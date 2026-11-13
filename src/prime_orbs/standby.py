"""Adaptive base + project-snapshot standby pool and async ORB_SNAPSHOT_ID memory promoter."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from prime_sandboxes import (
    APIError,
    CreateSandboxRequest,
    ImageClient,
    Sandbox,
    SandboxClient,
)

from .config import OrbSpec, ProviderConfig
from .dispatcher import (
    ORB_TEMPLATE_LABEL,
    TEMPLATE_PREWARM_SCRIPT,
    TERMINAL_STATUSES,
    LogFn,
    _execute_when_gateway_ready,
    _poll_until_paused,
    _poll_until_running,
    _prefetch_gateway_auth_async,
    _safe_delete_checkpoint,
    _safe_delete_sandbox,
    _speculative_resume_until_running,
    default_log,
    expire_stale_project_snapshots,
    project_standby_key,
    snapshot_memory_mode,
    warm_template_key,
)
from .images import resolve_prime_image
from .state import SnapshotStore

ORB_STANDBY_LABEL = "amp-orb-standby"


def standby_key_label(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"amp-orb-standby-key:{digest}"


class SnapshotPromoter:
    """Finalizes async pause checkpoints and promotes `ORB_SNAPSHOT_ID` to `restore_memory=True`."""

    def __init__(
        self,
        config: ProviderConfig,
        sandbox_client: SandboxClient,
        store: SnapshotStore,
        *,
        log_fn: LogFn = default_log,
        on_promoted: Callable[[str, OrbSpec], None] | None = None,
    ) -> None:
        self._config = config
        self._client = sandbox_client
        self._store = store
        self._log = log_fn
        self._on_promoted = on_promoted
        self._lock = threading.Lock()
        self._inflight: set[str] = set()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prime-snap-warm")
        self._closed = False

    def submit(self, orb_snapshot_id: str, disk_checkpoint_id: str, spec: OrbSpec) -> None:
        if spec.gpu_count > 0:
            return
        with self._lock:
            if self._closed or orb_snapshot_id in self._inflight:
                return
            self._inflight.add(orb_snapshot_id)
        self._executor.submit(self._promote_sync, orb_snapshot_id, disk_checkpoint_id, spec)

    def promote_now(
        self, orb_snapshot_id: str, disk_checkpoint_id: str, spec: OrbSpec
    ) -> str | None:
        """Run finalize + memory promotion synchronously (useful in deterministic tests)."""
        return self._promote_sync(orb_snapshot_id, disk_checkpoint_id, spec)

    def _promote_sync(
        self, orb_snapshot_id: str, disk_checkpoint_id: str, spec: OrbSpec
    ) -> str | None:
        try:
            current_ckpt, current_mode = self._store.get_project_snapshot_entry(orb_snapshot_id)
            target_mode = snapshot_memory_mode(spec)
            if current_ckpt and current_mode == target_mode:
                if self._on_promoted is not None:
                    self._on_promoted(orb_snapshot_id, spec)
                return current_ckpt

            deadline = time.monotonic() + 300.0

            # Step 1: Ensure the source pause checkpoint is DURABLE (off the thread's pause path!).
            durable_disk = self._client.wait_for_checkpoint(
                disk_checkpoint_id, timeout_seconds=max(10.0, deadline - time.monotonic())
            )
            disk_checkpoint_id = durable_disk.id
            self._store.set_project_snapshot(orb_snapshot_id, disk_checkpoint_id, mode="disk")

            # Step 2: Boot a clean VM from that disk checkpoint, warm `amp` in RAM, and pause it.
            temp_name = f"amp-snap-warm-{int(time.time())}"
            self._log(
                f"Promoting project snapshot {orb_snapshot_id} ({disk_checkpoint_id}) "
                "to a bootable memory checkpoint ..."
            )
            created = self._client.create(
                CreateSandboxRequest(
                    name=temp_name,
                    checkpoint_id=disk_checkpoint_id,
                    cpu_cores=spec.cpu_cores,
                    memory_gb=spec.memory_gb,
                    timeout_minutes=30,
                    region=spec.region,
                    labels=[ORB_TEMPLATE_LABEL],
                )
            )
            try:
                _poll_until_running(self._client, created, deadline)
                _execute_when_gateway_ready(
                    self._client,
                    created.id,
                    TEMPLATE_PREWARM_SCRIPT,
                    envs={"PRIME_ORB_UPDATE_AMP": "0"},
                    user=self._config.orb_user,
                    deadline=deadline,
                    action_label="snapshot memory prewarm",
                )
                paused = self._client.pause(created.id)
                _poll_until_paused(self._client, created.id, paused.error_message, deadline)
                durable = self._client.wait_for_checkpoint(
                    paused.checkpoint_id, timeout_seconds=max(10.0, deadline - time.monotonic())
                )
            finally:
                _safe_delete_sandbox(self._client, created.id)

            self._store.set_project_snapshot(
                orb_snapshot_id, durable.id, mode=target_mode
            )
            if disk_checkpoint_id not in self._store.referenced_checkpoint_ids():
                _safe_delete_checkpoint(self._client, disk_checkpoint_id, deadline)
            self._log(
                f"Promoted project snapshot {orb_snapshot_id} -> memory checkpoint {durable.id}"
            )
            if self._on_promoted is not None:
                self._on_promoted(orb_snapshot_id, spec)
            return durable.id
        except Exception as exc:
            self._log(f"Warning: snapshot memory promotion skipped for {orb_snapshot_id}: {exc}")
            return None
        finally:
            with self._lock:
                self._inflight.discard(orb_snapshot_id)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)


class StandbyPool:
    """Maintains pre-placed base + active-project standby VMs (single-RPC hot claim)."""

    def __init__(
        self,
        config: ProviderConfig,
        sandbox_client: SandboxClient,
        image_client: ImageClient,
        store: SnapshotStore,
        *,
        log_fn: LogFn = default_log,
        maintenance_interval_seconds: float = 15.0,
    ) -> None:
        self._config = config
        self._client = sandbox_client
        self._image_client = image_client
        self._store = store
        self._log = log_fn
        self._target_count = config.standby_count
        self._max_project_standbys = (
            config.max_project_standbys if config.standby_count > 0 else 0
        )
        self._hot_seconds = config.standby_hot_seconds
        self._maintenance_interval = maintenance_interval_seconds

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        # bucket_key -> list[(sandbox_id, status)] where status is "RUNNING" or "PAUSED"
        self._pool: dict[str, list[tuple[str, str]]] = {}
        self._creating_count: dict[str, int] = {}
        # LRU of active project snapshots: bucket_key -> (orb_snapshot_id, OrbSpec)
        self._project_targets: OrderedDict[str, tuple[str, OrbSpec]] = OrderedDict()
        self._last_activity = time.monotonic()
        self._default_key: str | None = None
        self._default_image: str | None = None
        self._thread: threading.Thread | None = None

    def start(self, *, run_background_worker: bool = True) -> None:
        resolved_image = resolve_prime_image(self._config.default_spec.image, self._image_client)
        self._default_image = resolved_image
        self._default_key = warm_template_key(resolved_image, self._config.default_spec)
        self.reconcile()
        if run_background_worker and self._target_count > 0:
            self._thread = threading.Thread(
                target=self._worker_loop, name="prime-orbs-standby", daemon=True
            )
            self._thread.start()

    def register_project_snapshot(self, orb_snapshot_id: str, spec: OrbSpec) -> None:
        """Mark `orb_snapshot_id` active so the standby worker pre-places a hot VM for it."""
        if self._max_project_standbys <= 0 or spec.gpu_count > 0:
            return
        key = project_standby_key(orb_snapshot_id, spec)
        evicted_ids: list[str] = []
        with self._lock:
            self._last_activity = time.monotonic()
            self._project_targets.pop(key, None)
            self._project_targets[key] = (orb_snapshot_id, spec)
            while len(self._project_targets) > self._max_project_standbys:
                evicted_key, _ = self._project_targets.popitem(last=False)
                for sbx_id, _ in self._pool.pop(evicted_key, ()):
                    evicted_ids.append(sbx_id)
                self._store.set_standby_ids(evicted_key, [])
            self._wake_event.set()

        for sbx_id in evicted_ids:
            self._log(f"Draining LRU-evicted project standby sandbox {sbx_id}")
            _safe_delete_sandbox(self._client, sbx_id)

    def reconcile(self) -> None:
        """Re-adopt live base & project standby VMs across restarts; drain stale/excess ones."""
        claimed_orb_ids = set(self._store.all_orb_sandboxes().values())
        valid_snapshots = self._store.all_project_snapshots()
        persisted = self._store.all_standby_ids()
        discovered_by_id: dict[str, Sandbox] = {}

        for key, ids in persisted.items():
            if key.startswith("snap:") and len(self._project_targets) < self._max_project_standbys:
                snap_part = key.split("|", 1)[0].removeprefix("snap:")
                if snap_part in valid_snapshots:
                    self._project_targets[key] = (snap_part, self._config.default_spec)
            for sbx_id in ids:
                if sbx_id in claimed_orb_ids:
                    continue
                with contextlib.suppress(APIError):
                    sbx = self._client.get(sbx_id)
                    if sbx.status not in TERMINAL_STATUSES:
                        discovered_by_id[sbx.id] = sbx

        cursor: str | None = None
        seen_cursors: set[str] = set()
        with contextlib.suppress(APIError):
            while True:
                page = self._client.list(labels=[ORB_STANDBY_LABEL], per_page=100, cursor=cursor)
                for sbx in page.sandboxes:
                    if sbx.id not in claimed_orb_ids and sbx.status not in TERMINAL_STATUSES:
                        discovered_by_id[sbx.id] = sbx
                if not page.has_next or not page.next_cursor or page.next_cursor in seen_cursors:
                    break
                seen_cursors.add(page.next_cursor)
                cursor = page.next_cursor

        allowed_quotas: dict[str, int] = {}
        if self._target_count > 0 and self._default_key:
            allowed_quotas[self._default_key] = self._target_count
        if self._max_project_standbys > 0:
            for proj_key in self._project_targets:
                allowed_quotas[proj_key] = 1

        label_to_key = {standby_key_label(k): k for k in allowed_quotas}
        grouped: dict[str, list[Sandbox]] = {k: [] for k in allowed_quotas}
        to_delete: list[str] = []

        for sbx in discovered_by_id.values():
            matched_key = next(
                (label_to_key[lbl] for lbl in sbx.labels if lbl in label_to_key), None
            )
            if (
                matched_key is not None
                and sbx.status in ("RUNNING", "PAUSED")
                and len(grouped[matched_key]) < allowed_quotas[matched_key]
            ):
                grouped[matched_key].append(sbx)
            else:
                to_delete.append(sbx.id)

        with self._lock:
            self._pool.clear()
            for old_key in set(persisted) - set(allowed_quotas):
                self._store.set_standby_ids(old_key, [])
            for key, items in grouped.items():
                items.sort(key=lambda s: 0 if s.status == "RUNNING" else 1)
                if items:
                    self._pool[key] = [(s.id, s.status) for s in items]
                self._store.set_standby_ids(key, [s.id for s in items])

        for sbx_id in to_delete:
            self._log(f"Draining unneeded standby sandbox {sbx_id}")
            _safe_delete_sandbox(self._client, sbx_id)

    def claim(
        self,
        resolved_image: str,
        spec: OrbSpec,
        deadline: float,
        log: LogFn,
    ) -> tuple[str, bool] | None:
        if self._target_count <= 0 or spec.gpu_count > 0:
            return None
        return self._claim_from_bucket(warm_template_key(resolved_image, spec), deadline, log)

    def claim_project(
        self,
        orb_snapshot_id: str,
        spec: OrbSpec,
        deadline: float,
        log: LogFn,
    ) -> tuple[str, bool] | None:
        if self._max_project_standbys <= 0 or spec.gpu_count > 0:
            return None
        return self._claim_from_bucket(project_standby_key(orb_snapshot_id, spec), deadline, log)

    def _claim_from_bucket(
        self,
        key: str,
        deadline: float,
        log: LogFn,
    ) -> tuple[str, bool] | None:
        while True:
            with self._lock:
                self._last_activity = time.monotonic()
                bucket = self._pool.get(key)
                if not bucket:
                    self._wake_event.set()
                    return None
                candidate_id, known_status = bucket.pop(0)
                self._store.set_standby_ids(key, [sid for sid, _ in bucket])
                self._wake_event.set()

            # Single-RPC fast path: skip GET /sandbox/{id} completely when already RUNNING!
            if known_status == "RUNNING":
                log(f"Claimed hot RUNNING standby sandbox {candidate_id} (0-GET fast path)")
                return candidate_id, True

            log(f"Waking pre-placed standby sandbox {candidate_id}")
            _prefetch_gateway_auth_async(self._client, candidate_id)
            resumed = _speculative_resume_until_running(self._client, candidate_id, deadline, log)
            if resumed is not None:
                return resumed.id, False
            _safe_delete_sandbox(self._client, candidate_id)

    def replenish_once(self) -> Sandbox | None:
        """Synchronously create one missing base or active-project standby VM."""
        if self._target_count <= 0:
            return None

        target = self._pick_next_replenish_target()
        if target is None:
            return None
        key, orb_snapshot_id, spec = target

        try:
            if orb_snapshot_id is not None:
                sbx = self._create_project_standby(key, orb_snapshot_id, spec)
            else:
                assert self._default_image is not None
                sbx = self._create_base_standby(key, self._default_image, spec)
            if sbx is None:
                return None

            status = "RUNNING"
            if self._is_idle():
                cooled = self._cool_standby(sbx)
                if cooled is not None:
                    sbx = cooled
                    status = "PAUSED"

            with self._lock:
                if self._stop_event.is_set():
                    return sbx
                bucket = self._pool.setdefault(key, [])
                if status == "RUNNING":
                    bucket.insert(0, (sbx.id, "RUNNING"))
                else:
                    bucket.append((sbx.id, "PAUSED"))
                self._store.set_standby_ids(key, [sid for sid, _ in bucket])
            return sbx
        finally:
            with self._lock:
                self._creating_count[key] = max(0, self._creating_count.get(key, 1) - 1)

    def _pick_next_replenish_target(self) -> tuple[str, str | None, OrbSpec] | None:
        valid_snapshots = self._store.all_project_snapshots()
        modes = self._store.all_project_snapshot_modes()
        stale_Ids: list[str] = []

        with self._lock:
            # Prune any project targets whose snapshot expired or is still uploading.
            for proj_key in list(self._project_targets):
                snap_id, _ = self._project_targets[proj_key]
                if snap_id not in valid_snapshots:
                    self._project_targets.pop(proj_key, None)
                    for sid, _ in self._pool.pop(proj_key, ()):
                        stale_Ids.append(sid)
                    self._store.set_standby_ids(proj_key, [])

            # 1. Base standby pool first
            if self._default_key is not None:
                cur = len(self._pool.get(self._default_key, ())) + self._creating_count.get(
                    self._default_key, 0
                )
                if cur < self._target_count:
                    self._creating_count[self._default_key] = (
                        self._creating_count.get(self._default_key, 0) + 1
                    )
                    target: tuple[str, str | None, OrbSpec] | None = (
                        self._default_key,
                        None,
                        self._config.default_spec,
                    )
                else:
                    target = None
            else:
                target = None

            # 2. Most-recently-used ready project snapshots next
            if target is None:
                for proj_key, (snap_id, spec) in reversed(list(self._project_targets.items())):
                    if modes.get(snap_id) == "uploading":
                        continue
                    cur = len(self._pool.get(proj_key, ())) + self._creating_count.get(proj_key, 0)
                    if cur < 1:
                        self._creating_count[proj_key] = (
                            self._creating_count.get(proj_key, 0) + 1
                        )
                        target = (proj_key, snap_id, spec)
                        break

        for sid in stale_Ids:
            _safe_delete_sandbox(self._client, sid)
        return target

    def cool_idle_now(self) -> int:
        """Pause any RUNNING standby VMs if the pool has been idle past `standby_hot_seconds`."""
        if not self._is_idle():
            return 0
        with self._lock:
            candidates = [
                (key, sid)
                for key, items in self._pool.items()
                for sid, st in items
                if st == "RUNNING"
            ]
        cooled = 0
        for key, sbx_id in candidates:
            with contextlib.suppress(APIError):
                sbx = self._client.get(sbx_id)
                if sbx.status == "RUNNING" and self._cool_standby(sbx) is not None:
                    self._mark_bucket_status(key, sbx_id, "PAUSED")
                    cooled += 1
                elif sbx.status == "PAUSED":
                    self._mark_bucket_status(key, sbx_id, "PAUSED")
        return cooled

    def refresh_standby_tokens_now(self) -> int:
        """Keep RUNNING standby `gatewayAuth` tokens warm across 30-minute expiry boundaries."""
        auth_cache = getattr(self._client, "_auth_cache", None)
        get_or_refresh = getattr(auth_cache, "get_or_refresh", None)
        if not callable(get_or_refresh):
            return 0
        with self._lock:
            running_ids = [
                sid
                for items in self._pool.values()
                for sid, st in items
                if st == "RUNNING"
            ]
        refreshed = 0
        for sbx_id in running_ids:
            with contextlib.suppress(Exception):
                get_or_refresh(sbx_id)
                refreshed += 1
        return refreshed

    def _mark_bucket_status(self, key: str, sbx_id: str, new_status: str) -> None:
        with self._lock:
            bucket = self._pool.get(key)
            if not bucket:
                return
            self._pool[key] = [
                (sid, new_status if sid == sbx_id else st) for sid, st in bucket
            ]

    def _is_idle(self) -> bool:
        with self._lock:
            elapsed = time.monotonic() - self._last_activity
        return elapsed >= self._hot_seconds

    def _create_base_standby(
        self, key: str, resolved_image: str, spec: OrbSpec
    ) -> Sandbox | None:
        if spec.gpu_count > 0:
            return None
        deadline = time.monotonic() + 240.0
        name = f"amp-orb-standby-{int(time.time())}"
        labels = [ORB_STANDBY_LABEL, standby_key_label(key)]
        warm_ckpt = (
            self._store.get_warm_template(key) if self._config.warm_template_enabled else None
        )

        created: Sandbox | None = None
        if warm_ckpt:
            try:
                created = self._client.create(
                    CreateSandboxRequest(
                        name=name,
                        checkpoint_id=warm_ckpt,
                        restore_memory=True,
                        timeout_minutes=spec.timeout_minutes,
                        idle_timeout_minutes=spec.idle_timeout_minutes,
                        region=spec.region,
                        network_allowlist=spec.network_allowlist,
                        network_denylist=spec.network_denylist,
                        labels=labels,
                    )
                )
            except APIError:
                self._store.remove_warm_template(key)
                created = None

        if created is None:
            created = self._client.create(
                CreateSandboxRequest(
                    name=name,
                    docker_image=resolved_image,
                    arch=spec.arch,
                    cpu_cores=spec.cpu_cores,
                    memory_gb=spec.memory_gb,
                    disk_size_gb=spec.disk_size_gb,
                    timeout_minutes=spec.timeout_minutes,
                    idle_timeout_minutes=spec.idle_timeout_minutes,
                    region=spec.region,
                    network_allowlist=spec.network_allowlist,
                    network_denylist=spec.network_denylist,
                    labels=labels,
                )
            )

        return self._finalize_standby_vm(created, deadline, f"base standby {created.id}")

    def _create_project_standby(
        self, key: str, orb_snapshot_id: str, spec: OrbSpec
    ) -> Sandbox | None:
        if spec.gpu_count > 0:
            return None
        ckpt_id, snap_mode = self._store.get_project_snapshot_entry(orb_snapshot_id)
        if not ckpt_id or snap_mode == "uploading":
            return None

        deadline = time.monotonic() + 240.0
        name = f"amp-proj-standby-{int(time.time())}"
        labels = [ORB_STANDBY_LABEL, standby_key_label(key)]
        use_memory = snap_mode == snapshot_memory_mode(spec)

        metadata_kwargs = dict(
            name=name,
            checkpoint_id=ckpt_id,
            timeout_minutes=spec.timeout_minutes,
            idle_timeout_minutes=spec.idle_timeout_minutes,
            region=spec.region,
            network_allowlist=spec.network_allowlist,
            network_denylist=spec.network_denylist,
            labels=labels,
        )
        try:
            if use_memory:
                created = self._client.create(
                    CreateSandboxRequest(restore_memory=True, **metadata_kwargs)
                )
            else:
                created = self._client.create(
                    CreateSandboxRequest(
                        cpu_cores=spec.cpu_cores,
                        memory_gb=spec.memory_gb,
                        **metadata_kwargs,
                    )
                )
        except APIError as exc:
            self._log(
                f"Warning: project standby create failed for {orb_snapshot_id} ({exc}); pruning"
            )
            self._store.remove_project_snapshot(orb_snapshot_id)
            return None

        return self._finalize_standby_vm(
            created, deadline, f"project standby ({orb_snapshot_id}) {created.id}"
        )

    def _finalize_standby_vm(
        self, created: Sandbox, deadline: float, label: str
    ) -> Sandbox | None:
        try:
            running = _poll_until_running(self._client, created, deadline)
            _execute_when_gateway_ready(
                self._client,
                running.id,
                TEMPLATE_PREWARM_SCRIPT,
                envs={"PRIME_ORB_UPDATE_AMP": "0"},
                user=self._config.orb_user,
                deadline=deadline,
                action_label="standby prewarm",
            )
            self._log(f"Prepared warm {label}")
            return running
        except Exception as exc:
            self._log(f"Warning: failed to prepare {label}: {exc}")
            _safe_delete_sandbox(self._client, created.id)
            return None

    def _cool_standby(self, sandbox: Sandbox) -> Sandbox | None:
        deadline = time.monotonic() + 120.0
        try:
            paused = self._client.pause(sandbox.id)
            _poll_until_paused(self._client, sandbox.id, paused.error_message, deadline)
            self._log(f"Auto-paused idle standby sandbox {sandbox.id}")
            return self._client.get(sandbox.id)
        except Exception as exc:
            self._log(f"Warning: could not auto-pause standby sandbox {sandbox.id}: {exc}")
            return None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                while not self._stop_event.is_set():
                    created = self.replenish_once()
                    if created is None:
                        break
                if not self._stop_event.is_set():
                    self.cool_idle_now()
                if not self._stop_event.is_set():
                    self.refresh_standby_tokens_now()
                if not self._stop_event.is_set():
                    expire_stale_project_snapshots(
                        self._client,
                        self._store,
                        ttl_seconds=self._config.snapshot_ttl_seconds,
                        log_fn=self._log,
                    )
            except Exception as exc:
                self._log(f"Warning: standby pool maintenance error: {exc}")

            self._wake_event.wait(timeout=self._maintenance_interval)
            self._wake_event.clear()

    def close(self) -> None:
        """Stop background worker without deleting persisted standby VMs (re-adopted on restart)."""
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
