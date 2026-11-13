"""Adaptive 1-VM standby pool and background ORB_SNAPSHOT_ID memory-checkpoint promoter."""

from __future__ import annotations

import contextlib
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from prime_sandboxes import (
    APIError,
    CreateSandboxRequest,
    ImageClient,
    Sandbox,
    SandboxClient,
    SandboxNotRunningError,
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
    _resume_until_running,
    _safe_delete_checkpoint,
    _safe_delete_sandbox,
    default_log,
    expire_stale_project_snapshots,
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
    """Promotes cold disk `ORB_SNAPSHOT_ID` checkpoints into clean `restore_memory=True` ones."""

    def __init__(
        self,
        config: ProviderConfig,
        sandbox_client: SandboxClient,
        store: SnapshotStore,
        *,
        log_fn: LogFn = default_log,
    ) -> None:
        self._config = config
        self._client = sandbox_client
        self._store = store
        self._log = log_fn
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
        """Run promotion synchronously (useful in deterministic tests)."""
        return self._promote_sync(orb_snapshot_id, disk_checkpoint_id, spec)

    def _promote_sync(
        self, orb_snapshot_id: str, disk_checkpoint_id: str, spec: OrbSpec
    ) -> str | None:
        try:
            current_ckpt, current_mode = self._store.get_project_snapshot_entry(orb_snapshot_id)
            target_mode = snapshot_memory_mode(spec)
            if current_ckpt and current_mode == target_mode:
                return current_ckpt

            deadline = time.monotonic() + 300.0
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
    """Maintains pre-placed standby VMs (RUNNING while active, auto-PAUSED once idle)."""

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
        self._hot_seconds = config.standby_hot_seconds
        self._maintenance_interval = maintenance_interval_seconds

        self._lock = threading.Lock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._pool: dict[str, list[str]] = {}
        self._creating_count: dict[str, int] = {}
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

    def reconcile(self) -> None:
        """Re-adopt live standby VMs across restarts and clean up stale/excess ones."""
        claimed_orb_ids = set(self._store.all_orb_sandboxes().values())
        discovered_by_id: dict[str, Sandbox] = {}

        # 1. Inspect persisted standby IDs from local state.
        for ids in self._store.all_standby_ids().values():
            for sbx_id in ids:
                if sbx_id in claimed_orb_ids:
                    continue
                with contextlib.suppress(APIError):
                    sbx = self._client.get(sbx_id)
                    if sbx.status not in TERMINAL_STATUSES:
                        discovered_by_id[sbx.id] = sbx

        # 2. Also discover any server-side `amp-orb-standby` VMs not yet in local state.
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

        target_key = self._default_key
        expected_label = standby_key_label(target_key) if target_key else None
        kept: list[Sandbox] = []
        to_delete: list[str] = []

        for sbx in discovered_by_id.values():
            if (
                self._target_count > 0
                and target_key is not None
                and expected_label in sbx.labels
                and sbx.status in ("RUNNING", "PAUSED")
                and len(kept) < self._target_count
            ):
                kept.append(sbx)
            else:
                to_delete.append(sbx.id)

        # Prefer already-RUNNING standbys ahead of PAUSED ones.
        kept.sort(key=lambda s: 0 if s.status == "RUNNING" else 1)
        kept_ids = [s.id for s in kept]

        with self._lock:
            self._pool.clear()
            if target_key and kept_ids:
                self._pool[target_key] = list(kept_ids)
                self._store.set_standby_ids(target_key, kept_ids)
            elif target_key:
                self._store.set_standby_ids(target_key, [])

        for sbx_id in to_delete:
            self._log(f"Draining unneeded standby sandbox {sbx_id}")
            _safe_delete_sandbox(self._client, sbx_id)

    def claim(
        self,
        resolved_image: str,
        spec: OrbSpec,
        deadline: float,
        log: LogFn,
    ) -> Sandbox | None:
        """Claim a pre-warmed standby VM for `(resolved_image, spec)`, or return None."""
        if self._target_count <= 0 or spec.gpu_count > 0:
            return None

        key = warm_template_key(resolved_image, spec)
        while True:
            with self._lock:
                self._last_activity = time.monotonic()
                bucket = self._pool.get(key)
                if not bucket:
                    self._wake_event.set()
                    return None
                candidate_id = bucket.pop(0)
                self._store.set_standby_ids(key, list(bucket))
                self._wake_event.set()

            try:
                sbx = self._client.get(candidate_id)
            except APIError:
                continue

            if sbx.status in TERMINAL_STATUSES:
                continue

            if sbx.status == "PAUSED":
                log(f"Waking pre-placed standby sandbox {sbx.id}")
                try:
                    return _resume_until_running(self._client, sbx, deadline, log)
                except (SandboxNotRunningError, RuntimeError, APIError):
                    _safe_delete_sandbox(self._client, sbx.id)
                    continue

            if sbx.status in ("PENDING", "PROVISIONING"):
                try:
                    return _poll_until_running(self._client, sbx, deadline)
                except (SandboxNotRunningError, APIError):
                    _safe_delete_sandbox(self._client, sbx.id)
                    continue

            log(f"Claimed hot RUNNING standby sandbox {sbx.id}")
            return sbx

    def replenish_once(self) -> Sandbox | None:
        """Synchronously create one standby VM if the default pool is below target_count."""
        if self._target_count <= 0 or not self._default_key or not self._default_image:
            return None
        key = self._default_key
        with self._lock:
            existing = len(self._pool.get(key, ())) + self._creating_count.get(key, 0)
            if existing >= self._target_count:
                return None
            self._creating_count[key] = self._creating_count.get(key, 0) + 1

        try:
            sbx = self._create_one_standby(key, self._default_image, self._config.default_spec)
            if sbx is None:
                return None
            # If hot_seconds == 0 or already idle past hot window, cool immediately to PAUSED.
            if self._is_idle():
                sbx = self._cool_standby(sbx) or sbx
            with self._lock:
                if self._stop_event.is_set():
                    return sbx
                bucket = self._pool.setdefault(key, [])
                # Keep RUNNING VMs at the front of the queue.
                if sbx.status == "RUNNING":
                    bucket.insert(0, sbx.id)
                else:
                    bucket.append(sbx.id)
                self._store.set_standby_ids(key, list(bucket))
            return sbx
        finally:
            with self._lock:
                self._creating_count[key] = max(0, self._creating_count.get(key, 1) - 1)

    def cool_idle_now(self) -> int:
        """Pause any RUNNING standby VMs if the pool has been idle past `standby_hot_seconds`."""
        if not self._is_idle() or not self._default_key:
            return 0
        key = self._default_key
        with self._lock:
            snapshot_ids = list(self._pool.get(key, ()))
        cooled = 0
        for sbx_id in snapshot_ids:
            with contextlib.suppress(APIError):
                sbx = self._client.get(sbx_id)
                if sbx.status == "RUNNING" and self._cool_standby(sbx) is not None:
                    cooled += 1
        return cooled

    def _is_idle(self) -> bool:
        with self._lock:
            elapsed = time.monotonic() - self._last_activity
        return elapsed >= self._hot_seconds

    def _create_one_standby(
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

        try:
            running = _poll_until_running(self._client, created, deadline)
            # Pre-seed gateway auth AND open a warm HTTP/2 ConnectRPC stream to this VM's gateway.
            _execute_when_gateway_ready(
                self._client,
                running.id,
                TEMPLATE_PREWARM_SCRIPT,
                envs={"PRIME_ORB_UPDATE_AMP": "0"},
                user=self._config.orb_user,
                deadline=deadline,
                action_label="standby prewarm",
            )
            self._log(f"Prepared warm standby sandbox {running.id}")
            return running
        except Exception as exc:
            self._log(f"Warning: failed to prepare standby sandbox {created.id}: {exc}")
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
