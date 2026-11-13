"""Core Amp self-hosted orb lifecycle dispatcher backed by Prime Sandboxes."""

from __future__ import annotations

import contextlib
import hashlib
import sys
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from prime_sandboxes import (
    APIClient,
    APIError,
    CommandTimeoutError,
    CreateSandboxRequest,
    ImageClient,
    Sandbox,
    SandboxClient,
    SandboxNotRunningError,
    UnauthorizedError,
)

try:
    from prime_sandboxes.sandbox import (
        _is_retryable_reachability_error as _sdk_is_retryable_reachability_error,
    )
except ImportError:  # pragma: no cover

    def _sdk_is_retryable_reachability_error(_exc: BaseException) -> bool:
        return False


from .config import DispatchJob, OrbSpec, ProviderConfig
from .images import resolve_prime_image
from .state import SnapshotStore, orb_lock

ORB_MANAGED_LABEL = "amp-orb"
ORB_TEMPLATE_LABEL = "amp-orb-template"
TERMINAL_STATUSES = frozenset({"TERMINATED", "TIMEOUT", "ERROR"})

BOOTSTRAP_SCRIPT = """\
set -euo pipefail
export HOME="${HOME:-/home/user}"
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
amp_dirs="$HOME/.amp/bin:/opt/git/bin:$HOME/.local/bin:$HOME/.nix-profile/bin"
export PATH="$amp_dirs:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
cd "$HOME" 2>/dev/null || cd /
if ! command -v amp >/dev/null 2>&1; then
  echo "amp CLI binary not found on PATH ($PATH)" >&2
  exit 127
fi
nohup setsid amp orb bootstrap </dev/null >>"${HOME}/.amp-bootstrap.log" 2>&1 &
"""

TEMPLATE_PREWARM_SCRIPT = """\
set -euo pipefail
export HOME="${HOME:-/home/user}"
export USER="${USER:-user}"
export LOGNAME="${LOGNAME:-$USER}"
amp_dirs="$HOME/.amp/bin:/opt/git/bin:$HOME/.local/bin:$HOME/.nix-profile/bin"
export PATH="$amp_dirs:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
cd "$HOME" 2>/dev/null || cd /
if [ "${PRIME_ORB_UPDATE_AMP:-1}" = "1" ] && command -v amp >/dev/null 2>&1; then
  amp update --porcelain || true
fi
command -v amp >/dev/null 2>&1 && amp --version >/dev/null 2>&1 || true
rm -f "${HOME}/.amp-bootstrap.log"
sync
"""

LogFn = Callable[[str], None]
StandbyClaimFn = Callable[[str, OrbSpec, float, LogFn], Sandbox | None]
SnapshotSavedFn = Callable[[str, str, OrbSpec], None]


def default_log(message: str) -> None:
    print(f"[prime-orbs] {message}", file=sys.stderr, flush=True)


def orb_id_label(orb_id: str) -> str:
    return f"amp-orb:{orb_id}"


def orb_labels(orb_id: str, orb_type: str | None = None) -> list[str]:
    labels = [ORB_MANAGED_LABEL, orb_id_label(orb_id)]
    if orb_type:
        labels.append(f"amp-orb-type:{orb_type}")
    return labels


def warm_template_key(resolved_image: str, spec: OrbSpec) -> str:
    arch = spec.arch or "amd64"
    return (
        f"{resolved_image}|cpu={spec.cpu_cores:g}|mem={spec.memory_gb:g}"
        f"|disk={spec.disk_size_gb:g}|arch={arch}"
    )


def snapshot_memory_mode(spec: OrbSpec) -> str:
    return f"memory:{spec.cpu_cores:g}:{spec.memory_gb:g}"


def _aware_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def _remaining(deadline: float, *, floor: float = 0.25) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("Dispatch operation exceeded configured time budget")
    return max(floor, left)


def _fast_poll_delay(elapsed: float) -> float:
    """Poll tightly (250ms) during normal 1-15s boot/resume/pause windows, then ease off."""
    if elapsed < 15.0:
        return 0.25
    if elapsed < 45.0:
        return 0.5
    return 1.0


def _deterministic_idempotency_key(orb_id: str, generation: int, source_tag: str) -> str:
    digest = hashlib.sha256(f"{orb_id}:{generation}:{source_tag}".encode()).hexdigest()
    return digest[:32]


def _invalidate_gateway_auth(client: SandboxClient, sandbox_id: str) -> None:
    auth_cache = getattr(client, "_auth_cache", None)
    invalidate = getattr(auth_cache, "invalidate", None)
    if callable(invalidate):
        with contextlib.suppress(Exception):
            invalidate(sandbox_id)


def _is_waiting_for_image_build(sandbox: Sandbox) -> bool:
    return sandbox.status == "PENDING" and bool(getattr(sandbox, "pending_image_build_id", None))


def _is_retryable_bootstrap_error(exc: BaseException) -> bool:
    if _sdk_is_retryable_reachability_error(exc):
        return True
    return isinstance(exc, (APIError, CommandTimeoutError))


def _extract_orb_id_from_labels(sandbox: Sandbox) -> str | None:
    prefix = "amp-orb:"
    for label in sandbox.labels:
        if label.startswith(prefix) and len(label) > len(prefix):
            return label[len(prefix) :]
    return None


def sync_known_orbs(
    client: SandboxClient,
    store: SnapshotStore,
    *,
    log_fn: LogFn = default_log,
) -> dict[str, str]:
    """Reconcile `store.orb_sandboxes` once on daemon startup for O(0)-GET new-thread checks."""
    reconciled: dict[str, str] = {}

    # 1. Preserve any persisted orb_id -> sandbox_id mappings (including claimed standby VMs).
    for orb_id, sbx_id in store.all_orb_sandboxes().items():
        with contextlib.suppress(APIError):
            sbx = client.get(sbx_id)
            if sbx.status not in TERMINAL_STATUSES:
                reconciled[orb_id] = sbx.id

    # 2. Discover any labeled `amp-orb` sandboxes created outside local state.
    by_orb: dict[str, list[Sandbox]] = {}
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        page = client.list(labels=[ORB_MANAGED_LABEL], per_page=100, cursor=cursor)
        for sbx in page.sandboxes:
            if sbx.status in TERMINAL_STATUSES:
                continue
            orb_id = _extract_orb_id_from_labels(sbx)
            if orb_id:
                by_orb.setdefault(orb_id, []).append(sbx)
        if not page.has_next or not page.next_cursor or page.next_cursor in seen_cursors:
            break
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor

    for orb_id, items in by_orb.items():
        items.sort(key=lambda s: _aware_utc(s.created_at), reverse=True)
        reconciled[orb_id] = items[0].id
        for extra in items[1:]:
            log_fn(f"Deleting duplicate active sandbox {extra.id} for {orb_id}")
            _safe_delete_sandbox(client, extra.id)

    store.replace_orb_sandboxes(reconciled)
    return reconciled


def list_sandboxes_for_orb(
    client: SandboxClient, orb_id: str
) -> tuple[list[Sandbox], list[Sandbox]]:
    """Return `(active, terminated)` sandboxes for `orb_id`, each sorted newest-first."""
    found: list[Sandbox] = []
    seen_ids: set[str] = set()
    seen_cursors: set[str] = set()
    cursor: str | None = None
    target_label = orb_id_label(orb_id)

    while True:
        page = client.list(labels=[target_label], per_page=100, cursor=cursor)
        for sbx in page.sandboxes:
            if target_label in sbx.labels and sbx.id not in seen_ids:
                seen_ids.add(sbx.id)
                found.append(sbx)
        if not page.has_next or not page.next_cursor or page.next_cursor in seen_cursors:
            break
        seen_cursors.add(page.next_cursor)
        cursor = page.next_cursor

    found.sort(key=lambda s: _aware_utc(s.created_at), reverse=True)
    active = [s for s in found if s.status not in TERMINAL_STATUSES]
    terminated = [s for s in found if s.status in TERMINAL_STATUSES]
    return active, terminated


def _resolve_existing_orb_sandbox(
    client: SandboxClient,
    store: SnapshotStore,
    orb_id: str,
    log: LogFn,
    *,
    authoritative_index: bool = False,
) -> tuple[Sandbox | None, list[Sandbox]]:
    """Fast-path lookup via cached `orb_id -> sandbox_id` (0 GETs when brand-new in daemon)."""
    cached_id = store.get_orb_sandbox(orb_id)
    if cached_id:
        try:
            sbx = client.get(cached_id)
            if sbx.status not in TERMINAL_STATUSES:
                return sbx, []
            store.remove_orb_sandbox(orb_id)
            return None, [sbx]
        except APIError as exc:
            if "HTTP 404" not in str(exc) and "not found" not in str(exc).lower():
                raise
            store.remove_orb_sandbox(orb_id)
            if authoritative_index:
                return None, []

    if authoritative_index:
        return None, []

    active, terminated = list_sandboxes_for_orb(client, orb_id)
    for extra in active[1:]:
        log(f"Deleting duplicate active sandbox {extra.id} for {orb_id}")
        _safe_delete_sandbox(client, extra.id)

    if active:
        chosen = active[0]
        store.set_orb_sandbox(orb_id, chosen.id)
        return chosen, terminated
    return None, terminated


def run_dispatch(
    job: DispatchJob,
    config: ProviderConfig,
    *,
    sandbox_client: SandboxClient | None = None,
    image_client: ImageClient | None = None,
    log_fn: LogFn = default_log,
    authoritative_index: bool = False,
    standby_claim_fn: StandbyClaimFn | None = None,
    on_snapshot_saved: SnapshotSavedFn | None = None,
) -> None:
    """Execute one `spawn`, `pause`, or `delete` job against Prime Sandboxes."""
    deadline = time.monotonic() + config.dispatch_timeout_seconds

    if sandbox_client is None or image_client is None:
        api_client = APIClient()
        if not api_client.config.api_key:
            raise UnauthorizedError(
                "Missing Prime API key. Set PRIME_API_KEY in the provider environment "
                "or authenticate with `prime login`."
            )
        sandbox_client = sandbox_client or SandboxClient(api_client)
        image_client = image_client or ImageClient(api_client)

    store = SnapshotStore(config.state_dir)

    with orb_lock(config.state_dir, job.orb_id, timeout_seconds=_remaining(deadline)):
        if job.operation == "spawn":
            _handle_spawn(
                job,
                config,
                sandbox_client,
                image_client,
                store,
                deadline,
                log_fn,
                authoritative_index=authoritative_index,
                standby_claim_fn=standby_claim_fn,
            )
        elif job.operation == "pause":
            _handle_pause(
                job,
                config,
                sandbox_client,
                store,
                deadline,
                log_fn,
                authoritative_index=authoritative_index,
                on_snapshot_saved=on_snapshot_saved,
            )
        elif job.operation == "delete":
            _handle_delete(job, sandbox_client, store, deadline, log_fn)
        else:
            raise ValueError(f"Unsupported operation: {job.operation!r}")


def _handle_spawn(
    job: DispatchJob,
    config: ProviderConfig,
    client: SandboxClient,
    image_client: ImageClient,
    store: SnapshotStore,
    deadline: float,
    log: LogFn,
    *,
    authoritative_index: bool = False,
    standby_claim_fn: StandbyClaimFn | None = None,
) -> None:
    spec = config.resolve_spec(job)
    candidate, terminated = _resolve_existing_orb_sandbox(
        client, store, job.orb_id, log, authoritative_index=authoritative_index
    )

    running_sandbox: Sandbox | None = None
    if candidate is not None:
        running_sandbox = _ensure_existing_running(client, candidate, job.orb_id, deadline, log)
        if running_sandbox is None:
            store.remove_orb_sandbox(job.orb_id)
            terminated.insert(0, candidate)

    if running_sandbox is None:
        running_sandbox = _create_orb_sandbox(
            job=job,
            spec=spec,
            config=config,
            client=client,
            image_client=image_client,
            store=store,
            terminated=terminated,
            deadline=deadline,
            log=log,
            standby_claim_fn=standby_claim_fn,
        )
        store.set_orb_sandbox(job.orb_id, running_sandbox.id)

    _launch_bootstrap_when_ready(client, running_sandbox.id, job, config, deadline, log)


def _ensure_existing_running(
    client: SandboxClient,
    sandbox: Sandbox,
    orb_id: str,
    deadline: float,
    log: LogFn,
) -> Sandbox | None:
    """Resume or wait on an existing sandbox; return None only if its node died irrecoverably."""
    if sandbox.status == "PAUSED":
        log(f"Resuming paused sandbox {sandbox.id} for {orb_id}")
        try:
            return _resume_until_running(client, sandbox, deadline, log)
        except SandboxNotRunningError as exc:
            log(
                f"Paused sandbox {sandbox.id} could not resume ({exc}); "
                "falling back to checkpoint/image recovery"
            )
            return None
    if sandbox.status in ("PENDING", "PROVISIONING"):
        log(f"Waiting for in-flight sandbox {sandbox.id} ({sandbox.status})")
        return _poll_until_running(client, sandbox, deadline)
    return sandbox


def _poll_until_running(
    client: SandboxClient,
    initial: Sandbox,
    deadline: float,
) -> Sandbox:
    """Poll tightly until `initial` reaches RUNNING (0 extra polls if already RUNNING)."""
    if initial.status == "RUNNING":
        return initial

    started = time.monotonic()
    while True:
        remaining = _remaining(deadline)
        current = client.get(initial.id)
        if current.status == "RUNNING":
            return current
        if current.status in TERMINAL_STATUSES:
            raise SandboxNotRunningError(
                initial.id,
                message=(
                    f"Sandbox entered {current.status} before reaching RUNNING: "
                    f"{current.error_message or current.termination_reason or 'no details'}"
                ),
            )
        delay = (
            5.0
            if _is_waiting_for_image_build(current)
            else _fast_poll_delay(time.monotonic() - started)
        )
        time.sleep(min(delay, remaining))


def _resume_until_running(
    client: SandboxClient,
    sandbox: Sandbox,
    deadline: float,
    log: LogFn,
) -> Sandbox:
    accepted = _request_resume_with_retry(client, sandbox.id, deadline)
    if accepted.status == "RUNNING":
        return accepted
    baseline_error = accepted.error_message

    retried_after_node_error = False
    started = time.monotonic()
    while True:
        remaining = _remaining(deadline)
        current = client.get(sandbox.id)
        if current.status == "RUNNING":
            return current
        if current.status in TERMINAL_STATUSES:
            raise SandboxNotRunningError(
                sandbox.id,
                message=(
                    f"Sandbox entered {current.status} during resume: "
                    f"{current.error_message or current.termination_reason or 'no details'}"
                ),
            )
        if (
            current.status == "PAUSED"
            and current.error_message
            and current.error_message != baseline_error
        ):
            if not retried_after_node_error:
                retried_after_node_error = True
                log(f"Resume reported error ({current.error_message}); retrying once")
                accepted_retry = _request_resume_with_retry(client, sandbox.id, deadline)
                if accepted_retry.status == "RUNNING":
                    return accepted_retry
                baseline_error = accepted_retry.error_message
            else:
                raise RuntimeError(
                    f"Failed to resume paused sandbox {sandbox.id}: {current.error_message}"
                )
        time.sleep(min(_fast_poll_delay(time.monotonic() - started), remaining))


def _request_resume_with_retry(
    client: SandboxClient, sandbox_id: str, deadline: float
) -> Sandbox:
    delay = 0.25
    for attempt in range(6):
        try:
            return client.resume(sandbox_id)
        except APIError as exc:
            msg = str(exc)
            is_conflict_or_5xx = any(
                code in msg for code in ("HTTP 409", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")
            )
            if not is_conflict_or_5xx or attempt == 5:
                raise
            remaining = _remaining(deadline)
            time.sleep(min(delay, remaining))
            delay = min(delay * 2.0, 2.0)
    raise AssertionError("unreachable")


def _get_durable_checkpoint_id(client: SandboxClient, checkpoint_id: str) -> str | None:
    try:
        ckpt = client.get_checkpoint(checkpoint_id)
    except APIError:
        return None
    if ckpt.state == "DURABLE" and ckpt.merge != "failed":
        return ckpt.id
    return None


def _find_orb_recovery_checkpoint(
    client: SandboxClient,
    store: SnapshotStore,
    orb_id: str,
    terminated: list[Sandbox],
) -> str | None:
    recorded = store.get_orb_checkpoint(orb_id)
    if recorded:
        durable = _get_durable_checkpoint_id(client, recorded)
        if durable:
            return durable
        store.remove_orb_checkpoint(orb_id)

    for past in terminated:
        try:
            checkpoints = client.list_all_checkpoints(
                sandbox_id=past.id, states=["DURABLE"]
            )
        except APIError:
            continue
        if checkpoints:
            return checkpoints[-1].id
    return None


def _build_create_request(
    *,
    orb_id: str,
    orb_type: str | None,
    spec: OrbSpec,
    generation: int,
    docker_image: str | None = None,
    checkpoint_id: str | None = None,
    restore_memory: bool = False,
) -> CreateSandboxRequest:
    mode_tag = "mem" if restore_memory else "disk"
    source_tag = f"ckpt:{mode_tag}:{checkpoint_id}" if checkpoint_id else f"img:{docker_image}"
    metadata_kwargs = dict(
        name=orb_id,
        timeout_minutes=spec.timeout_minutes,
        idle_timeout_minutes=spec.idle_timeout_minutes,
        region=spec.region,
        network_allowlist=spec.network_allowlist,
        network_denylist=spec.network_denylist,
        labels=orb_labels(orb_id, orb_type),
        idempotency_key=_deterministic_idempotency_key(orb_id, generation, source_tag),
    )
    if checkpoint_id is not None and restore_memory:
        # Memory restores inherit cpu_cores, memory_gb, gpu_count, gpu_type, disk, and arch
        # directly from the paused template VM; leave those fields unset on the model.
        return CreateSandboxRequest(
            checkpoint_id=checkpoint_id,
            restore_memory=True,
            **metadata_kwargs,
        )
    shape_kwargs = dict(
        cpu_cores=spec.cpu_cores,
        memory_gb=spec.memory_gb,
        gpu_count=spec.gpu_count,
        gpu_type=spec.gpu_type,
    )
    if checkpoint_id is not None:
        return CreateSandboxRequest(
            checkpoint_id=checkpoint_id,
            **shape_kwargs,
            **metadata_kwargs,
        )
    return CreateSandboxRequest(
        docker_image=docker_image,
        arch=spec.arch,
        disk_size_gb=spec.disk_size_gb,
        **shape_kwargs,
        **metadata_kwargs,
    )


def _create_orb_sandbox(
    *,
    job: DispatchJob,
    spec: OrbSpec,
    config: ProviderConfig,
    client: SandboxClient,
    image_client: ImageClient,
    store: SnapshotStore,
    terminated: list[Sandbox],
    deadline: float,
    log: LogFn,
    standby_claim_fn: StandbyClaimFn | None = None,
) -> Sandbox:
    generation = len(terminated)

    # 1. Recover this orb's disk if a previous sandbox was checkpointed (GPU pause or lost node).
    recovery_ckpt = _find_orb_recovery_checkpoint(client, store, job.orb_id, terminated)
    if recovery_ckpt:
        log(f"Restoring {job.orb_id} from orb checkpoint {recovery_ckpt}")
        try:
            created = client.create(
                _build_create_request(
                    orb_id=job.orb_id,
                    orb_type=job.orb_type,
                    spec=spec,
                    generation=generation,
                    checkpoint_id=recovery_ckpt,
                )
            )
            return _poll_until_running(client, created, deadline)
        except APIError as exc:
            log(f"Orb checkpoint {recovery_ckpt} restore failed ({exc}); falling back")
            store.remove_orb_checkpoint(job.orb_id)

    # 2. Boot from shared Amp project snapshot (ORB_SNAPSHOT_ID) when available.
    if config.snapshots_enabled and job.orb_snapshot_id:
        mapped_ckpt, snap_mode = store.get_project_snapshot_entry(job.orb_snapshot_id)
        if mapped_ckpt:
            use_memory = snap_mode == snapshot_memory_mode(spec) and spec.gpu_count == 0
            kind_desc = "memory-promoted" if use_memory else "disk"
            log(
                f"Creating {job.orb_id} from {kind_desc} project snapshot "
                f"{job.orb_snapshot_id} (checkpoint {mapped_ckpt})"
            )
            try:
                created = client.create(
                    _build_create_request(
                        orb_id=job.orb_id,
                        orb_type=job.orb_type,
                        spec=spec,
                        generation=generation,
                        checkpoint_id=mapped_ckpt,
                        restore_memory=use_memory,
                    )
                )
                return _poll_until_running(client, created, deadline)
            except APIError as exc:
                log(
                    f"Project snapshot {job.orb_snapshot_id} restore failed ({exc}); "
                    "pruning and falling back"
                )
                store.remove_project_snapshot(job.orb_snapshot_id)

    resolved_image = resolve_prime_image(spec.image, image_client, deadline=deadline)

    # 3. Claim a pre-placed RUNNING/PAUSED VM from the daemon's adaptive standby pool.
    if standby_claim_fn is not None and spec.gpu_count == 0:
        claimed = standby_claim_fn(resolved_image, spec, deadline, log)
        if claimed is not None:
            return claimed

    # 4. Fork from a pre-booted warm memory checkpoint (`restore_memory=True`) when available.
    if config.warm_template_enabled and spec.gpu_count == 0:
        tmpl_key = warm_template_key(resolved_image, spec)
        warm_ckpt = store.get_warm_template(tmpl_key)
        if warm_ckpt:
            log(f"Forking {job.orb_id} from warm memory checkpoint {warm_ckpt}")
            try:
                created = client.create(
                    _build_create_request(
                        orb_id=job.orb_id,
                        orb_type=job.orb_type,
                        spec=spec,
                        generation=generation,
                        checkpoint_id=warm_ckpt,
                        restore_memory=True,
                    )
                )
                return _poll_until_running(client, created, deadline)
            except APIError as exc:
                log(
                    f"Warm memory checkpoint {warm_ckpt} fork failed ({exc}); "
                    "pruning and cold-booting from image"
                )
                store.remove_warm_template(tmpl_key)

    # 5. Cold boot from resolved Prime/Docker Hub image.
    log(
        f"Creating {job.orb_id} from image {resolved_image} "
        f"({spec.cpu_cores:g} vCPU, {spec.memory_gb:g} GiB RAM, {spec.disk_size_gb:g} GiB disk)"
    )
    created = client.create(
        _build_create_request(
            orb_id=job.orb_id,
            orb_type=job.orb_type,
            spec=spec,
            generation=generation,
            docker_image=resolved_image,
        )
    )
    return _poll_until_running(client, created, deadline)


def _execute_when_gateway_ready(
    client: SandboxClient,
    sandbox_id: str,
    script: str,
    *,
    envs: dict[str, str] | None,
    user: str | None,
    deadline: float,
    action_label: str,
) -> None:
    """Run `script` as soon as the guest gateway registers, without a throwaway echo RPC."""
    started = time.monotonic()
    last_error: BaseException | None = None
    attempt = 0

    while True:
        remaining = _remaining(deadline)
        cmd_timeout = max(5, min(30, int(remaining)))
        try:
            result = client.execute_command(
                sandbox_id,
                script,
                working_dir="/",
                env=envs,
                timeout=cmd_timeout,
                user=user,
            )
            if result.exit_code != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"{action_label} failed in sandbox {sandbox_id} "
                    f"(exit {result.exit_code}): {detail}"
                )
            return
        except BaseException as exc:
            if not _is_retryable_bootstrap_error(exc):
                raise
            last_error = exc
            attempt += 1
            if (deadline - time.monotonic()) <= 0.25:
                break
            if attempt >= 2:
                _invalidate_gateway_auth(client, sandbox_id)
                with contextlib.suppress(APIError):
                    sbx = client.get(sandbox_id)
                    if sbx.status in TERMINAL_STATUSES:
                        raise SandboxNotRunningError(
                            sandbox_id,
                            message=(
                                f"Sandbox entered {sbx.status} before {action_label}: "
                                f"{sbx.error_message or sbx.termination_reason or 'no details'}"
                            ),
                        ) from exc
            delay = _fast_poll_delay(time.monotonic() - started)
            time.sleep(min(delay, _remaining(deadline)))

    assert last_error is not None
    raise last_error


def _launch_bootstrap_when_ready(
    client: SandboxClient,
    sandbox_id: str,
    job: DispatchJob,
    config: ProviderConfig,
    deadline: float,
    log: LogFn,
) -> None:
    assert job.orb_token is not None
    envs: dict[str, str] = {"ORB_TOKEN": job.orb_token}
    if job.amp_url:
        envs["AMP_URL"] = job.amp_url

    _execute_when_gateway_ready(
        client,
        sandbox_id,
        BOOTSTRAP_SCRIPT,
        envs=envs,
        user=config.orb_user,
        deadline=deadline,
        action_label="`amp orb bootstrap`",
    )
    log(f"Started `amp orb bootstrap` in sandbox {sandbox_id}")


def build_warm_template(
    config: ProviderConfig,
    *,
    sandbox_client: SandboxClient,
    image_client: ImageClient,
    orb_type: str | None = None,
    update_amp_cli: bool = True,
    timeout_seconds: float = 600.0,
    log_fn: LogFn = default_log,
) -> tuple[str, str]:
    """Boot a clean orb VM, prewarm `amp` in guest RAM, pause it, and store its memory checkpoint.

    Returns `(template_key, durable_checkpoint_id)`. The temporary VM is deleted once its
    pause checkpoint is DURABLE, leaving zero running compute.
    """
    deadline = time.monotonic() + timeout_seconds
    synthetic_job = DispatchJob(
        operation="spawn",
        orb_id="amp-warm-template",
        orb_token="unused",
        orb_type=orb_type,
    )
    spec = config.resolve_spec(synthetic_job)
    if spec.gpu_count > 0:
        raise ValueError(
            "GPU orb types cannot build memory-fork templates (Prime cannot pause GPUs)"
        )

    resolved_image = resolve_prime_image(spec.image, image_client, deadline=deadline)
    key = warm_template_key(resolved_image, spec)
    store = SnapshotStore(config.state_dir)

    template_name = f"amp-orb-template-{int(time.time())}"
    log_fn(
        f"Booting clean template VM ({spec.cpu_cores:g} vCPU, {spec.memory_gb:g} GiB RAM, "
        f"{spec.disk_size_gb:g} GiB disk) from {resolved_image} ..."
    )
    template_spec = replace(spec, timeout_minutes=30, idle_timeout_minutes=None)
    created = sandbox_client.create(
        CreateSandboxRequest(
            name=template_name,
            docker_image=resolved_image,
            arch=template_spec.arch,
            cpu_cores=template_spec.cpu_cores,
            memory_gb=template_spec.memory_gb,
            disk_size_gb=template_spec.disk_size_gb,
            timeout_minutes=30,
            region=template_spec.region,
            labels=[ORB_TEMPLATE_LABEL],
        )
    )
    try:
        _poll_until_running(sandbox_client, created, deadline)
        log_fn(f"Prewarming Amp CLI inside template VM {created.id} ...")
        _execute_when_gateway_ready(
            sandbox_client,
            created.id,
            TEMPLATE_PREWARM_SCRIPT,
            envs={"PRIME_ORB_UPDATE_AMP": "1" if update_amp_cli else "0"},
            user=config.orb_user,
            deadline=deadline,
            action_label="template prewarm",
        )

        log_fn(f"Freezing memory checkpoint on template VM {created.id} ...")
        paused = sandbox_client.pause(created.id)
        _poll_until_paused(sandbox_client, created.id, paused.error_message, deadline)
        durable = sandbox_client.wait_for_checkpoint(
            paused.checkpoint_id, timeout_seconds=_remaining(deadline, floor=5.0)
        )
    finally:
        with contextlib.suppress(APIError):
            sandbox_client.delete(created.id)

    previous_ckpt = store.set_warm_template(key, durable.id)
    if previous_ckpt and previous_ckpt != durable.id and (
        previous_ckpt not in store.referenced_checkpoint_ids()
    ):
        _safe_delete_checkpoint(sandbox_client, previous_ckpt, deadline)

    log_fn(f"Saved warm memory checkpoint {durable.id} for {key}")
    return key, durable.id


def _poll_until_paused(
    client: SandboxClient,
    sandbox_id: str,
    prior_error: str | None,
    deadline: float,
) -> None:
    started = time.monotonic()
    while True:
        remaining = _remaining(deadline)
        current = client.get(sandbox_id)
        if current.status == "PAUSED":
            return
        if current.status in TERMINAL_STATUSES:
            raise RuntimeError(
                f"Sandbox {sandbox_id} entered {current.status} while pausing: "
                f"{current.error_message or current.termination_reason or 'no details'}"
            )
        if (
            current.status == "RUNNING"
            and current.error_message
            and current.error_message != prior_error
        ):
            raise RuntimeError(f"Pause failed for sandbox {sandbox_id}: {current.error_message}")
        time.sleep(min(_fast_poll_delay(time.monotonic() - started), remaining))


def _record_project_snapshot_from_checkpoint(
    *,
    orb_snapshot_id: str,
    checkpoint_id: str,
    sandbox: Sandbox,
    config: ProviderConfig,
    client: SandboxClient,
    store: SnapshotStore,
    deadline: float,
    log: LogFn,
    on_snapshot_saved: SnapshotSavedFn | None,
) -> None:
    existing_ckpt, existing_mode = store.get_project_snapshot_entry(orb_snapshot_id)
    if existing_ckpt and _get_durable_checkpoint_id(client, existing_ckpt):
        if existing_mode == "disk" and on_snapshot_saved is not None:
            snap_spec = replace(
                config.default_spec,
                cpu_cores=sandbox.cpu_cores,
                memory_gb=sandbox.memory_gb,
                disk_size_gb=sandbox.disk_size_gb,
                gpu_count=sandbox.gpu_count,
                gpu_type=sandbox.gpu_type,
                region=sandbox.region,
            )
            on_snapshot_saved(orb_snapshot_id, existing_ckpt, snap_spec)
        return

    durable = client.wait_for_checkpoint(
        checkpoint_id, timeout_seconds=_remaining(deadline, floor=5.0)
    )
    previous = store.set_project_snapshot(orb_snapshot_id, durable.id, mode="disk")
    if previous and previous != durable.id and previous not in store.referenced_checkpoint_ids():
        _safe_delete_checkpoint(client, previous, deadline)

    log(f"Saved project snapshot {orb_snapshot_id} -> checkpoint {durable.id}")
    if on_snapshot_saved is not None and sandbox.gpu_count == 0:
        snap_spec = replace(
            config.default_spec,
            cpu_cores=sandbox.cpu_cores,
            memory_gb=sandbox.memory_gb,
            disk_size_gb=sandbox.disk_size_gb,
            gpu_count=sandbox.gpu_count,
            gpu_type=sandbox.gpu_type,
            region=sandbox.region,
        )
        on_snapshot_saved(orb_snapshot_id, durable.id, snap_spec)


def _handle_pause(
    job: DispatchJob,
    config: ProviderConfig,
    client: SandboxClient,
    store: SnapshotStore,
    deadline: float,
    log: LogFn,
    *,
    authoritative_index: bool = False,
    on_snapshot_saved: SnapshotSavedFn | None = None,
) -> None:
    sandbox, _ = _resolve_existing_orb_sandbox(
        client, store, job.orb_id, log, authoritative_index=authoritative_index
    )
    if sandbox is None:
        log(f"No active sandbox for {job.orb_id}; pause is a no-op")
        return

    if sandbox.status in ("PENDING", "PROVISIONING"):
        sandbox = _poll_until_running(client, sandbox, deadline)

    want_snapshot = bool(config.snapshots_enabled and job.orb_snapshot_id)

    # Already PAUSED: reuse its existing pause checkpoint if a project snapshot was requested.
    if sandbox.status == "PAUSED":
        if want_snapshot and job.orb_snapshot_id:
            ckpt_id = store.get_orb_checkpoint(job.orb_id)
            if not ckpt_id:
                checkpoints = client.list_checkpoints(sandbox.id)
                ckpt_id = checkpoints[-1].id if checkpoints else None
            if ckpt_id:
                _record_project_snapshot_from_checkpoint(
                    orb_snapshot_id=job.orb_snapshot_id,
                    checkpoint_id=ckpt_id,
                    sandbox=sandbox,
                    config=config,
                    client=client,
                    store=store,
                    deadline=deadline,
                    log=log,
                    on_snapshot_saved=on_snapshot_saved,
                )
        return

    if sandbox.status in TERMINAL_STATUSES:
        return

    # Prime GPU sandboxes cannot pause in memory; checkpoint rootfs once and delete compute.
    if sandbox.gpu_count > 0:
        _checkpoint_and_delete_for_pause(
            client=client,
            store=store,
            sandbox=sandbox,
            orb_id=job.orb_id,
            orb_snapshot_id=job.orb_snapshot_id if want_snapshot else None,
            deadline=deadline,
            log=log,
        )
        return

    # Standard CPU VM: a single `pause()` freezes both memory and a cold-bootable disk checkpoint!
    try:
        paused = client.pause(sandbox.id)
    except APIError as exc:
        if _is_pause_unsupported_error(exc):
            log(
                f"VM pause unsupported for sandbox {sandbox.id} ({exc}); "
                "falling back to filesystem checkpoint + delete"
            )
            _checkpoint_and_delete_for_pause(
                client=client,
                store=store,
                sandbox=sandbox,
                orb_id=job.orb_id,
                orb_snapshot_id=job.orb_snapshot_id if want_snapshot else None,
                deadline=deadline,
                log=log,
            )
            return
        raise

    pause_ckpt_id = getattr(paused, "checkpoint_id", None)
    if pause_ckpt_id:
        store.set_orb_checkpoint(job.orb_id, pause_ckpt_id)

    _poll_until_paused(client, sandbox.id, paused.error_message, deadline)
    log(f"Paused sandbox {sandbox.id} for {job.orb_id}")

    if want_snapshot and job.orb_snapshot_id:
        snap_source_ckpt = pause_ckpt_id
        if not snap_source_ckpt:
            checkpoints = client.list_checkpoints(sandbox.id)
            snap_source_ckpt = checkpoints[-1].id if checkpoints else None
        if snap_source_ckpt:
            _record_project_snapshot_from_checkpoint(
                orb_snapshot_id=job.orb_snapshot_id,
                checkpoint_id=snap_source_ckpt,
                sandbox=sandbox,
                config=config,
                client=client,
                store=store,
                deadline=deadline,
                log=log,
                on_snapshot_saved=on_snapshot_saved,
            )


def _is_pause_unsupported_error(exc: APIError) -> bool:
    lowered = str(exc).lower()
    return any(
        marker in lowered
        for marker in ("volume", "gpu", "cannot be paused", "can't be paused", "not supported")
    )


def _checkpoint_and_delete_for_pause(
    *,
    client: SandboxClient,
    store: SnapshotStore,
    sandbox: Sandbox,
    orb_id: str,
    orb_snapshot_id: str | None,
    deadline: float,
    log: LogFn,
) -> None:
    ckpt = client.checkpoint(sandbox.id, ttl_seconds=None)
    durable = client.wait_for_checkpoint(
        ckpt.id, timeout_seconds=_remaining(deadline, floor=5.0)
    )
    store.set_orb_checkpoint(orb_id, durable.id)
    if orb_snapshot_id:
        store.set_project_snapshot(orb_snapshot_id, durable.id, mode="disk")
    store.remove_orb_sandbox(orb_id)
    _safe_delete_sandbox(client, sandbox.id)
    log(f"Checkpointed ({durable.id}) and terminated non-pausable sandbox {sandbox.id}")


def expire_stale_project_snapshots(
    client: SandboxClient,
    store: SnapshotStore,
    *,
    ttl_seconds: int | None,
    log_fn: LogFn = default_log,
) -> int:
    """Delete project snapshot checkpoints older than `ttl_seconds` (including memory-promoted)."""
    if not ttl_seconds or ttl_seconds <= 0:
        return 0
    now = datetime.now(UTC)
    expired = 0
    deadline = time.monotonic() + 60.0

    for snap_id, ckpt_id in list(store.all_project_snapshots().items()):
        try:
            ckpt = client.get_checkpoint(ckpt_id)
        except APIError:
            store.remove_project_snapshot(snap_id)
            expired += 1
            continue

        age = (now - _aware_utc(ckpt.created_at)).total_seconds()
        if ckpt.state != "DURABLE" or ckpt.merge == "failed" or age >= ttl_seconds:
            store.remove_project_snapshot(snap_id)
            if ckpt_id not in store.referenced_checkpoint_ids():
                _safe_delete_checkpoint(client, ckpt_id, deadline)
            log_fn(f"Expired project snapshot {snap_id} (checkpoint {ckpt_id}, age={int(age)}s)")
            expired += 1
    return expired


def _handle_delete(
    job: DispatchJob,
    client: SandboxClient,
    store: SnapshotStore,
    deadline: float,
    log: LogFn,
) -> None:
    active, terminated = list_sandboxes_for_orb(client, job.orb_id)
    cached_id = store.get_orb_sandbox(job.orb_id)
    known_ids = {s.id for s in active} | {s.id for s in terminated}
    if cached_id and cached_id not in known_ids:
        with contextlib.suppress(APIError):
            sbx = client.get(cached_id)
            if sbx.status in TERMINAL_STATUSES:
                terminated.append(sbx)
            else:
                active.append(sbx)

    all_sandboxes = [*active, *terminated]

    for sbx in active:
        log(f"Deleting sandbox {sbx.id} for {job.orb_id}")
        _safe_delete_sandbox(client, sbx.id)

    store.remove_orb_checkpoint(job.orb_id)
    store.remove_orb_sandbox(job.orb_id)

    protected_checkpoints = store.referenced_checkpoint_ids()
    candidate_checkpoint_ids: set[str] = set()

    for sbx in all_sandboxes:
        try:
            for ckpt in client.list_checkpoints(sbx.id):
                if ckpt.id not in protected_checkpoints:
                    candidate_checkpoint_ids.add(ckpt.id)
        except APIError:
            continue

    for ckpt_id in sorted(candidate_checkpoint_ids):
        _safe_delete_checkpoint(client, ckpt_id, deadline)


def _safe_delete_sandbox(client: SandboxClient, sandbox_id: str) -> None:
    try:
        client.delete(sandbox_id)
    except APIError as exc:
        if "HTTP 404" in str(exc) or "not found" in str(exc).lower():
            return
        raise


def _safe_delete_checkpoint(
    client: SandboxClient, checkpoint_id: str, deadline: float
) -> None:
    delay = 0.25
    for attempt in range(4):
        try:
            client.delete_checkpoint(checkpoint_id)
            return
        except APIError as exc:
            msg = str(exc)
            if "HTTP 404" in msg or "not found" in msg.lower():
                return
            if "HTTP 409" in msg and attempt < 3 and (deadline - time.monotonic()) > delay:
                time.sleep(delay)
                delay *= 2.0
                continue
            default_log(f"Warning: could not delete checkpoint {checkpoint_id}: {exc}")
            return
