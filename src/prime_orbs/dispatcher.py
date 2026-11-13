"""Core Amp self-hosted orb lifecycle dispatcher backed by Prime Sandboxes."""

from __future__ import annotations

import contextlib
import hashlib
import sys
import time
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

from .config import DispatchJob, OrbSpec, ProviderConfig
from .images import resolve_prime_image
from .state import SnapshotStore, orb_lock

ORB_MANAGED_LABEL = "amp-orb"
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


def orb_id_label(orb_id: str) -> str:
    return f"amp-orb:{orb_id}"


def orb_labels(orb_id: str, orb_type: str | None = None) -> list[str]:
    labels = [ORB_MANAGED_LABEL, orb_id_label(orb_id)]
    if orb_type:
        labels.append(f"amp-orb-type:{orb_type}")
    return labels


def _log(message: str) -> None:
    print(f"[prime-orbs] {message}", file=sys.stderr, flush=True)


def _aware_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def _remaining(deadline: float, *, floor: float = 1.0) -> float:
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("Dispatch operation exceeded configured time budget")
    return max(floor, left)


def _max_attempts_for_phase_seconds(phase_seconds: float) -> int:
    # Matches prime_sandboxes._creation_timeout_seconds (_LEGACY_FAST_POLLS = 5).
    clamped = max(5.0, phase_seconds)
    if clamped <= 5.0:
        return 5
    return max(5, int(5 + (clamped - 5.0) / 2.0))


def _deterministic_idempotency_key(orb_id: str, generation: int, source_tag: str) -> str:
    digest = hashlib.sha256(f"{orb_id}:{generation}:{source_tag}".encode()).hexdigest()
    return digest[:32]


def _invalidate_gateway_auth(client: SandboxClient, sandbox_id: str) -> None:
    auth_cache = getattr(client, "_auth_cache", None)
    invalidate = getattr(auth_cache, "invalidate", None)
    if callable(invalidate):
        with contextlib.suppress(Exception):
            invalidate(sandbox_id)


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


def run_dispatch(
    job: DispatchJob,
    config: ProviderConfig,
    *,
    sandbox_client: SandboxClient | None = None,
    image_client: ImageClient | None = None,
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
            _handle_spawn(job, config, sandbox_client, image_client, store, deadline)
        elif job.operation == "pause":
            _handle_pause(job, config, sandbox_client, store, deadline)
        elif job.operation == "delete":
            _handle_delete(job, sandbox_client, store, deadline)
        else:
            raise ValueError(f"Unsupported operation: {job.operation!r}")


def _handle_spawn(
    job: DispatchJob,
    config: ProviderConfig,
    client: SandboxClient,
    image_client: ImageClient,
    store: SnapshotStore,
    deadline: float,
) -> None:
    spec = config.resolve_spec(job)
    active, terminated = list_sandboxes_for_orb(client, job.orb_id)

    # Clean up any unexpected older duplicate active sandboxes so only the newest remains.
    for extra in active[1:]:
        _log(f"Deleting duplicate active sandbox {extra.id} for {job.orb_id}")
        _safe_delete_sandbox(client, extra.id)

    running_sandbox: Sandbox | None = None
    if active:
        candidate = client.get(active[0].id)
        if candidate.status in TERMINAL_STATUSES:
            terminated.insert(0, candidate)
        else:
            running_sandbox = _ensure_existing_running(
                client, candidate, job.orb_id, deadline
            )
            if running_sandbox is None:
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
        )

    _launch_bootstrap(client, running_sandbox.id, job, config, deadline)


def _ensure_existing_running(
    client: SandboxClient,
    sandbox: Sandbox,
    orb_id: str,
    deadline: float,
) -> Sandbox | None:
    """Resume or wait on an existing sandbox; return None only if its node died irrecoverably."""
    if sandbox.status == "PAUSED":
        _log(f"Resuming paused sandbox {sandbox.id} for {orb_id}")
        try:
            _resume_until_running(client, sandbox, deadline)
        except SandboxNotRunningError as exc:
            _log(
                f"Paused sandbox {sandbox.id} could not resume ({exc}); "
                "falling back to checkpoint/image recovery"
            )
            return None
    elif sandbox.status in ("PENDING", "PROVISIONING"):
        _log(f"Waiting for in-flight sandbox {sandbox.id} ({sandbox.status})")

    _wait_for_gateway_ready(client, sandbox.id, deadline)
    return sandbox


def _resume_until_running(
    client: SandboxClient,
    sandbox: Sandbox,
    deadline: float,
) -> None:
    accepted = _request_resume_with_retry(client, sandbox.id, deadline)
    baseline_error = accepted.error_message
    _invalidate_gateway_auth(client, sandbox.id)

    retried_after_node_error = False
    delay = 0.5
    while True:
        remaining = _remaining(deadline)
        current = client.get(sandbox.id)
        if current.status == "RUNNING":
            _invalidate_gateway_auth(client, sandbox.id)
            return
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
                _log(f"Resume reported error ({current.error_message}); retrying once")
                accepted_retry = _request_resume_with_retry(client, sandbox.id, deadline)
                baseline_error = accepted_retry.error_message
            else:
                raise RuntimeError(
                    f"Failed to resume paused sandbox {sandbox.id}: {current.error_message}"
                )
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 3.0)


def _request_resume_with_retry(
    client: SandboxClient, sandbox_id: str, deadline: float
) -> Sandbox:
    delay = 0.5
    for attempt in range(5):
        try:
            return client.resume(sandbox_id)
        except APIError as exc:
            msg = str(exc)
            is_conflict_or_5xx = any(
                code in msg for code in ("HTTP 409", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504")
            )
            if not is_conflict_or_5xx or attempt == 4:
                raise
            remaining = _remaining(deadline)
            time.sleep(min(delay, remaining))
            delay = min(delay * 2.0, 4.0)
    raise AssertionError("unreachable")


def _wait_for_gateway_ready(
    client: SandboxClient, sandbox_id: str, deadline: float
) -> None:
    remaining = _remaining(deadline, floor=10.0)
    phase_seconds = max(10.0, remaining / 2.0)
    max_attempts = _max_attempts_for_phase_seconds(phase_seconds)
    image_build_budget = max(10, int(remaining - 15.0))
    client.wait_for_creation(
        sandbox_id,
        max_attempts=max_attempts,
        image_build_timeout_seconds=image_build_budget,
    )


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
) -> CreateSandboxRequest:
    source_tag = f"ckpt:{checkpoint_id}" if checkpoint_id else f"img:{docker_image}"
    common = dict(
        name=orb_id,
        cpu_cores=spec.cpu_cores,
        memory_gb=spec.memory_gb,
        gpu_count=spec.gpu_count,
        gpu_type=spec.gpu_type,
        timeout_minutes=spec.timeout_minutes,
        idle_timeout_minutes=spec.idle_timeout_minutes,
        region=spec.region,
        network_allowlist=spec.network_allowlist,
        network_denylist=spec.network_denylist,
        labels=orb_labels(orb_id, orb_type),
        idempotency_key=_deterministic_idempotency_key(orb_id, generation, source_tag),
    )
    if checkpoint_id is not None:
        # Prime checkpoint restores take image, disk_size_gb, and arch from the checkpoint.
        return CreateSandboxRequest(checkpoint_id=checkpoint_id, **common)
    return CreateSandboxRequest(
        docker_image=docker_image,
        arch=spec.arch,
        disk_size_gb=spec.disk_size_gb,
        **common,
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
) -> Sandbox:
    generation = len(terminated)

    # 1. Recover this orb's disk if a previous sandbox was checkpointed (GPU pause or lost node).
    recovery_ckpt = _find_orb_recovery_checkpoint(client, store, job.orb_id, terminated)
    if recovery_ckpt:
        _log(f"Restoring {job.orb_id} from orb checkpoint {recovery_ckpt}")
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
            _wait_for_gateway_ready(client, created.id, deadline)
            return created
        except APIError as exc:
            _log(f"Orb checkpoint {recovery_ckpt} restore failed ({exc}); falling back")
            store.remove_orb_checkpoint(job.orb_id)

    # 2. Boot from shared Amp project snapshot (ORB_SNAPSHOT_ID) when available.
    if config.snapshots_enabled and job.orb_snapshot_id:
        mapped_ckpt = store.get_project_snapshot(job.orb_snapshot_id)
        if mapped_ckpt:
            durable = _get_durable_checkpoint_id(client, mapped_ckpt)
            if durable is None:
                _log(
                    f"Pruning expired/missing project snapshot "
                    f"{job.orb_snapshot_id} -> {mapped_ckpt}"
                )
                store.remove_project_snapshot(job.orb_snapshot_id)
            else:
                _log(
                    f"Creating {job.orb_id} from project snapshot "
                    f"{job.orb_snapshot_id} (checkpoint {durable})"
                )
                try:
                    created = client.create(
                        _build_create_request(
                            orb_id=job.orb_id,
                            orb_type=job.orb_type,
                            spec=spec,
                            generation=generation,
                            checkpoint_id=durable,
                        )
                    )
                    _wait_for_gateway_ready(client, created.id, deadline)
                    return created
                except APIError as exc:
                    _log(
                        f"Project snapshot {job.orb_snapshot_id} restore failed ({exc}); "
                        "falling back to base image"
                    )
                    store.remove_project_snapshot(job.orb_snapshot_id)

    # 3. Cold boot from resolved Prime/Docker Hub image.
    resolved_image = resolve_prime_image(spec.image, image_client, deadline=deadline)
    _log(
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
    _wait_for_gateway_ready(client, created.id, deadline)
    return created


def _launch_bootstrap(
    client: SandboxClient,
    sandbox_id: str,
    job: DispatchJob,
    config: ProviderConfig,
    deadline: float,
) -> None:
    assert job.orb_token is not None
    envs: dict[str, str] = {"ORB_TOKEN": job.orb_token}
    if job.amp_url:
        envs["AMP_URL"] = job.amp_url

    last_error: Exception | None = None
    for attempt in range(3):
        cmd_timeout = max(10, min(30, int(_remaining(deadline, floor=10.0))))
        try:
            result = client.execute_command(
                sandbox_id,
                BOOTSTRAP_SCRIPT,
                working_dir="/",
                env=envs,
                timeout=cmd_timeout,
                user=config.orb_user,
            )
            if result.exit_code != 0:
                detail = (result.stderr or result.stdout).strip()
                raise RuntimeError(
                    f"`amp orb bootstrap` failed in sandbox {sandbox_id} "
                    f"(exit {result.exit_code}): {detail}"
                )
            _log(f"Started `amp orb bootstrap` in sandbox {sandbox_id}")
            return
        except (APIError, CommandTimeoutError) as exc:
            last_error = exc
            if attempt == 2:
                break
            _invalidate_gateway_auth(client, sandbox_id)
            time.sleep(min(1.0 * (2**attempt), _remaining(deadline)))

    assert last_error is not None
    raise last_error


def _handle_pause(
    job: DispatchJob,
    config: ProviderConfig,
    client: SandboxClient,
    store: SnapshotStore,
    deadline: float,
) -> None:
    active, _ = list_sandboxes_for_orb(client, job.orb_id)
    if not active:
        _log(f"No active sandbox for {job.orb_id}; pause is a no-op")
        return

    sandbox = client.get(active[0].id)
    if sandbox.status in TERMINAL_STATUSES:
        return

    if sandbox.status in ("PENDING", "PROVISIONING"):
        _wait_for_gateway_ready(client, sandbox.id, deadline)
        sandbox = client.get(sandbox.id)

    created_snapshot_ckpt_id: str | None = None
    if config.snapshots_enabled and job.orb_snapshot_id:
        existing_snap = store.get_project_snapshot(job.orb_snapshot_id)
        if existing_snap and _get_durable_checkpoint_id(client, existing_snap):
            created_snapshot_ckpt_id = existing_snap
        elif sandbox.status == "RUNNING":
            _log(
                f"Creating project snapshot {job.orb_snapshot_id} from sandbox {sandbox.id}"
            )
            ckpt = client.checkpoint(sandbox.id, ttl_seconds=config.snapshot_ttl_seconds)
            durable = client.wait_for_checkpoint(
                ckpt.id, timeout_seconds=_remaining(deadline, floor=5.0)
            )
            store.set_project_snapshot(job.orb_snapshot_id, durable.id)
            created_snapshot_ckpt_id = durable.id
        elif sandbox.status == "PAUSED":
            checkpoints = client.list_checkpoints(sandbox.id)
            if checkpoints:
                durable = client.wait_for_checkpoint(
                    checkpoints[-1].id, timeout_seconds=_remaining(deadline, floor=5.0)
                )
                store.set_project_snapshot(job.orb_snapshot_id, durable.id)
                created_snapshot_ckpt_id = durable.id
        sandbox = client.get(sandbox.id)

    if sandbox.status == "PAUSED" or sandbox.status in TERMINAL_STATUSES:
        return

    # Prime GPU sandboxes cannot pause in memory; checkpoint rootfs and delete compute instead.
    if sandbox.gpu_count > 0:
        _checkpoint_and_delete_for_pause(
            client=client,
            store=store,
            sandbox=sandbox,
            orb_id=job.orb_id,
            reusable_checkpoint_id=created_snapshot_ckpt_id,
            deadline=deadline,
        )
        return

    try:
        paused = client.pause(sandbox.id)
    except APIError as exc:
        if _is_pause_unsupported_error(exc):
            _log(
                f"VM pause unsupported for sandbox {sandbox.id} ({exc}); "
                "falling back to filesystem checkpoint + delete"
            )
            _checkpoint_and_delete_for_pause(
                client=client,
                store=store,
                sandbox=sandbox,
                orb_id=job.orb_id,
                reusable_checkpoint_id=created_snapshot_ckpt_id,
                deadline=deadline,
            )
            return
        raise

    prior_error = paused.error_message

    if getattr(paused, "checkpoint_id", None):
        store.set_orb_checkpoint(job.orb_id, paused.checkpoint_id)

    delay = 0.5
    while True:
        remaining = _remaining(deadline)
        current = client.get(sandbox.id)
        if current.status == "PAUSED":
            _log(f"Paused sandbox {sandbox.id} for {job.orb_id}")
            return
        if current.status in TERMINAL_STATUSES:
            raise RuntimeError(
                f"Sandbox {sandbox.id} entered {current.status} while pausing: "
                f"{current.error_message or current.termination_reason or 'no details'}"
            )
        if (
            current.status == "RUNNING"
            and current.error_message
            and current.error_message != prior_error
        ):
            raise RuntimeError(
                f"Pause failed for sandbox {sandbox.id}: {current.error_message}"
            )
        time.sleep(min(delay, remaining))
        delay = min(delay * 1.5, 3.0)


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
    reusable_checkpoint_id: str | None,
    deadline: float,
) -> None:
    # Project snapshots carry an expiry TTL; create a non-expiring checkpoint for this orb's disk.
    ckpt = client.checkpoint(sandbox.id, ttl_seconds=None)
    durable = client.wait_for_checkpoint(
        ckpt.id, timeout_seconds=_remaining(deadline, floor=5.0)
    )
    store.set_orb_checkpoint(orb_id, durable.id)
    _safe_delete_sandbox(client, sandbox.id)
    _log(
        f"Checkpointed ({durable.id}) and terminated non-pausable sandbox {sandbox.id} "
        f"(reusable snapshot={reusable_checkpoint_id})"
    )


def _handle_delete(
    job: DispatchJob,
    client: SandboxClient,
    store: SnapshotStore,
    deadline: float,
) -> None:
    active, terminated = list_sandboxes_for_orb(client, job.orb_id)
    all_sandboxes = [*active, *terminated]

    for sbx in active:
        _log(f"Deleting sandbox {sbx.id} for {job.orb_id}")
        _safe_delete_sandbox(client, sbx.id)

    protected_checkpoints = set(store.all_project_snapshots().values())
    candidate_checkpoint_ids: set[str] = set()

    recorded_orb_ckpt = store.get_orb_checkpoint(job.orb_id)
    if recorded_orb_ckpt and recorded_orb_ckpt not in protected_checkpoints:
        candidate_checkpoint_ids.add(recorded_orb_ckpt)

    for sbx in all_sandboxes:
        try:
            for ckpt in client.list_checkpoints(sbx.id):
                if ckpt.id not in protected_checkpoints:
                    candidate_checkpoint_ids.add(ckpt.id)
        except APIError:
            continue

    for ckpt_id in sorted(candidate_checkpoint_ids):
        _safe_delete_checkpoint(client, ckpt_id, deadline)

    store.remove_orb_checkpoint(job.orb_id)


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
    delay = 0.5
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
            _log(f"Warning: could not delete checkpoint {checkpoint_id}: {exc}")
            return
