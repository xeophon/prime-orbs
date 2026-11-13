"""Behavioral tests for the Amp self-hosted orb lifecycle dispatcher on Prime Sandboxes."""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from prime_sandboxes import (
    APIError,
    CommandResponse,
    CreateSandboxRequest,
    ImageListResponse,
    PausedSandbox,
    Sandbox,
    SandboxCheckpoint,
    SandboxListResponse,
    SandboxNotRunningError,
)

from prime_orbs.config import DEFAULT_AMP_ORB_IMAGE_ARM64, DispatchJob, ProviderConfig
from prime_orbs.dispatcher import build_warm_template, run_dispatch
from prime_orbs.state import SnapshotStore


class FakePrimeBackend:
    """In-memory simulation of Prime Sandboxes & Images APIs using real SDK Pydantic models."""

    def __init__(self) -> None:
        self._clock = datetime(2026, 11, 11, 12, 0, 0, tzinfo=UTC)
        self._next_sbx = 1
        self._next_ckpt = 1
        self.sandboxes: dict[str, dict[str, Any]] = {}
        self.checkpoints: dict[str, dict[str, Any]] = {}
        self.create_requests: list[CreateSandboxRequest] = []
        self.exec_calls: list[dict[str, Any]] = []
        self.pause_calls: list[str] = []
        self.resume_calls: list[str] = []
        self.delete_calls: list[str] = []
        self.deleted_checkpoints: list[str] = []
        self.invalidated_auth_ids: list[str] = []

        # Fault-injection hooks armed when pause() / resume() are invoked:
        self.next_pause_statuses: dict[str, list[tuple[str, str | None]]] = {}
        self.next_resume_statuses: dict[str, list[tuple[str, str | None]]] = {}
        self._active_poll_queue: dict[str, list[tuple[str, str | None]]] = {}
        self.resume_api_errors: list[APIError] = []

        self._auth_cache = self

    def _tick(self) -> str:
        self._clock += timedelta(seconds=1)
        return self._clock.isoformat()

    def invalidate(self, sandbox_id: str) -> None:
        self.invalidated_auth_ids.append(sandbox_id)

    # --- ImageClient surface ---
    def list_images_impl(self, **_kwargs: Any) -> ImageListResponse:
        return ImageListResponse.model_validate(
            {
                "data": [
                    {
                        "id": "img-1",
                        "artifactType": "VM_SANDBOX",
                        "imageName": "amp-orb",
                        "imageTag": "latest",
                        "status": "COMPLETED",
                        "visibility": "PRIVATE",
                        "createdAt": "2026-11-11T00:00:00Z",
                        "displayRef": "prime/xeophon/amp-orb:latest",
                    },
                    {
                        "id": "img-2",
                        "artifactType": "VM_SANDBOX",
                        "imageName": "amp-orb-arm64",
                        "imageTag": "latest",
                        "status": "COMPLETED",
                        "visibility": "PRIVATE",
                        "createdAt": "2026-11-11T00:00:00Z",
                        "displayRef": "prime/xeophon/amp-orb-arm64:latest",
                    },
                ],
                "totalCount": 2,
                "offset": 0,
                "limit": 100,
            }
        )

    # --- SandboxClient surface ---
    def create(self, request: CreateSandboxRequest) -> Sandbox:
        # Validate wire contract rules enforced by Prime's API:
        if request.checkpoint_id is not None:
            assert request.docker_image is None
            assert request.disk_size_gb is None
            assert request.arch is None, "Prime rejects arch on checkpoint restores"
            if request.restore_memory:
                for shape_field in ("cpu_cores", "memory_gb", "gpu_count", "gpu_type"):
                    assert (
                        shape_field not in request.model_fields_set
                    ), f"Memory restore must omit {shape_field} so paused shape is inherited"
            ckpt = self.checkpoints.get(request.checkpoint_id)
            if ckpt is None or ckpt["state"] != "DURABLE":
                raise APIError(f"HTTP 404: Checkpoint {request.checkpoint_id} not found")
            docker_image = ckpt["docker_image"]
            disk_size_gb = 20.0
        else:
            assert request.docker_image is not None
            assert request.docker_image.startswith("prime/") or "/" not in request.docker_image
            docker_image = request.docker_image
            disk_size_gb = request.disk_size_gb or 20.0

        self.create_requests.append(request)
        sbx_id = f"sbx_{self._next_sbx}"
        self._next_sbx += 1
        now = self._tick()
        row = {
            "id": sbx_id,
            "name": request.name,
            "dockerImage": docker_image,
            "cpuCores": request.cpu_cores,
            "memoryGB": request.memory_gb,
            "diskSizeGB": disk_size_gb,
            "diskMountPath": "/sandbox-workspace",
            "gpuCount": request.gpu_count,
            "gpuType": request.gpu_type,
            "vm": True,
            "status": "RUNNING",
            "timeoutMinutes": request.timeout_minutes,
            "idleTimeoutMinutes": request.idle_timeout_minutes,
            "labels": list(request.labels),
            "createdAt": now,
            "updatedAt": now,
            "errorMessage": None,
        }
        self.sandboxes[sbx_id] = row
        return Sandbox.model_validate(row)

    def list(
        self,
        *,
        labels: list[str] | None = None,
        per_page: int = 50,
        cursor: str | None = None,
        **_kwargs: Any,
    ) -> SandboxListResponse:
        del per_page
        matching = [
            row
            for row in self.sandboxes.values()
            if not labels or all(lbl in row["labels"] for lbl in labels)
        ]
        # Paginate 1 item per page when >1 exists to exercise cursor pagination thoroughly!
        idx = int(cursor) if cursor is not None else 0
        slice_rows = matching[idx : idx + 1] if len(matching) > 1 else matching
        has_next = len(matching) > 1 and (idx + 1) < len(matching)
        next_cursor = str(idx + 1) if has_next else None
        return SandboxListResponse.model_validate(
            {
                "sandboxes": slice_rows,
                "total": len(matching),
                "page": 1,
                "perPage": 1,
                "hasNext": has_next,
                "next_cursor": next_cursor,
            }
        )

    def get(self, sandbox_id: str) -> Sandbox:
        if sandbox_id not in self.sandboxes:
            raise APIError(f"HTTP 404: Sandbox {sandbox_id} not found")
        row = dict(self.sandboxes[sandbox_id])
        queue = self._active_poll_queue.get(sandbox_id)
        if queue:
            status, err = queue.pop(0)
            row["status"] = status
            row["errorMessage"] = err
            self.sandboxes[sandbox_id].update(row)
        return Sandbox.model_validate(row)

    def wait_for_creation(
        self,
        sandbox_id: str,
        max_attempts: int = 60,
        image_build_timeout_seconds: int = 3000,
    ) -> None:
        assert max_attempts >= 5
        assert image_build_timeout_seconds > 0
        sbx = self.get(sandbox_id)
        if sbx.status != "RUNNING":
            raise SandboxNotRunningError(sandbox_id, message=f"status={sbx.status}")

    def pause(self, sandbox_id: str) -> PausedSandbox:
        row = self.sandboxes[sandbox_id]
        if row["gpuCount"] > 0:
            raise APIError("HTTP 400: GPU sandboxes cannot be paused")
        self.pause_calls.append(sandbox_id)
        row["errorMessage"] = None
        ckpt = self._make_checkpoint(sandbox_id, ttl_seconds=None)
        returned = {**row, "checkpointId": ckpt["id"]}
        self._active_poll_queue[sandbox_id] = list(
            self.next_pause_statuses.pop(
                sandbox_id, [("RUNNING", None), ("PAUSED", None)]
            )
        )
        return PausedSandbox.model_validate(returned)

    def resume(self, sandbox_id: str) -> Sandbox:
        if self.resume_api_errors:
            raise self.resume_api_errors.pop(0)
        self.resume_calls.append(sandbox_id)
        row = self.sandboxes[sandbox_id]
        row["errorMessage"] = None
        self._active_poll_queue[sandbox_id] = list(
            self.next_resume_statuses.pop(
                sandbox_id, [("PAUSED", None), ("RUNNING", None)]
            )
        )
        return Sandbox.model_validate(row)

    def delete(self, sandbox_id: str) -> dict[str, Any]:
        self.delete_calls.append(sandbox_id)
        if sandbox_id not in self.sandboxes:
            raise APIError(f"HTTP 404: Sandbox {sandbox_id} not found")
        self.sandboxes[sandbox_id]["status"] = "TERMINATED"
        return {"status": "deleted"}

    def _make_checkpoint(self, sandbox_id: str, ttl_seconds: int | None) -> dict[str, Any]:
        ckpt_id = f"ckpt_{self._next_ckpt}"
        self._next_ckpt += 1
        now = self._tick()
        expires_at = (
            (self._clock + timedelta(seconds=ttl_seconds)).isoformat()
            if ttl_seconds
            else None
        )
        ckpt = {
            "id": ckpt_id,
            "sandbox_id": sandbox_id,
            "state": "DURABLE",
            "depth": 1,
            "docker_image": self.sandboxes[sandbox_id]["dockerImage"],
            "expires_at": expires_at,
            "created_at": now,
            "updated_at": now,
        }
        self.checkpoints[ckpt_id] = ckpt
        return ckpt

    def checkpoint(
        self, sandbox_id: str, ttl_seconds: int | None = None
    ) -> SandboxCheckpoint:
        return SandboxCheckpoint.model_validate(self._make_checkpoint(sandbox_id, ttl_seconds))

    def wait_for_checkpoint(
        self, checkpoint_id: str, timeout_seconds: float = 300
    ) -> SandboxCheckpoint:
        assert timeout_seconds > 0
        return self.get_checkpoint(checkpoint_id)

    def get_checkpoint(self, checkpoint_id: str) -> SandboxCheckpoint:
        if checkpoint_id not in self.checkpoints:
            raise APIError(f"HTTP 404: Checkpoint {checkpoint_id} not found")
        return SandboxCheckpoint.model_validate(self.checkpoints[checkpoint_id])

    def list_checkpoints(self, sandbox_id: str) -> list[SandboxCheckpoint]:
        return [
            SandboxCheckpoint.model_validate(c)
            for c in self.checkpoints.values()
            if c["sandbox_id"] == sandbox_id
        ]

    def list_all_checkpoints(
        self,
        *,
        sandbox_id: str | None = None,
        states: list[str] | None = None,
        **_kwargs: Any,
    ) -> list[SandboxCheckpoint]:
        return [
            SandboxCheckpoint.model_validate(c)
            for c in self.checkpoints.values()
            if (sandbox_id is None or c["sandbox_id"] == sandbox_id)
            and (states is None or c["state"] in states)
        ]

    def delete_checkpoint(self, checkpoint_id: str) -> None:
        self.deleted_checkpoints.append(checkpoint_id)
        if checkpoint_id not in self.checkpoints:
            raise APIError(f"HTTP 404: Checkpoint {checkpoint_id} not found")
        del self.checkpoints[checkpoint_id]

    def execute_command(
        self,
        sandbox_id: str,
        command: str,
        working_dir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        user: str | None = None,
    ) -> CommandResponse:
        self.exec_calls.append(
            {
                "sandbox_id": sandbox_id,
                "command": command,
                "working_dir": working_dir,
                "env": dict(env or {}),
                "timeout": timeout,
                "user": user,
            }
        )
        return CommandResponse(stdout="", stderr="", exit_code=0)


class _ImageClientAdapter:
    def __init__(self, backend: FakePrimeBackend) -> None:
        self._backend = backend

    def list(self, **kwargs: Any) -> ImageListResponse:
        return self._backend.list_images_impl(**kwargs)


def _run(
    backend: FakePrimeBackend,
    state_dir: Path,
    env: dict[str, str],
) -> None:
    full_env = {"PRIME_ORB_STATE_DIR": str(state_dir), **env}
    job = DispatchJob.from_env(full_env)
    cfg = ProviderConfig.from_env(full_env)
    run_dispatch(
        job,
        cfg,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=_ImageClientAdapter(backend),  # type: ignore[arg-type]
    )


def test_full_orb_lifecycle_spawn_pause_resume_and_delete(tmp_path: Path) -> None:
    backend = FakePrimeBackend()

    # 1. Cold spawn resolves default Google Artifact Registry ORB_IMAGE -> Prime Image
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-T-001",
            "ORB_TOKEN": "orb_tok_initial_secret",
            "AMP_URL": "https://ampcode.example",
        },
    )
    assert len(backend.create_requests) == 1
    created_req = backend.create_requests[0]
    assert created_req.docker_image == "prime/xeophon/amp-orb:latest"
    assert created_req.timeout_minutes == -1
    assert created_req.labels == ["amp-orb", "amp-orb:amp-T-001"]

    assert len(backend.exec_calls) == 1
    boot_call = backend.exec_calls[0]
    assert boot_call["user"] == "user"
    assert boot_call["env"] == {
        "ORB_TOKEN": "orb_tok_initial_secret",
        "AMP_URL": "https://ampcode.example",
    }
    # Secret token must never be interpolated into the shell script text!
    assert "orb_tok_initial_secret" not in boot_call["command"]
    assert "amp orb bootstrap" in boot_call["command"]

    # 2. Repeating spawn while RUNNING is idempotent (no duplicate VM created)
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-T-001",
            "ORB_TOKEN": "orb_tok_repeat",
        },
    )
    assert len(backend.create_requests) == 1
    assert backend.exec_calls[-1]["env"] == {"ORB_TOKEN": "orb_tok_repeat"}

    # 3. Pause freezes VM memory & disk and polls until PAUSED
    _run(backend, tmp_path, {"ORB_OPERATION": "pause", "ORB_ID": "amp-T-001"})
    assert backend.pause_calls == ["sbx_1"]
    assert backend.sandboxes["sbx_1"]["status"] == "PAUSED"
    # Repeating pause while already PAUSED is a safe no-op
    _run(backend, tmp_path, {"ORB_OPERATION": "pause", "ORB_ID": "amp-T-001"})
    assert backend.pause_calls == ["sbx_1"]

    # 4. Next spawn resumes the same VM (handling transient 409 conflict & node error retry)
    backend.resume_api_errors = [APIError("HTTP 409: pause still finalizing")]
    backend.next_resume_statuses["sbx_1"] = [("PAUSED", "node busy")]
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-T-001",
            "ORB_TOKEN": "orb_tok_wake",
        },
    )
    assert len(backend.create_requests) == 1  # Still the original sandbox!
    assert len(backend.resume_calls) == 2
    assert backend.exec_calls[-1]["env"] == {"ORB_TOKEN": "orb_tok_wake"}

    # 5. Delete terminates sandbox AND deletes its pause checkpoint so storage billing stops
    _run(backend, tmp_path, {"ORB_OPERATION": "delete", "ORB_ID": "amp-T-001"})
    assert backend.delete_calls == ["sbx_1"]
    assert backend.checkpoints == {}
    # Repeating delete is idempotent
    _run(backend, tmp_path, {"ORB_OPERATION": "delete", "ORB_ID": "amp-T-001"})


def test_project_snapshots_survive_source_thread_deletion_and_boot_arm64(tmp_path: Path) -> None:
    backend = FakePrimeBackend()

    # Thread A spawns on arm64, prepares setup, and pauses with ORB_SNAPSHOT_ID="snap-v1"
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-thread-A",
            "ORB_TOKEN": "tok-a",
            "ORB_IMAGE": DEFAULT_AMP_ORB_IMAGE_ARM64,
        },
    )
    assert backend.create_requests[0].arch == "arm64"

    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "pause",
            "ORB_ID": "amp-thread-A",
            "ORB_SNAPSHOT_ID": "snap-v1",
        },
    )
    store = SnapshotStore(tmp_path)
    shared_ckpt_id = store.get_project_snapshot("snap-v1")
    assert shared_ckpt_id is not None
    # Single-checkpoint pause: pause() created exactly 1 checkpoint reused for both orb & snapshot!
    assert len(backend.checkpoints) == 1

    # Deleting Thread A must KEEP shared project snapshot `shared_ckpt_id` for future threads!
    _run(backend, tmp_path, {"ORB_OPERATION": "delete", "ORB_ID": "amp-thread-A"})
    assert shared_ckpt_id in backend.checkpoints
    assert len(backend.checkpoints) == 1

    # Thread B spawns on arm64 from ORB_SNAPSHOT_ID="snap-v1":
    # Prime rejects `arch`, `docker_image`, and `disk_size_gb` on checkpoint restores!
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-thread-B",
            "ORB_TOKEN": "tok-b",
            "ORB_IMAGE": DEFAULT_AMP_ORB_IMAGE_ARM64,
            "ORB_SNAPSHOT_ID": "snap-v1",
        },
    )
    restored_req = backend.create_requests[-1]
    assert restored_req.checkpoint_id == shared_ckpt_id
    assert restored_req.arch is None
    assert restored_req.docker_image is None

    # If the checkpoint later expires server-side, Thread C prunes the stale mapping
    # and falls back cleanly to the base image instead of failing!
    del backend.checkpoints[shared_ckpt_id]
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-thread-C",
            "ORB_TOKEN": "tok-c",
            "ORB_IMAGE": DEFAULT_AMP_ORB_IMAGE_ARM64,
            "ORB_SNAPSHOT_ID": "snap-v1",
        },
    )
    assert store.get_project_snapshot("snap-v1") is None
    fallback_req = backend.create_requests[-1]
    assert fallback_req.checkpoint_id is None
    assert fallback_req.docker_image == "prime/xeophon/amp-orb-arm64:latest"
    assert fallback_req.arch == "arm64"


def test_lost_paused_host_node_recovers_disk_from_pause_checkpoint(tmp_path: Path) -> None:
    backend = FakePrimeBackend()
    _run(
        backend,
        tmp_path,
        {"ORB_OPERATION": "spawn", "ORB_ID": "amp-node-loss", "ORB_TOKEN": "tok-1"},
    )
    _run(backend, tmp_path, {"ORB_OPERATION": "pause", "ORB_ID": "amp-node-loss"})

    # Simulate physical node failure while paused:
    backend.sandboxes["sbx_1"]["status"] = "TERMINATED"

    # Next spawn automatically restores a replacement VM from sbx_1's durable pause checkpoint!
    _run(
        backend,
        tmp_path,
        {"ORB_OPERATION": "spawn", "ORB_ID": "amp-node-loss", "ORB_TOKEN": "tok-2"},
    )
    assert len(backend.create_requests) == 2
    assert backend.create_requests[1].checkpoint_id == "ckpt_1"
    assert backend.exec_calls[-1]["sandbox_id"] == "sbx_2"


def test_gpu_orb_type_checkpoints_and_terminates_on_pause_then_restores(tmp_path: Path) -> None:
    backend = FakePrimeBackend()
    gpu_types = (
        '{"gpu":{"cpu_cores":8,"memory_gb":32,"disk_size_gb":60,'
        '"gpu_count":1,"gpu_type":"RTX_PRO_6000"}}'
    )
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-gpu-1",
            "ORB_TOKEN": "tok-gpu-1",
            "ORB_TYPE": "gpu",
            "PRIME_ORB_TYPES": gpu_types,
        },
    )
    assert backend.create_requests[0].gpu_count == 1
    assert backend.create_requests[0].gpu_type == "RTX_PRO_6000"

    # Pause must NOT call VM pause (unsupported on GPU); it checkpoints rootfs & deletes GPU VM
    _run(
        backend,
        tmp_path,
        {"ORB_OPERATION": "pause", "ORB_ID": "amp-gpu-1", "PRIME_ORB_TYPES": gpu_types},
    )
    assert backend.pause_calls == []
    assert backend.sandboxes["sbx_1"]["status"] == "TERMINATED"

    # Next spawn restores disk into a new GPU sandbox from that checkpoint
    _run(
        backend,
        tmp_path,
        {
            "ORB_OPERATION": "spawn",
            "ORB_ID": "amp-gpu-1",
            "ORB_TOKEN": "tok-gpu-2",
            "ORB_TYPE": "gpu",
            "PRIME_ORB_TYPES": gpu_types,
        },
    )
    assert len(backend.create_requests) == 2
    assert backend.create_requests[1].checkpoint_id == "ckpt_1"
    assert backend.create_requests[1].gpu_count == 1
    assert backend.create_requests[1].gpu_type == "RTX_PRO_6000"


def test_async_pause_failure_on_node_raises_immediately(tmp_path: Path) -> None:
    backend = FakePrimeBackend()
    _run(
        backend,
        tmp_path,
        {"ORB_OPERATION": "spawn", "ORB_ID": "amp-pause-err", "ORB_TOKEN": "tok-1"},
    )
    backend.next_pause_statuses["sbx_1"] = [("RUNNING", "snapshot disk full")]
    with pytest.raises(RuntimeError, match="snapshot disk full"):
        _run(
            backend,
            tmp_path,
            {
                "ORB_OPERATION": "pause",
                "ORB_ID": "amp-pause-err",
                "PRIME_ORB_DISPATCH_TIMEOUT_SECONDS": "5",
            },
        )


def test_dispatch_executable_runs_via_system_python(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [str(repo_root / "dispatch.py")],
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "ORB_OPERATION": "invalid-op",
            "ORB_ID": "amp-test",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 1
    assert "Invalid or missing ORB_OPERATION" in result.stderr

    # Missing PRIME_API_KEY fails fast with clear actionable guidance
    missing_key = subprocess.run(
        [sys.executable, str(repo_root / "dispatch.py")],
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "ORB_OPERATION": "delete",
            "ORB_ID": "amp-test",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert missing_key.returncode == 1
    assert "PRIME_API_KEY" in missing_key.stderr


def test_bootstrap_script_detaches_cleanly_under_real_bash(tmp_path: Path) -> None:
    """Execute BOOTSTRAP_SCRIPT under real /bin/bash to verify fd closure, PATH, and env."""
    import time

    from prime_orbs.dispatcher import BOOTSTRAP_SCRIPT

    fake_home = tmp_path / "home"
    amp_bin_dir = fake_home / ".amp" / "bin"
    amp_bin_dir.mkdir(parents=True)
    marker_file = tmp_path / "bootstrap_observed.txt"

    fake_amp = amp_bin_dir / "amp"
    fake_amp.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s|%s|%s\\n" "$*" "${{ORB_TOKEN:-}}" "${{AMP_URL:-}}" > "{marker_file}"\n'
        "sleep 0.3\n",
        encoding="utf-8",
    )
    fake_amp.chmod(0o755)

    start = time.monotonic()
    proc = subprocess.run(
        ["/bin/bash", "-c", BOOTSTRAP_SCRIPT],
        cwd="/",
        env={
            "HOME": str(fake_home),
            "USER": "user",
            "PATH": "/usr/bin:/bin",
            "ORB_TOKEN": "tok_live_detached",
            "AMP_URL": "https://ampcode.example",
        },
        capture_output=True,
        text=True,
        timeout=2,
    )
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr
    # Parent shell must return immediately without waiting for the 0.3s background sleep!
    assert elapsed < 0.25

    deadline = time.monotonic() + 2.0
    while not marker_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert marker_file.read_text(encoding="utf-8").strip() == (
        "orb bootstrap|tok_live_detached|https://ampcode.example"
    )


def test_warm_memory_template_forks_with_restore_memory_and_survives_thread_delete(
    tmp_path: Path,
) -> None:
    backend = FakePrimeBackend()
    cfg = ProviderConfig.from_env({"PRIME_ORB_STATE_DIR": str(tmp_path)})
    img_adapter = _ImageClientAdapter(backend)

    # 1. Build a warm memory template once: creates clean VM, prewarms amp, pauses, deletes VM
    tmpl_key, warm_ckpt_id = build_warm_template(
        cfg,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
    )
    assert tmpl_key.startswith("prime/xeophon/amp-orb:latest|")
    assert backend.sandboxes["sbx_1"]["status"] == "TERMINATED"  # Template VM cleaned up!
    assert warm_ckpt_id in backend.checkpoints
    assert "amp update --porcelain" in backend.exec_calls[0]["command"]

    # 2. Rebuilding replaces the old template checkpoint and deletes the superseded checkpoint
    _, warm_ckpt_v2 = build_warm_template(
        cfg,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
    )
    assert warm_ckpt_v2 != warm_ckpt_id
    assert warm_ckpt_id not in backend.checkpoints
    assert warm_ckpt_v2 in backend.checkpoints

    # 3. Cold thread spawn forks from warm_ckpt_v2 with restore_memory=True (no shape conflict!)
    _run(
        backend,
        tmp_path,
        {"ORB_OPERATION": "spawn", "ORB_ID": "amp-fast-1", "ORB_TOKEN": "tok-fast-1"},
    )
    fork_req = backend.create_requests[-1]
    assert fork_req.checkpoint_id == warm_ckpt_v2
    assert fork_req.restore_memory is True
    assert "cpu_cores" not in fork_req.model_fields_set
    assert "memory_gb" not in fork_req.model_fields_set

    # 4. Pausing and deleting that thread keeps shared warm_ckpt_v2 intact for future threads!
    _run(backend, tmp_path, {"ORB_OPERATION": "pause", "ORB_ID": "amp-fast-1"})
    _run(backend, tmp_path, {"ORB_OPERATION": "delete", "ORB_ID": "amp-fast-1"})
    assert warm_ckpt_v2 in backend.checkpoints

    # 5. If warm checkpoint is later missing on Prime, spawn prunes it and cold-boots cleanly
    del backend.checkpoints[warm_ckpt_v2]
    _run(
        backend,
        tmp_path,
        {"ORB_OPERATION": "spawn", "ORB_ID": "amp-fast-2", "ORB_TOKEN": "tok-fast-2"},
    )
    cold_fallback = backend.create_requests[-1]
    assert cold_fallback.checkpoint_id is None
    assert cold_fallback.docker_image == "prime/xeophon/amp-orb:latest"


def test_project_snapshot_memory_promotion_and_custom_orb_type_shape_fallback(
    tmp_path: Path,
) -> None:
    from prime_orbs.dispatcher import expire_stale_project_snapshots
    from prime_orbs.standby import SnapshotPromoter

    backend = FakePrimeBackend()
    store = SnapshotStore(tmp_path)
    cfg = ProviderConfig.from_env({"PRIME_ORB_STATE_DIR": str(tmp_path)})
    promoter = SnapshotPromoter(cfg, backend, store)  # type: ignore[arg-type]

    try:
        # 1. Thread A spawns and pauses with ORB_SNAPSHOT_ID="snap-promo"
        _run(
            backend,
            tmp_path,
            {"ORB_OPERATION": "spawn", "ORB_ID": "amp-src", "ORB_TOKEN": "tok-src"},
        )
        _run(
            backend,
            tmp_path,
            {"ORB_OPERATION": "pause", "ORB_ID": "amp-src", "ORB_SNAPSHOT_ID": "snap-promo"},
        )
        disk_ckpt, initial_mode = store.get_project_snapshot_entry("snap-promo")
        assert disk_ckpt is not None
        assert initial_mode == "disk"

        # 2. Background promoter converts disk_ckpt -> clean restore_memory=True checkpoint
        promoted_ckpt = promoter.promote_now("snap-promo", disk_ckpt, cfg.default_spec)
        assert promoted_ckpt is not None and promoted_ckpt != disk_ckpt
        saved_ckpt, saved_mode = store.get_project_snapshot_entry("snap-promo")
        assert saved_ckpt == promoted_ckpt
        assert saved_mode == "memory:4:8"

        # 3. Default-sized Thread B forks with restore_memory=True!
        _run(
            backend,
            tmp_path,
            {
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-dst-default",
                "ORB_TOKEN": "tok-b",
                "ORB_SNAPSHOT_ID": "snap-promo",
            },
        )
        default_req = backend.create_requests[-1]
        assert default_req.checkpoint_id == promoted_ckpt
        assert default_req.restore_memory is True

        # 4. Larger ORB_TYPE="large" (16 vCPU, 32 GiB) safely cold-boots the same checkpoint
        #    so it gets 16 vCPUs / 32 GiB instead of failing Prime's memory-shape check!
        _run(
            backend,
            tmp_path,
            {
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-dst-large",
                "ORB_TOKEN": "tok-c",
                "ORB_TYPE": "large",
                "ORB_SNAPSHOT_ID": "snap-promo",
            },
        )
        large_req = backend.create_requests[-1]
        assert large_req.checkpoint_id == promoted_ckpt
        assert not large_req.restore_memory
        assert large_req.cpu_cores == 16.0
        assert large_req.memory_gb == 32.0

        # 5. TTL expiry cleans up aged-out project snapshots
        backend.checkpoints[promoted_ckpt]["created_at"] = "2026-11-01T00:00:00+00:00"
        expired = expire_stale_project_snapshots(
            backend,  # type: ignore[arg-type]
            store,
            ttl_seconds=96 * 3600,
        )
        assert expired == 1
        assert store.get_project_snapshot("snap-promo") is None
        assert promoted_ckpt not in backend.checkpoints
    finally:
        promoter.close()



