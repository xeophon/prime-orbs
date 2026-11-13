"""End-to-end tests for DispatchDaemon, zero-GET authoritative index, and StandbyPool."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from test_dispatcher import FakePrimeBackend, _ImageClientAdapter

from prime_orbs.config import ProviderConfig
from prime_orbs.daemon import DAEMON_SOCKET_ENV, DispatchDaemon, daemon_socket_path
from prime_orbs.dispatcher import sync_known_orbs
from prime_orbs.standby import StandbyPool
from prime_orbs.state import SnapshotStore


def test_dispatch_py_uses_warm_unix_socket_daemon_and_zero_list_fast_path(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    state_dir = tmp_path / "state"
    sock_path = daemon_socket_path(state_dir)
    backend = FakePrimeBackend()

    list_call_count = 0
    orig_list = backend.list

    def counting_list(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal list_call_count
        list_call_count += 1
        return orig_list(**kwargs)

    backend.list = counting_list  # type: ignore[method-assign]

    base_env = {
        "PRIME_ORB_STATE_DIR": str(state_dir),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
        "PRIME_ORB_STANDBY_COUNT": "0",
    }
    with DispatchDaemon(
        sock_path,
        base_env=base_env,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=_ImageClientAdapter(backend),  # type: ignore[arg-type]
        enable_standby_worker=False,
    ):
        assert sock_path.exists()
        assert (sock_path.stat().st_mode & 0o777) == 0o600
        list_calls_after_startup_sync = list_call_count

        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "PRIME_ORB_STATE_DIR": str(state_dir),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-daemon-1",
                "ORB_TOKEN": "tok_daemon_secret",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert proc.returncode == 0, proc.stderr
        # Zero GET /sandbox?labels=... calls during brand-new thread spawn!
        assert list_call_count == list_calls_after_startup_sync
        assert "[prime-orbs] Creating amp-daemon-1" in proc.stderr
        assert "[prime-orbs] Started `amp orb bootstrap`" in proc.stderr
        assert len(backend.create_requests) == 1
        assert backend.exec_calls[0]["env"] == {"ORB_TOKEN": "tok_daemon_secret"}
        assert elapsed_ms < 150.0, (
            f"Expected fast stdlib socket round-trip, got {elapsed_ms:.1f}ms"
        )

        # Validation error propagates exit code 1 and clear stderr through the socket
        bad = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-daemon-bad",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert bad.returncode == 1
        assert "ORB_TOKEN" in bad.stderr

    assert not sock_path.exists()


def test_standby_pool_hot_claim_restart_preservation_idle_cooling_and_drain(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    state_dir = tmp_path / "standby_state"
    sock_path = daemon_socket_path(state_dir)
    backend = FakePrimeBackend()
    img_adapter = _ImageClientAdapter(backend)
    store = SnapshotStore(state_dir)

    base_env = {
        "PRIME_ORB_STATE_DIR": str(state_dir),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
        "PRIME_ORB_STANDBY_COUNT": "1",
        "PRIME_ORB_STANDBY_HOT_SECONDS": "900",
    }

    with DispatchDaemon(
        sock_path,
        base_env=base_env,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
        enable_standby_worker=False,
    ) as daemon:
        # 1. Pre-populate 1 RUNNING standby VM in the pool
        standby_sbx = daemon.standby_pool.replenish_once()
        assert standby_sbx is not None
        assert standby_sbx.status == "RUNNING"
        creates_before_spawn = len(backend.create_requests)

        # 2. Spawning a new thread claims the RUNNING standby VM with ZERO create() calls!
        proc = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "PRIME_ORB_STATE_DIR": str(state_dir),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-claimed-1",
                "ORB_TOKEN": "tok_claimed",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert proc.returncode == 0, proc.stderr
        assert "Claimed hot RUNNING standby sandbox" in proc.stderr
        assert len(backend.create_requests) == creates_before_spawn
        assert store.get_orb_sandbox("amp-claimed-1") == standby_sbx.id

        # 3. Replenish a second standby VM and cool it to PAUSED when idle
        standby_two = daemon.standby_pool.replenish_once()
        assert standby_two is not None and standby_two.id != standby_sbx.id
        daemon.standby_pool._hot_seconds = 0.0
        assert daemon.standby_pool.cool_idle_now() == 1
        assert backend.sandboxes[standby_two.id]["status"] == "PAUSED"

    # 4. Simulate daemon restart: claimed standby VM MUST remain owned by `amp-claimed-1`
    #    (never reclaimed or drained!), while `standby_two` is re-adopted as PAUSED standby!
    sync_known_orbs(backend, store)  # type: ignore[arg-type]
    assert store.get_orb_sandbox("amp-claimed-1") == standby_sbx.id

    with DispatchDaemon(
        sock_path,
        base_env=base_env,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
        enable_standby_worker=False,
    ) as daemon2:
        creates_before_second = len(backend.create_requests)
        proc2 = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "PRIME_ORB_STATE_DIR": str(state_dir),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-claimed-2",
                "ORB_TOKEN": "tok_claimed_2",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert proc2.returncode == 0, proc2.stderr
        assert f"Waking pre-placed standby sandbox {standby_two.id}" in proc2.stderr
        assert len(backend.create_requests) == creates_before_second
        assert store.get_orb_sandbox("amp-claimed-2") == standby_two.id
        del daemon2

    # 5. Setting PRIME_ORB_STANDBY_COUNT=0 drains unclaimed standby VMs while keeping active orbs
    extra_standby_cfg = ProviderConfig.from_env(base_env)
    pool_temp = StandbyPool(extra_standby_cfg, backend, img_adapter, store)  # type: ignore[arg-type]
    pool_temp.start(run_background_worker=False)
    unclaimed = pool_temp.replenish_once()
    assert unclaimed is not None
    pool_temp.close()

    drain_cfg = ProviderConfig.from_env({**base_env, "PRIME_ORB_STANDBY_COUNT": "0"})
    drain_pool = StandbyPool(drain_cfg, backend, img_adapter, store)  # type: ignore[arg-type]
    drain_pool.start(run_background_worker=False)
    drain_pool.close()

    assert backend.sandboxes[unclaimed.id]["status"] == "TERMINATED"
    assert backend.sandboxes[standby_sbx.id]["status"] == "RUNNING"
    assert backend.sandboxes[standby_two.id]["status"] == "RUNNING"


def test_long_state_dir_falls_back_to_short_runtime_socket_path() -> None:
    deep_dir = Path("/tmp/" + ("very_long_segment_" * 8))
    sock = daemon_socket_path(deep_dir)
    assert len(str(sock).encode("utf-8")) <= 96
