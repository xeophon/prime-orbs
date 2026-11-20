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


def test_single_rpc_hot_standby_claim_and_dead_standby_recovery(tmp_path: Path) -> None:
    """Prove a hot standby claim makes 0 GET/LIST/CREATE calls (1 single ConnectRPC call),
    and recovers transparently if an idle standby VM died silently on Prime."""
    repo_root = Path(__file__).resolve().parents[1]
    state_dir = tmp_path / "single_rpc_state"
    sock_path = daemon_socket_path(state_dir)
    backend = FakePrimeBackend()
    img_adapter = _ImageClientAdapter(backend)

    get_calls: list[str] = []
    list_calls: int = 0
    orig_get = backend.get
    orig_list = backend.list
    orig_exec = backend.execute_command

    def tracked_get(sandbox_id: str):  # type: ignore[no-untyped-def]
        get_calls.append(sandbox_id)
        return orig_get(sandbox_id)

    def tracked_list(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal list_calls
        list_calls += 1
        return orig_list(**kwargs)

    def realistic_exec(sandbox_id: str, command: str, **kwargs):  # type: ignore[no-untyped-def]
        from prime_sandboxes import APIError

        if backend.sandboxes[sandbox_id]["status"] != "RUNNING":
            raise APIError(f"Sandbox {sandbox_id} not running")
        return orig_exec(sandbox_id, command, **kwargs)

    backend.get = tracked_get  # type: ignore[method-assign]
    backend.list = tracked_list  # type: ignore[method-assign]
    backend.execute_command = realistic_exec  # type: ignore[method-assign]

    base_env = {
        "PRIME_ORB_STATE_DIR": str(state_dir),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
        "PRIME_ORB_STANDBY_COUNT": "1",
    }
    with DispatchDaemon(
        sock_path,
        base_env=base_env,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
        enable_standby_worker=False,
    ) as daemon:
        hot_sbx = daemon.standby_pool.replenish_once()
        assert hot_sbx is not None

        # Reset counters right before spawn:
        get_calls.clear()
        list_calls = 0
        creates_before = len(backend.create_requests)
        execs_before = len(backend.exec_calls)

        proc = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "PRIME_ORB_STATE_DIR": str(state_dir),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-1rpc",
                "ORB_TOKEN": "tok_1rpc",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert proc.returncode == 0, proc.stderr
        assert get_calls == [], f"Expected 0 GET calls on hot claim, got {get_calls}"
        assert list_calls == 0
        assert len(backend.create_requests) == creates_before
        assert len(backend.exec_calls) == execs_before + 1

        # Now replenish another standby, silently kill it on the backend, and verify recovery:
        dead_standby = daemon.standby_pool.replenish_once()
        assert dead_standby is not None
        backend.sandboxes[dead_standby.id]["status"] = "TERMINATED"

        proc_recover = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "PRIME_ORB_STATE_DIR": str(state_dir),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-recover-dead-standby",
                "ORB_TOKEN": "tok_rec",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert proc_recover.returncode == 0, proc_recover.stderr
        assert "Optimistic standby sandbox" in proc_recover.stderr


def test_async_snapshot_pause_project_standby_claim_and_speculative_resume(
    tmp_path: Path,
) -> None:
    """Verify #1 (project-snapshot standby + LRU eviction), #2 (0-wait pause on ORB_SNAPSHOT_ID),
    and #3b (speculative POST /resume without prior GET)."""
    from prime_orbs.config import DispatchJob
    from prime_orbs.dispatcher import run_dispatch

    state_dir = tmp_path / "proj_state"
    backend = FakePrimeBackend()
    img_adapter = _ImageClientAdapter(backend)
    store = SnapshotStore(state_dir)

    wait_ckpt_calls: list[str] = []
    orig_wait_ckpt = backend.wait_for_checkpoint

    def tracked_wait_ckpt(checkpoint_id: str, timeout_seconds: float = 300):  # type: ignore[no-untyped-def]
        wait_ckpt_calls.append(checkpoint_id)
        return orig_wait_ckpt(checkpoint_id, timeout_seconds=timeout_seconds)

    backend.wait_for_checkpoint = tracked_wait_ckpt  # type: ignore[method-assign]

    env = {
        "PRIME_ORB_STATE_DIR": str(state_dir),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
        "PRIME_ORB_STANDBY_COUNT": "1",
        "PRIME_ORB_MAX_PROJECT_STANDBYS": "1",
    }
    cfg = ProviderConfig.from_env(env)
    pool = StandbyPool(cfg, backend, img_adapter, store)  # type: ignore[arg-type]
    pool.start(run_background_worker=False)

    promoted_queue: list[tuple[str, str]] = []
    try:
        # 1. Spawn Thread A and pause with ORB_SNAPSHOT_ID="snap-A" via async callback (#2):
        run_dispatch(
            DispatchJob.from_env(
                {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-A", "ORB_TOKEN": "tok-a"}
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
        )
        wait_ckpt_calls.clear()
        run_dispatch(
            DispatchJob.from_env(
                {**env, "ORB_OPERATION": "pause", "ORB_ID": "amp-A", "ORB_SNAPSHOT_ID": "snap-A"}
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
            on_snapshot_saved=lambda sid, cid, _sp: promoted_queue.append((sid, cid)),
        )
        # Pause returned immediately without blocking on wait_for_checkpoint!
        assert wait_ckpt_calls == []
        assert store.get_project_snapshot_entry("snap-A")[1] == "uploading"
        assert len(promoted_queue) == 1

        # 2. Speculative resume (#3b): waking Thread A calls resume() BEFORE any get()!
        call_order: list[str] = []
        orig_get = backend.get
        orig_resume = backend.resume

        def order_get(sbx_id: str):  # type: ignore[no-untyped-def]
            call_order.append(f"get:{sbx_id}")
            return orig_get(sbx_id)

        def order_resume(sbx_id: str):  # type: ignore[no-untyped-def]
            call_order.append(f"resume:{sbx_id}")
            return orig_resume(sbx_id)

        backend.get = order_get  # type: ignore[method-assign]
        backend.resume = order_resume  # type: ignore[method-assign]

        run_dispatch(
            DispatchJob.from_env(
                {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-A", "ORB_TOKEN": "tok-a2"}
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
        )
        assert call_order[0] == "resume:sbx_1"

        # 3. Promote snap-A and replenish standby pool -> warms base + snap-A project standby (#1)!
        from prime_orbs.standby import SnapshotPromoter

        promoter = SnapshotPromoter(
            cfg,
            backend,  # type: ignore[arg-type]
            store,
            on_promoted=pool.register_project_snapshot,
        )
        try:
            promoter.promote_now("snap-A", promoted_queue[0][1], cfg.default_spec)
        finally:
            promoter.close()

        base_sbx = pool.replenish_once()
        proj_sbx = pool.replenish_once()
        assert base_sbx is not None and proj_sbx is not None
        assert pool.replenish_once() is None  # Quotas full!

        # 4. New project thread with ORB_SNAPSHOT_ID="snap-A" claims proj_sbx in 0 creates & 0 GETs!
        call_order.clear()
        creates_before = len(backend.create_requests)
        run_dispatch(
            DispatchJob.from_env(
                {
                    **env,
                    "ORB_OPERATION": "spawn",
                    "ORB_ID": "amp-proj-fast",
                    "ORB_TOKEN": "tok-pf",
                    "ORB_SNAPSHOT_ID": "snap-A",
                }
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
            standby_claim_fn=pool.claim,
            project_standby_claim_fn=pool.claim_project,
            project_standby_touch_fn=pool.register_project_snapshot,
        )
        assert len(backend.create_requests) == creates_before
        assert call_order == []  # 0 get() and 0 resume() calls!
        assert store.get_orb_sandbox("amp-proj-fast") == proj_sbx.id

        # 5. LRU eviction when a second project snapshot registers under MAX_PROJECT_STANDBYS=1:
        proj_sbx_a2 = pool.replenish_once()
        assert proj_sbx_a2 is not None
        store.set_project_snapshot("snap-B", promoted_queue[0][1], mode="disk")
        pool.register_project_snapshot("snap-B", cfg.default_spec)
        assert backend.sandboxes[proj_sbx_a2.id]["status"] == "TERMINATED"
    finally:
        pool.close()


def test_long_state_dir_falls_back_to_short_runtime_socket_path() -> None:
    deep_dir = Path("/tmp/" + ("very_long_segment_" * 8))
    sock = daemon_socket_path(deep_dir)
    assert len(str(sock).encode("utf-8")) <= 96


def test_tuned_transport_pools_proactive_token_refresh_and_setup_timings(
    tmp_path: Path,
) -> None:
    from prime_sandboxes import APIClient, SandboxClient

    from prime_orbs.cli import main as cli_main
    from prime_orbs.dispatcher import TEMPLATE_PREWARM_SCRIPT
    from prime_orbs.transport import (
        API_KEEPALIVE_EXPIRY_SECONDS,
        tune_api_client,
        tune_sandbox_client,
    )

    # 1. Verify httpx keepalive_expiry is raised from 5.0s -> 300.0s & pyqwest HTTP/2 pool leases
    api = APIClient(api_key="test-key", require_auth=False)
    try:
        tune_api_client(api)
        assert api.client._transport._pool._keepalive_expiry == API_KEEPALIVE_EXPIRY_SECONDS  # type: ignore[attr-defined]
        sbx_client = tune_sandbox_client(SandboxClient(api))
        with sbx_client._command_pool.lease() as leased:
            assert leased is not None
    finally:
        api.client.close()

    # 2. Verify TEMPLATE_PREWARM_SCRIPT creates ~/.config/amp/settings.json only when missing
    fake_home = tmp_path / "guest_home"
    fake_home.mkdir()
    subprocess.run(
        ["/bin/bash", "-c", TEMPLATE_PREWARM_SCRIPT],
        env={"HOME": str(fake_home), "PATH": "/usr/bin:/bin", "PRIME_ORB_UPDATE_AMP": "0"},
        check=True,
    )
    settings_path = fake_home / ".config" / "amp" / "settings.json"
    assert settings_path.read_text(encoding="utf-8") == "{}\n"
    settings_path.write_text('{"custom": true}\n', encoding="utf-8")
    subprocess.run(
        ["/bin/bash", "-c", TEMPLATE_PREWARM_SCRIPT],
        env={"HOME": str(fake_home), "PATH": "/usr/bin:/bin", "PRIME_ORB_UPDATE_AMP": "0"},
        check=True,
    )
    assert settings_path.read_text(encoding="utf-8") == '{"custom": true}\n'

    # 3. Verify proactive standby token refresh & setup-log timing line
    state_dir = tmp_path / "timing_state"
    sock_path = daemon_socket_path(state_dir)
    backend = FakePrimeBackend()
    refreshed_ids: list[str] = []
    backend.get_or_refresh = lambda sid: refreshed_ids.append(sid) or {}  # type: ignore[attr-defined]

    with DispatchDaemon(
        sock_path,
        base_env={
            "PRIME_ORB_STATE_DIR": str(state_dir),
            "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
            "PRIME_ORB_STANDBY_COUNT": "1",
        },
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=_ImageClientAdapter(backend),  # type: ignore[arg-type]
        enable_standby_worker=False,
    ) as daemon:
        hot = daemon.standby_pool.replenish_once()
        assert hot is not None
        assert daemon.standby_pool.refresh_standby_tokens_now() == 1
        assert refreshed_ids == [hot.id]

        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, str(repo_root / "dispatch.py")],
            env={
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                DAEMON_SOCKET_ENV: str(sock_path),
                "PRIME_ORB_STATE_DIR": str(state_dir),
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-timed",
                "ORB_TOKEN": "tok_timed",
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert proc.returncode == 0, proc.stderr
        assert "via hot-standby (acquire=" in proc.stderr
        assert "bootstrap=" in proc.stderr

    # 4. Verify `prime-orbs install-service` writes 0600 env file, service unit, and daily timer
    env_file = tmp_path / "creds" / "prime-orbs.env"
    unit_dir = tmp_path / "systemd_user"
    rc = cli_main(
        [
            "install-service",
            "--env-file",
            str(env_file),
            "--unit-dir",
            str(unit_dir),
            "--standby-count",
            "1",
            "--team-id",
            "team_alpha",
            "--no-enable",
        ]
    )
    assert rc == 0
    assert (env_file.stat().st_mode & 0o777) == 0o600
    env_text = env_file.read_text(encoding="utf-8")
    assert "PRIME_TEAM_ID=team_alpha" in env_text
    svc_text = (unit_dir / "prime-orbs.service").read_text(encoding="utf-8")
    refresh_svc = (unit_dir / "prime-orbs-refresh.service").read_text(encoding="utf-8")
    refresh_timer = (unit_dir / "prime-orbs-refresh.timer").read_text(encoding="utf-8")
    assert "run --warm-template --team-id team_alpha --standby-count 1" in svc_text
    assert "warm-template --team-id team_alpha" in refresh_svc
    assert "OnCalendar=" in refresh_timer


def test_optimistic_running_rebootstrap_and_promoted_standby_adoption(tmp_path: Path) -> None:
    """Verify:
    - Re-spawning an already-RUNNING orb skips GET /sandbox/{id} and recovers if it died
    - SnapshotPromoter adopts the promoted VM directly into StandbyPool (no second VM boot)
    - Memory-forked standby VMs run a lightweight `true` readiness probe instead of full prewarm
    - build_warm_template(..., keep_standby_label=...) seeds a standby without deleting the VM
    """
    from prime_sandboxes import APIError

    from prime_orbs.config import DispatchJob
    from prime_orbs.dispatcher import (
        build_warm_template,
        run_dispatch,
        warm_template_key,
    )
    from prime_orbs.standby import SnapshotPromoter, standby_key_label

    state_dir = tmp_path / "opt_state"
    backend = FakePrimeBackend()
    img_adapter = _ImageClientAdapter(backend)
    store = SnapshotStore(state_dir)

    env = {
        "PRIME_ORB_STATE_DIR": str(state_dir),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
        "PRIME_ORB_STANDBY_COUNT": "1",
        "PRIME_ORB_MAX_PROJECT_STANDBYS": "2",
    }
    cfg = ProviderConfig.from_env(env)

    # 1. Seed warm template + initial standby in a single VM boot
    tmpl_key = warm_template_key("prime/xeophon/amp-orb:latest", cfg.default_spec)
    _, warm_ckpt = build_warm_template(
        cfg,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
        update_amp_cli=False,
        keep_standby_label=standby_key_label(tmpl_key),
    )
    seeded_ids = store.get_standby_ids(tmpl_key)
    assert len(seeded_ids) == 1
    assert backend.sandboxes[seeded_ids[0]]["status"] == "PAUSED"
    assert warm_ckpt in backend.checkpoints

    # 2. Memory-forked standby skips redundant full TEMPLATE_PREWARM_SCRIPT and runs `true`
    pool = StandbyPool(cfg, backend, img_adapter, store)  # type: ignore[arg-type]
    pool.start(run_background_worker=False)
    try:
        # Drain seeded paused standby by claiming it, then replenish from warm_ckpt
        claimed = pool.claim(
            "prime/xeophon/amp-orb:latest",
            cfg.default_spec,
            time.monotonic() + 30,
            lambda _: None,
        )
        assert claimed is not None and claimed[0] == seeded_ids[0]
        execs_before = len(backend.exec_calls)
        forked_standby = pool.replenish_once()
        assert forked_standby is not None
        assert backend.create_requests[-1].restore_memory is True
        assert backend.exec_calls[execs_before]["command"] == "true"

        # 3. Optimistic RUNNING re-spawn makes 0 GET calls, and falls back cleanly if dead
        get_calls: list[str] = []
        orig_get = backend.get
        orig_exec = backend.execute_command

        def tracked_get(sbx_id: str):  # type: ignore[no-untyped-def]
            get_calls.append(sbx_id)
            return orig_get(sbx_id)

        def realistic_exec(sbx_id: str, command: str, **kwargs):  # type: ignore[no-untyped-def]
            if backend.sandboxes[sbx_id]["status"] != "RUNNING":
                raise APIError(f"Sandbox {sbx_id} not running")
            return orig_exec(sbx_id, command, **kwargs)

        backend.get = tracked_get  # type: ignore[method-assign]
        backend.execute_command = realistic_exec  # type: ignore[method-assign]

        run_dispatch(
            DispatchJob.from_env(
                {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-opt-run", "ORB_TOKEN": "tok-1"}
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
            standby_claim_fn=pool.claim,
        )
        orb_sbx_id = store.get_orb_sandbox("amp-opt-run")
        assert orb_sbx_id == forked_standby.id

        get_calls.clear()
        run_dispatch(
            DispatchJob.from_env(
                {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-opt-run", "ORB_TOKEN": "tok-2"}
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
        )
        assert get_calls == [], f"Expected 0 GET calls on optimistic RUNNING spawn, got {get_calls}"

        # If that RUNNING sandbox silently died, optimistic spawn detects it and creates a new one
        backend.sandboxes[orb_sbx_id]["status"] = "TERMINATED"
        creates_before_fallback = len(backend.create_requests)
        run_dispatch(
            DispatchJob.from_env(
                {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-opt-run", "ORB_TOKEN": "tok-3"}
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
        )
        assert len(backend.create_requests) == creates_before_fallback + 1
        assert store.get_orb_sandbox("amp-opt-run") != orb_sbx_id

        # 4. SnapshotPromoter adopts the promoted VM directly into the project standby pool
        #    (resumed hot to RUNNING) so a new project thread claims it with 0 extra VM boots!
        run_dispatch(
            DispatchJob.from_env(
                {
                    **env,
                    "ORB_OPERATION": "pause",
                    "ORB_ID": "amp-opt-run",
                    "ORB_SNAPSHOT_ID": "snap-adopt",
                }
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
        )
        disk_ckpt, _ = store.get_project_snapshot_entry("snap-adopt")
        assert disk_ckpt is not None

        promoter = SnapshotPromoter(
            cfg,
            backend,  # type: ignore[arg-type]
            store,
            on_promoted=pool.register_project_snapshot,
            on_standby_ready=pool.adopt_project_standby,
        )
        try:
            creates_before_promo = len(backend.create_requests)
            promoted_ckpt = promoter.promote_now("snap-adopt", disk_ckpt, cfg.default_spec)
            assert promoted_ckpt is not None
            # Only 1 VM was created during promotion, and it was kept & resumed as RUNNING standby!
            assert len(backend.create_requests) == creates_before_promo + 1
        finally:
            promoter.close()

        creates_before_proj_spawn = len(backend.create_requests)
        get_calls.clear()
        run_dispatch(
            DispatchJob.from_env(
                {
                    **env,
                    "ORB_OPERATION": "spawn",
                    "ORB_ID": "amp-proj-adopted",
                    "ORB_TOKEN": "tok-adopted",
                    "ORB_SNAPSHOT_ID": "snap-adopt",
                }
            ),
            cfg,
            sandbox_client=backend,  # type: ignore[arg-type]
            image_client=img_adapter,  # type: ignore[arg-type]
            authoritative_index=True,
            project_standby_claim_fn=pool.claim_project,
            project_standby_touch_fn=pool.register_project_snapshot,
        )
        assert len(backend.create_requests) == creates_before_proj_spawn
        assert get_calls == []
    finally:
        pool.close()


def test_transient_bootstrap_miss_keeps_cached_gateway_auth_and_prewarms_transport(
    tmp_path: Path,
) -> None:
    """Verify transient gateway-not-registered errors don't needlessly invalidate cached auth."""
    from prime_sandboxes import APIError

    from prime_orbs.dispatcher import _execute_when_gateway_ready
    from prime_orbs.transport import prewarm_gateway_transport

    backend = FakePrimeBackend()
    cfg = ProviderConfig.from_env({"PRIME_ORB_STATE_DIR": str(tmp_path)})
    sbx = backend.create(
        __import__("prime_sandboxes").CreateSandboxRequest(
            name="amp-gw-test",
            docker_image="prime/xeophon/amp-orb:latest",
            cpu_cores=4,
            memory_gb=8,
            disk_size_gb=20,
        )
    )

    attempts = 0
    orig_exec = backend.execute_command

    def flaky_exec(sandbox_id: str, command: str, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise APIError("Connect RPC failed (unavailable): gateway still registering")
        return orig_exec(sandbox_id, command, **kwargs)

    backend.execute_command = flaky_exec  # type: ignore[method-assign]
    _execute_when_gateway_ready(
        backend,  # type: ignore[arg-type]
        sbx.id,
        "true",
        envs=None,
        user=cfg.orb_user,
        deadline=time.monotonic() + 10.0,
        action_label="test",
    )
    assert attempts == 3
    # Must NOT have invalidated a valid auth token on transient unavailable errors!
    assert backend.invalidated_auth_ids == []

    # Now verify auth errors DO invalidate cached auth immediately
    auth_attempts = 0

    def unauth_exec(sandbox_id: str, command: str, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal auth_attempts
        auth_attempts += 1
        if auth_attempts <= 2:
            raise APIError("Connect RPC failed (unauthenticated): HTTP 401")
        return orig_exec(sandbox_id, command, **kwargs)

    backend.execute_command = unauth_exec  # type: ignore[method-assign]
    _execute_when_gateway_ready(
        backend,  # type: ignore[arg-type]
        sbx.id,
        "true",
        envs=None,
        user=cfg.orb_user,
        deadline=time.monotonic() + 10.0,
        action_label="test-auth",
    )
    assert backend.invalidated_auth_ids == [sbx.id, sbx.id]

    # Verify prewarm_gateway_transport issues HEAD against the cached gateway_url via command pool
    head_urls: list[str] = []

    class FakeLeaseClient:
        def execute(self, method: str, url: str, **_kwargs):  # type: ignore[no-untyped-def]
            assert method == "HEAD"
            head_urls.append(url)

    class FakePool:
        def lease(self):  # type: ignore[no-untyped-def]
            import contextlib

            @contextlib.contextmanager
            def _cm():
                yield FakeLeaseClient()

            return _cm()

    class FakeSbxClient:
        def __init__(self) -> None:
            self._command_pool = FakePool()
            self._auth_cache = self

        def get_or_refresh(self, _sid: str) -> dict[str, str]:
            return {"gateway_url": "https://gateway.prime.example"}

    assert prewarm_gateway_transport(FakeSbxClient(), "sbx_1") is True  # type: ignore[arg-type]
    assert head_urls == ["https://gateway.prime.example/"]


def test_gateway_direct_create_skips_control_plane_poll_when_auth_seeded(tmp_path: Path) -> None:
    """When create() seeds gatewayAuth, cold spawn goes straight to gateway execute_command
    with zero GET /sandbox/{id} calls during normal placement."""
    from prime_sandboxes import APIError

    from prime_orbs.config import DispatchJob
    from prime_orbs.dispatcher import run_dispatch

    backend = FakePrimeBackend()
    img_adapter = _ImageClientAdapter(backend)
    cfg = ProviderConfig.from_env(
        {
            "PRIME_ORB_STATE_DIR": str(tmp_path),
            "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
            "PRIME_ORB_STANDBY_COUNT": "0",
        }
    )

    # Simulate real Prime SDK create() returning status=PENDING while seeding _auth_cache
    orig_create = backend.create
    orig_get = backend.get
    orig_exec = backend.execute_command
    get_calls: list[str] = []
    exec_attempts = 0
    auth_map: dict[str, dict[str, str]] = {}

    class _FakeAuthCache:
        def __init__(self) -> None:
            self._auth_cache = auth_map

        def invalidate(self, sandbox_id: str) -> None:
            backend.invalidate(sandbox_id)

    backend._auth_cache = _FakeAuthCache()  # type: ignore[assignment]

    def create_pending_with_auth(req):  # type: ignore[no-untyped-def]
        sbx = orig_create(req)
        backend.sandboxes[sbx.id]["status"] = "PENDING"
        auth_map[sbx.id] = {
            "gateway_url": "https://gateway.example",
            "user_ns": "ns",
            "job_id": sbx.id,
            "token": "tok",
        }
        return sbx.model_copy(update={"status": "PENDING"})

    def tracked_get(sbx_id: str):  # type: ignore[no-untyped-def]
        get_calls.append(sbx_id)
        return orig_get(sbx_id)

    def placing_then_ready_exec(sbx_id: str, command: str, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal exec_attempts
        exec_attempts += 1
        if exec_attempts <= 2:
            raise APIError(
                "Connect RPC failed (unavailable): "
                "The sandbox is being placed on a node; retry shortly"
            )
        backend.sandboxes[sbx_id]["status"] = "RUNNING"
        return orig_exec(sbx_id, command, **kwargs)

    backend.create = create_pending_with_auth  # type: ignore[method-assign]
    backend.get = tracked_get  # type: ignore[method-assign]
    backend.execute_command = placing_then_ready_exec  # type: ignore[method-assign]

    run_dispatch(
        DispatchJob.from_env(
            {
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-gw-direct",
                "ORB_TOKEN": "tok-gw",
                "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
            }
        ),
        cfg,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
        authoritative_index=True,
    )
    assert exec_attempts == 3
    assert get_calls == [], f"Expected 0 GET calls during fast gateway-direct boot, got {get_calls}"
    assert backend.invalidated_auth_ids == []


def test_reconcile_skips_redundant_gets_and_cool_idle_pops_before_pausing(tmp_path: Path) -> None:
    """Verify:
    - sync_known_orbs and StandbyPool.reconcile reuse list() results without N per-sandbox GETs
    - cool_idle_now pops a standby out of the pool while pausing so concurrent claims can't race it
    """
    from prime_orbs.config import DispatchJob
    from prime_orbs.dispatcher import run_dispatch

    backend = FakePrimeBackend()
    img_adapter = _ImageClientAdapter(backend)
    store = SnapshotStore(tmp_path)
    env = {
        "PRIME_ORB_STATE_DIR": str(tmp_path),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
        "PRIME_ORB_STANDBY_COUNT": "1",
        "PRIME_ORB_STANDBY_HOT_SECONDS": "900",
    }
    cfg = ProviderConfig.from_env(env)

    # Create 1 active orb and 1 standby sandbox
    run_dispatch(
        DispatchJob.from_env(
            {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-listed-orb", "ORB_TOKEN": "tok-1"}
        ),
        cfg,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=img_adapter,  # type: ignore[arg-type]
    )
    pool = StandbyPool(cfg, backend, img_adapter, store)  # type: ignore[arg-type]
    pool.start(run_background_worker=False)
    try:
        sbx = pool.replenish_once()
        assert sbx is not None

        get_calls: list[str] = []
        orig_get = backend.get

        def tracked_get(sbx_id: str):  # type: ignore[no-untyped-def]
            get_calls.append(sbx_id)
            return orig_get(sbx_id)

        backend.get = tracked_get  # type: ignore[method-assign]

        # Both startup syncs should find the persisted IDs via list() with 0 individual GETs!
        sync_known_orbs(backend, store)  # type: ignore[arg-type]
        pool.reconcile()
        assert get_calls == [], f"Expected 0 GETs during startup reconcile, got {get_calls}"

        # While cool_idle_now is pausing the standby, the pool bucket must not hand it to claim()
        claimed_during_pause: list[object] = []
        orig_pause = backend.pause

        def pause_with_concurrent_claim(sbx_id: str):  # type: ignore[no-untyped-def]
            claimed_during_pause.append(
                pool.claim(
                    "prime/xeophon/amp-orb:latest",
                    cfg.default_spec,
                    time.monotonic() + 10.0,
                    lambda _: None,
                )
            )
            return orig_pause(sbx_id)

        backend.pause = pause_with_concurrent_claim  # type: ignore[method-assign]
        pool._hot_seconds = 0.0
        assert pool.cool_idle_now() == 1
        assert claimed_during_pause == [None]
        assert backend.sandboxes[sbx.id]["status"] == "PAUSED"
    finally:
        pool.close()

