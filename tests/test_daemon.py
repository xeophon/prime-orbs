"""End-to-end tests for the warm UNIX-socket DispatchDaemon and stdlib dispatch.py fast path."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from test_dispatcher import FakePrimeBackend, _ImageClientAdapter

from prime_orbs.daemon import DAEMON_SOCKET_ENV, DispatchDaemon, daemon_socket_path


def test_dispatch_py_uses_warm_unix_socket_daemon_under_80ms(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    state_dir = tmp_path / "state"
    sock_path = daemon_socket_path(state_dir)
    backend = FakePrimeBackend()

    base_env = {
        "PRIME_ORB_STATE_DIR": str(state_dir),
        "ORB_IMAGE": "prime/xeophon/amp-orb:latest",
    }
    with DispatchDaemon(
        sock_path,
        base_env=base_env,
        sandbox_client=backend,  # type: ignore[arg-type]
        image_client=_ImageClientAdapter(backend),  # type: ignore[arg-type]
    ):
        assert sock_path.exists()
        assert (sock_path.stat().st_mode & 0o777) == 0o600

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
        assert "[prime-orbs] Creating amp-daemon-1" in proc.stderr
        assert "[prime-orbs] Started `amp orb bootstrap`" in proc.stderr
        assert len(backend.create_requests) == 1
        assert backend.exec_calls[0]["env"] == {"ORB_TOKEN": "tok_daemon_secret"}
        # Subprocess skips prime_sandboxes import over the warm socket (~20-50ms vs ~380ms)
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
                # Missing ORB_TOKEN
            },
            capture_output=True,
            text=True,
            timeout=5,
        )
        assert bad.returncode == 1
        assert "ORB_TOKEN" in bad.stderr

    assert not sock_path.exists()


def test_long_state_dir_falls_back_to_short_runtime_socket_path() -> None:
    deep_dir = Path("/tmp/" + ("very_long_segment_" * 8))
    sock = daemon_socket_path(deep_dir)
    assert len(str(sock).encode("utf-8")) <= 96
