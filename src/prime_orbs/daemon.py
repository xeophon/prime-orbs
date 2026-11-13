"""Warm local UNIX-socket worker keeping Prime HTTP/1.1 & HTTP/2 connection pools hot."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import socket
import socketserver
import threading
from collections.abc import Mapping
from pathlib import Path

from prime_sandboxes import APIClient, ImageClient, SandboxClient

from .config import DispatchJob, ProviderConfig
from .dispatcher import run_dispatch

DAEMON_SOCKET_ENV = "PRIME_ORBS_DAEMON_SOCKET"
_FORWARDED_ENV_PREFIXES = ("ORB_", "PRIME_ORB_", "AMP_", "PRIME_", "XDG_")


def daemon_socket_path(state_dir: Path) -> Path:
    """Return a private UNIX domain socket path safely within the 108-byte sun_path limit."""
    direct = state_dir / "daemon.sock"
    if len(str(direct).encode("utf-8")) <= 96:
        return direct
    digest = hashlib.sha256(str(state_dir).encode("utf-8")).hexdigest()[:12]
    runtime_dir = Path(f"/tmp/prime-orbs-{os.getuid()}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(runtime_dir, 0o700)
    return runtime_dir / f"{digest}.sock"


class _ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class DispatchDaemon:
    """In-process UNIX-socket server reusing warm Prime API & ConnectRPC transport pools."""

    def __init__(
        self,
        socket_path: Path,
        *,
        base_env: Mapping[str, str] | None = None,
        sandbox_client: SandboxClient | None = None,
        image_client: ImageClient | None = None,
    ) -> None:
        self.socket_path = socket_path
        self._base_env = dict(os.environ if base_env is None else base_env)
        if sandbox_client is None or image_client is None:
            api_client = APIClient()
            sandbox_client = sandbox_client or SandboxClient(api_client)
            image_client = image_client or ImageClient(api_client)
        self.sandbox_client = sandbox_client
        self.image_client = image_client
        self._server: _ThreadingUnixStreamServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            self.socket_path.unlink()

        daemon = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                raw_line = self.rfile.readline()
                if not raw_line:
                    return
                write_lock = threading.Lock()

                def send_event(payload: dict[str, object]) -> None:
                    encoded = (json.dumps(payload) + "\n").encode("utf-8")
                    with write_lock, contextlib.suppress(BrokenPipeError, OSError):
                        self.wfile.write(encoded)
                        self.wfile.flush()

                def log_fn(msg: str) -> None:
                    send_event({"type": "log", "line": f"[prime-orbs] {msg}"})

                try:
                    request = json.loads(raw_line.decode("utf-8"))
                    job_overrides = request.get("env") or {}
                    merged_env = dict(daemon._base_env)
                    # Clear job-scoped vars from base_env before overlaying the incoming job's env.
                    for key in list(merged_env):
                        if key.startswith("ORB_") and key != "ORB_IMAGE":
                            merged_env.pop(key, None)
                    for key, val in job_overrides.items():
                        merged_env[str(key)] = str(val)

                    job = DispatchJob.from_env(merged_env)
                    config = ProviderConfig.from_env(merged_env)
                    run_dispatch(
                        job,
                        config,
                        sandbox_client=daemon.sandbox_client,
                        image_client=daemon.image_client,
                        log_fn=log_fn,
                    )
                    send_event({"type": "done", "exit_code": 0})
                except Exception as exc:
                    send_event(
                        {
                            "type": "done",
                            "exit_code": 1,
                            "error": f"[prime-orbs] Error: {exc}",
                        }
                    )

        old_umask = os.umask(0o177)
        try:
            server = _ThreadingUnixStreamServer(str(self.socket_path), Handler)
        finally:
            os.umask(old_umask)
        os.chmod(self.socket_path, 0o600)

        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, name="prime-orbs-daemon", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        with contextlib.suppress(OSError):
            self.socket_path.unlink()

    def __enter__(self) -> DispatchDaemon:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def send_job_to_daemon(
    socket_path: str | Path,
    environ: Mapping[str, str] | None = None,
) -> int | None:
    """Send current ORB_* job to a running DispatchDaemon.

    Returns the integer exit code when handled by the daemon, or None if the daemon
    socket was unavailable (allowing caller fallback to standalone execution).
    """
    env = os.environ if environ is None else environ
    forwarded = {
        k: v for k, v in env.items() if k.startswith(_FORWARDED_ENV_PREFIXES)
    }
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(str(socket_path))
    except OSError:
        return None

    with sock:
        payload = (json.dumps({"env": forwarded}) + "\n").encode("utf-8")
        sock.sendall(payload)
        reader = sock.makefile("r", encoding="utf-8")
        for line in reader:
            event = json.loads(line)
            kind = event.get("type")
            if kind == "log":
                import sys

                print(event["line"], file=sys.stderr, flush=True)
            elif kind == "done":
                err = event.get("error")
                if err:
                    import sys

                    print(err, file=sys.stderr, flush=True)
                return int(event.get("exit_code", 1))
    return None
