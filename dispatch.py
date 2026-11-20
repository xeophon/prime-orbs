#!/usr/bin/env python3
"""Amp self-hosted orb dispatch entrypoint for Prime Intellect Sandboxes.

When `prime-orbs run` is hosting its warm UNIX-socket daemon (`PRIME_ORBS_DAEMON_SOCKET`),
this script forwards the job over the socket using only minimal Python standard library
imports (~10ms, no SDK import, pathlib, shutil, or virtualenv re-exec). When invoked
standalone (`amp orb provider ./dispatch.py`), it executes `prime_orbs` directly.
"""

from __future__ import annotations

import json
import os
import socket
import sys

_FORWARDED_PREFIXES = ("ORB_", "PRIME_ORB_", "AMP_", "PRIME_", "XDG_")


def _try_daemon_fast_path() -> int | None:
    sock_path = os.environ.get("PRIME_ORBS_DAEMON_SOCKET")
    if not sock_path:
        return None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(sock_path)
    except OSError:
        return None

    forwarded = {k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_PREFIXES)}
    with sock:
        sock.sendall((json.dumps({"env": forwarded}) + "\n").encode("utf-8"))
        for line in sock.makefile("r", encoding="utf-8"):
            event = json.loads(line)
            kind = event.get("type")
            if kind == "log":
                print(event["line"], file=sys.stderr, flush=True)
            elif kind == "done":
                err = event.get("error")
                if err:
                    print(err, file=sys.stderr, flush=True)
                return int(event.get("exit_code", 1))
    return None


def _ensure_environment() -> None:
    import shutil
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    try:
        import prime_sandboxes  # noqa: F401
        return
    except ImportError:
        pass

    if os.environ.get("_PRIME_ORBS_REEXEC") == "1":
        raise SystemExit(
            "prime-sandboxes is not installed. Run `uv sync` in "
            f"{repo_root} before starting `amp orb provider`."
        )

    env = dict(os.environ, _PRIME_ORBS_REEXEC="1")
    venv_python = repo_root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        os.execve(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]], env)

    uv_bin = shutil.which("uv")
    if uv_bin:
        os.execve(
            uv_bin,
            [uv_bin, "run", "--project", str(repo_root), "python", __file__, *sys.argv[1:]],
            env,
        )

    raise SystemExit(
        "prime-sandboxes is not installed and `uv` was not found on PATH. "
        "Install dependencies with `uv sync` or `pip install -e .`."
    )


def main() -> int:
    daemon_code = _try_daemon_fast_path()
    if daemon_code is not None:
        return daemon_code

    _ensure_environment()
    from prime_orbs.cli import cmd_dispatch

    return cmd_dispatch(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[prime-orbs] Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
