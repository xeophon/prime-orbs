#!/usr/bin/env python3
"""Amp self-hosted orb dispatch entrypoint for Prime Intellect Sandboxes.

Usage:
    amp orb provider ./dispatch.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"


def _ensure_environment() -> None:
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        import prime_sandboxes  # noqa: F401
        return
    except ImportError:
        pass

    if os.environ.get("_PRIME_ORBS_REEXEC") == "1":
        raise SystemExit(
            "prime-sandboxes is not installed. Run `uv sync` in "
            f"{REPO_ROOT} before starting `amp orb provider`."
        )

    env = dict(os.environ, _PRIME_ORBS_REEXEC="1")
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        os.execve(str(venv_python), [str(venv_python), __file__, *sys.argv[1:]], env)

    uv_bin = shutil.which("uv")
    if uv_bin:
        os.execve(
            uv_bin,
            [uv_bin, "run", "--project", str(REPO_ROOT), "python", __file__, *sys.argv[1:]],
            env,
        )

    raise SystemExit(
        "prime-sandboxes is not installed and `uv` was not found on PATH. "
        "Install dependencies with `uv sync` or `pip install -e .`."
    )


def main() -> int:
    _ensure_environment()
    from prime_orbs.cli import cmd_dispatch

    return cmd_dispatch(None)  # type: ignore[arg-type]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[prime-orbs] Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
