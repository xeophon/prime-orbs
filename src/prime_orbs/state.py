"""Concurrency locks and local checkpoint/sandbox mapping state for Prime orbs."""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_STORE_SECTIONS = (
    "project_snapshots",
    "orb_checkpoints",
    "orb_sandboxes",
    "warm_templates",
)


def _sanitize_key(key: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", key.strip())
    return cleaned[:120] or "default"


@contextlib.contextmanager
def acquire_file_lock(lock_path: Path, timeout_seconds: float = 270.0) -> Iterator[None]:
    """Acquire an exclusive flock on `lock_path` within `timeout_seconds`.

    Lockfiles are intentionally never unlinked on release so concurrent processes
    always lock the same inode.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    delay = 0.05
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Timed out waiting {timeout_seconds:g}s for lock {lock_path}"
                    ) from exc
                time.sleep(min(delay, remaining))
                delay = min(delay * 1.5, 1.0)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextlib.contextmanager
def orb_lock(state_dir: Path, orb_id: str, timeout_seconds: float = 270.0) -> Iterator[None]:
    """Serialize lifecycle operations for a single `orb_id` while allowing parallel orbs."""
    lock_path = state_dir / "locks" / f"{_sanitize_key(orb_id)}.lock"
    with acquire_file_lock(lock_path, timeout_seconds=timeout_seconds):
        yield


class SnapshotStore:
    """Atomic JSON store mapping Amp orb/snapshot/template IDs to Prime IDs."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.path = state_dir / "snapshots.json"
        self._lock_path = state_dir / "snapshots.lock"

    # --- Shared project snapshots (ORB_SNAPSHOT_ID -> checkpoint_id) ---
    def get_project_snapshot(self, orb_snapshot_id: str) -> str | None:
        return self._read()["project_snapshots"].get(orb_snapshot_id)

    def all_project_snapshots(self) -> dict[str, str]:
        return dict(self._read()["project_snapshots"])

    def set_project_snapshot(self, orb_snapshot_id: str, checkpoint_id: str) -> None:
        with self._mutate() as data:
            data["project_snapshots"][orb_snapshot_id] = checkpoint_id

    def remove_project_snapshot(self, orb_snapshot_id: str) -> None:
        with self._mutate() as data:
            data["project_snapshots"].pop(orb_snapshot_id, None)

    # --- Per-orb pause/GPU recovery checkpoints (ORB_ID -> checkpoint_id) ---
    def get_orb_checkpoint(self, orb_id: str) -> str | None:
        return self._read()["orb_checkpoints"].get(orb_id)

    def all_orb_checkpoints(self) -> dict[str, str]:
        return dict(self._read()["orb_checkpoints"])

    def set_orb_checkpoint(self, orb_id: str, checkpoint_id: str) -> None:
        with self._mutate() as data:
            data["orb_checkpoints"][orb_id] = checkpoint_id

    def remove_orb_checkpoint(self, orb_id: str) -> None:
        with self._mutate() as data:
            data["orb_checkpoints"].pop(orb_id, None)

    # --- Active sandbox ID cache (ORB_ID -> sandbox_id) ---
    def get_orb_sandbox(self, orb_id: str) -> str | None:
        return self._read()["orb_sandboxes"].get(orb_id)

    def all_orb_sandboxes(self) -> dict[str, str]:
        return dict(self._read()["orb_sandboxes"])

    def set_orb_sandbox(self, orb_id: str, sandbox_id: str) -> None:
        with self._mutate() as data:
            data["orb_sandboxes"][orb_id] = sandbox_id

    def remove_orb_sandbox(self, orb_id: str) -> None:
        with self._mutate() as data:
            data["orb_sandboxes"].pop(orb_id, None)

    # --- Warm memory-fork templates (template_key -> pause checkpoint_id) ---
    def get_warm_template(self, key: str) -> str | None:
        return self._read()["warm_templates"].get(key)

    def all_warm_templates(self) -> dict[str, str]:
        return dict(self._read()["warm_templates"])

    def set_warm_template(self, key: str, checkpoint_id: str) -> str | None:
        with self._mutate() as data:
            previous = data["warm_templates"].get(key)
            data["warm_templates"][key] = checkpoint_id
            return previous

    def remove_warm_template(self, key: str) -> None:
        with self._mutate() as data:
            data["warm_templates"].pop(key, None)

    def _empty(self) -> dict[str, dict[str, str]]:
        return {section: {} for section in _STORE_SECTIONS}

    def _read(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return self._empty()
        with acquire_file_lock(self._lock_path, timeout_seconds=30.0):
            return self._read_unlocked()

    def _read_unlocked(self) -> dict[str, dict[str, str]]:
        if not self.path.is_file():
            return self._empty()
        try:
            raw: Any = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty()
        if not isinstance(raw, dict):
            return self._empty()
        parsed = self._empty()
        for section in _STORE_SECTIONS:
            value = raw.get(section)
            if isinstance(value, dict):
                parsed[section] = {str(k): str(v) for k, v in value.items()}
        return parsed

    @contextlib.contextmanager
    def _mutate(self) -> Iterator[dict[str, dict[str, str]]]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with acquire_file_lock(self._lock_path, timeout_seconds=30.0):
            data = self._read_unlocked()
            yield data
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self.state_dir), prefix=".snapshots.", suffix=".tmp"
            )
            try:
                os.fchmod(fd, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, self.path)
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise
