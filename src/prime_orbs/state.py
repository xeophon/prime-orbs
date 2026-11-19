"""Concurrency locks and mtime-cached atomic state store for Prime orbs."""

from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_STORE_SECTIONS = (
    "project_snapshots",
    "project_snapshot_modes",
    "orb_checkpoints",
    "orb_sandboxes",
    "orb_states",
    "orb_teams",
    "warm_templates",
    "standby_sandboxes",
)

_CACHE_LOCK = threading.Lock()
# resolved_path -> (st_mtime_ns, st_size, parsed_data)
_MEMORY_CACHE: dict[Path, tuple[int, int, dict[str, dict[str, str]]]] = {}


def _scoped_snapshot_key(orb_snapshot_id: str, team_id: str | None) -> str:
    return f"team:{team_id}:{orb_snapshot_id}" if team_id else orb_snapshot_id


def _matches_snapshot_scope(stored_key: str, team_id: str | None) -> str | None:
    prefix = f"team:{team_id}:" if team_id else None
    if prefix is not None:
        return stored_key[len(prefix) :] if stored_key.startswith(prefix) else None
    return None if stored_key.startswith("team:") else stored_key


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
    """Atomic JSON store with an st_mtime_ns in-memory cache."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.path = state_dir / "snapshots.json"
        self._resolved_path = self.path.resolve()
        self._lock_path = state_dir / "snapshots.lock"

    # --- Shared project snapshots (scoped by team_id; backward-compatible for personal) ---
    def get_project_snapshot(
        self, orb_snapshot_id: str, *, team_id: str | None = None
    ) -> str | None:
        key = _scoped_snapshot_key(orb_snapshot_id, team_id)
        return self._read()["project_snapshots"].get(key)

    def get_project_snapshot_entry(
        self, orb_snapshot_id: str, *, team_id: str | None = None
    ) -> tuple[str | None, str]:
        """Return `(checkpoint_id, mode)` (`'disk'`, `'uploading'`, or `'memory:...'`)."""
        data = self._read()
        key = _scoped_snapshot_key(orb_snapshot_id, team_id)
        ckpt = data["project_snapshots"].get(key)
        mode = data["project_snapshot_modes"].get(key, "disk")
        return ckpt, mode

    def all_project_snapshots(
        self, *, team_id: str | None = None, all_scopes: bool = False
    ) -> dict[str, str]:
        raw = self._read()["project_snapshots"]
        if all_scopes:
            return dict(raw)
        result: dict[str, str] = {}
        for k, v in raw.items():
            unscoped = _matches_snapshot_scope(k, team_id)
            if unscoped is not None:
                result[unscoped] = v
        return result

    def all_project_snapshot_modes(
        self, *, team_id: str | None = None, all_scopes: bool = False
    ) -> dict[str, str]:
        raw = self._read()["project_snapshot_modes"]
        if all_scopes:
            return dict(raw)
        result: dict[str, str] = {}
        for k, v in raw.items():
            unscoped = _matches_snapshot_scope(k, team_id)
            if unscoped is not None:
                result[unscoped] = v
        return result

    def set_project_snapshot(
        self,
        orb_snapshot_id: str,
        checkpoint_id: str,
        *,
        mode: str = "disk",
        team_id: str | None = None,
    ) -> str | None:
        key = _scoped_snapshot_key(orb_snapshot_id, team_id)
        with self._mutate() as data:
            previous = data["project_snapshots"].get(key)
            data["project_snapshots"][key] = checkpoint_id
            data["project_snapshot_modes"][key] = mode
            return previous

    def remove_project_snapshot(
        self, orb_snapshot_id: str, *, team_id: str | None = None
    ) -> None:
        key = _scoped_snapshot_key(orb_snapshot_id, team_id)
        with self._mutate() as data:
            data["project_snapshots"].pop(key, None)
            data["project_snapshot_modes"].pop(key, None)

    def remove_raw_project_snapshot_key(self, raw_key: str) -> None:
        with self._mutate() as data:
            data["project_snapshots"].pop(raw_key, None)
            data["project_snapshot_modes"].pop(raw_key, None)

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

    # --- Active sandbox ID & state cache (ORB_ID -> sandbox_id, status) ---
    def get_orb_sandbox(self, orb_id: str) -> str | None:
        return self._read()["orb_sandboxes"].get(orb_id)

    def get_orb_sandbox_with_state(self, orb_id: str) -> tuple[str | None, str | None]:
        data = self._read()
        sbx_id = data["orb_sandboxes"].get(orb_id)
        if sbx_id is None:
            return None, None
        return sbx_id, data["orb_states"].get(orb_id)

    def get_orb_team(self, orb_id: str) -> str | None:
        val = self._read()["orb_teams"].get(orb_id, "").strip()
        return val or None

    def all_orb_sandboxes(self) -> dict[str, str]:
        return dict(self._read()["orb_sandboxes"])

    def set_orb_sandbox(
        self,
        orb_id: str,
        sandbox_id: str,
        *,
        status: str | None = None,
        team_id: str | None = None,
    ) -> None:
        with self._mutate() as data:
            data["orb_sandboxes"][orb_id] = sandbox_id
            if status is not None:
                data["orb_states"][orb_id] = status
            if team_id:
                data["orb_teams"][orb_id] = team_id
            elif team_id is None and orb_id not in data["orb_teams"]:
                pass

    def set_orb_state(self, orb_id: str, status: str) -> None:
        with self._mutate() as data:
            if orb_id in data["orb_sandboxes"]:
                data["orb_states"][orb_id] = status

    def replace_orb_sandboxes(
        self,
        mapping: dict[str, str],
        states: dict[str, str] | None = None,
        teams: dict[str, str | None] | None = None,
    ) -> None:
        with self._mutate() as data:
            data["orb_sandboxes"] = dict(mapping)
            if states is not None:
                data["orb_states"] = {
                    orb_id: st for orb_id, st in states.items() if orb_id in mapping
                }
            else:
                data["orb_states"] = {
                    orb_id: st
                    for orb_id, st in data["orb_states"].items()
                    if orb_id in mapping
                }
            if teams is not None:
                data["orb_teams"] = {
                    orb_id: tid
                    for orb_id, tid in teams.items()
                    if orb_id in mapping and tid
                }
            else:
                data["orb_teams"] = {
                    orb_id: tid
                    for orb_id, tid in data["orb_teams"].items()
                    if orb_id in mapping and tid
                }

    def remove_orb_sandbox(self, orb_id: str) -> None:
        with self._mutate() as data:
            data["orb_sandboxes"].pop(orb_id, None)
            data["orb_states"].pop(orb_id, None)
            data["orb_teams"].pop(orb_id, None)

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

    # --- Standby pool sandboxes (standby_key -> comma-separated sandbox_ids) ---
    def get_standby_ids(self, key: str) -> list[str]:
        raw = self._read()["standby_sandboxes"].get(key, "")
        return [item for item in raw.split(",") if item]

    def all_standby_ids(self) -> dict[str, list[str]]:
        return {
            key: [item for item in raw.split(",") if item]
            for key, raw in self._read()["standby_sandboxes"].items()
            if raw
        }

    def set_standby_ids(self, key: str, sandbox_ids: list[str]) -> None:
        deduped = list(dict.fromkeys(s for s in sandbox_ids if s))
        with self._mutate() as data:
            if deduped:
                data["standby_sandboxes"][key] = ",".join(deduped)
            else:
                data["standby_sandboxes"].pop(key, None)

    def referenced_checkpoint_ids(self) -> set[str]:
        """Return every checkpoint ID currently referenced across snapshots, orbs, or templates."""
        data = self._read()
        return (
            set(data["project_snapshots"].values())
            | set(data["orb_checkpoints"].values())
            | set(data["warm_templates"].values())
        )

    def _empty(self) -> dict[str, dict[str, str]]:
        return {section: {} for section in _STORE_SECTIONS}

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            with _CACHE_LOCK:
                _MEMORY_CACHE.pop(self._resolved_path, None)
            return self._empty()

        with _CACHE_LOCK:
            cached = _MEMORY_CACHE.get(self._resolved_path)
            if cached is not None and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                return cached[2]

        with acquire_file_lock(self._lock_path, timeout_seconds=30.0):
            return self._read_unlocked_and_cache()

    def _read_unlocked_and_cache(self) -> dict[str, dict[str, str]]:
        try:
            st = self.path.stat()
            raw_text = self.path.read_text(encoding="utf-8")
        except OSError:
            with _CACHE_LOCK:
                _MEMORY_CACHE.pop(self._resolved_path, None)
            return self._empty()

        try:
            raw: Any = json.loads(raw_text)
        except json.JSONDecodeError:
            return self._empty()
        if not isinstance(raw, dict):
            return self._empty()

        parsed = self._empty()
        for section in _STORE_SECTIONS:
            value = raw.get(section)
            if isinstance(value, dict):
                parsed[section] = {str(k): str(v) for k, v in value.items()}

        with _CACHE_LOCK:
            _MEMORY_CACHE[self._resolved_path] = (
                st.st_mtime_ns,
                st.st_size,
                copy.deepcopy(parsed),
            )
        return parsed

    @contextlib.contextmanager
    def _mutate(self) -> Iterator[dict[str, dict[str, str]]]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with acquire_file_lock(self._lock_path, timeout_seconds=30.0):
            data = self._read_unlocked_and_cache()
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
                st = self.path.stat()
                with _CACHE_LOCK:
                    _MEMORY_CACHE[self._resolved_path] = (
                        st.st_mtime_ns,
                        st.st_size,
                        copy.deepcopy(data),
                    )
            except BaseException:
                with contextlib.suppress(OSError):
                    os.unlink(tmp_name)
                raise
