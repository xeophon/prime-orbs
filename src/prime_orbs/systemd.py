"""Systemd user service and daily warm-template refresh timer generator/installer."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_REFRESH_ON_CALENDAR = "*-*-* 04:15:00"


@dataclass(frozen=True)
class SystemdUnits:
    service_name: str
    service_unit_path: Path
    service_content: str
    refresh_service_path: Path | None
    refresh_service_content: str | None
    refresh_timer_path: Path | None
    refresh_timer_content: str | None


def _resolve_prime_orbs_command(repo_root: Path) -> str:
    venv_entry = repo_root / ".venv" / "bin" / "prime-orbs"
    if venv_entry.is_file():
        return shlex.quote(str(venv_entry))
    uv_bin = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    return f"{shlex.quote(uv_bin)} run --project {shlex.quote(str(repo_root))} prime-orbs"


def _service_path_env() -> str:
    home = Path.home()
    entries = [
        str(home / ".amp" / "bin"),
        str(home / ".local" / "bin"),
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]
    amp_which = shutil.which("amp")
    if amp_which:
        amp_dir = str(Path(amp_which).resolve().parent)
        if amp_dir not in entries:
            entries.insert(0, amp_dir)
    uv_which = shutil.which("uv")
    if uv_which:
        uv_dir = str(Path(uv_which).resolve().parent)
        if uv_dir not in entries:
            entries.insert(0, uv_dir)
    return ":".join(entries)


def render_systemd_units(
    *,
    repo_root: Path,
    env_file: Path,
    unit_dir: Path,
    service_name: str = "prime-orbs",
    standby_count: int | None = None,
    team_id: str | None = None,
    force_personal: bool = False,
    include_refresh_timer: bool = True,
    refresh_on_calendar: str = DEFAULT_REFRESH_ON_CALENDAR,
) -> SystemdUnits:
    repo_root = repo_root.resolve()
    env_file = env_file.expanduser().resolve()
    unit_dir = unit_dir.expanduser().resolve()

    base_cmd = _resolve_prime_orbs_command(repo_root)
    scope_args: list[str] = []
    if force_personal:
        scope_args.append("--personal")
    elif team_id:
        scope_args.extend(["--team-id", shlex.quote(team_id)])

    run_args = ["run", "--warm-template", *scope_args]
    if standby_count is not None:
        run_args.extend(["--standby-count", str(standby_count)])
    exec_run = f"{base_cmd} {' '.join(run_args)}"
    warm_args = ["warm-template", *scope_args]
    exec_warm = f"{base_cmd} {' '.join(warm_args)}"
    path_env = _service_path_env()

    service_content = (
        "[Unit]\n"
        "Description=Amp Self-Hosted Orb Provider (Prime Sandboxes)\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=simple\n"
        f"WorkingDirectory={repo_root}\n"
        f"EnvironmentFile={env_file}\n"
        f'Environment="PATH={path_env}"\n'
        f"ExecStart={exec_run}\n"
        "Restart=always\n"
        "RestartSec=5\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    service_path = unit_dir / f"{service_name}.service"

    if not include_refresh_timer:
        return SystemdUnits(
            service_name=service_name,
            service_unit_path=service_path,
            service_content=service_content,
            refresh_service_path=None,
            refresh_service_content=None,
            refresh_timer_path=None,
            refresh_timer_content=None,
        )

    refresh_name = f"{service_name}-refresh"
    refresh_service_content = (
        "[Unit]\n"
        "Description=Refresh Prime Orbs Warm Memory Checkpoint & Amp CLI\n"
        "After=network-online.target\n"
        "Wants=network-online.target\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"WorkingDirectory={repo_root}\n"
        f"EnvironmentFile={env_file}\n"
        f'Environment="PATH={path_env}"\n'
        f"ExecStart={exec_warm}\n"
    )
    refresh_timer_content = (
        "[Unit]\n"
        "Description=Daily Prime Orbs Warm Memory Checkpoint Refresh\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar={refresh_on_calendar}\n"
        "Persistent=true\n"
        "RandomizedDelaySec=15m\n"
        f"Unit={refresh_name}.service\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    return SystemdUnits(
        service_name=service_name,
        service_unit_path=service_path,
        service_content=service_content,
        refresh_service_path=unit_dir / f"{refresh_name}.service",
        refresh_service_content=refresh_service_content,
        refresh_timer_path=unit_dir / f"{refresh_name}.timer",
        refresh_timer_content=refresh_timer_content,
    )


def ensure_env_file_placeholder(
    env_file: Path,
    *,
    team_id: str | None = None,
    force_personal: bool = False,
) -> bool:
    """Ensure `env_file` exists with mode 0600. Returns True if a new placeholder was created."""
    env_file = env_file.expanduser()
    if env_file.exists():
        os.chmod(env_file, 0o600)
        return False
    env_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(env_file, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    team_line = (
        'PRIME_TEAM_ID=""\n'
        if force_personal
        else (f"PRIME_TEAM_ID={team_id}\n" if team_id else "# PRIME_TEAM_ID=\n")
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(
            "# Prime Orbs credentials (mode 0600)\n"
            "PRIME_API_KEY=\n"
            "AMP_ORB_PROVIDER_TOKEN=\n"
            f"{team_line}"
        )
    return True


def write_systemd_units(units: SystemdUnits) -> list[Path]:
    units.service_unit_path.parent.mkdir(parents=True, exist_ok=True)
    written = [units.service_unit_path]
    units.service_unit_path.write_text(units.service_content, encoding="utf-8")

    if units.refresh_service_path and units.refresh_service_content:
        units.refresh_service_path.write_text(units.refresh_service_content, encoding="utf-8")
        written.append(units.refresh_service_path)
    if units.refresh_timer_path and units.refresh_timer_content:
        units.refresh_timer_path.write_text(units.refresh_timer_content, encoding="utf-8")
        written.append(units.refresh_timer_path)
    return written


def activate_systemd_units(units: SystemdUnits) -> list[str]:
    """Reload systemd --user, enable/start units, and return any operator follow-up notes."""
    notes: list[str] = []
    if not shutil.which("systemctl"):
        notes.append("`systemctl` not found on PATH; wrote unit files without enabling them.")
        return notes

    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", f"{units.service_name}.service"],
        check=True,
    )
    if units.refresh_timer_path is not None:
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", units.refresh_timer_path.name],
            check=True,
        )

    user = os.environ.get("USER") or ""
    if user and shutil.which("loginctl"):
        proc = subprocess.run(
            ["loginctl", "show-user", user, "-p", "Linger"],
            capture_output=True,
            text=True,
        )
        if "Linger=yes" not in proc.stdout:
            notes.append(
                f"Run `sudo loginctl enable-linger {shlex.quote(user)}` so the user service "
                "stays running after SSH logout and reboots."
            )
    return notes
