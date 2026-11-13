"""Configuration and Amp dispatch contract parsing for Prime Sandboxes orbs."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

DEFAULT_AMP_ORB_IMAGE_X64 = "us-central1-docker.pkg.dev/amp-prod-0/amp-orb/amp-orb:latest"
DEFAULT_AMP_ORB_IMAGE_ARM64 = "us-central1-docker.pkg.dev/amp-prod-0/amp-orb/amp-orb-arm64:latest"

DEFAULT_CPU_CORES = 4.0
DEFAULT_MEMORY_GB = 8.0
DEFAULT_DISK_SIZE_GB = 20.0
# Negative timeout disables Prime's 60-minute lifetime cap (paused time counts toward it).
DEFAULT_TIMEOUT_MINUTES = -1
# Amp expires project snapshots after 72h; 96h lets Prime auto-GC stale checkpoints safely.
DEFAULT_SNAPSHOT_TTL_SECONDS = 96 * 3600
# Amp enforces a 5-minute (300s) ceiling on each dispatch invocation.
DEFAULT_DISPATCH_TIMEOUT_SECONDS = 270.0

OrbOperation = Literal["spawn", "pause", "delete"]
OrbArch = Literal["amd64", "arm64"]


@dataclass(frozen=True)
class OrbSpec:
    """Resource and runtime parameters for a Prime Sandbox orb."""

    image: str
    arch: OrbArch | None = None
    cpu_cores: float = DEFAULT_CPU_CORES
    memory_gb: float = DEFAULT_MEMORY_GB
    disk_size_gb: float = DEFAULT_DISK_SIZE_GB
    gpu_count: int = 0
    gpu_type: str | None = None
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES
    idle_timeout_minutes: int | None = None
    region: str | None = None
    network_allowlist: list[str] | None = None
    network_denylist: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.image or not self.image.strip():
            raise ValueError("Orb image must be non-empty")
        if self.arch not in (None, "amd64", "arm64"):
            raise ValueError(f"Unsupported architecture {self.arch!r}; expected 'amd64' or 'arm64'")
        if not 1 <= self.cpu_cores <= 16:
            raise ValueError(
                f"cpu_cores must be between 1 and 16 on Prime Sandboxes (got {self.cpu_cores})"
            )
        if not 0.125 <= self.memory_gb <= 64:
            raise ValueError(
                f"memory_gb must be between 0.125 and 64 on Prime Sandboxes (got {self.memory_gb})"
            )
        if not 2 <= self.disk_size_gb <= 128:
            raise ValueError(
                f"disk_size_gb must be between 2 and 128 on Prime Sandboxes "
                f"(got {self.disk_size_gb})"
            )
        if not 0 <= self.gpu_count <= 8:
            raise ValueError(
                f"gpu_count must be between 0 and 8 on Prime Sandboxes (got {self.gpu_count})"
            )
        if self.gpu_count > 0 and not self.gpu_type:
            raise ValueError("gpu_type is required when gpu_count > 0")
        if self.gpu_count == 0 and self.gpu_type is not None:
            raise ValueError("gpu_type cannot be set when gpu_count == 0")
        if self.timeout_minutes == 0:
            raise ValueError("timeout_minutes must be positive or negative (-1 for unlimited)")
        if self.idle_timeout_minutes is not None:
            if not 1 <= self.idle_timeout_minutes <= 1440:
                raise ValueError(
                    f"idle_timeout_minutes must be between 1 and 1440 "
                    f"(got {self.idle_timeout_minutes})"
                )
            if self.timeout_minutes > 0 and self.idle_timeout_minutes > self.timeout_minutes:
                raise ValueError(
                    f"idle_timeout_minutes ({self.idle_timeout_minutes}) cannot exceed "
                    f"finite timeout_minutes ({self.timeout_minutes})"
                )
        if self.network_allowlist is not None and self.network_denylist is not None:
            raise ValueError("network_allowlist and network_denylist are mutually exclusive")


BUILTIN_ORB_TYPE_OVERRIDES: dict[str, dict[str, Any]] = {
    "tiny": {"cpu_cores": 2.0, "memory_gb": 4.0, "disk_size_gb": 10.0},
    "small": {"cpu_cores": 4.0, "memory_gb": 8.0, "disk_size_gb": 20.0},
    "medium": {"cpu_cores": 8.0, "memory_gb": 16.0, "disk_size_gb": 40.0},
    "large": {"cpu_cores": 16.0, "memory_gb": 32.0, "disk_size_gb": 80.0},
    "xlarge": {"cpu_cores": 16.0, "memory_gb": 64.0, "disk_size_gb": 128.0},
}


@dataclass(frozen=True)
class DispatchJob:
    """Parsed environment contract passed by `amp orb provider` for one lifecycle job."""

    operation: OrbOperation
    orb_id: str
    orb_token: str | None = None
    orb_image: str | None = None
    orb_snapshot_id: str | None = None
    orb_type: str | None = None
    amp_url: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> DispatchJob:
        env = os.environ if environ is None else environ
        raw_op = (env.get("ORB_OPERATION") or "").strip()
        if raw_op not in ("spawn", "pause", "delete"):
            raise ValueError(
                f"Invalid or missing ORB_OPERATION={raw_op!r}; "
                "expected 'spawn', 'pause', or 'delete'"
            )
        operation: OrbOperation = raw_op  # type: ignore[assignment]

        orb_id = (env.get("ORB_ID") or "").strip()
        if not orb_id:
            raise ValueError("Missing required environment variable ORB_ID")

        orb_token = _optional_str(env.get("ORB_TOKEN"))
        if operation == "spawn" and not orb_token:
            raise ValueError("Missing required environment variable ORB_TOKEN for spawn operation")

        return cls(
            operation=operation,
            orb_id=orb_id,
            orb_token=orb_token,
            orb_image=_optional_str(env.get("ORB_IMAGE")),
            orb_snapshot_id=_optional_str(env.get("ORB_SNAPSHOT_ID")),
            orb_type=_optional_str(env.get("ORB_TYPE")),
            amp_url=_optional_str(env.get("AMP_URL")),
        )


@dataclass(frozen=True)
class ProviderConfig:
    """Provider-level settings resolved from environment variables."""

    default_spec: OrbSpec
    orb_types: dict[str, dict[str, Any]]
    orb_user: str | None
    state_dir: Path
    snapshots_enabled: bool
    snapshot_ttl_seconds: int | None
    dispatch_timeout_seconds: float
    prime_orb_image_override: str | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProviderConfig:
        env = os.environ if environ is None else environ

        prime_image_override = _optional_str(env.get("PRIME_ORB_IMAGE"))
        job_image = _optional_str(env.get("ORB_IMAGE"))
        base_image = prime_image_override or job_image or DEFAULT_AMP_ORB_IMAGE_X64

        explicit_arch = _parse_arch(_optional_str(env.get("PRIME_ORB_ARCH")))
        inferred_arch: OrbArch | None = explicit_arch
        if inferred_arch is None and "arm64" in (job_image or base_image).lower():
            inferred_arch = "arm64"

        cpu_cores = _parse_float(env, "PRIME_ORB_CPU_CORES", DEFAULT_CPU_CORES)
        memory_gb = _parse_float(env, "PRIME_ORB_MEMORY_GB", DEFAULT_MEMORY_GB)
        disk_size_gb = _parse_float(env, "PRIME_ORB_DISK_SIZE_GB", DEFAULT_DISK_SIZE_GB)
        gpu_count = _parse_int(env, "PRIME_ORB_GPU_COUNT", 0)
        gpu_type = _optional_str(env.get("PRIME_ORB_GPU_TYPE"))
        timeout_minutes = _parse_int(env, "PRIME_ORB_TIMEOUT_MINUTES", DEFAULT_TIMEOUT_MINUTES)
        idle_timeout_raw = _optional_str(env.get("PRIME_ORB_IDLE_TIMEOUT_MINUTES"))
        idle_timeout_minutes = int(idle_timeout_raw) if idle_timeout_raw is not None else None
        region = _optional_str(env.get("PRIME_ORB_REGION"))
        allowlist = _parse_csv(env.get("PRIME_ORB_NETWORK_ALLOWLIST"))
        denylist = _parse_csv(env.get("PRIME_ORB_NETWORK_DENYLIST"))

        default_spec = OrbSpec(
            image=base_image,
            arch=inferred_arch,
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            disk_size_gb=disk_size_gb,
            gpu_count=gpu_count,
            gpu_type=gpu_type,
            timeout_minutes=timeout_minutes,
            idle_timeout_minutes=idle_timeout_minutes,
            region=region,
            network_allowlist=allowlist,
            network_denylist=denylist,
        )

        orb_types: dict[str, dict[str, Any]] = {
            name: dict(overrides) for name, overrides in BUILTIN_ORB_TYPE_OVERRIDES.items()
        }
        custom_types_raw = _optional_str(env.get("PRIME_ORB_TYPES"))
        if custom_types_raw:
            orb_types.update(_load_orb_types(custom_types_raw))

        raw_user = env.get("PRIME_ORB_USER")
        if raw_user is None:
            orb_user: str | None = "user"
        else:
            orb_user = raw_user.strip() or None

        state_dir_raw = _optional_str(env.get("PRIME_ORB_STATE_DIR"))
        if state_dir_raw:
            state_dir = Path(state_dir_raw).expanduser()
        else:
            xdg_state = _optional_str(env.get("XDG_STATE_HOME"))
            base_state = Path(xdg_state).expanduser() if xdg_state else Path.home() / ".local/state"
            state_dir = base_state / "prime-orbs"

        snapshots_enabled = _parse_bool(env.get("PRIME_ORB_SNAPSHOTS"), default=True)
        ttl_raw = _parse_int(env, "PRIME_ORB_SNAPSHOT_TTL_SECONDS", DEFAULT_SNAPSHOT_TTL_SECONDS)
        snapshot_ttl_seconds = ttl_raw if ttl_raw > 0 else None

        dispatch_timeout_seconds = _parse_float(
            env, "PRIME_ORB_DISPATCH_TIMEOUT_SECONDS", DEFAULT_DISPATCH_TIMEOUT_SECONDS
        )
        if dispatch_timeout_seconds <= 0:
            raise ValueError("PRIME_ORB_DISPATCH_TIMEOUT_SECONDS must be > 0")

        return cls(
            default_spec=default_spec,
            orb_types=orb_types,
            orb_user=orb_user,
            state_dir=state_dir,
            snapshots_enabled=snapshots_enabled,
            snapshot_ttl_seconds=snapshot_ttl_seconds,
            dispatch_timeout_seconds=dispatch_timeout_seconds,
            prime_orb_image_override=prime_image_override,
        )

    def resolve_spec(self, job: DispatchJob) -> OrbSpec:
        """Resolve the effective OrbSpec for a dispatch job, applying job image and ORB_TYPE."""
        image = self.prime_orb_image_override or job.orb_image or self.default_spec.image
        arch = self.default_spec.arch
        if arch is None and "arm64" in (job.orb_image or image).lower():
            arch = "arm64"

        spec = replace(self.default_spec, image=image, arch=arch)
        if not job.orb_type:
            return spec

        if job.orb_type not in self.orb_types:
            known = ", ".join(sorted(self.orb_types)) or "(none)"
            raise ValueError(
                f"Unknown ORB_TYPE={job.orb_type!r}. "
                f"Configured orb types: {known}. "
                "Add custom orb types via PRIME_ORB_TYPES (JSON object or file path)."
            )

        overrides = dict(self.orb_types[job.orb_type])
        if "arch" in overrides and overrides["arch"] is not None:
            overrides["arch"] = _parse_arch(str(overrides["arch"]))
        valid_fields = set(OrbSpec.__dataclass_fields__)
        unknown_fields = set(overrides) - valid_fields
        if unknown_fields:
            raise ValueError(
                f"ORB_TYPE {job.orb_type!r} contains unknown fields: {sorted(unknown_fields)}"
            )
        return replace(spec, **overrides)


def _optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_arch(value: str | None) -> OrbArch | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized == "arm":
        normalized = "arm64"
    if normalized in ("amd64", "x86_64", "linux/amd64"):
        return "amd64"
    if normalized in ("arm64", "aarch64", "linux/arm64"):
        return "arm64"
    raise ValueError(f"Unsupported architecture {value!r}; expected 'amd64' or 'arm64'")


def _parse_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _optional_str(env.get(key))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {key}={raw!r}") from exc


def _parse_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _optional_str(env.get(key))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for {key}={raw!r}") from exc


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    val = _optional_str(raw)
    if val is None:
        return default
    lowered = val.lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value {raw!r}; expected true/false or 1/0")


def _parse_csv(raw: str | None) -> list[str] | None:
    val = _optional_str(raw)
    if val is None:
        return None
    entries = [item.strip() for item in val.split(",") if item.strip()]
    return entries or None


def _load_orb_types(raw: str) -> dict[str, dict[str, Any]]:
    stripped = raw.strip()
    if stripped.startswith("{"):
        parsed = json.loads(stripped)
    else:
        path = Path(stripped).expanduser()
        if not path.is_file():
            raise ValueError(
                f"PRIME_ORB_TYPES must be a JSON object or path to an existing JSON file "
                f"(got {raw!r})"
            )
        parsed = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, dict) for k, v in parsed.items()
    ):
        raise ValueError(
            "PRIME_ORB_TYPES must map orb type names to JSON objects of OrbSpec fields"
        )
    return parsed
