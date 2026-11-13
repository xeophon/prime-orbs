"""Prime Images resolution and cloud Dockerfile builds for Amp orb images."""

from __future__ import annotations

import io
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from prime_sandboxes import (
    BuildImageRequest,
    BuildImageResponse,
    ImageBuildStatus,
    ImageClient,
    ImageListItem,
    ImageVisibility,
    RegistryAuth,
)
from prime_sandboxes.image_references import is_docker_hub_reference

from .config import DEFAULT_AMP_ORB_IMAGE_ARM64, DEFAULT_AMP_ORB_IMAGE_X64, OrbArch

_IN_PROGRESS_BUILD_STATUSES = {
    ImageBuildStatus.PENDING,
    ImageBuildStatus.UPLOADING,
    ImageBuildStatus.BUILDING,
}
_TERMINAL_BUILD_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}


class ImageResolutionError(RuntimeError):
    """Raised when an external registry image has not been built in Prime Images yet."""


def is_directly_bootable_image(reference: str) -> bool:
    """Return True when Prime Sandboxes accepts `reference` directly in CreateSandboxRequest."""
    cleaned = reference.strip()
    if not cleaned:
        return False
    if cleaned.startswith("prime/"):
        return True
    return is_docker_hub_reference(cleaned)


def parse_name_and_tag(reference: str) -> tuple[str, str]:
    """Extract `(image_name, image_tag)` from any OCI or Prime image reference."""
    cleaned = reference.strip()
    if not cleaned:
        raise ValueError("Image reference cannot be empty")
    without_digest = cleaned.split("@", 1)[0]
    last_segment = without_digest.rsplit("/", 1)[-1]
    if ":" in last_segment:
        name, tag = last_segment.rsplit(":", 1)
    else:
        name, tag = last_segment, "latest"
    if not name or not tag:
        raise ValueError(f"Invalid image reference: {reference!r}")
    return name, tag


def format_prime_image_ref(item: ImageListItem) -> str:
    """Return the canonical `prime/<owner>/<name>:<tag>` reference for a built Prime image."""
    for candidate in (item.display_ref, item.full_image_path):
        if candidate and candidate.startswith("prime/"):
            return candidate
    return item.display_ref or item.full_image_path or f"{item.image_name}:{item.image_tag}"


def _row_timestamp(row: ImageListItem) -> datetime:
    ts = row.pushed_at or row.completed_at or row.started_at or row.created_at
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def find_latest_image_row(
    image_client: ImageClient,
    image_name: str,
    image_tag: str,
    *,
    team_id: str | None = None,
) -> ImageListItem | None:
    """Find the most recent Prime Image artifact row matching `image_name:image_tag`."""
    listed = image_client.list(search=image_name, team_id=team_id, limit=100)
    matches = [
        row
        for row in listed.data
        if row.image_name == image_name and row.image_tag == image_tag
    ]
    if not matches:
        return None
    return max(matches, key=_row_timestamp)


def resolve_prime_image(
    requested_image: str,
    image_client: ImageClient,
    *,
    deadline: float | None = None,
    team_id: str | None = None,
    poll_interval_seconds: float = 5.0,
) -> str:
    """Resolve `requested_image` to a reference accepted by Prime `CreateSandboxRequest`.

    - `prime/<owner>/<name>:<tag>` and Docker Hub references pass through unchanged.
    - External registry references (such as Amp's default `us-central1-docker.pkg.dev/...`
      orb image) automatically resolve to a matching `<name>:<tag>` built in Prime Images.
    """
    if is_directly_bootable_image(requested_image):
        return requested_image.strip()

    image_name, image_tag = parse_name_and_tag(requested_image)

    while True:
        latest = find_latest_image_row(
            image_client, image_name, image_tag, team_id=team_id
        )
        if latest is None:
            break
        if latest.status == ImageBuildStatus.COMPLETED:
            resolved = format_prime_image_ref(latest)
            if resolved.startswith("prime/"):
                return resolved
            break
        if latest.status in _IN_PROGRESS_BUILD_STATUSES and deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining > 1.0:
                time.sleep(min(poll_interval_seconds, remaining))
                continue
            raise ImageResolutionError(
                f"Prime Image '{image_name}:{image_tag}' is still building "
                f"(status={latest.status.value}). Wait for it to reach Ready in "
                "`prime images list` and retry."
            )
        if latest.status == ImageBuildStatus.FAILED:
            detail = f": {latest.error_message}" if latest.error_message else ""
            raise ImageResolutionError(
                f"Prime Image build for '{image_name}:{image_tag}' failed{detail}. "
                f"Rebuild it with: uv run prime-orbs build-image --base-image {requested_image}"
            )
        break

    arch_flag = " --arch arm64" if "arm64" in requested_image.lower() else ""
    raise ImageResolutionError(
        f"Prime Sandboxes cannot boot external registry image {requested_image!r} directly. "
        f"Build it once in Prime Images first:\n"
        f"  uv run prime-orbs build-image --base-image {requested_image}"
        f" --name {image_name}:{image_tag}{arch_flag}\n"
        f"Once built, either keep default ORB_IMAGE (auto-resolved by name) or set "
        f"PRIME_ORB_IMAGE=prime/<owner>/{image_name}:{image_tag}."
    )


def package_dockerfile_tarball(
    *,
    base_image: str | None = None,
    dockerfile_path: Path | None = None,
    context_dir: Path | None = None,
) -> bytes:
    """Create a gzipped tar archive suitable for Prime's `/images/build` upload URL."""
    if dockerfile_path is not None:
        dockerfile_bytes = dockerfile_path.read_bytes()
    elif base_image is not None:
        dockerfile_bytes = f"FROM {base_image.strip()}\n".encode()
    else:
        raise ValueError("Either base_image or dockerfile_path must be provided")

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if context_dir is not None:
            for entry in sorted(context_dir.iterdir()):
                if entry.name == "Dockerfile" and dockerfile_path is not None:
                    continue
                archive.add(entry, arcname=entry.name)
        info = tarfile.TarInfo(name="Dockerfile")
        info.size = len(dockerfile_bytes)
        info.mtime = int(time.time())
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(dockerfile_bytes))
    return buffer.getvalue()


def default_base_image_for_arch(arch: OrbArch) -> str:
    return DEFAULT_AMP_ORB_IMAGE_ARM64 if arch == "arm64" else DEFAULT_AMP_ORB_IMAGE_X64


def build_orb_image(
    image_client: ImageClient,
    *,
    destination: str = "amp-orb:latest",
    base_image: str | None = None,
    dockerfile_path: Path | None = None,
    context_dir: Path | None = None,
    arch: OrbArch = "amd64",
    visibility: ImageVisibility | None = None,
    team_id: str | None = None,
    registry_auth: RegistryAuth | None = None,
    wait: bool = True,
    timeout_seconds: float = 1800.0,
    poll_interval_seconds: float = 10.0,
) -> tuple[str, str]:
    """Build an Amp orb VM image in Prime Images and return `(build_id, prime_image_ref)`."""
    image_name, image_tag = parse_name_and_tag(destination)
    if "/" in image_name:
        raise ValueError(
            f"Destination image name {image_name!r} must not contain '/'. "
            "Pass a simple name:tag such as 'amp-orb:latest'."
        )

    effective_base = base_image or default_base_image_for_arch(arch)
    tarball = package_dockerfile_tarball(
        base_image=None if dockerfile_path is not None else effective_base,
        dockerfile_path=dockerfile_path,
        context_dir=context_dir,
    )

    platform: Literal["linux/amd64", "linux/arm64"] = (
        "linux/arm64" if arch == "arm64" else "linux/amd64"
    )
    resolved_team_id = team_id if team_id is not None else image_client.client.config.team_id
    request = BuildImageRequest(
        image_name=image_name,
        image_tag=image_tag,
        dockerfile_path="Dockerfile",
        platform=platform,
        team_id=resolved_team_id,
        visibility=visibility,
        registry_auth=registry_auth,
    )
    initiated = image_client.initiate_build(request)
    if not isinstance(initiated, BuildImageResponse) or not initiated.upload_url:
        raise RuntimeError(f"Unexpected response from Prime image build initiation: {initiated}")

    upload_resp = httpx.put(
        initiated.upload_url,
        content=tarball,
        headers={"Content-Type": "application/octet-stream"},
        timeout=600.0,
    )
    upload_resp.raise_for_status()

    image_client.start_build(initiated.build_id, registry_auth=registry_auth)

    if not wait:
        return initiated.build_id, initiated.full_image_path

    deadline = time.monotonic() + timeout_seconds
    last_status: dict | None = None
    while time.monotonic() < deadline:
        last_status = image_client.get_build_status(initiated.build_id)
        state = str(last_status.get("status") or "").upper()
        if state in _TERMINAL_BUILD_STATUSES:
            if state != "COMPLETED":
                error_detail = (
                    last_status.get("errorMessage")
                    or last_status.get("error_message")
                    or last_status.get("error")
                    or last_status
                )
                raise RuntimeError(
                    f"Prime image build {initiated.build_id} ended with {state}: {error_detail}"
                )
            row = find_latest_image_row(
                image_client, image_name, image_tag, team_id=resolved_team_id
            )
            resolved_ref = format_prime_image_ref(row) if row else initiated.full_image_path
            return initiated.build_id, resolved_ref
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval_seconds, remaining))

    raise TimeoutError(
        f"Prime image build {initiated.build_id} did not finish within {timeout_seconds:g}s "
        f"(last status: {last_status})"
    )
