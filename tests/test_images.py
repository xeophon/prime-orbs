"""Tests for Prime Images resolution and cloud Dockerfile tarball builds."""

from __future__ import annotations

import io
import tarfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from prime_sandboxes import ImageClient, ImageListResponse

from prime_orbs.config import DEFAULT_AMP_ORB_IMAGE_ARM64, DEFAULT_AMP_ORB_IMAGE_X64
from prime_orbs.images import (
    ImageResolutionError,
    build_orb_image,
    is_directly_bootable_image,
    parse_name_and_tag,
    resolve_prime_image,
)


class FakeImageClient:
    def __init__(self, rows: list[dict[str, Any]], *, team_id: str | None = None) -> None:
        self._rows = rows
        self.list_calls: list[dict[str, Any]] = []
        self.initiated_requests: list[Any] = []
        self.started_builds: list[str] = []
        self.status_sequence: list[dict[str, Any]] = [{"status": "COMPLETED"}]
        self.client = SimpleNamespace(config=SimpleNamespace(team_id=team_id))

    def list(self, **kwargs: Any) -> ImageListResponse:
        self.list_calls.append(kwargs)
        return ImageListResponse.model_validate(
            {"data": self._rows, "totalCount": len(self._rows), "offset": 0, "limit": 100}
        )

    def initiate_build(self, request: Any) -> Any:
        self.initiated_requests.append(request)
        from prime_sandboxes import BuildImageResponse

        return BuildImageResponse.model_validate(
            {
                "build_id": "bld-42",
                "buildIds": ["bld-42"],
                "upload_url": "https://upload.example.test/context.tar.gz",
                "expires_in": 3600,
                "fullImagePath": f"{request.image_name}:{request.image_tag}",
            },
            context={"requires_upload": True},
        )

    def start_build(self, build_id: str, registry_auth: Any = None) -> dict[str, Any]:
        del registry_auth
        self.started_builds.append(build_id)
        return {"status": "BUILDING"}

    def get_build_status(self, _build_id: str) -> dict[str, Any]:
        if len(self.status_sequence) > 1:
            return self.status_sequence.pop(0)
        return self.status_sequence[0]


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("prime/xeophon/amp-orb:latest", True),
        ("prime/team-123/amp-orb:v1", True),
        ("ubuntu:24.04", True),
        ("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime", True),
        (DEFAULT_AMP_ORB_IMAGE_X64, False),
        (DEFAULT_AMP_ORB_IMAGE_ARM64, False),
        ("ghcr.io/acme/custom-orb:latest", False),
    ],
)
def test_is_directly_bootable_image(ref: str, expected: bool) -> None:
    assert is_directly_bootable_image(ref) is expected


def test_parse_name_and_tag_handles_registry_ports_and_digests() -> None:
    assert parse_name_and_tag(DEFAULT_AMP_ORB_IMAGE_X64) == ("amp-orb", "latest")
    assert parse_name_and_tag(DEFAULT_AMP_ORB_IMAGE_ARM64) == ("amp-orb-arm64", "latest")
    assert parse_name_and_tag("localhost:5000/org/my-orb@sha256:" + "a" * 64) == (
        "my-orb",
        "latest",
    )


def test_direct_prime_or_dockerhub_refs_skip_api_lookup() -> None:
    fake = FakeImageClient([])
    assert (
        resolve_prime_image("prime/xeophon/amp-orb:latest", fake)  # type: ignore[arg-type]
        == "prime/xeophon/amp-orb:latest"
    )
    assert resolve_prime_image("ubuntu:24.04", fake) == "ubuntu:24.04"  # type: ignore[arg-type]
    assert fake.list_calls == []


def test_resolves_latest_completed_row_over_older_failed_row() -> None:
    rows = [
        {
            "id": "old-failed",
            "artifactType": "VM_SANDBOX",
            "imageName": "amp-orb",
            "imageTag": "latest",
            "status": "FAILED",
            "errorMessage": "transient error",
            "visibility": "PRIVATE",
            "createdAt": "2026-11-10T01:00:00Z",
            "displayRef": "prime/xeophon/amp-orb:latest",
        },
        {
            "id": "new-ready",
            "artifactType": "VM_SANDBOX",
            "imageName": "amp-orb",
            "imageTag": "latest",
            "status": "COMPLETED",
            "visibility": "PRIVATE",
            "createdAt": "2026-11-11T01:00:00Z",
            "pushedAt": "2026-11-11T01:05:00Z",
            "fullImagePath": "amp-orb:latest",
            "displayRef": "prime/xeophon/amp-orb:latest",
        },
        {
            "id": "unrelated-other-tag",
            "artifactType": "VM_SANDBOX",
            "imageName": "amp-orb",
            "imageTag": "canary",
            "status": "COMPLETED",
            "visibility": "PRIVATE",
            "createdAt": "2026-11-11T02:00:00Z",
            "displayRef": "prime/xeophon/amp-orb:canary",
        },
    ]
    fake = FakeImageClient(rows)
    resolved = resolve_prime_image(DEFAULT_AMP_ORB_IMAGE_X64, fake)  # type: ignore[arg-type]
    assert resolved == "prime/xeophon/amp-orb:latest"


def test_newer_failed_row_is_not_masked_by_older_completed_row() -> None:
    rows = [
        {
            "id": "old-ready",
            "artifactType": "VM_SANDBOX",
            "imageName": "amp-orb",
            "imageTag": "latest",
            "status": "COMPLETED",
            "visibility": "PRIVATE",
            "createdAt": "2026-11-10T01:00:00Z",
            "displayRef": "prime/xeophon/amp-orb:latest",
        },
        {
            "id": "new-failed",
            "artifactType": "VM_SANDBOX",
            "imageName": "amp-orb",
            "imageTag": "latest",
            "status": "FAILED",
            "errorMessage": "layer unpack failed",
            "visibility": "PRIVATE",
            "createdAt": "2026-11-11T03:00:00Z",
            "displayRef": "prime/xeophon/amp-orb:latest",
        },
    ]
    fake = FakeImageClient(rows)
    with pytest.raises(ImageResolutionError, match="layer unpack failed"):
        resolve_prime_image(DEFAULT_AMP_ORB_IMAGE_X64, fake)  # type: ignore[arg-type]


def test_missing_external_image_gives_actionable_build_command() -> None:
    fake = FakeImageClient([])
    with pytest.raises(ImageResolutionError, match="prime-orbs build-image.*--arch arm64"):
        resolve_prime_image(DEFAULT_AMP_ORB_IMAGE_ARM64, fake)  # type: ignore[arg-type]


def test_build_orb_image_packages_dockerfile_and_returns_prime_display_ref() -> None:
    rows = [
        {
            "id": "built-1",
            "artifactType": "VM_SANDBOX",
            "imageName": "amp-orb",
            "imageTag": "latest",
            "status": "COMPLETED",
            "visibility": "PRIVATE",
            "createdAt": "2026-11-11T04:00:00Z",
            "fullImagePath": "amp-orb:latest",
            "displayRef": "prime/xeophon/amp-orb:latest",
        }
    ]
    fake = FakeImageClient(rows, team_id="team-99")
    fake.status_sequence = [{"status": "BUILDING"}, {"status": "COMPLETED"}]

    uploaded_archives: list[bytes] = []

    def fake_put(url: str, *, content: bytes, headers: dict[str, str], timeout: float) -> Any:
        assert url == "https://upload.example.test/context.tar.gz"
        assert headers["Content-Type"] == "application/octet-stream"
        assert timeout > 0
        uploaded_archives.append(content)
        return httpx.Response(200, request=httpx.Request("PUT", url))

    with patch("prime_orbs.images.httpx.put", side_effect=fake_put):
        build_id, ref = build_orb_image(
            fake,  # type: ignore[arg-type]
            destination="amp-orb:latest",
            arch="amd64",
            poll_interval_seconds=0.01,
        )

    assert build_id == "bld-42"
    assert ref == "prime/xeophon/amp-orb:latest"
    assert fake.started_builds == ["bld-42"]
    assert fake.initiated_requests[0].team_id == "team-99"
    assert fake.initiated_requests[0].platform == "linux/amd64"

    assert len(uploaded_archives) == 1
    with tarfile.open(fileobj=io.BytesIO(uploaded_archives[0]), mode="r:gz") as tar:
        dockerfile_member = tar.extractfile("Dockerfile")
        assert dockerfile_member is not None
        assert dockerfile_member.read().decode() == (
            f"FROM {DEFAULT_AMP_ORB_IMAGE_X64}\n"
            "USER user\n"
            "RUN curl -fsSL https://ampcode.com/install.sh | bash\n"
        )

    assert isinstance(ImageClient, type)
