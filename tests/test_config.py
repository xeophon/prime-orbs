"""Unit tests for DispatchJob, OrbSpec, and ProviderConfig validation and precedence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prime_orbs.config import (
    DEFAULT_AMP_ORB_IMAGE_ARM64,
    DEFAULT_AMP_ORB_IMAGE_X64,
    DispatchJob,
    OrbSpec,
    ProviderConfig,
)


class TestDispatchJob:
    def test_spawn_requires_orb_token_and_normalizes_blank_optionals(self) -> None:
        with pytest.raises(ValueError, match="ORB_TOKEN"):
            DispatchJob.from_env({"ORB_OPERATION": "spawn", "ORB_ID": "amp-1"})

        job = DispatchJob.from_env(
            {
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-T-123",
                "ORB_TOKEN": "tok_secret",
                "ORB_SNAPSHOT_ID": "   ",
                "ORB_TYPE": "",
                "AMP_URL": "",
            }
        )
        assert job.operation == "spawn"
        assert job.orb_id == "amp-T-123"
        assert job.orb_token == "tok_secret"
        assert job.orb_snapshot_id is None
        assert job.orb_type is None
        assert job.amp_url is None

    @pytest.mark.parametrize("op", ["pause", "delete"])
    def test_pause_and_delete_do_not_require_orb_token(self, op: str) -> None:
        job = DispatchJob.from_env({"ORB_OPERATION": op, "ORB_ID": "amp-T-123"})
        assert job.operation == op
        assert job.orb_token is None

    @pytest.mark.parametrize("bad_op", ["", "start", "stop", "SPAWN"])
    def test_rejects_unknown_operation(self, bad_op: str) -> None:
        with pytest.raises(ValueError, match="ORB_OPERATION"):
            DispatchJob.from_env({"ORB_OPERATION": bad_op, "ORB_ID": "amp-1"})


class TestOrbSpecValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"cpu_cores": 0}, "cpu_cores"),
            ({"cpu_cores": 17}, "cpu_cores"),
            ({"memory_gb": 0.05}, "memory_gb"),
            ({"memory_gb": 65}, "memory_gb"),
            ({"disk_size_gb": 1}, "disk_size_gb"),
            ({"disk_size_gb": 129}, "disk_size_gb"),
            ({"gpu_count": 1, "gpu_type": None}, "gpu_type is required"),
            ({"gpu_count": 0, "gpu_type": "RTX_PRO_6000"}, "gpu_type cannot be set"),
            ({"timeout_minutes": 0}, "timeout_minutes"),
            ({"idle_timeout_minutes": 0}, "idle_timeout_minutes"),
            ({"idle_timeout_minutes": 1441}, "idle_timeout_minutes"),
            ({"timeout_minutes": 30, "idle_timeout_minutes": 45}, "cannot exceed"),
            (
                {"network_allowlist": ["ampcode.com"], "network_denylist": ["1.1.1.1"]},
                "mutually exclusive",
            ),
        ],
    )
    def test_prime_resource_bounds(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            OrbSpec(image="prime/alice/amp-orb:latest", **kwargs)

    def test_boundary_values_accepted(self) -> None:
        min_spec = OrbSpec(
            image="prime/alice/amp-orb:latest",
            cpu_cores=1,
            memory_gb=0.125,
            disk_size_gb=2,
            timeout_minutes=-1,
            idle_timeout_minutes=1440,
        )
        max_spec = OrbSpec(
            image="prime/alice/amp-orb:latest",
            cpu_cores=16,
            memory_gb=64,
            disk_size_gb=128,
            gpu_count=8,
            gpu_type="H200_141GB",
            timeout_minutes=60,
            idle_timeout_minutes=60,
        )
        assert min_spec.timeout_minutes == -1
        assert max_spec.gpu_count == 8


class TestProviderConfigPrecedence:
    def test_defaults_disable_lifetime_cap_and_infer_arm64(self, tmp_path: Path) -> None:
        cfg_x64 = ProviderConfig.from_env({"PRIME_ORB_STATE_DIR": str(tmp_path)})
        assert cfg_x64.default_spec.image == DEFAULT_AMP_ORB_IMAGE_X64
        assert cfg_x64.default_spec.arch is None
        assert cfg_x64.default_spec.timeout_minutes == -1

        job_arm = DispatchJob.from_env(
            {
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-arm",
                "ORB_TOKEN": "tok",
                "ORB_IMAGE": DEFAULT_AMP_ORB_IMAGE_ARM64,
            }
        )
        resolved = cfg_x64.resolve_spec(job_arm)
        assert resolved.image == DEFAULT_AMP_ORB_IMAGE_ARM64
        assert resolved.arch == "arm64"

    def test_prime_orb_image_and_orb_type_file_override(self, tmp_path: Path) -> None:
        types_file = tmp_path / "orb-types.json"
        types_file.write_text(
            json.dumps(
                {
                    "gpu-dev": {
                        "cpu_cores": 8,
                        "memory_gb": 32,
                        "disk_size_gb": 60,
                        "gpu_count": 1,
                        "gpu_type": "RTX_PRO_6000",
                        "image": "prime/acme/amp-orb-cuda:v1",
                    }
                }
            ),
            encoding="utf-8",
        )
        env = {
            "PRIME_ORB_IMAGE": "prime/acme/amp-orb:v2",
            "ORB_IMAGE": DEFAULT_AMP_ORB_IMAGE_X64,
            "PRIME_ORB_TYPES": str(types_file),
            "PRIME_ORB_STATE_DIR": str(tmp_path),
        }
        cfg = ProviderConfig.from_env(env)

        default_job = DispatchJob.from_env(
            {**env, "ORB_OPERATION": "spawn", "ORB_ID": "amp-1", "ORB_TOKEN": "tok"}
        )
        assert cfg.resolve_spec(default_job).image == "prime/acme/amp-orb:v2"

        gpu_job = DispatchJob.from_env(
            {
                **env,
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-2",
                "ORB_TOKEN": "tok",
                "ORB_TYPE": "gpu-dev",
            }
        )
        gpu_spec = cfg.resolve_spec(gpu_job)
        assert gpu_spec.image == "prime/acme/amp-orb-cuda:v1"
        assert gpu_spec.gpu_count == 1
        assert gpu_spec.gpu_type == "RTX_PRO_6000"

        unknown_job = DispatchJob.from_env(
            {
                **env,
                "ORB_OPERATION": "spawn",
                "ORB_ID": "amp-3",
                "ORB_TOKEN": "tok",
                "ORB_TYPE": "nonexistent",
            }
        )
        with pytest.raises(ValueError, match="Unknown ORB_TYPE='nonexistent'"):
            cfg.resolve_spec(unknown_job)
