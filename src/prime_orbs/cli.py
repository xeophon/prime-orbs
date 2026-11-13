"""CLI utilities for building Prime orb images, inspecting orbs, and running the provider."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from prime_sandboxes import (
    APIClient,
    APIError,
    ImageClient,
    ImageVisibility,
    RegistryAuth,
    SandboxClient,
    UnauthorizedError,
)

from .config import DispatchJob, OrbArch, ProviderConfig
from .dispatcher import ORB_MANAGED_LABEL, TERMINAL_STATUSES, run_dispatch
from .images import (
    ImageResolutionError,
    build_orb_image,
    default_base_image_for_arch,
    format_prime_image_ref,
    resolve_prime_image,
)
from .state import SnapshotStore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_api_client() -> APIClient:
    client = APIClient()
    if not client.config.api_key:
        raise UnauthorizedError(
            "Missing Prime API key. Export PRIME_API_KEY or run `prime login` first."
        )
    return client


def cmd_dispatch(_args: argparse.Namespace) -> int:
    job = DispatchJob.from_env()
    config = ProviderConfig.from_env()
    run_dispatch(job, config)
    return 0


def cmd_build_image(args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    image_client = ImageClient(api_client)

    arch: OrbArch = "arm64" if args.arch in ("arm64", "arm") else "amd64"
    visibility: ImageVisibility | None = None
    if args.public:
        visibility = ImageVisibility.PUBLIC
    elif args.private:
        visibility = ImageVisibility.PRIVATE

    registry_auth: RegistryAuth | None = None
    if args.registry_server or args.registry_username:
        if not (args.registry_server and args.registry_username):
            print(
                "Error: --registry-server and --registry-username must be provided together.",
                file=sys.stderr,
            )
            return 2
        password = os.environ.get("PRIME_REGISTRY_PASSWORD")
        if not password:
            print(
                "Error: set PRIME_REGISTRY_PASSWORD when using --registry-server.",
                file=sys.stderr,
            )
            return 2
        registry_auth = RegistryAuth(
            server=args.registry_server,
            username=args.registry_username,
            password=password,
        )

    dockerfile_path = Path(args.dockerfile).resolve() if args.dockerfile else None
    context_dir = Path(args.context).resolve() if args.context else None
    base_image = args.base_image or default_base_image_for_arch(arch)
    destination = args.name or ("amp-orb-arm64:latest" if arch == "arm64" else "amp-orb:latest")

    source_desc = str(dockerfile_path) if dockerfile_path else base_image
    print(f"Building Prime VM image '{destination}' ({arch}) from {source_desc} ...")
    build_id, image_ref = build_orb_image(
        image_client,
        destination=destination,
        base_image=None if dockerfile_path else base_image,
        dockerfile_path=dockerfile_path,
        context_dir=context_dir,
        arch=arch,
        visibility=visibility,
        team_id=args.team_id,
        registry_auth=registry_auth,
        wait=not args.no_wait,
        timeout_seconds=args.timeout,
    )
    print(f"Build ID: {build_id}")
    print(f"Prime image reference: {image_ref}")
    if args.no_wait:
        print("Build is running in Prime Images. Check progress with `uv run prime-orbs status`.")
    else:
        print(f"\nReady! You can start the provider now (or set PRIME_ORB_IMAGE={image_ref}).")
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    sandbox_client = SandboxClient(api_client)
    image_client = ImageClient(api_client)
    config = ProviderConfig.from_env()
    store = SnapshotStore(config.state_dir)

    team_display = api_client.config.team_id or "personal"
    print(f"Scope: {team_display} ({api_client.config.base_url})")

    images = image_client.list(search="amp-orb", limit=50)
    print("\nPrime Images matching 'amp-orb':")
    if not images.data:
        print("  (none built yet — run `uv run prime-orbs build-image`)")
    else:
        for row in images.data:
            ref = format_prime_image_ref(row)
            print(f"  - {ref} [{row.status.value}] ({row.visibility.value})")

    sandboxes_resp = sandbox_client.list(labels=[ORB_MANAGED_LABEL], per_page=100)
    live_sandboxes = [
        sbx for sbx in sandboxes_resp.sandboxes if sbx.status not in TERMINAL_STATUSES
    ]
    print(f"\nActive/Paused Amp Orb Sandboxes ({len(live_sandboxes)}):")
    if not live_sandboxes:
        print("  (none)")
    else:
        for sbx in live_sandboxes:
            resources = (
                f"{sbx.cpu_cores:g} vCPU, {sbx.memory_gb:g} GiB RAM, "
                f"{sbx.disk_size_gb:g} GiB disk"
            )
            print(
                f"  - {sbx.name} ({sbx.id}): {sbx.status} | {resources} | "
                f"image={sbx.docker_image}"
            )

    snapshots = store.all_project_snapshots()
    print(f"\nCached Project Snapshots ({len(snapshots)}) in {store.path}:")
    for snap_id, ckpt_id in sorted(snapshots.items()):
        print(f"  - {snap_id} -> {ckpt_id}")

    return 0


def cmd_gc_snapshots(_args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    sandbox_client = SandboxClient(api_client)
    config = ProviderConfig.from_env()
    store = SnapshotStore(config.state_dir)

    removed = 0
    for snap_id, ckpt_id in store.all_project_snapshots().items():
        try:
            ckpt = sandbox_client.get_checkpoint(ckpt_id)
            if ckpt.state != "DURABLE" or ckpt.merge == "failed":
                store.remove_project_snapshot(snap_id)
                removed += 1
        except APIError:
            store.remove_project_snapshot(snap_id)
            removed += 1

    for orb_id, ckpt_id in store.all_orb_checkpoints().items():
        try:
            ckpt = sandbox_client.get_checkpoint(ckpt_id)
            if ckpt.state != "DURABLE":
                store.remove_orb_checkpoint(orb_id)
                removed += 1
        except APIError:
            store.remove_orb_checkpoint(orb_id)
            removed += 1

    print(f"Pruned {removed} stale snapshot entry/entries from {store.path}.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    image_client = ImageClient(api_client)
    config = ProviderConfig.from_env()

    resolved_image = resolve_prime_image(config.default_spec.image, image_client)
    amp_bin = shutil.which("amp")
    if not amp_bin:
        print("Error: `amp` CLI not found on PATH.", file=sys.stderr)
        return 1

    dispatch_script = Path(args.dispatch_script).resolve()
    if not dispatch_script.is_file():
        print(f"Error: dispatch script not found at {dispatch_script}", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env.setdefault("ORB_IMAGE", resolved_image)

    cmd = [amp_bin, "orb", "provider"]
    if args.token:
        cmd.extend(["--token", args.token])
    if args.allow_multiple:
        cmd.append("--allow-multiple")
    if args.amp_url:
        cmd.extend(["--amp-url", args.amp_url])
    cmd.append(str(dispatch_script))

    print(f"Starting Amp orb provider with Prime image: {resolved_image}", flush=True)
    os.execvpe(amp_bin, cmd, env)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prime-orbs",
        description="Manage and run Amp self-hosted orbs on Prime Intellect Sandboxes",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub_dispatch = sub.add_parser(
        "dispatch",
        help="Execute a single Amp orb lifecycle job from ORB_* environment variables",
    )
    sub_dispatch.set_defaults(func=cmd_dispatch)

    sub_build = sub.add_parser(
        "build-image",
        help="Build the Amp orb VM image in Prime Images (cloud build, no local Docker needed)",
    )
    sub_build.add_argument(
        "--name",
        default=None,
        help="Destination image name:tag in Prime Images (default: amp-orb:latest)",
    )
    sub_build.add_argument(
        "--base-image",
        default=None,
        help="Upstream OCI image for FROM (default: Amp's official orb image for --arch)",
    )
    sub_build.add_argument(
        "--dockerfile",
        default=None,
        help="Optional custom Dockerfile path instead of generated FROM <base-image>",
    )
    sub_build.add_argument(
        "--context",
        default=None,
        help="Optional build context directory when using --dockerfile",
    )
    sub_build.add_argument(
        "--arch",
        choices=("amd64", "arm64", "arm"),
        default="amd64",
        help="Target VM architecture (default: amd64)",
    )
    visibility_group = sub_build.add_mutually_exclusive_group()
    visibility_group.add_argument("--public", action="store_true", help="Publish image publicly")
    visibility_group.add_argument("--private", action="store_true", help="Keep image private")
    sub_build.add_argument("--team-id", default=None, help="Explicit Prime team ID")
    sub_build.add_argument("--registry-server", default=None, help="Private base registry host")
    sub_build.add_argument("--registry-username", default=None, help="Private base registry user")
    sub_build.add_argument(
        "--no-wait",
        action="store_true",
        help="Return immediately after starting the cloud image build",
    )
    sub_build.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for the cloud image build to finish (default: 1800)",
    )
    sub_build.set_defaults(func=cmd_build_image)

    sub_status = sub.add_parser(
        "status",
        help="Show built Prime orb images, active/paused orb sandboxes, and cached snapshots",
    )
    sub_status.set_defaults(func=cmd_status)

    sub_gc = sub.add_parser(
        "gc-snapshots",
        help="Remove expired or deleted checkpoint mappings from the local snapshot store",
    )
    sub_gc.set_defaults(func=cmd_gc_snapshots)

    sub_run = sub.add_parser(
        "run",
        help="Verify Prime credentials/image and run `amp orb provider ./dispatch.py`",
    )
    sub_run.add_argument(
        "--dispatch-script",
        default=str(_repo_root() / "dispatch.py"),
        help="Path to dispatch executable",
    )
    sub_run.add_argument(
        "--token",
        default=None,
        help="Orb provider token (defaults to AMP_ORB_PROVIDER_TOKEN or saved token)",
    )
    sub_run.add_argument(
        "--allow-multiple",
        action="store_true",
        help="Pass --allow-multiple to `amp orb provider`",
    )
    sub_run.add_argument("--amp-url", default=None, help="Custom Amp server URL")
    sub_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        ValueError,
        ImageResolutionError,
        UnauthorizedError,
        APIError,
        TimeoutError,
        RuntimeError,
    ) as exc:
        print(f"[prime-orbs] Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
