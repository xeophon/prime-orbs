"""CLI utilities for building Prime orb images, warming templates, and running the provider."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
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
from .daemon import DAEMON_SOCKET_ENV, DispatchDaemon, daemon_socket_path
from .dispatcher import (
    ORB_MANAGED_LABEL,
    TERMINAL_STATUSES,
    build_warm_template,
    expire_stale_project_snapshots,
    run_dispatch,
    warm_template_key,
)
from .images import (
    ImageResolutionError,
    build_orb_image,
    default_base_image_for_arch,
    format_prime_image_ref,
    resolve_prime_image,
)
from .standby import ORB_STANDBY_LABEL
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
        update_amp_cli=not args.no_update_amp,
        wait=not args.no_wait,
        timeout_seconds=args.timeout,
    )
    print(f"Build ID: {build_id}")
    print(f"Prime image reference: {image_ref}")
    if args.no_wait:
        print("Build is running in Prime Images. Check progress with `uv run prime-orbs status`.")
        return 0

    if args.warm_template:
        config = ProviderConfig.from_env({**os.environ, "ORB_IMAGE": image_ref})
        sandbox_client = SandboxClient(api_client)
        _, ckpt_id = build_warm_template(
            config,
            sandbox_client=sandbox_client,
            image_client=image_client,
            update_amp_cli=False,
        )
        print(f"Warm memory checkpoint ready: {ckpt_id}")

    print(f"\nReady! Start the provider with `uv run prime-orbs run` (image={image_ref}).")
    return 0


def cmd_warm_template(args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    sandbox_client = SandboxClient(api_client)
    image_client = ImageClient(api_client)
    config = ProviderConfig.from_env()

    key, ckpt_id = build_warm_template(
        config,
        sandbox_client=sandbox_client,
        image_client=image_client,
        orb_type=args.orb_type,
        update_amp_cli=not args.no_update_amp,
        timeout_seconds=args.timeout,
    )
    print(f"Warm memory template ready:\n  key: {key}\n  checkpoint: {ckpt_id}")
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

    warm_templates = store.all_warm_templates()
    print(f"\nWarm Memory-Fork Templates ({len(warm_templates)}):")
    if not warm_templates:
        print("  (none — run `uv run prime-orbs warm-template` to skip cold VM kernel boot)")
    else:
        for key, ckpt_id in sorted(warm_templates.items()):
            print(f"  - {key} -> {ckpt_id}")

    claimed_ids = set(store.all_orb_sandboxes().values())
    standby_resp = sandbox_client.list(labels=[ORB_STANDBY_LABEL], per_page=100)
    standby_live = [
        sbx
        for sbx in standby_resp.sandboxes
        if sbx.id not in claimed_ids and sbx.status not in TERMINAL_STATUSES
    ]
    print(f"\nPre-Placed Standby Sandboxes ({len(standby_live)}):")
    if not standby_live:
        print("  (none)")
    else:
        for sbx in standby_live:
            print(f"  - {sbx.name} ({sbx.id}): {sbx.status} | image={sbx.docker_image}")

    sandboxes_resp = sandbox_client.list(labels=[ORB_MANAGED_LABEL], per_page=100)
    live_by_id = {
        sbx.id: sbx for sbx in sandboxes_resp.sandboxes if sbx.status not in TERMINAL_STATUSES
    }
    for orb_id, sbx_id in store.all_orb_sandboxes().items():
        if sbx_id not in live_by_id:
            try:
                sbx = sandbox_client.get(sbx_id)
                if sbx.status not in TERMINAL_STATUSES:
                    live_by_id[sbx_id] = sbx
            except APIError:
                pass
        del orb_id

    live_sandboxes = list(live_by_id.values())
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
    modes = store.all_project_snapshot_modes()
    print(f"\nCached Project Snapshots ({len(snapshots)}) in {store.path}:")
    for snap_id, ckpt_id in sorted(snapshots.items()):
        mode = modes.get(snap_id, "disk")
        print(f"  - {snap_id} -> {ckpt_id} [{mode}]")

    return 0


def cmd_gc_snapshots(_args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    sandbox_client = SandboxClient(api_client)
    config = ProviderConfig.from_env()
    store = SnapshotStore(config.state_dir)

    removed = expire_stale_project_snapshots(
        sandbox_client,
        store,
        ttl_seconds=config.snapshot_ttl_seconds,
    )
    for snap_id, ckpt_id in store.all_project_snapshots().items():
        try:
            ckpt = sandbox_client.get_checkpoint(ckpt_id)
            if ckpt.state != "DURABLE" or ckpt.merge == "failed":
                store.remove_project_snapshot(snap_id)
                removed += 1
        except APIError:
            store.remove_project_snapshot(snap_id)
            removed += 1

    for tmpl_key, ckpt_id in store.all_warm_templates().items():
        try:
            ckpt = sandbox_client.get_checkpoint(ckpt_id)
            if ckpt.state != "DURABLE":
                store.remove_warm_template(tmpl_key)
                removed += 1
        except APIError:
            store.remove_warm_template(tmpl_key)
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

    for orb_id, sbx_id in store.all_orb_sandboxes().items():
        try:
            sbx = sandbox_client.get(sbx_id)
            if sbx.status in TERMINAL_STATUSES:
                store.remove_orb_sandbox(orb_id)
                removed += 1
        except APIError:
            store.remove_orb_sandbox(orb_id)
            removed += 1

    print(f"Pruned {removed} stale state entry/entries from {store.path}.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    api_client = _require_api_client()
    sandbox_client = SandboxClient(api_client)
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

    if args.warm_template or config.auto_warm_template:
        store = SnapshotStore(config.state_dir)
        key = warm_template_key(resolved_image, config.default_spec)
        existing_ckpt = store.get_warm_template(key)
        needs_build = True
        if existing_ckpt:
            try:
                needs_build = sandbox_client.get_checkpoint(existing_ckpt).state != "DURABLE"
            except APIError:
                needs_build = True
        if needs_build and config.default_spec.gpu_count == 0:
            env_with_img = {**os.environ, "ORB_IMAGE": resolved_image}
            build_warm_template(
                ProviderConfig.from_env(env_with_img),
                sandbox_client=sandbox_client,
                image_client=image_client,
            )

    env = dict(os.environ)
    env.setdefault("ORB_IMAGE", resolved_image)
    if args.standby_count is not None:
        env["PRIME_ORB_STANDBY_COUNT"] = str(args.standby_count)
    if args.standby_hot_seconds is not None:
        env["PRIME_ORB_STANDBY_HOT_SECONDS"] = str(args.standby_hot_seconds)

    cmd = [amp_bin, "orb", "provider"]
    if args.token:
        cmd.extend(["--token", args.token])
    if args.allow_multiple:
        cmd.append("--allow-multiple")
    if args.amp_url:
        cmd.extend(["--amp-url", args.amp_url])
    cmd.append(str(dispatch_script))

    if args.no_daemon:
        print(f"Starting Amp orb provider with Prime image: {resolved_image}", flush=True)
        os.execvpe(amp_bin, cmd, env)
        return 0

    sock_path = daemon_socket_path(config.state_dir)
    env[DAEMON_SOCKET_ENV] = str(sock_path)
    with DispatchDaemon(
        sock_path,
        base_env=env,
        sandbox_client=sandbox_client,
        image_client=image_client,
    ):
        print(
            f"Starting Amp orb provider with warm dispatch daemon "
            f"(image={resolved_image}, socket={sock_path})",
            flush=True,
        )
        proc = subprocess.Popen(cmd, env=env)

        def _forward_signal(signum: int, _frame: object) -> None:
            if proc.poll() is None:
                proc.send_signal(signum)

        prev_int = signal.signal(signal.SIGINT, _forward_signal)
        prev_term = signal.signal(signal.SIGTERM, _forward_signal)
        try:
            return proc.wait()
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()


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
        "--no-update-amp",
        action="store_true",
        help="Skip baking the latest `amp` CLI binary into the built image",
    )
    sub_build.add_argument(
        "--warm-template",
        action="store_true",
        help="Also build a warm memory-fork checkpoint right after the image finishes",
    )
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

    sub_warm = sub.add_parser(
        "warm-template",
        help="Create/refresh a paused memory checkpoint so new orbs skip cold VM kernel boot",
    )
    sub_warm.add_argument(
        "--orb-type",
        default=None,
        help="Optional ORB_TYPE name to warm (default: provider default spec)",
    )
    sub_warm.add_argument(
        "--no-update-amp",
        action="store_true",
        help="Skip running `amp update` inside the template VM before freezing",
    )
    sub_warm.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Seconds allowed for booting, prewarming, and checkpointing (default: 600)",
    )
    sub_warm.set_defaults(func=cmd_warm_template)

    sub_status = sub.add_parser(
        "status",
        help="Show built Prime orb images, warm templates, active/paused orbs, and snapshots",
    )
    sub_status.set_defaults(func=cmd_status)

    sub_gc = sub.add_parser(
        "gc-snapshots",
        help="Remove expired or deleted checkpoint/sandbox mappings from the local state store",
    )
    sub_gc.set_defaults(func=cmd_gc_snapshots)

    sub_run = sub.add_parser(
        "run",
        help="Run `amp orb provider ./dispatch.py` backed by a warm connection-pooled daemon",
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
    sub_run.add_argument(
        "--warm-template",
        action="store_true",
        help="Ensure a warm memory-fork checkpoint exists on startup before accepting jobs",
    )
    sub_run.add_argument(
        "--standby-count",
        type=int,
        default=None,
        help="Number of pre-placed standby VMs to maintain (default: PRIME_ORB_STANDBY_COUNT or 1)",
    )
    sub_run.add_argument(
        "--standby-hot-seconds",
        type=float,
        default=None,
        help="Seconds standby VMs stay RUNNING after activity before auto-pausing (default: 900)",
    )
    sub_run.add_argument(
        "--no-daemon",
        action="store_true",
        help="Exec `amp orb provider` directly without the in-memory UNIX-socket daemon",
    )
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
