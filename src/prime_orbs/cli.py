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

from .config import DispatchJob, OrbArch, ProviderConfig, scope_label, sdk_team_param
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
from .systemd import (
    DEFAULT_REFRESH_ON_CALENDAR,
    activate_systemd_units,
    ensure_env_file_placeholder,
    render_systemd_units,
    write_systemd_units,
)
from .transport import create_tuned_clients


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require_clients() -> tuple[APIClient, SandboxClient, ImageClient]:
    return create_tuned_clients(require_api_key=True)


def _scoped_env_from_args(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if getattr(args, "personal", False):
        env["PRIME_ORB_TEAM_ID"] = ""
        env["PRIME_TEAM_ID"] = ""
    elif getattr(args, "team_id", None):
        tid = str(args.team_id).strip()
        env["PRIME_ORB_TEAM_ID"] = tid
        env["PRIME_TEAM_ID"] = tid
    return env


def _add_scope_args(subparser: argparse.ArgumentParser) -> None:
    group = subparser.add_mutually_exclusive_group()
    group.add_argument(
        "--team-id",
        default=None,
        help="Prime team ID (overrides PRIME_TEAM_ID and ~/.prime/config.json)",
    )
    group.add_argument(
        "--personal",
        action="store_true",
        help="Force personal Prime scope even if PRIME_TEAM_ID or ~/.prime/config.json sets a team",
    )


def cmd_dispatch(_args: argparse.Namespace) -> int:
    job = DispatchJob.from_env()
    config = ProviderConfig.from_env()
    run_dispatch(job, config)
    return 0


def cmd_build_image(args: argparse.Namespace) -> int:
    _, sandbox_client, image_client = _require_clients()

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

    scoped_env = _scoped_env_from_args(args)
    scoped_cfg = ProviderConfig.from_env(scoped_env)
    effective_team_id = scoped_cfg.default_spec.team_id

    dockerfile_path = Path(args.dockerfile).resolve() if args.dockerfile else None
    context_dir = Path(args.context).resolve() if args.context else None
    base_image = args.base_image or default_base_image_for_arch(arch)
    destination = args.name or ("amp-orb-arm64:latest" if arch == "arm64" else "amp-orb:latest")

    source_desc = str(dockerfile_path) if dockerfile_path else base_image
    print(
        f"Building Prime VM image '{destination}' ({arch}, "
        f"scope={scope_label(effective_team_id)}) from {source_desc} ..."
    )
    build_id, image_ref = build_orb_image(
        image_client,
        destination=destination,
        base_image=None if dockerfile_path else base_image,
        dockerfile_path=dockerfile_path,
        context_dir=context_dir,
        arch=arch,
        visibility=visibility,
        team_id=effective_team_id,
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
        config = ProviderConfig.from_env({**scoped_env, "ORB_IMAGE": image_ref})
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
    _, sandbox_client, image_client = _require_clients()
    config = ProviderConfig.from_env(_scoped_env_from_args(args))

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


def cmd_status(args: argparse.Namespace) -> int:
    api_client, sandbox_client, image_client = _require_clients()
    config = ProviderConfig.from_env(_scoped_env_from_args(args))
    store = SnapshotStore(config.state_dir)
    team_id = config.default_spec.team_id
    team_param = sdk_team_param(team_id)

    print(f"Scope: {scope_label(team_id)} ({api_client.config.base_url})")

    images = image_client.list(search="amp-orb", team_id=team_param, limit=50)
    print("\nPrime Images matching 'amp-orb':")
    if not images.data:
        team_hint = f" --team-id {team_id}" if team_id else ""
        print(f"  (none built yet — run `uv run prime-orbs build-image{team_hint}`)")
    else:
        for row in images.data:
            ref = format_prime_image_ref(row)
            print(f"  - {ref} [{row.status.value}] ({row.visibility.value})")

    expected_scope_suffix = f"|scope={scope_label(team_id)}"
    warm_templates = {
        k: v
        for k, v in store.all_warm_templates().items()
        if k.endswith(expected_scope_suffix)
        or (team_id is None and "|scope=" not in k)
    }
    print(f"\nWarm Memory-Fork Templates ({len(warm_templates)}):")
    if not warm_templates:
        print("  (none — run `uv run prime-orbs warm-template` to skip cold VM kernel boot)")
    else:
        for key, ckpt_id in sorted(warm_templates.items()):
            print(f"  - {key} -> {ckpt_id}")

    claimed_ids = set(store.all_orb_sandboxes().values())
    standby_resp = sandbox_client.list(
        team_id=team_param, labels=[ORB_STANDBY_LABEL], per_page=100
    )
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

    sandboxes_resp = sandbox_client.list(
        team_id=team_param, labels=[ORB_MANAGED_LABEL], per_page=100
    )
    live_by_id = {
        sbx.id: sbx for sbx in sandboxes_resp.sandboxes if sbx.status not in TERMINAL_STATUSES
    }
    for orb_id, sbx_id in store.all_orb_sandboxes().items():
        if store.get_orb_team(orb_id) != team_id:
            continue
        if sbx_id not in live_by_id:
            try:
                sbx = sandbox_client.get(sbx_id)
                if sbx.status not in TERMINAL_STATUSES:
                    live_by_id[sbx_id] = sbx
            except APIError:
                pass

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

    snapshots = store.all_project_snapshots(team_id=team_id)
    modes = store.all_project_snapshot_modes(team_id=team_id)
    print(f"\nCached Project Snapshots ({len(snapshots)}) in {store.path}:")
    for snap_id, ckpt_id in sorted(snapshots.items()):
        mode = modes.get(snap_id, "disk")
        print(f"  - {snap_id} -> {ckpt_id} [{mode}]")

    return 0


def cmd_gc_snapshots(args: argparse.Namespace) -> int:
    _, sandbox_client, _ = _require_clients()
    config = ProviderConfig.from_env(_scoped_env_from_args(args))
    store = SnapshotStore(config.state_dir)

    removed = expire_stale_project_snapshots(
        sandbox_client,
        store,
        ttl_seconds=config.snapshot_ttl_seconds,
    )
    for raw_key, ckpt_id in store.all_project_snapshots(all_scopes=True).items():
        try:
            ckpt = sandbox_client.get_checkpoint(ckpt_id)
            if ckpt.state != "DURABLE" or ckpt.merge == "failed":
                store.remove_raw_project_snapshot_key(raw_key)
                removed += 1
        except APIError:
            store.remove_raw_project_snapshot_key(raw_key)
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
    _, sandbox_client, image_client = _require_clients()
    scoped_env = _scoped_env_from_args(args)
    config = ProviderConfig.from_env(scoped_env)
    team_id = config.default_spec.team_id

    resolved_image = resolve_prime_image(
        config.default_spec.image, image_client, team_id=team_id
    )
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
            env_with_img = {**scoped_env, "ORB_IMAGE": resolved_image}
            build_warm_template(
                ProviderConfig.from_env(env_with_img),
                sandbox_client=sandbox_client,
                image_client=image_client,
            )

    env = dict(scoped_env)
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

    scope_desc = scope_label(team_id)
    if args.no_daemon:
        print(
            f"Starting Amp orb provider (scope={scope_desc}, image={resolved_image})",
            flush=True,
        )
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
            f"(scope={scope_desc}, image={resolved_image}, socket={sock_path})",
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


def cmd_install_service(args: argparse.Namespace) -> int:
    repo_root = _repo_root()
    env_file = Path(args.env_file).expanduser()
    unit_dir = Path(args.unit_dir).expanduser()

    units = render_systemd_units(
        repo_root=repo_root,
        env_file=env_file,
        unit_dir=unit_dir,
        service_name=args.service_name,
        standby_count=args.standby_count,
        team_id=args.team_id,
        force_personal=args.personal,
        include_refresh_timer=not args.no_refresh_timer,
        refresh_on_calendar=args.refresh_schedule,
    )

    if args.print_only:
        print(f"# {units.service_unit_path}\n{units.service_content}")
        if units.refresh_service_path and units.refresh_service_content:
            print(f"# {units.refresh_service_path}\n{units.refresh_service_content}")
        if units.refresh_timer_path and units.refresh_timer_content:
            print(f"# {units.refresh_timer_path}\n{units.refresh_timer_content}")
        return 0

    created_placeholder = ensure_env_file_placeholder(
        env_file, team_id=args.team_id, force_personal=args.personal
    )
    written = write_systemd_units(units)
    for path in written:
        print(f"Wrote {path}")

    if created_placeholder:
        print(
            f"\nCreated 0600 credentials template at {env_file}.\n"
            "Fill in PRIME_API_KEY and AMP_ORB_PROVIDER_TOKEN before starting the service."
        )
        return 0

    if args.no_enable:
        print("Skipped `systemctl --user enable --now` (--no-enable).")
        return 0

    notes = activate_systemd_units(units)
    print(f"Enabled and started {units.service_name}.service")
    if units.refresh_timer_path is not None:
        print(f"Enabled and started {units.refresh_timer_path.name}")
    for note in notes:
        print(f"Note: {note}")
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
    _add_scope_args(sub_build)
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
    _add_scope_args(sub_warm)
    sub_warm.set_defaults(func=cmd_warm_template)

    sub_status = sub.add_parser(
        "status",
        help="Show built Prime orb images, warm templates, active/paused orbs, and snapshots",
    )
    _add_scope_args(sub_status)
    sub_status.set_defaults(func=cmd_status)

    sub_gc = sub.add_parser(
        "gc-snapshots",
        help="Remove expired or deleted checkpoint/sandbox mappings from the local state store",
    )
    _add_scope_args(sub_gc)
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
    _add_scope_args(sub_run)
    sub_run.set_defaults(func=cmd_run)

    sub_install = sub.add_parser(
        "install-service",
        help="Write and enable `systemd --user` service + daily warm-template refresh timer",
    )
    sub_install.add_argument(
        "--env-file",
        default=str(Path.home() / ".config" / "prime-orbs.env"),
        help="Path to 0600 credentials file (default: ~/.config/prime-orbs.env)",
    )
    sub_install.add_argument(
        "--unit-dir",
        default=str(Path.home() / ".config" / "systemd" / "user"),
        help="Systemd user unit directory (default: ~/.config/systemd/user)",
    )
    sub_install.add_argument(
        "--service-name",
        default="prime-orbs",
        help="Base systemd unit name (default: prime-orbs)",
    )
    sub_install.add_argument(
        "--standby-count",
        type=int,
        default=None,
        help="Optional --standby-count flag baked into ExecStart",
    )
    sub_install.add_argument(
        "--refresh-schedule",
        default=DEFAULT_REFRESH_ON_CALENDAR,
        help=(
            f"Systemd OnCalendar schedule for warm-template refresh "
            f"(default: {DEFAULT_REFRESH_ON_CALENDAR})"
        ),
    )
    sub_install.add_argument(
        "--no-refresh-timer",
        action="store_true",
        help="Skip creating the daily warm-template refresh service and timer",
    )
    sub_install.add_argument(
        "--no-enable",
        action="store_true",
        help="Write unit files without running `systemctl --user enable --now`",
    )
    sub_install.add_argument(
        "--print-only",
        action="store_true",
        help="Print generated systemd unit contents to stdout without writing files",
    )
    _add_scope_args(sub_install)
    sub_install.set_defaults(func=cmd_install_service)

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
