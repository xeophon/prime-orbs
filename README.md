# Prime Orbs

Run [Amp self-hosted orbs](https://ampcode.com/docs/orbs/self-hosted) on [Prime Intellect Sandboxes](https://docs.primeintellect.ai/sandboxes/overview).

Prime Sandboxes are hardware-isolated Linux microVMs with VM memory pause/resume and durable root-filesystem checkpoints, so orbs on Prime preserve **memory, disk, and project snapshots**:

| Capability | Support on Prime Sandboxes |
| --- | --- |
| **Memory** (processes survive `pause`/`spawn`) | Yes — `pause` freezes VM memory and disk; `resume` restores running processes, Terminal sessions, and services |
| **Disk** (files survive `pause`/`spawn`) | Yes — rootfs persists across pauses, and if a paused host node is ever lost, `spawn` automatically restores from the sandbox's durable pause checkpoint |
| **Project snapshots** (`ORB_SNAPSHOT_ID`) | Yes — `pause` saves a durable Prime filesystem checkpoint (with a 96h TTL by default) and later `spawn` jobs cold-boot new orbs directly from it |
| **Orb types** (`ORB_TYPE`) | Yes — built-in `tiny`/`small`/`medium`/`large`/`xlarge` presets plus custom CPU, memory, disk, GPU, image, and network rules via `PRIME_ORB_TYPES` |

## Prerequisites

- [Amp CLI](https://ampcode.com/docs) logged in, with a personal or workspace orb provider token (`aop_live_…`) from [Settings → Orbs](https://ampcode.com/settings/orbs)
- A Prime Intellect API key (`PRIME_API_KEY` or `prime login`) with sandbox and image permissions
- Python 3.11+ and [`uv`](https://docs.astral.sh/uv/)

## Quickstart

### 1. Install dependencies

```bash copy Copy Install Command
uv sync
```

### 2. Build the Amp orb image in Prime Images (once)

Prime Sandboxes only boot `prime/<owner>/<name>:<tag>` or Docker Hub images directly, and `prime images push --source-image` rejects Google Artifact Registry (`*.pkg.dev`). Building a one-line `Dockerfile` (`FROM us-central1-docker.pkg.dev/amp-prod-0/amp-orb/amp-orb:latest`) in Prime Images converts Amp's official orb image into a Prime VM image in the cloud (no local Docker required):

```bash copy Copy Image Build Command
export PRIME_API_KEY="<your-prime-api-key>"
uv run prime-orbs build-image --warm-template
```

`build-image` builds Amp's orb image with the latest `amp` CLI baked in, and `--warm-template` boots one clean template VM, faults `amp` into guest RAM, pauses it into a durable memory checkpoint, and deletes the temporary VM so future threads skip cold Linux kernel + `systemd` boot via `restore_memory=True`.

For ARM64 orbs, pass `--arch arm64` (builds `amp-orb-arm64:latest`). You can also refresh the warm memory checkpoint anytime without rebuilding the Docker image:

```bash
uv run prime-orbs warm-template
```

### 3. Run the orb provider

Run **one** `amp orb provider` process per provider token on a host with outbound HTTPS access:

```bash copy Copy Provider Command
export PRIME_API_KEY="<your-prime-api-key>"
export AMP_ORB_PROVIDER_TOKEN="aop_live_..."
amp orb provider /path/to/prime-orbs/dispatch.py
```

Or use `prime-orbs run`, which verifies Prime credentials, resolves your `prime/<owner>/amp-orb:latest` image, and hosts an in-memory UNIX-socket worker so `dispatch.py` completes in ~15ms reusing warm HTTP/1.1 & HTTP/2 connection pools:

```bash copy Copy Managed Run Command
uv run prime-orbs run --warm-template
```

Even when `ORB_IMAGE` is left unset (so Amp passes `us-central1-docker.pkg.dev/amp-prod-0/amp-orb/amp-orb:latest`), `dispatch.py` automatically looks up your built `amp-orb:latest` in Prime Images and resolves its `prime/<owner>/amp-orb:latest` reference.

### 4. Start a thread on Prime

Select your provider (for example, `prime`) under **Self-Hosted** in the Amp thread composer, or from the CLI:

```bash copy Copy Test Thread Command
amp -ox "uname -a && lscpu" --orb-provider prime
```

## Configuration

Configure defaults in the `amp orb provider` environment:

| Variable | Default | Description |
| --- | --- | --- |
| `PRIME_API_KEY` | `~/.prime/config.json` | Prime Intellect API key |
| `PRIME_TEAM_ID` | `~/.prime/config.json` | Prime team ID (`""` forces personal scope) |
| `PRIME_ORB_IMAGE` | `ORB_IMAGE` | Explicit `prime/<owner>/<image>:<tag>` override |
| `PRIME_ORB_CPU_CORES` | `4` | Default vCPUs per orb (`1`–`16`) |
| `PRIME_ORB_MEMORY_GB` | `8` | Default RAM in GiB (`0.125`–`64`) |
| `PRIME_ORB_DISK_SIZE_GB` | `20` | Default root disk in GiB (`2`–`128`) |
| `PRIME_ORB_ARCH` | inferred (`amd64`) | `amd64` or `arm64` |
| `PRIME_ORB_TIMEOUT_MINUTES` | `-1` | Sandbox lifetime cap (`-1` disables Prime's 60m cap so paused orbs are not terminated) |
| `PRIME_ORB_IDLE_TIMEOUT_MINUTES` | unset | Optional Prime idle timeout (`1`–`1440` minutes) |
| `PRIME_ORB_REGION` | unset | Optional Prime region placement |
| `PRIME_ORB_NETWORK_ALLOWLIST` | unset | Comma-separated outbound allowlist (domains/IPv4 CIDRs) |
| `PRIME_ORB_NETWORK_DENYLIST` | unset | Comma-separated outbound denylist |
| `PRIME_ORB_USER` | `user` | Guest user running `amp orb bootstrap` (`""` for sandbox default) |
| `PRIME_ORB_WARM_TEMPLATE` | `true` | Fork new CPU orbs from a cached `warm-template` memory checkpoint (`restore_memory=True`) when available |
| `PRIME_ORB_AUTO_WARM_TEMPLATE` | `false` | Build a warm memory template automatically during `prime-orbs run` if none exists |
| `PRIME_ORB_STANDBY_COUNT` | `1` | Pre-placed base standby VMs maintained by `prime-orbs run` (`0` disables all standby VMs) |
| `PRIME_ORB_MAX_PROJECT_STANDBYS` | `2` | Maximum recently used `ORB_SNAPSHOT_ID` project snapshots with pre-placed standby VMs |
| `PRIME_ORB_STANDBY_HOT_SECONDS` | `900` (15m) | How long standby VMs stay `RUNNING` after activity before auto-pausing (`0` pauses immediately) |
| `PRIME_ORB_SNAPSHOTS` | `true` | Enable `ORB_SNAPSHOT_ID` project snapshots (reusing pause checkpoints & promoting them to memory checkpoints in the background) |
| `PRIME_ORB_SNAPSHOT_TTL_SECONDS` | `345600` (96h) | Server-side TTL for project snapshot checkpoints (`0` disables expiry) |
| `PRIME_ORB_STATE_DIR` | `~/.local/state/prime-orbs` | Local directory for per-orb locks and `ORB_SNAPSHOT_ID` mappings |
| `PRIME_ORB_DISPATCH_TIMEOUT_SECONDS` | `270` | Per-invocation deadline (under Amp's 300s dispatch limit) |
| `PRIME_ORB_TYPES` | built-in presets | JSON object or path to JSON file defining/overriding `ORB_TYPE` specs |

### Orb Types (`ORB_TYPE`)

Built-in presets work immediately if you add matching Orb Type names (`tiny`, `small`, `medium`, `large`, `xlarge`) in **Settings → Orbs → Edit Provider**:

- `tiny`: 2 vCPUs, 4 GiB RAM, 10 GiB disk
- `small`: 4 vCPUs, 8 GiB RAM, 20 GiB disk
- `medium`: 8 vCPUs, 16 GiB RAM, 40 GiB disk
- `large`: 16 vCPUs, 32 GiB RAM, 80 GiB disk
- `xlarge`: 16 vCPUs, 64 GiB RAM, 128 GiB disk

Define custom orb types (including GPU sandboxes once GPU access is enabled on your Prime account) via `PRIME_ORB_TYPES`:

```json
{
  "dev": { "cpu_cores": 8, "memory_gb": 16, "disk_size_gb": 40 },
  "gpu": {
    "cpu_cores": 8,
    "memory_gb": 32,
    "disk_size_gb": 60,
    "gpu_count": 1,
    "gpu_type": "RTX_PRO_6000",
    "image": "prime/my-team/amp-orb-cuda:latest"
  }
}
```

Note: Prime GPU sandboxes do not support VM memory pause; when `ORB_OPERATION=pause` runs on a GPU orb, `prime-orbs` automatically checkpoints its root filesystem and terminates the GPU VM, then restores from that checkpoint on the next `spawn`.

## Helper Commands

```bash
# Inspect built Prime orb images, warm templates, standby VMs, active/paused orbs, and snapshots
uv run prime-orbs status

# Install & enable 24/7 systemd --user service + daily warm-template refresh timer on Linux VPS
uv run prime-orbs install-service

# Prune expired/deleted checkpoints from the local snapshot mapping store
uv run prime-orbs gc-snapshots
```
