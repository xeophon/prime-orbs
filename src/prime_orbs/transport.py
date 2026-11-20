"""Long-lived HTTP/1.1 & HTTP/2 client factory preventing idle TLS socket eviction."""

from __future__ import annotations

import contextlib
import functools
import socket

import httpx
from prime_sandboxes import APIClient, ImageClient, SandboxClient, UnauthorizedError

API_KEEPALIVE_EXPIRY_SECONDS = 300.0
API_TCP_KEEPALIVE_SECONDS = 15
GATEWAY_POOL_IDLE_TIMEOUT_SECONDS = 600.0
GATEWAY_TCP_KEEPALIVE_SECONDS = 15.0


@functools.lru_cache(maxsize=1)
def _ca_bundle_bytes() -> bytes:
    import certifi

    with open(certifi.where(), "rb") as handle:
        return handle.read()


def _tcp_keepalive_socket_options() -> list[tuple[int, int, int]]:
    """Build portable TCP_NODELAY + SO_KEEPALIVE socket options for long-lived HTTPS pools."""
    opts: list[tuple[int, int, int]] = [
        (socket.IPPROTO_TCP, socket.TCP_NODELAY, 1),
        (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    ]
    if hasattr(socket, "TCP_KEEPIDLE"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, API_TCP_KEEPALIVE_SECONDS))
    elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, API_TCP_KEEPALIVE_SECONDS))
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, API_TCP_KEEPALIVE_SECONDS))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4))
    return opts


class _ExplicitScopeConfigProxy:
    """Wraps `prime_sandboxes.Config` so `team_id` is never injected implicitly behind `OrbSpec`."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    @property
    def team_id(self) -> str | None:
        return None

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def tune_api_client(api_client: APIClient) -> APIClient:
    """Replace `APIClient.client` (default 5s keepalive) with a 300s TCP-keepalive pool."""
    if not isinstance(getattr(api_client, "config", None), _ExplicitScopeConfigProxy):
        api_client.config = _ExplicitScopeConfigProxy(api_client.config)  # type: ignore[assignment]
    old_http = getattr(api_client, "client", None)
    if not isinstance(old_http, httpx.Client):
        return api_client

    headers = dict(old_http.headers)
    timeout = old_http.timeout
    old_http.close()
    limits = httpx.Limits(
        max_connections=64,
        max_keepalive_connections=32,
        keepalive_expiry=API_KEEPALIVE_EXPIRY_SECONDS,
    )
    transport = httpx.HTTPTransport(
        limits=limits,
        socket_options=_tcp_keepalive_socket_options(),
    )
    api_client.client = httpx.Client(
        headers=headers,
        follow_redirects=True,
        timeout=timeout,
        transport=transport,
    )
    return api_client


def tune_sandbox_client(sandbox_client: SandboxClient) -> SandboxClient:
    """Extend `SandboxClient._command_pool` HTTP/2 idle timeout from 90s to 600s + TCP keepalive."""
    try:
        from prime_sandboxes.sandbox import _CommandTransportPool
        from pyqwest import HTTPVersion, SyncClient, SyncHTTPTransport
    except ImportError:  # pragma: no cover
        return sandbox_client

    if hasattr(sandbox_client, "_command_pool"):
        ca_cert = _ca_bundle_bytes()
        sandbox_client._command_pool = _CommandTransportPool(
            lambda: SyncClient(
                transport=SyncHTTPTransport(
                    tls_ca_cert=ca_cert,
                    http_version=HTTPVersion.HTTP2,
                    pool_idle_timeout=GATEWAY_POOL_IDLE_TIMEOUT_SECONDS,
                    tcp_keepalive_interval=GATEWAY_TCP_KEEPALIVE_SECONDS,
                )
            )
        )
    return sandbox_client


def prewarm_gateway_transport(sandbox_client: SandboxClient, sandbox_id: str) -> bool:
    """Open the gateway HTTP/2 TLS connection early so the first bootstrap RPC skips handshake."""
    auth_cache = getattr(sandbox_client, "_auth_cache", None)
    get_or_refresh = getattr(auth_cache, "get_or_refresh", None)
    command_pool = getattr(sandbox_client, "_command_pool", None)
    if not callable(get_or_refresh) or command_pool is None:
        return False

    try:
        auth = get_or_refresh(sandbox_id)
    except Exception:
        return False
    if not isinstance(auth, dict):
        return False
    gateway_url = str(auth.get("gateway_url") or "").rstrip("/")
    if not gateway_url.startswith("https://"):
        return False

    try:
        with command_pool.lease() as http_client:
            http_client.execute("HEAD", f"{gateway_url}/", timeout=5.0)
        return True
    except Exception:
        return False


def prewarm_api_connection(api_client: APIClient) -> bool:
    """Warm the Prime control-plane TLS socket so the first API call avoids a cold handshake."""
    http_client = getattr(api_client, "client", None)
    base_url = str(getattr(api_client, "base_url", "") or "").rstrip("/")
    if not isinstance(http_client, httpx.Client) or not base_url.startswith("https://"):
        return False
    with contextlib.suppress(Exception):
        http_client.head(f"{base_url}/health", timeout=5.0)
        return True
    return False


def create_tuned_clients(
    *, require_api_key: bool = True
) -> tuple[APIClient, SandboxClient, ImageClient]:
    """Build tuned `(APIClient, SandboxClient, ImageClient)` sharing warm connection pools."""
    api_client = APIClient()
    if require_api_key and not api_client.config.api_key:
        raise UnauthorizedError(
            "Missing Prime API key. Set PRIME_API_KEY in the provider environment "
            "or authenticate with `prime login`."
        )
    tune_api_client(api_client)
    sandbox_client = tune_sandbox_client(SandboxClient(api_client))
    image_client = ImageClient(api_client)
    return api_client, sandbox_client, image_client
