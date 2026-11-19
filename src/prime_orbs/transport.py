"""Long-lived HTTP/1.1 & HTTP/2 client factory preventing idle TLS socket eviction."""

from __future__ import annotations

import functools

import httpx
from prime_sandboxes import APIClient, ImageClient, SandboxClient, UnauthorizedError

API_KEEPALIVE_EXPIRY_SECONDS = 300.0
GATEWAY_POOL_IDLE_TIMEOUT_SECONDS = 600.0
GATEWAY_TCP_KEEPALIVE_SECONDS = 15.0


@functools.lru_cache(maxsize=1)
def _ca_bundle_bytes() -> bytes:
    import certifi

    with open(certifi.where(), "rb") as handle:
        return handle.read()


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
    """Replace `APIClient.client` (default 5s keepalive) with a 300s keepalive pool."""
    if not isinstance(getattr(api_client, "config", None), _ExplicitScopeConfigProxy):
        api_client.config = _ExplicitScopeConfigProxy(api_client.config)  # type: ignore[assignment]
    old_http = getattr(api_client, "client", None)
    if not isinstance(old_http, httpx.Client):
        return api_client

    headers = dict(old_http.headers)
    timeout = old_http.timeout
    old_http.close()
    api_client.client = httpx.Client(
        headers=headers,
        follow_redirects=True,
        timeout=timeout,
        limits=httpx.Limits(
            max_connections=64,
            max_keepalive_connections=32,
            keepalive_expiry=API_KEEPALIVE_EXPIRY_SECONDS,
        ),
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
