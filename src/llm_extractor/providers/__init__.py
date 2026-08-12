"""Backend registry and provider construction.

``build_provider(settings)`` returns a ready-to-use provider for the selected
API, already wrapped in the response cache when caching is enabled.
"""
from __future__ import annotations

from .aimodelhub import AIModelHubProvider
from .base import Completion, HTTPProvider, ProviderError, Usage, image_part, text_part, user_message
from .llmhub import LLMHubProvider

BACKENDS = {
    "llmhub": LLMHubProvider,
    "aimodelhub": AIModelHubProvider,
}

__all__ = [
    "AIModelHubProvider",
    "BACKENDS",
    "Completion",
    "HTTPProvider",
    "LLMHubProvider",
    "ProviderError",
    "Usage",
    "build_provider",
    "image_part",
    "text_part",
    "user_message",
]


def build_provider(settings, cache=None, bus=None, bypass_cache: bool = False):
    """Instantiate the provider named by ``settings.api``.

    OAuth2 client credentials take precedence over a static API key, matching
    how gateways usually roll out short-lived tokens. Unless caching is
    disabled the result is wrapped in :class:`~llm_extractor.cache.CachedProvider`,
    so every stage transparently reuses previous responses.
    """
    try:
        provider_cls = BACKENDS[settings.api]
    except KeyError:
        raise ProviderError(
            f"unknown api '{settings.api}'; available: {sorted(BACKENDS)}"
        ) from None

    token_provider = None
    if settings.client_id and settings.client_secret and settings.token_url:
        from ..auth import OAuthTokenProvider

        token_provider = OAuthTokenProvider(
            settings.client_id, settings.client_secret,
            token_url=settings.token_url, timeout=settings.timeout,
        )

    provider = provider_cls(
        name=settings.api,
        base_url=settings.base_url,
        api_key=settings.api_key,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        token_provider=token_provider,
    )

    if cache is None and settings.cache_enabled:
        from ..cache import ResponseCache

        cache = ResponseCache(settings.cache_dir)
    if cache is not None:
        from ..cache import CachedProvider

        provider = CachedProvider(provider, cache, bus=bus, bypass=bypass_cache)
    return provider
