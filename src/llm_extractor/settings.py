"""Runtime settings: defaults, environment, ``.env``, CLI overrides.

One :class:`Settings` object carries everything the pipeline needs, so the CLI
is the only place that knows about argument parsing. Backend-specific values
(base URL, key, OAuth pair) are read from that backend's environment prefix,
which is how the same command works against either API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .credentials import get_env, resolve_secret
from .credstore import stored_api_key, stored_base_url

DEFAULT_API = "llmhub"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_OCR_MODEL = "gpt-4.1"
DEFAULT_AGENT_MODEL = "gpt-4.1-mini"
DEFAULT_CACHE_DIR = ".llm_cache"

# Environment prefix per backend, so both APIs can be configured side by side.
ENV_PREFIX = {
    "llmhub": "LLM_HUB",
    "aimodelhub": "AI_MODEL_HUB",
}


@dataclass
class Settings:
    api: str = DEFAULT_API
    base_url: str = ""
    api_key: str = ""
    client_id: str = ""
    client_secret: str = ""
    token_url: str = ""

    model: str = DEFAULT_MODEL
    ocr_model: str = DEFAULT_OCR_MODEL
    agent_model: str = DEFAULT_AGENT_MODEL

    timeout: float = 90.0
    max_retries: int = 4
    max_workers: int = 8
    max_output_tokens: int = 16000
    temperature: float = 0.0

    cache_dir: str = DEFAULT_CACHE_DIR
    cache_enabled: bool = True

    template: str = "generic"
    ocr: str = "auto"          # auto | always | never
    aggregate: bool = True     # run the aggregation agent over text + OCR JSON
    max_figures: int = 20
    output_format: str = "both"  # jsonl | csv | both

    extra: dict = field(default_factory=dict)

    @property
    def env_prefix(self) -> str:
        return ENV_PREFIX.get(self.api, self.api.upper().replace("-", "_"))

    def describe(self) -> dict:
        """Non-secret summary suitable for logs and the ``check`` command."""
        from .credentials import mask

        return {
            "api": self.api,
            "base_url": self.base_url or "(not set)",
            "api_key": mask(self.api_key),
            "client_id": mask(self.client_id),
            "client_secret": mask(self.client_secret),
            "token_url": self.token_url or "(not set)",
            "model": self.model,
            "ocr_model": self.ocr_model,
            "agent_model": self.agent_model,
            "template": self.template,
            "cache": self.cache_dir if self.cache_enabled else "(disabled)",
        }


def build_settings(api: str | None = None, *, base_url: str | None = None,
                   api_key: str | None = None, model: str | None = None,
                   ocr_model: str | None = None, agent_model: str | None = None,
                   cache_dir: str | None = None, cache_enabled: bool = True,
                   allow_prompt: bool = False, prompter=None, **overrides) -> Settings:
    """Resolve settings from defaults + environment + explicit overrides."""
    api = (api or get_env("LLM_EXTRACTOR_API") or DEFAULT_API).strip().lower()
    prefix = ENV_PREFIX.get(api, api.upper().replace("-", "_"))

    settings = Settings(api=api)
    settings.base_url = (
        base_url or get_env(f"{prefix}_BASE_URL") or stored_base_url(api)
    ).rstrip("/")
    settings.client_id = get_env(f"{prefix}_CLIENT_ID")
    settings.client_secret = get_env(f"{prefix}_CLIENT_SECRET")
    settings.token_url = get_env(f"{prefix}_TOKEN_URL")

    # Only prompt for a key when OAuth credentials are absent.
    has_oauth = bool(settings.client_id and settings.client_secret and settings.token_url)
    settings.api_key = resolve_secret(
        api_key,
        [f"{prefix}_API_KEY", f"{prefix}_KEY"],
        label=f"{api} API key",
        allow_prompt=allow_prompt and not has_oauth,
        prompter=prompter,
        fallback=None if has_oauth else (lambda: stored_api_key(api)),
    )

    settings.model = model or get_env("LLM_EXTRACTOR_MODEL") or DEFAULT_MODEL
    settings.ocr_model = ocr_model or get_env("LLM_EXTRACTOR_OCR_MODEL") or settings.model
    settings.agent_model = (
        agent_model or get_env("LLM_EXTRACTOR_AGENT_MODEL") or settings.model
    )

    settings.cache_dir = str(
        Path(cache_dir or get_env("LLM_EXTRACTOR_CACHE_DIR") or DEFAULT_CACHE_DIR)
    )
    settings.cache_enabled = cache_enabled

    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    return settings
