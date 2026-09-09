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

# Per-backend model defaults. A gateway only serves the models it hosts, so the
# right default depends on which one is being called: picking a name the
# backend has never heard of turns a first run into a 404 hunt.
#
# ``review`` is the model used to judge extracted records rather than produce
# them. A different model family is deliberate — a second opinion from the same
# model that wrote the answer mostly restates it.
BACKEND_DEFAULTS = {
    "aimodelhub": {
        "model": "gpt-5.6-sol",
        "ocr_model": "gpt-5.6-sol",
        "agent_model": "gpt-5.6-sol",
        # A different model, not merely a different name: scout is a separate
        # family from sol. Anthropic models are visible on this gateway but keys
        # are commonly scoped away from them, and a default that 403s at call
        # time is worse than a slightly weaker reviewer that runs.
        "review_model": "scout-gpt-5.1",
    },
    "llmhub": {
        "model": DEFAULT_MODEL,
        "ocr_model": DEFAULT_OCR_MODEL,
        "agent_model": DEFAULT_AGENT_MODEL,
        "review_model": DEFAULT_AGENT_MODEL,
    },
}


def backend_default(api: str, role: str, fallback: str = "") -> str:
    """The default model for one role on one backend."""
    defaults = BACKEND_DEFAULTS.get(api) or {}
    return defaults.get(role) or fallback


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
    chart: str = "auto"        # auto | always | never — vector figure digitisation
    chart_model: str = ""      # model that names panels/groups; defaults to model
    review_model: str = ""     # model that judges records; a second opinion
    aggregate: bool = True     # run the aggregation agent over text + OCR JSON
    max_figures: int = 20
    max_chart_pages: int = 20
    max_chart_points: int = 20000
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
            "review_model": self.review_model,
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

    # A model chosen explicitly drives the derived roles too: someone passing
    # --model expects the whole run to use it, not for OCR to silently fall
    # back to the backend's default. The per-backend defaults apply only when
    # nothing was chosen at all.
    chosen = model or get_env("LLM_EXTRACTOR_MODEL")
    settings.model = chosen or backend_default(api, "model", DEFAULT_MODEL)
    settings.ocr_model = (ocr_model or get_env("LLM_EXTRACTOR_OCR_MODEL")
                          or chosen or backend_default(api, "ocr_model", settings.model))
    settings.agent_model = (agent_model or get_env("LLM_EXTRACTOR_AGENT_MODEL")
                            or chosen
                            or backend_default(api, "agent_model", settings.model))
    settings.review_model = (get_env("LLM_EXTRACTOR_REVIEW_MODEL")
                             or backend_default(api, "review_model", settings.agent_model))
    settings.chart_model = get_env("LLM_EXTRACTOR_CHART_MODEL") or settings.model

    settings.cache_dir = str(
        Path(cache_dir or get_env("LLM_EXTRACTOR_CACHE_DIR") or DEFAULT_CACHE_DIR)
    )
    settings.cache_enabled = cache_enabled

    for key, value in overrides.items():
        if value is not None and hasattr(settings, key):
            setattr(settings, key, value)
    return settings
