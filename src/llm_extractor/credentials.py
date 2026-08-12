"""Credential resolution: CLI flag, environment, ``.env`` file, or paste.

Precedence (first hit wins):

1. an explicit value passed on the command line (``--api-key``);
2. a real environment variable;
3. a ``.env`` file (searched upward from the current directory, then ``~``);
4. an interactive paste prompt (only when explicitly requested with ``-`` or
   when running on a TTY with ``allow_prompt=True``).

Secrets are never written to disk by this module and never logged.
"""
from __future__ import annotations

import os
from pathlib import Path

PROMPT_SENTINEL = "-"

_ENV_CACHE: dict[str, str] = {}
_ENV_LOADED_FROM: list[str] = []


def find_env_file(start: str | os.PathLike | None = None) -> Path | None:
    """Return the nearest ``.env``, searching upward then the home directory."""
    here = Path(start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        env_path = candidate / ".env"
        if env_path.is_file():
            return env_path
    home_env = Path.home() / ".env"
    return home_env if home_env.is_file() else None


def load_env_file(path: str | os.PathLike | None = None, override: bool = False) -> dict:
    """Parse a ``.env`` file into a dict and cache it for later lookups.

    Real environment variables win unless ``override`` is set, so CI/secret
    managers always beat a stale local file.
    """
    env_path = Path(path) if path else find_env_file()
    values: dict[str, str] = {}
    if env_path and env_path.is_file():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key.lower().startswith("export "):
                key = key[7:].strip()
            values[key] = value.strip().strip('"').strip("'")
        _ENV_LOADED_FROM[:] = [str(env_path)]
    _ENV_CACHE.update(values)
    if override:
        os.environ.update(values)
    return values


def env_file_in_use() -> str | None:
    return _ENV_LOADED_FROM[0] if _ENV_LOADED_FROM else None


def get_env(name: str, default: str = "") -> str:
    """Read ``name`` from the real environment, falling back to the ``.env`` file."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if name not in _ENV_CACHE and not _ENV_LOADED_FROM:
        load_env_file()
    return _ENV_CACHE.get(name, default).strip()


def prompt_secret(label: str) -> str:
    """Prompt once for a pasted secret (hidden input, falls back to plain input)."""
    import getpass

    try:
        value = getpass.getpass(f"Paste {label} (input hidden): ")
    except Exception:  # pragma: no cover - non-interactive terminals
        value = input(f"Paste {label}: ")
    return value.strip()


def resolve_secret(cli_value: str | None, env_names, label: str,
                   allow_prompt: bool = False, prompter=None) -> str:
    """Resolve one secret from CLI / env / .env / interactive paste.

    Passing ``-`` as the CLI value forces the paste prompt, which is the
    "copy-paste" path for users who do not keep a ``.env`` file.
    """
    prompter = prompter or prompt_secret
    if cli_value == PROMPT_SENTINEL:
        return prompter(label)
    if cli_value:
        return cli_value.strip()
    for name in ([env_names] if isinstance(env_names, str) else env_names):
        value = get_env(name)
        if value:
            return value
    if allow_prompt:
        return prompter(label)
    return ""


def mask(value: str) -> str:
    """Render a secret for logs: presence and length only, never the content."""
    if not value:
        return "(not set)"
    return f"set (len={len(value)}, prefix={value[:3]}***)"
