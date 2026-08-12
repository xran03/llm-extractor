"""Per-user credential store: paste a key once, then just run.

Editing a ``.env`` file is a poor fit for people who do not live in a terminal,
so ``llm-extract login`` asks for the key, verifies it against the gateway, and
records it here. Every later run picks it up with no further setup.

The store is a small JSON file in the user's own configuration directory:

* Windows — ``%APPDATA%\\llm-extractor\\credentials.json``
* macOS / Linux — ``$XDG_CONFIG_HOME`` (or ``~/.config``) ``/llm-extractor/``

It is written owner-only (``0600``) inside an owner-only directory (``0700``),
and is created with those permissions rather than relaxed-then-tightened, so
there is never a moment where another account on a shared machine could read
it. The value is never echoed, logged, or copied into the working directory —
which matters here, because a key pasted into ``.env`` inside a git repository
is one ``git add -A`` away from being published.

Resolution order is deliberate: an explicit ``--api-key``, a real environment
variable and a ``.env`` file all take precedence over the store, so shared
machines, CI and site-wide configuration keep behaving exactly as before.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_NAME = "llm-extractor"
STORE_FILENAME = "credentials.json"

#: Set this to relocate the store (used by the tests, and by anyone who keeps
#: their configuration on a different volume).
CONFIG_DIR_ENV = "LLM_EXTRACTOR_CONFIG_DIR"


def config_dir() -> Path:
    """Return the per-user configuration directory for this tool."""
    override = os.environ.get(CONFIG_DIR_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = os.environ.get("APPDATA", "").strip()
        root = Path(base) if base else Path.home() / "AppData" / "Roaming"
        return root / APP_NAME
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(base) if base else Path.home() / ".config"
    return root / APP_NAME


def store_path() -> Path:
    return config_dir() / STORE_FILENAME


def _restrict(path: Path, mode: int) -> None:
    """Best-effort permission tightening.

    ``chmod`` is meaningful on POSIX. On Windows the per-user profile is
    already protected by ACLs, and some network filesystems reject the call
    outright, so a failure here must never break an otherwise valid save.
    """
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def load_store() -> dict:
    """Return the whole store, or ``{}`` when it is missing or unreadable.

    A corrupt store is treated as absent rather than fatal: the user can always
    recover by running ``llm-extract login`` again.
    """
    path = store_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_credentials(api: str) -> dict:
    """Return ``{"api_key": ..., "base_url": ...}`` for one backend."""
    entry = load_store().get(api)
    if not isinstance(entry, dict):
        return {}
    return {
        "api_key": str(entry.get("api_key") or "").strip(),
        "base_url": str(entry.get("base_url") or "").strip().rstrip("/"),
    }


def stored_api_key(api: str) -> str:
    return read_credentials(api).get("api_key", "")


def stored_base_url(api: str) -> str:
    return read_credentials(api).get("base_url", "")


def _write_store(data: dict) -> Path:
    path = store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _restrict(path.parent, 0o700)

    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    # Opened with the final mode so the secret is never briefly world-readable.
    handle = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    _restrict(path, 0o600)
    return path


def save_credentials(api: str, api_key: str = "", base_url: str = "") -> Path:
    """Record credentials for one backend, leaving the others untouched."""
    data = load_store()
    entry = dict(data.get(api) or {}) if isinstance(data.get(api), dict) else {}
    if api_key:
        entry["api_key"] = api_key.strip()
    if base_url:
        entry["base_url"] = base_url.strip().rstrip("/")
    data[api] = entry
    return _write_store(data)


def delete_credentials(api: str = "") -> bool:
    """Forget one backend, or the entire store when ``api`` is empty."""
    if not api:
        path = store_path()
        try:
            path.unlink()
            return True
        except OSError:
            return False

    data = load_store()
    if api not in data:
        return False
    del data[api]
    if data:
        _write_store(data)
    else:
        try:
            store_path().unlink()
        except OSError:
            pass
    return True


def describe_store() -> str:
    """One non-secret line about the store, for ``llm-extract check``."""
    path = store_path()
    if not path.is_file():
        return "(none - run 'llm-extract login')"
    names = sorted(k for k, v in load_store().items() if isinstance(v, dict) and v.get("api_key"))
    if not names:
        return f"{path} (no keys saved)"
    return f"{path} ({', '.join(names)})"
