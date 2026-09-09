"""Command-line interface.

The common case is one line::

    llm-extract -i ./docs -o ./out --api llmhub

``run`` is the default subcommand, so the flags above work without naming it.
Everything else (``sources``, ``models``, ``check``, ``cache``, ``audit``,
``serve``, ``templates``) is a named subcommand.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from ._exec import BACKEND as EXEC_BACKEND, BACKEND_DETAIL as EXEC_BACKEND_DETAIL
from .bus import (DOC_COMPLETED, DOC_FAILED, DOC_SKIPPED, EventBus, JOB_COMPLETED,
                  JOB_STARTED)
from .credentials import PROMPT_SENTINEL, env_file_in_use, load_env_file, prompt_secret
from .credstore import (delete_credentials, describe_store, save_credentials,
                        store_path, stored_base_url)
from .ingest import describe_formats
from .providers import ProviderError
from .settings import DEFAULT_CACHE_DIR, build_settings
from .sources import SOURCES, available_sources, build_source
from .templates import (BUILTIN_TEMPLATES, STARTER_TEMPLATE, TemplateError,
                        load_template)

#: Diagnostics probe the gateway with a short budget: 'check' exists to report
#: an unreachable host quickly, not to keep retrying one.
PROBE_TIMEOUT = 15.0

SUBCOMMANDS = {"run", "sources", "formats", "models", "check", "cache", "audit", "serve",
               "templates", "login", "logout", "batch"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-extract",
        description="Folder-in / JSON-out document extraction powered by an LLM API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  llm-extract login\n"
            "  llm-extract -i ./docs -o ./out --api llmhub\n"
            "  llm-extract -i ./docs -o ./out --api aimodelhub --template immunogenicity\n"
            "  llm-extract run --source europepmc --param query='pneumococcal conjugate' -o ./out\n"
            "  llm-extract audit --n 25 --strategy oldest --referee-model gpt-4.1\n"
            "  llm-extract serve --port 8080\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"llm-extract {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="extract a folder (or any registered source)")
    _add_common(run)
    run.add_argument("-i", "--input", help="input folder (or file) to extract")
    run.add_argument("-o", "--output", default="out", help="output directory (default: out)")
    run.add_argument("--source", default="folder",
                     help=f"document source: {', '.join(sorted(available_sources()))}")
    run.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                     help="source parameter; repeatable (e.g. --param query=vaccine)")
    run.add_argument("--extensions", help="comma-separated extension filter, e.g. .pdf,.docx")
    run.add_argument("--limit", type=int, default=0, help="stop after N documents")
    run.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    run.add_argument("--rate-limit", type=int, default=0,
                     help="max API calls per minute")
    run.add_argument("--ocr", choices=["auto", "always", "never"], default=None,
                     help="figure OCR policy (default: auto)")
    run.add_argument("--chart", choices=["auto", "always", "never"], default=None,
                     help="digitise vector figures, one record per plotted point "
                          "(default: auto, which also skips OCR on pages it measured)")
    run.add_argument("--format", dest="output_format", default=None,
                     choices=["jsonl", "csv", "both"],
                     help="record artifacts to write (default: both)")
    run.add_argument("--no-aggregate", action="store_true",
                     help="skip the reconciliation agent (no extra API call)")
    run.add_argument("--no-resume", action="store_true",
                     help="re-extract documents even if unchanged since a previous run")
    run.add_argument("--quiet", action="store_true", help="only print the final summary")

    for name, help_text in (("sources", "list registered document sources"),
                            ("formats", "list supported document formats"),
                            ("templates", "list, inspect or scaffold extraction templates"),
                            ("models", "list models offered by the API"),
                            ("check", "verify credentials and configuration")):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        if name == "check":
            p.add_argument("--no-verify", action="store_true",
                           help="report the configuration without contacting the gateway")
        if name == "templates":
            p.add_argument("--show", help="print one template with its JSON schema")
            p.add_argument("--init", metavar="PATH",
                           help="write a starter template JSON you can edit")
            p.add_argument("--validate", metavar="PATH",
                           help="check a template JSON and report any problem")

    cache = sub.add_parser("cache", help="inspect or clear the response cache")
    _add_common(cache)
    cache.add_argument("action", choices=["stats", "clear", "entries"], default="stats",
                       nargs="?")
    cache.add_argument("--stage", default="", help="filter by stage (extract/ocr/aggregate)")
    cache.add_argument("--limit", type=int, default=20)

    audit = sub.add_parser("audit", help="re-validate a sample of cached extractions")
    _add_common(audit)
    audit.add_argument("--n", type=int, default=20, help="sample size (default 20)")
    audit.add_argument("--stage", default="", help="restrict to one stage")
    audit.add_argument("--strategy", default="random",
                       choices=["random", "oldest", "newest", "largest", "unverified"])
    audit.add_argument("--seed", type=int, default=None, help="make sampling reproducible")
    audit.add_argument("--referee-model", default="",
                       help="replay with a different (usually stronger) model")
    audit.add_argument("--workers", type=int, default=1, help=argparse.SUPPRESS)
    audit.add_argument("--invalidate-drifted", action="store_true",
                       help="delete entries that failed the audit so they re-extract")
    audit.add_argument("-o", "--output", default="", help="write the audit report JSON here")

    batch = sub.add_parser("batch", help="queue a corpus through the provider's batch API")
    batch_sub = batch.add_subparsers(dest="batch_cmd", required=True)

    bp = batch_sub.add_parser("preflight", help="check whether this gateway can run batches")
    _add_common(bp)

    bs = batch_sub.add_parser("submit", help="upload a corpus as one queued batch")
    _add_common(bs)
    bs.add_argument("-i", "--input", required=True, help="folder (or file) to extract")
    bs.add_argument("-o", "--output", default="out", help="where the manifest and results go")
    bs.add_argument("--extensions", help="comma-separated extension filter")
    bs.add_argument("--limit", type=int, default=0, help="stop after N documents")
    bs.add_argument("--figures", action="store_true",
                    help="queue figure OCR in the same batch")
    bs.add_argument("--completion-window", default="24h",
                    help="how long the gateway may take (default 24h)")

    for name, help_text in (("status", "report how far along a batch is"),
                            ("fetch", "download a finished batch and write artifacts"),
                            ("cancel", "cancel a running batch")):
        p = batch_sub.add_parser(name, help=help_text)
        _add_common(p)
        p.add_argument("-o", "--output", default="out",
                       help="directory holding the batch manifest")
        p.add_argument("--batch-id", default="",
                       help="batch to act on (default: the one in the manifest)")
        if name == "fetch":
            p.add_argument("--wait", action="store_true",
                           help="poll until the batch finishes, then collect")
            p.add_argument("--poll-seconds", type=int, default=60,
                           help="seconds between polls when waiting")

    bl = batch_sub.add_parser("list", help="list recent batches on the gateway")
    _add_common(bl)
    bl.add_argument("--limit", type=int, default=20)

    serve = sub.add_parser("serve", help="run the HTTP API for a frontend")
    _add_common(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)
    serve.add_argument("-o", "--output", default="out", help="artifact directory")
    serve.add_argument("--token", default="", help="require this bearer token")

    login = sub.add_parser("login", help="paste an API key once and save it securely")
    _add_common(login)
    login.add_argument("--no-verify", action="store_true",
                       help="save without checking the key against the gateway")

    logout = sub.add_parser("logout", help="forget the saved API key")
    _add_common(logout)
    logout.add_argument("--all", action="store_true",
                        help="forget every backend, not just the selected one")
    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--api", default=None,
                        help="API backend: llmhub (chat completions) or aimodelhub (responses)")
    parser.add_argument("--base-url", default=None, help="override the gateway base URL")
    parser.add_argument("--api-key", default=None,
                        help=f"API key; use '{PROMPT_SENTINEL}' to paste it interactively")
    parser.add_argument("--env-file", default=None, help="path to a .env file")
    parser.add_argument("--model", default=None, help="extraction model")
    parser.add_argument("--ocr-model", default=None, help="vision model for figures")
    parser.add_argument("--agent-model", default=None, help="model for the aggregation agent")
    parser.add_argument("--template", default=None,
                        help=f"{', '.join(sorted(BUILTIN_TEMPLATES))}, or a template JSON path")
    parser.add_argument("--cache-dir", default=None, help=f"cache directory (default {DEFAULT_CACHE_DIR})")
    parser.add_argument("--no-cache", action="store_true", help="disable the response cache")


def settings_from_args(args, allow_prompt: bool = False):
    """Build settings from parsed CLI arguments.

    Interactive pasting is explicit: it happens when the user passes
    ``--api-key -``. We never prompt implicitly, because that would block
    scripts, CI and piped invocations that are simply missing configuration.
    """
    if getattr(args, "env_file", None):
        load_env_file(args.env_file, override=True)
    return build_settings(
        api=args.api,
        base_url=getattr(args, "base_url", None),
        api_key=getattr(args, "api_key", None),
        model=args.model,
        ocr_model=args.ocr_model,
        agent_model=args.agent_model,
        cache_dir=args.cache_dir,
        cache_enabled=not args.no_cache,
        allow_prompt=allow_prompt,
        template=args.template,
        ocr=getattr(args, "ocr", None),
        chart=getattr(args, "chart", None),
        output_format=getattr(args, "output_format", None),
        aggregate=False if getattr(args, "no_aggregate", False) else None,
        max_workers=getattr(args, "workers", None),
    )


def _interactive() -> bool:
    """True when a human is present to answer a prompt.

    Scripts, CI and piped invocations must keep failing fast with a message
    rather than blocking forever on a prompt nobody will ever see, so every
    implicit prompt is gated on this.
    """
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def parse_params(pairs) -> dict:
    """Parse ``--param key=value`` into a dict, decoding JSON values when possible."""
    params: dict = {}
    for pair in pairs or []:
        key, _, value = str(pair).partition("=")
        key = key.strip()
        if not key:
            continue
        try:
            params[key] = json.loads(value)
        except json.JSONDecodeError:
            params[key] = value
    return params


# --------------------------------- commands ---------------------------------
def cmd_run(args) -> int:
    # Validate arguments before touching credentials or the network.
    if args.source == "folder" and not args.input and \
            "input_dir" not in parse_params(args.param):
        print("error: -i/--input is required for the folder source", file=sys.stderr)
        return 2

    settings = settings_from_args(args, allow_prompt=_interactive())
    template = load_template(settings.template)
    source_params = parse_params(args.param)

    if args.source == "folder":
        source_params.setdefault("input_dir", args.input)
        if args.extensions:
            source_params["extensions"] = [
                e if e.startswith(".") else f".{e}"
                for e in args.extensions.split(",") if e.strip()
            ]
    if args.limit:
        source_params.setdefault("limit", args.limit)

    bus = EventBus()
    if not args.quiet:
        _attach_progress(bus)

    from .runner import run_job

    summary = run_job(
        settings, source_name=args.source, source_params=source_params,
        out_dir=args.output, bus=bus, resume=not args.no_resume,
        rate_limit=args.rate_limit,
    )
    print()
    print(f"documents : {summary.ok} ok, {summary.skipped} skipped, {summary.failed} failed "
          f"(of {summary.total})")
    print(f"records   : {summary.records}   figures OCR'd: {summary.figures}")
    print(f"tokens    : {summary.prompt_tokens} in / {summary.completion_tokens} out"
          f"   cached calls: {summary.cached_calls}")
    if summary.cache:
        print(f"cache     : {summary.cache.get('entries', 0)} entries, "
              f"hit rate {summary.cache.get('hit_rate', 0)}")
    print(f"template  : {template.name}   api: {settings.api}   model: {settings.model}")
    print(f"output    : {Path(args.output).resolve()}")
    for name, path in summary.tables.items():
        print(f"  {name:<12}: {path}")
    for error in summary.errors[:5]:
        print(f"  ! {error}", file=sys.stderr)
    return 1 if summary.failed and not summary.ok else 0


def cmd_sources(args) -> int:
    for name, description in sorted(available_sources().items()):
        print(f"{name:<14} {description}")
    return 0


def cmd_formats(args) -> int:
    print(f"{'format':<10}{'kind':<13}{'text':<6}{'figures':<10}extensions")
    for fmt in sorted(describe_formats(), key=lambda f: (f["kind"], f["name"])):
        needs = f"  (needs {fmt['requires']})" if fmt["requires"] else ""
        print(f"{fmt['name']:<10}{fmt['kind']:<13}"
              f"{'yes' if fmt['text_layer'] else 'no':<6}{fmt['figures']:<10}"
              f"{' '.join(fmt['extensions'])}{needs}")
    print("\nformats are detected from content, so a mislabelled or extension-less "
          "file is still read correctly")
    return 0


def cmd_templates(args) -> int:
    if getattr(args, "init", None):
        path = Path(args.init)
        if path.exists():
            print(f"error: {path} already exists", file=sys.stderr)
            return 2
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(STARTER_TEMPLATE, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"wrote starter template -> {path}")
        print("edit the fields, then run:")
        print(f"  llm-extract templates --validate {path}")
        print(f"  llm-extract -i ./docs -o ./out --template {path}")
        return 0

    if getattr(args, "validate", None):
        try:
            template = load_template(args.validate)
        except TemplateError as exc:
            print(f"invalid: {exc}", file=sys.stderr)
            return 1
        print(f"valid: '{template.name}' with {len(template.fields)} fields")
        print(f"  fields    : {', '.join(template.field_names)}")
        print(f"  key_fields: {', '.join(template.key_fields) or '(none)'}")
        return 0

    if getattr(args, "show", None):
        template = load_template(args.show)
        print(json.dumps({**template.to_dict(), "json_schema": template.json_schema()},
                         ensure_ascii=False, indent=2))
        return 0

    for name, template in sorted(BUILTIN_TEMPLATES.items()):
        print(f"{name:<16} {template.description}")
        print(f"{'':<16} fields: {', '.join(template.field_names)}")
    print("\ncustom schemas: llm-extract templates --init my-template.json")
    return 0


def cmd_models(args) -> int:
    from .providers import build_provider

    settings = settings_from_args(args)
    # Model discovery is a live probe; caching it would defeat the purpose.
    settings.cache_enabled = False
    for model in build_provider(settings).list_models():
        print(model)
    return 0


def cmd_check(args) -> int:
    settings = settings_from_args(args)
    print(f"env file  : {env_file_in_use() or '(none found)'}")
    print(f"saved key : {describe_store()}")
    for key, value in settings.describe().items():
        print(f"{key:<10}: {value}")
    print(f"sources   : {', '.join(SOURCES.names())}")
    print(f"templates : {', '.join(sorted(BUILTIN_TEMPLATES))}")
    print(f"execution : {EXEC_BACKEND} ({EXEC_BACKEND_DETAIL})")

    if not settings.base_url:
        print(f"\nnot configured: run 'llm-extract login' to paste a key, or set "
              f"{settings.env_prefix}_BASE_URL in .env (see .env.example)", file=sys.stderr)
        return 1
    if not (settings.api_key or (settings.client_id and settings.client_secret)):
        print(f"\nnot configured: no credentials. Run 'llm-extract login' to paste a key, "
              f"or set {settings.env_prefix}_API_KEY in .env",
              file=sys.stderr)
        return 1

    if getattr(args, "no_verify", False):
        print("\nconnectivity: not checked (--no-verify)")
        return 0

    try:
        from .providers import build_provider

        settings.cache_enabled = False
        # A diagnostic must fail fast: the full retry budget would take minutes
        # to report an unreachable host, which is the one thing it exists to say.
        settings.timeout = min(settings.timeout, PROBE_TIMEOUT)
        settings.max_retries = 1
        models = build_provider(settings).list_models()
        print(f"\nconnectivity: OK ({len(models)} models visible)")
    except Exception as exc:
        print(f"\nconnectivity: FAILED - {exc}", file=sys.stderr)
        return 1

    # Visible is not the same as permitted, and /v1/models is not always
    # exhaustive: gateways serve aliases and deployments they do not list. So
    # this is advisory — worth saying, not worth failing a working setup over.
    configured = {
        "model": settings.model,
        "ocr_model": settings.ocr_model,
        "agent_model": settings.agent_model,
        "review_model": settings.review_model,
    }
    offered = set(models)
    missing = {role: name for role, name in configured.items()
               if name and name not in offered}
    if missing:
        print("\nmodels not listed by this gateway (it may still serve them):")
        for role, name in sorted(missing.items()):
            print(f"  {role:<12} {name}")
        print("  run 'llm-extract models' to see what it does list")
    else:
        print(f"models    : {', '.join(sorted(set(configured.values())))} all listed")
    return 0


def cmd_login(args) -> int:
    """Paste a key once, verify it, and save it for every later run.

    Verification happens *before* saving, so a mistyped or expired key is
    reported here — while the user still has it on the clipboard — instead of
    failing later inside an extraction run.
    """
    if getattr(args, "env_file", None):
        load_env_file(args.env_file, override=True)

    # Resolve the backend and any base URL already configured. The key found
    # here is deliberately ignored: this command exists to replace it.
    probe = build_settings(api=args.api, base_url=args.base_url, cache_enabled=False)
    api, base_url = probe.api, probe.base_url

    if not base_url:
        if not _interactive():
            print(f"error: no base URL for '{api}'. Pass --base-url https://your-gateway",
                  file=sys.stderr)
            return 2
        base_url = input(f"Gateway base URL for {api} (e.g. https://your-gateway): ").strip()
        if not base_url:
            print("error: a base URL is required", file=sys.stderr)
            return 2

    supplied = getattr(args, "api_key", None)
    if supplied and supplied != PROMPT_SENTINEL:
        key = supplied.strip()
    elif _interactive() or supplied == PROMPT_SENTINEL:
        key = prompt_secret(f"{api} API key")
    else:
        print("error: no terminal to paste into. Pass --api-key, or run this "
              "in an interactive shell.", file=sys.stderr)
        return 2
    if not key:
        print("error: no key entered; nothing was saved", file=sys.stderr)
        return 2

    settings = build_settings(api=api, base_url=base_url, api_key=key, cache_enabled=False)
    if not args.no_verify:
        ok, detail = verify_credentials(settings)
        if not ok:
            print(f"\nnot saved: {detail}", file=sys.stderr)
            return 1
        print(f"verified : {detail}")

    path = save_credentials(api, api_key=key, base_url=base_url)
    print(f"saved    : {path} (readable only by you)")
    # Name the backend in the hint only when it is not the one picked up by
    # default, so the suggested command works as printed.
    default_api = build_settings(cache_enabled=False).api
    suffix = "" if api == default_api else f" --api {api}"
    print(f"\nYou are ready. Try:\n  llm-extract -i ./docs -o ./out{suffix}")
    return 0


def _is_auth_failure(exc) -> bool:
    """True when a provider error is specifically a rejected credential."""
    message = str(exc)
    return "HTTP 401" in message or "HTTP 403" in message


def verify_credentials(settings) -> tuple:
    """Prove a key actually works, and say how confident that verdict is.

    Listing models is not sufficient evidence on its own: some gateways serve
    ``/v1/models`` without authentication, so a model list proves the host is
    reachable but says nothing about the key. A single one-token completion is
    what actually exercises authentication, so it decides the verdict.

    Returns ``(ok, detail)``. A non-authentication failure (an unavailable
    model, a transient error) is reported but does not veto the save, because
    the key itself was never shown to be wrong.
    """
    from .providers import build_provider

    provider = build_provider(settings)
    try:
        models = provider.list_models()
    except Exception as exc:
        if _is_auth_failure(exc):
            return False, f"the gateway rejected this key - {exc}"
        return False, f"the gateway could not be reached - {exc}"

    # Prefer a model the gateway actually offers, so a wrong default model
    # cannot be mistaken for a wrong key.
    model = settings.model if settings.model in models else (models[0] if models else settings.model)
    try:
        provider.complete([{"role": "user", "content": "ping"}],
                          model=model, temperature=0.0, max_tokens=1)
    except Exception as exc:
        if _is_auth_failure(exc):
            return False, f"the gateway rejected this key - {exc}"
        return True, (f"{len(models)} models visible; the key could not be fully "
                      f"exercised ({exc})")
    return True, f"{len(models)} models visible, test call accepted"


def cmd_logout(args) -> int:
    """Forget a saved key."""
    if args.all:
        removed = delete_credentials()
        print("removed every saved credential" if removed else "nothing was saved")
        return 0
    api = build_settings(api=args.api, cache_enabled=False).api
    if delete_credentials(api):
        print(f"removed the saved credentials for {api} ({store_path()})")
    else:
        print(f"nothing saved for {api}")
    return 0


def cmd_cache(args) -> int:
    from .cache import ResponseCache

    settings = settings_from_args(args)
    cache = ResponseCache(settings.cache_dir)
    try:
        if args.action == "clear":
            print(f"removed {cache.clear()} cache entries from {settings.cache_dir}")
            return 0
        if args.action == "entries":
            for row in cache.query(stage=args.stage, limit=args.limit):
                print(f"{row['key'][:12]}  {row.get('stage') or '-':<10} "
                      f"{row.get('doc_id') or '-':<28} {row.get('model') or '-':<18} "
                      f"{row.get('verdict') or 'unverified'}")
            return 0
        print(json.dumps(cache.summary(), ensure_ascii=False, indent=2))
        return 0
    finally:
        cache.close()


def cmd_audit(args) -> int:
    from .audit import audit_cache
    from .cache import ResponseCache
    from .providers import build_provider

    settings = settings_from_args(args)
    cache = ResponseCache(settings.cache_dir)
    try:
        # Bypass the cache on replay, otherwise we would compare an entry to itself.
        provider = build_provider(settings, cache=cache, bypass_cache=True)
        report = audit_cache(
            cache, provider, n=args.n, stage=args.stage, strategy=args.strategy,
            seed=args.seed, referee_model=args.referee_model, max_workers=args.workers,
            invalidate_drifted=args.invalidate_drifted,
            only_unverified=args.strategy == "unverified",
        )
        data = report.to_dict()
    finally:
        cache.close()

    print(f"sampled {data['sampled']} of {data['population']} cached calls "
          f"(strategy={data['strategy']}, referee={data['referee_model'] or 'same model'})")
    print(f"confirmed {data['confirmed']}  drifted {data['drifted']}  "
          f"suspect {data['suspect']}  errors {data['errors']}")
    print(f"pass rate {data['pass_rate']}  95% CI {data['pass_rate_ci95']}  "
          f"mean agreement {data['mean_agreement']}")
    for entry in data["entries"]:
        if entry["verdict"] != "confirmed":
            print(f"  {entry['verdict']:<10} {entry['doc_id'] or entry['key'][:12]:<28} "
                  f"agreement={entry['agreement']} changed={entry['changed_fields']}"
                  f"{' ' + entry['error'] if entry['error'] else ''}")
    if args.output:
        Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
        print(f"report -> {args.output}")
    return 0


def cmd_batch(args) -> int:
    """Queue work through the provider's batch API.

    Split into submit / status / fetch because a batch outlives the process
    that starts it: the point is to hand the corpus over and come back, not to
    hold a terminal open for the completion window.
    """
    import time as _time

    from . import batchapi
    from .providers import build_provider

    settings = settings_from_args(args)
    settings.cache_enabled = False          # a queued request is never a cache hit
    provider = build_provider(settings)
    action = args.batch_cmd

    if action == "preflight":
        report = batchapi.preflight(provider)
        print(f"api          : {settings.api} ({provider.INFERENCE_PATH})")
        print(f"list batches : {report['list_batches']}")
        print(f"upload files : {report['upload_files'] or 'not reached'}")
        if report["detail"]:
            print(f"detail       : {report['detail']}")
        print(f"\nbatch ready  : {'yes' if report['ready'] else 'no'}")
        if not report["ready"]:
            print("the gateway exposes the batch routes but cannot store input files; "
                  "this is a deployment setting on the gateway, not a client problem",
                  file=sys.stderr)
        return 0 if report["ready"] else 1

    if action == "list":
        for entry in provider.list_batches(limit=args.limit).get("data", []):
            counts = entry.get("request_counts") or {}
            print(f"{entry.get('id'):<40} {entry.get('status'):<12} "
                  f"{counts.get('completed', 0)}/{counts.get('total', 0)}")
        return 0

    out_dir = Path(args.output)
    manifest_path = out_dir / batchapi.MANIFEST_NAME

    if action == "submit":
        template = load_template(settings.template)
        source_params = {"input_dir": args.input}
        if args.extensions:
            source_params["extensions"] = [
                e if e.startswith(".") else f".{e}"
                for e in args.extensions.split(",") if e.strip()
            ]
        if args.limit:
            source_params["limit"] = args.limit

        documents = list(build_source("folder", **source_params).iter_documents())
        if not documents:
            print(f"error: no readable documents under {args.input}", file=sys.stderr)
            return 2

        manifest = batchapi.submit(provider, documents, template, settings, out_dir,
                                   completion_window=args.completion_window,
                                   with_ocr=args.figures)
        stages = {}
        for task in manifest.tasks:
            stages[task.stage] = stages.get(task.stage, 0) + 1
        print(f"batch     : {manifest.batch_id}")
        print(f"documents : {len(manifest.documents())}")
        print(f"requests  : {len(manifest.tasks)}  ({', '.join(f'{k}={v}' for k, v in sorted(stages.items()))})")
        print(f"manifest  : {manifest_path}")
        print(f"\nnext: llm-extract batch fetch -o {args.output} --wait")
        return 0

    if not manifest_path.exists() and not args.batch_id:
        print(f"error: no manifest at {manifest_path}; pass --batch-id or submit first",
              file=sys.stderr)
        return 2
    manifest = (batchapi.Manifest.load(manifest_path) if manifest_path.exists()
                else batchapi.Manifest(batch_id=args.batch_id))
    batch_id = args.batch_id or manifest.batch_id

    if action == "cancel":
        entry = provider.cancel_batch(batch_id)
        print(f"{batch_id}: {entry.get('status')}")
        return 0

    if action == "status":
        entry = provider.get_batch(batch_id)
        counts = entry.get("request_counts") or {}
        print(f"batch    : {batch_id}")
        print(f"status   : {entry.get('status')}")
        print(f"requests : {counts.get('completed', 0)} done, "
              f"{counts.get('failed', 0)} failed, of {counts.get('total', 0)}")
        if entry.get("output_file_id"):
            print(f"output   : {entry['output_file_id']}")
        return 0

    # fetch
    if args.wait:
        while True:
            entry = provider.get_batch(batch_id)
            state = entry.get("status")
            counts = entry.get("request_counts") or {}
            print(f"  {state}: {counts.get('completed', 0)}/{counts.get('total', 0)}")
            if state in batchapi.FINISHED:
                break
            _time.sleep(max(5, args.poll_seconds))

    template = load_template(manifest.template or settings.template)
    try:
        summary = batchapi.collect(provider, manifest, template, settings, out_dir)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\nstatus    : {summary['status']}")
    print(f"documents : {summary['documents']}")
    print(f"records   : {summary['records']}   figures: {summary['figures']}")
    print(f"output    : {out_dir.resolve()}")
    for problem in summary["errors"][:5]:
        print(f"  ! {problem}", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    from .service import serve

    settings = settings_from_args(args)
    serve(host=args.host, port=args.port, out_dir=args.output,
          cache_dir=settings.cache_dir, token=args.token)
    return 0


COMMANDS = {
    "run": cmd_run,
    "sources": cmd_sources,
    "formats": cmd_formats,
    "templates": cmd_templates,
    "models": cmd_models,
    "check": cmd_check,
    "cache": cmd_cache,
    "audit": cmd_audit,
    "batch": cmd_batch,
    "serve": cmd_serve,
    "login": cmd_login,
    "logout": cmd_logout,
}


def _attach_progress(bus: EventBus) -> None:
    """Print one line per document so long runs stay legible."""
    state = {"done": 0, "total": 0}

    def handler(event):
        if event.type == JOB_STARTED:
            state["total"] = event.payload.get("total", 0)
            print(f"job {event.job_id}: {state['total']} documents via "
                  f"{event.payload.get('source')} -> {event.payload.get('out_dir')}")
        elif event.type in (DOC_COMPLETED, DOC_FAILED, DOC_SKIPPED):
            state["done"] += 1
            mark = {DOC_COMPLETED: "ok", DOC_FAILED: "FAIL", DOC_SKIPPED: "skip"}[event.type]
            suffix = f"  {event.message}" if event.message else ""
            print(f"[{state['done']}/{state['total']}] {mark:<4} {event.doc_id}{suffix}")
        elif event.type == JOB_COMPLETED:
            print(f"job {event.job_id} finished in "
                  f"{event.payload.get('duration_s', 0)}s")

    bus.subscribe(handler, types=[JOB_STARTED, DOC_COMPLETED, DOC_FAILED,
                                  DOC_SKIPPED, JOB_COMPLETED])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Make `run` the default so `llm-extract -i docs -o out` just works.
    if not argv or (argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help", "--version")):
        argv.insert(0, "run")

    args = build_parser().parse_args(argv)
    if not args.cmd:
        build_parser().print_help()
        return 2
    try:
        return COMMANDS[args.cmd](args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except ProviderError as exc:
        # Almost always "no key yet" or "no gateway yet": point at the fix
        # instead of letting a traceback reach a non-technical user.
        print(f"error: {exc}\n\nIf you have not set this up yet, run:\n"
              f"  llm-extract login", file=sys.stderr)
        return 2
    except (TemplateError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
