"""HTTP API — the service surface for a frontend.

Deliberately built on ``http.server`` so the package keeps zero runtime
dependencies: installing ``llm-extractor`` is enough to serve an API. The
handler is a thin translation layer; all behaviour lives in
:mod:`llm_extractor.runner`, :mod:`llm_extractor.audit` and the registries, so
the API automatically exposes any newly installed source or provider plugin.

Endpoints
---------
``GET  /health``                    liveness
``GET  /v1/capabilities``           providers, sources (+ their parameters), templates
``GET  /v1/templates/{name}``       full template incl. its JSON schema
``POST /v1/jobs``                   start a job (returns 202 + job id)
``GET  /v1/jobs``                   list jobs
``GET  /v1/jobs/{id}``              job status and counters
``GET  /v1/jobs/{id}/tasks``        per-document task rows
``GET  /v1/jobs/{id}/events``       Server-Sent Events progress stream
``GET  /v1/documents/{doc_id}``     aggregated document JSON
``GET  /v1/cache``                  cache statistics
``POST /v1/cache/audit``            sample + revalidate cached extractions
``DELETE /v1/cache``                clear the cache

Auth is an optional shared bearer token (``LLM_EXTRACTOR_SERVICE_TOKEN``).
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..audit import audit_cache
from ..bus import JOB_COMPLETED, JOB_FAILED, Event, EventBus
from ..cache import ResponseCache
from ..credentials import get_env
from ..jobstore import JobStore
from ..providers import BACKENDS, build_provider
from ..runner import run_job
from ..settings import build_settings
from ..sources import SOURCES
from ..templates import (BUILTIN_TEMPLATES, STARTER_TEMPLATE, TemplateError,
                         load_template)

#: Keep-alive cadence and hard cap for a single SSE connection.
SSE_HEARTBEAT_SECONDS = 10.0
SSE_MAX_SECONDS = 3600.0
TERMINAL_JOB_STATUSES = {"ok", "error", "cancelled"}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class AppState:
    """Everything the handler needs, created once per server."""

    def __init__(self, out_dir: str = "out", cache_dir: str = ".llm_cache",
                 token: str = "", default_settings=None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.token = token or get_env("LLM_EXTRACTOR_SERVICE_TOKEN")
        self.bus = EventBus(history=5000)
        self.store = JobStore(str(Path(cache_dir) / "jobs.sqlite3"))
        self.cache = ResponseCache(cache_dir)
        self.default_settings = default_settings
        self._threads: dict = {}

    # ------------------------------- jobs ----------------------------------
    def start_job(self, body: dict) -> dict:
        source = body.get("source", "folder")
        if source not in SOURCES:
            raise ApiError(400, f"unknown source '{source}'; available: {SOURCES.names()}")

        # A template may be a built-in name, a path, or an inline JSON object,
        # so a frontend can define its own schema without shipping a file.
        template_spec = body.get("template")
        try:
            template = load_template(template_spec)
        except TemplateError as exc:
            raise ApiError(400, str(exc)) from exc

        settings = build_settings(
            api=body.get("api"),
            model=body.get("model"),
            ocr_model=body.get("ocr_model"),
            agent_model=body.get("agent_model"),
            cache_dir=self.cache_dir,
            cache_enabled=body.get("cache", True),
            template=template,
            ocr=body.get("ocr"),
            aggregate=body.get("aggregate"),
            output_format=body.get("format"),
            max_workers=body.get("workers"),
        )

        out_dir = Path(body.get("out_dir") or (self.out_dir / "jobs"))
        job_id = self.store.create_job(source, {"pending": True})
        params = dict(body.get("params") or {})

        def _run():
            try:
                run_job(settings, source_name=source, source_params=params,
                        out_dir=str(out_dir / job_id), bus=self.bus,
                        store=self.store, job_id=job_id,
                        resume=body.get("resume", True),
                        rate_limit=int(body.get("rate_limit") or 0))
            except Exception as exc:  # pragma: no cover - defensive
                message = f"{type(exc).__name__}: {exc}"
                self.store.update_job(job_id, status="error", error=message)
                # Always emit a terminal event, otherwise SSE clients wait forever.
                self.bus.publish(Event(type=JOB_FAILED, job_id=job_id, message=message))

        # A job id is reserved synchronously so the client can poll immediately,
        # then the run itself proceeds on a background thread.
        thread = threading.Thread(target=_run, name=f"job-{job_id}", daemon=True)
        self._threads[job_id] = thread
        thread.start()
        return {"job_id": job_id, "status": "accepted",
                "events": f"/v1/jobs/{job_id}/events",
                "out_dir": str(out_dir / job_id)}

    def find_document(self, doc_id: str):
        matches = sorted(self.out_dir.rglob(f"{doc_id}.document.json"))
        if not matches:
            raise ApiError(404, f"no document artifact for '{doc_id}'")
        return json.loads(matches[-1].read_text(encoding="utf-8"))

    def run_audit(self, body: dict) -> dict:
        settings = build_settings(api=body.get("api"), model=body.get("model"),
                                  cache_dir=self.cache_dir)
        provider = build_provider(settings, cache=self.cache, bus=self.bus,
                                  bypass_cache=True)
        report = audit_cache(
            self.cache, provider,
            n=int(body.get("n", 20)),
            stage=body.get("stage", ""),
            strategy=body.get("strategy", "random"),
            seed=body.get("seed"),
            referee_model=body.get("referee_model", ""),
            max_workers=int(body.get("workers", 4)),
            invalidate_drifted=bool(body.get("invalidate_drifted")),
            only_unverified=bool(body.get("only_unverified")),
            bus=self.bus,
        )
        return report.to_dict()

    def capabilities(self) -> dict:
        return {
            "providers": sorted(BACKENDS),
            "sources": {
                name: {
                    "description": getattr(cls, "description", ""),
                    "parameters": getattr(cls, "parameters", {}),
                }
                for name, cls in SOURCES.items().items()
            },
            "templates": {
                name: template.description
                for name, template in BUILTIN_TEMPLATES.items()
            },
        }


class Handler(BaseHTTPRequestHandler):
    server_version = "llm-extractor/0.1"
    state: AppState = None  # injected by make_server

    # ----------------------------- plumbing --------------------------------
    def log_message(self, fmt, *args):  # keep the console clean
        pass

    def _authorized(self) -> bool:
        if not self.state.token:
            return True
        header = self.headers.get("Authorization", "")
        return header == f"Bearer {self.state.token}"

    def _send(self, status: int, payload, content_type="application/json"):
        body = (json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
                if content_type == "application/json" else payload.encode("utf-8"))
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ApiError(400, f"invalid JSON body: {exc}") from exc

    def do_OPTIONS(self):  # noqa: N802 - CORS preflight
        self._send(204, {})

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str):
        parsed = urlparse(self.path)
        parts = [p for p in parsed.path.strip("/").split("/") if p]
        query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        try:
            if parts and parts[0] != "health" and not self._authorized():
                raise ApiError(401, "missing or invalid bearer token")
            self._route(method, parts, query)
        except ApiError as exc:
            self._send(exc.status, {"error": exc.message})
        except BrokenPipeError:  # client disconnected from an SSE stream
            pass
        except Exception as exc:  # pragma: no cover - defensive
            self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc(limit=3)})

    # ------------------------------ routes ---------------------------------
    def _route(self, method: str, parts: list, query: dict):
        state = self.state

        if not parts or parts == ["health"]:
            return self._send(200, {"status": "ok", "service": "llm-extractor"})

        if parts[0] != "v1":
            raise ApiError(404, f"unknown path: /{'/'.join(parts)}")
        rest = parts[1:]

        if method == "GET" and rest == ["capabilities"]:
            return self._send(200, state.capabilities())

        if rest and rest[0] == "templates":
            if method == "GET" and len(rest) == 1:
                return self._send(200, {
                    "templates": {n: t.description for n, t in BUILTIN_TEMPLATES.items()},
                    "starter": STARTER_TEMPLATE,
                })
            if method == "POST" and rest[1:] == ["validate"]:
                # Lets a frontend check a user-authored schema before spending tokens.
                try:
                    template = load_template(self._body().get("template"))
                except TemplateError as exc:
                    return self._send(200, {"valid": False, "error": str(exc)})
                return self._send(200, {"valid": True, "name": template.name,
                                        "fields": template.field_names,
                                        "json_schema": template.json_schema()})
            if method == "GET" and len(rest) == 2:
                try:
                    template = load_template(rest[1])
                except TemplateError as exc:
                    raise ApiError(404, str(exc)) from exc
                return self._send(200, {**template.to_dict(),
                                        "json_schema": template.json_schema()})

        if rest and rest[0] == "jobs":
            return self._jobs(method, rest[1:], query)

        if method == "GET" and len(rest) == 2 and rest[0] == "documents":
            return self._send(200, state.find_document(rest[1]))

        if rest and rest[0] == "cache":
            if method == "GET" and len(rest) == 1:
                return self._send(200, state.cache.summary())
            if method == "DELETE" and len(rest) == 1:
                return self._send(200, {"removed": state.cache.clear()})
            if method == "GET" and rest[1:] == ["entries"]:
                return self._send(200, {"entries": state.cache.query(
                    stage=query.get("stage", ""), doc_id=query.get("doc_id", ""),
                    verdict=query.get("verdict", ""),
                    limit=int(query.get("limit", 100)))})
            if method == "POST" and rest[1:] == ["audit"]:
                return self._send(200, state.run_audit(self._body()))

        raise ApiError(404, f"unknown path: {method} /{'/'.join(parts)}")

    def _jobs(self, method: str, rest: list, query: dict):
        state = self.state
        if method == "POST" and not rest:
            return self._send(202, state.start_job(self._body()))
        if method == "GET" and not rest:
            return self._send(200, {"jobs": [
                j.to_dict() for j in state.store.list_jobs(
                    status=query.get("status", ""), limit=int(query.get("limit", 50)))
            ]})
        if not rest:
            raise ApiError(405, f"{method} not allowed on /v1/jobs")

        job_id = rest[0]
        job = state.store.get_job(job_id)
        if job is None:
            raise ApiError(404, f"unknown job '{job_id}'")

        if method == "GET" and len(rest) == 1:
            return self._send(200, job.to_dict())
        if method == "GET" and rest[1:] == ["tasks"]:
            return self._send(200, {"tasks": state.store.list_tasks(
                job_id, status=query.get("status", ""),
                limit=int(query.get("limit", 1000)))})
        if method == "GET" and rest[1:] == ["events"]:
            return self._stream_events(job_id, replay=query.get("replay", "1") != "0")
        raise ApiError(404, f"unknown job path: {method} /v1/jobs/{'/'.join(rest)}")

    def _stream_events(self, job_id: str, replay: bool = True):
        """Server-Sent Events: the frontend's live progress channel.

        A keep-alive comment is sent while idle so proxies do not drop the
        connection, and the stream ends on the job's terminal event or after
        ``SSE_MAX_SECONDS`` — a client is never left hanging forever.
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        job = self.state.store.get_job(job_id)
        already_finished = job is not None and job.status in TERMINAL_JOB_STATUSES

        events, close = self.state.bus.stream(replay=replay, timeout=SSE_HEARTBEAT_SECONDS)
        deadline = time.monotonic() + SSE_MAX_SECONDS
        try:
            for event in events:
                if time.monotonic() > deadline:
                    break
                if event is None:  # idle heartbeat
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    if already_finished:
                        break
                    continue
                if job_id and event.job_id and event.job_id != job_id:
                    continue
                payload = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
                self.wfile.write(f"id: {event.seq}\nevent: {event.type}\n"
                                 f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                if event.type in (JOB_COMPLETED, JOB_FAILED) and event.job_id == job_id:
                    break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            close()


def make_server(host: str = "127.0.0.1", port: int = 8080, out_dir: str = "out",
                cache_dir: str = ".llm_cache", token: str = ""):
    """Build a configured (not yet serving) HTTP server."""
    state = AppState(out_dir=out_dir, cache_dir=cache_dir, token=token)
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.state = state
    return server


def serve(host: str = "127.0.0.1", port: int = 8080, out_dir: str = "out",
          cache_dir: str = ".llm_cache", token: str = "") -> None:
    server = make_server(host, port, out_dir, cache_dir, token)
    print(f"llm-extractor API on http://{host}:{port}  (out={out_dir}, cache={cache_dir})")
    print("  GET  /v1/capabilities    POST /v1/jobs    GET /v1/jobs/{id}/events")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
