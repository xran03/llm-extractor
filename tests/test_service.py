"""HTTP API contract — what a frontend can rely on."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from llm_extractor.service.app import make_server


def request(url, method="GET", body=None, token=""):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read().decode("utf-8")
            return resp.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        payload = exc.read().decode("utf-8")
        return exc.code, json.loads(payload) if payload else {}


class ServiceTestBase(unittest.TestCase):
    token = ""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.server = make_server(host="127.0.0.1", port=0,
                                  out_dir=str(self.dir / "out"),
                                  cache_dir=str(self.dir / "cache"),
                                  token=self.token)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.server.state.store.close()
        self.server.state.cache.close()
        self.tmp.cleanup()


class ServiceTest(ServiceTestBase):
    def test_health(self):
        status, payload = request(f"{self.base}/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "ok")

    def test_capabilities_lists_plugins(self):
        status, payload = request(f"{self.base}/v1/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(sorted(payload["providers"]), ["aimodelhub", "llmhub"])
        self.assertIn("folder", payload["sources"])
        self.assertIn("generic", payload["templates"])

    def test_source_parameters_are_exposed_for_a_form(self):
        _, payload = request(f"{self.base}/v1/capabilities")
        self.assertIn("input_dir", payload["sources"]["folder"]["parameters"])

    def test_template_endpoint_returns_the_json_schema(self):
        status, payload = request(f"{self.base}/v1/templates/generic")
        self.assertEqual(status, 200)
        self.assertEqual(payload["json_schema"]["name"], "generic_records")
        self.assertIn("fields", payload)

    def test_unknown_template_is_400(self):
        status, _ = request(f"{self.base}/v1/templates/nope")
        self.assertEqual(status, 404)

    def test_templates_index_includes_a_starter_schema(self):
        status, payload = request(f"{self.base}/v1/templates")
        self.assertEqual(status, 200)
        self.assertIn("generic", payload["templates"])
        self.assertIn("fields", payload["starter"])

    def test_inline_custom_schema_is_validated(self):
        template = {
            "name": "inline", "instructions": "extract",
            "fields": [
                {"name": "subject", "type": "string", "description": "s"},
                {"name": "source_span", "type": "string", "description": "e"},
            ],
        }
        status, payload = request(f"{self.base}/v1/templates/validate", "POST",
                                  {"template": template})
        self.assertEqual(status, 200)
        self.assertTrue(payload["valid"])
        self.assertEqual(payload["json_schema"]["name"], "inline_records")

    def test_invalid_inline_schema_reports_why(self):
        status, payload = request(f"{self.base}/v1/templates/validate", "POST",
                                  {"template": {"name": "bad", "fields": [
                                      {"name": "x", "type": "string", "description": "d"}]}})
        self.assertEqual(status, 200)
        self.assertFalse(payload["valid"])
        self.assertIn("source_span", payload["error"])

    def test_job_with_a_bad_inline_schema_is_rejected(self):
        status, payload = request(f"{self.base}/v1/jobs", "POST", {
            "source": "folder", "params": {"input_dir": "."},
            "template": {"name": "bad", "fields": []},
        })
        self.assertEqual(status, 400)
        self.assertIn("fields", payload["error"])

    def test_jobs_list_is_empty_initially(self):
        status, payload = request(f"{self.base}/v1/jobs")
        self.assertEqual(status, 200)
        self.assertEqual(payload["jobs"], [])

    def test_unknown_job_is_404(self):
        status, _ = request(f"{self.base}/v1/jobs/does-not-exist")
        self.assertEqual(status, 404)

    def test_unknown_source_is_rejected_with_400(self):
        status, payload = request(f"{self.base}/v1/jobs", "POST",
                                  {"source": "not-a-source"})
        self.assertEqual(status, 400)
        self.assertIn("unknown source", payload["error"])

    def test_job_submission_returns_an_id_and_event_url(self):
        status, payload = request(f"{self.base}/v1/jobs", "POST", {
            "source": "folder",
            "params": {"input_dir": str(self.dir / "missing")},
            "api": "llmhub",
        })
        self.assertEqual(status, 202)
        self.assertIn("job_id", payload)
        self.assertTrue(payload["events"].endswith("/events"))

    def test_submitted_job_becomes_queryable(self):
        _, created = request(f"{self.base}/v1/jobs", "POST", {
            "source": "folder", "params": {"input_dir": str(self.dir / "missing")}})
        status, payload = request(f"{self.base}/v1/jobs/{created['job_id']}")
        self.assertEqual(status, 200)
        self.assertIn("status", payload)
        self.assertIn("progress", payload)

    def test_job_tasks_endpoint(self):
        _, created = request(f"{self.base}/v1/jobs", "POST", {
            "source": "folder", "params": {"input_dir": str(self.dir / "missing")}})
        status, payload = request(f"{self.base}/v1/jobs/{created['job_id']}/tasks")
        self.assertEqual(status, 200)
        self.assertIn("tasks", payload)

    def test_cache_stats(self):
        status, payload = request(f"{self.base}/v1/cache")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], 0)
        self.assertIn("by_stage", payload)

    def test_cache_entries_listing(self):
        status, payload = request(f"{self.base}/v1/cache/entries?limit=5")
        self.assertEqual(status, 200)
        self.assertEqual(payload["entries"], [])

    def test_cache_clear(self):
        status, payload = request(f"{self.base}/v1/cache", "DELETE")
        self.assertEqual(status, 200)
        self.assertIn("removed", payload)

    def test_missing_document_is_404(self):
        status, _ = request(f"{self.base}/v1/documents/nope")
        self.assertEqual(status, 404)

    def test_existing_document_artifact_is_served(self):
        out = self.dir / "out"
        out.mkdir(parents=True, exist_ok=True)
        (out / "doc1.document.json").write_text(
            json.dumps({"doc_id": "doc1", "records": []}), encoding="utf-8")
        status, payload = request(f"{self.base}/v1/documents/doc1")
        self.assertEqual(status, 200)
        self.assertEqual(payload["doc_id"], "doc1")

    def test_unknown_path_is_404(self):
        status, _ = request(f"{self.base}/v1/nope")
        self.assertEqual(status, 404)

    def test_cors_preflight_is_allowed(self):
        req = urllib.request.Request(f"{self.base}/v1/capabilities", method="OPTIONS")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.status, 204)

    def test_events_stream_emits_server_sent_events(self):
        _, created = request(f"{self.base}/v1/jobs", "POST", {
            "source": "folder", "params": {"input_dir": str(self.dir / "missing")}})
        job_id = created["job_id"]
        req = urllib.request.Request(f"{self.base}/v1/jobs/{job_id}/events")
        with urllib.request.urlopen(req, timeout=10) as resp:
            self.assertEqual(resp.headers["Content-Type"], "text/event-stream")
            chunk = resp.read(200).decode("utf-8", "replace")
        self.assertIn("event:", chunk)


class ServiceAuthTest(ServiceTestBase):
    token = "s3cret"

    def test_health_is_public(self):
        status, _ = request(f"{self.base}/health")
        self.assertEqual(status, 200)

    def test_api_requires_the_token(self):
        status, payload = request(f"{self.base}/v1/capabilities")
        self.assertEqual(status, 401)
        self.assertIn("bearer", payload["error"])

    def test_valid_token_is_accepted(self):
        status, _ = request(f"{self.base}/v1/capabilities", token=self.token)
        self.assertEqual(status, 200)

    def test_wrong_token_is_rejected(self):
        status, _ = request(f"{self.base}/v1/capabilities", token="wrong")
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
