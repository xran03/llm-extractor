"""End-to-end: the real CLI, real HTTP transport, against a fake API gateway.

Everything except the model itself is production code here — argument parsing,
credential resolution, the folder source, format readers, both provider wire
formats, the cache, the scheduler and artifact writing. This is the test that
would catch a break in the path a user actually exercises.
"""
from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from llm_extractor import cli
from llm_extractor.serialize import read_csv

from ._fakes import DEFAULT_AGGREGATE, DEFAULT_OCR, DEFAULT_RECORDS, write_docx, write_png, write_pptx, write_txt, write_xml


class FakeGatewayHandler(BaseHTTPRequestHandler):
    """Implements just enough of both API styles to answer the pipeline."""

    requests: list = []

    def log_message(self, fmt, *args):
        pass

    def _json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/v1/models":
            return self._json(200, {"data": [{"id": "fake-model"}]})
        self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        self.requests.append({"path": self.path,
                              "auth": self.headers.get("Authorization"),
                              "payload": payload})

        answer = json.dumps(_answer_for(self.path, payload))
        if self.path == "/v1/chat/completions":
            return self._json(200, {
                "choices": [{"message": {"content": answer}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 60},
            })
        if self.path == "/v1/responses":
            return self._json(200, {
                "output_text": answer,
                "usage": {"input_tokens": 120, "output_tokens": 60},
            })
        self._json(404, {"error": "not found"})


def _answer_for(path: str, payload: dict) -> object:
    """Pick a reply based on the JSON schema the pipeline asked for."""
    if path == "/v1/responses":
        schema_name = ((payload.get("text") or {}).get("format") or {}).get("name", "")
    else:
        schema_name = ((payload.get("response_format") or {}).get("json_schema")
                       or {}).get("name", "")
    if schema_name == "figure_ocr":
        return DEFAULT_OCR
    if schema_name == "document_aggregate":
        return DEFAULT_AGGREGATE
    return {"records": DEFAULT_RECORDS}


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        FakeGatewayHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGatewayHandler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"

        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.docs = self.dir / "docs"
        self.docs.mkdir()
        write_txt(self.docs, "report.txt")
        write_xml(self.docs, "article.xml")
        write_docx(self.docs, "memo.docx")
        write_pptx(self.docs, "deck.pptx")
        write_png(self.docs, "figure.png")
        self.out = self.dir / "out"
        self.cache = self.dir / "cache"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def _run(self, *extra):
        argv = ["run", "-i", str(self.docs), "-o", str(self.out),
                "--base-url", self.base_url, "--api-key", "test-key",
                "--model", "fake-model", "--cache-dir", str(self.cache), *extra]
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(argv)
        return code, buffer.getvalue()

    # ------------------------------------------------------------------
    def test_folder_of_mixed_formats_extracts_end_to_end(self):
        code, output = self._run("--api", "llmhub")
        self.assertEqual(code, 0)
        self.assertIn("5 ok", output)

    def test_all_three_artifacts_are_written_per_document(self):
        self._run("--api", "llmhub")
        self.assertTrue((self.out / "report.records.jsonl").exists())
        self.assertTrue((self.out / "report.document.json").exists())
        self.assertTrue((self.out / "figure.ocr.json").exists())
        self.assertTrue((self.out / "summary.json").exists())

    def test_combined_csv_is_the_headline_table(self):
        self._run("--api", "llmhub")
        rows = read_csv(self.out / "records.csv")
        # 5 documents; the image-only one yields no text records.
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["unit"], "µg/mL")
        self.assertIn("doc_id", rows[0])

    def test_per_document_csv_is_written(self):
        self._run("--api", "llmhub")
        rows = read_csv(self.out / "report.records.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["subject"], "group A")

    def test_figures_csv_captures_vision_readings(self):
        self._run("--api", "llmhub", "--ocr", "always")
        rows = read_csv(self.out / "figures.csv")
        self.assertTrue(rows)
        self.assertIn("group C", {r["label"] for r in rows})

    def test_csv_only_format_skips_jsonl(self):
        self._run("--api", "llmhub", "--format", "csv")
        self.assertTrue((self.out / "report.records.csv").exists())
        self.assertFalse((self.out / "report.records.jsonl").exists())

    def test_jsonl_only_format_skips_csv(self):
        self._run("--api", "llmhub", "--format", "jsonl")
        self.assertTrue((self.out / "report.records.jsonl").exists())
        self.assertFalse((self.out / "report.records.csv").exists())
        self.assertFalse((self.out / "records.csv").exists())

    def test_csv_columns_follow_the_template(self):
        self._run("--api", "llmhub", "--template", "immunogenicity")
        header = (self.out / "records.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertIn("assay", header)
        self.assertIn("_value_grounded", header)

    def test_custom_schema_file_drives_the_csv_columns(self):
        template_path = self.dir / "custom.json"
        template_path.write_text(json.dumps({
            "name": "custom_demo",
            "instructions": "Extract facts.",
            "key_fields": ["subject"],
            "fields": [
                {"name": "subject", "type": "string", "description": "who"},
                {"name": "value", "type": "number", "description": "how much"},
                {"name": "source_span", "type": "string", "description": "evidence"},
            ],
        }), encoding="utf-8")
        code, _ = self._run("--api", "llmhub", "--template", str(template_path))
        self.assertEqual(code, 0)
        header = (self.out / "records.csv").read_text(encoding="utf-8-sig").splitlines()[0]
        self.assertEqual(header.strip().split(","),
                         ["doc_id", "doc_title", "subject", "value", "source_span",
                          "_grounded", "_value_grounded", "_unit_grounded",
                          "_ungrounded"])

    def test_document_json_has_records_ocr_and_aggregate(self):
        self._run("--api", "llmhub", "--ocr", "always")
        payload = json.loads((self.out / "deck.document.json").read_text(encoding="utf-8"))
        self.assertEqual(len(payload["records"]), 2)
        self.assertTrue(payload["figures"])
        self.assertIn("summary", payload["aggregate"])
        self.assertEqual(payload["stats"]["api"], "llmhub")

    def test_records_are_normalized_and_grounded(self):
        self._run("--api", "llmhub")
        line = (self.out / "report.records.jsonl").read_text(encoding="utf-8").splitlines()[0]
        record = json.loads(line)
        self.assertEqual(record["unit"], "µg/mL")
        self.assertTrue(record["_grounded"])
        self.assertTrue(record["_value_grounded"])

    def test_llmhub_backend_calls_chat_completions(self):
        self._run("--api", "llmhub")
        paths = {r["path"] for r in FakeGatewayHandler.requests}
        self.assertEqual(paths, {"/v1/chat/completions"})

    def test_aimodelhub_backend_calls_the_responses_endpoint(self):
        self._run("--api", "aimodelhub")
        paths = {r["path"] for r in FakeGatewayHandler.requests}
        self.assertEqual(paths, {"/v1/responses"})

    def test_both_backends_produce_the_same_records(self):
        self._run("--api", "llmhub")
        first = (self.out / "report.records.jsonl").read_text(encoding="utf-8")
        second_out = self.dir / "out2"
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli.main(["run", "-i", str(self.docs), "-o", str(second_out),
                      "--api", "aimodelhub", "--base-url", self.base_url,
                      "--api-key", "k", "--model", "fake-model",
                      "--cache-dir", str(self.dir / "cache2")])
        second = (second_out / "report.records.jsonl").read_text(encoding="utf-8")
        self.assertEqual(json.loads(first.splitlines()[0])["value"],
                         json.loads(second.splitlines()[0])["value"])

    def test_bearer_token_is_sent(self):
        self._run("--api", "llmhub")
        self.assertEqual(FakeGatewayHandler.requests[0]["auth"], "Bearer test-key")

    def test_json_schema_is_requested_from_the_gateway(self):
        self._run("--api", "llmhub")
        extract = [r for r in FakeGatewayHandler.requests
                   if (r["payload"].get("response_format") or {}).get("json_schema", {})
                   .get("name") == "generic_records"]
        self.assertTrue(extract)

    def test_cache_prevents_a_second_round_of_api_calls(self):
        self._run("--api", "llmhub")
        first = len(FakeGatewayHandler.requests)
        self._run("--api", "llmhub", "--no-resume", "-o", str(self.dir / "out3"))
        self.assertEqual(len(FakeGatewayHandler.requests), first)

    def test_no_cache_flag_re_issues_the_calls(self):
        self._run("--api", "llmhub", "--no-cache")
        first = len(FakeGatewayHandler.requests)
        self._run("--api", "llmhub", "--no-cache", "--no-resume",
                  "-o", str(self.dir / "out4"))
        self.assertGreater(len(FakeGatewayHandler.requests), first)

    def test_resume_skips_unchanged_documents_on_a_second_run(self):
        self._run("--api", "llmhub")
        _, output = self._run("--api", "llmhub")
        self.assertIn("5 skipped", output)

    def test_extension_filter_limits_the_run(self):
        code, output = self._run("--api", "llmhub", "--extensions", ".xml")
        self.assertEqual(code, 0)
        self.assertIn("1 ok", output)

    def test_immunogenicity_template_changes_the_requested_schema(self):
        self._run("--api", "llmhub", "--template", "immunogenicity")
        names = {(r["payload"].get("response_format") or {})
                 .get("json_schema", {}).get("name")
                 for r in FakeGatewayHandler.requests}
        self.assertIn("immunogenicity_records", names)

    def test_ocr_never_makes_no_vision_calls(self):
        self._run("--api", "llmhub", "--ocr", "never")
        ocr_calls = [r for r in FakeGatewayHandler.requests
                     if (r["payload"].get("response_format") or {})
                     .get("json_schema", {}).get("name") == "figure_ocr"]
        self.assertEqual(ocr_calls, [])

    def test_no_aggregate_skips_the_agent_call(self):
        self._run("--api", "llmhub", "--no-aggregate")
        agent_calls = [r for r in FakeGatewayHandler.requests
                       if (r["payload"].get("response_format") or {})
                       .get("json_schema", {}).get("name") == "document_aggregate"]
        self.assertEqual(agent_calls, [])

    def test_audit_replays_cached_calls_and_confirms_them(self):
        self._run("--api", "llmhub")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["audit", "--n", "3", "--seed", "1",
                             "--base-url", self.base_url, "--api-key", "k",
                             "--cache-dir", str(self.cache)])
        output = buffer.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("confirmed 3", output)
        self.assertIn("pass rate 1.0", output)

    def test_cache_stats_reflect_the_run(self):
        self._run("--api", "llmhub")
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli.main(["cache", "stats", "--cache-dir", str(self.cache)])
        stats = json.loads(buffer.getvalue())
        self.assertGreater(stats["entries"], 0)
        self.assertIn("extract", stats["by_stage"])

    def test_models_command_reaches_the_gateway(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["models", "--base-url", self.base_url, "--api-key", "k",
                             "--cache-dir", str(self.cache)])
        self.assertEqual(code, 0)
        self.assertIn("fake-model", buffer.getvalue())

    def test_check_reports_a_working_configuration(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = cli.main(["check", "--base-url", self.base_url, "--api-key", "k",
                             "--cache-dir", str(self.cache)])
        self.assertEqual(code, 0)
        self.assertIn("connectivity: OK", buffer.getvalue())

    def test_check_reports_the_execution_backend(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            cli.main(["check", "--base-url", self.base_url, "--api-key", "k",
                      "--cache-dir", str(self.cache)])
        output = buffer.getvalue()
        self.assertIn("execution :", output)
        self.assertTrue(
            "sequential" in output or "accelerated" in output, output)


if __name__ == "__main__":
    unittest.main()
