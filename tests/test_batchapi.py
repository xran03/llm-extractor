"""Batch API: wire protocol, manifest, and asynchronous reassembly.

The hard part of batching is that answers come back hours later as one flat
stream identified only by ``custom_id``. These tests pin that down: shuffled,
partial and failed answers must still land in the right document, in order.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor import batchapi
from llm_extractor.batchapi import (Manifest, STAGE_EXTRACT, STAGE_OCR, Task,
                                    build_requests, parse_output, preflight,
                                    records_for_document, figures_for_document)
from llm_extractor.providers.aimodelhub import AIModelHubProvider
from llm_extractor.providers.base import ProviderError, encode_multipart
from llm_extractor.providers.llmhub import LLMHubProvider
from llm_extractor.settings import Settings
from llm_extractor.sources import SourceDocument
from llm_extractor.templates import BUILTIN_TEMPLATES

from ._fakes import write_png, write_txt


def make_provider(cls=AIModelHubProvider, **routes):
    """A provider whose HTTP layer is replaced by recorded routes."""
    provider = cls(name="test", base_url="https://gw", api_key="k")
    calls = []

    def request(method, path, payload=None):
        calls.append({"method": method, "path": path, "payload": payload})
        for prefix, response in routes.items():
            if path.startswith(prefix):
                if isinstance(response, Exception):
                    raise response
                return response
        return {}

    def request_raw(method, path, data=None, content_type="", accept=""):
        calls.append({"method": method, "path": path, "data": data,
                      "content_type": content_type})
        for prefix, response in routes.items():
            if path.startswith(prefix):
                if isinstance(response, Exception):
                    raise response
                if isinstance(response, bytes):
                    return response
                return json.dumps(response).encode()
        return b"{}"

    provider.request = request
    provider.request_raw = request_raw
    provider.calls = calls
    return provider


def settings_for(tmp) -> Settings:
    return Settings(api="aimodelhub", base_url="https://gw", api_key="k",
                    model="gpt-5.6-sol", ocr_model="gpt-5.6-sol",
                    agent_model="gpt-5.6-sol", cache_dir=str(Path(tmp) / "c"),
                    cache_enabled=False, aggregate=False, max_figures=5)


class MultipartTest(unittest.TestCase):
    def test_body_carries_the_field_and_the_file(self):
        body, content_type = encode_multipart({"purpose": "batch"}, "file",
                                              "requests.jsonl", b'{"a":1}')
        self.assertIn("multipart/form-data; boundary=", content_type)
        text = body.decode()
        self.assertIn('name="purpose"', text)
        self.assertIn("batch", text)
        self.assertIn('filename="requests.jsonl"', text)
        self.assertIn('{"a":1}', text)

    def test_boundary_matches_the_declared_one(self):
        body, content_type = encode_multipart({}, "file", "f.jsonl", b"x")
        boundary = content_type.split("boundary=")[1]
        self.assertTrue(body.decode().startswith(f"--{boundary}"))
        self.assertTrue(body.decode().rstrip().endswith(f"--{boundary}--"))


class ProtocolTest(unittest.TestCase):
    """Every call must be the endpoint the OpenAI batch protocol specifies."""

    def test_upload_posts_multipart_to_files(self):
        provider = make_provider(**{"/v1/files": {"id": "file-1"}})
        self.assertEqual(provider.upload_file(b"{}")["id"], "file-1")
        call = provider.calls[-1]
        self.assertEqual((call["method"], call["path"]), ("POST", "/v1/files"))
        self.assertIn("multipart/form-data", call["content_type"])

    def test_create_batch_names_the_backend_endpoint(self):
        provider = make_provider(**{"/v1/batches": {"id": "batch-1"}})
        provider.create_batch("file-1")
        payload = provider.calls[-1]["payload"]
        self.assertEqual(payload["input_file_id"], "file-1")
        self.assertEqual(payload["endpoint"], "/v1/responses")
        self.assertEqual(payload["completion_window"], "24h")

    def test_chat_backend_names_its_own_endpoint(self):
        provider = make_provider(LLMHubProvider, **{"/v1/batches": {"id": "b"}})
        provider.create_batch("file-1")
        self.assertEqual(provider.calls[-1]["payload"]["endpoint"],
                         "/v1/chat/completions")

    def test_lifecycle_paths(self):
        provider = make_provider(**{"/v1/batches": {"id": "b", "status": "completed"}})
        provider.get_batch("b")
        self.assertEqual(provider.calls[-1]["path"], "/v1/batches/b")
        provider.cancel_batch("b")
        self.assertEqual(provider.calls[-1]["path"], "/v1/batches/b/cancel")
        provider.list_batches(limit=5)
        self.assertIn("limit=5", provider.calls[-1]["path"])

    def test_download_reads_the_content_route(self):
        provider = make_provider(**{"/v1/files": b"line"})
        self.assertEqual(provider.download_file("f-1"), b"line")
        self.assertEqual(provider.calls[-1]["path"], "/v1/files/f-1/content")


class PreflightTest(unittest.TestCase):
    def test_ready_when_both_routes_answer(self):
        provider = make_provider(**{"/v1/batches": {"data": []},
                                    "/v1/files": {"id": "file-probe"}})
        report = preflight(provider)
        self.assertTrue(report["ready"])
        self.assertEqual(report["list_batches"], "ok")

    def test_upload_failure_is_reported_as_not_ready(self):
        provider = make_provider(**{"/v1/batches": {"data": []},
                                    "/v1/files": ProviderError("files_settings is not set")})
        report = preflight(provider)
        self.assertFalse(report["ready"])
        self.assertEqual(report["upload_files"], "failed")
        self.assertIn("files_settings", report["detail"])

    def test_listing_failure_stops_before_uploading(self):
        provider = make_provider(**{"/v1/batches": ProviderError("404 not found")})
        report = preflight(provider)
        self.assertEqual(report["list_batches"], "failed")
        self.assertIsNone(report["upload_files"])


class BuildRequestsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.template = BUILTIN_TEMPLATES["generic"]
        self.settings = settings_for(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def _docs(self, *paths):
        return [SourceDocument(doc_id=p.stem, path=str(p), title=p.stem,
                               source_name="folder") for p in paths]

    def test_one_request_per_document(self):
        docs = self._docs(write_txt(self.dir, "a.txt"), write_txt(self.dir, "b.txt"))
        lines, tasks = build_requests(make_provider(), docs, self.template, self.settings)
        self.assertEqual(len(lines), 2)
        self.assertEqual({t.doc_id for t in tasks}, {"a", "b"})

    def test_request_bodies_are_the_backend_wire_format(self):
        docs = self._docs(write_txt(self.dir, "a.txt"))
        lines, _ = build_requests(make_provider(), docs, self.template, self.settings)
        body = lines[0]["body"]
        self.assertIn("input", body)            # responses shape, not messages
        self.assertEqual(lines[0]["url"], "/v1/responses")
        self.assertEqual(body["model"], "gpt-5.6-sol")

    def test_the_json_schema_travels_with_each_request(self):
        docs = self._docs(write_txt(self.dir, "a.txt"))
        lines, _ = build_requests(make_provider(), docs, self.template, self.settings)
        self.assertEqual(lines[0]["body"]["text"]["format"]["name"], "generic_records")

    def test_custom_ids_are_unique(self):
        docs = self._docs(*[write_txt(self.dir, f"d{i}.txt") for i in range(5)])
        lines, tasks = build_requests(make_provider(), docs, self.template, self.settings)
        ids = [line["custom_id"] for line in lines]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [t.custom_id for t in tasks])

    def test_a_long_document_becomes_several_ordered_chunks(self):
        long_doc = write_txt(self.dir, "long.txt", text="paragraph\n\n" * 20000)
        lines, tasks = build_requests(make_provider(), self._docs(long_doc),
                                      self.template, self.settings)
        self.assertGreater(len(lines), 1)
        self.assertEqual([t.chunk for t in tasks], list(range(len(tasks))))
        self.assertTrue(all(t.n_chunks == len(tasks) for t in tasks))

    def test_figures_are_queued_only_when_asked(self):
        png = write_png(self.dir, "fig.png")
        docs = self._docs(png)
        _, without = build_requests(make_provider(), docs, self.template, self.settings)
        self.assertEqual([t.stage for t in without], [])

        _, with_ocr = build_requests(make_provider(), docs, self.template,
                                     self.settings, with_ocr=True)
        self.assertEqual([t.stage for t in with_ocr], [STAGE_OCR])
        self.assertEqual(with_ocr[0].figure, "fig.png")

    def test_ocr_requests_carry_the_image_and_its_schema(self):
        docs = self._docs(write_png(self.dir, "fig.png"))
        lines, _ = build_requests(make_provider(), docs, self.template,
                                  self.settings, with_ocr=True)
        body = lines[0]["body"]
        self.assertEqual(body["text"]["format"]["name"], "figure_ocr")
        parts = body["input"][0]["content"]
        self.assertTrue(any(p["type"] == "input_image" for p in parts))


class ManifestTest(unittest.TestCase):
    def _manifest(self):
        return Manifest(batch_id="b-1", input_file_id="f-1", api="aimodelhub",
                        model="m", template="generic", endpoint="/v1/responses",
                        tasks=[Task(custom_id="r-0", doc_id="a", chunk=0, n_chunks=2),
                               Task(custom_id="r-1", doc_id="a", chunk=1, n_chunks=2),
                               Task(custom_id="r-2", doc_id="b", stage=STAGE_OCR,
                                    figure="f.png")])

    def test_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._manifest().save(Path(tmp) / "m.json")
            loaded = Manifest.load(path)
        self.assertEqual(loaded.batch_id, "b-1")
        self.assertEqual(len(loaded.tasks), 3)
        self.assertEqual(loaded.tasks[2].figure, "f.png")

    def test_documents_keep_submission_order(self):
        self.assertEqual(self._manifest().documents(), ["a", "b"])

    def test_tasks_can_be_selected_by_document_and_stage(self):
        manifest = self._manifest()
        self.assertEqual(len(manifest.tasks_for("a")), 2)
        self.assertEqual(len(manifest.tasks_for("b", STAGE_OCR)), 1)
        self.assertEqual(manifest.tasks_for("a", STAGE_OCR), [])


class ParseOutputTest(unittest.TestCase):
    def setUp(self):
        self.provider = make_provider()

    def _line(self, custom_id, text, status=200):
        return json.dumps({
            "custom_id": custom_id,
            "response": {"status_code": status,
                         "body": {"output": [{"type": "message", "content": [
                             {"type": "output_text", "text": text}]}]}},
        })

    def test_answers_are_keyed_by_custom_id(self):
        raw = "\n".join([self._line("r-0", "first"), self._line("r-1", "second")])
        answers = parse_output(self.provider, raw.encode())
        self.assertEqual(answers["r-0"]["text"], "first")
        self.assertEqual(answers["r-1"]["text"], "second")

    def test_a_per_request_error_is_captured_not_raised(self):
        raw = json.dumps({"custom_id": "r-0", "error": {"message": "rate limited"}})
        answers = parse_output(self.provider, raw.encode())
        self.assertIn("rate limited", answers["r-0"]["error"])
        self.assertEqual(answers["r-0"]["text"], "")

    def test_an_http_error_status_is_captured(self):
        answers = parse_output(self.provider, self._line("r-0", "", 429).encode())
        self.assertIn("429", answers["r-0"]["error"])

    def test_a_string_body_is_decoded(self):
        raw = json.dumps({"custom_id": "r-0", "response": {
            "status_code": 200,
            "body": json.dumps({"output": [{"type": "message", "content": [
                {"type": "output_text", "text": "inner"}]}]})}})
        self.assertEqual(parse_output(self.provider, raw.encode())["r-0"]["text"], "inner")

    def test_blank_and_malformed_lines_are_skipped(self):
        raw = "\n".join(["", "not json", self._line("r-0", "kept"), ""])
        answers = parse_output(self.provider, raw.encode())
        self.assertEqual(list(answers), ["r-0"])


class ReassemblyTest(unittest.TestCase):
    """Answers arrive unordered; records must still come back in chunk order."""

    def setUp(self):
        self.template = BUILTIN_TEMPLATES["generic"]
        self.text = ("Group A reached 12.5 ug/mL. "
                     "Group B reached 4.0 ug/mL. "
                     "Group C reached 7.25 ug/mL.")
        self.tasks = [Task(custom_id=f"r-{i}", doc_id="d", chunk=i, n_chunks=3)
                      for i in range(3)]

    def _answer(self, subject, value, span):
        return {"error": "", "text": json.dumps({"records": [
            {"subject": subject, "attribute": "concentration", "value": value,
             "unit": "ug/mL", "source_span": span}]})}

    def test_shuffled_answers_reassemble_in_chunk_order(self):
        answers = {
            "r-2": self._answer("group C", 7.25, "Group C reached 7.25 ug/mL"),
            "r-0": self._answer("group A", 12.5, "Group A reached 12.5 ug/mL"),
            "r-1": self._answer("group B", 4.0, "Group B reached 4.0 ug/mL"),
        }
        records, errors = records_for_document(answers, self.tasks, self.template,
                                               self.text)
        self.assertEqual([r["subject"] for r in records],
                         ["group A", "group B", "group C"])
        self.assertEqual(errors, [])

    def test_records_are_grounded_against_the_reread_document(self):
        answers = {"r-0": self._answer("group A", 12.5, "Group A reached 12.5 ug/mL")}
        records, _ = records_for_document(answers, self.tasks[:1], self.template,
                                          self.text)
        self.assertTrue(records[0]["_grounded"])
        self.assertTrue(records[0]["_value_grounded"])

    def test_a_fabricated_value_is_still_caught_in_batch_mode(self):
        answers = {"r-0": self._answer("ghost", 999.0, "Group A reached 12.5 ug/mL")}
        records, _ = records_for_document(answers, self.tasks[:1], self.template,
                                          self.text)
        self.assertFalse(records[0]["_value_grounded"])

    def test_a_missing_answer_is_reported_and_the_rest_survive(self):
        answers = {"r-0": self._answer("group A", 12.5, "Group A reached 12.5 ug/mL")}
        records, errors = records_for_document(answers, self.tasks, self.template,
                                               self.text)
        self.assertEqual(len(records), 1)
        self.assertEqual(len(errors), 2)
        self.assertIn("no answer returned", errors[0])

    def test_one_failed_chunk_does_not_lose_the_others(self):
        answers = {
            "r-0": self._answer("group A", 12.5, "Group A reached 12.5 ug/mL"),
            "r-1": {"error": "HTTP 500", "text": ""},
            "r-2": self._answer("group C", 7.25, "Group C reached 7.25 ug/mL"),
        }
        records, errors = records_for_document(answers, self.tasks, self.template,
                                               self.text)
        self.assertEqual([r["subject"] for r in records], ["group A", "group C"])
        self.assertIn("HTTP 500", errors[0])

    def test_unparsable_output_is_an_error_not_a_crash(self):
        answers = {"r-0": {"error": "", "text": "not json at all"}}
        records, errors = records_for_document(answers, self.tasks[:1], self.template,
                                               self.text)
        self.assertEqual(records, [])
        self.assertIn("unparsable", errors[0])

    def test_duplicate_records_across_overlapping_chunks_collapse(self):
        same = self._answer("group A", 12.5, "Group A reached 12.5 ug/mL")
        answers = {"r-0": same, "r-1": same, "r-2": same}
        records, _ = records_for_document(answers, self.tasks, self.template, self.text)
        self.assertEqual(len(records), 1)


class FigureReassemblyTest(unittest.TestCase):
    def setUp(self):
        self.tasks = [Task(custom_id="r-0", doc_id="d", stage=STAGE_OCR, figure="a.png"),
                      Task(custom_id="r-1", doc_id="d", stage=STAGE_OCR, figure="b.png")]

    def test_figures_come_back_with_their_image_names(self):
        answers = {
            "r-0": {"error": "", "text": json.dumps({"figure_type": "chart",
                                                     "items": [{"label": "x", "value": 1}]})},
            "r-1": {"error": "", "text": json.dumps({"figure_type": "table"})},
        }
        figures, errors = figures_for_document(answers, self.tasks)
        self.assertEqual([f["image"] for f in figures], ["a.png", "b.png"])
        self.assertEqual(figures[0]["ocr"]["figure_type"], "chart")
        self.assertEqual(errors, [])

    def test_a_partial_payload_is_filled_out_to_the_schema(self):
        answers = {"r-0": {"error": "", "text": json.dumps({"figure_type": "chart"})}}
        figures, _ = figures_for_document(answers, self.tasks[:1])
        self.assertEqual(figures[0]["ocr"]["items"], [])
        self.assertEqual(figures[0]["ocr"]["tables"], [])

    def test_a_failed_figure_is_reported(self):
        answers = {"r-0": {"error": "timeout", "text": ""}}
        figures, errors = figures_for_document(answers, self.tasks[:1])
        self.assertEqual(figures, [])
        self.assertIn("timeout", errors[0])


class CollectGuardTest(unittest.TestCase):
    def test_an_unfinished_batch_is_refused(self):
        provider = make_provider(**{"/v1/batches": {"id": "b", "status": "in_progress"}})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProviderError) as ctx:
                batchapi.collect(provider, Manifest(batch_id="b"),
                                 BUILTIN_TEMPLATES["generic"], settings_for(tmp), tmp)
        self.assertIn("still in_progress", str(ctx.exception))

    def test_a_finished_batch_without_output_is_refused(self):
        provider = make_provider(**{"/v1/batches": {"id": "b", "status": "failed"}})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProviderError) as ctx:
                batchapi.collect(provider, Manifest(batch_id="b"),
                                 BUILTIN_TEMPLATES["generic"], settings_for(tmp), tmp)
        self.assertIn("no output file", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
