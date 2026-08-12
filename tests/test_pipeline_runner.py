"""End-to-end document pipeline and multi-document job runner (offline)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor.cache import ResponseCache
from llm_extractor.formats import FORMATS
from llm_extractor.jobstore import JobStore
from llm_extractor.pipeline import run_document, should_run_ocr, load_records
from llm_extractor.runner import run_job
from llm_extractor.settings import Settings
from llm_extractor.sources import SourceDocument
from llm_extractor.templates import load_template
from llm_extractor.ingest import Document

from ._fakes import (FakeProvider, write_docx, write_png, write_pptx, write_txt,
                     write_xml)


def make_settings(tmp: Path, **overrides) -> Settings:
    settings = Settings(api="fake", base_url="https://x", api_key="k",
                        model="m", ocr_model="v", agent_model="a",
                        cache_dir=str(tmp / "cache"), cache_enabled=False,
                        max_workers=2)
    for key, value in overrides.items():
        setattr(settings, key, value)
    return settings


class OcrPolicyTest(unittest.TestCase):
    """OCR routing is driven by the format's declared text layer, not extensions."""

    def _document(self, format_name="plain", text="text", figures=("a.png",)):
        return Document(doc_id="d", text=text, figures=list(figures),
                        fmt=FORMATS[format_name])

    def test_never_disables_ocr(self):
        self.assertFalse(should_run_ocr(self._document(), "never"))

    def test_always_enables_ocr_when_figures_exist(self):
        self.assertTrue(should_run_ocr(self._document(), "always"))

    def test_no_figures_means_no_ocr(self):
        self.assertFalse(should_run_ocr(self._document(figures=()), "always"))

    def test_auto_triggers_when_the_format_has_no_text_layer(self):
        self.assertTrue(should_run_ocr(self._document("png", text=""), "auto"))

    def test_auto_triggers_on_a_thin_text_layer(self):
        self.assertTrue(should_run_ocr(self._document(text="tiny"), "auto"))

    def test_auto_skips_a_healthy_text_document(self):
        self.assertFalse(should_run_ocr(self._document(text="x" * 500), "auto"))


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "out"
        self.template = load_template("generic")
        self.provider = FakeProvider()

    def tearDown(self):
        self.tmp.cleanup()

    def _source_doc(self, path):
        return SourceDocument(doc_id=Path(path).stem, path=str(path),
                              title=Path(path).stem, source_name="folder")

    def test_text_document_produces_records_and_artifacts(self):
        path = write_txt(self.dir)
        result = run_document(self.provider, self._source_doc(path),
                              make_settings(self.dir), self.template, self.out)
        self.assertEqual(len(result.records), 2)
        self.assertTrue(Path(result.artifacts["records"]).exists())
        self.assertTrue(Path(result.artifacts["document"]).exists())

    def test_document_json_contains_all_three_views(self):
        path = write_txt(self.dir)
        result = run_document(self.provider, self._source_doc(path),
                              make_settings(self.dir), self.template, self.out)
        payload = json.loads(Path(result.artifacts["document"]).read_text(encoding="utf-8"))
        self.assertIn("records", payload)
        self.assertIn("aggregate", payload)
        self.assertIn("stats", payload)

    def test_records_artifact_is_jsonl(self):
        path = write_txt(self.dir)
        result = run_document(self.provider, self._source_doc(path),
                              make_settings(self.dir), self.template, self.out)
        lines = Path(result.artifacts["records"]).read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["subject"], "group A")

    def test_image_only_document_uses_ocr(self):
        path = write_png(self.dir)
        result = run_document(self.provider, self._source_doc(path),
                              make_settings(self.dir), self.template, self.out)
        self.assertEqual(len(result.figures), 1)
        self.assertEqual(len(self.provider.stage_calls("extract")), 0)
        self.assertTrue(Path(result.artifacts["ocr"]).exists())

    def test_ocr_always_runs_both_passes(self):
        pptx = write_pptx(self.dir)
        settings = make_settings(self.dir, ocr="always")
        result = run_document(self.provider, self._source_doc(pptx), settings,
                              self.template, self.out)
        self.assertTrue(self.provider.stage_calls("extract"))
        self.assertTrue(self.provider.stage_calls("ocr"))
        self.assertTrue(result.stats["ocr_executed"])

    def test_ocr_never_skips_the_vision_model(self):
        pptx = write_pptx(self.dir)
        settings = make_settings(self.dir, ocr="never")
        run_document(self.provider, self._source_doc(pptx), settings,
                     self.template, self.out)
        self.assertEqual(self.provider.stage_calls("ocr"), [])

    def test_aggregation_can_be_disabled(self):
        path = write_txt(self.dir)
        settings = make_settings(self.dir, aggregate=False)
        result = run_document(self.provider, self._source_doc(path), settings,
                              self.template, self.out)
        self.assertEqual(self.provider.stage_calls("aggregate"), [])
        self.assertTrue(result.aggregate["summary"])

    def test_stage_failure_is_captured_not_raised(self):
        provider = FakeProvider(fail_stages={"extract"})
        path = write_txt(self.dir)
        result = run_document(provider, self._source_doc(path),
                              make_settings(self.dir), self.template, self.out)
        self.assertTrue(result.errors)
        self.assertEqual(result.records, [])

    def test_api_backed_document_without_a_file(self):
        source_doc = SourceDocument(doc_id="api1", text="Group A reached 12.5 ug/mL.",
                                    source_name="rest")
        result = run_document(self.provider, source_doc, make_settings(self.dir),
                              self.template, self.out)
        self.assertEqual(len(result.records), 2)

    def test_blob_backed_document_is_spooled_and_read(self):
        source_doc = SourceDocument(doc_id="blob1", blob=b"Group A reached 12.5 ug/mL.",
                                    media_type="txt", source_name="rest")
        result = run_document(self.provider, source_doc, make_settings(self.dir),
                              self.template, self.out)
        self.assertEqual(len(result.records), 2)

    def test_stats_capture_tokens_and_grounding(self):
        path = write_txt(self.dir)
        result = run_document(self.provider, self._source_doc(path),
                              make_settings(self.dir), self.template, self.out)
        self.assertGreater(result.stats["prompt_tokens"], 0)
        self.assertEqual(result.stats["grounded"], 2)
        self.assertEqual(result.stats["template"], "generic")

    def test_load_records_reads_a_directory(self):
        for name in ("a.txt", "b.txt"):
            run_document(self.provider, self._source_doc(write_txt(self.dir, name)),
                         make_settings(self.dir), self.template, self.out)
        self.assertEqual(len(load_records(self.out)), 4)


class RunnerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.docs = self.dir / "docs"
        self.docs.mkdir()
        write_txt(self.docs, "a.txt")
        write_xml(self.docs, "b.xml")
        write_docx(self.docs, "c.docx")
        self.out = self.dir / "out"
        self.store = JobStore(str(self.dir / "jobs.sqlite3"))
        self._patch_provider()

    def tearDown(self):
        self.store.close()
        import llm_extractor.runner as runner

        runner.build_provider = self._original
        self.tmp.cleanup()

    def _patch_provider(self):
        import llm_extractor.runner as runner

        self._original = runner.build_provider
        self.provider = FakeProvider()
        runner.build_provider = lambda settings, **kwargs: self.provider

    def _run(self, **kwargs):
        return run_job(make_settings(self.dir), source_name="folder",
                       source_params={"input_dir": str(self.docs)},
                       out_dir=str(self.out), store=self.store, **kwargs)

    def test_every_document_is_processed(self):
        summary = self._run()
        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.ok, 3)
        self.assertEqual(summary.failed, 0)

    def test_records_are_counted(self):
        self.assertEqual(self._run().records, 6)

    def test_summary_json_is_written(self):
        self._run()
        payload = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["ok"], 3)

    def test_job_and_tasks_are_persisted(self):
        summary = self._run()
        job = self.store.get_job(summary.job_id)
        self.assertEqual(job.total, 3)
        self.assertEqual(len(self.store.list_tasks(summary.job_id)), 3)

    def test_resume_skips_unchanged_documents(self):
        self._run()
        second = self._run()
        self.assertEqual(second.skipped, 3)
        self.assertEqual(second.ok, 0)

    def test_no_resume_reprocesses_everything(self):
        self._run()
        second = self._run(resume=False)
        self.assertEqual(second.ok, 3)

    def test_a_changed_document_is_reprocessed(self):
        self._run()
        write_txt(self.docs, "a.txt", text="Group A reached 99.9 ug/mL now.")
        second = self._run()
        self.assertEqual(second.ok, 1)
        self.assertEqual(second.skipped, 2)

    def test_unknown_source_fails_the_job_cleanly(self):
        summary = run_job(make_settings(self.dir), source_name="does-not-exist",
                          out_dir=str(self.out), store=self.store)
        self.assertTrue(summary.errors)
        self.assertEqual(summary.ok, 0)

    def test_missing_input_dir_fails_the_job_cleanly(self):
        summary = run_job(make_settings(self.dir), source_name="folder",
                          source_params={"input_dir": str(self.dir / "nope")},
                          out_dir=str(self.out), store=self.store)
        self.assertTrue(summary.errors)

    def test_events_are_emitted_for_the_run(self):
        from llm_extractor.bus import EventBus

        bus = EventBus()
        seen = []
        bus.subscribe(seen.append)
        run_job(make_settings(self.dir), source_name="folder",
                source_params={"input_dir": str(self.docs)},
                out_dir=str(self.out), store=self.store, bus=bus)
        types = {e.type for e in seen}
        self.assertIn("job.started", types)
        self.assertIn("job.completed", types)

    def test_cache_reuse_across_runs_avoids_api_calls(self):
        import llm_extractor.runner as runner

        cache = ResponseCache(str(self.dir / "shared-cache"))
        inner = FakeProvider()
        from llm_extractor.cache import CachedProvider

        runner.build_provider = lambda settings, **kwargs: CachedProvider(inner, cache)
        settings = make_settings(self.dir, cache_enabled=True)
        run_job(settings, source_name="folder",
                source_params={"input_dir": str(self.docs)},
                out_dir=str(self.out), store=self.store)
        first_calls = len(inner.calls)

        second_store = JobStore(str(self.dir / "j2.sqlite3"))
        try:
            run_job(settings, source_name="folder",
                    source_params={"input_dir": str(self.docs)},
                    out_dir=str(self.dir / "out2"), store=second_store, resume=False)
        finally:
            second_store.close()
            cache.close()
        self.assertEqual(len(inner.calls), first_calls)


if __name__ == "__main__":
    unittest.main()
