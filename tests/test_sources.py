"""Source connectors: local folders and the REST base used by patent/literature APIs."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from llm_extractor.cli import source_from_args

from llm_extractor.sources import SOURCES, available_sources, build_source
from llm_extractor.sources.base import (Source, SourceConfigError, SourceDocument,
                                        load_source_config)
from llm_extractor.sources.literature import EuropePMCSource, OpenAlexSource
from llm_extractor.sources.patents import PatentSearchSource
from llm_extractor.sources.rest import (STARTER_CONNECTOR, RestSource,
                                        RestSourceError, dig, sniff_media)

from ._fakes import write_docx, write_png, write_txt, write_xml


class RegistryWiringTest(unittest.TestCase):
    def test_builtin_sources_are_registered(self):
        for name in ("folder", "rest", "patents", "europepmc", "openalex"):
            self.assertIn(name, SOURCES)

    def test_available_sources_have_descriptions(self):
        for name, description in available_sources().items():
            self.assertTrue(description, name)

    def test_a_third_party_source_can_be_registered(self):
        @SOURCES.register("unit-test-source")
        class Custom(Source):
            name = "unit-test-source"
            description = "test"

            def iter_documents(self):
                yield SourceDocument(doc_id="x", text="hello")

        try:
            documents = list(build_source("unit-test-source").iter_documents())
            self.assertEqual(documents[0].doc_id, "x")
        finally:
            SOURCES._items.pop("unit-test-source", None)


class FolderSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_txt(self.dir, "a.txt")
        write_xml(self.dir, "b.xml")
        nested = self.dir / "sub"
        nested.mkdir()
        write_docx(nested, "c.docx")
        write_png(nested, "d.png")

    def tearDown(self):
        self.tmp.cleanup()

    def test_yields_every_supported_document(self):
        documents = list(build_source("folder", input_dir=str(self.dir)).iter_documents())
        self.assertEqual(len(documents), 4)

    def test_count_matches_iteration(self):
        source = build_source("folder", input_dir=str(self.dir))
        self.assertEqual(source.count(), len(list(source.iter_documents())))

    def test_nested_documents_get_unique_ids(self):
        documents = list(build_source("folder", input_dir=str(self.dir)).iter_documents())
        ids = [d.doc_id for d in documents]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("sub__c", ids)

    def test_one_stem_in_two_formats_does_not_share_an_id(self):
        # Artifacts are named after the id, so a shared id loses a document.
        write_txt(self.dir, "figure.txt")
        write_png(self.dir, "figure.png")
        documents = list(build_source("folder", input_dir=str(self.dir)).iter_documents())
        ids = [d.doc_id for d in documents]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("figure__txt", ids)
        self.assertIn("figure__png", ids)

    def test_ids_keep_their_spelling_when_nothing_collides(self):
        documents = list(build_source("folder", input_dir=str(self.dir)).iter_documents())
        self.assertEqual(sorted(d.doc_id for d in documents),
                         ["a", "b", "sub__c", "sub__d"])

    def test_extension_filter(self):
        documents = list(build_source("folder", input_dir=str(self.dir),
                                      extensions=[".xml"]).iter_documents())
        self.assertEqual([d.doc_id for d in documents], ["b"])

    def test_limit(self):
        documents = list(build_source("folder", input_dir=str(self.dir),
                                      limit=2).iter_documents())
        self.assertEqual(len(documents), 2)

    def test_missing_directory_raises(self):
        source = build_source("folder", input_dir=str(self.dir / "nope"))
        with self.assertRaises(FileNotFoundError):
            list(source.iter_documents())

    def test_content_hash_is_stable_and_content_sensitive(self):
        source = build_source("folder", input_dir=str(self.dir), extensions=[".txt"])
        first = list(source.iter_documents())[0].content_hash()
        self.assertEqual(first, list(source.iter_documents())[0].content_hash())
        write_txt(self.dir, "a.txt", text="different content entirely")
        self.assertNotEqual(first, list(source.iter_documents())[0].content_hash())


class DigTest(unittest.TestCase):
    def test_nested_path(self):
        self.assertEqual(dig({"a": {"b": {"c": 1}}}, "a.b.c"), 1)

    def test_list_index(self):
        self.assertEqual(dig({"a": [{"b": 2}]}, "a.0.b"), 2)

    def test_missing_path_returns_default(self):
        self.assertEqual(dig({"a": 1}, "x.y", "fallback"), "fallback")

    def test_empty_path_returns_payload(self):
        self.assertEqual(dig({"a": 1}, ""), {"a": 1})


class RestSourceTest(unittest.TestCase):
    def _source(self, pages, **kwargs):
        calls = []

        def fetcher(url):
            calls.append(url)
            return pages[min(len(calls) - 1, len(pages) - 1)]

        params = {
            "base_url": "https://api.example", "path": "/search",
            "records_path": "results", "id_field": "id", "title_field": "title",
            "text_fields": ["abstract"], "uri_field": "url",
            "page_size": 2, "fetcher": fetcher,
        }
        params.update(kwargs)
        return RestSource(**params), calls

    def test_base_url_is_required(self):
        with self.assertRaises(RestSourceError):
            RestSource(base_url="")

    def test_records_are_mapped_to_documents(self):
        page = {"results": [{"id": "1", "title": "T", "abstract": "A", "url": "u"}]}
        source, _ = self._source([page])
        document = list(source.iter_documents())[0]
        self.assertEqual(document.doc_id, "1")
        self.assertEqual(document.title, "T")
        self.assertIn("A", document.text)
        self.assertEqual(document.uri, "u")

    def test_paging_stops_on_a_short_page(self):
        full = {"results": [{"id": "1"}, {"id": "2"}]}
        short = {"results": [{"id": "3"}]}
        source, calls = self._source([full, short])
        documents = list(source.iter_documents())
        self.assertEqual([d.doc_id for d in documents], ["1", "2", "3"])
        self.assertEqual(len(calls), 2)

    def test_max_records_caps_output(self):
        page = {"results": [{"id": "1"}, {"id": "2"}]}
        source, _ = self._source([page], max_records=1)
        self.assertEqual(len(list(source.iter_documents())), 1)

    def test_search_term_is_added_to_the_query(self):
        source, calls = self._source([{"results": []}], search="vaccine",
                                     query_param="q")
        list(source.iter_documents())
        self.assertIn("q=vaccine", calls[0])

    def test_paging_none_makes_a_single_request(self):
        source, calls = self._source([{"results": [{"id": "1"}, {"id": "2"}]}],
                                     paging="none")
        list(source.iter_documents())
        self.assertEqual(len(calls), 1)

    def test_cursor_paging_follows_the_cursor(self):
        first = {"results": [{"id": "1"}, {"id": "2"}], "meta": {"next": "c2"}}
        second = {"results": [{"id": "3"}], "meta": {"next": None}}
        source, calls = self._source([first, second], paging="cursor",
                                     cursor_path="meta.next", cursor_param="cursor")
        list(source.iter_documents())
        self.assertIn("cursor=c2", calls[1])

    def test_total_is_exposed_after_iteration(self):
        page = {"results": [{"id": "1"}], "total": 42}
        source, _ = self._source([page], total_path="total")
        list(source.iter_documents())
        self.assertEqual(source.count(), 42)

    def test_records_without_an_id_are_skipped(self):
        page = {"results": [{"title": "no id"}, {"id": "1"}]}
        source, _ = self._source([page, {"results": []}])
        self.assertEqual([d.doc_id for d in source.iter_documents()], ["1"])

    def test_a_repeating_page_terminates_instead_of_looping(self):
        page = {"results": [{"id": "1"}, {"id": "2"}]}
        source, calls = self._source([page])  # the fetcher always returns this page
        documents = list(source.iter_documents())
        self.assertEqual([d.doc_id for d in documents], ["1", "2"])
        self.assertEqual(len(calls), 2)

    def test_max_pages_bounds_the_walk(self):
        def fetcher(url):
            fetcher.n += 1
            return {"results": [{"id": f"{fetcher.n}-a"}, {"id": f"{fetcher.n}-b"}]}

        fetcher.n = 0
        source = RestSource(base_url="https://api.example", path="/s",
                            records_path="results", id_field="id", page_size=2,
                            max_pages=3, fetcher=fetcher)
        self.assertEqual(len(list(source.iter_documents())), 6)

    def test_query_auth_appends_the_credential(self):
        import os

        os.environ["UNIT_TEST_SRC_KEY"] = "secret123"
        try:
            source, calls = self._source([{"results": []}], auth="query",
                                         auth_env="UNIT_TEST_SRC_KEY",
                                         auth_query_param="api_key")
            list(source.iter_documents())
            self.assertIn("api_key=secret123", calls[0])
        finally:
            os.environ.pop("UNIT_TEST_SRC_KEY", None)


class ConnectorPresetTest(unittest.TestCase):
    def test_europepmc_defaults(self):
        source = EuropePMCSource(fetcher=lambda url: {"resultList": {"result": []}})
        self.assertEqual(source.records_path, "resultList.result")
        self.assertEqual(source.paging, "page")

    def test_europepmc_maps_doi_to_a_resolvable_uri(self):
        page = {"resultList": {"result": [
            {"id": "PMC1", "title": "T", "abstractText": "A", "doi": "10.1/xyz",
             "pubYear": "2024"}]}}
        source = EuropePMCSource(fetcher=lambda url: page)
        document = list(source.iter_documents())[0]
        self.assertEqual(document.uri, "https://doi.org/10.1/xyz")
        self.assertEqual(document.metadata["year"], "2024")

    def test_openalex_rebuilds_the_inverted_abstract(self):
        page = {"results": [{
            "id": "W1", "display_name": "Title",
            "abstract_inverted_index": {"Group": [0], "A": [1], "responded": [2]},
        }], "meta": {"count": 1, "next_cursor": None}}
        source = OpenAlexSource(fetcher=lambda url: page)
        document = list(source.iter_documents())[0]
        self.assertIn("Group A responded", document.text)

    def test_patent_preset_maps_bibliographic_metadata(self):
        page = {"results": [{
            "patent_id": "US123", "patent_title": "A patent",
            "patent_abstract": "abstract text", "claims": ["c1", "c2"],
            "assignee": "ACME", "filing_date": "2020-01-01",
        }], "total": 1}
        source = PatentSearchSource(base_url="https://patents.example",
                                    fetcher=lambda url: page)
        document = list(source.iter_documents())[0]
        self.assertEqual(document.doc_id, "US123")
        self.assertIn("abstract text", document.text)
        self.assertIn("c1", document.text)
        self.assertEqual(document.metadata["assignee"], "ACME")

    def test_presets_advertise_parameters_for_a_frontend(self):
        for cls in (EuropePMCSource, OpenAlexSource, PatentSearchSource):
            self.assertIn("base_url", cls.parameters)


class SourceDocumentTest(unittest.TestCase):
    def test_text_documents_hash_their_content(self):
        a = SourceDocument(doc_id="1", text="same")
        b = SourceDocument(doc_id="2", text="same")
        self.assertEqual(a.content_hash(), b.content_hash())

    def test_blob_and_text_hash_differently(self):
        self.assertNotEqual(
            SourceDocument(doc_id="1", text="x").content_hash(),
            SourceDocument(doc_id="1", blob=b"y").content_hash(),
        )

    def test_to_dict_is_serializable(self):
        data = SourceDocument(doc_id="1", text="x", metadata={"a": 1}).to_dict()
        self.assertTrue(data["has_text"])
        self.assertFalse(data["has_blob"])


class SourceConfigTest(unittest.TestCase):
    """A connector defined in a file, so it can be shared instead of retyped."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, text: str) -> Path:
        path = Path(self.tmp.name) / "connector.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_reads_a_connector_definition(self):
        config = load_source_config(self._write('{"source":"rest","base_url":"https://x"}'))
        self.assertEqual(config["base_url"], "https://x")

    def test_missing_file_is_reported(self):
        with self.assertRaises(SourceConfigError):
            load_source_config(Path(self.tmp.name) / "absent.json")

    def test_invalid_json_is_reported(self):
        with self.assertRaises(SourceConfigError):
            load_source_config(self._write("{not json"))

    def test_a_non_object_is_rejected(self):
        with self.assertRaises(SourceConfigError):
            load_source_config(self._write('["rest"]'))

    def test_the_starter_connector_builds_a_working_source(self):
        config = dict(STARTER_CONNECTOR)
        source = build_source(config.pop("source"), **config)
        self.assertEqual(source.base_url, "https://api.crossref.org")
        self.assertEqual(source.size_param, "rows")


class SourceFromArgsTest(unittest.TestCase):
    """``run`` and ``batch submit`` must resolve a source identically."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _args(self, **overrides):
        base = {"source": None, "param": [], "source_config": "", "input": "",
                "extensions": "", "exclude": [], "limit": 0}
        return argparse.Namespace(**{**base, **overrides})

    def _config(self, payload: dict) -> str:
        path = Path(self.tmp.name) / "connector.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_defaults_to_the_folder_source(self):
        name, params = source_from_args(self._args(input="docs"))
        self.assertEqual(name, "folder")
        self.assertEqual(params["input_dir"], "docs")

    def test_a_config_supplies_both_the_connector_and_its_settings(self):
        path = self._config({"source": "rest", "base_url": "https://x", "page_size": 7})
        name, params = source_from_args(self._args(source_config=path))
        self.assertEqual(name, "rest")
        self.assertEqual(params["page_size"], 7)
        self.assertNotIn("source", params)

    def test_param_overrides_the_config(self):
        path = self._config({"source": "rest", "base_url": "https://x", "page_size": 7})
        _, params = source_from_args(self._args(source_config=path, param=["page_size=50"]))
        self.assertEqual(params["page_size"], 50)

    def test_an_explicit_source_wins_over_the_config(self):
        path = self._config({"source": "rest", "base_url": "https://x"})
        name, _ = source_from_args(self._args(source="openalex", source_config=path))
        self.assertEqual(name, "openalex")

    def test_extensions_and_excludes_reach_the_folder_source(self):
        _, params = source_from_args(
            self._args(input="docs", extensions="pdf,.docx", exclude=["out"]))
        self.assertEqual(params["extensions"], [".pdf", ".docx"])
        self.assertEqual(params["exclude"], ["out"])


class QueryOrderTest(unittest.TestCase):
    """Europe PMC answers hitCount=0 unless the search term comes first."""

    def test_the_search_term_leads_the_query_string(self):
        source = EuropePMCSource(search="pneumococcal")
        url = source._build_url({"page": 1, "pageSize": 100})
        self.assertIn("?query=", url, url)

    def test_other_parameters_survive(self):
        source = EuropePMCSource(search="pneumococcal")
        url = source._build_url({"page": 2})
        for expected in ("format=json", "resultType=core", "page=2"):
            self.assertIn(expected, url)


class SniffMediaTest(unittest.TestCase):
    def test_recognises_what_it_can(self):
        self.assertEqual(sniff_media(b"%PDF-1.7 ..."), "pdf")
        self.assertEqual(sniff_media(b"<!DOCTYPE html><html>"), "html")
        self.assertEqual(sniff_media(b"  <?xml version='1.0'?><article>"), "xml")
        self.assertEqual(sniff_media(b"\n<!DOCTYPE article PUBLIC>"), "xml")

    def test_says_nothing_when_unsure(self):
        self.assertEqual(sniff_media(b"\x00\x01binary"), "")


class FullTextTest(unittest.TestCase):
    RECORD = {"id": "1", "title": "T", "abstractText": "an abstract"}

    def _source(self, fetcher, **params):
        return RestSource(base_url="https://x", id_field="id", title_field="title",
                          text_fields=["abstractText"], fulltext=True,
                          fulltext_field="pdf", fulltext_fetcher=fetcher, **params)

    def test_a_fetched_body_replaces_the_abstract(self):
        source = self._source(lambda url: b"%PDF-1.7 body")
        document = source.to_document({**self.RECORD, "pdf": "https://x/a.pdf"})
        self.assertEqual(document.blob, b"%PDF-1.7 body")
        self.assertEqual(document.media_type, "pdf")
        self.assertIn("an abstract", document.text, "abstract must survive as fallback")

    def test_a_failed_download_leaves_the_abstract_in_place(self):
        source = self._source(lambda url: None)
        document = source.to_document({**self.RECORD, "pdf": "https://x/a.pdf"})
        self.assertEqual(document.blob, b"")
        self.assertIn("an abstract", document.text)
        self.assertIn("unavailable", document.metadata["fulltext"])

    def test_a_landing_page_is_refused_rather_than_extracted(self):
        source = self._source(lambda url: b"<!DOCTYPE html><html>sign in</html>")
        document = source.to_document({**self.RECORD, "pdf": "https://x/a.pdf"})
        self.assertEqual(document.blob, b"")
        self.assertIn("returned html", document.metadata["fulltext"])

    def test_a_record_advertising_nothing_is_recorded_as_such(self):
        source = self._source(lambda url: b"%PDF")
        document = source.to_document(self.RECORD)
        self.assertEqual(document.metadata["fulltext"], "none advertised")

    def test_nothing_is_fetched_unless_asked(self):
        calls = []
        source = RestSource(base_url="https://x", id_field="id",
                            fulltext_field="pdf",
                            fulltext_fetcher=lambda url: calls.append(url))
        source.to_document({**self.RECORD, "pdf": "https://x/a.pdf"})
        self.assertEqual(calls, [])


class LiteratureFullTextTargetTest(unittest.TestCase):
    OPEN = {"pmcid": "PMC1", "isOpenAccess": "Y", "inEPMC": "Y"}

    def test_europepmc_uses_the_jats_endpoint_for_open_articles(self):
        url, media = EuropePMCSource().fulltext_target(self.OPEN)
        self.assertTrue(url.endswith("/PMC1/fullTextXML"))
        self.assertEqual(media, "xml")

    def test_europepmc_refuses_anything_not_openly_held(self):
        for missing in ("pmcid", "isOpenAccess", "inEPMC"):
            record = {**self.OPEN, missing: "N" if missing != "pmcid" else ""}
            self.assertIsNone(EuropePMCSource().fulltext_target(record), missing)

    def test_openalex_prefers_a_real_pdf_link(self):
        record = {"open_access": {"is_oa": True, "oa_url": "https://x/landing"},
                  "best_oa_location": {"pdf_url": "https://x/a.pdf"}}
        self.assertEqual(OpenAlexSource().fulltext_target(record),
                         ("https://x/a.pdf", "pdf"))

    def test_openalex_ignores_an_oa_url_that_is_not_a_pdf(self):
        record = {"open_access": {"is_oa": True, "oa_url": "https://x/landing"},
                  "best_oa_location": {}}
        self.assertIsNone(OpenAlexSource().fulltext_target(record))

    def test_openalex_refuses_closed_works(self):
        record = {"open_access": {"is_oa": False},
                  "best_oa_location": {"pdf_url": "https://x/a.pdf"}}
        self.assertIsNone(OpenAlexSource().fulltext_target(record))

    def test_an_openalex_record_without_an_abstract_still_carries_its_title(self):
        document = OpenAlexSource().to_document(
            {"id": "W1", "display_name": "A study of things"})
        self.assertIn("A study of things", document.text)


if __name__ == "__main__":
    unittest.main()
