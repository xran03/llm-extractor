"""Source connectors: local folders and the REST base used by patent/literature APIs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llm_extractor.sources import SOURCES, available_sources, build_source
from llm_extractor.sources.base import Source, SourceDocument
from llm_extractor.sources.literature import EuropePMCSource, OpenAlexSource
from llm_extractor.sources.patents import PatentSearchSource
from llm_extractor.sources.rest import RestSource, RestSourceError, dig

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


if __name__ == "__main__":
    unittest.main()
