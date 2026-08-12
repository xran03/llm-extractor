"""Format detection, reading and figure discovery.

Coverage is table-driven: one row per format, so adding a format to
``formats.BUILTIN`` without a reader, a MIME type or a processing path fails
here rather than in production.
"""
from __future__ import annotations

import tempfile
import unittest
import unittest.mock
import zipfile
from pathlib import Path

from llm_extractor import formats
from llm_extractor.ingest import (Document, UnsupportedFormat, discover, find_figures,
                                  load_document)
from llm_extractor.readers import CELL, READERS
from llm_extractor.formats import (SELF, decode_bytes, detect_format, looks_like_text,
                                   mime_for, sniff, supported_extensions)

from ._fakes import (PNG_BYTES, write_csv, write_docx, write_eml, write_json, write_odt,
                     write_png, write_pptx, write_rtf, write_txt, write_xlsx, write_xml)


class FormatTableTest(unittest.TestCase):
    """Invariants every registered format must satisfy."""

    def test_every_format_has_a_reader(self):
        for name in formats.FORMATS:
            with self.subTest(format=name):
                self.assertIn(name, READERS)

    def test_every_format_is_well_formed(self):
        for name, fmt in formats.FORMATS.items():
            with self.subTest(format=name):
                self.assertTrue(fmt.extensions)
                self.assertTrue(all(e.startswith(".") for e in fmt.extensions))
                self.assertTrue(fmt.description)
                self.assertIn(fmt.figures, ("none", "self", "embedded", "render"))
                self.assertIn("/", fmt.mime)

    def test_extensions_are_not_claimed_twice(self):
        seen = {}
        for name, fmt in formats.FORMATS.items():
            for ext in fmt.extensions:
                self.assertNotIn(ext, seen, f"{ext} claimed by {seen.get(ext)} and {name}")
                seen[ext] = name

    def test_images_declare_no_text_layer(self):
        for fmt in formats.FORMATS.values():
            if fmt.figures == SELF:
                self.assertFalse(fmt.text_layer, fmt.name)

    def test_required_formats_are_supported(self):
        required = {".pdf", ".xml", ".docx", ".pptx", ".xlsx", ".csv", ".png", ".jpg",
                    ".jpeg", ".txt", ".md", ".html", ".json", ".rtf", ".eml", ".odt"}
        self.assertTrue(required.issubset(set(supported_extensions())))


class DetectionTest(unittest.TestCase):
    """Content decides; the suffix only breaks ties."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_by_content(self):
        cases = [
            (b"%PDF-1.7\n", "pdf"),
            (PNG_BYTES, "png"),
            (b"\xff\xd8\xff\xe0", "jpeg"),
            (b"GIF89a", "gif"),
            (b"{\\rtf1 hello}", "rtf"),
            (b"<?xml version='1.0'?><a/>", "xml"),
            (b"<!DOCTYPE html><html></html>", "html"),
            (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "doc"),
        ]
        for payload, expected in cases:
            with self.subTest(expected=expected):
                path = self.dir / f"probe-{expected}"
                path.write_bytes(payload)
                self.assertEqual(detect_format(path).name, expected)

    def test_content_beats_a_lying_suffix(self):
        mislabelled = self.dir / "report.pdf"
        mislabelled.write_bytes(PNG_BYTES)
        self.assertEqual(detect_format(mislabelled).name, "png")

    def test_extension_less_file_is_still_identified(self):
        path = self.dir / "no_extension"
        path.write_bytes(b"%PDF-1.4\n")
        self.assertEqual(detect_format(path).name, "pdf")

    def test_ooxml_containers_are_told_apart(self):
        for writer, expected in ((write_docx, "docx"), (write_pptx, "pptx"),
                                 (write_xlsx, "xlsx")):
            with self.subTest(expected=expected):
                self.assertEqual(detect_format(writer(self.dir)).name, expected)

    def test_opendocument_is_identified_by_its_mimetype_entry(self):
        self.assertEqual(detect_format(write_odt(self.dir)).name, "odt")

    def test_suffix_decides_between_text_formats(self):
        for name, expected in (("a.md", "markdown"), ("a.csv", "csv"),
                               ("a.json", "json"), ("a.txt", "plain")):
            with self.subTest(name=name):
                path = self.dir / name
                path.write_text("group,value\na,1\n", encoding="utf-8")
                self.assertEqual(detect_format(path).name, expected)

    def test_unknown_suffix_with_text_content_reads_as_text(self):
        path = self.dir / "export.zzz"
        path.write_text("Group A reached 12.5 ug/mL.", encoding="utf-8")
        self.assertEqual(detect_format(path).name, "plain")

    def test_unrecognised_binary_is_refused(self):
        path = self.dir / "blob.zzz"
        path.write_bytes(bytes(range(256)) * 4)
        self.assertIsNone(detect_format(path))
        with self.assertRaises(UnsupportedFormat):
            load_document(path)

    def test_missing_file_is_reported_clearly(self):
        with self.assertRaises(FileNotFoundError):
            load_document(self.dir / "nope.txt")

    def test_sniff_is_undecided_on_ambiguous_bytes(self):
        self.assertEqual(sniff(b"group,value\n"), "")

    def test_looks_like_text_rejects_binary(self):
        self.assertTrue(looks_like_text(b"plain words"))
        self.assertFalse(looks_like_text(b"\x00\x01\x02"))

    def test_mime_for_images(self):
        self.assertEqual(mime_for("a.png"), "image/png")
        self.assertEqual(mime_for("a.JPG"), "image/jpeg")
        self.assertEqual(mime_for("a.unknown"), "image/png")


class DecodingTest(unittest.TestCase):
    def test_encodings_round_trip(self):
        cases = [
            ("utf-8", "Group A 12.5 µg/mL".encode("utf-8")),
            ("utf-8-sig", "\ufeffGroup A".encode("utf-8")),
            ("utf-16", "Group A".encode("utf-16")),
            ("cp1252", "Groupe A 12,5 µg".encode("cp1252")),
        ]
        for label, payload in cases:
            with self.subTest(encoding=label):
                self.assertIn("Group", decode_bytes(payload).replace("Groupe", "Group"))

    def test_undecodable_bytes_never_raise(self):
        self.assertIsInstance(decode_bytes(b"\xff\xfe\xfa\x00 bad"), str)


class ReaderTest(unittest.TestCase):
    """Each reader must surface the content that carries the data."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_readers_surface_expected_content(self):
        cases = [
            (write_txt, ["12.5 ug/mL"]),
            (write_xml, ["Study title", "Group B reached 4.0 ug/mL"]),
            (write_docx, ["Group A reached 12.5 ug/mL"]),
            (write_pptx, ["--- slide 1 ---", "Group A reached 12.5 ug/mL"]),
            (write_xlsx, ["--- sheet: Measurements ---", "group A", "12.5"]),
            (write_odt, ["Group A reached 12.5 ug/mL."]),
            (write_csv, ["group A", "12.5"]),
            (write_json, ["group A", "12.5"]),
            (write_rtf, ["Group A reached 12.5 ug/mL."]),
            (write_eml, ["Subject: Results", "12.5 ug/mL"]),
        ]
        for writer, expected in cases:
            with self.subTest(writer=writer.__name__):
                text = load_document(writer(self.dir)).text
                for fragment in expected:
                    self.assertIn(fragment, text)

    def test_tables_keep_their_rows_intact(self):
        """A value must stay on the same line as its row label."""
        for writer in (write_xlsx, write_odt, write_csv):
            with self.subTest(writer=writer.__name__):
                text = load_document(writer(self.dir)).text
                data_rows = [line for line in text.splitlines() if CELL in line]
                self.assertTrue(data_rows, "no delimited rows produced")
                self.assertTrue(any("12.5" in line and "group a" in line.lower()
                                    for line in data_rows), data_rows)

    def test_xlsx_reads_shared_and_inline_strings(self):
        text = load_document(write_xlsx(self.dir)).text
        self.assertIn("group A | 12.5", text)
        self.assertIn("group B | 4.0", text)

    def test_docx_tables_are_read_as_rows(self):
        path = write_docx(self.dir, "with_table.docx", with_table=True)
        rows = [line for line in load_document(path).text.splitlines() if CELL in line]
        self.assertTrue(any("group A" in r and "12.5" in r for r in rows), rows)

    def test_pptx_includes_speaker_notes(self):
        path = write_pptx(self.dir, "notes.pptx", notes="Raw value was 12.5 ug/mL")
        self.assertIn("[speaker notes] Raw value was 12.5 ug/mL",
                      load_document(path).text)

    def test_html_drops_scripts_and_keeps_content(self):
        path = self.dir / "page.html"
        path.write_text("<html><script>bad()</script><p>Value 9 ug/mL</p></html>",
                        encoding="utf-8")
        text = load_document(path).text
        self.assertIn("Value 9 ug/mL", text)
        self.assertNotIn("bad()", text)

    def test_malformed_xml_degrades_instead_of_raising(self):
        path = self.dir / "broken.xml"
        path.write_text("<a><b>text without closing", encoding="utf-8")
        self.assertIn("text without closing", load_document(path).text)

    def test_images_have_no_text_and_route_to_vision(self):
        document = load_document(write_png(self.dir))
        self.assertEqual(document.text, "")
        self.assertTrue(document.is_image_only)
        self.assertFalse(document.has_text_layer)
        self.assertEqual(len(document.figures), 1)

    def test_corrupt_container_raises_a_clear_error(self):
        for name in ("bad.docx", "bad.xlsx", "bad.pptx"):
            with self.subTest(name=name):
                path = self.dir / name
                path.write_bytes(b"PK\x03\x04 not really a zip")
                with self.assertRaises((RuntimeError, UnsupportedFormat)):
                    load_document(path)

    def test_an_unreadable_pdf_falls_back_to_its_pages(self):
        """A damaged text layer is not a lost document when it can be looked at."""
        path = self.dir / "broken.pdf"
        path.write_bytes(b"%PDF-1.4\ntruncated")
        with self.assertRaises(UnsupportedFormat):
            load_document(path)          # no renderer available -> no figures

        with unittest.mock.patch("llm_extractor.ingest.find_figures",
                                 return_value=[write_png(self.dir, "page1.png")]):
            document = load_document(path)
        self.assertEqual(document.text, "")
        self.assertTrue(document.read_error)
        self.assertFalse(document.has_text_layer)
        self.assertEqual(len(document.figures), 1)

    def test_document_exposes_its_format(self):
        document = load_document(write_docx(self.dir))
        self.assertEqual(document.format_name, "docx")
        self.assertEqual(document.media_type, formats.DOCUMENT)
        self.assertTrue(document.has_text_layer)


class FigureDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cache = self.dir / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_image_is_its_own_figure(self):
        path = write_png(self.dir)
        self.assertEqual(find_figures(path), [path])

    def test_embedded_media_is_unpacked(self):
        figures = find_figures(write_pptx(self.dir), cache_dir=str(self.cache))
        self.assertEqual(len(figures), 1)
        self.assertTrue(figures[0].exists())

    def test_docx_media_is_unpacked(self):
        path = write_docx(self.dir, "withimage.docx", with_image=True)
        figures = find_figures(path, cache_dir=str(self.cache))
        self.assertEqual(len(figures), 1)

    def test_sibling_files_directory_is_used(self):
        doc = write_txt(self.dir, "report.txt")
        sibling = self.dir / "report_files"
        sibling.mkdir()
        write_png(sibling, "f1.png")
        write_png(sibling, "f2.png")
        self.assertEqual(len(find_figures(doc)), 2)

    def test_max_figures_is_enforced(self):
        doc = write_txt(self.dir, "report.txt")
        sibling = self.dir / "report_files"
        sibling.mkdir()
        for i in range(5):
            write_png(sibling, f"f{i}.png")
        self.assertEqual(len(find_figures(doc, max_figures=2)), 2)

    def test_formats_without_figures_yield_none(self):
        self.assertEqual(find_figures(write_csv(self.dir)), [])


class DiscoverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        write_txt(self.dir, "a.txt")
        write_xml(self.dir, "b.xml")
        nested = self.dir / "nested"
        nested.mkdir()
        write_docx(nested, "c.docx")
        write_png(nested, "d.png")

    def tearDown(self):
        self.tmp.cleanup()

    def _names(self, **kwargs):
        return {p.name for p in discover(self.dir, **kwargs)}

    def test_discovers_every_readable_file_recursively(self):
        self.assertEqual(self._names(), {"a.txt", "b.xml", "c.docx", "d.png"})

    def test_binary_noise_is_skipped(self):
        (self.dir / "archive.zip").write_bytes(b"PK\x03\x04junk")
        (self.dir / "lib.dll").write_bytes(b"\x00\x01\x02")
        self.assertEqual(self._names(), {"a.txt", "b.xml", "c.docx", "d.png"})

    def test_hidden_files_are_skipped(self):
        (self.dir / ".env").write_text("SECRET=1", encoding="utf-8")
        self.assertNotIn(".env", self._names())

    def test_extension_filter(self):
        self.assertEqual(self._names(extensions=[".xml", ".docx"]), {"b.xml", "c.docx"})

    def test_extension_filter_accepts_bare_names(self):
        self.assertEqual(self._names(extensions=["xml"]), {"b.xml"})

    def test_mislabelled_file_is_still_discovered(self):
        (self.dir / "scan.pdf").write_bytes(PNG_BYTES)
        self.assertIn("scan.pdf", self._names())

    def test_single_file_input(self):
        self.assertEqual(len(discover(self.dir / "a.txt")), 1)


if __name__ == "__main__":
    unittest.main()
