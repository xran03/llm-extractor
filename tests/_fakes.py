"""Deterministic fakes so the whole pipeline is testable without a network."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from llm_extractor.providers.base import Completion, Usage

DEFAULT_RECORDS = [
    {
        "subject": "group A",
        "attribute": "antibody concentration",
        "value": 12.5,
        "unit": "ug/mL",
        "direction": "higher",
        "significant": "yes",
        "p_value": "<0.01",
        "value_source": "text",
        "source_span": "Group A reached 12.5 ug/mL, higher than group B (p<0.01).",
    },
    {
        "subject": "group B",
        "attribute": "antibody concentration",
        "value": 4.0,
        "unit": "ug/mL",
        "value_source": "text",
        "source_span": "Group B reached 4.0 ug/mL at the same timepoint.",
    },
]

DEFAULT_OCR = {
    "figure_type": "chart",
    "caption": "Figure 1. Antibody concentration by group",
    "axis_x": "group",
    "axis_y": "concentration (ug/mL)",
    "items": [
        {"label": "group A", "series": None, "value": 12.5, "value_text": None,
         "unit": "ug/mL", "note": None},
        {"label": "group C", "series": None, "value": 7.25, "value_text": None,
         "unit": "ug/mL", "note": "figure only"},
    ],
    "tables": [],
    "text_blocks": ["Figure 1"],
    "notes": None,
}

DEFAULT_AGGREGATE = {
    "summary": "Group A responded more strongly than group B.",
    "key_findings": ["Group A reached 12.5 ug/mL"],
    "figure_insights": [
        {"image": "fig1.png", "finding": "group C", "value": 7.25, "unit": "ug/mL"}
    ],
    "conflicts": [],
    "coverage_gaps": [],
}

SAMPLE_TEXT = (
    "Results\n\n"
    "Group A reached 12.5 ug/mL, higher than group B (p<0.01).\n"
    "Group B reached 4.0 ug/mL at the same timepoint.\n"
    "See Figure 1 for the full distribution.\n"
)


class FakeProvider:
    """Answers per stage, records every call, and never touches the network."""

    API_STYLE = "fake"

    def __init__(self, records=None, ocr=None, aggregate=None, fail_stages=(),
                 name="fake"):
        self.name = name
        self.records = DEFAULT_RECORDS if records is None else records
        self.ocr = DEFAULT_OCR if ocr is None else ocr
        self.aggregate = DEFAULT_AGGREGATE if aggregate is None else aggregate
        self.fail_stages = set(fail_stages)
        self.calls = []

    def list_models(self):
        return ["fake-model", "fake-model-mini"]

    def complete(self, messages, model, temperature=0.0, max_tokens=None,
                 json_schema=None, meta=None, **kwargs):
        stage = (meta or {}).get("stage", "extract")
        self.calls.append({"stage": stage, "model": model, "messages": messages,
                           "json_schema": json_schema, "meta": meta})
        if stage in self.fail_stages:
            raise RuntimeError(f"fake failure in stage {stage}")
        if stage == "ocr":
            payload = json.dumps(self.ocr)
        elif stage == "aggregate":
            payload = json.dumps(self.aggregate)
        else:
            payload = json.dumps({"records": self.records})
        return Completion(text=payload,
                          usage=Usage(prompt_tokens=100, completion_tokens=50,
                                      total_tokens=150))

    def complete_text(self, messages, model, **kwargs):
        return self.complete(messages, model, **kwargs).text

    def stage_calls(self, stage):
        return [c for c in self.calls if c["stage"] == stage]


class RecordingTransport:
    """Captures the exact HTTP payload a provider would send."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, method, path, payload=None):
        self.calls.append({"method": method, "path": path, "payload": payload})
        return self.response


def write_txt(directory, name="doc.txt", text=SAMPLE_TEXT) -> Path:
    path = Path(directory) / name
    path.write_text(text, encoding="utf-8")
    return path


def write_xml(directory, name="doc.xml") -> Path:
    path = Path(directory) / name
    path.write_text(
        '<?xml version="1.0"?><article><front><title>Study title</title></front>'
        "<body><sec><p>Group A reached 12.5 ug/mL, higher than group B (p&lt;0.01).</p>"
        "<p>Group B reached 4.0 ug/mL at the same timepoint.</p></sec></body></article>",
        encoding="utf-8",
    )
    return path


def write_docx(directory, name="doc.docx", with_table=False, with_image=False) -> Path:
    """Minimal but valid OOXML word document."""
    path = Path(directory) / name
    body = (
        "<w:p><w:r><w:t>Group A reached 12.5 ug/mL, higher than group B (p&lt;0.01).</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Group B reached 4.0 ug/mL at the same timepoint.</w:t></w:r></w:p>"
    )
    if with_table:
        body += (
            "<w:tbl>"
            "<w:tr><w:tc><w:p><w:r><w:t>group</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>concentration</w:t></w:r></w:p></w:tc></w:tr>"
            "<w:tr><w:tc><w:p><w:r><w:t>group A</w:t></w:r></w:p></w:tc>"
            "<w:tc><w:p><w:r><w:t>12.5</w:t></w:r></w:p></w:tc></w:tr>"
            "</w:tbl>"
        )
    document = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", document)
        if with_image:
            zf.writestr("word/media/image1.png", PNG_BYTES)
    return path


def write_pptx(directory, name="deck.pptx", with_media=True, notes="") -> Path:
    path = Path(directory) / name
    slide = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
        ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        "<p:cSld><p:spTree>"
        "<p:sp><p:txBody><a:p><a:r><a:t>Group A reached 12.5 ug/mL</a:t></a:r></a:p></p:txBody></p:sp>"
        "</p:spTree></p:cSld></p:sld>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("ppt/slides/slide1.xml", slide)
        if notes:
            zf.writestr("ppt/notesSlides/notesSlide1.xml",
                        '<?xml version="1.0" encoding="UTF-8"?>'
                        '<p:notes xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
                        ' xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                        f"<a:p><a:r><a:t>{notes}</a:t></a:r></a:p></p:notes>")
        if with_media:
            zf.writestr("ppt/media/image1.png", PNG_BYTES)
    return path


#: Smallest valid 1x1 PNG.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


def write_png(directory, name="fig1.png") -> Path:
    path = Path(directory) / name
    path.write_bytes(PNG_BYTES)
    return path


def write_xlsx(directory, name="book.xlsx") -> Path:
    """Minimal but valid OOXML workbook: a header row, a data row, an inline string."""
    path = Path(directory) / name
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("xl/workbook.xml",
                    f'<workbook xmlns="{ns}"><sheets>'
                    f'<sheet name="Measurements" sheetId="1"/></sheets></workbook>')
        zf.writestr("xl/sharedStrings.xml",
                    f'<sst xmlns="{ns}">'
                    f"<si><t>group</t></si><si><t>concentration</t></si>"
                    f"<si><t>group A</t></si></sst>")
        zf.writestr("xl/worksheets/sheet1.xml",
                    f'<worksheet xmlns="{ns}"><sheetData>'
                    f'<row r="1"><c r="A1" t="s"><v>0</v></c>'
                    f'<c r="B1" t="s"><v>1</v></c></row>'
                    f'<row r="2"><c r="A2" t="s"><v>2</v></c>'
                    f'<c r="B2"><v>12.5</v></c></row>'
                    f'<row r="3"><c r="A3" t="inlineStr"><is><t>group B</t></is></c>'
                    f'<c r="B3"><v>4.0</v></c></row>'
                    f"</sheetData></worksheet>")
    return path


def write_odt(directory, name="doc.odt") -> Path:
    """Minimal OpenDocument text with a paragraph and a one-row table."""
    path = Path(directory) / name
    text_ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
    table_ns = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        zf.writestr("content.xml",
                    f'<document xmlns:text="{text_ns}" xmlns:table="{table_ns}">'
                    f"<text:p>Group A reached 12.5 ug/mL.</text:p>"
                    f"<table:table><table:table-row>"
                    f"<table:table-cell><text:p>group A</text:p></table:table-cell>"
                    f"<table:table-cell><text:p>12.5</text:p></table:table-cell>"
                    f"</table:table-row></table:table></document>")
    return path


def write_eml(directory, name="mail.eml") -> Path:
    path = Path(directory) / name
    path.write_text(
        "From: sender@example.com\nTo: reader@example.com\n"
        "Subject: Results\nDate: Mon, 1 Jan 2024 09:00:00 +0000\n"
        "Content-Type: text/plain; charset=utf-8\n\n"
        "Group A reached 12.5 ug/mL, higher than group B (p<0.01).\n",
        encoding="utf-8")
    return path


def write_rtf(directory, name="note.rtf") -> Path:
    path = Path(directory) / name
    path.write_text(
        r"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times;}}"
        r"\f0\fs24 Group A reached 12.5 ug/mL.\par Group B reached 4.0 ug/mL.\par}",
        encoding="ascii")
    return path


def write_csv(directory, name="table.csv") -> Path:
    path = Path(directory) / name
    path.write_text("group,concentration,unit\ngroup A,12.5,ug/mL\ngroup B,4.0,ug/mL\n",
                    encoding="utf-8")
    return path


def write_json(directory, name="data.json") -> Path:
    path = Path(directory) / name
    path.write_text('{"group": "group A", "value": 12.5, "unit": "ug/mL"}',
                    encoding="utf-8")
    return path

