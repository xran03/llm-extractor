"""Format readers: a file in, plain text out.

One reader per format, registered by format name so :mod:`ingest` never
branches on an extension. Readers preserve the structure that carries meaning
for extraction:

* table rows stay on one line with a delimiter, because a value separated from
  its row label is unusable;
* slides, sheets and pages are labelled so a fact can be traced back;
* anything unreadable degrades to the best available text instead of raising,
  since one awkward file must not fail a batch.

Only PDF text needs a third-party package; everything else is stdlib.
"""
from __future__ import annotations

import csv
import html
import io
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from .formats import decode_bytes, read_text

READERS: dict = {}

#: Separator used whenever a table row is flattened onto one line.
CELL = " | "

# XML namespaces, by the vocabulary they belong to.
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
}


def reader(*format_names):
    def deco(fn):
        for name in format_names:
            READERS[name] = fn
        return fn

    return deco


def read(format_name: str, path) -> str:
    try:
        fn = READERS[format_name]
    except KeyError:
        raise ValueError(f"no reader for format '{format_name}'") from None
    return fn(Path(path))


def tag_of(element) -> str:
    """Local tag name, namespace stripped."""
    return element.tag.rsplit("}", 1)[-1]


def node_text(element) -> str:
    """All text under an element, whitespace-collapsed."""
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def row(cells) -> str:
    return CELL.join(cells)


# --------------------------------------------------------------------------
# Text families
# --------------------------------------------------------------------------
@reader("plain", "markdown")
def read_plain(path: Path) -> str:
    return read_text(path)


def strip_html(raw: str) -> str:
    """Drop markup, keeping block boundaries as line breaks."""
    raw = re.sub(r"(?is)<(script|style|head)[^>]*>.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<(br|/p|/div|/tr|/h[1-6]|/li)\s*/?>", "\n", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    text = re.sub(r"[ \t]+", " ", html.unescape(raw))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


@reader("html")
def read_html(path: Path) -> str:
    return strip_html(read_text(path))


@reader("xml")
def read_xml(path: Path) -> str:
    """Flatten an XML tree (JATS/PMC article markup included) to text."""
    raw = read_text(path)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return strip_html(raw)

    chunks = []
    for element in root.iter():
        if tag_of(element).lower() in ("script", "style"):
            continue
        for value in (element.text, element.tail):
            if value and value.strip():
                chunks.append(value.strip())
    return "\n".join(chunks)


@reader("json")
def read_json(path: Path) -> str:
    """Pretty-print JSON, or pass newline-delimited JSON through per record."""
    raw = read_text(path)
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        pass

    lines = []
    for line in filter(None, (line.strip() for line in raw.splitlines())):
        try:
            lines.append(json.dumps(json.loads(line), ensure_ascii=False))
        except json.JSONDecodeError:
            lines.append(line)
    return "\n".join(lines)


@reader("csv")
def read_delimited(path: Path) -> str:
    """Read delimited text, sniffing the delimiter (comma, tab, semicolon…)."""
    raw = read_text(path)
    try:
        delimiter = csv.Sniffer().sniff(raw[:8192], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = "\t" if path.suffix.lower() in (".tsv", ".tab") else ","
    return "\n".join(row((cell or "").strip() for cell in record)
                     for record in csv.reader(io.StringIO(raw), delimiter=delimiter))


@reader("rtf")
def read_rtf(path: Path) -> str:
    """Strip RTF control words. Enough for text; not a full renderer."""
    raw = decode_bytes(path.read_bytes())
    raw = re.sub(r"\\'([0-9a-fA-F]{2})",
                 lambda m: bytes([int(m.group(1), 16)]).decode("cp1252", "replace"), raw)
    raw = re.sub(r"(?s)\{\\\*.*?\}", " ", raw)       # annotations, fonts, styles
    raw = re.sub(r"\\par[d]?\b", "\n", raw)
    raw = re.sub(r"\\tab\b", "\t", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d*\s?", " ", raw)   # remaining control words
    raw = raw.replace("{", " ").replace("}", " ")
    return re.sub(r"[ \t]{2,}", " ", raw).strip()


@reader("email")
def read_email(path: Path) -> str:
    """Headers plus the text body of an RFC-822 message."""
    from email import policy
    from email.parser import BytesParser

    message = BytesParser(policy=policy.default).parsebytes(path.read_bytes())
    lines = [f"{field}: {message[field]}"
             for field in ("From", "To", "Cc", "Date", "Subject") if message[field]]
    try:
        part = message.get_body(preferencelist=("plain", "html"))
        body = part.get_content() if part is not None else ""
        if part is not None and part.get_content_type() == "text/html":
            body = strip_html(body)
    except Exception:
        body = decode_bytes(message.get_payload(decode=True) or b"")

    attachments = [p.get_filename() for p in message.iter_attachments() if p.get_filename()]
    if attachments:
        lines.append(f"Attachments: {', '.join(attachments)}")
    return "\n".join(lines) + "\n\n" + (body or "")


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------
@reader("pdf")
def read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PDF text extraction requires pypdf: pip install 'llm-extractor[pdf]'"
        ) from exc

    pages = []
    for index, page in enumerate(PdfReader(str(path)).pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""            # a damaged page must not lose the whole file
        if text.strip():
            pages.append(f"--- page {index} ---\n{text}")
    return "\n\n".join(pages)


# --------------------------------------------------------------------------
# Word / PowerPoint / Excel (OOXML)
# --------------------------------------------------------------------------
def _open_xml(path: Path, member: str, label: str):
    try:
        with zipfile.ZipFile(path) as zf:
            return ET.fromstring(zf.read(member))
    except (KeyError, zipfile.BadZipFile, ET.ParseError) as exc:
        raise RuntimeError(f"not a readable {label} file: {path}") from exc


@reader("docx")
def read_docx(path: Path) -> str:
    """Word OOXML: blocks in document order, tables kept as delimited rows.

    Body children are walked in order so a table appears where it sits, and each
    row stays on one line — a number divorced from its row label cannot be
    extracted correctly.
    """
    root = _open_xml(path, "word/document.xml", ".docx")
    body = root.find("w:body", NS)
    return "\n".join(_docx_blocks(body if body is not None else root))


def _docx_blocks(node):
    """Yield paragraphs and table rows, recursing into nested tables."""
    for child in node:
        tag = tag_of(child)
        if tag == "p":
            text = _docx_paragraph(child)
            if text:
                yield text
        elif tag == "tbl":
            for tr in child.findall("w:tr", NS):
                cells = [" ".join(_docx_blocks(tc)).strip()
                         for tc in tr.findall("w:tc", NS)]
                if any(cells):
                    yield row(cells)


def _docx_paragraph(paragraph) -> str:
    parts = []
    for node in paragraph.iter():
        tag = tag_of(node)
        if tag == "t" and node.text:
            parts.append(node.text)
        elif tag == "tab":
            parts.append("\t")
        elif tag in ("br", "cr"):
            parts.append("\n")
    return "".join(parts).strip()


@reader("pptx")
def read_pptx(path: Path) -> str:
    """Slide text in presentation order, plus speaker notes.

    Notes routinely carry the numbers behind a chart, so dropping them loses
    exactly the data worth extracting.
    """
    try:
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
            slides = sorted((n for n in names
                             if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                            key=_trailing_number)
            lines = []
            for index, name in enumerate(slides, start=1):
                lines.append(f"--- slide {index} ---")
                lines.extend(_drawing_text(ET.fromstring(zf.read(name))))

                notes = f"ppt/notesSlides/notesSlide{index}.xml"
                if notes in names:
                    spoken = [t for t in _drawing_text(ET.fromstring(zf.read(notes)))
                              if t != str(index)]
                    if spoken:
                        lines.append(f"[speaker notes] {' '.join(spoken)}")
                lines.append("")
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise RuntimeError(f"not a readable .pptx file: {path}") from exc
    return "\n".join(lines).strip()


def _drawing_text(root) -> list:
    return [text for text in (node_text(p) for p in root.findall(".//a:p", NS)) if text]


def _trailing_number(name: str) -> int:
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


@reader("xlsx")
def read_xlsx(path: Path) -> str:
    """Every sheet flattened to delimited rows, stdlib only."""
    try:
        with zipfile.ZipFile(path) as zf:
            shared = _xlsx_strings(zf)
            titles = _xlsx_titles(zf)
            sheets = sorted((n for n in zf.namelist()
                             if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
                            key=_trailing_number)
            out = []
            for index, name in enumerate(sheets):
                title = titles[index] if index < len(titles) else f"sheet{index + 1}"
                out.append(f"--- sheet: {title} ---")
                out.extend(_xlsx_rows(ET.fromstring(zf.read(name)), shared))
                out.append("")
    except (zipfile.BadZipFile, ET.ParseError) as exc:
        raise RuntimeError(f"not a readable .xlsx file: {path}") from exc
    return "\n".join(out).strip()


def _xlsx_strings(zf) -> list:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return [node_text(si) for si in root.findall("s:si", NS)]


def _xlsx_titles(zf) -> list:
    if "xl/workbook.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/workbook.xml"))
    return [sheet.get("name", "") for sheet in root.findall(".//s:sheet", NS)]


def _xlsx_rows(root, shared) -> list:
    rows = []
    for record in root.findall(".//s:row", NS):
        cells = [_xlsx_cell(cell, shared) for cell in record.findall("s:c", NS)]
        while cells and not cells[-1]:
            cells.pop()
        if cells:
            rows.append(row(cells))
    return rows


def _xlsx_cell(cell, shared) -> str:
    if cell.get("t") == "inlineStr":
        inline = cell.find("s:is", NS)
        return node_text(inline) if inline is not None else ""
    value = cell.find("s:v", NS)
    if value is None or value.text is None:
        return ""
    if cell.get("t") == "s":
        try:
            return shared[int(value.text)].strip()
        except (ValueError, IndexError):
            return ""
    return value.text.strip()


# --------------------------------------------------------------------------
# OpenDocument
# --------------------------------------------------------------------------
@reader("odt", "odp", "ods")
def read_opendocument(path: Path) -> str:
    """OpenDocument text, presentation or spreadsheet; tables stay as rows."""
    root = _open_xml(path, "content.xml", "OpenDocument")
    return "\n".join(_odf_blocks(root))


def _odf_blocks(node):
    """Yield paragraphs and table rows, never descending into an emitted block."""
    for child in node:
        tag, namespace = tag_of(child), child.tag[1:].split("}")[0]
        if namespace == NS["table"] and tag == "table-row":
            cells = [node_text(cell) for cell in child
                     if tag_of(cell) == "table-cell"]
            if any(cells):
                yield row(cells)
        elif namespace == NS["text"] and tag in ("p", "h"):
            text = node_text(child)
            if text:
                yield text
        else:
            yield from _odf_blocks(child)


# --------------------------------------------------------------------------
# EPUB
# --------------------------------------------------------------------------
@reader("epub")
def read_epub(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as zf:
            chapters = sorted(n for n in zf.namelist()
                              if n.lower().endswith((".xhtml", ".html", ".htm")))
            parts = [strip_html(decode_bytes(zf.read(name))) for name in chapters]
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"not a readable .epub file: {path}") from exc
    return "\n\n".join(part for part in parts if part)


# --------------------------------------------------------------------------
# Legacy Word
# --------------------------------------------------------------------------
@reader("doc")
def read_doc(path: Path) -> str:
    """Legacy .doc via a local converter, with a clear error when absent.

    Files carrying a .doc extension are often really RTF or OOXML; those are
    handled directly, so a converter is only needed for true OLE2 documents.
    """
    header = path.read_bytes()[:8]
    if header.startswith(b"{\\rtf"):
        return read_rtf(path)
    if header.startswith(b"PK\x03\x04"):
        return read_docx(path)

    if shutil.which("antiword"):
        proc = subprocess.run(["antiword", str(path)], capture_output=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip():
            return decode_bytes(proc.stdout)

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [soffice, "--headless", "--convert-to", "txt:Text", "--outdir", tmp,
                 str(path)], capture_output=True, check=False)
            converted = Path(tmp) / f"{path.stem}.txt"
            if proc.returncode == 0 and converted.exists():
                return decode_bytes(converted.read_bytes())

    raise RuntimeError(
        f"reading legacy .doc requires a converter: install antiword or LibreOffice, "
        f"or convert {path.name} to .docx/.pdf first"
    )


# --------------------------------------------------------------------------
# Images carry no text layer; the vision pass supplies their content.
# --------------------------------------------------------------------------
@reader("png", "jpeg", "gif", "webp", "tiff", "bmp")
def read_image(path: Path) -> str:
    return ""
