"""What a file *is*, and how it must be processed.

Suffixes lie: archives hand you `report.pdf` that is really a PNG, exports drop
the extension entirely, and `.doc` is as likely to be RTF as OLE2. So detection
is content-first — magic bytes decide, and the suffix only breaks ties between
formats sharing a container (every OOXML file is a ZIP) or that are plain text
either way (`.md` vs `.txt` vs `.csv`).

Each format declares its own processing path instead of the pipeline
special-casing extensions:

``kind``       what the content is, for routing and reporting;
``text_layer`` whether the bytes normally yield extractable text at all;
``figures``    where images come from — the file itself, embedded media, or
               rendered pages.

Adding a format is one :class:`Format` entry plus a reader; nothing downstream
changes.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

# What the content is.
TEXT, DOCUMENT, SLIDES, SPREADSHEET, IMAGE, EMAIL, DATA, BOOK = (
    "text", "document", "slides", "spreadsheet", "image", "email", "data", "book")

# Where figures come from.
NO_FIGURES = "none"        # nothing to look at
SELF = "self"              # the file *is* the image
EMBEDDED = "embedded"      # images are packed inside the container
RENDER = "render"          # pages must be rasterised to be seen


@dataclass(frozen=True)
class Format:
    name: str
    extensions: tuple
    kind: str
    mime: str = "application/octet-stream"
    text_layer: bool = True
    figures: str = NO_FIGURES
    requires: str = ""
    description: str = ""

    @property
    def is_image(self) -> bool:
        return self.figures == SELF

    def to_dict(self) -> dict:
        return {
            "name": self.name, "extensions": list(self.extensions), "kind": self.kind,
            "text_layer": self.text_layer, "figures": self.figures,
            "requires": self.requires or None, "description": self.description,
        }


def _image(name, extensions, mime, description) -> Format:
    return Format(name, extensions, IMAGE, mime, text_layer=False, figures=SELF,
                  description=description)


BUILTIN = (
    Format("plain", (".txt", ".text", ".log"), TEXT, "text/plain",
           description="Plain text."),
    Format("markdown", (".md", ".markdown", ".rst"), TEXT, "text/markdown",
           description="Markdown or reStructuredText."),
    Format("html", (".html", ".htm", ".xhtml"), TEXT, "text/html",
           description="HTML page; tags and scripts stripped."),
    Format("xml", (".xml", ".jats", ".nxml"), TEXT, "application/xml",
           description="XML such as JATS/PMC article markup; flattened to text."),
    Format("json", (".json", ".jsonl", ".ndjson"), DATA, "application/json",
           description="JSON or newline-delimited JSON."),
    Format("csv", (".csv", ".tsv", ".tab"), SPREADSHEET, "text/csv",
           description="Delimited text; the delimiter is sniffed."),
    Format("rtf", (".rtf",), DOCUMENT, "application/rtf",
           description="Rich Text Format; control words stripped."),
    Format("pdf", (".pdf",), DOCUMENT, "application/pdf", figures=RENDER,
           requires="pypdf",
           description="PDF; scans have no text layer and are rendered for OCR."),
    Format("docx", (".docx", ".docm"), DOCUMENT,
           "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
           figures=EMBEDDED,
           description="Word OOXML; paragraphs, tables and embedded images."),
    Format("pptx", (".pptx", ".pptm"), SLIDES,
           "application/vnd.openxmlformats-officedocument.presentationml.presentation",
           figures=EMBEDDED,
           description="PowerPoint OOXML; slide text, speaker notes, slide images."),
    Format("xlsx", (".xlsx", ".xlsm"), SPREADSHEET,
           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
           description="Excel OOXML; every sheet flattened to delimited rows."),
    Format("odt", (".odt",), DOCUMENT, "application/vnd.oasis.opendocument.text",
           figures=EMBEDDED, description="OpenDocument text."),
    Format("odp", (".odp",), SLIDES, "application/vnd.oasis.opendocument.presentation",
           figures=EMBEDDED, description="OpenDocument presentation."),
    Format("ods", (".ods",), SPREADSHEET,
           "application/vnd.oasis.opendocument.spreadsheet",
           description="OpenDocument spreadsheet."),
    Format("epub", (".epub",), BOOK, "application/epub+zip", figures=EMBEDDED,
           description="EPUB book; chapters concatenated."),
    Format("email", (".eml", ".mbox"), EMAIL, "message/rfc822",
           description="RFC-822 email; headers plus the text body."),
    Format("doc", (".doc", ".dot"), DOCUMENT, "application/msword",
           requires="antiword or libreoffice",
           description="Legacy Word; needs a local converter."),
    _image("png", (".png",), "image/png", "PNG image."),
    _image("jpeg", (".jpg", ".jpeg", ".jpe"), "image/jpeg", "JPEG image."),
    _image("gif", (".gif",), "image/gif", "GIF image."),
    _image("webp", (".webp",), "image/webp", "WebP image."),
    _image("tiff", (".tif", ".tiff"), "image/tiff", "TIFF image."),
    _image("bmp", (".bmp",), "image/bmp", "BMP image."),
)

FORMATS: dict = {}
_BY_EXTENSION: dict = {}


def register_format(fmt: Format) -> Format:
    """Add a format; third-party packages use this to extend detection."""
    FORMATS[fmt.name] = fmt
    for ext in fmt.extensions:
        _BY_EXTENSION.setdefault(ext.lower(), fmt.name)
    return fmt


for _fmt in BUILTIN:
    register_format(_fmt)

IMAGE_EXTENSIONS = tuple(e for f in BUILTIN if f.is_image for e in f.extensions)
#: Files that are never documents; skipped before sniffing to keep scans cheap.
SKIP_EXTENSIONS = frozenset((
    ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".bin", ".dat", ".db",
    ".sqlite", ".sqlite3", ".zip", ".gz", ".bz2", ".xz", ".tar", ".7z", ".rar",
    ".mp3", ".mp4", ".mov", ".avi", ".wav", ".ttf", ".otf", ".woff", ".woff2",
    ".ico", ".lock", ".whl", ".egg",
))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------
_MAGIC = (
    (b"%PDF", "pdf"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
    (b"BM", "bmp"),
    (b"II*\x00", "tiff"),
    (b"MM\x00*", "tiff"),
    (b"{\\rtf", "rtf"),
    # OLE2 compound file: legacy .doc/.xls/.ppt all share it.
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "doc"),
)
_ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
HEADER_BYTES = 512


def read_header(path, size: int = HEADER_BYTES) -> bytes:
    try:
        with open(path, "rb") as fh:
            return fh.read(size)
    except OSError:
        return b""


def sniff(header: bytes, path=None) -> str:
    """Return a format name from content, or "" when the bytes are ambiguous."""
    if not header:
        return ""
    for signature, name in _MAGIC:
        if header.startswith(signature):
            return name
    if header[:4] in _ZIP_MAGIC:
        return _zip_format(path) if path else ""
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    start = header.lstrip()[:64].lower()
    if start.startswith(b"<?xml"):
        return "xml"
    if start.startswith((b"<!doctype html", b"<html")):
        return "html"
    return ""


def _zip_format(path) -> str:
    """Disambiguate ZIP-based formats by what is inside the container."""
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            mimetype = (zf.read("mimetype").decode("ascii", "replace").strip()
                        if "mimetype" in names else "")
    except (zipfile.BadZipFile, OSError, KeyError):
        return ""

    if mimetype:
        for fmt in FORMATS.values():
            if fmt.mime == mimetype:
                return fmt.name
    for prefix, name in (("word/", "docx"), ("ppt/", "pptx"), ("xl/", "xlsx")):
        if any(n.startswith(prefix) for n in names):
            return name
    return "epub" if "META-INF/container.xml" in names else ""


def format_for_extension(suffix: str):
    return FORMATS.get(_BY_EXTENSION.get((suffix or "").lower(), ""))


def detect_format(path):
    """Identify a file's format from its content, falling back to its suffix.

    Content wins for anything with a signature, which is what makes mislabelled
    and extension-less files work. The suffix only decides between formats that
    are indistinguishable byte-wise, and is preferred over a sniffed result of
    the same format so `.docm` and `.jsonl` keep their specific entry.
    """
    path = Path(path)
    by_suffix = format_for_extension(path.suffix)
    header = read_header(path)
    sniffed = FORMATS.get(sniff(header, path))

    if sniffed is not None:
        return by_suffix if (by_suffix is not None and by_suffix.name == sniffed.name) \
            else sniffed
    if by_suffix is not None:
        return by_suffix
    return FORMATS["plain"] if looks_like_text(header) else None


def looks_like_text(header: bytes) -> bool:
    if not header or b"\x00" in header:
        return False
    try:
        header.decode("utf-8")
        return True
    except UnicodeDecodeError:
        printable = sum(32 <= b < 127 or b in (9, 10, 13) for b in header)
        return printable / len(header) > 0.9


def supported_extensions() -> list:
    return sorted(_BY_EXTENSION)


def describe_formats() -> list:
    return [fmt.to_dict() for fmt in FORMATS.values()]


def mime_for(path) -> str:
    """MIME type for an image being sent to a vision model."""
    fmt = format_for_extension(Path(path).suffix) or detect_format(path)
    return fmt.mime if fmt is not None and fmt.is_image else "image/png"


# --------------------------------------------------------------------------
# Text decoding
# --------------------------------------------------------------------------
_BOMS = (
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16"),
)


def decode_bytes(data: bytes, encodings=("utf-8", "cp1252", "latin-1")) -> str:
    """Decode text of unknown encoding without ever raising.

    A byte-order mark decides when present; otherwise UTF-8 is tried first and
    single-byte fallbacks follow, so a Windows-1252 export neither turns into
    mojibake nor aborts a batch.
    """
    for bom, encoding in _BOMS:
        if data.startswith(bom):
            return data.decode(encoding, errors="replace")
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def read_text(path) -> str:
    return decode_bytes(Path(path).read_bytes())
