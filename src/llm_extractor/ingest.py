"""Document ingestion: detect the format, read it, collect its figures.

This module holds no format knowledge of its own. :mod:`llm_extractor.formats`
decides what a file is and which processing path it declares;
:mod:`llm_extractor.readers` turns it into text. Ingestion applies them, so
supporting a new format never touches the pipeline.

The figure strategy comes from the format rather than an extension check:

``self``      the file is an image — it goes straight to the vision pass;
``embedded``  images are unpacked from the container (pptx, docx, odp, epub);
``render``    pages are rasterised so a scanned PDF can still be read;
``none``      nothing to look at.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from . import formats
from .formats import (EMBEDDED, IMAGE_EXTENSIONS, NO_FIGURES, RENDER, SELF,
                      SKIP_EXTENSIONS, describe_formats, detect_format,
                      format_for_extension, mime_for, supported_extensions)
from .readers import read as read_format

__all__ = [
    "Document", "IMAGE_EXTENSIONS", "UnsupportedFormat", "describe_formats",
    "detect_format", "discover", "find_figures", "load_document", "mime_for",
    "supported_extensions",
]

PLAIN = formats.FORMATS["plain"]


class UnsupportedFormat(ValueError):
    """Raised when a file's format cannot be determined or read."""


@dataclass
class Document:
    doc_id: str
    text: str
    figures: list = field(default_factory=list)
    source_path: str = ""
    fmt: formats.Format = PLAIN
    read_error: str = ""

    @property
    def format_name(self) -> str:
        return self.fmt.name

    @property
    def media_type(self) -> str:
        """The format's kind: text, document, slides, spreadsheet, image…"""
        return self.fmt.kind

    @property
    def has_text_layer(self) -> bool:
        return self.fmt.text_layer and not self.read_error

    @property
    def is_image_only(self) -> bool:
        return self.fmt.is_image


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
#: Where each container keeps its images.
MEDIA_PREFIXES = {
    "pptx": ("ppt/media/",),
    "docx": ("word/media/",),
    "odt": ("Pictures/",),
    "odp": ("Pictures/",),
    "epub": (),
}


def find_figures(path, fmt=None, cache_dir=None, max_figures: int = 20) -> list:
    """Return the images to send to the vision model for this document."""
    path = Path(path)
    fmt = fmt or detect_format(path)
    if fmt is None:
        return []
    if fmt.figures == SELF:
        return [path]

    # A sibling "<stem>_files" directory, typical of HTML and Office exports.
    sibling = Path(f"{path.with_suffix('')}_files")
    figures = ([p for p in sorted(sibling.iterdir())
                if p.suffix.lower() in IMAGE_EXTENSIONS] if sibling.is_dir() else [])

    if not figures and fmt.figures == EMBEDDED:
        figures = _unpack_media(path, fmt.name, cache_dir)
    if not figures and fmt.figures == RENDER:
        figures = _render_pages(path, cache_dir)
    return figures[:max_figures]


def _media_dir(cache_dir, kind: str, stem: str) -> Path:
    return Path(cache_dir or ".llm_cache") / "media" / kind / stem


def _unpack_media(path: Path, format_name: str, cache_dir) -> list:
    prefixes = MEDIA_PREFIXES.get(format_name, ())
    out_dir = _media_dir(cache_dir, format_name, path.stem)
    figures = []
    try:
        with zipfile.ZipFile(path) as zf:
            names = sorted(n for n in zf.namelist()
                           if n.lower().endswith(IMAGE_EXTENSIONS)
                           and (not prefixes or n.startswith(prefixes)))
            if names:
                out_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                target = out_dir / Path(name).name
                if not target.exists():
                    target.write_bytes(zf.read(name))
                figures.append(target)
    except (zipfile.BadZipFile, OSError):
        return []
    return figures


#: A page whose text layer is at least this long, and which draws nothing, is
#: treated as prose that the text pass already read in full.
TEXT_ONLY_MIN_CHARS = 200

#: Vector drawings below this count are rules, borders and underlines rather
#: than a chart.
VECTOR_DRAWING_MIN = 12

#: An image covering this much of the page is the page — a scan, not a figure.
FULL_PAGE_IMAGE_RATIO = 0.5


def page_needs_vision(page, min_chars: int = TEXT_ONLY_MIN_CHARS) -> bool:
    """Decide whether a PDF page carries anything the text pass could not read.

    The vision pass is by far the most expensive stage — measured at roughly
    2,100 input and 1,700 output tokens per page — and ``figures=RENDER`` sends
    *every* page to it. On a born-digital paper most pages are prose that the
    text extractor already has verbatim, so paying a vision model to re-read
    them buys nothing.

    PyMuPDF already exposes the signal, so no model and no training data are
    needed: a page is worth rendering when it draws a chart, embeds a picture,
    or has too little text to be prose (a scan, whose "text" is often empty).
    When the signal cannot be read at all the answer is yes, because skipping a
    page that did hold a figure is the more expensive mistake.
    """
    try:
        images = page.get_images()
        drawings = page.get_drawings()
        text_length = len(page.get_text().strip())
    except Exception:
        return True

    if text_length < min_chars:
        return True                      # scan, cover page, or figure-only page
    if images:
        return True
    return len(drawings) >= VECTOR_DRAWING_MIN


def _render_pages(path: Path, cache_dir, triage: bool = True) -> list:
    """Rasterise the PDF pages a vision model could actually add something to."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []
    try:
        pdf = fitz.open(str(path))
    except Exception:
        return []

    out_dir = _media_dir(cache_dir, "pdf", path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    for index in range(len(pdf)):
        page = pdf.load_page(index)
        if triage and not page_needs_vision(page):
            continue
        image = out_dir / f"{path.stem}-p{index + 1:03d}.png"
        if not image.exists():
            page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(str(image))
        figures.append(image)
    return figures


# --------------------------------------------------------------------------
# Loading and discovery
# --------------------------------------------------------------------------
def load_document(path, doc_id: str | None = None, with_figures: bool = False,
                  cache_dir=None, max_figures: int = 20) -> Document:
    """Detect, read and optionally collect the figures of one file."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    fmt = detect_format(path)
    if fmt is None:
        raise UnsupportedFormat(
            f"cannot determine the format of '{path.name}'; supported extensions: "
            f"{', '.join(supported_extensions())}"
        )

    # An image has nothing but its figure, so it is always collected.
    wants_figures = with_figures or fmt.figures == SELF
    figures = (find_figures(path, fmt=fmt, cache_dir=cache_dir, max_figures=max_figures)
               if wants_figures else [])

    text, read_error = "", ""
    try:
        text = read_format(fmt.name, path)
    except Exception as exc:
        # A damaged or encrypted file is not a lost cause when its pages can
        # still be looked at, so degrade to the vision path instead of failing.
        if not figures and fmt.figures != NO_FIGURES and not wants_figures:
            figures = find_figures(path, fmt=fmt, cache_dir=cache_dir,
                                   max_figures=max_figures)
        if not figures:
            raise (exc if isinstance(exc, RuntimeError) else UnsupportedFormat(
                f"failed to read {path.name} as {fmt.name}: {exc}")) from exc
        read_error = f"{type(exc).__name__}: {exc}"

    return Document(doc_id=doc_id or path.stem, text=text, figures=figures,
                    source_path=str(path), fmt=fmt, read_error=read_error)


def discover(input_dir, extensions=None) -> list:
    """Recursively list every readable document under ``input_dir``.

    Without a filter, files are included when their *content* is recognised, so
    extension-less and mislabelled files are picked up too.
    """
    root = Path(input_dir)
    if root.is_file():
        return [root]

    allowed = ({e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
               if extensions else None)
    found = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if allowed is not None:
            if suffix in allowed:
                found.append(path)
            continue
        if suffix in SKIP_EXTENSIONS:
            continue
        if format_for_extension(suffix) is not None or detect_format(path) is not None:
            found.append(path)
    return found
