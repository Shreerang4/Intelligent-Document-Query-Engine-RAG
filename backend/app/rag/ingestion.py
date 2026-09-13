"""Queue-independent PDF cleaning and page-aware chunking."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import TypedDict

import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.app.rag.config import get_chunk_overlap, get_chunk_size, get_max_pdf_bytes


logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    text: str
    page: int
    chunk_id: int


class PdfContentError(ValueError):
    """Base class for deterministic PDF content failures."""


class PdfTooLargeError(PdfContentError):
    """Raised when source bytes exceed the configured ingestion limit."""


class InvalidPdfError(PdfContentError):
    """Raised when PyMuPDF cannot safely open or read a source PDF."""


class NoMeaningfulTextError(PdfContentError):
    """Raised when PDF extraction produces no usable text chunks."""


def ascii_normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _is_board_coordinate_line(line: str) -> bool:
    compact = " ".join(line.lower().split())
    if re.fullmatch(r"(?:[a-h](?:\s+[a-h]){3,7})", compact):
        return True
    if re.fullmatch(r"(?:[1-8](?:\s+[1-8]){0,7})", compact):
        return True
    return False


def _line_has_language_content(line: str) -> bool:
    non_space = sum(1 for char in line if not char.isspace())
    alpha_chars = sum(1 for char in line if char.isalpha())
    if non_space == 0 or alpha_chars == 0:
        return False
    if alpha_chars < 2 and non_space < 12:
        return False
    if alpha_chars / max(non_space, 1) < 0.18 and alpha_chars < 12:
        return False
    return True


def _clean_extracted_line(line: str) -> str:
    cleaned = ascii_normalize(line)
    cleaned = re.sub(r"[^\x20-\x7E]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned or _is_board_coordinate_line(cleaned):
        return ""
    if re.fullmatch(r"[1-8]", cleaned) or not _line_has_language_content(cleaned):
        return ""
    return cleaned


def _clean_extracted_page_text(page_text: str) -> str:
    cleaned_lines: list[str] = []
    previous_line = ""
    for raw_line in page_text.splitlines():
        cleaned_line = _clean_extracted_line(raw_line)
        if not cleaned_line or cleaned_line == previous_line:
            continue
        cleaned_lines.append(cleaned_line)
        previous_line = cleaned_line
    return "\n".join(cleaned_lines).strip()


def _is_low_quality_chunk(text: str) -> bool:
    non_space = sum(1 for char in text if not char.isspace())
    alpha_chars = sum(1 for char in text if char.isalpha())
    if non_space == 0 or alpha_chars == 0:
        return True
    if len(text) < 30 and alpha_chars < 12:
        return True
    if alpha_chars / max(non_space, 1) < 0.22 and alpha_chars < 80:
        return True
    return False


def parse_and_chunk_pdf_bytes(pdf_bytes: bytes) -> list[ChunkRecord]:
    """Parse PDF bytes into the canonical page-aware chunk sequence."""
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise InvalidPdfError("PDF source is empty or invalid.")
    if len(pdf_bytes) > get_max_pdf_bytes():
        raise PdfTooLargeError("PDF exceeds maximum allowed size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=get_chunk_size(),
        chunk_overlap=get_chunk_overlap(),
    )
    chunk_records: list[ChunkRecord] = []
    chunk_id = 0
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            if document.needs_pass:
                raise InvalidPdfError("Encrypted PDF cannot be read without a password.")
            for page_number, page in enumerate(document, start=1):
                page_text = _clean_extracted_page_text(page.get_text("text"))
                if not page_text:
                    continue
                for chunk_text in splitter.split_text(page_text):
                    normalized_text = chunk_text.strip()
                    if not normalized_text or _is_low_quality_chunk(normalized_text):
                        continue
                    chunk_records.append(
                        {"text": normalized_text, "page": page_number, "chunk_id": chunk_id}
                    )
                    chunk_id += 1
    except PdfContentError:
        raise
    except Exception as exc:
        raise InvalidPdfError("Failed to parse PDF document.") from exc

    if not chunk_records:
        raise NoMeaningfulTextError("No meaningful text found in the PDF.")
    logger.info("Document parsed into %s chunks.", len(chunk_records))
    return chunk_records
