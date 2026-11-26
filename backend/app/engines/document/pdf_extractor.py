"""
TerraMind AI - PDF Extractor
------------------------------
Extracts text from PDF documents using PyMuPDF (fitz).

PyMuPDF is fast, handles scanned and digital PDFs, and extracts
text with page-level metadata. It is free and open source.

Supported inputs:
    - File path to a PDF
    - Bytes object containing PDF data

Output:
    - List of page dicts with page_number, text, char_count
    - Full document text as a single string
    - Document metadata (title, author, page count)
"""

from pathlib import Path
from typing import Union
import pymupdf as fitz  # PyMuPDF
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class PDFExtractor:
    """
    Extracts text and metadata from PDF documents.

    Usage
    -----
        extractor = PDFExtractor()
        result = extractor.extract(path_to_pdf)
        # result["pages"] -> list of page dicts
        # result["full_text"] -> complete document text
        # result["metadata"] -> title, author, page_count etc
    """

    def extract(self, source: Union[str, Path, bytes]) -> dict:
        """
        Extract text and metadata from a PDF.

        Args:
            source: File path or bytes of the PDF.

        Returns:
            Dict with keys:
                pages     - list of {page_number, text, char_count}
                full_text - all pages joined with newlines
                metadata  - {title, author, page_count, file_size_bytes}
                page_count - integer

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If the PDF cannot be opened.
        """
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"PDF not found: {path}")
            logger.info(f"Extracting text from PDF: {path.name}")
            doc = fitz.open(str(path))
            file_size = path.stat().st_size
            filename = path.name
        elif isinstance(source, bytes):
            logger.info("Extracting text from PDF bytes")
            doc = fitz.open(stream=source, filetype="pdf")
            file_size = len(source)
            filename = "uploaded.pdf"
        else:
            raise ValueError(f"Unsupported source type: {type(source)}")

        pages = []
        full_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")

            # Basic cleaning
            text = text.strip()
            text = "\n".join(
                line.strip() for line in text.splitlines()
                if line.strip()
            )

            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "char_count": len(text),
            })
            full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)

        # Extract PDF metadata
        meta = doc.metadata or {}
        metadata = {
            "filename":    filename,
            "title":       meta.get("title", ""),
            "author":      meta.get("author", ""),
            "subject":     meta.get("subject", ""),
            "page_count":  len(doc),
            "file_size_bytes": file_size,
            "total_chars": len(full_text),
        }

        doc.close()

        logger.info(
            f"Extracted {len(pages)} pages, "
            f"{metadata['total_chars']:,} characters from {filename}"
        )

        return {
            "pages":      pages,
            "full_text":  full_text,
            "metadata":   metadata,
            "page_count": len(pages),
        }

    def extract_pages_text(self, source: Union[str, Path, bytes]) -> list[str]:
        """
        Extract text page by page as a simple list of strings.

        Args:
            source: File path or bytes.

        Returns:
            List of strings, one per page.
        """
        result = self.extract(source)
        return [p["text"] for p in result["pages"]]
