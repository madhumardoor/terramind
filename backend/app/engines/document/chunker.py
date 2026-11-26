"""
TerraMind AI - Document Chunker
---------------------------------
Splits extracted document text into overlapping chunks
suitable for embedding and vector search.

Why chunking:
    Embedding models have a token limit (typically 512 tokens).
    Documents are much longer. Chunking splits the document into
    segments that fit within the model limit while preserving context.

Why overlap:
    Without overlap, important context at chunk boundaries is lost.
    Overlap ensures sentences are not arbitrarily cut off.

Chunk metadata:
    Every chunk carries metadata about its source document and position.
    This metadata is stored alongside the embedding in the vector database
    and returned when a chunk is retrieved — enabling source citations.
"""

from typing import Optional
from app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_CHUNK_SIZE = 500      # characters per chunk
DEFAULT_CHUNK_OVERLAP = 100   # characters of overlap between chunks


class DocumentChunker:
    """
    Splits document text into overlapping chunks for RAG.

    Usage
    -----
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        chunks = chunker.chunk_text(
            text=full_document_text,
            document_id="fao_report_2024",
            source_name="FAO World Food Report 2024",
        )
        # Each chunk: {chunk_id, text, char_count, document_id,
        #              source_name, chunk_index, total_chunks}
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Args:
            chunk_size:    Target character count per chunk.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})."
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        document_id: str,
        source_name: str,
        page_number: Optional[int] = None,
    ) -> list[dict]:
        """
        Split text into overlapping chunks with metadata.

        Strategy:
            1. Split text into sentences (by period + newline boundaries)
            2. Accumulate sentences into chunks up to chunk_size characters
            3. When a chunk is full, save it and start the next chunk
               beginning from the overlap point

        Args:
            text:        The document text to chunk.
            document_id: Unique identifier for the source document.
            source_name: Human-readable document name for citations.
            page_number: Optional page number if chunking one page at a time.

        Returns:
            List of chunk dicts, each containing:
                chunk_id     - unique identifier
                text         - chunk content
                char_count   - character count
                document_id  - source document ID
                source_name  - source document name
                chunk_index  - position in document (0-based)
                total_chunks - total number of chunks (filled after)
                page_number  - source page if provided
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for document: {document_id}")
            return []

        # Split into sentences at period or newline boundaries
        # This is a simple but effective sentence splitter
        import re
        sentences = re.split(r"(?<=[.!?])\s+|\n\n+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)

                # Start next chunk with overlap
                # Keep sentences from the end of the current chunk
                overlap_text = chunk_text[-self.chunk_overlap:]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space

        # Add the final chunk
        if current_chunk:
            final_text = " ".join(current_chunk)
            if final_text.strip():
                chunks.append(final_text.strip())

        # Build chunk dicts with metadata
        result = []
        total = len(chunks)

        for idx, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{idx:04d}"
            result.append({
                "chunk_id":    chunk_id,
                "text":        chunk_text,
                "char_count":  len(chunk_text),
                "document_id": document_id,
                "source_name": source_name,
                "chunk_index": idx,
                "total_chunks": total,
                "page_number": page_number,
            })

        logger.info(
            f"Chunked document '{document_id}' into {total} chunks "
            f"(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return result

    def chunk_document(
        self,
        extraction_result: dict,
        document_id: str,
        source_name: str,
        strategy: str = "full_document",
    ) -> list[dict]:
        """
        Chunk a document from a PDFExtractor result.

        Two strategies:
            full_document: chunk the entire document as one text block.
                           Faster, loses page boundary information.
            page_by_page:  chunk each page separately, preserving page numbers.
                           Better for documents where page boundaries matter.

        Args:
            extraction_result: Output from PDFExtractor.extract().
            document_id:       Unique document identifier.
            source_name:       Human-readable name for citations.
            strategy:          "full_document" or "page_by_page".

        Returns:
            List of chunk dicts with full metadata.
        """
        if strategy == "full_document":
            return self.chunk_text(
                text=extraction_result["full_text"],
                document_id=document_id,
                source_name=source_name,
            )

        elif strategy == "page_by_page":
            all_chunks = []
            for page in extraction_result["pages"]:
                if not page["text"].strip():
                    continue
                page_chunks = self.chunk_text(
                    text=page["text"],
                    document_id=document_id,
                    source_name=source_name,
                    page_number=page["page_number"],
                )
                all_chunks.extend(page_chunks)

            # Re-index chunks sequentially across all pages
            for idx, chunk in enumerate(all_chunks):
                chunk["chunk_index"] = idx
                chunk["total_chunks"] = len(all_chunks)
                chunk["chunk_id"] = f"{document_id}_chunk_{idx:04d}"

            logger.info(
                f"Page-by-page chunking produced {len(all_chunks)} chunks "
                f"from {len(extraction_result['pages'])} pages"
            )
            return all_chunks

        else:
            raise ValueError(
                f"Unknown strategy: {strategy}. "
                f"Use 'full_document' or 'page_by_page'."
            )
