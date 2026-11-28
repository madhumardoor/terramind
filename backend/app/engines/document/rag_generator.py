"""
TerraMind AI - RAG Answer Generator
--------------------------------------
Combines document retrieval with LLM generation to answer
questions grounded in source documents.

Pipeline:
    1. User asks a question
    2. Question is embedded using Sentence Transformers
    3. FAISS retrieves the most similar document chunks
    4. Retrieved chunks are assembled into a context prompt
    5. Gemini LLM generates an answer grounded in the context
    6. Answer is returned with source citations

Uses the new google-genai package (google-generativeai is deprecated).
"""

from pathlib import Path
from typing import Optional
from google import genai
from app.core.config import settings
from app.core.logging_config import get_logger
from app.engines.document.embedder import DocumentEmbedder

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are TerraMind AI, an intelligent document assistant
for geospatial and agricultural intelligence.

Answer questions using ONLY the provided document context.
If the context does not contain enough information to answer,
say so clearly — do not invent facts.

Always cite your sources by mentioning the document name.
Keep answers concise, accurate and professional."""


class RAGGenerator:
    """
    Retrieval-Augmented Generation for document question answering.

    Usage
    -----
        generator = RAGGenerator(embedder)
        result = generator.answer("What is the crop yield forecast?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        embedder: DocumentEmbedder,
        model: Optional[str] = None,
        top_k: int = 5,
        max_context_chars: int = 3000,
    ):
        """
        Args:
            embedder:          Loaded DocumentEmbedder with indexed chunks.
            model:             Gemini model name. Defaults to config value.
            top_k:             Number of chunks to retrieve per query.
            max_context_chars: Maximum characters of context to send to LLM.
        """
        self.embedder = embedder
        self.model = model or settings.gemini_model
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.client = genai.Client(api_key=settings.gemini_api_key)
        logger.info(f"RAGGenerator initialised with model: {self.model}")

    def _build_context(self, chunks: list[dict]) -> tuple[str, list[dict]]:
        """
        Assemble retrieved chunks into a context string for the LLM.

        Args:
            chunks: Retrieved chunk dicts from DocumentEmbedder.search().

        Returns:
            Tuple of (context_string, sources_list)
        """
        context_parts = []
        sources = []
        total_chars = 0

        for chunk in chunks:
            chunk_text = chunk["text"]
            source = chunk.get("source_name", "Unknown")
            page = chunk.get("page_number")

            if total_chars + len(chunk_text) > self.max_context_chars:
                break

            page_info = f" (page {page})" if page else ""
            context_parts.append(
                f"[Source: {source}{page_info}]\n{chunk_text}"
            )
            total_chars += len(chunk_text)

            source_entry = {
                "source_name": source,
                "document_id": chunk.get("document_id"),
                "chunk_id":    chunk.get("chunk_id"),
                "score":       chunk.get("score"),
                "page_number": page,
            }
            if source_entry not in sources:
                sources.append(source_entry)

        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def answer(
        self,
        question: str,
        score_threshold: float = 0.0,
    ) -> dict:
        """
        Answer a question using retrieved document context.

        Args:
            question:        The user's question.
            score_threshold: Minimum similarity score for retrieved chunks.

        Returns:
            Dict with question, answer, sources, chunks_used, model,
            retrieval_scores.
        """
        logger.info(f"RAG query: {question!r}")

        # Step 1: Retrieve relevant chunks
        retrieved = self.embedder.search(
            query=question,
            top_k=self.top_k,
            score_threshold=score_threshold,
        )

        if not retrieved:
            logger.warning("No relevant chunks found for query")
            return {
                "question":         question,
                "answer":           (
                    "I could not find relevant information in the loaded "
                    "documents to answer this question."
                ),
                "sources":          [],
                "chunks_used":      0,
                "model":            self.model,
                "retrieval_scores": [],
            }

        # Step 2: Build context
        context, sources = self._build_context(retrieved)
        retrieval_scores = [r["score"] for r in retrieved]

        logger.info(
            f"Retrieved {len(retrieved)} chunks, "
            f"using {len(sources)} unique sources"
        )

        # Step 3: Build prompt
        full_prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Document context:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer based only on the document context above:"
        )

        # Step 4: Generate answer with Gemini
        logger.info(f"Generating answer with {self.model}...")

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
        )

        answer = response.text.strip()

        logger.info(
            f"Answer generated: {len(answer)} characters, "
            f"{len(sources)} sources cited"
        )

        return {
            "question":         question,
            "answer":           answer,
            "sources":          sources,
            "chunks_used":      len(retrieved),
            "model":            self.model,
            "retrieval_scores": retrieval_scores,
        }