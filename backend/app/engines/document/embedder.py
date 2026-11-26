"""
TerraMind AI - Document Embedder
-----------------------------------
Converts text chunks into vector embeddings using Sentence Transformers.
Stores and retrieves embeddings using FAISS for fast similarity search.

Model: all-MiniLM-L6-v2
    - 384-dimensional embeddings
    - Fast CPU inference
    - Good semantic understanding for English text
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.core.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class DocumentEmbedder:
    """
    Embeds document chunks and provides semantic similarity search.

    Usage
    -----
        embedder = DocumentEmbedder()
        embedder.add_chunks(chunks)
        results = embedder.search("query", top_k=5)
        embedder.save(Path("data/processed/rag_index"))
        embedder.load(Path("data/processed/rag_index"))
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[dict] = []
        self.is_built = False

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Embedding model loaded. Dimension: {EMBEDDING_DIM}")
        return self.model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Convert a list of text strings into embedding vectors.

        Returns:
            Numpy array of shape (N, 384), float32, L2-normalised.
        """
        model = self._load_model()
        logger.info(f"Embedding {len(texts)} texts...")

        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        logger.info(f"Embeddings shape: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def add_chunks(self, chunks: list[dict]) -> dict:
        """
        Embed a list of chunks and add them to the FAISS index.

        Args:
            chunks: List of chunk dicts from DocumentChunker.

        Returns:
            Dict with n_chunks_added, total_chunks, embedding_dim.
        """
        if not chunks:
            logger.warning("No chunks provided to add_chunks")
            return {"n_chunks_added": 0, "total_chunks": len(self.chunks)}

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_texts(texts)

        if self.index is None:
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            logger.info(f"Initialised FAISS IndexFlatIP (dim={EMBEDDING_DIM})")

        self.index.add(embeddings)
        self.chunks.extend(chunks)
        self.is_built = True

        logger.info(
            f"Added {len(chunks)} chunks. "
            f"Total in index: {self.index.ntotal}"
        )

        return {
            "n_chunks_added": len(chunks),
            "total_chunks": len(self.chunks),
            "embedding_dim": EMBEDDING_DIM,
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Find the most semantically similar chunks to a query.

        Args:
            query:           The search query or question.
            top_k:           Number of results to return.
            score_threshold: Minimum similarity score (0.0 to 1.0).

        Returns:
            List of result dicts sorted by similarity descending.

        Raises:
            RuntimeError: If no chunks have been added yet.
        """
        if not self.is_built or self.index is None:
            raise RuntimeError(
                "No chunks in the index. Call add_chunks() first."
            )

        logger.info(f"Searching for: {query!r} (top_k={top_k})")

        query_embedding = self.embed_texts([query])

        actual_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_embedding, actual_k)

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx == -1:
                continue
            if float(score) < score_threshold:
                continue

            chunk = self.chunks[idx]
            results.append({
                "rank":        rank,
                "score":       round(float(score), 4),
                "chunk_id":    chunk.get("chunk_id"),
                "text":        chunk.get("text"),
                "source_name": chunk.get("source_name"),
                "document_id": chunk.get("document_id"),
                "chunk_index": chunk.get("chunk_index"),
                "page_number": chunk.get("page_number"),
            })

        logger.info(f"Search returned {len(results)} results")
        return results

    def save(self, save_dir: Path) -> dict:
        """
        Save the FAISS index and chunk metadata to disk.
        """
        if not self.is_built:
            raise RuntimeError("Nothing to save. Add chunks first.")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        index_path = save_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))

        chunks_path = save_dir / "chunks_metadata.pkl"
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        config_path = save_dir / "embedder_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump({
                "model_name":    self.model_name,
                "embedding_dim": EMBEDDING_DIM,
                "total_chunks":  len(self.chunks),
            }, f, indent=2)

        logger.info(f"Index saved to: {save_dir} ({len(self.chunks)} chunks)")
        return {
            "index_path":  str(index_path),
            "chunks_path": str(chunks_path),
            "config_path": str(config_path),
        }

    def load(self, save_dir: Path) -> dict:
        """
        Load a previously saved FAISS index and chunk metadata.
        """
        save_dir = Path(save_dir)
        index_path  = save_dir / "faiss_index.bin"
        chunks_path = save_dir / "chunks_metadata.pkl"
        config_path = save_dir / "embedder_config.json"

        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index found at: {index_path}"
            )

        self.index = faiss.read_index(str(index_path))

        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        config = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)

        self.is_built = True
        logger.info(
            f"Index loaded: {len(self.chunks)} chunks, "
            f"model={config.get('model_name')}"
        )
        return config