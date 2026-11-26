import sys
from pathlib import Path
import os

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.logging_config import setup_logging
setup_logging(log_level="INFO")

from app.engines.document.pdf_extractor import PDFExtractor
from app.engines.document.chunker import DocumentChunker
from app.engines.document.embedder import DocumentEmbedder

print("\n" + "="*60)
print("TerraMind AI - RAG Pipeline Test")
print("="*60)

pdf_path = Path("data/raw/documents/fao_food_security_2020.pdf")
fallback  = Path("data/raw/documents/test_document.pdf")
pdf_file  = pdf_path if pdf_path.exists() else fallback
print(f"\nUsing PDF: {pdf_file.name}")

print("\n[1] Extracting text...")
extractor = PDFExtractor()
result = extractor.extract(pdf_file)
print(f"    Pages      : {result['page_count']}")
print(f"    Characters : {result['metadata']['total_chars']:,}")

print("\n[2] Chunking...")
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(result, "test_doc_001", pdf_file.stem)
print(f"    Chunks created : {len(chunks)}")

print("\n[3] Embedding chunks...")
embedder = DocumentEmbedder()
stats = embedder.add_chunks(chunks)
print(f"    Chunks embedded : {stats['n_chunks_added']}")
print(f"    Embedding dim   : {stats['embedding_dim']}")

print("\n[4] Running semantic search...")
queries = ["food security", "agricultural production", "nutrition"]
for query in queries:
    results = embedder.search(query, top_k=2)
    print(f"\n    Query: '{query}'")
    for r in results:
        preview = r['text'][:100].replace("\n", " ")
        print(f"      Rank {r['rank']} (score={r['score']:.4f}): {preview}...")

print("\n[5] Saving index...")
save_dir = Path("data/processed/rag_index")
embedder.save(save_dir)
print(f"    Saved to: {save_dir}")

print("\n[6] Reloading and searching...")
embedder2 = DocumentEmbedder()
embedder2.load(save_dir)
results2 = embedder2.search("crop yield", top_k=1)
if results2:
    print(f"    Result: score={results2[0]['score']:.4f}")
    print(f"    Text  : {results2[0]['text'][:150]}...")

print("\n" + "="*60)
print("RAG pipeline test complete.")
print("="*60)