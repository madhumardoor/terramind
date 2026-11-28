"""
TerraMind AI - Full RAG Pipeline Test
Runs: PDF extraction -> chunking -> embedding -> retrieval -> LLM answer
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.logging_config import setup_logging
setup_logging(log_level="INFO")

from app.engines.document.pdf_extractor import PDFExtractor
from app.engines.document.chunker import DocumentChunker
from app.engines.document.embedder import DocumentEmbedder
from app.engines.document.rag_generator import RAGGenerator

print("\n" + "="*60)
print("TerraMind AI - Full RAG Pipeline Test")
print("="*60)

# Find available PDF
import glob
import os

pdfs = (
    glob.glob("data/raw/documents/fao_*.pdf") +
    glob.glob("data/raw/documents/worldbank_*.pdf") +
    glob.glob("data/raw/documents/ipcc_*.pdf") +
    glob.glob("data/raw/documents/*.pdf")
)

if not pdfs:
    print("No PDF found. Place any PDF in data/raw/documents/")
    sys.exit(1)

pdf_file = Path(pdfs[0])
print(f"\nUsing: {pdf_file.name}")

# Step 1: Extract
print("\n[1] Extracting text from PDF...")
extractor = PDFExtractor()
result = extractor.extract(pdf_file)
print(f"    Pages      : {result['page_count']}")
print(f"    Characters : {result['metadata']['total_chars']:,}")

# Step 2: Chunk
print("\n[2] Chunking document...")
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(
    result, pdf_file.stem, pdf_file.stem.replace("_", " ").title()
)
print(f"    Chunks : {len(chunks)}")

# Step 3: Embed
print("\n[3] Embedding chunks...")
embedder = DocumentEmbedder()
stats = embedder.add_chunks(chunks)
print(f"    Embedded : {stats['n_chunks_added']} chunks")

# Step 4: RAG answers
print("\n[4] Running RAG question answering...")
generator = RAGGenerator(embedder=embedder, top_k=3)

questions = [
    "What is the main topic of this document?",
    "What are the key findings or recommendations?",
    "What data or statistics are mentioned?",
]

for question in questions:
    print(f"\n    Q: {question}")
    result_qa = generator.answer(question)
    print(f"    A: {result_qa['answer'][:300]}")
    print(f"    Sources: {[s['source_name'] for s in result_qa['sources']]}")
    print(f"    Retrieval scores: {result_qa['retrieval_scores']}")

print("\n" + "="*60)
print("Full RAG pipeline test complete.")
print("="*60)