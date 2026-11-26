import sys
from pathlib import Path

# Write a minimal test PDF using reportlab if available, otherwise use a sample
backend_dir = Path("backend")
sys.path.insert(0, str(backend_dir))

from app.core.logging_config import setup_logging
setup_logging(log_level="INFO")

from app.engines.document.pdf_extractor import PDFExtractor
from app.engines.document.chunker import DocumentChunker

print("\n" + "="*60)
print("TerraMind AI - Document Intelligence Engine Test")
print("="*60)

# Download a small public domain PDF for testing
import urllib.request

pdf_url = "https://www.w3.org/WAI/WCAG21/wcag21.pdf"
test_pdf_path = Path("data/raw/documents")
test_pdf_path.mkdir(parents=True, exist_ok=True)
pdf_file = test_pdf_path / "test_document.pdf"

# Use a smaller public PDF
small_pdf_url = "https://www.africau.edu/images/default/sample.pdf"

print("\n[1] Downloading sample PDF...")
try:
    urllib.request.urlretrieve(small_pdf_url, pdf_file)
    print(f"    Downloaded to: {pdf_file}")
except Exception as e:
    print(f"    Download failed: {e}")
    print("    Creating a minimal test PDF manually...")
    # Create minimal PDF manually if download fails
    minimal_pdf = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 200>>stream
BT /F1 12 Tf 72 720 Td
(TerraMind AI Test Document) Tj
0 -20 Td (This is a test document for the document intelligence engine.) Tj
0 -20 Td (It contains information about crop monitoring and agricultural analysis.) Tj
0 -20 Td (NDVI values indicate vegetation health across different seasons.) Tj
0 -20 Td (Rainfall data from NASA POWER shows precipitation patterns.) Tj
ET
endstream endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000266 00000 n
0000000517 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
585
%%EOF"""
    with open(pdf_file, "wb") as f:
        f.write(minimal_pdf)
    print(f"    Created minimal test PDF: {pdf_file}")

print("\n[2] Extracting text from PDF...")
extractor = PDFExtractor()
result = extractor.extract(pdf_file)
print(f"    Pages extracted  : {result['page_count']}")
print(f"    Total characters : {result['metadata']['total_chars']:,}")
print(f"    File size        : {result['metadata']['file_size_bytes']:,} bytes")
if result['metadata']['title']:
    print(f"    Title            : {result['metadata']['title']}")

print("\n    First 300 characters of extracted text:")
print("   ", result['full_text'][:300].replace('\n', ' '))

print("\n[3] Chunking the document...")
chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
chunks = chunker.chunk_document(
    extraction_result=result,
    document_id="test_document_001",
    source_name="TerraMind Test Document",
    strategy="full_document",
)
print(f"    Total chunks created : {len(chunks)}")
if chunks:
    print(f"    First chunk ID       : {chunks[0]['chunk_id']}")
    print(f"    First chunk chars    : {chunks[0]['char_count']}")
    print(f"    First chunk preview  : {chunks[0]['text'][:150]}...")

print("\n[4] Testing page-by-page chunking...")
page_chunks = chunker.chunk_document(
    extraction_result=result,
    document_id="test_document_001",
    source_name="TerraMind Test Document",
    strategy="page_by_page",
)
print(f"    Total chunks (page-by-page): {len(page_chunks)}")

print("\n" + "="*60)
print("Document intelligence engine test complete.")
print("="*60)