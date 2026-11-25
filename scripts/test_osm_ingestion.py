"""
TerraMind AI — OSM Ingestion Test Script
------------------------------------------
Runs the OSM connector and prints the result.
This is a manual integration test — it makes a real API call.

Run from the terramind root with .venv active:
    cd backend
    python ..\scripts\test_osm_ingestion.py
"""

import sys
from pathlib import Path

# Add backend to path so app imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.logging_config import setup_logging
from app.core.config import settings
from app.engines.ingestion.osm_connector import OSMConnector, BENGALURU_BBOX

# Set up logging so we see output
setup_logging(log_level="INFO")

def main():
    print("\n" + "="*60)
    print("TerraMind AI — OSM Ingestion Test")
    print("="*60)

    connector = OSMConnector()

    print("\n[1] Fetching amenities data for Bengaluru...")
    print("    (This makes a real API call — may take 10-30 seconds)\n")

    result = connector.fetch(
        bbox=BENGALURU_BBOX,
        feature_type="amenities",
        area_name="Bengaluru",
    )

    record = result["record"]
    data   = result["data"]

    print("\n" + "-"*60)
    print("INGESTION RESULT")
    print("-"*60)
    print(f"Dataset ID       : {record.dataset_id}")
    print(f"Status           : {record.processing_status}")
    print(f"Validation       : {record.validation_status}")
    print(f"Records fetched  : {record.record_count}")
    print(f"File size        : {record.file_size_bytes:,} bytes" if record.file_size_bytes else "File size: N/A")
    print(f"Saved to         : {record.file_path}")
    print(f"Licence          : {record.licence}")

    if record.error_message:
        print(f"Error            : {record.error_message}")

    if data and data.get("elements"):
        print("\nSample elements (first 3):")
        for el in data["elements"][:3]:
            tags = el.get("tags", {})
            amenity = tags.get("amenity", "unknown")
            name    = tags.get("name", "unnamed")
            print(f"  - {amenity}: {name}")

    print("\n" + "="*60)
    print("Check these directories for output files:")
    print(f"  Raw data : {settings.raw_data_dir / 'osm'}")
    print(f"  Metadata : {settings.metadata_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()