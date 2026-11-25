"""
TerraMind AI — Geospatial Engine Test Script
----------------------------------------------
Tests the OSM processor against the real ingested data file.
No API calls — reads from data/raw/osm/ directly.

Run from terramind root with .venv active:
    cd backend
    python ..\scripts\test_geospatial_engine.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.logging_config import setup_logging
from app.core.config import settings
from app.engines.geospatial.osm_processor import OSMProcessor

setup_logging(log_level="INFO")


def main():
    print("\n" + "=" * 60)
    print("TerraMind AI — Geospatial Engine Test")
    print("=" * 60)

    # Find the most recent OSM amenities file
    osm_dir = settings.raw_data_dir / "osm"
    osm_files = sorted(osm_dir.glob("osm_amenities_*.json"), reverse=True)

    if not osm_files:
        print("ERROR: No OSM amenity files found.")
        print(f"Run the OSM ingestion first.")
        return

    osm_file = osm_files[0]
    print(f"\nUsing file: {osm_file.name}")

    processor = OSMProcessor()

    # --- Test 1: Load all features ---
    print("\n[1] Loading all OSM nodes into GeoDataFrame...")
    gdf = processor.load_from_file(osm_file)
    print(f"    Total features : {len(gdf)}")
    print(f"    CRS            : {gdf.crs}")
    print(f"    Columns        : {list(gdf.columns)}")

    # --- Test 2: Amenity summary ---
    print("\n[2] Amenity type summary (top 10):")
    summary = processor.get_amenity_summary(gdf)
    for amenity, count in list(summary.items())[:10]:
        print(f"    {amenity:<20} : {count}")

    # --- Test 3: Filter by amenity ---
    print("\n[3] Filtering to hospitals...")
    hospitals = processor.filter_by_amenity(gdf, "hospital")
    print(f"    Hospitals found : {len(hospitals)}")
    if len(hospitals) > 0:
        print("    First 3:")
        for _, row in hospitals.head(3).iterrows():
            print(f"      - {row['name']} ({row['lat']:.4f}, {row['lon']:.4f})")

    # --- Test 4: Proximity query ---
    print("\n[4] Finding amenities within 3km of Bengaluru city centre...")
    print("    Centre: 12.9716°N, 77.5946°E (MG Road area)")

    nearby = processor.find_within_radius(
        gdf=gdf,
        centre_lat=12.9716,
        centre_lon=77.5946,
        radius_metres=3000,
    )
    print(f"    Features within 3km : {len(nearby)}")

    if len(nearby) > 0:
        print("    Closest 5 features:")
        for _, row in nearby.head(5).iterrows():
            print(
                f"      {row['distance_m']:>7.0f}m — "
                f"{row['amenity']:<15} — {row['name']}"
            )

    # --- Test 5: Hospitals within 5km ---
    print("\n[5] Finding hospitals within 5km of city centre...")
    hospitals_nearby = processor.find_within_radius(
        gdf=processor.filter_by_amenity(gdf, "hospital"),
        centre_lat=12.9716,
        centre_lon=77.5946,
        radius_metres=5000,
    )
    print(f"    Hospitals within 5km : {len(hospitals_nearby)}")
    if len(hospitals_nearby) > 0:
        for _, row in hospitals_nearby.head(5).iterrows():
            print(f"      {row['distance_m']:>7.0f}m — {row['name']}")

    print("\n" + "=" * 60)
    print("Geospatial engine test complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()