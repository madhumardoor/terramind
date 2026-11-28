import sys
from pathlib import Path
import numpy as np

backend_dir = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.core.logging_config import setup_logging
setup_logging(log_level="INFO")

from app.engines.geospatial.satellite_processor import SatelliteProcessor

print("\n" + "="*60)
print("TerraMind AI - NDVI Engine Test")
print("="*60)

processor = SatelliteProcessor()

print("\n[1] Generating synthetic NDVI scene...")
result = processor.generate_synthetic_ndvi(rows=200, cols=200)
ndvi   = result["ndvi_array"]
stats  = result["statistics"]

print(f"    Shape          : {ndvi.shape}")
print(f"    NDVI min       : {stats['min']:.4f}")
print(f"    NDVI max       : {stats['max']:.4f}")
print(f"    NDVI mean      : {stats['mean']:.4f}")
print(f"    Dominant class : {stats['dominant_class']}")

print("\n[2] Land cover breakdown:")
for class_name, pct in stats["class_coverage"].items():
    bar = "#" * int(pct / 2)
    print(f"    {class_name:<25} : {pct:5.1f}% {bar}")

print("\n[3] Testing manual NDVI calculation...")
red = np.array([[0.05, 0.10], [0.08, 0.20]])
nir = np.array([[0.45, 0.60], [0.55, 0.30]])
ndvi_manual = processor.calculate_ndvi(red, nir)
expected_00 = (0.45 - 0.05) / (0.45 + 0.05)
print(f"    Expected [0,0]: {expected_00:.4f}")
print(f"    Got [0,0]     : {ndvi_manual[0,0]:.4f}")
assert abs(ndvi_manual[0,0] - expected_00) < 0.001, "NDVI calculation error"
print("    Calculation verified correctly")

print("\n[4] Checking for real Sentinel-2 data...")
satellite_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "satellite"
safe_folders  = list(satellite_dir.glob("*.SAFE")) if satellite_dir.exists() else []
zip_files     = list(satellite_dir.glob("*.zip")) if satellite_dir.exists() else []

if safe_folders:
    print(f"    Found .SAFE folder: {safe_folders[0].name}")
    output_dir  = Path(__file__).resolve().parent.parent / "data" / "processed" / "satellite"
    real_result = processor.process_safe_folder(safe_folders[0], output_dir)
    real_stats  = real_result["statistics"]
    print(f"    Tile        : {real_result['tile_name']}")
    print(f"    NDVI mean   : {real_stats['mean']:.4f}")
    print(f"    Dominant    : {real_stats['dominant_class']}")
elif zip_files:
    print(f"    Found zip: {zip_files[0].name}")
    print("    Please extract the zip to get the .SAFE folder")
else:
    print("    No Sentinel-2 data found yet.")
    print("    Synthetic test completed successfully above.")

print("\n" + "="*60)
print("NDVI engine test complete.")
print("="*60)