
"""
TerraMind AI - Satellite Image Processor
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import rasterio
from rasterio.enums import Resampling
from app.core.logging_config import get_logger

logger = get_logger(__name__)

SENTINEL2_SCALE = 10000.0

NDVI_CLASSES = {
    "dense_vegetation": (0.6, 1.0),
    "moderate_vegetation": (0.2, 0.6),
    "sparse_vegetation": (0.0, 0.2),
    "non_vegetation": (-1.0, 0.0),
}


class SatelliteProcessor:

    def find_band_file(self, safe_folder, band, resolution="10m"):
        pattern = f"**/*{band}_{resolution}.jp2"
        matches = list(Path(safe_folder).glob(pattern))
        if not matches:
            logger.warning(f"Band {band} at {resolution} not found in {Path(safe_folder).name}")
            return None
        logger.info(f"Found band {band}: {matches[0].name}")
        return matches[0]

    def load_band(self, band_path, target_shape=None):
        with rasterio.open(band_path) as src:
            profile = src.profile.copy()
            if target_shape and (src.height, src.width) != target_shape:
                data = src.read(1, out_shape=(target_shape[0], target_shape[1]),
                                resampling=Resampling.bilinear).astype(np.float32)
            else:
                data = src.read(1).astype(np.float32)
        data = data / SENTINEL2_SCALE
        data = np.clip(data, 0.0, 1.0)
        logger.info(f"Loaded band {Path(band_path).name}: shape={data.shape}, min={data.min():.4f}, max={data.max():.4f}")
        return data, profile

    def calculate_ndvi(self, red_array, nir_array):
        red_array = np.asarray(red_array, dtype=np.float32)
        nir_array = np.asarray(nir_array, dtype=np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            numerator = nir_array - red_array
            denominator = nir_array + red_array
            ndvi = np.where(denominator != 0, numerator / denominator, 0.0)
        ndvi = ndvi.astype(np.float32)
        ndvi = np.clip(ndvi, -1.0, 1.0)
        logger.info(f"NDVI calculated: shape={ndvi.shape}, min={ndvi.min():.4f}, max={ndvi.max():.4f}, mean={ndvi.mean():.4f}")
        return ndvi

    def get_ndvi_statistics(self, ndvi):
        valid_mask = ndvi != 0
        valid_ndvi = ndvi[valid_mask]
        if len(valid_ndvi) == 0:
            logger.warning("No valid NDVI pixels found")
            return {}
        total_pixels = len(valid_ndvi)
        class_coverage = {}
        for class_name, (low, high) in NDVI_CLASSES.items():
            class_pixels = np.sum((valid_ndvi >= low) & (valid_ndvi < high))
            class_coverage[class_name] = round(float(class_pixels / total_pixels * 100), 2)
        stats = {
            "min":    round(float(valid_ndvi.min()), 4),
            "max":    round(float(valid_ndvi.max()), 4),
            "mean":   round(float(valid_ndvi.mean()), 4),
            "median": round(float(np.median(valid_ndvi)), 4),
            "std":    round(float(valid_ndvi.std()), 4),
            "p25":    round(float(np.percentile(valid_ndvi, 25)), 4),
            "p75":    round(float(np.percentile(valid_ndvi, 75)), 4),
            "valid_pixels":   int(total_pixels),
            "class_coverage": class_coverage,
            "dominant_class": max(class_coverage, key=class_coverage.get),
        }
        logger.info(f"NDVI stats: mean={stats['mean']:.4f}, dominant={stats['dominant_class']}")
        return stats

    def save_ndvi_geotiff(self, ndvi, profile, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        profile.update(dtype=rasterio.float32, count=1, compress="lzw", driver="GTiff")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndvi, 1)
        logger.info(f"NDVI GeoTIFF saved: {output_path} ({output_path.stat().st_size:,} bytes)")
        return output_path

    def process_safe_folder(self, safe_folder, output_dir=None):
        safe_folder = Path(safe_folder)
        if output_dir is None:
            output_dir = Path("../data/processed/satellite")
        logger.info(f"Processing Sentinel-2 tile: {safe_folder.name}")
        red_path = self.find_band_file(safe_folder, "B04", "10m")
        nir_path = self.find_band_file(safe_folder, "B08", "10m")
        if not red_path or not nir_path:
            raise FileNotFoundError(f"Could not find B04 or B08 bands in {safe_folder}")
        red_array, profile = self.load_band(red_path)
        nir_array, _ = self.load_band(nir_path, target_shape=red_array.shape)
        ndvi = self.calculate_ndvi(red_array, nir_array)
        stats = self.get_ndvi_statistics(ndvi)
        tile_name = safe_folder.stem.replace(".SAFE", "")
        output_path = Path(output_dir) / f"{tile_name}_NDVI.tif"
        self.save_ndvi_geotiff(ndvi, profile, output_path)
        stats_path = Path(output_dir) / f"{tile_name}_NDVI_stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        result = {
            "tile_name":  tile_name,
            "ndvi_array": ndvi,
            "statistics": stats,
            "output_path": str(output_path),
            "stats_path": str(stats_path),
            "shape":      ndvi.shape,
            "licence":    "Contains modified Copernicus Sentinel data",
        }
        logger.info(f"Processing complete: {tile_name} NDVI mean={stats.get('mean', 'N/A')}")
        return result

    def generate_synthetic_ndvi(self, rows=100, cols=100, seed=42):
        rng = np.random.default_rng(seed)
        ndvi = np.zeros((rows, cols), dtype=np.float32)
        ndvi[:rows//2, :cols//2] = rng.uniform(0.0, 0.15, (rows//2, cols//2))
        ndvi[:rows//2, cols//2:] = rng.uniform(0.25, 0.55, (rows//2, cols//2))
        ndvi[rows//2:, :cols//2] = rng.uniform(0.55, 0.85, (rows//2, cols//2))
        ndvi[rows//2:, cols//2:] = rng.uniform(-0.3, -0.05, (rows//2, cols//2))
        stats = self.get_ndvi_statistics(ndvi)
        logger.info("Generated synthetic NDVI scene for testing")
        return {"ndvi_array": ndvi, "statistics": stats, "synthetic": True}
