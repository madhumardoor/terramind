"""
TerraMind AI - Satellite Intelligence API
-------------------------------------------
Exposes NDVI and satellite analysis as REST endpoints.

Endpoints
---------
GET /satellite/ndvi/latest
    Return NDVI statistics from the most recently processed tile.

GET /satellite/ndvi/tiles
    List all processed NDVI tiles.

POST /satellite/ndvi/process
    Trigger NDVI processing for the latest available .SAFE folder.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.core.config import settings
from app.core.logging_config import get_logger
from app.engines.geospatial.satellite_processor import SatelliteProcessor

logger = get_logger(__name__)
router = APIRouter(prefix="/satellite", tags=["Satellite"])

SATELLITE_RAW_DIR       = Path("../data/raw/satellite")
SATELLITE_PROCESSED_DIR = Path("../data/processed/satellite")


@router.get("/ndvi/tiles")
async def list_ndvi_tiles() -> dict:
    """List all processed NDVI tiles and their statistics."""
    if not SATELLITE_PROCESSED_DIR.exists():
        return {"total": 0, "tiles": []}

    stats_files = sorted(
        SATELLITE_PROCESSED_DIR.glob("*_NDVI_stats.json"),
        reverse=True,
    )

    tiles = []
    for stats_file in stats_files:
        try:
            with open(stats_file, encoding="utf-8") as f:
                stats = json.load(f)
            tile_name = stats_file.stem.replace("_NDVI_stats", "")
            tif_path  = SATELLITE_PROCESSED_DIR / f"{tile_name}_NDVI.tif"
            tiles.append({
                "tile_name":    tile_name,
                "statistics":   stats,
                "tif_exists":   tif_path.exists(),
                "tif_size_mb":  round(tif_path.stat().st_size / 1e6, 1) if tif_path.exists() else None,
            })
        except Exception as e:
            logger.warning(f"Could not read stats file {stats_file}: {e}")

    return {"total": len(tiles), "tiles": tiles}


@router.get("/ndvi/latest")
async def get_latest_ndvi() -> dict:
    """Return NDVI statistics from the most recently processed tile."""
    if not SATELLITE_PROCESSED_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail="No processed satellite data found. Run POST /satellite/ndvi/process first.",
        )

    stats_files = sorted(
        SATELLITE_PROCESSED_DIR.glob("*_NDVI_stats.json"),
        reverse=True,
    )

    if not stats_files:
        raise HTTPException(
            status_code=404,
            detail="No NDVI statistics found. Run POST /satellite/ndvi/process first.",
        )

    with open(stats_files[0], encoding="utf-8") as f:
        stats = json.load(f)

    tile_name = stats_files[0].stem.replace("_NDVI_stats", "")

    return {
        "tile_name":   tile_name,
        "statistics":  stats,
        "licence":     "Contains modified Copernicus Sentinel data",
        "interpretation": {
            "mean_ndvi":       stats.get("mean"),
            "dominant_class":  stats.get("dominant_class"),
            "vegetation_pct":  round(
                stats.get("class_coverage", {}).get("dense_vegetation", 0) +
                stats.get("class_coverage", {}).get("moderate_vegetation", 0), 2
            ),
        },
    }


@router.post("/ndvi/process")
async def process_ndvi() -> dict:
    """
    Trigger NDVI processing for the latest available Sentinel-2 .SAFE folder.
    Saves output GeoTIFF and statistics to data/processed/satellite/.
    """
    if not SATELLITE_RAW_DIR.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Satellite data directory not found: {SATELLITE_RAW_DIR}",
        )

    safe_folders = sorted(SATELLITE_RAW_DIR.glob("*.SAFE"), reverse=True)

    if not safe_folders:
        raise HTTPException(
            status_code=404,
            detail="No .SAFE folders found. Download a Sentinel-2 tile first.",
        )

    safe_folder = safe_folders[0]
    logger.info(f"Processing NDVI for: {safe_folder.name}")

    try:
        processor = SatelliteProcessor()
        result    = processor.process_safe_folder(
            safe_folder,
            output_dir=SATELLITE_PROCESSED_DIR,
        )
        stats = result["statistics"]

        return {
            "status":       "success",
            "tile_name":    result["tile_name"],
            "shape":        list(result["shape"]),
            "output_path":  result["output_path"],
            "statistics":   stats,
            "licence":      result["licence"],
            "message":      (
                f"NDVI processed successfully. "
                f"Mean NDVI: {stats.get('mean', 'N/A')}, "
                f"Dominant class: {stats.get('dominant_class', 'N/A')}"
            ),
        }

    except Exception as e:
        logger.error(f"NDVI processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))