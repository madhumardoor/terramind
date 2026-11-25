"""
TerraMind AI — Geospatial Intelligence API Endpoints
------------------------------------------------------
Exposes spatial analysis capabilities as REST endpoints.

Endpoints
---------
GET /geospatial/osm/summary
    Amenity type counts for the loaded OSM dataset.

GET /geospatial/osm/amenities
    List amenities, optionally filtered by type.

GET /geospatial/osm/nearby
    Find amenities within a radius of a given point.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engines.geospatial.osm_processor import OSMProcessor

logger = get_logger(__name__)

router = APIRouter(prefix="/geospatial", tags=["Geospatial"])


def _get_latest_osm_file() -> Path:
    """
    Find the most recently ingested OSM amenities file.

    Raises:
        HTTPException 404 if no OSM data has been ingested yet.
    """
    osm_dir = settings.raw_data_dir / "osm"

    if not osm_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="No OSM data directory found. Run POST /ingestion/osm first.",
        )

    osm_files = sorted(osm_dir.glob("osm_amenities_*.json"), reverse=True)

    if not osm_files:
        raise HTTPException(
            status_code=404,
            detail="No OSM amenity files found. Run POST /ingestion/osm first.",
        )

    logger.info(f"Using OSM file: {osm_files[0].name}")
    return osm_files[0]


@router.get("/osm/summary")
async def get_osm_summary() -> dict:
    """
    Return a count of each amenity type in the ingested OSM dataset.
    """
    file_path = _get_latest_osm_file()
    processor = OSMProcessor()

    try:
        gdf = processor.load_from_file(file_path)
        summary = processor.get_amenity_summary(gdf)

        return {
            "file": file_path.name,
            "total_features": len(gdf),
            "unique_amenity_types": len(summary),
            "amenity_counts": summary,
        }

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/osm/amenities")
async def get_amenities(
    amenity_type: Optional[str] = Query(
        default=None,
        description="Filter by amenity type, e.g. hospital, school, bank",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of features to return",
    ),
) -> dict:
    """
    Return amenity features from the ingested OSM dataset.
    Optionally filter by amenity type.
    """
    file_path = _get_latest_osm_file()
    processor = OSMProcessor()

    try:
        gdf = processor.load_from_file(file_path)

        if amenity_type:
            gdf = processor.filter_by_amenity(gdf, amenity_type)

        gdf = gdf.head(limit)
        geojson = processor.to_geojson_dict(gdf)

        return {
            "amenity_filter": amenity_type,
            "returned": len(gdf),
            "limit": limit,
            "geojson": geojson,
        }

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/osm/nearby")
async def get_nearby_amenities(
    lat: float = Query(
        default=12.9716,
        description="Centre latitude in decimal degrees",
    ),
    lon: float = Query(
        default=77.5946,
        description="Centre longitude in decimal degrees",
    ),
    radius_m: float = Query(
        default=3000,
        ge=100,
        le=50000,
        description="Search radius in metres (100 to 50000)",
    ),
    amenity_type: Optional[str] = Query(
        default=None,
        description="Filter by amenity type, e.g. hospital, school",
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of results to return",
    ),
) -> dict:
    """
    Find amenities within a radius of a given point.
    Uses UTM projection for accurate metre-based distance calculations.
    Results sorted by distance from centre point.
    """
    file_path = _get_latest_osm_file()
    processor = OSMProcessor()

    try:
        gdf = processor.load_from_file(file_path)

        if amenity_type:
            gdf = processor.filter_by_amenity(gdf, amenity_type)

        nearby = processor.find_within_radius(
            gdf=gdf,
            centre_lat=lat,
            centre_lon=lon,
            radius_metres=radius_m,
        )

        nearby = nearby.head(limit)
        geojson = processor.to_geojson_dict(nearby)

        return {
            "centre": {"lat": lat, "lon": lon},
            "radius_m": radius_m,
            "amenity_filter": amenity_type,
            "total_found": len(nearby),
            "limit": limit,
            "geojson": geojson,
        }

    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))