"""
TerraMind AI — Ingestion API Endpoints
----------------------------------------
Exposes data ingestion connectors as REST API endpoints.

Endpoints
---------
POST /ingestion/osm
    Trigger OpenStreetMap data ingestion for a bounding box.

POST /ingestion/nasa-power
    Trigger NASA POWER weather data ingestion for a location.

GET  /ingestion/status/{dataset_id}
    Retrieve the metadata record for a previously ingested dataset.

GET  /ingestion/catalogue
    List all ingested datasets from the metadata directory.

Design note
-----------
Ingestion runs synchronously in this phase.
The HTTP client will wait until the fetch completes.
In Phase 10 this moves to background tasks (Celery + Redis)
so the endpoint returns immediately with a task ID.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engines.ingestion.osm_connector import OSMConnector, BENGALURU_BBOX
from app.engines.ingestion.nasa_power_connector import (
    NASAPowerConnector,
    BENGALURU_LOCATION,
    AGRICULTURE_PARAMETERS,
)
from app.models.ingestion import IngestionRecord, IngestionStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/ingestion", tags=["Ingestion"])


# ---------------------------------------------------------------------------
# Request and response schemas
# ---------------------------------------------------------------------------

class BoundingBox(BaseModel):
    """Geographic bounding box for OSM queries."""
    south: float = Field(default=12.834, description="Southern latitude boundary")
    west:  float = Field(default=77.460, description="Western longitude boundary")
    north: float = Field(default=13.139, description="Northern latitude boundary")
    east:  float = Field(default=77.784, description="Eastern longitude boundary")


class OSMIngestionRequest(BaseModel):
    """Request body for OSM ingestion."""
    feature_type: str = Field(
        default="amenities",
        description="Type of OSM features to fetch: 'roads' or 'amenities'",
    )
    area_name: str = Field(
        default="Bengaluru",
        description="Human-readable name for the study area",
    )
    bbox: BoundingBox = Field(
        default_factory=BoundingBox,
        description="Geographic bounding box to query",
    )


class NASAPowerIngestionRequest(BaseModel):
    """Request body for NASA POWER ingestion."""
    latitude: float = Field(
        default=12.9716,
        description="Latitude of the point to query (decimal degrees)",
    )
    longitude: float = Field(
        default=77.5946,
        description="Longitude of the point to query (decimal degrees)",
    )
    location_name: str = Field(
        default="Bengaluru",
        description="Human-readable location name",
    )
    start_date: Optional[date] = Field(
        default=None,
        description="Start date (YYYY-MM-DD). Defaults to 30 days before end_date.",
    )
    end_date: Optional[date] = Field(
        default=None,
        description="End date (YYYY-MM-DD). Defaults to 8 days ago (NASA data lag).",
    )


class IngestionResponse(BaseModel):
    """Response returned after an ingestion run."""
    dataset_id: str
    status: str
    validation_status: str
    record_count: Optional[int]
    file_size_bytes: Optional[int]
    file_path: Optional[str]
    licence: str
    error_message: Optional[str] = None
    message: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _load_metadata_record(dataset_id: str) -> dict:
    """
    Load a metadata record from the metadata directory.

    Args:
        dataset_id: The dataset ID to look up.

    Returns:
        The metadata dict.

    Raises:
        HTTPException 404 if the record does not exist.
    """
    metadata_path = settings.metadata_dir / f"{dataset_id}.json"

    if not metadata_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No metadata record found for dataset_id: '{dataset_id}'. "
                   f"Run ingestion first.",
        )

    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/osm", response_model=IngestionResponse)
async def ingest_osm(request: OSMIngestionRequest) -> IngestionResponse:
    """
    Trigger OpenStreetMap data ingestion.

    Fetches road or amenity data for the given bounding box
    from the Overpass API and saves it to data/raw/osm/.

    This call blocks until the fetch completes (typically 10-60 seconds).
    """
    logger.info(
        f"OSM ingestion requested: {request.feature_type} "
        f"for {request.area_name}"
    )

    # Validate feature type before hitting the API
    if request.feature_type not in ("roads", "amenities"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature_type '{request.feature_type}'. "
                   f"Must be 'roads' or 'amenities'.",
        )

    connector = OSMConnector()

    bbox_dict = {
        "south": request.bbox.south,
        "west":  request.bbox.west,
        "north": request.bbox.north,
        "east":  request.bbox.east,
    }

    result = connector.fetch(
        bbox=bbox_dict,
        feature_type=request.feature_type,
        area_name=request.area_name,
    )

    record = result["record"]

    if record.processing_status == IngestionStatus.FAILED:
        logger.error(f"OSM ingestion failed: {record.error_message}")
        raise HTTPException(
            status_code=502,
            detail=f"OSM ingestion failed: {record.error_message}",
        )

    return IngestionResponse(
        dataset_id=record.dataset_id,
        status=record.processing_status.value,
        validation_status=record.validation_status.value,
        record_count=record.record_count,
        file_size_bytes=record.file_size_bytes,
        file_path=record.file_path,
        licence=record.licence,
        error_message=record.error_message,
        message=(
            f"OSM {request.feature_type} ingestion complete. "
            f"{record.record_count} elements saved."
        ),
    )


@router.post("/nasa-power", response_model=IngestionResponse)
async def ingest_nasa_power(
    request: NASAPowerIngestionRequest,
) -> IngestionResponse:
    """
    Trigger NASA POWER weather data ingestion.

    Fetches daily weather parameters for the given location
    and date range. Saves raw data to data/raw/nasa_power/.

    NASA POWER has approximately a 7-day data lag.
    Requesting data more recent than 7 days ago may return incomplete results.
    """
    logger.info(
        f"NASA POWER ingestion requested: {request.location_name} "
        f"({request.start_date} to {request.end_date})"
    )

    connector = NASAPowerConnector()

    result = connector.fetch(
        latitude=request.latitude,
        longitude=request.longitude,
        location_name=request.location_name,
        start_date=request.start_date,
        end_date=request.end_date,
    )

    record = result["record"]

    if record.processing_status == IngestionStatus.FAILED:
        logger.error(f"NASA POWER ingestion failed: {record.error_message}")
        raise HTTPException(
            status_code=502,
            detail=f"NASA POWER ingestion failed: {record.error_message}",
        )

    summary = result.get("summary", {})
    summary_str = (
        ", ".join(f"{k}: {v}" for k, v in summary.items())
        if summary else "no summary available"
    )

    return IngestionResponse(
        dataset_id=record.dataset_id,
        status=record.processing_status.value,
        validation_status=record.validation_status.value,
        record_count=record.record_count,
        file_size_bytes=record.file_size_bytes,
        file_path=record.file_path,
        licence=record.licence,
        error_message=record.error_message,
        message=(
            f"NASA POWER ingestion complete. "
            f"{record.record_count} days fetched. "
            f"Summary: {summary_str}"
        ),
    )


@router.get("/status/{dataset_id}")
async def get_ingestion_status(dataset_id: str) -> dict:
    """
    Retrieve the metadata record for a previously ingested dataset.

    Returns the full ingestion record including status, file path,
    record count, licence, and any error messages.
    """
    logger.info(f"Status requested for dataset: {dataset_id}")
    return _load_metadata_record(dataset_id)


@router.get("/catalogue")
async def list_catalogue(
    industry: Optional[str] = Query(
        default=None,
        description="Filter by industry: agriculture, urban, manufacturing",
    ),
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: success, failed, invalid",
    ),
) -> dict:
    """
    List all ingested datasets from the metadata catalogue.

    Optionally filter by industry or processing status.
    Returns datasets in reverse chronological order (newest first).
    """
    logger.info(
        f"Catalogue requested — filters: industry={industry}, status={status}"
    )

    metadata_dir = settings.metadata_dir
    if not metadata_dir.exists():
        return {"total": 0, "datasets": []}

    records = []
    for metadata_file in sorted(
        metadata_dir.glob("*.json"), reverse=True
    ):
        try:
            with open(metadata_file, encoding="utf-8") as f:
                record = json.load(f)

            # Apply filters
            if industry and record.get("industry") != industry:
                continue
            if status and record.get("processing_status") != status:
                continue

            records.append({
                "dataset_id":        record.get("dataset_id"),
                "dataset_name":      record.get("dataset_name"),
                "industry":          record.get("industry"),
                "processing_status": record.get("processing_status"),
                "record_count":      record.get("record_count"),
                "retrieval_timestamp": record.get("retrieval_timestamp"),
                "licence":           record.get("licence"),
            })

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not read metadata file {metadata_file}: {e}")
            continue

    return {
        "total": len(records),
        "filters_applied": {
            "industry": industry,
            "status": status,
        },
        "datasets": records,
    }