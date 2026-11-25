"""
TerraMind AI — Ingestion Metadata Models
-----------------------------------------
Pydantic models that describe every dataset ingested into the platform.
Every connector must produce an IngestionRecord when it runs.

This creates a metadata trail:
- What was fetched
- From where
- When
- In what format
- Under what licence
- Whether it is valid
"""

from pydantic import BaseModel, Field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class IngestionStatus(str, Enum):
    """Tracks the lifecycle of a dataset through ingestion."""
    PENDING = "pending"
    FETCHING = "fetching"
    SUCCESS = "success"
    FAILED = "failed"
    INVALID = "invalid"


class DataFormat(str, Enum):
    """Supported raw data formats."""
    JSON = "json"
    GEOJSON = "geojson"
    CSV = "csv"
    TIFF = "tiff"
    SHAPEFILE = "shapefile"
    PDF = "pdf"
    XML = "xml"


class IngestionRecord(BaseModel):
    """
    Metadata record for one ingested dataset.

    One record is created per ingestion run.
    Records are saved as JSON files in data/metadata/.
    Later these will be stored in PostgreSQL.

    Fields match the data catalogue requirements from the project spec:
    dataset_id, name, source_url, retrieval_timestamp, geographic_coverage,
    industry, format, licence, version, processing_status, validation_status.
    """

    dataset_id: str = Field(
        description="Unique identifier for this dataset, e.g. osm_roads_bengaluru_20240101"
    )
    dataset_name: str = Field(
        description="Human-readable name, e.g. OpenStreetMap Roads — Bengaluru"
    )
    source_url: str = Field(
        description="The URL or API endpoint the data was fetched from"
    )
    retrieval_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the data was fetched"
    )
    geographic_coverage: str = Field(
        description="Human-readable coverage area, e.g. Bengaluru, Karnataka, India"
    )
    industry: str = Field(
        description="Primary industry this dataset serves, e.g. urban, agriculture"
    )
    data_format: DataFormat = Field(
        description="Format of the raw saved file"
    )
    licence: str = Field(
        description="Dataset licence, e.g. ODbL 1.0, CC-BY, Public Domain"
    )
    dataset_version: Optional[str] = Field(
        default=None,
        description="Version string if the source provides one"
    )
    processing_status: IngestionStatus = Field(
        default=IngestionStatus.PENDING
    )
    validation_status: IngestionStatus = Field(
        default=IngestionStatus.PENDING
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Relative path to the saved raw data file"
    )
    record_count: Optional[int] = Field(
        default=None,
        description="Number of records or features in the dataset"
    )
    file_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of the saved file in bytes"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error detail if ingestion failed"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Any additional context about this dataset"
    )