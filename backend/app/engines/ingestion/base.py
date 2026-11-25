"""
TerraMind AI — Base Ingestion Connector
-----------------------------------------
Abstract base class that all data source connectors must inherit from.

This enforces a consistent interface across every connector:
- OSM connector
- NASA POWER connector
- Sentinel-2 connector
- Industrial dataset connector
- Document connector

Every connector must implement fetch() and validate().
The save_raw() and save_metadata() methods are shared utilities.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger
from app.models.ingestion import IngestionRecord, IngestionStatus

logger = get_logger(__name__)


class BaseConnector(ABC):
    """
    Abstract base class for all TerraMind AI data connectors.

    Subclasses must implement:
        fetch()    — retrieve data from the source
        validate() — check that the fetched data is usable

    Subclasses inherit:
        save_raw()      — save raw data to data/raw/
        save_metadata() — save ingestion record to data/metadata/
    """

    def __init__(self, connector_name: str):
        """
        Args:
            connector_name: Short identifier, e.g. "osm", "nasa_power"
        """
        self.connector_name = connector_name
        self.logger = get_logger(f"connector.{connector_name}")

    @abstractmethod
    def fetch(self, **kwargs) -> dict[str, Any]:
        """
        Fetch raw data from the source.

        Must return a dict with at least:
            {
                "data": <the raw fetched content>,
                "record": <IngestionRecord with status updated>
            }
        """
        pass

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        Validate that fetched data meets minimum quality requirements.

        Args:
            data: The raw fetched content to validate.

        Returns:
            True if data is valid and usable, False otherwise.
        """
        pass

    def save_raw(
        self,
        data: Any,
        filename: str,
        subdirectory: str = "",
    ) -> Path:
        """
        Save raw fetched data to data/raw/.

        Args:
            data:         The data to save (dict or list → JSON).
            filename:     The filename to save as, e.g. osm_roads_20240101.json
            subdirectory: Optional subfolder inside raw/, e.g. "osm" or "nasa"

        Returns:
            The Path where the file was saved.
        """
        save_dir = settings.raw_data_dir
        if subdirectory:
            save_dir = save_dir / subdirectory

        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / filename

        if isinstance(data, (dict, list)):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(data))

        file_size = file_path.stat().st_size
        self.logger.info(
            f"Raw data saved: {file_path} ({file_size:,} bytes)"
        )
        return file_path

    def save_metadata(self, record: IngestionRecord) -> Path:
        """
        Save the ingestion metadata record to data/metadata/.

        Args:
            record: The completed IngestionRecord for this ingestion run.

        Returns:
            The Path where the metadata was saved.
        """
        metadata_dir = settings.metadata_dir
        metadata_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{record.dataset_id}.json"
        file_path = metadata_dir / filename

        with open(file_path, "w", encoding="utf-8") as f:
            # model_dump_json gives us proper datetime serialization
            f.write(record.model_dump_json(indent=2))

        self.logger.info(f"Metadata saved: {file_path}")
        return file_path