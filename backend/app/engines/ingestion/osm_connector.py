"""
TerraMind AI — OpenStreetMap Ingestion Connector
--------------------------------------------------
Fetches road network and amenity data from the OpenStreetMap Overpass API.

Data licence: OpenStreetMap data is © OpenStreetMap contributors,
licensed under the Open Database Licence (ODbL 1.0).
Attribution is required for any public use.
Reference: https://www.openstreetmap.org/copyright

Overpass API documentation: https://wiki.openstreetmap.org/wiki/Overpass_API
No API key required. Rate limits apply — do not hammer the endpoint.
"""

import time
import requests
from datetime import datetime, timezone
from typing import Any

from app.core.config import settings
from app.core.logging_config import get_logger
from app.engines.ingestion.base import BaseConnector
from app.models.ingestion import (
    DataFormat,
    IngestionRecord,
    IngestionStatus,
)

logger = get_logger(__name__)


# Bounding box for Bengaluru, Karnataka, India
# Format: (south, west, north, east) in decimal degrees
BENGALURU_BBOX = {
    "south": 12.834,
    "west": 77.460,
    "north": 13.139,
    "east": 77.784,
}


class OSMConnector(BaseConnector):
    """
    Connector for the OpenStreetMap Overpass API.

    Fetches vector features (roads, amenities, buildings) for a
    specified geographic bounding box.

    Usage:
        connector = OSMConnector()
        result = connector.fetch(
            bbox=BENGALURU_BBOX,
            feature_type="roads",
            area_name="Bengaluru"
        )
    """

    def __init__(self):
        super().__init__(connector_name="osm")
        self.base_url = settings.osm_overpass_url
        self.timeout_seconds = 60   # Overpass can be slow for large areas
        self.retry_attempts = 3
        self.retry_delay_seconds = 5

    def _build_roads_query(self, bbox: dict) -> str:
        """
        Build an Overpass QL query for the road network.

        highway=* captures all road types:
        motorway, trunk, primary, secondary, tertiary, residential, etc.

        Returns GeoJSON format for easy processing downstream.
        """
        south = bbox["south"]
        west  = bbox["west"]
        north = bbox["north"]
        east  = bbox["east"]

        return f"""
        [out:json][timeout:60];
        (
          way["highway"]({south},{west},{north},{east});
        );
        out geom;
        """

    def _build_amenities_query(self, bbox: dict) -> str:
        """
        Build an Overpass QL query for key urban amenities.

        Fetches: hospitals, schools, universities, banks, restaurants.
        These are useful for urban intelligence and accessibility analysis.
        """
        south = bbox["south"]
        west  = bbox["west"]
        north = bbox["north"]
        east  = bbox["east"]

        return f"""
        [out:json][timeout:60];
        (
          node["amenity"~"hospital|school|university|bank|restaurant"]
            ({south},{west},{north},{east});
        );
        out body;
        """

    def _execute_query(self, query: str) -> dict:
        """
        Send a query to the Overpass API with retry logic.

        Args:
            query: Overpass QL query string.

        Returns:
            Parsed JSON response as a dict.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self.logger.info(
                    f"Overpass API request — attempt {attempt}/{self.retry_attempts}"
                )

                response = requests.post(
                    self.base_url,
                    data={"data": query},
                    timeout=self.timeout_seconds,
                    headers={"User-Agent": "TerraMindAI/0.1 (portfolio research project)"},
                )

                # Raise an exception for HTTP error codes (4xx, 5xx)
                response.raise_for_status()

                data = response.json()
                self.logger.info(
                    f"Overpass API response received — "
                    f"{len(data.get('elements', []))} elements"
                )
                return data

            except requests.exceptions.Timeout:
                last_error = f"Request timed out after {self.timeout_seconds}s"
                self.logger.warning(f"Attempt {attempt} timed out")

            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP error: {e}"
                self.logger.warning(f"Attempt {attempt} HTTP error: {e}")

            except requests.exceptions.RequestException as e:
                last_error = f"Request error: {e}"
                self.logger.warning(f"Attempt {attempt} request error: {e}")

            if attempt < self.retry_attempts:
                self.logger.info(f"Retrying in {self.retry_delay_seconds}s...")
                time.sleep(self.retry_delay_seconds)

        raise RuntimeError(
            f"Overpass API failed after {self.retry_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def validate(self, data: Any) -> bool:
        """
        Validate that the Overpass response is usable.

        A valid response must:
        - Be a dict
        - Contain the 'elements' key
        - Have at least one element
        """
        if not isinstance(data, dict):
            self.logger.error("Validation failed: response is not a dict")
            return False

        if "elements" not in data:
            self.logger.error("Validation failed: 'elements' key missing")
            return False

        element_count = len(data["elements"])
        if element_count == 0:
            self.logger.warning("Validation warning: response contains 0 elements")
            return False

        self.logger.info(f"Validation passed: {element_count} elements found")
        return True

    def fetch(
        self,
        bbox: dict = None,
        feature_type: str = "roads",
        area_name: str = "Bengaluru",
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fetch OSM data for a bounding box.

        Args:
            bbox:         Bounding box dict with south/west/north/east keys.
                          Defaults to BENGALURU_BBOX.
            feature_type: "roads" or "amenities".
            area_name:    Human-readable location name for metadata.

        Returns:
            dict with keys:
                "data"   — the raw Overpass response
                "record" — the completed IngestionRecord
        """
        if bbox is None:
            bbox = BENGALURU_BBOX

        # Build a unique dataset ID using feature type and date
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        dataset_id = f"osm_{feature_type}_{area_name.lower().replace(' ', '_')}_{date_str}"

        # Initialise the metadata record
        record = IngestionRecord(
            dataset_id=dataset_id,
            dataset_name=f"OpenStreetMap {feature_type.title()} — {area_name}",
            source_url=self.base_url,
            geographic_coverage=f"{area_name}, Karnataka, India",
            industry="urban",
            data_format=DataFormat.JSON,
            licence="ODbL 1.0 — © OpenStreetMap contributors",
            processing_status=IngestionStatus.FETCHING,
            validation_status=IngestionStatus.PENDING,
            notes=f"Bounding box: {bbox}",
        )

        self.logger.info(
            f"Starting OSM ingestion: {feature_type} for {area_name}"
        )

        try:
            # Build the appropriate query
            if feature_type == "roads":
                query = self._build_roads_query(bbox)
            elif feature_type == "amenities":
                query = self._build_amenities_query(bbox)
            else:
                raise ValueError(
                    f"Unknown feature_type '{feature_type}'. "
                    f"Use 'roads' or 'amenities'."
                )

            # Execute query against Overpass API
            raw_data = self._execute_query(query)

            # Validate the response
            is_valid = self.validate(raw_data)
            record.validation_status = (
                IngestionStatus.SUCCESS if is_valid else IngestionStatus.INVALID
            )

            if not is_valid:
                record.processing_status = IngestionStatus.INVALID
                record.error_message = "Validation failed — empty or malformed response"
                self.save_metadata(record)
                return {"data": None, "record": record}

            # Save raw data
            filename = f"{dataset_id}.json"
            file_path = self.save_raw(
                data=raw_data,
                filename=filename,
                subdirectory="osm",
            )

            # Update record with results
            record.processing_status = IngestionStatus.SUCCESS
            record.file_path = str(file_path)
            record.record_count = len(raw_data.get("elements", []))
            record.file_size_bytes = file_path.stat().st_size

            self.logger.info(
                f"OSM ingestion complete: {record.record_count} elements, "
                f"{record.file_size_bytes:,} bytes"
            )

        except Exception as e:
            self.logger.error(f"OSM ingestion failed: {e}")
            record.processing_status = IngestionStatus.FAILED
            record.error_message = str(e)

        finally:
            # Always save metadata, even on failure
            self.save_metadata(record)

        return {"data": raw_data if record.processing_status == IngestionStatus.SUCCESS else None,
                "record": record}