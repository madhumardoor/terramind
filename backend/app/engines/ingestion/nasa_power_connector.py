"""
TerraMind AI — NASA POWER Weather Ingestion Connector
------------------------------------------------------
Fetches daily meteorological data from the NASA POWER API.

Data source  : NASA Prediction of Worldwide Energy Resources (POWER)
API docs     : https://power.larc.nasa.gov/docs/services/api/
Licence      : NASA Open Data — public domain, free for any use
               https://www.earthdata.nasa.gov/learn/use-data/data-use-policy
No API key required. No registration required.

Parameters fetched
------------------
T2M             : Temperature at 2 metres (degrees C)
T2M_MAX         : Maximum temperature at 2 metres (degrees C)
T2M_MIN         : Minimum temperature at 2 metres (degrees C)
PRECTOTCORR     : Precipitation corrected (mm/day)
RH2M            : Relative humidity at 2 metres (%)
WS2M            : Wind speed at 2 metres (m/s)
ALLSKY_SFC_SW_DWN : Solar irradiance (kW-hr/m2/day)
"""

import time
import requests
from datetime import datetime, timezone, date, timedelta
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


BENGALURU_LOCATION = {
    "latitude": 12.9716,
    "longitude": 77.5946,
    "name": "Bengaluru",
}

AGRICULTURE_PARAMETERS = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "ALLSKY_SFC_SW_DWN",
]


class NASAPowerConnector(BaseConnector):
    """
    Connector for the NASA POWER Daily Point API.

    Fetches daily weather data for a geographic point
    over a specified date range.
    """

    def __init__(self):
        super().__init__(connector_name="nasa_power")
        self.base_url = settings.nasa_power_base_url
        self.timeout_seconds = 60
        self.retry_attempts = 3
        self.retry_delay_seconds = 5

    def _build_request_params(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
        parameters: list[str],
    ) -> dict:
        """
        Build query parameters for the NASA POWER API request.
        NASA POWER date format: YYYYMMDD (no dashes).
        Community AG = Agroclimatology.
        """
        return {
            "parameters": ",".join(parameters),
            "community": "AG",
            "longitude": longitude,
            "latitude": latitude,
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON",
        }

    def _execute_request(self, params: dict) -> dict:
        """
        Make the GET request to NASA POWER with retry logic.

        Returns:
            Parsed JSON response.

        Raises:
            RuntimeError: If all retry attempts fail.
        """
        last_error = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                self.logger.info(
                    f"NASA POWER API request — attempt {attempt}/{self.retry_attempts}"
                )

                response = requests.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout_seconds,
                    headers={
                        "User-Agent": "TerraMindAI/0.1 (portfolio research project)"
                    },
                )

                response.raise_for_status()
                data = response.json()
                self.logger.info("NASA POWER API response received")
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
            f"NASA POWER API failed after {self.retry_attempts} attempts. "
            f"Last error: {last_error}"
        )

    def validate(self, data: Any) -> bool:
        """
        Validate the NASA POWER response structure.

        A valid response must contain properties.parameter
        with at least one parameter containing daily values.
        """
        if not isinstance(data, dict):
            self.logger.error("Validation failed: response is not a dict")
            return False

        properties = data.get("properties")
        if not properties:
            self.logger.error("Validation failed: 'properties' key missing")
            return False

        parameter_data = properties.get("parameter")
        if not parameter_data:
            self.logger.error("Validation failed: 'properties.parameter' key missing")
            return False

        if len(parameter_data) == 0:
            self.logger.error("Validation failed: no parameters in response")
            return False

        first_param = next(iter(parameter_data.values()))
        if not isinstance(first_param, dict) or len(first_param) == 0:
            self.logger.error("Validation failed: parameter data is empty")
            return False

        day_count = len(first_param)
        self.logger.info(
            f"Validation passed: {len(parameter_data)} parameters, "
            f"{day_count} daily records"
        )
        return True

    def _extract_summary(self, data: dict) -> dict:
        """
        Compute mean values per parameter, excluding NASA fill value -999.

        Returns:
            Dict mapping parameter name to mean value over the period.
        """
        parameter_data = data["properties"]["parameter"]
        summary = {}

        for param_name, daily_values in parameter_data.items():
            valid_values = [
                v for v in daily_values.values()
                if isinstance(v, (int, float)) and v != -999
            ]
            if valid_values:
                summary[param_name] = round(
                    sum(valid_values) / len(valid_values), 3
                )

        return summary

    def fetch(
        self,
        latitude: float = None,
        longitude: float = None,
        location_name: str = "Bengaluru",
        start_date: date = None,
        end_date: date = None,
        parameters: list[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Fetch daily weather data from NASA POWER.

        Args:
            latitude:      Decimal degrees. Defaults to Bengaluru.
            longitude:     Decimal degrees. Defaults to Bengaluru.
            location_name: Human-readable name for metadata.
            start_date:    Start of date range. Defaults to 30 days before end.
            end_date:      End of date range. Defaults to 8 days ago.
            parameters:    NASA POWER parameter codes. Defaults to AGRICULTURE_PARAMETERS.

        Returns:
            dict with keys: data, record, summary
        """
        if latitude is None:
            latitude = BENGALURU_LOCATION["latitude"]
        if longitude is None:
            longitude = BENGALURU_LOCATION["longitude"]
        if parameters is None:
            parameters = AGRICULTURE_PARAMETERS
        if end_date is None:
            end_date = date.today() - timedelta(days=8)
        if start_date is None:
            start_date = end_date - timedelta(days=29)

        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        location_slug = location_name.lower().replace(" ", "_")
        dataset_id = (
            f"nasa_power_weather_{location_slug}_"
            f"{start_date.strftime('%Y%m%d')}_"
            f"{end_date.strftime('%Y%m%d')}_"
            f"fetched{date_str}"
        )

        record = IngestionRecord(
            dataset_id=dataset_id,
            dataset_name=(
                f"NASA POWER Daily Weather — {location_name} "
                f"({start_date} to {end_date})"
            ),
            source_url=self.base_url,
            geographic_coverage=f"{location_name} ({latitude}N, {longitude}E)",
            industry="agriculture",
            data_format=DataFormat.JSON,
            licence="NASA Open Data — Public Domain",
            processing_status=IngestionStatus.FETCHING,
            validation_status=IngestionStatus.PENDING,
            notes=(
                f"Parameters: {', '.join(parameters)}. "
                f"Community: AG (Agroclimatology)."
            ),
        )

        self.logger.info(
            f"Starting NASA POWER ingestion: {location_name} "
            f"({start_date} to {end_date})"
        )

        raw_data = None
        summary = {}

        try:
            params = self._build_request_params(
                latitude=latitude,
                longitude=longitude,
                start_date=start_date,
                end_date=end_date,
                parameters=parameters,
            )

            raw_data = self._execute_request(params)

            is_valid = self.validate(raw_data)
            record.validation_status = (
                IngestionStatus.SUCCESS if is_valid else IngestionStatus.INVALID
            )

            if not is_valid:
                record.processing_status = IngestionStatus.INVALID
                record.error_message = "Validation failed — unexpected response structure"
                self.save_metadata(record)
                return {"data": None, "record": record, "summary": {}}

            summary = self._extract_summary(raw_data)
            self.logger.info(f"Weather summary: {summary}")

            first_param = next(
                iter(raw_data["properties"]["parameter"].values())
            )
            day_count = len(first_param)

            filename = f"{dataset_id}.json"
            file_path = self.save_raw(
                data=raw_data,
                filename=filename,
                subdirectory="nasa_power",
            )

            record.processing_status = IngestionStatus.SUCCESS
            record.file_path = str(file_path)
            record.record_count = day_count
            record.file_size_bytes = file_path.stat().st_size

            self.logger.info(
                f"NASA POWER ingestion complete: {day_count} days, "
                f"{record.file_size_bytes:,} bytes"
            )

        except Exception as e:
            self.logger.error(f"NASA POWER ingestion failed: {e}")
            record.processing_status = IngestionStatus.FAILED
            record.error_message = str(e)

        finally:
            self.save_metadata(record)

        return {
            "data": raw_data if record.processing_status == IngestionStatus.SUCCESS else None,
            "record": record,
            "summary": summary,
        }