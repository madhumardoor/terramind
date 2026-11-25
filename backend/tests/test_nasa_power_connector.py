"""
TerraMind AI — NASA POWER Connector Tests
------------------------------------------
Automated tests for the NASA POWER ingestion connector.
No real API calls are made — responses are mocked.

Run with:
    cd backend
    pytest tests/test_nasa_power_connector.py -v
"""

import json
import pytest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.ingestion.nasa_power_connector import (
    NASAPowerConnector,
    BENGALURU_LOCATION,
    AGRICULTURE_PARAMETERS,
)
from app.models.ingestion import IngestionStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def connector():
    return NASAPowerConnector()


@pytest.fixture
def valid_nasa_response():
    """
    Minimal realistic NASA POWER response.
    Uses 5 days of data to keep the fixture small.
    -999 is NASA POWER's fill value for missing data.
    """
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [77.5946, 12.9716, 920.87]
        },
        "properties": {
            "parameter": {
                "T2M": {
                    "20241001": 24.5,
                    "20241002": 25.1,
                    "20241003": 23.8,
                    "20241004": 24.2,
                    "20241005": 25.0,
                },
                "PRECTOTCORR": {
                    "20241001": 0.0,
                    "20241002": 2.3,
                    "20241003": 0.0,
                    "20241004": 5.1,
                    "20241005": 0.0,
                },
                "RH2M": {
                    "20241001": 72.3,
                    "20241002": 75.1,
                    "20241003": 68.9,
                    "20241004": 80.2,
                    "20241005": 71.5,
                },
            }
        },
        "header": {
            "title": "NASA/POWER CERES/MERRA2 Native Resolution Daily Data"
        },
    }


@pytest.fixture
def missing_data_nasa_response():
    """Response where all values are -999 (NASA fill value for missing data)."""
    return {
        "type": "Feature",
        "properties": {
            "parameter": {
                "T2M": {
                    "20241001": -999,
                    "20241002": -999,
                }
            }
        }
    }


@pytest.fixture
def test_dates():
    end   = date(2024, 10, 5)
    start = date(2024, 10, 1)
    return start, end


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

class TestNASAPowerValidation:

    def test_valid_response_passes(self, connector, valid_nasa_response):
        assert connector.validate(valid_nasa_response) is True

    def test_missing_properties_fails(self, connector):
        assert connector.validate({"type": "Feature"}) is False

    def test_missing_parameter_key_fails(self, connector):
        assert connector.validate({"properties": {}}) is False

    def test_empty_parameter_dict_fails(self, connector):
        assert connector.validate({"properties": {"parameter": {}}}) is False

    def test_non_dict_fails(self, connector):
        assert connector.validate(None) is False
        assert connector.validate("string") is False
        assert connector.validate(42) is False


# ---------------------------------------------------------------------------
# Request parameter building tests
# ---------------------------------------------------------------------------

class TestNASAPowerRequestParams:

    def test_date_format_is_correct(self, connector, test_dates):
        """NASA POWER requires YYYYMMDD format without dashes."""
        start, end = test_dates
        params = connector._build_request_params(
            latitude=12.97,
            longitude=77.59,
            start_date=start,
            end_date=end,
            parameters=["T2M"],
        )
        assert params["start"] == "20241001"
        assert params["end"] == "20241005"

    def test_community_is_ag(self, connector, test_dates):
        """Community must be AG for agroclimatology data."""
        start, end = test_dates
        params = connector._build_request_params(
            latitude=12.97,
            longitude=77.59,
            start_date=start,
            end_date=end,
            parameters=["T2M"],
        )
        assert params["community"] == "AG"

    def test_parameters_are_joined(self, connector, test_dates):
        """Multiple parameters must be comma-joined in the request."""
        start, end = test_dates
        params = connector._build_request_params(
            latitude=12.97,
            longitude=77.59,
            start_date=start,
            end_date=end,
            parameters=["T2M", "RH2M", "PRECTOTCORR"],
        )
        assert params["parameters"] == "T2M,RH2M,PRECTOTCORR"


# ---------------------------------------------------------------------------
# Summary extraction tests
# ---------------------------------------------------------------------------

class TestNASAPowerSummary:

    def test_summary_excludes_fill_values(
        self, connector, missing_data_nasa_response
    ):
        """
        NASA POWER uses -999 for missing data.
        Summary must exclude these fill values from averages.
        """
        summary = connector._extract_summary(missing_data_nasa_response)
        # All values are -999 so no valid data — summary should be empty
        assert summary == {}

    def test_summary_computes_mean(self, connector, valid_nasa_response):
        """Mean temperature should be average of the 5 daily values."""
        summary = connector._extract_summary(valid_nasa_response)
        expected_t2m = round((24.5 + 25.1 + 23.8 + 24.2 + 25.0) / 5, 3)
        assert "T2M" in summary
        assert summary["T2M"] == expected_t2m


# ---------------------------------------------------------------------------
# Fetch tests with mocked API
# ---------------------------------------------------------------------------

class TestNASAPowerFetch:

    @patch("app.engines.ingestion.nasa_power_connector.requests.get")
    def test_successful_fetch_returns_success(
        self, mock_get, connector, valid_nasa_response,
        tmp_path, monkeypatch, test_dates
    ):
        monkeypatch.setattr(
            "app.core.config.settings.raw_data_dir", tmp_path / "raw"
        )
        monkeypatch.setattr(
            "app.core.config.settings.metadata_dir", tmp_path / "metadata"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = valid_nasa_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        start, end = test_dates
        result = connector.fetch(
            latitude=BENGALURU_LOCATION["latitude"],
            longitude=BENGALURU_LOCATION["longitude"],
            location_name="Bengaluru",
            start_date=start,
            end_date=end,
        )

        assert result["record"].processing_status == IngestionStatus.SUCCESS
        assert result["record"].record_count == 5
        assert result["data"] is not None
        assert "T2M" in result["summary"]

    @patch("app.engines.ingestion.nasa_power_connector.requests.get")
    def test_metadata_saved_on_success(
        self, mock_get, connector, valid_nasa_response,
        tmp_path, monkeypatch, test_dates
    ):
        monkeypatch.setattr(
            "app.core.config.settings.raw_data_dir", tmp_path / "raw"
        )
        monkeypatch.setattr(
            "app.core.config.settings.metadata_dir", tmp_path / "metadata"
        )

        mock_response = MagicMock()
        mock_response.json.return_value = valid_nasa_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        start, end = test_dates
        connector.fetch(
            latitude=BENGALURU_LOCATION["latitude"],
            longitude=BENGALURU_LOCATION["longitude"],
            location_name="Bengaluru",
            start_date=start,
            end_date=end,
        )

        metadata_files = list((tmp_path / "metadata").glob("*.json"))
        assert len(metadata_files) == 1

        with open(metadata_files[0], encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["processing_status"] == "success"
        assert metadata["industry"] == "agriculture"
        assert metadata["licence"] == "NASA Open Data — Public Domain"

    @patch("app.engines.ingestion.nasa_power_connector.requests.get")
    def test_api_failure_returns_failed_status(
        self, mock_get, connector, tmp_path, monkeypatch, test_dates
    ):
        monkeypatch.setattr(
            "app.core.config.settings.raw_data_dir", tmp_path / "raw"
        )
        monkeypatch.setattr(
            "app.core.config.settings.metadata_dir", tmp_path / "metadata"
        )

        mock_get.side_effect = Exception("Simulated network failure")

        start, end = test_dates
        result = connector.fetch(
            latitude=BENGALURU_LOCATION["latitude"],
            longitude=BENGALURU_LOCATION["longitude"],
            location_name="Bengaluru",
            start_date=start,
            end_date=end,
        )

        assert result["record"].processing_status == IngestionStatus.FAILED
        assert result["record"].error_message is not None