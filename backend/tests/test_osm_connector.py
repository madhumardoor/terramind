"""
TerraMind AI — OSM Connector Tests
------------------------------------
Automated tests for the OSM ingestion connector.

These tests do NOT make real API calls.
They test the connector's logic using mocked responses.

Why mock the API:
- Tests must be fast (under 1 second each)
- Tests must not depend on external services being available
- Tests must be deterministic — same result every run
- Real API calls belong in integration tests run separately

Run with:
    cd backend
    pytest tests/test_osm_connector.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.ingestion.osm_connector import OSMConnector, BENGALURU_BBOX
from app.models.ingestion import IngestionStatus


# ---------------------------------------------------------------------------
# Fixtures — reusable test data
# ---------------------------------------------------------------------------

@pytest.fixture
def connector():
    """Returns a fresh OSMConnector instance for each test."""
    return OSMConnector()


@pytest.fixture
def valid_osm_response():
    """A minimal but realistic Overpass API response."""
    return {
        "version": 0.6,
        "generator": "Overpass API",
        "elements": [
            {
                "type": "node",
                "id": 123456,
                "lat": 12.9716,
                "lon": 77.5946,
                "tags": {
                    "amenity": "hospital",
                    "name": "Test Hospital"
                }
            },
            {
                "type": "node",
                "id": 789012,
                "lat": 12.9800,
                "lon": 77.6000,
                "tags": {
                    "amenity": "school",
                    "name": "Test School"
                }
            }
        ]
    }


@pytest.fixture
def empty_osm_response():
    """An Overpass response with no elements — represents a failed query."""
    return {
        "version": 0.6,
        "generator": "Overpass API",
        "elements": []
    }


# ---------------------------------------------------------------------------
# Validation tests — no network calls needed
# ---------------------------------------------------------------------------

class TestOSMValidation:

    def test_valid_response_passes(self, connector, valid_osm_response):
        """A response with elements should pass validation."""
        assert connector.validate(valid_osm_response) is True

    def test_empty_elements_fails(self, connector, empty_osm_response):
        """A response with zero elements should fail validation."""
        assert connector.validate(empty_osm_response) is False

    def test_missing_elements_key_fails(self, connector):
        """A response without the elements key should fail validation."""
        assert connector.validate({"version": 0.6}) is False

    def test_non_dict_response_fails(self, connector):
        """A non-dict response should fail validation."""
        assert connector.validate("not a dict") is False
        assert connector.validate(None) is False
        assert connector.validate([]) is False


# ---------------------------------------------------------------------------
# Query building tests — no network calls needed
# ---------------------------------------------------------------------------

class TestOSMQueryBuilding:

    def test_roads_query_contains_highway(self, connector):
        """Roads query must filter on the highway tag."""
        query = connector._build_roads_query(BENGALURU_BBOX)
        assert "highway" in query

    def test_roads_query_contains_bbox(self, connector):
        """Roads query must include the bounding box coordinates."""
        query = connector._build_roads_query(BENGALURU_BBOX)
        assert str(BENGALURU_BBOX["south"]) in query
        assert str(BENGALURU_BBOX["north"]) in query

    def test_amenities_query_contains_amenity(self, connector):
        """Amenities query must filter on the amenity tag."""
        query = connector._build_amenities_query(BENGALURU_BBOX)
        assert "amenity" in query
        assert "hospital" in query

    def test_amenities_query_contains_bbox(self, connector):
        """Amenities query must include the bounding box coordinates."""
        query = connector._build_amenities_query(BENGALURU_BBOX)
        assert str(BENGALURU_BBOX["west"]) in query
        assert str(BENGALURU_BBOX["east"]) in query


# ---------------------------------------------------------------------------
# Fetch tests — API call is mocked
# ---------------------------------------------------------------------------

class TestOSMFetch:

    @patch("app.engines.ingestion.osm_connector.requests.post")
    def test_successful_fetch_returns_success_status(
        self, mock_post, connector, valid_osm_response, tmp_path, monkeypatch
    ):
        """
        A successful API response should produce IngestionStatus.SUCCESS.
        We mock requests.post so no real network call is made.
        """
        # Redirect data directories to a temp folder so no real files are written
        monkeypatch.setattr("app.core.config.settings.raw_data_dir", tmp_path / "raw")
        monkeypatch.setattr("app.core.config.settings.metadata_dir", tmp_path / "metadata")

        # Mock the HTTP response
        mock_response = MagicMock()
        mock_response.json.return_value = valid_osm_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = connector.fetch(
            bbox=BENGALURU_BBOX,
            feature_type="amenities",
            area_name="TestCity",
        )

        assert result["record"].processing_status == IngestionStatus.SUCCESS
        assert result["record"].record_count == 2
        assert result["data"] is not None

    @patch("app.engines.ingestion.osm_connector.requests.post")
    def test_empty_response_returns_invalid_status(
        self, mock_post, connector, empty_osm_response, tmp_path, monkeypatch
    ):
        """An empty API response should produce IngestionStatus.INVALID."""
        monkeypatch.setattr("app.core.config.settings.raw_data_dir", tmp_path / "raw")
        monkeypatch.setattr("app.core.config.settings.metadata_dir", tmp_path / "metadata")

        mock_response = MagicMock()
        mock_response.json.return_value = empty_osm_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = connector.fetch(
            bbox=BENGALURU_BBOX,
            feature_type="amenities",
            area_name="TestCity",
        )

        assert result["record"].processing_status == IngestionStatus.INVALID

    @patch("app.engines.ingestion.osm_connector.requests.post")
    def test_invalid_feature_type_raises_error(
        self, mock_post, connector, tmp_path, monkeypatch
    ):
        """An unknown feature_type should result in a FAILED status."""
        monkeypatch.setattr("app.core.config.settings.raw_data_dir", tmp_path / "raw")
        monkeypatch.setattr("app.core.config.settings.metadata_dir", tmp_path / "metadata")

        result = connector.fetch(
            bbox=BENGALURU_BBOX,
            feature_type="unknown_type",
            area_name="TestCity",
        )

        assert result["record"].processing_status == IngestionStatus.FAILED
        assert "unknown_type" in result["record"].error_message


# ---------------------------------------------------------------------------
# Metadata tests
# ---------------------------------------------------------------------------

class TestOSMMetadata:

    @patch("app.engines.ingestion.osm_connector.requests.post")
    def test_metadata_file_is_created(
        self, mock_post, connector, valid_osm_response, tmp_path, monkeypatch
    ):
        """A metadata JSON file must be created after every fetch."""
        monkeypatch.setattr("app.core.config.settings.raw_data_dir", tmp_path / "raw")
        monkeypatch.setattr("app.core.config.settings.metadata_dir", tmp_path / "metadata")

        mock_response = MagicMock()
        mock_response.json.return_value = valid_osm_response
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        result = connector.fetch(
            bbox=BENGALURU_BBOX,
            feature_type="amenities",
            area_name="TestCity",
        )

        metadata_dir = tmp_path / "metadata"
        metadata_files = list(metadata_dir.glob("*.json"))

        assert len(metadata_files) == 1

        # Verify the metadata file is valid JSON with expected fields
        with open(metadata_files[0], encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["processing_status"] == "success"
        assert metadata["licence"] == "ODbL 1.0 — © OpenStreetMap contributors"
        assert metadata["record_count"] == 2