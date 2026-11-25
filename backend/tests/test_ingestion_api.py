"""
TerraMind AI — Ingestion API Tests
------------------------------------
Tests for the ingestion REST endpoints.
Uses FastAPI's TestClient — no real server needed.
Connectors are mocked — no real API calls made.

Run with:
    cd backend
    pytest tests/test_ingestion_api.py -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.models.ingestion import IngestionStatus, IngestionRecord, DataFormat
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Test client and shared mock record
# ---------------------------------------------------------------------------

client = TestClient(app)


def make_mock_record(status: IngestionStatus = IngestionStatus.SUCCESS) -> IngestionRecord:
    """Build a realistic IngestionRecord for use in mocks."""
    return IngestionRecord(
        dataset_id="test_dataset_001",
        dataset_name="Test Dataset",
        source_url="https://example.com/api",
        geographic_coverage="Bengaluru, Karnataka, India",
        industry="urban",
        data_format=DataFormat.JSON,
        licence="ODbL 1.0",
        processing_status=status,
        validation_status=IngestionStatus.SUCCESS if status == IngestionStatus.SUCCESS else IngestionStatus.INVALID,
        file_path="data/raw/test/test_dataset_001.json",
        record_count=100,
        file_size_bytes=50000,
    )


# ---------------------------------------------------------------------------
# Health endpoint sanity check
# ---------------------------------------------------------------------------

class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# OSM ingestion endpoint
# ---------------------------------------------------------------------------

class TestOSMIngestionEndpoint:

    @patch("app.api.ingestion.OSMConnector")
    def test_successful_osm_ingestion(self, mock_connector_class):
        """POST /ingestion/osm should return 200 on success."""
        mock_instance = MagicMock()
        mock_instance.fetch.return_value = {
            "data": {"elements": [{"type": "node"}] * 100},
            "record": make_mock_record(IngestionStatus.SUCCESS),
        }
        mock_connector_class.return_value = mock_instance

        response = client.post("/ingestion/osm", json={
            "feature_type": "amenities",
            "area_name": "Bengaluru",
        })

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["record_count"] == 100

    @patch("app.api.ingestion.OSMConnector")
    def test_invalid_feature_type_returns_422(self, mock_connector_class):
        """POST /ingestion/osm with invalid feature_type should return 422."""
        response = client.post("/ingestion/osm", json={
            "feature_type": "invalid_type",
            "area_name": "Bengaluru",
        })
        assert response.status_code == 422

    @patch("app.api.ingestion.OSMConnector")
    def test_failed_ingestion_returns_502(self, mock_connector_class):
        """If the connector fails, the endpoint should return 502."""
        failed_record = make_mock_record(IngestionStatus.FAILED)
        failed_record.error_message = "Overpass API unreachable"

        mock_instance = MagicMock()
        mock_instance.fetch.return_value = {
            "data": None,
            "record": failed_record,
        }
        mock_connector_class.return_value = mock_instance

        response = client.post("/ingestion/osm", json={
            "feature_type": "roads",
            "area_name": "Bengaluru",
        })

        assert response.status_code == 502


# ---------------------------------------------------------------------------
# NASA POWER ingestion endpoint
# ---------------------------------------------------------------------------

class TestNASAPowerIngestionEndpoint:

    @patch("app.api.ingestion.NASAPowerConnector")
    def test_successful_nasa_power_ingestion(self, mock_connector_class):
        """POST /ingestion/nasa-power should return 200 on success."""
        nasa_record = make_mock_record(IngestionStatus.SUCCESS)
        nasa_record.industry = "agriculture"
        nasa_record.record_count = 30

        mock_instance = MagicMock()
        mock_instance.fetch.return_value = {
            "data": {"properties": {"parameter": {}}},
            "record": nasa_record,
            "summary": {"T2M": 22.5, "PRECTOTCORR": 3.1},
        }
        mock_connector_class.return_value = mock_instance

        response = client.post("/ingestion/nasa-power", json={
            "latitude": 12.9716,
            "longitude": 77.5946,
            "location_name": "Bengaluru",
        })

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["record_count"] == 30


# ---------------------------------------------------------------------------
# Catalogue endpoint
# ---------------------------------------------------------------------------

class TestCatalogueEndpoint:

    def test_catalogue_returns_200(self):
        """GET /ingestion/catalogue should always return 200."""
        response = client.get("/ingestion/catalogue")
        assert response.status_code == 200
        body = response.json()
        assert "total" in body
        assert "datasets" in body

    def test_catalogue_with_industry_filter(self):
        """Filter by industry should return 200."""
        response = client.get("/ingestion/catalogue?industry=agriculture")
        assert response.status_code == 200

    def test_catalogue_with_status_filter(self):
        """Filter by status should return 200."""
        response = client.get("/ingestion/catalogue?status=success")
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------

class TestStatusEndpoint:

    def test_nonexistent_dataset_returns_404(self):
        """GET /ingestion/status with unknown ID should return 404."""
        response = client.get("/ingestion/status/nonexistent_dataset_xyz")
        assert response.status_code == 404

    def test_existing_dataset_returns_200(self, tmp_path, monkeypatch):
        """GET /ingestion/status with a known ID should return the record."""
        # Write a fake metadata file
        monkeypatch.setattr(
            "app.core.config.settings.metadata_dir", tmp_path
        )

        fake_record = {
            "dataset_id": "test_ds_123",
            "dataset_name": "Test",
            "processing_status": "success",
        }
        (tmp_path / "test_ds_123.json").write_text(
            json.dumps(fake_record), encoding="utf-8"
        )

        response = client.get("/ingestion/status/test_ds_123")
        assert response.status_code == 200
        assert response.json()["dataset_id"] == "test_ds_123"