"""
TerraMind AI — OSM Processor Tests
------------------------------------
Automated tests for the OSM geospatial processor.
No file I/O, no network calls — all data is built in fixtures.

Run with:
    cd backend
    pytest tests/test_osm_processor.py -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd
from shapely.geometry import Point

from app.engines.geospatial.osm_processor import OSMProcessor, WGS84_CRS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def processor():
    return OSMProcessor()


@pytest.fixture
def sample_elements():
    """
    Realistic OSM node elements covering multiple amenity types
    and geographic spread across Bengaluru.
    """
    return [
        {
            "type": "node",
            "id": 1001,
            "lat": 12.9716,
            "lon": 77.5946,
            "tags": {"amenity": "hospital", "name": "City Hospital"}
        },
        {
            "type": "node",
            "id": 1002,
            "lat": 12.9750,
            "lon": 77.5980,
            "tags": {"amenity": "school", "name": "Central School"}
        },
        {
            "type": "node",
            "id": 1003,
            "lat": 12.9680,
            "lon": 77.5900,
            "tags": {"amenity": "restaurant", "name": "Good Food"}
        },
        {
            "type": "node",
            "id": 1004,
            "lat": 12.9800,
            "lon": 77.6100,
            "tags": {"amenity": "hospital", "name": "North Hospital"}
        },
        {
            "type": "node",
            "id": 1005,
            "lat": 13.0500,
            "lon": 77.6500,
            "tags": {"amenity": "bank", "name": "Far Away Bank"}
        },
    ]


@pytest.fixture
def sample_gdf(processor, sample_elements):
    """GeoDataFrame built from sample elements."""
    return processor.parse_osm_nodes(sample_elements)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestOSMParsing:

    def test_correct_number_of_nodes_parsed(self, processor, sample_elements):
        """All 5 node elements should be parsed."""
        gdf = processor.parse_osm_nodes(sample_elements)
        assert len(gdf) == 5

    def test_crs_is_wgs84(self, processor, sample_elements):
        """GeoDataFrame must be in EPSG:4326."""
        gdf = processor.parse_osm_nodes(sample_elements)
        assert str(gdf.crs) == WGS84_CRS

    def test_geometry_column_contains_points(self, processor, sample_elements):
        """Every geometry must be a Shapely Point."""
        gdf = processor.parse_osm_nodes(sample_elements)
        for geom in gdf.geometry:
            assert isinstance(geom, Point)

    def test_coordinates_are_correct(self, processor, sample_elements):
        """Point coordinates must match the source lat/lon values."""
        gdf = processor.parse_osm_nodes(sample_elements)
        first_row = gdf.iloc[0]
        # Shapely Point: x=longitude, y=latitude
        assert abs(first_row.geometry.x - 77.5946) < 0.0001
        assert abs(first_row.geometry.y - 12.9716) < 0.0001

    def test_required_columns_present(self, processor, sample_elements):
        """osm_id, amenity, name, lat, lon, geometry must all exist."""
        gdf = processor.parse_osm_nodes(sample_elements)
        for col in ("osm_id", "amenity", "name", "lat", "lon", "geometry"):
            assert col in gdf.columns

    def test_non_node_elements_are_skipped(self, processor):
        """Way and relation elements must be ignored."""
        elements = [
            {"type": "way", "id": 999, "nodes": [1, 2, 3], "tags": {}},
            {"type": "node", "id": 1, "lat": 12.97, "lon": 77.59,
             "tags": {"amenity": "bank", "name": "Test Bank"}},
        ]
        gdf = processor.parse_osm_nodes(elements)
        assert len(gdf) == 1

    def test_elements_without_coordinates_are_skipped(self, processor):
        """Nodes missing lat or lon must be excluded."""
        elements = [
            {"type": "node", "id": 1, "tags": {"amenity": "bank"}},
            {"type": "node", "id": 2, "lat": 12.97, "lon": 77.59,
             "tags": {"amenity": "bank", "name": "Valid Bank"}},
        ]
        gdf = processor.parse_osm_nodes(elements)
        assert len(gdf) == 1

    def test_empty_elements_raises_value_error(self, processor):
        """Empty element list must raise ValueError."""
        with pytest.raises(ValueError, match="No valid node elements"):
            processor.parse_osm_nodes([])

    def test_only_non_node_elements_raises_value_error(self, processor):
        """List with only ways must raise ValueError."""
        elements = [
            {"type": "way", "id": 1, "nodes": [], "tags": {}}
        ]
        with pytest.raises(ValueError):
            processor.parse_osm_nodes(elements)


# ---------------------------------------------------------------------------
# Filtering tests
# ---------------------------------------------------------------------------

class TestAmenityFiltering:

    def test_filter_returns_correct_amenity_type(self, processor, sample_gdf):
        """Filter for hospital must return only hospitals."""
        hospitals = processor.filter_by_amenity(sample_gdf, "hospital")
        assert len(hospitals) == 2
        assert all(hospitals["amenity"] == "hospital")

    def test_filter_returns_empty_for_unknown_type(self, processor, sample_gdf):
        """Filter for a type not in the data must return empty GeoDataFrame."""
        result = processor.filter_by_amenity(sample_gdf, "zoo")
        assert len(result) == 0

    def test_filter_does_not_modify_original(self, processor, sample_gdf):
        """Filtering must not change the original GeoDataFrame."""
        original_len = len(sample_gdf)
        processor.filter_by_amenity(sample_gdf, "hospital")
        assert len(sample_gdf) == original_len


# ---------------------------------------------------------------------------
# Proximity search tests
# ---------------------------------------------------------------------------

class TestProximitySearch:

    def test_finds_features_within_radius(self, processor, sample_gdf):
        """
        Centre at city hospital (12.9716, 77.5946).
        Nearby school (12.9750, 77.5980) is ~500m away — should be found.
        Far away bank (13.0500, 77.6500) is ~10km away — must not appear.
        """
        nearby = processor.find_within_radius(
            gdf=sample_gdf,
            centre_lat=12.9716,
            centre_lon=77.5946,
            radius_metres=2000,
        )
        names = list(nearby["name"])
        assert "City Hospital" in names
        assert "Far Away Bank" not in names

    def test_distance_column_is_present(self, processor, sample_gdf):
        """Results must include a distance_m column."""
        nearby = processor.find_within_radius(
            gdf=sample_gdf,
            centre_lat=12.9716,
            centre_lon=77.5946,
            radius_metres=2000,
        )
        assert "distance_m" in nearby.columns

    def test_results_sorted_by_distance(self, processor, sample_gdf):
        """Results must be sorted ascending by distance_m."""
        nearby = processor.find_within_radius(
            gdf=sample_gdf,
            centre_lat=12.9716,
            centre_lon=77.5946,
            radius_metres=5000,
        )
        distances = list(nearby["distance_m"])
        assert distances == sorted(distances)

    def test_zero_results_for_tiny_radius(self, processor, sample_gdf):
        """A 1-metre radius in an empty area must return no results."""
        nearby = processor.find_within_radius(
            gdf=sample_gdf,
            centre_lat=0.0,
            centre_lon=0.0,
            radius_metres=1,
        )
        assert len(nearby) == 0

    def test_result_crs_is_wgs84(self, processor, sample_gdf):
        """Results must be returned in EPSG:4326, not UTM."""
        nearby = processor.find_within_radius(
            gdf=sample_gdf,
            centre_lat=12.9716,
            centre_lon=77.5946,
            radius_metres=5000,
        )
        assert str(nearby.crs) == WGS84_CRS


# ---------------------------------------------------------------------------
# Summary tests
# ---------------------------------------------------------------------------

class TestAmenitySummary:

    def test_summary_counts_are_correct(self, processor, sample_gdf):
        """Hospital count must be 2, school 1, restaurant 1, bank 1."""
        summary = processor.get_amenity_summary(sample_gdf)
        assert summary["hospital"] == 2
        assert summary["school"] == 1
        assert summary["restaurant"] == 1
        assert summary["bank"] == 1

    def test_summary_total_matches_dataframe(self, processor, sample_gdf):
        """Sum of all counts must equal total rows in GeoDataFrame."""
        summary = processor.get_amenity_summary(sample_gdf)
        assert sum(summary.values()) == len(sample_gdf)