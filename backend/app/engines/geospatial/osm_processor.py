"""
TerraMind AI — OSM Geospatial Processor
-----------------------------------------
Loads raw OSM Overpass API data and converts it into
a GeoPandas GeoDataFrame for spatial analysis.

CRS strategy
------------
EPSG:4326  — WGS84 geographic (latitude/longitude). Used for storage and display.
EPSG:32643 — UTM Zone 43N (metres). Used for distance calculations in Karnataka.
"""

import json
from pathlib import Path
from typing import Optional

import geopandas as gpd
from shapely.geometry import Point

from app.core.logging_config import get_logger

logger = get_logger(__name__)

KARNATAKA_UTM_CRS = "EPSG:32643"
WGS84_CRS = "EPSG:4326"


class OSMProcessor:
    """
    Processes raw OSM Overpass API data into spatial datasets.

    Usage
    -----
        processor = OSMProcessor()
        gdf = processor.load_from_file(path_to_raw_json)
        hospitals = processor.filter_by_amenity(gdf, "hospital")
        nearby = processor.find_within_radius(
            gdf, centre_lat=12.9716, centre_lon=77.5946, radius_metres=5000
        )
    """

    def parse_osm_nodes(self, elements: list[dict]) -> gpd.GeoDataFrame:
        """
        Convert a list of OSM node elements into a GeoDataFrame.

        Only processes elements of type 'node' that have lat and lon.
        Ways and relations are skipped.

        Args:
            elements: List of OSM element dicts from Overpass response.

        Returns:
            GeoDataFrame with columns: osm_id, amenity, name, lat, lon, geometry.
            CRS is EPSG:4326.

        Raises:
            ValueError: If no valid node elements are found.
        """
        rows = []

        for element in elements:
            if element.get("type") != "node":
                continue

            lat = element.get("lat")
            lon = element.get("lon")

            if lat is None or lon is None:
                continue

            tags = element.get("tags", {})

            row = {
                "osm_id":   element.get("id"),
                "amenity":  tags.get("amenity", "unknown"),
                "name":     tags.get("name", "unnamed"),
                "lat":      lat,
                "lon":      lon,
                "geometry": Point(lon, lat),
            }

            for tag_key in ("operator", "opening_hours", "addr:street",
                            "addr:city", "phone", "website"):
                if tag_key in tags:
                    row[tag_key] = tags[tag_key]

            rows.append(row)

        if not rows:
            raise ValueError(
                "No valid node elements found in OSM data. "
                "Check that the Overpass query returned node features."
            )

        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=WGS84_CRS)

        logger.info(
            f"Parsed {len(gdf)} OSM nodes into GeoDataFrame "
            f"with {len(gdf.columns)} columns"
        )
        return gdf

    def load_from_file(self, file_path: Path) -> gpd.GeoDataFrame:
        """
        Load a raw OSM JSON file and return a GeoDataFrame.

        Args:
            file_path: Path to the raw OSM JSON file saved by OSMConnector.

        Returns:
            GeoDataFrame of OSM nodes with EPSG:4326 CRS.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains no valid nodes.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"OSM data file not found: {file_path}. "
                f"Run the OSM ingestion connector first."
            )

        logger.info(f"Loading OSM data from: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            raw_data = json.load(f)

        elements = raw_data.get("elements", [])
        logger.info(f"Found {len(elements)} total elements in raw file")

        return self.parse_osm_nodes(elements)

    def filter_by_amenity(
        self,
        gdf: gpd.GeoDataFrame,
        amenity_type: str,
    ) -> gpd.GeoDataFrame:
        """
        Filter the GeoDataFrame to a specific amenity type.

        Args:
            gdf:          Full GeoDataFrame of OSM nodes.
            amenity_type: Amenity value to filter on, e.g. 'hospital', 'school'.

        Returns:
            Filtered GeoDataFrame. May be empty if no matches.
        """
        filtered = gdf[gdf["amenity"] == amenity_type].copy()
        logger.info(
            f"Filtered to amenity='{amenity_type}': {len(filtered)} features"
        )
        return filtered

    def find_within_radius(
        self,
        gdf: gpd.GeoDataFrame,
        centre_lat: float,
        centre_lon: float,
        radius_metres: float,
    ) -> gpd.GeoDataFrame:
        """
        Find all features within a given radius of a centre point.

        Reprojects to UTM for accurate metre-based distance calculations,
        then reprojects results back to EPSG:4326 for return.

        Args:
            gdf:            GeoDataFrame in EPSG:4326.
            centre_lat:     Latitude of the centre point.
            centre_lon:     Longitude of the centre point.
            radius_metres:  Search radius in metres.

        Returns:
            GeoDataFrame of features within the radius, with distance_m column.
            Sorted by distance ascending. Returned in EPSG:4326.
        """
        centre_point_wgs84 = gpd.GeoDataFrame(
            geometry=[Point(centre_lon, centre_lat)],
            crs=WGS84_CRS,
        )

        gdf_utm = gdf.to_crs(KARNATAKA_UTM_CRS)
        centre_utm = centre_point_wgs84.to_crs(KARNATAKA_UTM_CRS)
        centre_geom_utm = centre_utm.geometry.iloc[0]

        buffer = centre_geom_utm.buffer(radius_metres)
        within_mask = gdf_utm.geometry.within(buffer)
        result_utm = gdf_utm[within_mask].copy()

        result_utm["distance_m"] = result_utm.geometry.apply(
            lambda geom: round(geom.distance(centre_geom_utm), 1)
        )

        result_wgs84 = result_utm.to_crs(WGS84_CRS)

        logger.info(
            f"Found {len(result_wgs84)} features within "
            f"{radius_metres:,.0f}m of ({centre_lat}, {centre_lon})"
        )

        return result_wgs84.sort_values("distance_m")

    def get_amenity_summary(self, gdf: gpd.GeoDataFrame) -> dict:
        """
        Return a count of features by amenity type, sorted descending.

        Args:
            gdf: GeoDataFrame of OSM nodes.

        Returns:
            Dict mapping amenity type to count.
        """
        counts = gdf["amenity"].value_counts().to_dict()
        logger.info(f"Amenity summary: {len(counts)} unique types")
        return counts

    def to_geojson_dict(self, gdf: gpd.GeoDataFrame) -> dict:
        """
        Convert a GeoDataFrame to a GeoJSON-compatible dict.

        Args:
            gdf: GeoDataFrame to convert.

        Returns:
            GeoJSON FeatureCollection dict.
        """
        import json as _json
        export_gdf = gdf.copy()

        if "distance_m" in export_gdf.columns:
            export_gdf["distance_m"] = export_gdf["distance_m"].round(1)

        return _json.loads(export_gdf.to_json())