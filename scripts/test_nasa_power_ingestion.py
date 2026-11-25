"""
TerraMind AI — NASA POWER Ingestion Test Script
------------------------------------------------
Runs the NASA POWER connector and prints the result.
Makes a real API call — requires internet connection.

Run from the terramind root with .venv active:
    cd backend
    python ..\scripts\test_nasa_power_ingestion.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from app.core.logging_config import setup_logging
from app.core.config import settings
from app.engines.ingestion.nasa_power_connector import (
    NASAPowerConnector,
    BENGALURU_LOCATION,
    AGRICULTURE_PARAMETERS,
)

setup_logging(log_level="INFO")


def main():
    print("\n" + "=" * 60)
    print("TerraMind AI — NASA POWER Ingestion Test")
    print("=" * 60)

    connector = NASAPowerConnector()

    # Fetch last 30 days of weather data
    end_date   = date.today() - timedelta(days=8)
    start_date = end_date - timedelta(days=29)

    print(f"\n[1] Fetching weather data for {BENGALURU_LOCATION['name']}")
    print(f"    Date range : {start_date} to {end_date}")
    print(f"    Parameters : {', '.join(AGRICULTURE_PARAMETERS)}")
    print("    (This makes a real API call — may take 10-20 seconds)\n")

    result = connector.fetch(
        latitude=BENGALURU_LOCATION["latitude"],
        longitude=BENGALURU_LOCATION["longitude"],
        location_name=BENGALURU_LOCATION["name"],
        start_date=start_date,
        end_date=end_date,
    )

    record  = result["record"]
    summary = result["summary"]

    print("\n" + "-" * 60)
    print("INGESTION RESULT")
    print("-" * 60)
    print(f"Dataset ID     : {record.dataset_id}")
    print(f"Status         : {record.processing_status}")
    print(f"Validation     : {record.validation_status}")
    print(f"Days fetched   : {record.record_count}")
    print(f"File size      : {record.file_size_bytes:,} bytes" if record.file_size_bytes else "File size: N/A")
    print(f"Saved to       : {record.file_path}")
    print(f"Licence        : {record.licence}")

    if record.error_message:
        print(f"Error          : {record.error_message}")

    if summary:
        print("\nWeather summary (30-day averages):")
        labels = {
            "T2M":               "Mean temperature     ",
            "T2M_MAX":           "Max temperature      ",
            "T2M_MIN":           "Min temperature      ",
            "PRECTOTCORR":       "Precipitation        ",
            "RH2M":              "Relative humidity    ",
            "WS2M":              "Wind speed           ",
            "ALLSKY_SFC_SW_DWN": "Solar irradiance     ",
        }
        units = {
            "T2M":               "°C",
            "T2M_MAX":           "°C",
            "T2M_MIN":           "°C",
            "PRECTOTCORR":       "mm/day",
            "RH2M":              "%",
            "WS2M":              "m/s",
            "ALLSKY_SFC_SW_DWN": "kW-hr/m²/day",
        }
        for param, value in summary.items():
            label = labels.get(param, param)
            unit  = units.get(param, "")
            print(f"  {label}: {value} {unit}")

    print("\n" + "=" * 60)
    print("Output files:")
    print(f"  Raw data : {settings.raw_data_dir / 'nasa_power'}")
    print(f"  Metadata : {settings.metadata_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()