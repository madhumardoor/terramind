"""
TerraMind AI — Application Configuration
-----------------------------------------
Loads all environment variables from backend/.env using Pydantic Settings.
Every other module imports from here. Never read os.environ directly elsewhere.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


# Resolve the backend directory so relative paths in .env work correctly
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Central configuration object.
    Values are loaded from the .env file automatically.
    Type hints ensure values are validated on startup.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",        # Ignore unknown keys in .env safely
    )

    # Application identity
    app_name: str = Field(default="TerraMind AI")
    app_version: str = Field(default="0.1.0")
    app_env: str = Field(default="development")
    debug: bool = Field(default=True)

    # Server
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # Data directories (resolved as absolute paths)
    raw_data_dir: Path = Field(default=BACKEND_DIR.parent / "data" / "raw")
    processed_data_dir: Path = Field(default=BACKEND_DIR.parent / "data" / "processed")
    metadata_dir: Path = Field(default=BACKEND_DIR.parent / "data" / "metadata")

    # External API base URLs
    nasa_power_base_url: str = Field(
        default="https://power.larc.nasa.gov/api/temporal/daily/point"
    )
    osm_overpass_url: str = Field(
        default="https://overpass-api.de/api/interpreter"
    )

    # Logging
    log_level: str = Field(default="INFO")
    log_dir: Path = Field(default=BACKEND_DIR.parent / "logs")


def get_settings() -> Settings:
    """
    Returns a Settings instance.
    Call this function wherever you need configuration.
    In FastAPI, use it as a dependency with Depends(get_settings).
    """
    return Settings()


# Module-level singleton — imported directly by other modules
settings = get_settings()