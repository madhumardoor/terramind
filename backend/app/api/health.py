"""
TerraMind AI — Health Check Endpoints
---------------------------------------
Provides /health and /health/detailed endpoints.
These are called by monitoring systems to verify the API is running.
They also confirm configuration is loading correctly.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


class HealthResponse(BaseModel):
    """Basic health check response schema."""
    status: str
    app_name: str
    version: str
    environment: str
    timestamp: str


class DetailedHealthResponse(HealthResponse):
    """Detailed health check — includes configuration summary."""
    debug_mode: bool
    data_dirs: dict[str, str]
    external_apis: dict[str, str]


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check.
    Returns HTTP 200 when the API is running.
    Called by load balancers and monitoring tools.
    """
    logger.info("Health check called")
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check() -> DetailedHealthResponse:
    """
    Detailed health check.
    Returns configuration state — useful during development and deployment.
    In production, restrict this endpoint to internal networks only.
    """
    logger.info("Detailed health check called")
    return DetailedHealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        timestamp=datetime.now(timezone.utc).isoformat(),
        debug_mode=settings.debug,
        data_dirs={
            "raw": str(settings.raw_data_dir),
            "processed": str(settings.processed_data_dir),
            "metadata": str(settings.metadata_dir),
        },
        external_apis={
            "nasa_power": settings.nasa_power_base_url,
            "osm_overpass": settings.osm_overpass_url,
        },
    )