"""
TerraMind AI — Application Entry Point
----------------------------------------
Creates and configures the FastAPI application.
Registers all API routers.
Handles startup and shutdown events.

Run with:
    cd backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging, get_logger
from app.api.health import router as health_router

# Initialise logging before anything else runs
setup_logging(
    log_level=settings.log_level,
    log_dir=settings.log_dir,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.

    The lifespan context manager replaces the older @app.on_event approach.
    Code before `yield` runs at startup.
    Code after `yield` runs at shutdown.

    This is where we will later:
    - Connect to PostgreSQL
    - Connect to Redis
    - Initialise vector database connections
    - Start background schedulers
    """
    # --- Startup ---
    logger.info("=" * 60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Debug mode  : {settings.debug}")
    logger.info(f"Log level   : {settings.log_level}")
    logger.info("=" * 60)

    # Ensure data directories exist on startup
    for dir_path in [
        settings.raw_data_dir,
        settings.processed_data_dir,
        settings.metadata_dir,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory verified: {dir_path}")

    logger.info("TerraMind AI is ready")

    yield  # Application runs here

    # --- Shutdown ---
    logger.info("TerraMind AI is shutting down")


# Create the FastAPI application instance
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "An autonomous, multimodal, multi-industry geospatial intelligence platform. "
        "Supports Agriculture, Manufacturing, Urban Infrastructure, Environment, "
        "Logistics, Healthcare, and Real Estate intelligence modules."
    ),
    docs_url="/docs",        # Swagger UI at http://localhost:8000/docs
    redoc_url="/redoc",      # ReDoc UI at http://localhost:8000/redoc
    lifespan=lifespan,
)

# CORS middleware — allows the React frontend to call this API
# In production, replace ["*"] with your actual frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
# Each router handles a group of related endpoints
app.include_router(health_router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> dict:
    """
    Root endpoint. Confirms the API is reachable.
    """
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }