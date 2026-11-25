"""
TerraMind AI — Logging Configuration
--------------------------------------
Sets up structured logging for the entire application.
Call setup_logging() once at application startup.
Then in any module: logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_dir: Path = None) -> None:
    """
    Configures the root logger for the application.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_dir:   Directory where log files will be written.
                   If None, logs go to console only.
    """

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Log format: timestamp | level | module name | message
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = []

    # Console handler — always active
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    handlers.append(console_handler)

    # File handler — active if log_dir is provided
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        log_filename = log_dir / f"terramind_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        handlers.append(file_handler)

    # Apply configuration to the root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True,   # Override any previously set handlers
    )

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger for the given module.

    Usage in any module:
        from app.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened")

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A configured Logger instance.
    """
    return logging.getLogger(name)