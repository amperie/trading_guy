import logging
import sys
from pathlib import Path
from typing import Optional
from utils.config_manager import ConfigManager


class Logger:
    """
    Singleton logger that configures itself from config.yaml.

    Usage:
        from utils.logger import Logger

        logger = Logger().get_logger(__name__)
        logger.info("Trading algorithm started")
        logger.error("Order failed", exc_info=True)
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            Logger._initialized = True

    def _setup_logging(self):
        """Configure logging from config.yaml."""
        config = ConfigManager()

        # Get logging config with defaults
        log_level = config.get("logging.level", "INFO").upper()
        log_folder = config.get("logging.folder", "logs")
        log_filename = config.get("logging.filename", "trading.log")
        log_format = config.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log_to_console = config.get("logging.console", True)

        # Create handlers
        handlers = []

        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(console_handler)

        if log_filename:
            log_path = Path(log_folder) / log_filename
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)

        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, log_level),
            handlers=handlers,
            force=True
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        return logging.getLogger(name)
