import glob
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from utils.config_manager import ConfigManager


class Logger:
    """
    Singleton logger that configures itself from config.yaml.

    Supports console output and optional daily-rotating file output.
    Old log files are automatically deleted based on ``retention_days``.

    Config (config.yaml):
        logging:
          level: INFO
          console: true
          file_logging: false        # set true to enable file output
          folder: "logs"
          filename: "trading.log"    # base name; date suffix added automatically
          retention_days: 7          # how many days of log files to keep
          format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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

        log_level = config.get("logging.level", "INFO").upper()
        log_to_console = config.get("logging.console", True)
        log_to_file = config.get("logging.file_logging", False)
        log_folder = config.get("logging.folder", "logs")
        log_filename = config.get("logging.filename", "trading.log")
        self._retention_days = config.get("logging.retention_days", 7)
        log_format = config.get(
            "logging.format",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        handlers = []

        if log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(console_handler)

        if log_to_file and log_filename:
            log_path = Path(log_folder) / log_filename
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = TimedRotatingFileHandler(
                filename=str(log_path),
                when="midnight",
                interval=1,
                backupCount=self._retention_days,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)

            # Clean up old log files beyond retention
            self._cleanup_old_logs(log_path)

        logging.basicConfig(
            level=getattr(logging, log_level),
            handlers=handlers,
            force=True
        )

    def _cleanup_old_logs(self, log_path: Path):
        """Delete rotated log files older than retention_days."""
        import time

        cutoff = time.time() - (self._retention_days * 86400)
        parent = log_path.parent
        stem = log_path.name

        for path in parent.glob(f"{stem}.*"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                pass

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        return logging.getLogger(name)
