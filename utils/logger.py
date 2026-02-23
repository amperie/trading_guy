import glob
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from utils.config_manager import ConfigManager


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes based on log level."""

    COLORS = {
        logging.DEBUG: "\033[36m",      # Cyan
        logging.INFO: "\033[32m",       # Green
        logging.WARNING: "\033[33m",    # Yellow
        logging.ERROR: "\033[31m",      # Red
        logging.CRITICAL: "\033[1;31m", # Bold Red
    }
    RESET = "\033[0m"

    COLOR_NAMES = {
        "red": "\033[31m",
        "bold_red": "\033[1;31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
    }

    def format(self, record):
        msg = super().format(record)
        override = getattr(record, "color", None)
        if override:
            color = self.COLOR_NAMES.get(override, "")
        else:
            color = self.COLORS.get(record.levelno, "")
        return f"{color}{msg}{self.RESET}" if color else msg


class ColorLogger:
    """Wrapper around logging.Logger that supports a ``color`` keyword argument."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def __getattr__(self, name):
        return getattr(self._logger, name)

    def _log_with_color(self, method_name, msg, *args, **kwargs):
        color = kwargs.pop("color", None)
        extra = kwargs.pop("extra", None) or {}
        if color:
            extra["color"] = color
        getattr(self._logger, method_name)(msg, *args, extra=extra, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._log_with_color("debug", msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self._log_with_color("info", msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._log_with_color("warning", msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._log_with_color("error", msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._log_with_color("critical", msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):
        color = kwargs.pop("color", None)
        extra = kwargs.pop("extra", None) or {}
        if color:
            extra["color"] = color
        self._logger.log(level, msg, *args, extra=extra, **kwargs)


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
            console_handler.setFormatter(ColorFormatter(log_format))
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

    def get_logger(self, name: str) -> ColorLogger:
        """Get a logger instance for a specific module."""
        return ColorLogger(logging.getLogger(name))
