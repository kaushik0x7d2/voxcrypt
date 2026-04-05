"""
Structured logging for Orion Voice.

Supports JSON format (for production log aggregation) and text format (for development).
"""

import json
import logging
import threading
import uuid


_request_id_var = threading.local()


def get_request_id():
    """Get the current request's correlation ID."""
    return getattr(_request_id_var, "request_id", None)


def set_request_id(request_id=None):
    """Set the current request's correlation ID."""
    _request_id_var.request_id = request_id or str(uuid.uuid4())[:8]
    return _request_id_var.request_id


class JSONFormatter(logging.Formatter):
    """JSON log formatter for production use."""

    SENSITIVE_KEYS = {"key", "secret", "password", "token", "ciphertext"}

    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        request_id = get_request_id()
        if request_id:
            log_data["request_id"] = request_id

        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra_data"):
            for k, v in record.extra_data.items():
                if any(s in k.lower() for s in self.SENSITIVE_KEYS):
                    log_data[k] = "[REDACTED]"
                else:
                    log_data[k] = v

        return json.dumps(log_data)


class TextFormatter(logging.Formatter):
    """Human-readable log formatter for development."""

    def format(self, record):
        request_id = get_request_id()
        rid = f"[{request_id}] " if request_id else ""
        return (
            f"{self.formatTime(record)} {record.levelname:>8s} {rid}"
            f"{record.name}: {record.getMessage()}"
        )


def setup_logging(level="INFO", fmt="json", log_file=""):
    """
    Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Format type ("json" or "text").
        log_file: Optional file path for log output.
    """
    root = logging.getLogger("orion_voice")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root.handlers.clear()

    formatter = JSONFormatter() if fmt == "json" else TextFormatter()

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Suppress noisy loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    return root


def get_logger(name):
    """Get a logger under the orion_voice namespace."""
    return logging.getLogger(f"orion_voice.{name}")
