"""Structured application logging without sensitive values."""

from __future__ import annotations

import logging
import sys

from app.core.config import get_settings

_SENSITIVE_KEYS = ("api_key", "apikey", "authorization", "password", "secret", "token")


class RedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = str(record.getMessage())
        lowered = message.lower()
        if any(key in lowered for key in _SENSITIVE_KEYS):
            record.msg = "[redacted log message containing sensitive key]"
            record.args = ()
        return True


def configure_logging() -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger("reqguard")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


logger = configure_logging()
