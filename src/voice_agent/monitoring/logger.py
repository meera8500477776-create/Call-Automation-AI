"""Structured logging infrastructure using structlog."""

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from voice_agent.config.settings import settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add application context to log entries."""
    event_dict["app"] = "voice-agent-platform"
    event_dict["version"] = "0.1.0"
    return event_dict


def configure_logging() -> None:
    """Configure structured logging for the application."""
    # Determine processors based on log format
    if settings.log_format == "json":
        processors: list[Processor] = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_app_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:  # console format
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            add_app_context,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(settings.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Optional logger name (typically __name__ of the module)

    Returns:
        Configured structlog logger
    """
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()


def bind_context(**kwargs: Any) -> None:
    """Bind context variables to all subsequent log entries in this context.

    Args:
        **kwargs: Key-value pairs to add to logging context
    """
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(**kwargs)


def unbind_context(*keys: str) -> None:
    """Remove context variables from logging context.

    Args:
        *keys: Keys to remove from context
    """
    structlog.contextvars.unbind_contextvars(*keys)


def clear_context() -> None:
    """Clear all context variables."""
    structlog.contextvars.clear_contextvars()


# Initialize logging on module import
configure_logging()
