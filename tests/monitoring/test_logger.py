"""Tests for structured logging."""


from voice_agent.monitoring.logger import (
    bind_context,
    clear_context,
    get_logger,
    unbind_context,
)


def test_get_logger() -> None:
    """Test getting a logger instance."""
    logger = get_logger("test_module")

    assert logger is not None
    # Check that logger has the expected logging methods
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_get_logger_without_name() -> None:
    """Test getting a logger without a name."""
    logger = get_logger()

    assert logger is not None
    # Check that logger has the expected logging methods
    assert hasattr(logger, "info")
    assert hasattr(logger, "debug")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


def test_bind_context() -> None:
    """Test binding context variables."""
    clear_context()  # Start fresh

    bind_context(session_id="test-session", tenant_id="test-tenant")

    # Context should be bound for subsequent log calls
    # We can't easily test the output, but we can verify no errors occur
    logger = get_logger("test")
    logger.info("test message")


def test_unbind_context() -> None:
    """Test unbinding specific context variables."""
    clear_context()
    bind_context(session_id="test-session", tenant_id="test-tenant")

    unbind_context("session_id")

    # Should not raise any errors
    logger = get_logger("test")
    logger.info("test message")


def test_clear_context() -> None:
    """Test clearing all context variables."""
    bind_context(session_id="test-session", tenant_id="test-tenant")

    clear_context()

    # Should not raise any errors
    logger = get_logger("test")
    logger.info("test message")
