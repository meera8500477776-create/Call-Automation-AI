"""Unit tests for AuditLogger."""

from datetime import datetime, timedelta

import pytest

from voice_agent.monitoring.audit_logger import AuditLogger, DateRange, LogEntry
from voice_agent.security.encryption import EncryptionService


@pytest.fixture
def audit_logger() -> AuditLogger:
    """Create an AuditLogger instance for testing."""
    return AuditLogger()


@pytest.fixture
def encryption_service() -> EncryptionService:
    """Create an EncryptionService instance for testing."""
    return EncryptionService()


def test_log_entry_immutability() -> None:
    """Test that LogEntry is immutable after creation."""
    entry = LogEntry(
        log_id="test-123",
        timestamp=datetime.utcnow(),
        event_type="test_event",
        session_id="session-123",
        data={"key": "value"},
    )

    # Attempt to modify should raise AttributeError
    with pytest.raises(AttributeError, match="immutable"):
        entry.log_id = "modified"

    with pytest.raises(AttributeError, match="immutable"):
        entry.event_type = "modified_event"

    with pytest.raises(AttributeError, match="immutable"):
        entry.data = {"new": "data"}


def test_log_event_basic(audit_logger: AuditLogger) -> None:
    """Test basic event logging."""
    session_id = "session-123"
    event_type = AuditLogger.EVENT_CALL_INITIATED
    data = {"caller_id": "+1234567890", "tenant_id": "healthcare-001"}

    log_entry = audit_logger.log_event(event_type, session_id, data)

    assert log_entry.log_id is not None
    assert log_entry.event_type == event_type
    assert log_entry.session_id == session_id
    assert log_entry.data == data
    assert log_entry.timestamp is not None
    assert isinstance(log_entry.timestamp, datetime)


def test_log_event_with_pii_tokenization(audit_logger: AuditLogger) -> None:
    """Test event logging with PII tokenization."""
    session_id = "session-123"
    event_type = AuditLogger.EVENT_TRANSCRIPT_SEGMENT
    data = {
        "transcript": "My email is john.doe@example.com and phone is 555-123-4567",
        "speaker": "user",
    }

    log_entry = audit_logger.log_event(event_type, session_id, data, tokenize_pii=True)

    assert log_entry.tokenized_data is not None
    # Check that PII was tokenized
    assert "john.doe@example.com" not in log_entry.tokenized_data.text
    assert "555-123-4567" not in log_entry.tokenized_data.text
    # Check that tokens were created
    assert len(log_entry.tokenized_data.tokens) > 0


def test_log_event_without_pii_tokenization(audit_logger: AuditLogger) -> None:
    """Test event logging without PII tokenization."""
    session_id = "session-123"
    event_type = AuditLogger.EVENT_ERROR
    data = {"error": "Connection timeout", "component": "STT"}

    log_entry = audit_logger.log_event(
        event_type, session_id, data, tokenize_pii=False
    )

    assert log_entry.tokenized_data is None


def test_log_api_call(audit_logger: AuditLogger) -> None:
    """Test API call logging."""
    tool_name = "search_appointments"
    session_id = "session-123"
    parameters = {"patient_id": "P123", "date": "2024-01-15"}
    result = {"slots": [{"id": "slot-1", "time": "10:00"}]}
    execution_time_ms = 250.5

    log_entry = audit_logger.log_api_call(
        tool_name, session_id, parameters, result, execution_time_ms
    )

    assert log_entry.event_type == AuditLogger.EVENT_TOOL_EXECUTION
    assert log_entry.session_id == session_id
    assert log_entry.data["tool_name"] == tool_name
    assert log_entry.data["parameters"] == parameters
    assert log_entry.data["result"] == result
    assert log_entry.data["execution_time_ms"] == execution_time_ms


def test_log_pii_access(audit_logger: AuditLogger) -> None:
    """Test PII access logging."""
    user_id = "system-component-llm"
    session_id = "session-123"
    data_type = "patient_record"
    reason = "appointment_scheduling"
    accessed_fields = ["name", "date_of_birth", "phone"]

    log_entry = audit_logger.log_pii_access(
        user_id, session_id, data_type, reason, accessed_fields
    )

    assert log_entry.event_type == AuditLogger.EVENT_PII_ACCESS
    assert log_entry.session_id == session_id
    assert log_entry.data["user_id"] == user_id
    assert log_entry.data["data_type"] == data_type
    assert log_entry.data["reason"] == reason
    assert log_entry.data["accessed_fields"] == accessed_fields


def test_query_logs_by_event_type(audit_logger: AuditLogger) -> None:
    """Test querying logs by event type."""
    # Create multiple log entries
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "session-1", {"data": "1"}
    )
    audit_logger.log_event(AuditLogger.EVENT_ERROR, "session-2", {"data": "2"})
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "session-3", {"data": "3"}
    )

    # Query for specific event type
    results = audit_logger.query_logs(
        filters={"event_type": AuditLogger.EVENT_CALL_INITIATED}
    )

    assert len(results) == 2
    assert all(log.event_type == AuditLogger.EVENT_CALL_INITIATED for log in results)


def test_query_logs_by_session_id(audit_logger: AuditLogger) -> None:
    """Test querying logs by session ID."""
    session_id = "session-123"

    # Create multiple log entries
    audit_logger.log_event(AuditLogger.EVENT_CALL_INITIATED, session_id, {"data": "1"})
    audit_logger.log_event(AuditLogger.EVENT_LLM_REQUEST, session_id, {"data": "2"})
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "other-session", {"data": "3"}
    )

    # Query for specific session
    results = audit_logger.query_logs(filters={"session_id": session_id})

    assert len(results) == 2
    assert all(log.session_id == session_id for log in results)


def test_query_logs_by_date_range(audit_logger: AuditLogger) -> None:
    """Test querying logs by date range."""
    now = datetime.utcnow()

    # Create log entries
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "session-1", {"data": "1"}
    )

    # Query with date range
    date_range = DateRange(start=now - timedelta(minutes=1), end=now + timedelta(minutes=1))
    results = audit_logger.query_logs(date_range=date_range)

    assert len(results) == 1


def test_query_logs_with_limit(audit_logger: AuditLogger) -> None:
    """Test querying logs with limit."""
    # Create multiple log entries
    for i in range(10):
        audit_logger.log_event(
            AuditLogger.EVENT_CALL_INITIATED, f"session-{i}", {"data": str(i)}
        )

    # Query with limit
    results = audit_logger.query_logs(limit=5)

    assert len(results) == 5


def test_query_logs_by_data_field(audit_logger: AuditLogger) -> None:
    """Test querying logs by data field."""
    # Create log entries with different data
    audit_logger.log_event(
        AuditLogger.EVENT_ERROR, "session-1", {"component": "STT"}
    )
    audit_logger.log_event(
        AuditLogger.EVENT_ERROR, "session-2", {"component": "LLM"}
    )
    audit_logger.log_event(
        AuditLogger.EVENT_ERROR, "session-3", {"component": "STT"}
    )

    # Query by data field
    results = audit_logger.query_logs(filters={"component": "STT"})

    assert len(results) == 2
    assert all(log.data["component"] == "STT" for log in results)


def test_query_logs_combined_filters(audit_logger: AuditLogger) -> None:
    """Test querying logs with combined filters."""
    session_id = "session-123"

    # Create log entries
    audit_logger.log_event(
        AuditLogger.EVENT_ERROR, session_id, {"component": "STT"}
    )
    audit_logger.log_event(
        AuditLogger.EVENT_ERROR, session_id, {"component": "LLM"}
    )
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, session_id, {"component": "STT"}
    )

    # Query with combined filters
    results = audit_logger.query_logs(
        filters={"event_type": AuditLogger.EVENT_ERROR, "session_id": session_id}
    )

    assert len(results) == 2
    assert all(log.event_type == AuditLogger.EVENT_ERROR for log in results)
    assert all(log.session_id == session_id for log in results)


def test_get_log_count(audit_logger: AuditLogger) -> None:
    """Test getting log count."""
    assert audit_logger.get_log_count() == 0

    # Add some logs
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "session-1", {"data": "1"}
    )
    audit_logger.log_event(AuditLogger.EVENT_ERROR, "session-2", {"data": "2"})

    assert audit_logger.get_log_count() == 2


def test_clear_logs(audit_logger: AuditLogger) -> None:
    """Test clearing logs."""
    # Add some logs
    audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "session-1", {"data": "1"}
    )
    audit_logger.log_event(AuditLogger.EVENT_ERROR, "session-2", {"data": "2"})

    assert audit_logger.get_log_count() == 2

    # Clear logs
    audit_logger.clear_logs()

    assert audit_logger.get_log_count() == 0


def test_log_entry_data_isolation(audit_logger: AuditLogger) -> None:
    """Test that modifying original data doesn't affect logged data."""
    data = {"key": "original_value"}
    log_entry = audit_logger.log_event(
        AuditLogger.EVENT_CALL_INITIATED, "session-1", data
    )

    # Modify original data
    data["key"] = "modified_value"

    # Log entry should still have original value
    assert log_entry.data["key"] == "original_value"


def test_multiple_sessions_logging(audit_logger: AuditLogger) -> None:
    """Test logging events from multiple sessions."""
    sessions = ["session-1", "session-2", "session-3"]

    for session_id in sessions:
        audit_logger.log_event(
            AuditLogger.EVENT_CALL_INITIATED, session_id, {"data": session_id}
        )
        audit_logger.log_event(
            AuditLogger.EVENT_CALL_TERMINATED, session_id, {"data": session_id}
        )

    # Verify all logs were created
    assert audit_logger.get_log_count() == 6

    # Verify each session has 2 logs
    for session_id in sessions:
        results = audit_logger.query_logs(filters={"session_id": session_id})
        assert len(results) == 2


def test_pii_tokenization_consistency(audit_logger: AuditLogger) -> None:
    """Test that PII tokenization is consistent across multiple logs."""
    email = "test@example.com"
    data1 = {"message": f"Contact me at {email}"}
    data2 = {"message": f"My email is {email}"}

    log1 = audit_logger.log_event(
        AuditLogger.EVENT_TRANSCRIPT_SEGMENT, "session-1", data1, tokenize_pii=True
    )
    log2 = audit_logger.log_event(
        AuditLogger.EVENT_TRANSCRIPT_SEGMENT, "session-2", data2, tokenize_pii=True
    )

    # Both logs should have tokenized data
    assert log1.tokenized_data is not None
    assert log2.tokenized_data is not None

    # Email should not appear in tokenized text
    assert email not in log1.tokenized_data.text
    assert email not in log2.tokenized_data.text
