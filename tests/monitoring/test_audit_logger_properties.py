"""Property-based tests for AuditLogger using Hypothesis.

**Validates: Requirements 7.7**
"""

from datetime import datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from voice_agent.monitoring.audit_logger import AuditLogger


class TestAuditLoggerProperties:
    """Property-based tests for AuditLogger."""

    @given(
        event_type=st.sampled_from([
            AuditLogger.EVENT_CALL_INITIATED,
            AuditLogger.EVENT_CALL_TERMINATED,
            AuditLogger.EVENT_TRANSCRIPT_SEGMENT,
            AuditLogger.EVENT_LLM_REQUEST,
            AuditLogger.EVENT_LLM_RESPONSE,
            AuditLogger.EVENT_TOOL_EXECUTION,
            AuditLogger.EVENT_PII_ACCESS,
            AuditLogger.EVENT_AUTHENTICATION,
            AuditLogger.EVENT_ERROR,
            AuditLogger.EVENT_LATENCY_THRESHOLD,
        ]),
        session_id=st.text(min_size=1, max_size=100),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.one_of(
                st.text(max_size=100),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans(),
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_log_immutability_attributes(
        self, event_type: str, session_id: str, data: dict
    ):
        """Property 11: Log immutability - For any logged event, the log entry should never be modified after creation.

        **Validates: Requirements 7.7**

        This property ensures that once a log entry is created, all of its
        attributes (log_id, timestamp, event_type, session_id, data) are
        immutable and cannot be changed.
        """
        audit_logger = AuditLogger()

        # Create log entry
        log_entry = audit_logger.log_event(event_type, session_id, data)

        # Store original values
        original_log_id = log_entry.log_id
        original_timestamp = log_entry.timestamp
        original_event_type = log_entry.event_type
        original_session_id = log_entry.session_id
        original_data = log_entry.data.copy()

        # Property: Attempting to modify any attribute should raise AttributeError
        with pytest.raises(AttributeError, match="immutable"):
            log_entry.log_id = "modified_id"

        with pytest.raises(AttributeError, match="immutable"):
            log_entry.timestamp = datetime.utcnow()

        with pytest.raises(AttributeError, match="immutable"):
            log_entry.event_type = "modified_event"

        with pytest.raises(AttributeError, match="immutable"):
            log_entry.session_id = "modified_session"

        with pytest.raises(AttributeError, match="immutable"):
            log_entry.data = {"modified": "data"}

        # Property: All original values should remain unchanged
        assert log_entry.log_id == original_log_id
        assert log_entry.timestamp == original_timestamp
        assert log_entry.event_type == original_event_type
        assert log_entry.session_id == original_session_id
        assert log_entry.data == original_data

    @given(
        event_type=st.sampled_from([
            AuditLogger.EVENT_CALL_INITIATED,
            AuditLogger.EVENT_CALL_TERMINATED,
            AuditLogger.EVENT_TRANSCRIPT_SEGMENT,
            AuditLogger.EVENT_LLM_REQUEST,
            AuditLogger.EVENT_LLM_RESPONSE,
            AuditLogger.EVENT_TOOL_EXECUTION,
            AuditLogger.EVENT_PII_ACCESS,
            AuditLogger.EVENT_AUTHENTICATION,
            AuditLogger.EVENT_ERROR,
            AuditLogger.EVENT_LATENCY_THRESHOLD,
        ]),
        session_id=st.text(min_size=1, max_size=100),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_log_immutability_data_isolation(
        self, event_type: str, session_id: str, data: dict
    ):
        """Property 11 (variant): Log immutability - Modifying the original data dict should not affect the logged data.

        **Validates: Requirements 7.7**

        This property ensures that the log entry stores a copy of the data,
        so modifications to the original data dictionary do not affect the
        logged data.
        """
        audit_logger = AuditLogger()

        # Store original data
        original_data = data.copy()

        # Create log entry
        log_entry = audit_logger.log_event(event_type, session_id, data)

        # Modify the original data dictionary
        for key in list(data.keys()):
            data[key] = "modified_value"
        data["new_key"] = "new_value"

        # Property: Log entry data should remain unchanged
        assert log_entry.data == original_data
        assert "new_key" not in log_entry.data

    @given(
        tool_name=st.text(min_size=1, max_size=50),
        session_id=st.text(min_size=1, max_size=100),
        parameters=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            min_size=0,
            max_size=10,
        ),
        result=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            min_size=0,
            max_size=10,
        ),
        execution_time_ms=st.floats(min_value=0.0, max_value=10000.0),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_log_immutability_api_call(
        self,
        tool_name: str,
        session_id: str,
        parameters: dict,
        result: dict,
        execution_time_ms: float,
    ):
        """Property 11 (variant): Log immutability - API call logs should be immutable.

        **Validates: Requirements 7.7**

        This property ensures that API call logs are also immutable and
        cannot be modified after creation.
        """
        audit_logger = AuditLogger()

        # Create API call log
        log_entry = audit_logger.log_api_call(
            tool_name, session_id, parameters, result, execution_time_ms
        )

        # Store original values
        original_tool_name = log_entry.data["tool_name"]
        original_parameters = log_entry.data["parameters"].copy()
        original_result = log_entry.data["result"].copy()

        # Property: Attempting to modify log entry should raise AttributeError
        with pytest.raises(AttributeError, match="immutable"):
            log_entry.data = {"modified": "data"}

        # Property: Original values should remain unchanged
        assert log_entry.data["tool_name"] == original_tool_name
        assert log_entry.data["parameters"] == original_parameters
        assert log_entry.data["result"] == original_result

    @given(
        user_id=st.text(min_size=1, max_size=50),
        session_id=st.text(min_size=1, max_size=100),
        data_type=st.text(min_size=1, max_size=50),
        reason=st.text(min_size=1, max_size=100),
        accessed_fields=st.lists(
            st.text(min_size=1, max_size=50), min_size=0, max_size=10
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_log_immutability_pii_access(
        self,
        user_id: str,
        session_id: str,
        data_type: str,
        reason: str,
        accessed_fields: list,
    ):
        """Property 11 (variant): Log immutability - PII access logs should be immutable.

        **Validates: Requirements 7.7**

        This property ensures that PII access logs are immutable and
        cannot be modified after creation.
        """
        audit_logger = AuditLogger()

        # Create PII access log
        log_entry = audit_logger.log_pii_access(
            user_id, session_id, data_type, reason, accessed_fields
        )

        # Store original values
        original_user_id = log_entry.data["user_id"]
        original_data_type = log_entry.data["data_type"]
        original_reason = log_entry.data["reason"]
        original_accessed_fields = log_entry.data["accessed_fields"].copy()

        # Property: Attempting to modify log entry should raise AttributeError
        with pytest.raises(AttributeError, match="immutable"):
            log_entry.data = {"modified": "data"}

        # Property: Original values should remain unchanged
        assert log_entry.data["user_id"] == original_user_id
        assert log_entry.data["data_type"] == original_data_type
        assert log_entry.data["reason"] == original_reason
        assert log_entry.data["accessed_fields"] == original_accessed_fields

    @given(
        event_type=st.sampled_from([
            AuditLogger.EVENT_CALL_INITIATED,
            AuditLogger.EVENT_CALL_TERMINATED,
            AuditLogger.EVENT_ERROR,
        ]),
        session_id=st.text(min_size=1, max_size=100),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            min_size=1,
            max_size=10,
        ),
        num_queries=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_log_immutability_after_queries(
        self, event_type: str, session_id: str, data: dict, num_queries: int
    ):
        """Property 11 (variant): Log immutability - Log entries should remain immutable even after multiple queries.

        **Validates: Requirements 7.7**

        This property ensures that querying logs does not affect their
        immutability. Log entries should remain unchanged even after
        being retrieved multiple times.
        """
        audit_logger = AuditLogger()

        # Create log entry
        log_entry = audit_logger.log_event(event_type, session_id, data)

        # Store original values
        original_log_id = log_entry.log_id
        original_timestamp = log_entry.timestamp
        original_event_type = log_entry.event_type
        original_session_id = log_entry.session_id
        original_data = log_entry.data.copy()

        # Query logs multiple times
        for _ in range(num_queries):
            results = audit_logger.query_logs(filters={"session_id": session_id})
            assert len(results) > 0

            # Get the log entry from query results
            queried_entry = results[0]

            # Property: Queried entry should have same values as original
            assert queried_entry.log_id == original_log_id
            assert queried_entry.timestamp == original_timestamp
            assert queried_entry.event_type == original_event_type
            assert queried_entry.session_id == original_session_id
            assert queried_entry.data == original_data

            # Property: Attempting to modify queried entry should raise AttributeError
            with pytest.raises(AttributeError, match="immutable"):
                queried_entry.log_id = "modified_id"

    @given(
        events=st.lists(
            st.tuples(
                st.sampled_from([
                    AuditLogger.EVENT_CALL_INITIATED,
                    AuditLogger.EVENT_CALL_TERMINATED,
                    AuditLogger.EVENT_ERROR,
                    AuditLogger.EVENT_LLM_REQUEST,
                ]),
                st.text(min_size=1, max_size=50),
                st.dictionaries(
                    keys=st.text(min_size=1, max_size=20),
                    values=st.text(max_size=50),
                    min_size=1,
                    max_size=5,
                ),
            ),
            min_size=1,
            max_size=20,
        )
    )
    @settings(max_examples=50, deadline=None)
    def test_property_log_immutability_multiple_logs(self, events: list):
        """Property 11 (variant): Log immutability - All log entries should be immutable regardless of how many logs exist.

        **Validates: Requirements 7.7**

        This property ensures that immutability holds for all log entries
        when multiple logs are created, not just individual logs.
        """
        audit_logger = AuditLogger()

        # Create multiple log entries
        log_entries = []
        for event_type, session_id, data in events:
            log_entry = audit_logger.log_event(event_type, session_id, data)
            log_entries.append((log_entry, event_type, session_id, data.copy()))

        # Property: All log entries should be immutable
        for log_entry, original_event_type, original_session_id, original_data in log_entries:
            # Verify immutability
            with pytest.raises(AttributeError, match="immutable"):
                log_entry.log_id = "modified_id"

            with pytest.raises(AttributeError, match="immutable"):
                log_entry.event_type = "modified_event"

            with pytest.raises(AttributeError, match="immutable"):
                log_entry.session_id = "modified_session"

            # Verify original values remain unchanged
            assert log_entry.event_type == original_event_type
            assert log_entry.session_id == original_session_id
            assert log_entry.data == original_data

    @given(
        event_type=st.sampled_from([
            AuditLogger.EVENT_CALL_INITIATED,
            AuditLogger.EVENT_CALL_TERMINATED,
        ]),
        session_id=st.text(min_size=1, max_size=100),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_log_immutability_tokenized_data(
        self, event_type: str, session_id: str, data: dict
    ):
        """Property 11 (variant): Log immutability - Tokenized data should also be immutable.

        **Validates: Requirements 7.7**

        This property ensures that when PII tokenization is enabled,
        the tokenized_data field is also immutable.
        """
        audit_logger = AuditLogger()

        # Create log entry with PII tokenization
        log_entry = audit_logger.log_event(
            event_type, session_id, data, tokenize_pii=True
        )

        # Property: Attempting to modify tokenized_data should raise AttributeError
        if log_entry.tokenized_data is not None:
            with pytest.raises(AttributeError, match="immutable"):
                log_entry.tokenized_data = None

    @given(
        event_type=st.sampled_from([
            AuditLogger.EVENT_CALL_INITIATED,
            AuditLogger.EVENT_ERROR,
        ]),
        session_id=st.text(min_size=1, max_size=100),
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=50),
            values=st.text(max_size=100),
            min_size=1,
            max_size=10,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_property_log_immutability_timestamp_never_changes(
        self, event_type: str, session_id: str, data: dict
    ):
        """Property 11 (variant): Log immutability - Timestamp should never change after log creation.

        **Validates: Requirements 7.7**

        This property specifically tests that the timestamp field,
        which is critical for audit trails, remains immutable.
        """
        audit_logger = AuditLogger()

        # Create log entry
        log_entry = audit_logger.log_event(event_type, session_id, data)

        # Store original timestamp
        original_timestamp = log_entry.timestamp

        # Property: Attempting to modify timestamp should raise AttributeError
        with pytest.raises(AttributeError, match="immutable"):
            log_entry.timestamp = datetime.utcnow()

        with pytest.raises(AttributeError, match="immutable"):
            log_entry.timestamp = original_timestamp + timedelta(seconds=1)

        # Property: Timestamp should remain unchanged
        assert log_entry.timestamp == original_timestamp
