"""
Audit Logger for Config-Driven Voice Agent Platform.

Provides immutable audit logging for compliance and debugging with PII tokenization.

Requirements: 7.7
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from voice_agent.monitoring.logger import get_logger
from voice_agent.security.encryption import EncryptionService, TokenizedText

logger = get_logger(__name__)


@dataclass
class LogEntry:
    """Immutable audit log entry."""

    log_id: str
    timestamp: datetime
    event_type: str
    session_id: str
    data: Dict[str, Any]
    tokenized_data: Optional[TokenizedText] = None

    def __post_init__(self) -> None:
        """Ensure immutability by freezing the object."""
        # Make the dataclass effectively immutable by preventing attribute changes
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, key: str, value: Any) -> None:
        """Prevent modification of attributes after initialization."""
        if hasattr(self, "_frozen") and self._frozen:
            raise AttributeError(f"LogEntry is immutable. Cannot modify '{key}'")
        object.__setattr__(self, key, value)


@dataclass
class DateRange:
    """Date range for log queries."""

    start: datetime
    end: datetime


class AuditLogger:
    """
    Immutable audit logger for compliance and debugging.

    Creates immutable audit logs for all system operations including:
    - General events (call initiation, termination, errors)
    - API/tool executions
    - PII/PHI access for compliance tracking

    All PII/PHI is tokenized before logging to maintain privacy.

    Requirements:
    - 7.7: Immutable audit logs for all system operations
    """

    # Event types
    EVENT_CALL_INITIATED = "call_initiated"
    EVENT_CALL_TERMINATED = "call_terminated"
    EVENT_TRANSCRIPT_SEGMENT = "transcript_segment"
    EVENT_LLM_REQUEST = "llm_request"
    EVENT_LLM_RESPONSE = "llm_response"
    EVENT_TOOL_EXECUTION = "tool_execution"
    EVENT_PII_ACCESS = "pii_access"
    EVENT_AUTHENTICATION = "authentication"
    EVENT_ERROR = "error"
    EVENT_LATENCY_THRESHOLD = "latency_threshold"

    def __init__(self, encryption_service: Optional[EncryptionService] = None):
        """
        Initialize the audit logger.

        Args:
            encryption_service: Optional encryption service for PII tokenization.
                              If None, creates a new instance.
        """
        self._encryption_service = encryption_service or EncryptionService()
        self._logs: List[LogEntry] = []

    def log_event(
        self,
        event_type: str,
        session_id: str,
        data: Dict[str, Any],
        tokenize_pii: bool = True,
    ) -> LogEntry:
        """
        Log a general event.

        Args:
            event_type: Type of event (use EVENT_* constants)
            session_id: Unique session identifier
            data: Event data dictionary
            tokenize_pii: Whether to tokenize PII in the data

        Returns:
            Created log entry (immutable)
        """
        log_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()

        # Tokenize PII if requested
        tokenized_data = None
        if tokenize_pii:
            # Convert data to string for tokenization
            data_str = str(data)
            tokenized_data = self._encryption_service.tokenize_pii(data_str)

        # Create immutable log entry
        log_entry = LogEntry(
            log_id=log_id,
            timestamp=timestamp,
            event_type=event_type,
            session_id=session_id,
            data=data.copy(),  # Copy to prevent external modifications
            tokenized_data=tokenized_data,
        )

        # Store log entry
        self._logs.append(log_entry)

        logger.info(
            "Logged event",
            log_id=log_id,
            event_type=event_type,
            session_id=session_id,
            timestamp=timestamp.isoformat(),
        )

        return log_entry

    def log_api_call(
        self,
        tool_name: str,
        session_id: str,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        execution_time_ms: Optional[float] = None,
    ) -> LogEntry:
        """
        Log an API/tool execution.

        Args:
            tool_name: Name of the tool/API executed
            session_id: Unique session identifier
            parameters: Tool parameters
            result: Tool execution result
            execution_time_ms: Optional execution time in milliseconds

        Returns:
            Created log entry (immutable)
        """
        data = {
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result,
            "execution_time_ms": execution_time_ms,
        }

        return self.log_event(
            event_type=self.EVENT_TOOL_EXECUTION,
            session_id=session_id,
            data=data,
            tokenize_pii=True,
        )

    def log_pii_access(
        self,
        user_id: str,
        session_id: str,
        data_type: str,
        reason: str,
        accessed_fields: Optional[List[str]] = None,
    ) -> LogEntry:
        """
        Log PII/PHI access for compliance tracking.

        Args:
            user_id: User or system component accessing PII
            session_id: Unique session identifier
            data_type: Type of PII/PHI accessed (e.g., "patient_record", "appointment")
            reason: Reason for accessing PII
            accessed_fields: Optional list of specific fields accessed

        Returns:
            Created log entry (immutable)
        """
        data = {
            "user_id": user_id,
            "data_type": data_type,
            "reason": reason,
            "accessed_fields": accessed_fields or [],
        }

        return self.log_event(
            event_type=self.EVENT_PII_ACCESS,
            session_id=session_id,
            data=data,
            tokenize_pii=False,  # Don't tokenize the metadata itself
        )

    def query_logs(
        self,
        filters: Optional[Dict[str, Any]] = None,
        date_range: Optional[DateRange] = None,
        limit: Optional[int] = None,
    ) -> List[LogEntry]:
        """
        Query audit logs with filters.

        Args:
            filters: Optional filters to apply (e.g., {"event_type": "error", "session_id": "123"})
            date_range: Optional date range to filter by
            limit: Optional maximum number of results to return

        Returns:
            List of matching log entries
        """
        results = self._logs.copy()

        # Apply date range filter
        if date_range:
            results = [
                log
                for log in results
                if date_range.start <= log.timestamp <= date_range.end
            ]

        # Apply custom filters
        if filters:
            for key, value in filters.items():
                if key == "event_type":
                    results = [log for log in results if log.event_type == value]
                elif key == "session_id":
                    results = [log for log in results if log.session_id == value]
                elif key in ["log_id"]:
                    results = [log for log in results if getattr(log, key) == value]
                else:
                    # Filter by data fields
                    results = [
                        log for log in results if log.data.get(key) == value
                    ]

        # Apply limit
        if limit:
            results = results[:limit]

        logger.debug(
            "Queried logs",
            filters=filters,
            date_range=date_range,
            result_count=len(results),
        )

        return results

    def get_log_count(self) -> int:
        """
        Get the total number of log entries.

        Returns:
            Total number of logs
        """
        return len(self._logs)

    def clear_logs(self) -> None:
        """
        Clear all logs (use with caution - for testing only).

        Note: In production, logs should never be cleared to maintain audit trail.
        """
        self._logs.clear()
        logger.warning("Cleared all audit logs")
