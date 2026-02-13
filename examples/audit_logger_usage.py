"""
Example usage of AuditLogger for compliance and debugging.

This example demonstrates:
1. Logging general events (call lifecycle)
2. Logging API/tool executions
3. Logging PII access for compliance
4. Querying logs with filters
5. PII tokenization in logs
"""

from datetime import datetime, timedelta

from voice_agent.monitoring.audit_logger import AuditLogger, DateRange
from voice_agent.security.encryption import EncryptionService


def main() -> None:
    """Demonstrate AuditLogger usage."""
    # Initialize audit logger
    encryption_service = EncryptionService()
    audit_logger = AuditLogger(encryption_service=encryption_service)

    print("=== AuditLogger Example ===\n")

    # Example 1: Log call initiation
    print("1. Logging call initiation...")
    session_id = "session-12345"
    call_log = audit_logger.log_event(
        event_type=AuditLogger.EVENT_CALL_INITIATED,
        session_id=session_id,
        data={
            "caller_id": "+1-555-123-4567",
            "tenant_id": "healthcare-clinic-001",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )
    print(f"   Log ID: {call_log.log_id}")
    print(f"   Event Type: {call_log.event_type}")
    print(f"   Session ID: {call_log.session_id}\n")

    # Example 2: Log transcript with PII tokenization
    print("2. Logging transcript with PII tokenization...")
    transcript_log = audit_logger.log_event(
        event_type=AuditLogger.EVENT_TRANSCRIPT_SEGMENT,
        session_id=session_id,
        data={
            "transcript": "My name is John Doe, email john.doe@example.com, phone 555-987-6543",
            "speaker": "user",
            "timestamp": datetime.utcnow().isoformat(),
        },
        tokenize_pii=True,
    )
    print(f"   Original data: {transcript_log.data['transcript']}")
    if transcript_log.tokenized_data:
        print(f"   Tokenized: {transcript_log.tokenized_data.text}")
        print(f"   Tokens created: {len(transcript_log.tokenized_data.tokens)}\n")

    # Example 3: Log API/tool execution
    print("3. Logging API call (FHIR appointment search)...")
    api_log = audit_logger.log_api_call(
        tool_name="search_appointments",
        session_id=session_id,
        parameters={
            "patient_id": "P12345",
            "specialty": "cardiology",
            "date_range": "2024-01-15 to 2024-01-22",
        },
        result={
            "slots_found": 3,
            "slots": [
                {"id": "slot-1", "time": "2024-01-16 10:00"},
                {"id": "slot-2", "time": "2024-01-17 14:00"},
                {"id": "slot-3", "time": "2024-01-19 09:00"},
            ],
        },
        execution_time_ms=245.7,
    )
    print(f"   Tool: {api_log.data['tool_name']}")
    print(f"   Execution time: {api_log.data['execution_time_ms']}ms")
    print(f"   Result: {api_log.data['result']['slots_found']} slots found\n")

    # Example 4: Log PII access for compliance
    print("4. Logging PII access for compliance...")
    pii_log = audit_logger.log_pii_access(
        user_id="llm-service",
        session_id=session_id,
        data_type="patient_record",
        reason="appointment_scheduling",
        accessed_fields=["name", "date_of_birth", "phone", "email"],
    )
    print(f"   User: {pii_log.data['user_id']}")
    print(f"   Data type: {pii_log.data['data_type']}")
    print(f"   Reason: {pii_log.data['reason']}")
    print(f"   Fields accessed: {', '.join(pii_log.data['accessed_fields'])}\n")

    # Example 5: Log LLM request and response
    print("5. Logging LLM interaction...")
    audit_logger.log_event(
        event_type=AuditLogger.EVENT_LLM_REQUEST,
        session_id=session_id,
        data={
            "model": "gpt-4",
            "prompt": "User wants to schedule cardiology appointment",
            "temperature": 0.7,
        },
    )
    audit_logger.log_event(
        event_type=AuditLogger.EVENT_LLM_RESPONSE,
        session_id=session_id,
        data={
            "response": "I'd be happy to help schedule a cardiology appointment",
            "intent": "schedule_appointment",
            "confidence": 0.95,
            "tool_call": "search_appointments",
        },
    )
    print("   LLM request and response logged\n")

    # Example 6: Log error
    print("6. Logging error event...")
    audit_logger.log_event(
        event_type=AuditLogger.EVENT_ERROR,
        session_id=session_id,
        data={
            "component": "FHIR_Adapter",
            "error": "Connection timeout",
            "retry_count": 2,
        },
    )
    print("   Error logged\n")

    # Example 7: Log call termination
    print("7. Logging call termination...")
    audit_logger.log_event(
        event_type=AuditLogger.EVENT_CALL_TERMINATED,
        session_id=session_id,
        data={
            "duration_seconds": 180,
            "reason": "completed",
            "outcome": "appointment_scheduled",
        },
    )
    print("   Call termination logged\n")

    # Example 8: Query logs by session
    print("8. Querying logs by session ID...")
    session_logs = audit_logger.query_logs(filters={"session_id": session_id})
    print(f"   Found {len(session_logs)} logs for session {session_id}")
    for log in session_logs:
        print(f"   - {log.event_type} at {log.timestamp.isoformat()}")
    print()

    # Example 9: Query logs by event type
    print("9. Querying logs by event type (errors)...")
    error_logs = audit_logger.query_logs(
        filters={"event_type": AuditLogger.EVENT_ERROR}
    )
    print(f"   Found {len(error_logs)} error logs")
    for log in error_logs:
        print(f"   - {log.data.get('component')}: {log.data.get('error')}")
    print()

    # Example 10: Query logs by date range
    print("10. Querying logs by date range (last 5 minutes)...")
    now = datetime.utcnow()
    date_range = DateRange(start=now - timedelta(minutes=5), end=now)
    recent_logs = audit_logger.query_logs(date_range=date_range)
    print(f"   Found {len(recent_logs)} logs in the last 5 minutes\n")

    # Example 11: Query PII access logs
    print("11. Querying PII access logs...")
    pii_access_logs = audit_logger.query_logs(
        filters={"event_type": AuditLogger.EVENT_PII_ACCESS}
    )
    print(f"   Found {len(pii_access_logs)} PII access logs")
    for log in pii_access_logs:
        print(
            f"   - {log.data['user_id']} accessed {log.data['data_type']} "
            f"for {log.data['reason']}"
        )
    print()

    # Example 12: Query API calls with execution time
    print("12. Querying API calls...")
    api_logs = audit_logger.query_logs(
        filters={"event_type": AuditLogger.EVENT_TOOL_EXECUTION}
    )
    print(f"   Found {len(api_logs)} API call logs")
    for log in api_logs:
        tool_name = log.data.get("tool_name")
        exec_time = log.data.get("execution_time_ms")
        print(f"   - {tool_name}: {exec_time}ms")
    print()

    # Example 13: Demonstrate log immutability
    print("13. Demonstrating log immutability...")
    try:
        call_log.event_type = "modified_event"
        print("   ERROR: Log was modified (should not happen)")
    except AttributeError:
        print("   ✓ Log is immutable - modification prevented\n")

    # Summary
    print("=== Summary ===")
    print(f"Total logs created: {audit_logger.get_log_count()}")
    print(f"Session logs: {len(session_logs)}")
    print(f"Error logs: {len(error_logs)}")
    print(f"PII access logs: {len(pii_access_logs)}")
    print(f"API call logs: {len(api_logs)}")


if __name__ == "__main__":
    main()
