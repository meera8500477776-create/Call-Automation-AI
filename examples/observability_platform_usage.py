"""
Example usage of ObservabilityPlatform for monitoring and tracing.

This example demonstrates:
- Tracing voice interactions
- Sentiment analysis
- Hallucination detection
- Alert generation
- Dashboard metrics retrieval
"""

from voice_agent.monitoring.observability_platform import (
    AlertSeverity,
    ObservabilityPlatform,
)


def main():
    """Demonstrate ObservabilityPlatform usage."""
    print("=== Observability Platform Example ===\n")

    # Initialize platform with API keys
    platform = ObservabilityPlatform(
        langsmith_api_key="your_langsmith_key",
        helicone_api_key="your_helicone_key",
        twilio_ci_enabled=True,
    )

    # Example 1: Trace an interaction
    print("1. Tracing interaction...")
    session_id = "session_12345"
    trace_data = {
        "transcript": "I need to schedule an appointment",
        "llm_request": "User wants to schedule appointment",
        "llm_response": "I'd be happy to help you schedule an appointment. What date works for you?",
        "llm_tokens": 45,
        "llm_cost": 0.0009,
        "latency_ms": 750,
        "tool_calls": ["search_appointments"],
    }
    platform.trace_interaction(session_id, trace_data)
    print(f"   Traced interaction for session {session_id}")
    print(f"   Total traces: {platform.get_trace_count()}\n")

    # Example 2: Analyze sentiment
    print("2. Analyzing sentiment...")
    positive_transcript = "Thank you so much! This is great and very helpful."
    sentiment = platform.analyze_sentiment(session_id, positive_transcript)
    print(f"   Transcript: {positive_transcript}")
    print(f"   Sentiment: {sentiment.label} (score: {sentiment.score:.2f})")
    print(f"   Confidence: {sentiment.confidence:.2f}\n")

    negative_transcript = "This is terrible and very frustrating."
    sentiment = platform.analyze_sentiment(session_id, negative_transcript)
    print(f"   Transcript: {negative_transcript}")
    print(f"   Sentiment: {sentiment.label} (score: {sentiment.score:.2f})")
    print(f"   Confidence: {sentiment.confidence:.2f}\n")

    # Example 3: Detect hallucinations
    print("3. Detecting hallucinations...")

    # Good response - acknowledges boundaries
    good_response = "I'm not sure about that specific detail. Let me check with the system."
    context = {"facts": ["appointment system available"]}
    hallucination = platform.detect_hallucination(good_response, context)
    print(f"   Response: {good_response}")
    print(f"   Hallucination score: {hallucination.score:.2f}")
    print(f"   Is hallucination: {hallucination.is_hallucination}")
    print(f"   Reasons: {hallucination.reasons}\n")

    # Potentially problematic response
    problematic_response = "Definitely, I can absolutely guarantee your appointment is at 3:45 PM on December 25th."
    hallucination = platform.detect_hallucination(problematic_response, context)
    print(f"   Response: {problematic_response}")
    print(f"   Hallucination score: {hallucination.score:.2f}")
    print(f"   Is hallucination: {hallucination.is_hallucination}")
    print(f"   Reasons: {hallucination.reasons}\n")

    # Example 4: Generate alerts
    print("4. Generating alerts...")

    # High latency alert
    platform.generate_alert(
        alert_type=platform.ALERT_HIGH_LATENCY,
        severity=AlertSeverity.WARNING,
        details={
            "session_id": session_id,
            "latency_ms": 1500,
            "stage": "llm_total",
        },
    )
    print("   Generated high latency alert")

    # System error alert
    platform.generate_alert(
        alert_type=platform.ALERT_SYSTEM_ERROR,
        severity=AlertSeverity.ERROR,
        details={
            "error": "Failed to connect to FHIR server",
            "session_id": session_id,
        },
    )
    print("   Generated system error alert")
    print(f"   Total alerts: {platform.get_alert_count()}\n")

    # Example 5: Track session and call metrics
    print("5. Tracking metrics...")
    platform.increment_session_count(5)  # 5 active sessions
    for _ in range(10):
        platform.increment_call_count()  # 10 total calls
    platform.increment_error_count()  # 1 error
    print("   Updated session and call metrics\n")

    # Example 6: Get dashboard metrics
    print("6. Retrieving dashboard metrics...")
    metrics = platform.get_dashboard_metrics()
    print(f"   Timestamp: {metrics.timestamp}")
    print(f"   Active sessions: {metrics.active_sessions}")
    print(f"   Total calls today: {metrics.total_calls_today}")
    print(f"   Average latency: {metrics.average_latency_ms:.2f} ms")
    print(f"   Error rate: {metrics.error_rate:.2%}")
    print(f"   Hallucination rate: {metrics.hallucination_rate:.2%}")
    print(f"   Average sentiment: {metrics.average_sentiment:.2f}")
    print(f"   LLM token usage: {metrics.llm_token_usage}")
    print(f"   LLM cost: ${metrics.llm_cost_usd:.4f}\n")

    # Example 7: Simulate a complete call flow
    print("7. Simulating complete call flow...")
    call_session = "session_67890"

    # Start call
    platform.increment_session_count(1)
    platform.increment_call_count()

    # Trace initial interaction
    platform.trace_interaction(
        call_session,
        {
            "transcript": "Hello, I need help",
            "llm_response": "Hello! I'm here to help. What can I do for you?",
            "llm_tokens": 30,
            "llm_cost": 0.0006,
        },
    )

    # Analyze sentiment
    sentiment = platform.analyze_sentiment(
        call_session,
        "Hello, I need help with scheduling",
    )

    # Check for hallucinations
    hallucination = platform.detect_hallucination(
        "I can help you schedule an appointment.",
        {"facts": ["scheduling available"]},
    )

    # End call
    platform.increment_session_count(-1)

    print(f"   Completed call flow for {call_session}")
    print(f"   Sentiment: {sentiment.label}")
    print(f"   Hallucination detected: {hallucination.is_hallucination}\n")

    # Final metrics
    print("8. Final system state...")
    final_metrics = platform.get_dashboard_metrics()
    print(f"   Active sessions: {final_metrics.active_sessions}")
    print(f"   Total calls: {final_metrics.total_calls_today}")
    print(f"   Total traces: {platform.get_trace_count()}")
    print(f"   Total alerts: {platform.get_alert_count()}")


if __name__ == "__main__":
    main()
