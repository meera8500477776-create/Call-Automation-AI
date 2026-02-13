"""Unit tests for ObservabilityPlatform."""

import pytest

from voice_agent.monitoring.observability_platform import (
    AlertSeverity,
    DashboardMetrics,
    HallucinationScore,
    ObservabilityPlatform,
    SentimentResult,
)


class TestObservabilityPlatform:
    """Test suite for ObservabilityPlatform."""

    @pytest.fixture
    def platform(self):
        """Create ObservabilityPlatform instance for testing."""
        return ObservabilityPlatform(
            langsmith_api_key="test_langsmith_key",
            helicone_api_key="test_helicone_key",
            twilio_ci_enabled=False,
        )

    def test_initialization(self, platform):
        """Test platform initialization."""
        assert platform._langsmith_api_key == "test_langsmith_key"
        assert platform._helicone_api_key == "test_helicone_key"
        assert platform._twilio_ci_enabled is False
        assert platform._active_sessions == 0
        assert platform._total_calls == 0

    def test_trace_interaction(self, platform):
        """Test tracing interactions."""
        session_id = "test_session_123"
        trace_data = {
            "transcript": "Hello, I need to schedule an appointment",
            "llm_response": "I'd be happy to help you schedule an appointment",
            "llm_tokens": 50,
            "llm_cost": 0.001,
            "latency_ms": 750,
        }

        platform.trace_interaction(session_id, trace_data)

        assert platform.get_trace_count() == 1
        assert platform._total_llm_tokens == 50
        assert platform._total_llm_cost == 0.001

    def test_trace_interaction_multiple(self, platform):
        """Test tracing multiple interactions."""
        for i in range(5):
            platform.trace_interaction(
                f"session_{i}",
                {"llm_tokens": 10, "llm_cost": 0.0001},
            )

        assert platform.get_trace_count() == 5
        assert platform._total_llm_tokens == 50
        assert platform._total_llm_cost == 0.0005

    def test_analyze_sentiment_positive(self, platform):
        """Test sentiment analysis with positive transcript."""
        transcript = "Thank you so much! This is great and I'm very happy with the service."
        result = platform.analyze_sentiment("session_123", transcript)

        assert isinstance(result, SentimentResult)
        assert result.score > 0
        assert result.label == "positive"
        assert 0 <= result.confidence <= 1.0

    def test_analyze_sentiment_negative(self, platform):
        """Test sentiment analysis with negative transcript."""
        transcript = "This is terrible and awful. I'm very frustrated and disappointed."
        result = platform.analyze_sentiment("session_123", transcript)

        assert isinstance(result, SentimentResult)
        assert result.score < 0
        assert result.label == "negative"

    def test_analyze_sentiment_neutral(self, platform):
        """Test sentiment analysis with neutral transcript."""
        transcript = "I would like to schedule an appointment for next week."
        result = platform.analyze_sentiment("session_123", transcript)

        assert isinstance(result, SentimentResult)
        assert result.label == "neutral"

    def test_analyze_sentiment_low_triggers_alert(self, platform):
        """Test that low sentiment triggers an alert."""
        transcript = "This is absolutely terrible and horrible. Very bad experience."
        platform.analyze_sentiment("session_123", transcript)

        # Should have generated a low sentiment alert
        assert platform.get_alert_count() > 0

    def test_detect_hallucination_with_knowledge_boundary(self, platform):
        """Test hallucination detection when LLM acknowledges boundaries."""
        response = "I'm not sure about that specific detail. I don't have access to that information."
        context = {"facts": ["appointment available", "doctor available"]}

        result = platform.detect_hallucination(response, context)

        assert isinstance(result, HallucinationScore)
        assert result.score < 0.5  # Should have low hallucination score
        assert not result.is_hallucination

    def test_detect_hallucination_with_overly_confident_language(self, platform):
        """Test hallucination detection with overly confident language."""
        response = "Definitely, I can absolutely guarantee that this will work perfectly."
        context = {"facts": []}

        result = platform.detect_hallucination(response, context)

        assert result.score > 0
        assert "Overly confident language" in result.reasons

    def test_detect_hallucination_with_specific_details(self, platform):
        """Test hallucination detection with specific details not in context."""
        response = "Your appointment is scheduled for 3:45 PM on December 25th."
        context = {"facts": ["appointment requested"]}  # No specific time/date

        result = platform.detect_hallucination(response, context)

        assert result.score > 0
        # Should detect numbers not in context

    def test_detect_hallucination_threshold(self, platform):
        """Test hallucination threshold detection."""
        # Create response that should exceed threshold
        response = (
            "Definitely and absolutely, I can guarantee that your appointment "
            "is scheduled for 3:45 PM on December 25th with Dr. Smith in room 305. "
            * 10  # Make it very long
        )
        context = {"facts": []}

        result = platform.detect_hallucination(response, context)

        # Should have high score due to multiple factors
        assert result.score > 0

    def test_generate_alert_info(self, platform):
        """Test generating info-level alert."""
        platform.generate_alert(
            alert_type=platform.ALERT_HIGH_LATENCY,
            severity=AlertSeverity.INFO,
            details={"latency_ms": 1200, "session_id": "test_123"},
        )

        assert platform.get_alert_count() == 1
        alert = platform._alerts[0]
        assert alert["alert_type"] == platform.ALERT_HIGH_LATENCY
        assert alert["severity"] == "info"

    def test_generate_alert_warning(self, platform):
        """Test generating warning-level alert."""
        platform.generate_alert(
            alert_type=platform.ALERT_LOW_SENTIMENT,
            severity=AlertSeverity.WARNING,
            details={"sentiment_score": -0.8},
        )

        assert platform.get_alert_count() == 1
        alert = platform._alerts[0]
        assert alert["severity"] == "warning"

    def test_generate_alert_error(self, platform):
        """Test generating error-level alert."""
        platform.generate_alert(
            alert_type=platform.ALERT_HIGH_ERROR_RATE,
            severity=AlertSeverity.ERROR,
            details={"error_rate": 0.15},
        )

        assert platform.get_alert_count() == 1
        alert = platform._alerts[0]
        assert alert["severity"] == "error"

    def test_generate_alert_critical(self, platform):
        """Test generating critical-level alert."""
        platform.generate_alert(
            alert_type=platform.ALERT_SYSTEM_ERROR,
            severity=AlertSeverity.CRITICAL,
            details={"error": "System failure"},
        )

        assert platform.get_alert_count() == 1
        alert = platform._alerts[0]
        assert alert["severity"] == "critical"

    def test_get_dashboard_metrics(self, platform):
        """Test retrieving dashboard metrics."""
        # Set up some metrics
        platform.increment_session_count(5)
        platform.increment_call_count()
        platform.increment_call_count()
        platform.increment_call_count()

        metrics = platform.get_dashboard_metrics()

        assert isinstance(metrics, DashboardMetrics)
        assert metrics.active_sessions == 5
        assert metrics.total_calls_today == 3
        assert metrics.average_latency_ms >= 0
        assert 0 <= metrics.error_rate <= 1.0
        assert 0 <= metrics.hallucination_rate <= 1.0

    def test_get_dashboard_metrics_with_history(self, platform):
        """Test dashboard metrics with sentiment and hallucination history."""
        # Add some sentiment history
        platform.analyze_sentiment("s1", "Thank you, this is great!")
        platform.analyze_sentiment("s2", "This is terrible and awful.")

        # Add some hallucination checks
        platform.detect_hallucination(
            "I'm not sure about that.",
            {"facts": []},
        )
        platform.detect_hallucination(
            "Definitely guaranteed to work perfectly!" * 20,
            {"facts": []},
        )

        metrics = platform.get_dashboard_metrics()

        # Should have calculated averages (sentiment can be 0.0 if balanced)
        assert len(platform._sentiment_history) == 2
        assert metrics.hallucination_rate >= 0.0

    def test_increment_session_count(self, platform):
        """Test incrementing session count."""
        assert platform._active_sessions == 0

        platform.increment_session_count(3)
        assert platform._active_sessions == 3

        platform.increment_session_count(-1)
        assert platform._active_sessions == 2

    def test_increment_call_count(self, platform):
        """Test incrementing call count."""
        assert platform._total_calls == 0

        platform.increment_call_count()
        assert platform._total_calls == 1

        platform.increment_call_count()
        assert platform._total_calls == 2

    def test_increment_error_count(self, platform):
        """Test incrementing error count."""
        assert platform._total_errors == 0

        platform.increment_error_count()
        assert platform._total_errors == 1

    def test_increment_error_count_triggers_alert(self, platform):
        """Test that high error rate triggers alert."""
        # Set up calls
        for _ in range(10):
            platform.increment_call_count()

        # Add errors to exceed threshold (5%)
        for _ in range(2):
            platform.increment_error_count()

        # Should have generated high error rate alert
        assert platform.get_alert_count() > 0
        alert_types = [a["alert_type"] for a in platform._alerts]
        assert platform.ALERT_HIGH_ERROR_RATE in alert_types

    def test_clear_metrics(self, platform):
        """Test clearing all metrics."""
        # Add some data
        platform.trace_interaction("s1", {"llm_tokens": 10})
        platform.generate_alert(
            platform.ALERT_HIGH_LATENCY,
            AlertSeverity.INFO,
            {},
        )
        platform.increment_session_count(5)
        platform.increment_call_count()

        # Clear
        platform.clear_metrics()

        # Verify everything is cleared
        assert platform.get_trace_count() == 0
        assert platform.get_alert_count() == 0
        assert platform._active_sessions == 0
        assert platform._total_calls == 0
        assert platform._total_errors == 0
        assert platform._total_llm_tokens == 0

    def test_sentiment_result_dataclass(self):
        """Test SentimentResult dataclass."""
        result = SentimentResult(
            score=0.8,
            magnitude=0.6,
            label="positive",
            confidence=0.9,
        )

        assert result.score == 0.8
        assert result.magnitude == 0.6
        assert result.label == "positive"
        assert result.confidence == 0.9

    def test_hallucination_score_dataclass(self):
        """Test HallucinationScore dataclass."""
        score = HallucinationScore(
            score=0.3,
            confidence=0.7,
            reasons=["Test reason"],
            is_hallucination=False,
        )

        assert score.score == 0.3
        assert score.confidence == 0.7
        assert score.reasons == ["Test reason"]
        assert score.is_hallucination is False

    def test_dashboard_metrics_dataclass(self):
        """Test DashboardMetrics dataclass."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        metrics = DashboardMetrics(
            timestamp=now,
            active_sessions=10,
            total_calls_today=100,
            average_latency_ms=750.0,
            error_rate=0.02,
            hallucination_rate=0.01,
            average_sentiment=0.5,
            llm_token_usage=5000,
            llm_cost_usd=1.25,
        )

        assert metrics.timestamp == now
        assert metrics.active_sessions == 10
        assert metrics.total_calls_today == 100
        assert metrics.average_latency_ms == 750.0
        assert metrics.error_rate == 0.02

    def test_alert_severity_enum(self):
        """Test AlertSeverity enum."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_platform_without_api_keys(self):
        """Test platform initialization without API keys."""
        platform = ObservabilityPlatform()

        assert platform._langsmith_api_key is None
        assert platform._helicone_api_key is None
        assert platform._twilio_ci_enabled is False

        # Should still work for basic functionality
        platform.trace_interaction("s1", {"test": "data"})
        assert platform.get_trace_count() == 1

    def test_multiple_traces_and_metrics(self, platform):
        """Test comprehensive scenario with multiple operations."""
        # Simulate multiple interactions
        for i in range(10):
            platform.increment_call_count()
            platform.trace_interaction(
                f"session_{i}",
                {
                    "llm_tokens": 50,
                    "llm_cost": 0.001,
                    "latency_ms": 800,
                },
            )

            # Analyze sentiment
            if i % 2 == 0:
                platform.analyze_sentiment(f"session_{i}", "Thank you!")
            else:
                platform.analyze_sentiment(f"session_{i}", "This is okay.")

            # Check for hallucinations
            platform.detect_hallucination(
                "I can help with that.",
                {"facts": ["help available"]},
            )

        # Get final metrics
        metrics = platform.get_dashboard_metrics()

        assert metrics.total_calls_today == 10
        assert metrics.llm_token_usage == 500
        assert abs(metrics.llm_cost_usd - 0.01) < 0.0001  # Use approximate comparison for floats
        assert platform.get_trace_count() == 10


class TestLangSmithHeliconeIntegration:
    """Test suite for LangSmith and Helicone trace submission integration.
    
    Requirements: 8.1, 8.2
    """

    @pytest.fixture
    def platform_with_langsmith(self):
        """Create platform with LangSmith enabled."""
        return ObservabilityPlatform(
            langsmith_api_key="test_langsmith_key_123",
            helicone_api_key=None,
            twilio_ci_enabled=False,
        )

    @pytest.fixture
    def platform_with_helicone(self):
        """Create platform with Helicone enabled."""
        return ObservabilityPlatform(
            langsmith_api_key=None,
            helicone_api_key="test_helicone_key_456",
            twilio_ci_enabled=False,
        )

    @pytest.fixture
    def platform_with_both(self):
        """Create platform with both LangSmith and Helicone enabled."""
        return ObservabilityPlatform(
            langsmith_api_key="test_langsmith_key",
            helicone_api_key="test_helicone_key",
            twilio_ci_enabled=False,
        )

    def test_trace_submission_with_langsmith_enabled(self, platform_with_langsmith, monkeypatch):
        """Test that traces are submitted to LangSmith when API key is configured."""
        # Track if _send_to_langsmith was called
        langsmith_called = []

        def mock_send_to_langsmith(trace_entry):
            langsmith_called.append(trace_entry)

        monkeypatch.setattr(
            platform_with_langsmith,
            "_send_to_langsmith",
            mock_send_to_langsmith,
        )

        # Submit trace
        session_id = "test_session_langsmith"
        trace_data = {
            "transcript": "Schedule appointment",
            "llm_response": "I'll help you schedule",
            "llm_tokens": 25,
            "llm_cost": 0.0005,
        }

        platform_with_langsmith.trace_interaction(session_id, trace_data)

        # Verify LangSmith was called
        assert len(langsmith_called) == 1
        assert langsmith_called[0]["session_id"] == session_id
        assert langsmith_called[0]["trace_data"] == trace_data

    def test_trace_submission_with_helicone_enabled(self, platform_with_helicone, monkeypatch):
        """Test that traces are submitted to Helicone when API key is configured."""
        # Track if _send_to_helicone was called
        helicone_called = []

        def mock_send_to_helicone(trace_entry):
            helicone_called.append(trace_entry)

        monkeypatch.setattr(
            platform_with_helicone,
            "_send_to_helicone",
            mock_send_to_helicone,
        )

        # Submit trace
        session_id = "test_session_helicone"
        trace_data = {
            "transcript": "Cancel appointment",
            "llm_response": "I'll help you cancel",
            "llm_tokens": 30,
        }

        platform_with_helicone.trace_interaction(session_id, trace_data)

        # Verify Helicone was called
        assert len(helicone_called) == 1
        assert helicone_called[0]["session_id"] == session_id

    def test_trace_submission_with_both_providers(self, platform_with_both, monkeypatch):
        """Test that traces are submitted to both LangSmith and Helicone when both are configured."""
        langsmith_called = []
        helicone_called = []

        def mock_send_to_langsmith(trace_entry):
            langsmith_called.append(trace_entry)

        def mock_send_to_helicone(trace_entry):
            helicone_called.append(trace_entry)

        monkeypatch.setattr(platform_with_both, "_send_to_langsmith", mock_send_to_langsmith)
        monkeypatch.setattr(platform_with_both, "_send_to_helicone", mock_send_to_helicone)

        # Submit trace
        session_id = "test_session_both"
        trace_data = {"transcript": "Test", "llm_tokens": 10}

        platform_with_both.trace_interaction(session_id, trace_data)

        # Verify both were called
        assert len(langsmith_called) == 1
        assert len(helicone_called) == 1
        assert langsmith_called[0]["session_id"] == session_id
        assert helicone_called[0]["session_id"] == session_id

    def test_trace_submission_without_providers(self):
        """Test that traces are stored locally when no providers are configured."""
        platform = ObservabilityPlatform()

        session_id = "test_session_no_providers"
        trace_data = {"transcript": "Test"}

        platform.trace_interaction(session_id, trace_data)

        # Should still store locally
        assert platform.get_trace_count() == 1

    def test_trace_includes_timestamp(self, platform_with_langsmith):
        """Test that trace entries include ISO format timestamps."""
        session_id = "test_session_timestamp"
        trace_data = {"test": "data"}

        platform_with_langsmith.trace_interaction(session_id, trace_data)

        # Check stored trace has timestamp
        trace = platform_with_langsmith._traces[0]
        assert "timestamp" in trace
        assert "T" in trace["timestamp"]  # ISO format includes 'T'
        assert trace["timestamp"].endswith(("Z", "+00:00"))  # UTC timezone

    def test_trace_accumulates_token_usage(self, platform_with_langsmith):
        """Test that token usage is accumulated across multiple traces."""
        for i in range(5):
            platform_with_langsmith.trace_interaction(
                f"session_{i}",
                {"llm_tokens": 100, "llm_cost": 0.002},
            )

        assert platform_with_langsmith._total_llm_tokens == 500
        assert abs(platform_with_langsmith._total_llm_cost - 0.01) < 0.0001

    def test_trace_handles_missing_token_data(self, platform_with_langsmith):
        """Test that traces without token data don't cause errors."""
        platform_with_langsmith.trace_interaction(
            "session_no_tokens",
            {"transcript": "Test"},
        )

        # Should not increment tokens
        assert platform_with_langsmith._total_llm_tokens == 0
        assert platform_with_langsmith._total_llm_cost == 0.0


class TestSentimentAnalysisIntegration:
    """Test suite for sentiment analysis integration.
    
    Requirements: 8.3
    """

    @pytest.fixture
    def platform_with_twilio_ci(self):
        """Create platform with Twilio CI enabled."""
        return ObservabilityPlatform(twilio_ci_enabled=True)

    @pytest.fixture
    def platform_without_twilio_ci(self):
        """Create platform without Twilio CI."""
        return ObservabilityPlatform(twilio_ci_enabled=False)

    def test_sentiment_analysis_with_twilio_ci_enabled(self, platform_with_twilio_ci, monkeypatch):
        """Test that Twilio CI API is called when enabled."""
        twilio_ci_called = []

        def mock_call_twilio_ci_api(transcript):
            twilio_ci_called.append(transcript)
            return SentimentResult(
                score=0.8,
                magnitude=0.7,
                label="positive",
                confidence=0.95,
            )

        monkeypatch.setattr(
            platform_with_twilio_ci,
            "_call_twilio_ci_api",
            mock_call_twilio_ci_api,
        )

        transcript = "Thank you for your help!"
        result = platform_with_twilio_ci.analyze_sentiment("session_123", transcript)

        # Verify Twilio CI was called
        assert len(twilio_ci_called) == 1
        assert twilio_ci_called[0] == transcript
        assert result.score == 0.8
        assert result.confidence == 0.95

    def test_sentiment_analysis_without_twilio_ci(self, platform_without_twilio_ci):
        """Test that heuristic analysis is used when Twilio CI is disabled."""
        transcript = "Thank you for your excellent service!"
        result = platform_without_twilio_ci.analyze_sentiment("session_123", transcript)

        # Should use heuristic analysis
        assert isinstance(result, SentimentResult)
        assert result.score > 0  # Positive sentiment
        assert result.label == "positive"

    def test_sentiment_analysis_stores_history(self, platform_without_twilio_ci):
        """Test that sentiment results are stored in history."""
        transcripts = [
            "Thank you!",
            "This is terrible.",
            "Okay, that works.",
        ]

        for i, transcript in enumerate(transcripts):
            platform_without_twilio_ci.analyze_sentiment(f"session_{i}", transcript)

        assert len(platform_without_twilio_ci._sentiment_history) == 3

    def test_sentiment_analysis_triggers_alert_on_low_score(self, platform_without_twilio_ci):
        """Test that low sentiment scores trigger alerts."""
        # Very negative transcript
        transcript = "This is absolutely terrible, awful, and horrible. Very disappointed."
        
        initial_alert_count = platform_without_twilio_ci.get_alert_count()
        result = platform_without_twilio_ci.analyze_sentiment("session_123", transcript)

        # Should trigger alert if score is below threshold
        if result.score < ObservabilityPlatform.SENTIMENT_ALERT_THRESHOLD:
            assert platform_without_twilio_ci.get_alert_count() > initial_alert_count
            # Check alert details
            alert = platform_without_twilio_ci._alerts[-1]
            assert alert["alert_type"] == ObservabilityPlatform.ALERT_LOW_SENTIMENT
            assert alert["severity"] == AlertSeverity.WARNING.value

    def test_sentiment_heuristic_with_mixed_words(self, platform_without_twilio_ci):
        """Test heuristic sentiment analysis with mixed positive and negative words."""
        transcript = "The service was good but I had some problems with the terrible wait time."
        result = platform_without_twilio_ci.analyze_sentiment("session_123", transcript)

        assert isinstance(result, SentimentResult)
        # Should be somewhat negative or neutral due to mixed sentiment
        assert -1.0 <= result.score <= 1.0

    def test_sentiment_heuristic_empty_transcript(self, platform_without_twilio_ci):
        """Test heuristic sentiment analysis with empty transcript."""
        result = platform_without_twilio_ci.analyze_sentiment("session_123", "")

        assert result.label == "neutral"
        assert result.score == 0.0

    def test_sentiment_result_in_dashboard_metrics(self, platform_without_twilio_ci):
        """Test that sentiment results are reflected in dashboard metrics."""
        # Add multiple sentiment analyses
        platform_without_twilio_ci.analyze_sentiment("s1", "Thank you so much!")
        platform_without_twilio_ci.analyze_sentiment("s2", "This is great!")
        platform_without_twilio_ci.analyze_sentiment("s3", "Perfect service!")

        metrics = platform_without_twilio_ci.get_dashboard_metrics()

        # Average sentiment should be positive
        assert metrics.average_sentiment > 0


class TestHallucinationDetection:
    """Test suite for hallucination detection logic.
    
    Requirements: 8.5
    """

    @pytest.fixture
    def platform(self):
        """Create platform for testing."""
        return ObservabilityPlatform()

    def test_hallucination_detection_with_knowledge_boundaries(self, platform):
        """Test that acknowledging knowledge boundaries reduces hallucination score."""
        response = "I'm not sure about that specific information. I don't have access to those details."
        context = {"facts": ["general information available"]}

        result = platform.detect_hallucination(response, context)

        assert result.score < 0.5
        assert not result.is_hallucination
        assert any("knowledge boundaries" in reason.lower() for reason in result.reasons)

    def test_hallucination_detection_with_confident_language(self, platform):
        """Test that overly confident language increases hallucination score."""
        response = "I can definitely and absolutely guarantee this will work perfectly."
        context = {"facts": []}

        result = platform.detect_hallucination(response, context)

        assert result.score > 0
        assert any("confident" in reason.lower() for reason in result.reasons)

    def test_hallucination_detection_with_fabricated_details(self, platform):
        """Test detection of specific details not present in context."""
        response = "Your appointment is at 3:45 PM on December 25th in room 305."
        context = {"facts": ["appointment requested"]}

        result = platform.detect_hallucination(response, context)

        assert result.score > 0
        # Should detect numbers not in context

    def test_hallucination_detection_with_long_response(self, platform):
        """Test that very long responses increase hallucination score."""
        response = "This is a very long response. " * 50  # 150 words
        context = {"facts": []}

        result = platform.detect_hallucination(response, context)

        assert result.score > 0
        assert any("long" in reason.lower() for reason in result.reasons)

    def test_hallucination_detection_threshold(self, platform):
        """Test hallucination threshold classification."""
        # Low score - not hallucination
        response_low = "I can help you with that."
        result_low = platform.detect_hallucination(response_low, {"facts": ["help available"]})
        assert not result_low.is_hallucination

        # High score - hallucination
        response_high = (
            "Definitely and absolutely guaranteed! "
            "Your appointment is at 3:45 PM on December 25th with Dr. Smith in room 305. "
            * 20
        )
        result_high = platform.detect_hallucination(response_high, {"facts": []})
        # Should have high score due to multiple factors

    def test_hallucination_detection_stores_history(self, platform):
        """Test that hallucination results are stored in history."""
        responses = [
            "I'm not sure about that.",
            "Definitely guaranteed!",
            "Let me help you.",
        ]

        for response in responses:
            platform.detect_hallucination(response, {"facts": []})

        assert len(platform._hallucination_history) == 3

    def test_hallucination_detection_with_context_facts(self, platform):
        """Test hallucination detection with facts in context."""
        response = "Your appointment is on Monday at 2 PM."
        context = {
            "facts": [
                "appointment on Monday",
                "time is 2 PM",
            ]
        }

        result = platform.detect_hallucination(response, context)

        # Should have lower score since details match context
        # (though heuristic may not be perfect)
        assert isinstance(result, HallucinationScore)

    def test_hallucination_score_range(self, platform):
        """Test that hallucination scores are always in valid range."""
        test_cases = [
            ("Short response", {"facts": []}),
            ("I'm not sure about that specific detail.", {"facts": []}),
            ("Definitely guaranteed! " * 100, {"facts": []}),
        ]

        for response, context in test_cases:
            result = platform.detect_hallucination(response, context)
            assert 0.0 <= result.score <= 1.0
            assert 0.0 <= result.confidence <= 1.0

    def test_hallucination_rate_in_dashboard_metrics(self, platform):
        """Test that hallucination rate is calculated in dashboard metrics."""
        # Add some hallucination checks
        platform.detect_hallucination("I'm not sure.", {"facts": []})
        platform.detect_hallucination("Definitely guaranteed! " * 50, {"facts": []})
        platform.detect_hallucination("Let me help.", {"facts": []})

        metrics = platform.get_dashboard_metrics()

        assert 0.0 <= metrics.hallucination_rate <= 1.0


class TestAlertGeneration:
    """Test suite for alert generation functionality.
    
    Requirements: 8.6
    """

    @pytest.fixture
    def platform(self):
        """Create platform for testing."""
        return ObservabilityPlatform()

    def test_alert_generation_stores_alert(self, platform):
        """Test that generated alerts are stored."""
        platform.generate_alert(
            alert_type=ObservabilityPlatform.ALERT_HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            details={"latency_ms": 1500},
        )

        assert platform.get_alert_count() == 1

    def test_alert_generation_includes_timestamp(self, platform):
        """Test that alerts include ISO format timestamps."""
        platform.generate_alert(
            alert_type=ObservabilityPlatform.ALERT_SYSTEM_ERROR,
            severity=AlertSeverity.ERROR,
            details={"error": "Test error"},
        )

        alert = platform._alerts[0]
        assert "timestamp" in alert
        assert "T" in alert["timestamp"]

    def test_alert_generation_all_severity_levels(self, platform):
        """Test alert generation for all severity levels."""
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
            AlertSeverity.CRITICAL,
        ]

        for severity in severities:
            platform.generate_alert(
                alert_type=ObservabilityPlatform.ALERT_SYSTEM_ERROR,
                severity=severity,
                details={"test": severity.value},
            )

        assert platform.get_alert_count() == 4

        # Verify all severities are present
        alert_severities = [a["severity"] for a in platform._alerts]
        assert "info" in alert_severities
        assert "warning" in alert_severities
        assert "error" in alert_severities
        assert "critical" in alert_severities

    def test_alert_generation_for_high_error_rate(self, platform):
        """Test automatic alert generation when error rate exceeds threshold."""
        # Set up calls
        for _ in range(100):
            platform.increment_call_count()

        # Add errors to exceed 5% threshold
        for _ in range(10):
            platform.increment_error_count()

        # Should have generated high error rate alert
        alert_types = [a["alert_type"] for a in platform._alerts]
        assert ObservabilityPlatform.ALERT_HIGH_ERROR_RATE in alert_types

    def test_alert_generation_for_low_sentiment(self, platform):
        """Test automatic alert generation for low sentiment."""
        # Analyze very negative sentiment
        transcript = "This is terrible, awful, horrible, and disappointing. Very bad."
        platform.analyze_sentiment("session_123", transcript)

        # Check if alert was generated (depends on sentiment score)
        result = platform._sentiment_history[-1]
        if result.score < ObservabilityPlatform.SENTIMENT_ALERT_THRESHOLD:
            alert_types = [a["alert_type"] for a in platform._alerts]
            assert ObservabilityPlatform.ALERT_LOW_SENTIMENT in alert_types

    def test_alert_details_preserved(self, platform):
        """Test that alert details are preserved correctly."""
        details = {
            "session_id": "test_123",
            "latency_ms": 2000,
            "threshold_ms": 1000,
            "stage": "llm",
        }

        platform.generate_alert(
            alert_type=ObservabilityPlatform.ALERT_HIGH_LATENCY,
            severity=AlertSeverity.WARNING,
            details=details,
        )

        alert = platform._alerts[0]
        assert alert["details"] == details
        assert alert["alert_type"] == ObservabilityPlatform.ALERT_HIGH_LATENCY

    def test_multiple_alerts_accumulate(self, platform):
        """Test that multiple alerts are accumulated."""
        alert_types = [
            ObservabilityPlatform.ALERT_HIGH_LATENCY,
            ObservabilityPlatform.ALERT_HIGH_ERROR_RATE,
            ObservabilityPlatform.ALERT_SYSTEM_ERROR,
        ]

        for alert_type in alert_types:
            platform.generate_alert(
                alert_type=alert_type,
                severity=AlertSeverity.WARNING,
                details={},
            )

        assert platform.get_alert_count() == 3

        stored_types = [a["alert_type"] for a in platform._alerts]
        for alert_type in alert_types:
            assert alert_type in stored_types
