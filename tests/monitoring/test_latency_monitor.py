"""Unit tests for LatencyMonitor."""

import time
from unittest.mock import patch

import pytest

from voice_agent.monitoring.latency_monitor import (
    AggregateMetrics,
    LatencyMetrics,
    LatencyMonitor,
)


class TestLatencyMonitor:
    """Test suite for LatencyMonitor class."""

    def test_start_timer(self):
        """Test starting a timer for a pipeline stage."""
        monitor = LatencyMonitor()
        session_id = "test-session-1"
        stage = LatencyMonitor.STAGE_STT

        monitor.start_timer(session_id, stage)

        # Verify timer was started
        assert session_id in monitor._timers
        assert stage in monitor._timers[session_id]
        assert isinstance(monitor._timers[session_id][stage], float)

    def test_end_timer_returns_duration(self):
        """Test ending a timer returns the correct duration."""
        monitor = LatencyMonitor()
        session_id = "test-session-2"
        stage = LatencyMonitor.STAGE_LLM_TTFT

        monitor.start_timer(session_id, stage)
        time.sleep(0.01)  # Sleep for 10ms
        duration_ms = monitor.end_timer(session_id, stage)

        # Duration should be at least 10ms
        assert duration_ms >= 10.0
        # Timer should be cleaned up
        assert stage not in monitor._timers.get(session_id, {})

    def test_end_timer_without_start_raises_error(self):
        """Test ending a timer that wasn't started raises ValueError."""
        monitor = LatencyMonitor()
        session_id = "test-session-3"
        stage = LatencyMonitor.STAGE_TTS_TTFB

        with pytest.raises(ValueError, match="Timer not started"):
            monitor.end_timer(session_id, stage)

    def test_record_latency_creates_session_metrics(self):
        """Test recording latency creates session metrics."""
        monitor = LatencyMonitor()
        session_id = "test-session-4"
        stage = LatencyMonitor.STAGE_STT
        duration_ms = 150.5

        monitor.record_latency(session_id, stage, duration_ms)

        # Verify session metrics were created
        metrics = monitor.get_session_metrics(session_id)
        assert metrics is not None
        assert metrics.stt_latency_ms == duration_ms

    def test_record_latency_updates_all_stages(self):
        """Test recording latency for all pipeline stages."""
        monitor = LatencyMonitor()
        session_id = "test-session-5"

        # Record latency for each stage
        monitor.record_latency(session_id, LatencyMonitor.STAGE_STT, 100.0)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_LLM_TTFT, 200.0)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_LLM_TOTAL, 300.0)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_TTS_TTFB, 150.0)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_TTS_TOTAL, 250.0)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_END_TO_END, 800.0)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_TOOL_EXECUTION, 50.0)

        # Verify all metrics were recorded
        metrics = monitor.get_session_metrics(session_id)
        assert metrics is not None
        assert metrics.stt_latency_ms == 100.0
        assert metrics.llm_ttft_ms == 200.0
        assert metrics.llm_total_ms == 300.0
        assert metrics.tts_ttfb_ms == 150.0
        assert metrics.tts_total_ms == 250.0
        assert metrics.end_to_end_ms == 800.0
        assert metrics.tool_execution_ms == 50.0

    def test_get_session_metrics_returns_none_for_unknown_session(self):
        """Test getting metrics for unknown session returns None."""
        monitor = LatencyMonitor()
        metrics = monitor.get_session_metrics("unknown-session")
        assert metrics is None

    def test_get_aggregate_metrics_single_stage(self):
        """Test getting aggregate metrics for a single stage."""
        monitor = LatencyMonitor()

        # Record multiple latencies for STT stage
        for i in range(10):
            monitor.record_latency(f"session-{i}", LatencyMonitor.STAGE_STT, 100.0 + i * 10)

        # Get aggregate metrics for STT
        agg_metrics = monitor.get_aggregate_metrics(LatencyMonitor.STAGE_STT)

        assert LatencyMonitor.STAGE_STT in agg_metrics
        metrics = agg_metrics[LatencyMonitor.STAGE_STT]
        assert metrics.count == 10
        assert metrics.mean_ms == 145.0  # (100 + 190) / 2
        assert metrics.min_ms == 100.0
        assert metrics.max_ms == 190.0

    def test_get_aggregate_metrics_all_stages(self):
        """Test getting aggregate metrics for all stages."""
        monitor = LatencyMonitor()

        # Record latencies for multiple stages
        monitor.record_latency("session-1", LatencyMonitor.STAGE_STT, 100.0)
        monitor.record_latency("session-1", LatencyMonitor.STAGE_LLM_TTFT, 200.0)
        monitor.record_latency("session-2", LatencyMonitor.STAGE_STT, 150.0)
        monitor.record_latency("session-2", LatencyMonitor.STAGE_LLM_TTFT, 250.0)

        # Get all aggregate metrics
        agg_metrics = monitor.get_aggregate_metrics()

        assert len(agg_metrics) == 2
        assert LatencyMonitor.STAGE_STT in agg_metrics
        assert LatencyMonitor.STAGE_LLM_TTFT in agg_metrics

        stt_metrics = agg_metrics[LatencyMonitor.STAGE_STT]
        assert stt_metrics.count == 2
        assert stt_metrics.mean_ms == 125.0

        llm_metrics = agg_metrics[LatencyMonitor.STAGE_LLM_TTFT]
        assert llm_metrics.count == 2
        assert llm_metrics.mean_ms == 225.0

    def test_get_aggregate_metrics_empty_returns_empty_dict(self):
        """Test getting aggregate metrics with no data returns empty dict."""
        monitor = LatencyMonitor()
        agg_metrics = monitor.get_aggregate_metrics()
        assert agg_metrics == {}

    def test_aggregate_metrics_percentiles(self):
        """Test aggregate metrics calculates percentiles correctly."""
        monitor = LatencyMonitor()

        # Record 100 latencies with known distribution
        for i in range(100):
            monitor.record_latency(f"session-{i}", LatencyMonitor.STAGE_STT, float(i))

        agg_metrics = monitor.get_aggregate_metrics(LatencyMonitor.STAGE_STT)
        metrics = agg_metrics[LatencyMonitor.STAGE_STT]

        assert metrics.count == 100
        assert metrics.median_ms == pytest.approx(49.5, rel=0.1)
        assert metrics.p95_ms == pytest.approx(94.05, rel=0.1)
        assert metrics.p99_ms == pytest.approx(98.01, rel=0.1)

    def test_clear_session(self):
        """Test clearing session metrics and timers."""
        monitor = LatencyMonitor()
        session_id = "test-session-6"

        # Start timer and record latency
        monitor.start_timer(session_id, LatencyMonitor.STAGE_STT)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_LLM_TTFT, 200.0)

        # Verify data exists
        assert session_id in monitor._timers
        assert monitor.get_session_metrics(session_id) is not None

        # Clear session
        monitor.clear_session(session_id)

        # Verify data was cleared
        assert session_id not in monitor._timers
        assert monitor.get_session_metrics(session_id) is None

    @patch("voice_agent.monitoring.latency_monitor.logger")
    def test_threshold_warning_logged(self, mock_logger):
        """Test that exceeding warning threshold logs a warning."""
        monitor = LatencyMonitor()
        session_id = "test-session-7"
        stage = LatencyMonitor.STAGE_END_TO_END
        duration_ms = 1500.0  # Exceeds 1000ms warning threshold

        monitor.record_latency(session_id, stage, duration_ms)

        # Verify warning was logged
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "Latency threshold exceeded" in call_args[0]
        assert call_args[1]["duration_ms"] == duration_ms
        assert call_args[1]["severity"] == "warning"

    @patch("voice_agent.monitoring.latency_monitor.logger")
    def test_threshold_target_logged(self, mock_logger):
        """Test that exceeding target threshold logs an info message."""
        monitor = LatencyMonitor()
        session_id = "test-session-8"
        stage = LatencyMonitor.STAGE_LLM_TTFT
        duration_ms = 750.0  # Exceeds 500ms target but not 1000ms warning

        monitor.record_latency(session_id, stage, duration_ms)

        # Verify info was logged (not warning)
        mock_logger.warning.assert_not_called()
        # Check that info was called with threshold message
        info_calls = [call for call in mock_logger.info.call_args_list
                      if "Latency above target" in call[0]]
        assert len(info_calls) == 1

    @patch("voice_agent.monitoring.latency_monitor.logger")
    def test_no_threshold_alert_for_good_latency(self, mock_logger):
        """Test that good latency doesn't trigger threshold alerts."""
        monitor = LatencyMonitor()
        session_id = "test-session-9"
        stage = LatencyMonitor.STAGE_STT
        duration_ms = 300.0  # Below 500ms target

        monitor.record_latency(session_id, stage, duration_ms)

        # Verify no warning or threshold info was logged
        mock_logger.warning.assert_not_called()
        info_calls = [call for call in mock_logger.info.call_args_list
                      if "Latency above target" in call[0]]
        assert len(info_calls) == 0

    def test_end_timer_records_latency(self):
        """Test that end_timer automatically records latency."""
        monitor = LatencyMonitor()
        session_id = "test-session-10"
        stage = LatencyMonitor.STAGE_TTS_TOTAL

        monitor.start_timer(session_id, stage)
        time.sleep(0.01)
        duration_ms = monitor.end_timer(session_id, stage)

        # Verify latency was recorded
        metrics = monitor.get_session_metrics(session_id)
        assert metrics is not None
        assert metrics.tts_total_ms == duration_ms

    def test_multiple_sessions_independent(self):
        """Test that multiple sessions maintain independent metrics."""
        monitor = LatencyMonitor()

        # Record different latencies for different sessions
        monitor.record_latency("session-a", LatencyMonitor.STAGE_STT, 100.0)
        monitor.record_latency("session-b", LatencyMonitor.STAGE_STT, 200.0)
        monitor.record_latency("session-c", LatencyMonitor.STAGE_STT, 300.0)

        # Verify each session has correct metrics
        metrics_a = monitor.get_session_metrics("session-a")
        metrics_b = monitor.get_session_metrics("session-b")
        metrics_c = monitor.get_session_metrics("session-c")

        assert metrics_a.stt_latency_ms == 100.0
        assert metrics_b.stt_latency_ms == 200.0
        assert metrics_c.stt_latency_ms == 300.0

    def test_latency_metrics_dataclass_defaults(self):
        """Test LatencyMetrics dataclass has correct defaults."""
        metrics = LatencyMetrics()

        assert metrics.stt_latency_ms == 0.0
        assert metrics.llm_ttft_ms == 0.0
        assert metrics.llm_total_ms == 0.0
        assert metrics.tts_ttfb_ms == 0.0
        assert metrics.tts_total_ms == 0.0
        assert metrics.end_to_end_ms == 0.0
        assert metrics.tool_execution_ms is None

    def test_aggregate_metrics_dataclass(self):
        """Test AggregateMetrics dataclass structure."""
        metrics = AggregateMetrics(
            stage="test_stage",
            count=100,
            mean_ms=150.0,
            median_ms=145.0,
            p95_ms=200.0,
            p99_ms=250.0,
            min_ms=50.0,
            max_ms=300.0,
        )

        assert metrics.stage == "test_stage"
        assert metrics.count == 100
        assert metrics.mean_ms == 150.0
        assert metrics.median_ms == 145.0
        assert metrics.p95_ms == 200.0
        assert metrics.p99_ms == 250.0
        assert metrics.min_ms == 50.0
        assert metrics.max_ms == 300.0
