"""Property-based tests for LatencyMonitor using Hypothesis.

**Validates: Requirements 2.7**
"""

import time
from hypothesis import given, strategies as st, settings
import pytest

from voice_agent.monitoring.latency_monitor import LatencyMonitor


class TestLatencyMonitorProperties:
    """Property-based tests for LatencyMonitor."""

    @given(
        session_id=st.text(min_size=1, max_size=100),
        stage=st.sampled_from([
            LatencyMonitor.STAGE_STT,
            LatencyMonitor.STAGE_LLM_TTFT,
            LatencyMonitor.STAGE_LLM_TOTAL,
            LatencyMonitor.STAGE_TTS_TTFB,
            LatencyMonitor.STAGE_TTS_TOTAL,
            LatencyMonitor.STAGE_END_TO_END,
            LatencyMonitor.STAGE_TOOL_EXECUTION,
        ]),
        sleep_ms=st.floats(min_value=0.001, max_value=0.1),  # 1-100ms sleep
    )
    @settings(max_examples=100, deadline=None)
    def test_property_timer_consistency_non_negative_duration(
        self, session_id: str, stage: str, sleep_ms: float
    ):
        """Property 7: Timer consistency - For any timer, end_timer() should return a non-negative duration.

        **Validates: Requirements 2.7**

        This property ensures that the latency monitor always returns valid,
        non-negative durations for any timer, regardless of the session ID,
        stage, or duration.
        """
        monitor = LatencyMonitor()

        # Start timer
        monitor.start_timer(session_id, stage)

        # Wait for a small amount of time
        time.sleep(sleep_ms)

        # End timer and get duration
        duration_ms = monitor.end_timer(session_id, stage)

        # Property: Duration must be non-negative
        assert duration_ms >= 0.0, (
            f"Timer duration must be non-negative, got {duration_ms}ms "
            f"for session={session_id}, stage={stage}"
        )

        # Additional check: Duration should be at least close to the sleep time
        # (accounting for system overhead and timing precision)
        expected_min_ms = sleep_ms * 1000 * 0.5  # Allow 50% margin for overhead
        assert duration_ms >= expected_min_ms, (
            f"Timer duration {duration_ms}ms is suspiciously low "
            f"for sleep time {sleep_ms * 1000}ms"
        )

    @given(
        session_id=st.text(min_size=1, max_size=100),
        stage=st.sampled_from([
            LatencyMonitor.STAGE_STT,
            LatencyMonitor.STAGE_LLM_TTFT,
            LatencyMonitor.STAGE_LLM_TOTAL,
            LatencyMonitor.STAGE_TTS_TTFB,
            LatencyMonitor.STAGE_TTS_TOTAL,
            LatencyMonitor.STAGE_END_TO_END,
            LatencyMonitor.STAGE_TOOL_EXECUTION,
        ]),
    )
    @settings(max_examples=100)
    def test_property_timer_consistency_immediate_end(
        self, session_id: str, stage: str
    ):
        """Property 7 (variant): Timer consistency - Even with immediate end_timer(), duration should be non-negative.

        **Validates: Requirements 2.7**

        This tests the edge case where end_timer() is called immediately after
        start_timer() with no sleep in between.
        """
        monitor = LatencyMonitor()

        # Start and immediately end timer
        monitor.start_timer(session_id, stage)
        duration_ms = monitor.end_timer(session_id, stage)

        # Property: Duration must be non-negative even for immediate calls
        assert duration_ms >= 0.0, (
            f"Timer duration must be non-negative even for immediate end, "
            f"got {duration_ms}ms for session={session_id}, stage={stage}"
        )

    @given(
        session_id=st.text(min_size=1, max_size=100),
        stage=st.sampled_from([
            LatencyMonitor.STAGE_STT,
            LatencyMonitor.STAGE_LLM_TTFT,
            LatencyMonitor.STAGE_LLM_TOTAL,
            LatencyMonitor.STAGE_TTS_TTFB,
            LatencyMonitor.STAGE_TTS_TOTAL,
            LatencyMonitor.STAGE_END_TO_END,
            LatencyMonitor.STAGE_TOOL_EXECUTION,
        ]),
        duration_ms=st.floats(min_value=0.0, max_value=10000.0),
    )
    @settings(max_examples=100)
    def test_property_recorded_latency_non_negative(
        self, session_id: str, stage: str, duration_ms: float
    ):
        """Property 7 (variant): Timer consistency - Recorded latencies should always be non-negative.

        **Validates: Requirements 2.7**

        This tests that the record_latency() method properly handles
        non-negative durations and stores them correctly.
        """
        monitor = LatencyMonitor()

        # Record latency
        monitor.record_latency(session_id, stage, duration_ms)

        # Get session metrics
        metrics = monitor.get_session_metrics(session_id)

        # Property: Recorded latency must be non-negative
        assert metrics is not None, f"Session metrics should exist for {session_id}"

        # Check the appropriate field based on stage
        if stage == LatencyMonitor.STAGE_STT:
            recorded = metrics.stt_latency_ms
        elif stage == LatencyMonitor.STAGE_LLM_TTFT:
            recorded = metrics.llm_ttft_ms
        elif stage == LatencyMonitor.STAGE_LLM_TOTAL:
            recorded = metrics.llm_total_ms
        elif stage == LatencyMonitor.STAGE_TTS_TTFB:
            recorded = metrics.tts_ttfb_ms
        elif stage == LatencyMonitor.STAGE_TTS_TOTAL:
            recorded = metrics.tts_total_ms
        elif stage == LatencyMonitor.STAGE_END_TO_END:
            recorded = metrics.end_to_end_ms
        elif stage == LatencyMonitor.STAGE_TOOL_EXECUTION:
            recorded = metrics.tool_execution_ms
        else:
            pytest.fail(f"Unknown stage: {stage}")

        assert recorded >= 0.0, (
            f"Recorded latency must be non-negative, got {recorded}ms "
            f"for session={session_id}, stage={stage}"
        )

        # Property: Recorded value should match input
        assert recorded == duration_ms, (
            f"Recorded latency {recorded}ms should match input {duration_ms}ms"
        )

    @given(
        session_id=st.text(min_size=1, max_size=100),
        stages=st.lists(
            st.sampled_from([
                LatencyMonitor.STAGE_STT,
                LatencyMonitor.STAGE_LLM_TTFT,
                LatencyMonitor.STAGE_LLM_TOTAL,
                LatencyMonitor.STAGE_TTS_TTFB,
                LatencyMonitor.STAGE_TTS_TOTAL,
                LatencyMonitor.STAGE_END_TO_END,
                LatencyMonitor.STAGE_TOOL_EXECUTION,
            ]),
            min_size=1,
            max_size=7,
            unique=True,
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_multiple_timers_all_non_negative(
        self, session_id: str, stages: list
    ):
        """Property 7 (variant): Timer consistency - Multiple timers should all return non-negative durations.

        **Validates: Requirements 2.7**

        This tests that when multiple timers are used for different stages,
        all of them return non-negative durations.
        """
        monitor = LatencyMonitor()

        # Start all timers
        for stage in stages:
            monitor.start_timer(session_id, stage)

        # Small sleep to ensure some time passes
        time.sleep(0.001)

        # End all timers and collect durations
        durations = {}
        for stage in stages:
            durations[stage] = monitor.end_timer(session_id, stage)

        # Property: All durations must be non-negative
        for stage, duration_ms in durations.items():
            assert duration_ms >= 0.0, (
                f"Timer duration must be non-negative, got {duration_ms}ms "
                f"for session={session_id}, stage={stage}"
            )

    @given(
        session_ids=st.lists(
            st.text(min_size=1, max_size=50),
            min_size=1,
            max_size=20,
            unique=True,
        ),
        stage=st.sampled_from([
            LatencyMonitor.STAGE_STT,
            LatencyMonitor.STAGE_LLM_TTFT,
            LatencyMonitor.STAGE_LLM_TOTAL,
            LatencyMonitor.STAGE_TTS_TTFB,
            LatencyMonitor.STAGE_TTS_TOTAL,
            LatencyMonitor.STAGE_END_TO_END,
            LatencyMonitor.STAGE_TOOL_EXECUTION,
        ]),
    )
    @settings(max_examples=50, deadline=None)
    def test_property_multiple_sessions_all_non_negative(
        self, session_ids: list, stage: str
    ):
        """Property 7 (variant): Timer consistency - Multiple sessions should all have non-negative durations.

        **Validates: Requirements 2.7**

        This tests that when multiple sessions are tracked simultaneously,
        all of them return non-negative durations.
        """
        monitor = LatencyMonitor()

        # Start timers for all sessions
        for session_id in session_ids:
            monitor.start_timer(session_id, stage)

        # Small sleep to ensure some time passes
        time.sleep(0.001)

        # End all timers and collect durations
        durations = {}
        for session_id in session_ids:
            durations[session_id] = monitor.end_timer(session_id, stage)

        # Property: All durations must be non-negative
        for session_id, duration_ms in durations.items():
            assert duration_ms >= 0.0, (
                f"Timer duration must be non-negative, got {duration_ms}ms "
                f"for session={session_id}, stage={stage}"
            )
