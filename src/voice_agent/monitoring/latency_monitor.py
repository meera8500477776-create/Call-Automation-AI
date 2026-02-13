"""Latency monitoring for voice pipeline stages."""

import time
from collections import defaultdict
from dataclasses import dataclass

from voice_agent.monitoring.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyMetrics:
    """Latency metrics for a voice pipeline session."""

    stt_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0  # Time To First Token
    llm_total_ms: float = 0.0
    tts_ttfb_ms: float = 0.0  # Time To First Byte
    tts_total_ms: float = 0.0
    end_to_end_ms: float = 0.0
    tool_execution_ms: float | None = None


@dataclass
class AggregateMetrics:
    """Aggregate latency metrics across multiple sessions."""

    stage: str
    count: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


class LatencyMonitor:
    """Monitors and tracks latency metrics for voice pipeline stages.

    Tracks latency for each pipeline stage (STT, LLM TTFT, TTS TTFB) and provides
    per-session and aggregate reporting. Alerts when latency exceeds thresholds.
    """

    # Pipeline stages
    STAGE_STT = "stt"
    STAGE_LLM_TTFT = "llm_ttft"
    STAGE_LLM_TOTAL = "llm_total"
    STAGE_TTS_TTFB = "tts_ttfb"
    STAGE_TTS_TOTAL = "tts_total"
    STAGE_END_TO_END = "end_to_end"
    STAGE_TOOL_EXECUTION = "tool_execution"

    # Latency thresholds (in milliseconds)
    THRESHOLD_WARNING_MS = 1000.0
    THRESHOLD_TARGET_MS = 500.0

    def __init__(self) -> None:
        """Initialize the latency monitor."""
        # Active timers: {session_id: {stage: start_time}}
        self._timers: dict[str, dict[str, float]] = defaultdict(dict)

        # Session metrics: {session_id: LatencyMetrics}
        self._session_metrics: dict[str, LatencyMetrics] = {}

        # Historical latency data: {stage: [durations]}
        self._latency_history: dict[str, list[float]] = defaultdict(list)

    def start_timer(self, session_id: str, stage: str) -> None:
        """Start a timer for a specific pipeline stage.

        Args:
            session_id: Unique session identifier
            stage: Pipeline stage name (use STAGE_* constants)
        """
        self._timers[session_id][stage] = time.perf_counter()
        logger.debug(
            "Started timer",
            session_id=session_id,
            stage=stage,
        )

    def end_timer(self, session_id: str, stage: str) -> float:
        """End a timer and return the duration.

        Args:
            session_id: Unique session identifier
            stage: Pipeline stage name (use STAGE_* constants)

        Returns:
            Duration in milliseconds

        Raises:
            ValueError: If timer was not started for this session/stage
        """
        if session_id not in self._timers or stage not in self._timers[session_id]:
            raise ValueError(
                f"Timer not started for session {session_id}, stage {stage}"
            )

        start_time = self._timers[session_id][stage]
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # Clean up timer
        del self._timers[session_id][stage]

        logger.debug(
            "Ended timer",
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
        )

        # Record the latency
        self.record_latency(session_id, stage, duration_ms)

        return duration_ms

    def record_latency(self, session_id: str, stage: str, duration_ms: float) -> None:
        """Record latency for a specific pipeline stage.

        Args:
            session_id: Unique session identifier
            stage: Pipeline stage name (use STAGE_* constants)
            duration_ms: Duration in milliseconds
        """
        # Initialize session metrics if needed
        if session_id not in self._session_metrics:
            self._session_metrics[session_id] = LatencyMetrics()

        # Update session metrics
        metrics = self._session_metrics[session_id]
        if stage == self.STAGE_STT:
            metrics.stt_latency_ms = duration_ms
        elif stage == self.STAGE_LLM_TTFT:
            metrics.llm_ttft_ms = duration_ms
        elif stage == self.STAGE_LLM_TOTAL:
            metrics.llm_total_ms = duration_ms
        elif stage == self.STAGE_TTS_TTFB:
            metrics.tts_ttfb_ms = duration_ms
        elif stage == self.STAGE_TTS_TOTAL:
            metrics.tts_total_ms = duration_ms
        elif stage == self.STAGE_END_TO_END:
            metrics.end_to_end_ms = duration_ms
        elif stage == self.STAGE_TOOL_EXECUTION:
            metrics.tool_execution_ms = duration_ms

        # Add to historical data
        self._latency_history[stage].append(duration_ms)

        # Check thresholds and alert if needed
        self._check_threshold(session_id, stage, duration_ms)

        logger.info(
            "Recorded latency",
            session_id=session_id,
            stage=stage,
            duration_ms=duration_ms,
        )

    def get_session_metrics(self, session_id: str) -> LatencyMetrics | None:
        """Get latency metrics for a specific session.

        Args:
            session_id: Unique session identifier

        Returns:
            LatencyMetrics for the session, or None if session not found
        """
        return self._session_metrics.get(session_id)

    def get_aggregate_metrics(
        self, stage: str | None = None
    ) -> dict[str, AggregateMetrics]:
        """Get aggregate latency metrics across all sessions.

        Args:
            stage: Optional specific stage to get metrics for.
                   If None, returns metrics for all stages.

        Returns:
            Dictionary mapping stage names to AggregateMetrics
        """
        result = {}

        stages_to_process = (
            [stage] if stage else list(self._latency_history.keys())
        )

        for stage_name in stages_to_process:
            durations = self._latency_history.get(stage_name, [])
            if not durations:
                continue

            sorted_durations = sorted(durations)
            count = len(sorted_durations)

            # Calculate statistics
            mean_ms = sum(sorted_durations) / count
            median_ms = self._percentile(sorted_durations, 50)
            p95_ms = self._percentile(sorted_durations, 95)
            p99_ms = self._percentile(sorted_durations, 99)
            min_ms = sorted_durations[0]
            max_ms = sorted_durations[-1]

            result[stage_name] = AggregateMetrics(
                stage=stage_name,
                count=count,
                mean_ms=mean_ms,
                median_ms=median_ms,
                p95_ms=p95_ms,
                p99_ms=p99_ms,
                min_ms=min_ms,
                max_ms=max_ms,
            )

        return result

    def clear_session(self, session_id: str) -> None:
        """Clear metrics and timers for a specific session.

        Args:
            session_id: Unique session identifier
        """
        if session_id in self._timers:
            del self._timers[session_id]
        if session_id in self._session_metrics:
            del self._session_metrics[session_id]

        logger.debug("Cleared session metrics", session_id=session_id)

    def _check_threshold(
        self, session_id: str, stage: str, duration_ms: float
    ) -> None:
        """Check if latency exceeds thresholds and log alerts.

        Args:
            session_id: Unique session identifier
            stage: Pipeline stage name
            duration_ms: Duration in milliseconds
        """
        if duration_ms > self.THRESHOLD_WARNING_MS:
            logger.warning(
                "Latency threshold exceeded",
                session_id=session_id,
                stage=stage,
                duration_ms=duration_ms,
                threshold_ms=self.THRESHOLD_WARNING_MS,
                severity="warning",
            )
        elif duration_ms > self.THRESHOLD_TARGET_MS:
            logger.info(
                "Latency above target",
                session_id=session_id,
                stage=stage,
                duration_ms=duration_ms,
                threshold_ms=self.THRESHOLD_TARGET_MS,
                severity="info",
            )

    @staticmethod
    def _percentile(sorted_values: list[float], percentile: int) -> float:
        """Calculate percentile from sorted values.

        Args:
            sorted_values: List of values in sorted order
            percentile: Percentile to calculate (0-100)

        Returns:
            Percentile value
        """
        if not sorted_values:
            return 0.0

        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1

        if c >= len(sorted_values):
            return sorted_values[-1]

        # Linear interpolation
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])
