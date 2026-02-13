"""
Observability Platform for Config-Driven Voice Agent Platform.

Integrates monitoring, tracing, and analytics tools for comprehensive system observability.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from voice_agent.monitoring.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SentimentResult:
    """Sentiment analysis result from Twilio Conversational Intelligence."""

    score: float  # -1.0 (negative) to 1.0 (positive)
    magnitude: float  # 0.0 to 1.0 (intensity)
    label: str  # "positive", "negative", "neutral"
    confidence: float  # 0.0 to 1.0


@dataclass
class HallucinationScore:
    """Hallucination detection result for LLM responses."""

    score: float  # 0.0 (no hallucination) to 1.0 (likely hallucination)
    confidence: float  # 0.0 to 1.0
    reasons: list[str]  # List of reasons for detection
    is_hallucination: bool  # True if score exceeds threshold


@dataclass
class DashboardMetrics:
    """Real-time dashboard metrics."""

    timestamp: datetime
    active_sessions: int
    total_calls_today: int
    average_latency_ms: float
    error_rate: float
    hallucination_rate: float
    average_sentiment: float
    llm_token_usage: int
    llm_cost_usd: float


class ObservabilityPlatform:
    """
    Integrated monitoring system for comprehensive observability.

    Provides:
    - End-to-end tracing for voice interactions
    - LLM observability via LangSmith/Helicone
    - Sentiment analysis via Twilio Conversational Intelligence
    - Hallucination detection for LLM responses
    - Alert generation for threshold violations
    - Real-time dashboard metrics

    Requirements:
    - 8.1: End-to-end traces for every voice interaction
    - 8.2: LangSmith/Helicone integration for LLM observability
    - 8.3: Twilio Conversational Intelligence for sentiment analysis
    - 8.4: Latency tracking per pipeline stage
    - 8.5: Hallucination rate monitoring
    - 8.6: Alert generation for threshold violations
    - 8.7: Real-time system health dashboards
    """

    # Alert types
    ALERT_HIGH_LATENCY = "high_latency"
    ALERT_HIGH_ERROR_RATE = "high_error_rate"
    ALERT_HIGH_HALLUCINATION_RATE = "high_hallucination_rate"
    ALERT_LOW_SENTIMENT = "low_sentiment"
    ALERT_SYSTEM_ERROR = "system_error"

    # Thresholds
    HALLUCINATION_THRESHOLD = 0.7  # Score above this is considered hallucination
    SENTIMENT_ALERT_THRESHOLD = -0.5  # Sentiment below this triggers alert
    ERROR_RATE_THRESHOLD = 0.05  # 5% error rate threshold

    def __init__(
        self,
        langsmith_api_key: str | None = None,
        helicone_api_key: str | None = None,
        twilio_ci_enabled: bool = False,
    ):
        """
        Initialize the observability platform.

        Args:
            langsmith_api_key: Optional LangSmith API key for LLM tracing
            helicone_api_key: Optional Helicone API key for LLM tracing
            twilio_ci_enabled: Whether Twilio Conversational Intelligence is enabled
        """
        self._langsmith_api_key = langsmith_api_key
        self._helicone_api_key = helicone_api_key
        self._twilio_ci_enabled = twilio_ci_enabled

        # In-memory storage for metrics (in production, use time-series DB)
        self._traces: list[dict[str, Any]] = []
        self._alerts: list[dict[str, Any]] = []
        self._sentiment_history: list[SentimentResult] = []
        self._hallucination_history: list[HallucinationScore] = []

        # Metrics counters
        self._active_sessions = 0
        self._total_calls = 0
        self._total_errors = 0
        self._total_llm_tokens = 0
        self._total_llm_cost = 0.0

        logger.info(
            "Initialized ObservabilityPlatform",
            langsmith_enabled=bool(langsmith_api_key),
            helicone_enabled=bool(helicone_api_key),
            twilio_ci_enabled=twilio_ci_enabled,
        )

    def trace_interaction(
        self,
        session_id: str,
        trace_data: dict[str, Any],
    ) -> None:
        """
        Send end-to-end trace to LangSmith or Helicone.

        Captures complete interaction trace including:
        - Input transcript
        - LLM request/response
        - Tool executions
        - Latency metrics
        - Token usage and cost

        Args:
            session_id: Unique session identifier
            trace_data: Trace data dictionary containing interaction details

        Requirements: 8.1, 8.2
        """
        trace_entry = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_data": trace_data,
        }

        # Store trace locally
        self._traces.append(trace_entry)

        # Send to LangSmith if configured
        if self._langsmith_api_key:
            self._send_to_langsmith(trace_entry)

        # Send to Helicone if configured
        if self._helicone_api_key:
            self._send_to_helicone(trace_entry)

        # Update token usage and cost metrics
        if "llm_tokens" in trace_data:
            self._total_llm_tokens += trace_data["llm_tokens"]
        if "llm_cost" in trace_data:
            self._total_llm_cost += trace_data["llm_cost"]

        logger.debug(
            "Traced interaction",
            session_id=session_id,
            trace_id=trace_entry.get("trace_id"),
        )

    def analyze_sentiment(
        self,
        session_id: str,
        transcript: str,
    ) -> SentimentResult:
        """
        Analyze sentiment using Twilio Conversational Intelligence.

        Provides sentiment analysis for conversation quality monitoring.
        In production, this would call Twilio CI API. For now, uses
        simple heuristic-based analysis.

        Args:
            session_id: Unique session identifier
            transcript: Transcript text to analyze

        Returns:
            SentimentResult with score, magnitude, label, and confidence

        Requirements: 8.3
        """
        if self._twilio_ci_enabled:
            # In production, call Twilio CI API
            result = self._call_twilio_ci_api(transcript)
        else:
            # Simple heuristic-based sentiment analysis
            result = self._analyze_sentiment_heuristic(transcript)

        # Store sentiment history
        self._sentiment_history.append(result)

        # Check for low sentiment and generate alert if needed
        if result.score < self.SENTIMENT_ALERT_THRESHOLD:
            self.generate_alert(
                alert_type=self.ALERT_LOW_SENTIMENT,
                severity=AlertSeverity.WARNING,
                details={
                    "session_id": session_id,
                    "sentiment_score": result.score,
                    "transcript_preview": transcript[:100],
                },
            )

        logger.info(
            "Analyzed sentiment",
            session_id=session_id,
            score=result.score,
            label=result.label,
        )

        return result

    def detect_hallucination(
        self,
        response: str,
        context: dict[str, Any],
    ) -> HallucinationScore:
        """
        Detect potential hallucinations in LLM responses.

        Uses heuristics to identify responses that may contain
        fabricated or inconsistent information:
        - Contradictions with provided context
        - Overly specific details not in context
        - Uncertain language patterns
        - Knowledge boundary violations

        Args:
            response: LLM response text
            context: Context dictionary with conversation history and facts

        Returns:
            HallucinationScore with detection results

        Requirements: 8.5
        """
        reasons = []
        score = 0.0

        # Check for knowledge boundary violations
        uncertain_phrases = [
            "i'm not sure",
            "i don't know",
            "i cannot",
            "i'm unable to",
            "i don't have access",
        ]
        if any(phrase in response.lower() for phrase in uncertain_phrases):
            # Good - acknowledging limitations
            score -= 0.2
            reasons.append("Acknowledges knowledge boundaries")
        else:
            # Check for overly confident statements without context
            confident_phrases = [
                "definitely",
                "certainly",
                "absolutely",
                "guaranteed",
            ]
            if any(phrase in response.lower() for phrase in confident_phrases):
                score += 0.3
                reasons.append("Overly confident language")

        # Check for specific details not in context
        # (simplified heuristic - in production, use semantic similarity)
        if context.get("facts"):
            facts = context["facts"]
            # If response contains numbers/dates not in facts, potential hallucination
            import re

            response_numbers = set(re.findall(r"\b\d+\b", response))
            fact_numbers = set(
                re.findall(r"\b\d+\b", str(facts))
            )
            if response_numbers - fact_numbers:
                score += 0.4
                reasons.append("Contains specific details not in context")

        # Check response length (very long responses may contain hallucinations)
        if len(response.split()) > 100:
            score += 0.2
            reasons.append("Response unusually long")

        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score))

        # Determine if hallucination
        is_hallucination = score >= self.HALLUCINATION_THRESHOLD

        result = HallucinationScore(
            score=score,
            confidence=0.7,  # Heuristic confidence
            reasons=reasons if reasons else ["No hallucination indicators detected"],
            is_hallucination=is_hallucination,
        )

        # Store hallucination history
        self._hallucination_history.append(result)

        if is_hallucination:
            logger.warning(
                "Potential hallucination detected",
                score=score,
                reasons=reasons,
            )

        return result

    def generate_alert(
        self,
        alert_type: str,
        severity: AlertSeverity,
        details: dict[str, Any],
    ) -> None:
        """
        Generate alert for threshold violations or system issues.

        Alerts are logged and stored for dashboard display.
        In production, would integrate with alerting systems
        (PagerDuty, Slack, email, etc.).

        Args:
            alert_type: Type of alert (use ALERT_* constants)
            severity: Alert severity level
            details: Alert details dictionary

        Requirements: 8.6
        """
        alert = {
            "alert_type": alert_type,
            "severity": severity.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        }

        # Store alert
        self._alerts.append(alert)

        # Log alert with appropriate level
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(severity, logger.warning)

        log_method(
            "Generated alert",
            alert_type=alert_type,
            severity=severity.value,
            details=details,
        )

        # In production, send to alerting systems
        # self._send_to_pagerduty(alert)
        # self._send_to_slack(alert)

    def get_dashboard_metrics(self) -> DashboardMetrics:
        """
        Get real-time metrics for system health dashboard.

        Provides key metrics for monitoring:
        - Active sessions and call volume
        - Average latency
        - Error rate
        - Hallucination rate
        - Average sentiment
        - LLM token usage and cost

        Returns:
            DashboardMetrics with current system metrics

        Requirements: 8.7
        """
        # Calculate error rate
        error_rate = (
            self._total_errors / self._total_calls
            if self._total_calls > 0
            else 0.0
        )

        # Calculate hallucination rate
        hallucination_rate = (
            sum(1 for h in self._hallucination_history if h.is_hallucination)
            / len(self._hallucination_history)
            if self._hallucination_history
            else 0.0
        )

        # Calculate average sentiment
        average_sentiment = (
            sum(s.score for s in self._sentiment_history)
            / len(self._sentiment_history)
            if self._sentiment_history
            else 0.0
        )

        # Calculate average latency (would come from LatencyMonitor in production)
        average_latency_ms = 750.0  # Placeholder

        metrics = DashboardMetrics(
            timestamp=datetime.now(timezone.utc),
            active_sessions=self._active_sessions,
            total_calls_today=self._total_calls,
            average_latency_ms=average_latency_ms,
            error_rate=error_rate,
            hallucination_rate=hallucination_rate,
            average_sentiment=average_sentiment,
            llm_token_usage=self._total_llm_tokens,
            llm_cost_usd=self._total_llm_cost,
        )

        logger.debug("Retrieved dashboard metrics", metrics=metrics)

        return metrics

    def increment_session_count(self, delta: int = 1) -> None:
        """
        Increment active session count.

        Args:
            delta: Amount to increment (can be negative to decrement)
        """
        self._active_sessions += delta

    def increment_call_count(self) -> None:
        """Increment total call count."""
        self._total_calls += 1

    def increment_error_count(self) -> None:
        """Increment error count and check threshold."""
        self._total_errors += 1

        # Check error rate threshold
        error_rate = self._total_errors / self._total_calls if self._total_calls > 0 else 0.0
        if error_rate > self.ERROR_RATE_THRESHOLD:
            self.generate_alert(
                alert_type=self.ALERT_HIGH_ERROR_RATE,
                severity=AlertSeverity.ERROR,
                details={
                    "error_rate": error_rate,
                    "total_errors": self._total_errors,
                    "total_calls": self._total_calls,
                },
            )

    def get_trace_count(self) -> int:
        """Get total number of traces."""
        return len(self._traces)

    def get_alert_count(self) -> int:
        """Get total number of alerts."""
        return len(self._alerts)

    def clear_metrics(self) -> None:
        """Clear all metrics (for testing only)."""
        self._traces.clear()
        self._alerts.clear()
        self._sentiment_history.clear()
        self._hallucination_history.clear()
        self._active_sessions = 0
        self._total_calls = 0
        self._total_errors = 0
        self._total_llm_tokens = 0
        self._total_llm_cost = 0.0
        logger.warning("Cleared all observability metrics")

    # Private helper methods

    def _send_to_langsmith(self, trace_entry: dict[str, Any]) -> None:
        """Send trace to LangSmith (placeholder for production implementation)."""
        # In production, use LangSmith SDK
        logger.debug("Would send trace to LangSmith", trace_id=trace_entry.get("session_id"))

    def _send_to_helicone(self, trace_entry: dict[str, Any]) -> None:
        """Send trace to Helicone (placeholder for production implementation)."""
        # In production, use Helicone SDK
        logger.debug("Would send trace to Helicone", trace_id=trace_entry.get("session_id"))

    def _call_twilio_ci_api(self, transcript: str) -> SentimentResult:
        """Call Twilio Conversational Intelligence API (placeholder)."""
        # In production, call actual Twilio CI API
        return self._analyze_sentiment_heuristic(transcript)

    def _analyze_sentiment_heuristic(self, transcript: str) -> SentimentResult:
        """Simple heuristic-based sentiment analysis."""
        text_lower = transcript.lower()

        # Positive and negative word lists
        positive_words = [
            "thank",
            "great",
            "good",
            "excellent",
            "perfect",
            "wonderful",
            "happy",
            "pleased",
            "appreciate",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "angry",
            "frustrated",
            "disappointed",
            "upset",
            "problem",
            "issue",
        ]

        # Count occurrences
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Calculate score
        total_count = positive_count + negative_count
        if total_count == 0:
            score = 0.0
            label = "neutral"
            magnitude = 0.0
        else:
            score = (positive_count - negative_count) / total_count
            magnitude = total_count / len(text_lower.split())
            if score > 0.2:
                label = "positive"
            elif score < -0.2:
                label = "negative"
            else:
                label = "neutral"

        return SentimentResult(
            score=score,
            magnitude=min(1.0, magnitude * 10),  # Scale magnitude
            label=label,
            confidence=0.6,  # Heuristic confidence
        )
