"""
Twilio-Voice Pipeline Bridge.

This module provides the bridge between Twilio telephony integration and
the Voice Pipeline, handling audio routing, session lifecycle, and event
coordination.

**Validates: Requirements 11.2, 11.3, 11.5, 11.6**
"""

import asyncio
import logging
from dataclasses import dataclass

from ..core.pipecat_orchestrator import CallMetadata as OrchestratorCallMetadata
from ..core.pipecat_orchestrator import PipecatOrchestrator
from ..core.voice_pipeline import LatencyMetrics, VoicePipeline
from .twilio_integration import CallMetadata as TwilioCallMetadata
from .twilio_integration import TwilioIntegration

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for Twilio-Voice Pipeline bridge.

    Attributes:
        session_id: Unique session identifier
        enable_audio_logging: Enable logging of audio data (for debugging)
        audio_buffer_size: Size of audio buffer for routing
    """
    session_id: str
    enable_audio_logging: bool = False
    audio_buffer_size: int = 8192


class TwilioVoiceBridge:
    """
    Bridge connecting Twilio telephony to Voice Pipeline.

    Handles bidirectional audio streaming, session lifecycle management,
    and event coordination between Twilio and the voice processing pipeline.

    **Validates: Requirements 11.2, 11.3, 11.5, 11.6**
    """

    def __init__(
        self,
        config: BridgeConfig,
        twilio_integration: TwilioIntegration,
        orchestrator: PipecatOrchestrator,
    ):
        """
        Initialize Twilio-Voice Pipeline bridge.

        Args:
            config: Bridge configuration
            twilio_integration: Twilio integration instance
            orchestrator: Pipecat orchestrator instance
        """
        self.config = config
        self._twilio = twilio_integration
        self._orchestrator = orchestrator
        self._pipeline: VoicePipeline | None = None
        self._is_active = False
        self._audio_routing_task: asyncio.Task | None = None

        logger.info(
            "Twilio-Voice Bridge initialized (session: %s)",
            config.session_id
        )

    @property
    def is_active(self) -> bool:
        """Check if bridge is active."""
        return self._is_active

    @property
    def pipeline(self) -> VoicePipeline | None:
        """Get active voice pipeline."""
        return self._pipeline

    async def start(self) -> None:
        """
        Start the bridge and wire up connections.

        Registers callbacks for Twilio events and initializes the voice pipeline
        when a call starts.

        **Validates: Requirements 11.2, 11.3, 11.5**
        """
        if self._is_active:
            logger.warning("Bridge already active")
            return

        logger.info("Starting Twilio-Voice Bridge")

        # Register Twilio callbacks
        self._twilio.on_call_started(self._handle_call_started)
        self._twilio.on_call_ended(self._handle_call_ended)
        self._twilio.on_audio_received(self._handle_twilio_audio)
        self._twilio.on_error(self._handle_twilio_error)

        self._is_active = True
        logger.info("Twilio-Voice Bridge started")

    async def stop(self) -> None:
        """
        Stop the bridge and cleanup resources.

        **Validates: Requirements 11.6**
        """
        if not self._is_active:
            logger.debug("Bridge already stopped")
            return

        logger.info("Stopping Twilio-Voice Bridge")

        self._is_active = False

        # Cancel audio routing task
        if self._audio_routing_task and not self._audio_routing_task.done():
            self._audio_routing_task.cancel()
            try:
                await self._audio_routing_task
            except asyncio.CancelledError:
                pass

        # Stop pipeline if active
        if self._pipeline:
            try:
                await self._orchestrator.stop_session(self.config.session_id)
                self._pipeline = None
            except Exception as e:
                logger.error("Error stopping pipeline: %s", str(e))

        logger.info("Twilio-Voice Bridge stopped")

    def _handle_call_started(self, call_metadata: TwilioCallMetadata) -> None:
        """
        Handle call started event from Twilio.

        Creates a new session and initializes the voice pipeline.

        Args:
            call_metadata: Metadata from Twilio about the call

        **Validates: Requirements 11.5**
        """
        logger.info(
            "Call started: %s from %s",
            call_metadata.call_sid,
            call_metadata.caller_id
        )

        # Create orchestrator call metadata
        orchestrator_metadata = OrchestratorCallMetadata(
            caller_id=call_metadata.caller_id,
            call_sid=call_metadata.call_sid,
            from_number=call_metadata.caller_id,
            to_number=call_metadata.called_number,
            additional_data={
                "direction": call_metadata.direction,
                "from_city": call_metadata.from_city,
                "from_state": call_metadata.from_state,
                "from_country": call_metadata.from_country,
                "custom_parameters": call_metadata.custom_parameters,
            },
        )

        # Start session asynchronously
        asyncio.create_task(self._start_session(orchestrator_metadata))

    async def _start_session(self, call_metadata: OrchestratorCallMetadata) -> None:
        """
        Start voice pipeline session.

        Args:
            call_metadata: Call metadata for session creation
        """
        try:
            # Start session and get pipeline
            self._pipeline = await self._orchestrator.start_session(
                session_id=self.config.session_id,
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

            # Register pipeline callbacks for audio output
            self._pipeline.on_audio_generated(self._handle_pipeline_audio)
            self._pipeline.on_latency_metric(self._handle_latency_metrics)

            logger.info("Voice pipeline session started: %s", self.config.session_id)

        except Exception as e:
            logger.error("Failed to start session: %s", str(e))
            # Notify Twilio of error
            if self._twilio._on_error:
                self._twilio._on_error(e)

    def _handle_call_ended(self, call_metadata: TwilioCallMetadata) -> None:
        """
        Handle call ended event from Twilio.

        Cleans up the session and stops the voice pipeline.

        Args:
            call_metadata: Metadata from Twilio about the ended call

        **Validates: Requirements 11.6**
        """
        logger.info(
            "Call ended: %s (duration: %.2fs)",
            call_metadata.call_sid,
            call_metadata.duration_seconds
        )

        # Stop session asynchronously
        asyncio.create_task(self._stop_session())

    async def _stop_session(self) -> None:
        """Stop voice pipeline session and cleanup."""
        try:
            if self._pipeline:
                summary = await self._orchestrator.stop_session(
                    session_id=self.config.session_id,
                    persist=True,
                )

                if summary:
                    logger.info(
                        "Session stopped: %s (turns: %d)",
                        self.config.session_id,
                        summary.get("turn_count", 0)
                    )

                self._pipeline = None

        except Exception as e:
            logger.error("Error stopping session: %s", str(e))

    def _handle_twilio_audio(self, audio_data: bytes) -> None:
        """
        Handle incoming audio from Twilio.

        Routes audio to the voice pipeline for processing.

        Args:
            audio_data: Raw audio bytes from Twilio (mulaw/pcm)

        **Validates: Requirements 11.2**
        """
        if not self._pipeline or not self._pipeline.is_active:
            logger.debug("Pipeline not active, dropping audio")
            return

        if self.config.enable_audio_logging:
            logger.debug("Received audio from Twilio: %d bytes", len(audio_data))

        # Route audio to pipeline asynchronously
        asyncio.create_task(self._route_audio_to_pipeline(audio_data))

    async def _route_audio_to_pipeline(self, audio_data: bytes) -> None:
        """
        Route audio data to voice pipeline.

        Args:
            audio_data: Audio bytes to route
        """
        try:
            if self._pipeline and self._pipeline.is_active:
                await self._pipeline.process_audio_chunk(audio_data)
        except Exception as e:
            logger.error("Error routing audio to pipeline: %s", str(e))

    def _handle_pipeline_audio(self, audio_data: bytes) -> None:
        """
        Handle generated audio from voice pipeline.

        Routes audio to Twilio for playback to the caller.

        Args:
            audio_data: Generated audio bytes from TTS

        **Validates: Requirements 11.3**
        """
        if not self._twilio.is_connected:
            logger.debug("Twilio not connected, dropping audio")
            return

        if self.config.enable_audio_logging:
            logger.debug("Sending audio to Twilio: %d bytes", len(audio_data))

        # Route audio to Twilio asynchronously
        asyncio.create_task(self._route_audio_to_twilio(audio_data))

    async def _route_audio_to_twilio(self, audio_data: bytes) -> None:
        """
        Route audio data to Twilio.

        Args:
            audio_data: Audio bytes to route
        """
        try:
            if self._twilio.is_connected:
                await self._twilio.send_audio(audio_data)
        except Exception as e:
            logger.error("Error routing audio to Twilio: %s", str(e))

    def _handle_latency_metrics(self, metrics: LatencyMetrics) -> None:
        """
        Handle latency metrics from voice pipeline.

        Args:
            metrics: Latency metrics from pipeline
        """
        logger.info(
            "Pipeline latency metrics - E2E: %.2fms, STT: %.2fms, LLM: %.2fms, TTS: %.2fms",
            metrics.end_to_end_ms,
            metrics.stt_latency_ms,
            metrics.llm_total_ms,
            metrics.tts_total_ms,
        )

    def _handle_twilio_error(self, error: Exception) -> None:
        """
        Handle error from Twilio integration.

        Args:
            error: Exception from Twilio
        """
        logger.error("Twilio error: %s", str(error))

        # Stop session on error
        asyncio.create_task(self._stop_session())

    def get_session_id(self) -> str:
        """
        Get current session ID.

        Returns:
            Session identifier
        """
        return self.config.session_id

    def get_call_metadata(self) -> TwilioCallMetadata | None:
        """
        Get current call metadata from Twilio.

        Returns:
            Call metadata if available, None otherwise
        """
        return self._twilio.get_call_metadata()

    def get_pipeline_metrics(self) -> LatencyMetrics | None:
        """
        Get current pipeline latency metrics.

        Returns:
            Latency metrics if pipeline is active, None otherwise
        """
        if self._pipeline:
            return self._pipeline.get_latency_metrics()
        return None
