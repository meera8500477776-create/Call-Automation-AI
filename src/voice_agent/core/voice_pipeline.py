"""
Voice Pipeline orchestrator.

This module provides the VoicePipeline class that coordinates STT, LLM, TTS, and VAD
components to create a real-time voice processing pipeline with ultra-low latency.
Supports concurrent processing, barge-in, and streaming optimizations.

**Validates: Requirements 2.1, 2.2, 2.3, 2.5, 2.6, 12.6**
"""

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..services.elevenlabs_tts_service import VoiceConfig
from ..services.llm_service import ConversationContext, LLMResponse, LLMService
from ..services.stt_service import STTConfig, STTService
from ..services.tts_factory import TTSProviderFactory
from ..services.vad_module import VADConfig, VADModule

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """Voice pipeline states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class LatencyMetrics:
    """
    Latency metrics for pipeline stages.

    Tracks timing for each stage of the voice pipeline to monitor
    performance and identify bottlenecks.

    **Validates: Requirements 2.1, 2.2, 2.7**
    """
    stt_latency_ms: float = 0.0
    llm_ttft_ms: float = 0.0  # Time To First Token
    llm_total_ms: float = 0.0
    tts_ttfb_ms: float = 0.0  # Time To First Byte
    tts_total_ms: float = 0.0
    end_to_end_ms: float = 0.0
    tool_execution_ms: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "stt_latency_ms": self.stt_latency_ms,
            "llm_ttft_ms": self.llm_ttft_ms,
            "llm_total_ms": self.llm_total_ms,
            "tts_ttfb_ms": self.tts_ttfb_ms,
            "tts_total_ms": self.tts_total_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "tool_execution_ms": self.tool_execution_ms,
        }


@dataclass
class StreamingConfig:
    """
    Configuration for streaming optimizations.

    Controls concurrent processing, buffering, and flow control
    for the voice pipeline.

    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    enable_concurrent_processing: bool = True
    max_audio_buffer_size: int = 32768  # 32KB
    audio_chunk_size: int = 8192  # 8KB
    max_concurrent_llm_requests: int = 1
    max_concurrent_tts_requests: int = 1
    enable_early_llm_start: bool = False  # Start LLM on interim results
    audio_queue_max_size: int = 100
    backpressure_threshold: float = 0.8  # 80% of queue capacity


@dataclass
class PipelineConfig:
    """
    Configuration for voice pipeline.

    Attributes:
        session_id: Unique session identifier
        stt_config: STT service configuration
        llm_service: LLM service instance
        vad_config: VAD module configuration
        voice_config: TTS voice configuration
        tts_provider_config: TTS provider-specific configuration
        enable_barge_in: Enable user interruption capability
        enable_interim_processing: Process interim STT results
        buffer_size: Audio buffer size in bytes
        streaming_config: Streaming optimization configuration
    """
    session_id: str
    stt_config: STTConfig
    llm_service: LLMService
    vad_config: VADConfig
    voice_config: VoiceConfig
    tts_provider_config: dict[str, Any] | None = None
    enable_barge_in: bool = True
    enable_interim_processing: bool = False
    buffer_size: int = 8192
    streaming_config: StreamingConfig | None = None


class VoicePipeline:
    """
    Voice Pipeline orchestrator coordinating STT, LLM, TTS, and VAD.

    Manages the real-time audio processing chain from audio input through
    speech recognition, language understanding, response generation, and
    speech synthesis. Supports concurrent processing stages, barge-in
    capability, and streaming optimizations for ultra-low latency.

    Pipeline flow:
        Audio Input → VAD → STT → LLM → TTS → Audio Output
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize voice pipeline.

        Args:
            config: Pipeline configuration including all component configs
        """
        self.config = config
        self._state = PipelineState.IDLE

        # Streaming configuration
        self._streaming_config = config.streaming_config or StreamingConfig()

        # Initialize components
        self._stt_service = STTService(config.stt_config)
        self._llm_service = config.llm_service
        self._vad_module = VADModule(config.vad_config)
        self._tts_factory = TTSProviderFactory()
        self._tts_service = None

        # Conversation context
        self._context = ConversationContext(
            session_id=config.session_id,
            conversation_history=[],
            extracted_parameters={},
        )

        # Callback handlers
        self._transcript_callback: Callable[[str, bool], None] | None = None
        self._llm_response_callback: Callable[[LLMResponse], None] | None = None
        self._audio_generated_callback: Callable[[bytes], None] | None = None
        self._latency_callback: Callable[[LatencyMetrics], None] | None = None

        # Barge-in state
        self._barge_in_enabled = config.enable_barge_in
        self._is_agent_speaking = False
        self._current_tts_task: asyncio.Task | None = None

        # Audio buffer for processing with flow control
        self._audio_buffer = bytearray()
        self._audio_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self._streaming_config.audio_queue_max_size
        )

        # Processing locks and semaphores for concurrent control
        self._processing_lock = asyncio.Lock()
        self._llm_semaphore = asyncio.Semaphore(
            self._streaming_config.max_concurrent_llm_requests
        )
        self._tts_semaphore = asyncio.Semaphore(
            self._streaming_config.max_concurrent_tts_requests
        )

        # Latency tracking
        self._current_metrics = LatencyMetrics()
        self._stage_timers: dict[str, float] = {}

        # Concurrent processing tasks
        self._processing_tasks: list[asyncio.Task] = []

        logger.info(
            "Voice pipeline initialized (session: %s, barge_in: %s, concurrent: %s)",
            config.session_id,
            config.enable_barge_in,
            self._streaming_config.enable_concurrent_processing,
        )

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """Check if pipeline is active."""
        return self._state == PipelineState.ACTIVE

    @property
    def is_agent_speaking(self) -> bool:
        """Check if agent is currently speaking."""
        return self._is_agent_speaking

    async def initialize(self) -> None:
        """
        Initialize pipeline components and establish connections.

        Raises:
            RuntimeError: If initialization fails
        """
        if self._state != PipelineState.IDLE:
            logger.warning("Pipeline already initialized")
            return

        self._state = PipelineState.INITIALIZING

        try:
            # Connect STT service
            await self._stt_service.connect(self.config.session_id)

            # Register STT callbacks
            self._stt_service.on_interim_result(self._handle_interim_transcript)
            self._stt_service.on_final_result(self._handle_final_transcript)
            self._stt_service.on_error(self._handle_stt_error)

            # Register VAD callbacks
            self._vad_module.on_speech_start(self._handle_speech_start)
            self._vad_module.on_speech_end(self._handle_speech_end)

            # Initialize TTS service
            self._tts_service = await self._tts_factory.create_tts_service(
                voice_config=self.config.voice_config,
                provider_config=self.config.tts_provider_config,
            )

            self._state = PipelineState.ACTIVE
            logger.info("Voice pipeline initialized successfully")

        except Exception as e:
            self._state = PipelineState.ERROR
            logger.error("Failed to initialize pipeline: %s", str(e))
            raise RuntimeError(f"Pipeline initialization failed: {e}") from e

    async def shutdown(self) -> None:
        """
        Shutdown pipeline and cleanup resources.
        """
        if self._state == PipelineState.IDLE:
            logger.debug("Pipeline already shutdown")
            return

        self._state = PipelineState.STOPPING

        try:
            # Cancel any ongoing processing tasks
            for task in self._processing_tasks:
                if not task.done():
                    task.cancel()

            # Wait for all tasks to complete
            if self._processing_tasks:
                await asyncio.gather(*self._processing_tasks, return_exceptions=True)

            self._processing_tasks.clear()

            # Cancel any ongoing TTS
            if self._current_tts_task and not self._current_tts_task.done():
                self._current_tts_task.cancel()
                try:
                    await self._current_tts_task
                except asyncio.CancelledError:
                    pass

            # Disconnect STT service
            if self._stt_service.is_connected:
                await self._stt_service.disconnect()

            # Reset VAD
            self._vad_module.reset()

            # Clear buffers
            self._audio_buffer.clear()

            # Clear audio queue
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            self._state = PipelineState.IDLE
            logger.info("Voice pipeline shutdown complete")

        except Exception as e:
            logger.error("Error during pipeline shutdown: %s", str(e))
            self._state = PipelineState.ERROR
            raise

    def on_transcript(self, callback: Callable[[str, bool], None]) -> None:
        """
        Register callback for transcript events.

        Args:
            callback: Function to call with (transcript, is_final) when transcript arrives
        """
        self._transcript_callback = callback
        logger.debug("Transcript callback registered")

    def on_llm_response(self, callback: Callable[[LLMResponse], None]) -> None:
        """
        Register callback for LLM response events.

        Args:
            callback: Function to call with LLMResponse when LLM generates response
        """
        self._llm_response_callback = callback
        logger.debug("LLM response callback registered")

    def on_audio_generated(self, callback: Callable[[bytes], None]) -> None:
        """
        Register callback for TTS audio generation events.

        Args:
            callback: Function to call with audio bytes when TTS generates audio
        """
        self._audio_generated_callback = callback
        logger.debug("Audio generated callback registered")

    def on_latency_metric(self, callback: Callable[[LatencyMetrics], None]) -> None:
        """
        Register callback for latency metric events.

        Args:
            callback: Function to call with LatencyMetrics when metrics are recorded
        """
        self._latency_callback = callback
        logger.debug("Latency metric callback registered")

    def enable_barge_in(self, enabled: bool = True) -> None:
        """
        Enable or disable barge-in capability.

        When enabled, user speech will interrupt agent speech output.

        Args:
            enabled: True to enable barge-in, False to disable
        """
        self._barge_in_enabled = enabled
        logger.info("Barge-in %s", "enabled" if enabled else "disabled")

    def _start_latency_timer(self, stage: str) -> None:
        """
        Start latency timer for a pipeline stage.

        Args:
            stage: Stage name (stt, llm, tts, end_to_end)
        """
        self._stage_timers[stage] = time.perf_counter()
        logger.debug("Started timer for stage: %s", stage)

    def _end_latency_timer(self, stage: str) -> float:
        """
        End latency timer for a pipeline stage and return duration.

        Args:
            stage: Stage name (stt, llm, tts, end_to_end)

        Returns:
            Duration in milliseconds
        """
        if stage not in self._stage_timers:
            logger.warning("No timer found for stage: %s", stage)
            return 0.0

        start_time = self._stage_timers.pop(stage)
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.debug("Stage %s completed in %.2f ms", stage, duration_ms)
        return duration_ms

    def _record_latency(self, stage: str, duration_ms: float) -> None:
        """
        Record latency for a pipeline stage.

        Args:
            stage: Stage name
            duration_ms: Duration in milliseconds
        """
        if stage == "stt":
            self._current_metrics.stt_latency_ms = duration_ms
        elif stage == "llm_ttft":
            self._current_metrics.llm_ttft_ms = duration_ms
        elif stage == "llm_total":
            self._current_metrics.llm_total_ms = duration_ms
        elif stage == "tts_ttfb":
            self._current_metrics.tts_ttfb_ms = duration_ms
        elif stage == "tts_total":
            self._current_metrics.tts_total_ms = duration_ms
        elif stage == "end_to_end":
            self._current_metrics.end_to_end_ms = duration_ms

        logger.debug("Recorded latency for %s: %.2f ms", stage, duration_ms)

    def get_latency_metrics(self) -> LatencyMetrics:
        """
        Get current latency metrics.

        Returns:
            Current latency metrics
        """
        return self._current_metrics

    def _check_backpressure(self) -> bool:
        """
        Check if audio queue is experiencing backpressure.

        Returns:
            True if backpressure threshold exceeded
        """
        if self._audio_queue.maxsize == 0:
            return False

        queue_utilization = self._audio_queue.qsize() / self._audio_queue.maxsize
        return queue_utilization >= self._streaming_config.backpressure_threshold

    async def process_audio_chunk(self, audio_data: bytes) -> None:
        """
        Process incoming audio chunk through the pipeline.

        Handles audio buffering, VAD processing, and STT streaming.
        Implements barge-in detection when enabled. Includes flow control
        to prevent buffer overflow.

        Args:
            audio_data: Raw audio data bytes (16-bit PCM)

        Raises:
            RuntimeError: If pipeline is not active
            ValueError: If audio data is invalid
        """
        if self._state != PipelineState.ACTIVE:
            raise RuntimeError(f"Pipeline not active (state: {self._state.value})")

        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        try:
            # Check for backpressure
            if self._check_backpressure():
                logger.warning("Audio queue backpressure detected, applying flow control")
                # Could implement adaptive buffering here
                await asyncio.sleep(0.01)  # Brief pause to allow queue to drain

            # Add to buffer
            self._audio_buffer.extend(audio_data)

            # Process buffer in chunks
            while len(self._audio_buffer) >= self._streaming_config.audio_chunk_size:
                chunk = bytes(self._audio_buffer[:self._streaming_config.audio_chunk_size])
                del self._audio_buffer[:self._streaming_config.audio_chunk_size]

                # Process through VAD for speech detection
                vad_result = self._vad_module.process_audio(chunk)

                # Check for barge-in
                if self._barge_in_enabled and vad_result.is_speech and self._is_agent_speaking:
                    logger.info("Barge-in detected, interrupting agent speech")
                    await self._handle_barge_in()

                # Send audio to STT service
                if self._stt_service.is_connected:
                    await self._stt_service.send_audio(chunk)

            # Enforce max buffer size
            if len(self._audio_buffer) > self._streaming_config.max_audio_buffer_size:
                logger.warning("Audio buffer overflow, discarding oldest data")
                excess = len(self._audio_buffer) - self._streaming_config.max_audio_buffer_size
                del self._audio_buffer[:excess]

            logger.debug("Processed audio chunk: %d bytes (buffer: %d bytes)",
                        len(audio_data), len(self._audio_buffer))

        except Exception as e:
            logger.error("Error processing audio chunk: %s", str(e))
            raise

    async def _handle_interim_transcript(self, transcript: str, metadata: dict[str, Any]) -> None:
        """
        Handle interim transcript from STT service.

        Args:
            transcript: Interim transcript text
            metadata: Transcript metadata
        """
        logger.debug("Interim transcript: %s", transcript)

        # Notify callback
        if self._transcript_callback:
            self._transcript_callback(transcript, False)

        # Process interim results if enabled for early LLM start
        if (self.config.enable_interim_processing and
            self._streaming_config.enable_early_llm_start and
            transcript.strip()):
            # Start LLM processing early for lower latency
            # This is experimental and may result in incomplete context
            logger.debug("Early LLM start enabled, processing interim transcript")
            # Could trigger early LLM processing here

    async def _handle_final_transcript(self, transcript: str, metadata: dict[str, Any]) -> None:
        """
        Handle final transcript from STT service.

        Triggers LLM processing and response generation.
        Tracks STT latency.

        Args:
            transcript: Final transcript text
            metadata: Transcript metadata
        """
        # Record STT latency if timer was started
        if "stt" in self._stage_timers:
            stt_latency = self._end_latency_timer("stt")
            self._record_latency("stt", stt_latency)

        logger.info("Final transcript: %s (confidence: %.2f)", transcript, metadata.get("confidence", 0.0))

        # Notify callback
        if self._transcript_callback:
            self._transcript_callback(transcript, True)

        # Process through LLM
        if transcript.strip():
            # Start end-to-end timer
            self._start_latency_timer("end_to_end")

            # Process LLM concurrently if enabled
            if self._streaming_config.enable_concurrent_processing:
                task = asyncio.create_task(self._process_llm(transcript))
                self._processing_tasks.append(task)
            else:
                await self._process_llm(transcript)

    async def _process_llm(self, transcript: str) -> None:
        """
        Process transcript through LLM and generate response.

        Uses semaphore for concurrent request control and tracks
        LLM latency (TTFT and total).

        Args:
            transcript: User transcript to process
        """
        async with self._llm_semaphore:
            async with self._processing_lock:
                try:
                    # Add user turn to context
                    self._context.conversation_history.append({
                        "role": "user",
                        "content": transcript,
                    })

                    # Start LLM timer
                    self._start_latency_timer("llm_total")
                    self._start_latency_timer("llm_ttft")

                    # Generate LLM response
                    logger.debug("Generating LLM response for: %s", transcript)
                    llm_response = await self._llm_service.generate_response(
                        transcript=transcript,
                        context=self._context,
                    )

                    # Record TTFT (approximation - actual TTFT would need streaming support)
                    llm_ttft = self._end_latency_timer("llm_ttft")
                    self._record_latency("llm_ttft", llm_ttft)

                    logger.info("LLM response generated: %s", llm_response.text[:100])

                    # Record total LLM time
                    llm_total = self._end_latency_timer("llm_total")
                    self._record_latency("llm_total", llm_total)

                    # Update context with extracted parameters
                    self._context.extracted_parameters.update(llm_response.parameters)

                    # Add assistant turn to context
                    self._context.conversation_history.append({
                        "role": "assistant",
                        "content": llm_response.text,
                    })

                    # Notify callback
                    if self._llm_response_callback:
                        self._llm_response_callback(llm_response)

                    # Generate audio response (concurrent if enabled)
                    if self._streaming_config.enable_concurrent_processing:
                        task = asyncio.create_task(self._generate_audio(llm_response))
                        self._processing_tasks.append(task)
                    else:
                        await self._generate_audio(llm_response)

                except Exception as e:
                    logger.error("Error processing LLM: %s", str(e))
                    # Clean up timers
                    self._stage_timers.pop("llm_total", None)
                    self._stage_timers.pop("llm_ttft", None)
                    raise

    async def _generate_audio(self, llm_response: LLMResponse) -> None:
        """
        Generate audio from LLM response using TTS.

        Tracks TTS latency (TTFB and total) and uses semaphore
        for concurrent request control.

        Args:
            llm_response: LLM response to synthesize
        """
        if not self._tts_service:
            logger.error("TTS service not initialized")
            return

        async with self._tts_semaphore:
            try:
                # Apply emotional context if available
                text = llm_response.text
                if llm_response.emotional_tags:
                    text = await self._tts_service.apply_emotional_context(
                        text,
                        llm_response.emotional_tags,
                    )

                # Mark agent as speaking
                self._is_agent_speaking = True

                # Start TTS timer
                self._start_latency_timer("tts_total")
                self._start_latency_timer("tts_ttfb")

                # Create TTS task
                self._current_tts_task = asyncio.create_task(
                    self._stream_tts_audio(text)
                )

                # Wait for TTS to complete
                await self._current_tts_task

                # Record total TTS time
                tts_total = self._end_latency_timer("tts_total")
                self._record_latency("tts_total", tts_total)

                # Record end-to-end latency
                if "end_to_end" in self._stage_timers:
                    e2e_latency = self._end_latency_timer("end_to_end")
                    self._record_latency("end_to_end", e2e_latency)

                    # Notify latency callback
                    if self._latency_callback:
                        self._latency_callback(self._current_metrics)

                    # Log metrics
                    logger.info("Pipeline metrics: %s", self._current_metrics.to_dict())

            except asyncio.CancelledError:
                logger.info("TTS generation cancelled (barge-in)")
                # Clean up timers
                self._stage_timers.pop("tts_total", None)
                self._stage_timers.pop("tts_ttfb", None)
            except Exception as e:
                logger.error("Error generating audio: %s", str(e))
                # Clean up timers
                self._stage_timers.pop("tts_total", None)
                self._stage_timers.pop("tts_ttfb", None)
                raise
            finally:
                self._is_agent_speaking = False
                self._current_tts_task = None

    async def _stream_tts_audio(self, text: str) -> None:
        """
        Stream TTS audio chunks.

        Tracks TTFB (time to first byte) for the first audio chunk.

        Args:
            text: Text to synthesize
        """
        first_chunk = True

        try:
            async for audio_chunk in self._tts_service.synthesize_streaming(text):
                # Record TTFB on first chunk
                if first_chunk and "tts_ttfb" in self._stage_timers:
                    tts_ttfb = self._end_latency_timer("tts_ttfb")
                    self._record_latency("tts_ttfb", tts_ttfb)
                    first_chunk = False

                # Notify callback with audio chunk
                if self._audio_generated_callback:
                    self._audio_generated_callback(audio_chunk)

                logger.debug("Generated audio chunk: %d bytes", len(audio_chunk))

        except Exception as e:
            logger.error("Error streaming TTS audio: %s", str(e))
            raise

    async def _handle_barge_in(self) -> None:
        """
        Handle barge-in event by interrupting agent speech.
        """
        if self._current_tts_task and not self._current_tts_task.done():
            logger.info("Cancelling current TTS due to barge-in")
            self._current_tts_task.cancel()
            self._is_agent_speaking = False

    def _handle_speech_start(self, timestamp: float) -> None:
        """
        Handle speech start event from VAD.

        Args:
            timestamp: Timestamp when speech started
        """
        logger.debug("Speech started at %.3f", timestamp)

        # Could trigger barge-in here if agent is speaking
        if self._barge_in_enabled and self._is_agent_speaking:
            asyncio.create_task(self._handle_barge_in())

    def _handle_speech_end(self, timestamp: float) -> None:
        """
        Handle speech end event from VAD.

        Args:
            timestamp: Timestamp when speech ended
        """
        logger.debug("Speech ended at %.3f", timestamp)

    def _handle_stt_error(self, error: Exception) -> None:
        """
        Handle STT service error.

        Args:
            error: Exception from STT service
        """
        logger.error("STT error: %s", str(error))
        self._state = PipelineState.ERROR

    def get_context(self) -> ConversationContext:
        """
        Get current conversation context.

        Returns:
            Current conversation context
        """
        return self._context

    def update_context(self, **kwargs) -> None:
        """
        Update conversation context fields.

        Args:
            **kwargs: Fields to update in context
        """
        for key, value in kwargs.items():
            if hasattr(self._context, key):
                setattr(self._context, key, value)
                logger.debug("Updated context field: %s", key)
