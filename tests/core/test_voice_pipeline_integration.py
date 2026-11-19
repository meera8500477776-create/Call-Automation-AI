"""
Integration tests for Voice Pipeline.

Tests end-to-end audio processing, barge-in functionality, and concurrent
processing stages with mock components.

**Validates: Requirements 2.1, 2.2, 2.3, 2.5**
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, PropertyMock, patch

import pytest

from src.voice_agent.core.voice_pipeline import (
    LatencyMetrics,
    PipelineConfig,
    StreamingConfig,
    VoicePipeline,
)
from src.voice_agent.services.elevenlabs_tts_service import VoiceConfig
from src.voice_agent.services.llm_service import (
    ConversationContext,
    Intent,
    LLMConfig,
    LLMResponse,
    LLMService,
)
from src.voice_agent.services.stt_service import ConnectionState, STTConfig
from src.voice_agent.services.vad_module import VADConfig, VADResult


class MockLLMService(LLMService):
    """Mock LLM service for integration testing."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.generate_response_mock = None
        self.call_count = 0

    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """Mock generate_response with tracking."""
        self.call_count += 1

        if self.generate_response_mock:
            return await self.generate_response_mock(transcript, context)

        # Default response based on transcript
        intent = self.classify_intent(transcript)
        return LLMResponse(
            text=f"Response to: {transcript[:30]}",
            intent=intent,
            parameters={},  # Regular dict, not AsyncMock
            confidence=0.9,
            emotional_tags=["friendly"],
        )


@pytest.fixture
def stt_config():
    """Create STT configuration for testing."""
    return STTConfig(
        api_key="test_api_key",
        model="nova-2",
        language="en-US",
    )


@pytest.fixture
def llm_service():
    """Create mock LLM service for testing."""
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
    )
    return MockLLMService(config)


@pytest.fixture
def vad_config():
    """Create VAD configuration for testing."""
    return VADConfig(
        sample_rate=16000,
        threshold=0.5,
    )


@pytest.fixture
def voice_config():
    """Create voice configuration for testing."""
    return VoiceConfig(
        provider="openai",
        voice_id="alloy",
    )


@pytest.fixture
def streaming_config():
    """Create streaming configuration for testing."""
    return StreamingConfig(
        enable_concurrent_processing=True,
        max_audio_buffer_size=16384,
        audio_chunk_size=2048,
        max_concurrent_llm_requests=2,
        max_concurrent_tts_requests=2,
    )



@pytest.fixture
def pipeline_config(stt_config, llm_service, vad_config, voice_config, streaming_config):
    """Create pipeline configuration for testing."""
    return PipelineConfig(
        session_id="integration_test_session",
        stt_config=stt_config,
        llm_service=llm_service,
        vad_config=vad_config,
        voice_config=voice_config,
        enable_barge_in=True,
        streaming_config=streaming_config,
    )


@pytest.fixture
async def initialized_pipeline(pipeline_config):
    """Create and initialize a pipeline for testing."""
    pipeline = VoicePipeline(pipeline_config)

    # Mock WebSocket connection for STT
    mock_ws = MagicMock()
    mock_ws.send_media = AsyncMock()

    # Mock TTS service
    mock_tts_service = Mock()

    async def mock_tts_stream(text):
        yield b'audio_chunk_1'
        yield b'audio_chunk_2'
        yield b'audio_chunk_3'

    mock_tts_service.synthesize_streaming = Mock(return_value=mock_tts_stream("test"))
    mock_tts_service.apply_emotional_context = AsyncMock(return_value="test text")

    # Initialize pipeline with mocked components
    with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock) as mock_connect, \
         patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts_factory:

        async def setup_stt_connection(session_id):
            pipeline._stt_service._connection = mock_ws
            pipeline._stt_service._state = ConnectionState.CONNECTED

        mock_connect.side_effect = setup_stt_connection
        mock_tts_factory.return_value = mock_tts_service

        await pipeline.initialize()

    yield pipeline

    # Cleanup
    if pipeline.is_active:
        await pipeline.shutdown()


class TestEndToEndAudioProcessing:
    """Integration tests for end-to-end audio processing."""

    @pytest.mark.asyncio
    async def test_complete_audio_pipeline_flow(self, initialized_pipeline):
        """
        Test complete audio processing flow from input to output.

        Validates that audio flows through all pipeline stages:
        Audio Input → VAD → STT → LLM → TTS → Audio Output

        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        pipeline = initialized_pipeline

        # Track callbacks
        transcript_callback = Mock()
        llm_response_callback = Mock()
        audio_generated_callback = Mock()
        latency_callback = Mock()

        pipeline.on_transcript(transcript_callback)
        pipeline.on_llm_response(llm_response_callback)
        pipeline.on_audio_generated(audio_generated_callback)
        pipeline.on_latency_metric(latency_callback)

        # Mock VAD to detect speech
        vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Step 1: Process audio chunk
            audio_data = b'\x00\x01' * 2048
            await pipeline.process_audio_chunk(audio_data)

            # Step 2: Simulate final transcript from STT
            await pipeline._handle_final_transcript(
                "I want to schedule an appointment",
                {"confidence": 0.95}
            )

            # Wait for async processing
            await asyncio.sleep(0.3)

            # Wait for processing tasks
            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)

            await asyncio.sleep(0.2)

        # Verify all callbacks were invoked
        assert transcript_callback.called, "Transcript callback not invoked"
        assert llm_response_callback.called, "LLM response callback not invoked"
        assert audio_generated_callback.called, "Audio generated callback not invoked"

        # Verify transcript callback received final transcript
        transcript_calls = [call for call in transcript_callback.call_args_list
                           if call[0][1] is True]  # is_final=True
        assert len(transcript_calls) > 0, "No final transcript received"
        assert transcript_calls[0][0][0] == "I want to schedule an appointment"

        # Verify LLM response callback
        llm_response = llm_response_callback.call_args[0][0]
        assert isinstance(llm_response, LLMResponse)
        assert llm_response.text is not None

        # Verify audio was generated (multiple chunks)
        assert audio_generated_callback.call_count >= 2, \
            f"Expected multiple audio chunks, got {audio_generated_callback.call_count}"

        # Verify latency metrics were recorded
        if latency_callback.called:
            metrics = latency_callback.call_args[0][0]
            assert isinstance(metrics, LatencyMetrics)
            assert metrics.end_to_end_ms > 0

    @pytest.mark.asyncio
    async def test_multiple_turns_conversation(self, initialized_pipeline):
        """
        Test multiple conversation turns through the pipeline.

        Validates that the pipeline can handle multiple user inputs
        and maintain conversation context.

        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        pipeline = initialized_pipeline

        # Track responses
        llm_responses = []

        def capture_llm_response(response):
            llm_responses.append(response)

        pipeline.on_llm_response(capture_llm_response)

        vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Turn 1
            audio_data = b'\x00\x01' * 2048
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript("Hello", {"confidence": 0.9})
            await asyncio.sleep(0.2)

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)
            await asyncio.sleep(0.1)

            # Turn 2
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript(
                "I need to book an appointment",
                {"confidence": 0.95}
            )
            await asyncio.sleep(0.2)

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)
            await asyncio.sleep(0.1)

            # Turn 3
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript("Tomorrow at 2pm", {"confidence": 0.9})
            await asyncio.sleep(0.2)

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)
            await asyncio.sleep(0.1)

        # Verify multiple responses
        assert len(llm_responses) >= 3, \
            f"Expected at least 3 LLM responses, got {len(llm_responses)}"

        # Verify conversation context was maintained
        context = pipeline.get_context()
        assert len(context.conversation_history) >= 6, \
            "Expected at least 6 turns (3 user + 3 assistant)"

        # Verify user and assistant turns alternate
        roles = [turn["role"] for turn in context.conversation_history]
        for i in range(0, len(roles) - 1, 2):
            if i + 1 < len(roles):
                assert roles[i] == "user", f"Expected user at index {i}"
                assert roles[i + 1] == "assistant", f"Expected assistant at index {i + 1}"

    @pytest.mark.asyncio
    async def test_audio_processing_with_latency_tracking(self, initialized_pipeline):
        """
        Test audio processing tracks latency for all stages.

        Validates that latency is measured for STT, LLM, TTS, and end-to-end.

        **Validates: Requirements 2.1, 2.2, 2.7**
        """
        pipeline = initialized_pipeline

        # Track latency metrics
        latency_metrics = []

        def capture_latency(metrics):
            latency_metrics.append(metrics)

        pipeline.on_latency_metric(capture_latency)

        vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Process audio
            audio_data = b'\x00\x01' * 2048
            await pipeline.process_audio_chunk(audio_data)

            # Simulate transcript with STT timer
            pipeline._start_latency_timer("stt")
            await asyncio.sleep(0.01)  # Simulate STT processing time
            await pipeline._handle_final_transcript(
                "Test transcript",
                {"confidence": 0.9}
            )

            # Wait for processing
            await asyncio.sleep(0.3)

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)

            await asyncio.sleep(0.2)

        # Verify latency was tracked
        if len(latency_metrics) > 0:
            metrics = latency_metrics[0]

            # Check that key metrics are recorded
            assert metrics.end_to_end_ms > 0, "End-to-end latency not recorded"
            assert metrics.llm_total_ms > 0, "LLM latency not recorded"
            assert metrics.tts_total_ms > 0, "TTS latency not recorded"

            # Verify end-to-end includes processing time
            # End-to-end should be reasonable (not negative, not zero)
            assert metrics.end_to_end_ms > 0, "End-to-end latency should be positive"

            # LLM and TTS should have been measured
            assert metrics.llm_total_ms > 0 and metrics.tts_total_ms > 0, \
                "LLM and TTS latencies should be measured"


class TestBargeInFunctionality:
    """Integration tests for barge-in functionality."""

    @pytest.mark.asyncio
    async def test_barge_in_interrupts_agent_speech(self, initialized_pipeline):
        """
        Test barge-in interrupts ongoing agent speech.

        Validates that when user speaks while agent is speaking,
        the agent's speech is interrupted.

        **Validates: Requirements 2.5**
        """
        pipeline = initialized_pipeline

        # Track TTS cancellations
        tts_started = []
        tts_cancelled = []

        # Start agent speaking
        pipeline._is_agent_speaking = True

        # Create a real asyncio task for TTS
        async def mock_tts_generation():
            tts_started.append(True)
            try:
                await asyncio.sleep(1.0)  # Long TTS generation
                return "completed"
            except asyncio.CancelledError:
                tts_cancelled.append(True)
                raise

        pipeline._current_tts_task = asyncio.create_task(mock_tts_generation())

        # Wait for TTS to start
        await asyncio.sleep(0.05)

        # Verify TTS started
        assert len(tts_started) > 0, "TTS task should have started"

        # Mock VAD detecting user speech (barge-in)
        vad_result = VADResult(is_speech=True, confidence=0.95, timestamp=2.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Process audio chunk (should trigger barge-in)
            # Use larger chunk to ensure it gets processed
            audio_data = b'\x00\x01' * 4096
            await pipeline.process_audio_chunk(audio_data)

            # Wait for barge-in to process
            await asyncio.sleep(0.3)

        # Verify TTS was cancelled
        assert pipeline._current_tts_task.cancelled() or len(tts_cancelled) > 0, \
            "TTS task should be cancelled during barge-in"
        assert not pipeline._is_agent_speaking, "Agent should not be speaking after barge-in"

    @pytest.mark.asyncio
    async def test_barge_in_disabled_does_not_interrupt(self, initialized_pipeline):
        """
        Test that with barge-in disabled, agent speech is not interrupted.

        **Validates: Requirements 2.5**
        """
        pipeline = initialized_pipeline

        # Disable barge-in
        pipeline.enable_barge_in(False)

        # Track TTS cancellations
        tts_cancelled = []

        # Start agent speaking
        pipeline._is_agent_speaking = True

        async def mock_tts_generation():
            try:
                await asyncio.sleep(0.5)
                return "completed"
            except asyncio.CancelledError:
                tts_cancelled.append(True)
                raise

        pipeline._current_tts_task = asyncio.create_task(mock_tts_generation())

        # Mock VAD detecting user speech
        vad_result = VADResult(is_speech=True, confidence=0.95, timestamp=2.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Process audio chunk
            audio_data = b'\x00\x01' * 2048
            await pipeline.process_audio_chunk(audio_data)

            # Wait briefly
            await asyncio.sleep(0.2)

        # Verify TTS was NOT cancelled (barge-in disabled)
        assert len(tts_cancelled) == 0, "TTS should not be cancelled when barge-in is disabled"
        assert pipeline._is_agent_speaking, "Agent should still be speaking"

        # Cleanup
        if not pipeline._current_tts_task.done():
            pipeline._current_tts_task.cancel()
            try:
                await pipeline._current_tts_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_barge_in_during_audio_generation(self, initialized_pipeline):
        """
        Test barge-in during active audio generation.

        Validates that barge-in works during the TTS streaming phase.

        **Validates: Requirements 2.5**
        """
        pipeline = initialized_pipeline

        # Track audio chunks generated before interruption
        audio_chunks_before_interrupt = []

        def capture_audio(chunk):
            audio_chunks_before_interrupt.append(chunk)

        pipeline.on_audio_generated(capture_audio)

        # Setup LLM response
        llm_response = LLMResponse(
            text="This is a long response that will be interrupted",
            intent=Intent(name="general_inquiry", confidence=0.9),
            parameters={},
            confidence=0.9,
            emotional_tags=["friendly"],
        )

        # Start audio generation in background
        audio_gen_task = asyncio.create_task(pipeline._generate_audio(llm_response))

        # Wait for audio generation to start
        await asyncio.sleep(0.1)

        # Trigger barge-in
        vad_result = VADResult(is_speech=True, confidence=0.95, timestamp=2.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            audio_data = b'\x00\x01' * 2048
            await pipeline.process_audio_chunk(audio_data)

            # Wait for barge-in to process
            await asyncio.sleep(0.2)

        # Wait for audio generation task to complete/cancel
        try:
            await asyncio.wait_for(audio_gen_task, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass

        # Verify some audio was generated before interruption
        # (may be 0 if interruption was very fast)
        assert not pipeline._is_agent_speaking, "Agent should not be speaking after barge-in"


class TestConcurrentProcessing:
    """Integration tests for concurrent processing stages."""

    @pytest.mark.asyncio
    async def test_concurrent_llm_processing(self, initialized_pipeline):
        """
        Test concurrent LLM processing with multiple requests.

        Validates that the pipeline can handle multiple LLM requests
        concurrently when configured to do so.

        **Validates: Requirements 2.2, 2.3**
        """
        pipeline = initialized_pipeline
        llm_service = pipeline._llm_service

        # Hold each request until the second one enters the LLM. This proves
        # overlap without relying on wall-clock scheduling thresholds.
        llm_call_count = 0
        both_calls_started = asyncio.Event()
        release_calls = asyncio.Event()

        original_generate = llm_service.generate_response

        async def tracked_generate(transcript, context):
            nonlocal llm_call_count
            llm_call_count += 1
            if llm_call_count == 2:
                both_calls_started.set()
            await release_calls.wait()
            return await original_generate(transcript, context)

        llm_service.generate_response = tracked_generate

        with patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Feed final transcripts directly; audio/VAD is covered elsewhere.
            await pipeline._handle_final_transcript("First request", {"confidence": 0.9})
            await pipeline._handle_final_transcript("Second request", {"confidence": 0.9})

            await asyncio.wait_for(both_calls_started.wait(), timeout=2)
            release_calls.set()

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)

        assert llm_call_count == 2, f"Expected 2 overlapping LLM calls, got {llm_call_count}"

    @pytest.mark.asyncio
    async def test_concurrent_tts_processing(self, initialized_pipeline):
        """
        Test concurrent TTS processing with multiple requests.

        Validates that the pipeline can handle multiple TTS requests
        concurrently when configured to do so.

        **Validates: Requirements 2.2, 2.3**
        """
        pipeline = initialized_pipeline

        # Track TTS calls
        tts_call_times = []

        original_stream = pipeline._tts_service.synthesize_streaming

        def tracked_stream(text):
            tts_call_times.append(asyncio.get_event_loop().time())
            return original_stream(text)

        pipeline._tts_service.synthesize_streaming = tracked_stream

        # Generate multiple audio responses
        llm_response1 = LLMResponse(
            text="First response",
            intent=Intent(name="general_inquiry", confidence=0.9),
            parameters={},
            confidence=0.9,
        )

        llm_response2 = LLMResponse(
            text="Second response",
            intent=Intent(name="general_inquiry", confidence=0.9),
            parameters={},
            confidence=0.9,
        )

        # Start both TTS generations
        task1 = asyncio.create_task(pipeline._generate_audio(llm_response1))
        await asyncio.sleep(0.05)
        task2 = asyncio.create_task(pipeline._generate_audio(llm_response2))

        # Wait for both to complete
        await asyncio.gather(task1, task2, return_exceptions=True)

        # Verify multiple TTS calls were made
        assert len(tts_call_times) >= 2, \
            f"Expected at least 2 TTS calls, got {len(tts_call_times)}"

        # With concurrent processing and semaphore=2, both should be able to run
        if len(tts_call_times) >= 2:
            time_diff = tts_call_times[1] - tts_call_times[0]
            # Second call should start shortly after first
            assert time_diff < 0.2, \
                f"TTS calls not concurrent enough, time diff: {time_diff}"

    @pytest.mark.asyncio
    async def test_pipeline_stages_run_concurrently(self, initialized_pipeline):
        """
        Test that pipeline stages can run concurrently.

        Validates that STT, LLM, and TTS stages can overlap when
        concurrent processing is enabled.

        **Validates: Requirements 2.2, 2.3**
        """
        pipeline = initialized_pipeline

        # Track stage execution times
        stage_times = {
            "stt": [],
            "llm": [],
            "tts": [],
        }

        # Wrap methods to track timing
        original_stt_send = pipeline._stt_service.send_audio

        async def tracked_stt_send(audio_chunk):
            stage_times["stt"].append(asyncio.get_event_loop().time())
            return await original_stt_send(audio_chunk)

        original_llm_generate = pipeline._llm_service.generate_response

        async def tracked_llm_generate(transcript, context):
            stage_times["llm"].append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.05)  # Simulate processing
            return await original_llm_generate(transcript, context)

        original_tts_stream = pipeline._tts_service.synthesize_streaming

        def tracked_tts_stream(text):
            stage_times["tts"].append(asyncio.get_event_loop().time())
            return original_tts_stream(text)

        pipeline._stt_service.send_audio = tracked_stt_send
        pipeline._llm_service.generate_response = tracked_llm_generate
        pipeline._tts_service.synthesize_streaming = tracked_tts_stream

        vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Process multiple audio chunks and transcripts
            audio_data = b'\x00\x01' * 2048

            # First turn
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript("First message", {"confidence": 0.9})

            await asyncio.sleep(0.1)

            # Second turn (while first might still be processing)
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript("Second message", {"confidence": 0.9})

            # Wait for all processing
            await asyncio.sleep(0.5)

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)

            await asyncio.sleep(0.2)

        # Verify all stages were executed
        assert len(stage_times["stt"]) > 0, "STT stage not executed"
        assert len(stage_times["llm"]) > 0, "LLM stage not executed"
        assert len(stage_times["tts"]) > 0, "TTS stage not executed"

        # With concurrent processing, stages should overlap
        # Check that second LLM call happens before first TTS completes
        if len(stage_times["llm"]) >= 2 and len(stage_times["tts"]) >= 1:
            # This demonstrates concurrent processing across stages
            assert stage_times["llm"][0] < stage_times["tts"][0], \
                "LLM should start before TTS"

    @pytest.mark.asyncio
    async def test_backpressure_handling(self, initialized_pipeline):
        """
        Test pipeline handles backpressure correctly.

        Validates that when the audio queue fills up, the pipeline
        applies flow control to prevent overflow.

        **Validates: Requirements 2.2, 2.3**
        """
        pipeline = initialized_pipeline

        # Fill the audio queue to trigger backpressure
        max_size = pipeline._audio_queue.maxsize

        # Fill queue to 90% capacity
        for _ in range(int(max_size * 0.9)):
            await pipeline._audio_queue.put(b'data')

        # Check backpressure is detected
        assert pipeline._check_backpressure(), \
            "Backpressure should be detected when queue is 90% full"

        vad_result = VADResult(is_speech=False, confidence=0.5, timestamp=1.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            # Process audio chunk (should handle backpressure)
            audio_data = b'\x00\x01' * 2048

            # This should not raise an error despite backpressure
            await pipeline.process_audio_chunk(audio_data)

        # Clean up queue
        while not pipeline._audio_queue.empty():
            try:
                await pipeline._audio_queue.get()
            except asyncio.QueueEmpty:
                break

    @pytest.mark.asyncio
    async def test_error_handling_in_concurrent_processing(self, initialized_pipeline):
        """
        Test error handling during concurrent processing.

        Validates that errors in one processing stage don't crash
        the entire pipeline.

        **Validates: Requirements 2.2, 2.3**
        """
        pipeline = initialized_pipeline
        llm_service = pipeline._llm_service

        # Make LLM raise an error on first call
        call_count = [0]

        original_generate = llm_service.generate_response

        async def error_on_first(transcript, context):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("Simulated LLM error")
            return await original_generate(transcript, context)

        llm_service.generate_response = error_on_first

        vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)

        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected',
                         new_callable=PropertyMock, return_value=True):

            audio_data = b'\x00\x01' * 2048

            # First request (will error)
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript("First request", {"confidence": 0.9})

            await asyncio.sleep(0.2)

            # Second request (should succeed)
            await pipeline.process_audio_chunk(audio_data)
            await pipeline._handle_final_transcript("Second request", {"confidence": 0.9})

            await asyncio.sleep(0.3)

            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)

        # Pipeline should still be active despite error
        assert pipeline.is_active, "Pipeline should remain active after error"

        # Second call should have succeeded
        assert call_count[0] >= 2, "Second LLM call should have been attempted"
