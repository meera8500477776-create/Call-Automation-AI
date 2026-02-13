"""
Unit tests for Voice Pipeline orchestrator.

Tests the VoicePipeline class that coordinates STT, LLM, TTS, and VAD
components for real-time voice processing with barge-in support.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock, PropertyMock
from typing import AsyncIterator

from src.voice_agent.core.voice_pipeline import (
    VoicePipeline,
    PipelineConfig,
    PipelineState,
    LatencyMetrics,
    StreamingConfig,
)
from src.voice_agent.services.stt_service import STTConfig
from src.voice_agent.services.llm_service import (
    LLMService,
    LLMResponse,
    Intent,
    ConversationContext,
    LLMConfig,
)
from src.voice_agent.services.vad_module import VADConfig, VADResult
from src.voice_agent.services.elevenlabs_tts_service import VoiceConfig


class MockLLMService(LLMService):
    """Mock LLM service for testing."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.generate_response_mock = AsyncMock()
    
    async def generate_response(
        self,
        transcript: str,
        context: ConversationContext,
    ) -> LLMResponse:
        """Mock generate_response."""
        if self.generate_response_mock:
            return await self.generate_response_mock(transcript, context)
        
        # Default response
        intent = self.classify_intent(transcript)
        return LLMResponse(
            text="This is a test response.",
            intent=intent,
            parameters={},
            confidence=0.9,
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
def pipeline_config(stt_config, llm_service, vad_config, voice_config):
    """Create pipeline configuration for testing."""
    return PipelineConfig(
        session_id="test_session_123",
        stt_config=stt_config,
        llm_service=llm_service,
        vad_config=vad_config,
        voice_config=voice_config,
        enable_barge_in=True,
    )


@pytest.fixture
def pipeline(pipeline_config):
    """Create voice pipeline for testing."""
    return VoicePipeline(pipeline_config)


class TestVoicePipelineInitialization:
    """Test voice pipeline initialization."""
    
    def test_pipeline_creation(self, pipeline):
        """Test pipeline is created with correct initial state."""
        assert pipeline.state == PipelineState.IDLE
        assert not pipeline.is_active
        assert not pipeline.is_agent_speaking
    
    def test_pipeline_config_stored(self, pipeline, pipeline_config):
        """Test pipeline stores configuration."""
        assert pipeline.config == pipeline_config
        assert pipeline.config.session_id == "test_session_123"
        assert pipeline.config.enable_barge_in is True
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization connects all components."""
        # Mock component connections
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock) as mock_connect, \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            
            await pipeline.initialize()
            
            assert pipeline.state == PipelineState.ACTIVE
            assert pipeline.is_active
            mock_connect.assert_called_once_with("test_session_123")
            mock_tts.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization_failure(self, pipeline):
        """Test pipeline handles initialization failure."""
        # Mock STT connection failure
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = ConnectionError("Connection failed")
            
            with pytest.raises(RuntimeError, match="Pipeline initialization failed"):
                await pipeline.initialize()
            
            assert pipeline.state == PipelineState.ERROR
    
    @pytest.mark.asyncio
    async def test_pipeline_double_initialization(self, pipeline):
        """Test pipeline handles double initialization gracefully."""
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock), \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            
            await pipeline.initialize()
            assert pipeline.state == PipelineState.ACTIVE
            
            # Second initialization should be no-op
            await pipeline.initialize()
            assert pipeline.state == PipelineState.ACTIVE


class TestVoicePipelineCallbacks:
    """Test voice pipeline callback registration."""
    
    def test_on_transcript_callback(self, pipeline):
        """Test transcript callback registration."""
        callback = Mock()
        pipeline.on_transcript(callback)
        assert pipeline._transcript_callback == callback
    
    def test_on_llm_response_callback(self, pipeline):
        """Test LLM response callback registration."""
        callback = Mock()
        pipeline.on_llm_response(callback)
        assert pipeline._llm_response_callback == callback
    
    def test_on_audio_generated_callback(self, pipeline):
        """Test audio generated callback registration."""
        callback = Mock()
        pipeline.on_audio_generated(callback)
        assert pipeline._audio_generated_callback == callback


class TestVoicePipelineAudioProcessing:
    """Test voice pipeline audio processing."""
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk_not_active(self, pipeline):
        """Test processing audio when pipeline is not active raises error."""
        audio_data = b'\x00\x01' * 1000
        
        with pytest.raises(RuntimeError, match="Pipeline not active"):
            await pipeline.process_audio_chunk(audio_data)
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk_empty_data(self, pipeline):
        """Test processing empty audio data raises error."""
        # Set pipeline to active state
        pipeline._state = PipelineState.ACTIVE
        
        with pytest.raises(ValueError, match="Audio data cannot be empty"):
            await pipeline.process_audio_chunk(b'')
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk_success(self, pipeline):
        """Test successful audio chunk processing."""
        # Initialize pipeline
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock), \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            await pipeline.initialize()
        
        # Mock VAD and STT - need to patch is_connected property
        # Create audio data larger than chunk size to trigger processing
        chunk_size = pipeline._streaming_config.audio_chunk_size
        audio_data = b'\x00\x01' * (chunk_size // 2 + 100)
        vad_result = VADResult(is_speech=True, confidence=0.8, timestamp=1.0)
        
        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock) as mock_send, \
             patch.object(type(pipeline._stt_service), 'is_connected', new_callable=PropertyMock, return_value=True):
            
            await pipeline.process_audio_chunk(audio_data)
            
            # Should have sent at least one chunk (audio is buffered and processed in chunks)
            assert mock_send.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_process_audio_chunk_with_barge_in(self, pipeline):
        """Test audio processing triggers barge-in when agent is speaking."""
        # Initialize pipeline
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock), \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            await pipeline.initialize()
        
        # Set agent as speaking
        pipeline._is_agent_speaking = True
        pipeline._current_tts_task = Mock()
        pipeline._current_tts_task.done.return_value = False
        pipeline._current_tts_task.cancel = Mock()
        
        # Mock VAD detecting speech
        # Create audio data larger than chunk size to trigger processing
        chunk_size = pipeline._streaming_config.audio_chunk_size
        audio_data = b'\x00\x01' * (chunk_size // 2 + 100)
        vad_result = VADResult(is_speech=True, confidence=0.9, timestamp=1.0)
        
        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected', new_callable=PropertyMock, return_value=True):
            
            await pipeline.process_audio_chunk(audio_data)
            
            # Barge-in should cancel TTS (at least once for the chunks processed)
            assert pipeline._current_tts_task.cancel.call_count >= 1


class TestVoicePipelineTranscriptHandling:
    """Test voice pipeline transcript handling."""
    
    @pytest.mark.asyncio
    async def test_handle_interim_transcript(self, pipeline):
        """Test interim transcript handling."""
        callback = Mock()
        pipeline.on_transcript(callback)
        
        await pipeline._handle_interim_transcript("Hello", {"confidence": 0.7})
        
        callback.assert_called_once_with("Hello", False)
    
    @pytest.mark.asyncio
    async def test_handle_final_transcript(self, pipeline):
        """Test final transcript handling triggers LLM processing."""
        callback = Mock()
        pipeline.on_transcript(callback)
        
        # Mock LLM processing
        with patch.object(pipeline, '_process_llm', new_callable=AsyncMock) as mock_llm:
            await pipeline._handle_final_transcript("Hello there", {"confidence": 0.9})
            
            callback.assert_called_once_with("Hello there", True)
            mock_llm.assert_called_once_with("Hello there")
    
    @pytest.mark.asyncio
    async def test_handle_final_transcript_empty(self, pipeline):
        """Test empty final transcript does not trigger LLM."""
        with patch.object(pipeline, '_process_llm', new_callable=AsyncMock) as mock_llm:
            await pipeline._handle_final_transcript("", {"confidence": 0.0})
            
            mock_llm.assert_not_called()


class TestVoicePipelineLLMProcessing:
    """Test voice pipeline LLM processing."""
    
    @pytest.mark.asyncio
    async def test_process_llm_success(self, pipeline, llm_service):
        """Test successful LLM processing."""
        # Setup mock response
        llm_response = LLMResponse(
            text="I can help you with that.",
            intent=Intent(name="general_inquiry", confidence=0.9),
            parameters={"topic": "help"},
            confidence=0.9,
        )
        llm_service.generate_response_mock.return_value = llm_response
        
        # Mock TTS generation
        with patch.object(pipeline, '_generate_audio', new_callable=AsyncMock) as mock_audio:
            await pipeline._process_llm("I need help")
            
            # Check context updated
            assert len(pipeline._context.conversation_history) == 2
            assert pipeline._context.conversation_history[0]["role"] == "user"
            assert pipeline._context.conversation_history[0]["content"] == "I need help"
            assert pipeline._context.conversation_history[1]["role"] == "assistant"
            
            # Check audio generation called
            mock_audio.assert_called_once_with(llm_response)
    
    @pytest.mark.asyncio
    async def test_process_llm_with_callback(self, pipeline, llm_service):
        """Test LLM processing invokes callback."""
        callback = Mock()
        pipeline.on_llm_response(callback)
        
        llm_response = LLMResponse(
            text="Test response",
            intent=Intent(name="general_inquiry", confidence=0.9),
            confidence=0.9,
        )
        llm_service.generate_response_mock.return_value = llm_response
        
        with patch.object(pipeline, '_generate_audio', new_callable=AsyncMock):
            await pipeline._process_llm("Test input")
            
            callback.assert_called_once_with(llm_response)
    
    @pytest.mark.asyncio
    async def test_process_llm_updates_parameters(self, pipeline, llm_service):
        """Test LLM processing updates extracted parameters."""
        llm_response = LLMResponse(
            text="Your appointment is scheduled.",
            intent=Intent(name="schedule_appointment", confidence=0.9),
            parameters={"date": "tomorrow", "time": "2pm"},
            confidence=0.9,
        )
        llm_service.generate_response_mock.return_value = llm_response
        
        with patch.object(pipeline, '_generate_audio', new_callable=AsyncMock):
            await pipeline._process_llm("Book appointment tomorrow at 2pm")
            
            assert pipeline._context.extracted_parameters["date"] == "tomorrow"
            assert pipeline._context.extracted_parameters["time"] == "2pm"


class TestVoicePipelineAudioGeneration:
    """Test voice pipeline audio generation."""
    
    @pytest.mark.asyncio
    async def test_generate_audio_success(self, pipeline):
        """Test successful audio generation."""
        # Initialize TTS service
        pipeline._tts_service = Mock()
        pipeline._tts_service.apply_emotional_context = AsyncMock(return_value="Enhanced text")
        
        async def mock_stream():
            yield b'audio_chunk_1'
            yield b'audio_chunk_2'
        
        pipeline._tts_service.synthesize_streaming = Mock(return_value=mock_stream())
        
        # Setup callback
        callback = Mock()
        pipeline.on_audio_generated(callback)
        
        llm_response = LLMResponse(
            text="Hello there",
            intent=Intent(name="general_inquiry", confidence=0.9),
            emotional_tags=["friendly"],
            confidence=0.9,
        )
        
        await pipeline._generate_audio(llm_response)
        
        # Check audio chunks were sent to callback
        assert callback.call_count == 2
        callback.assert_any_call(b'audio_chunk_1')
        callback.assert_any_call(b'audio_chunk_2')
    
    @pytest.mark.asyncio
    async def test_generate_audio_no_tts_service(self, pipeline):
        """Test audio generation handles missing TTS service."""
        llm_response = LLMResponse(
            text="Test",
            intent=Intent(name="general_inquiry", confidence=0.9),
            confidence=0.9,
        )
        
        # Should not raise error, just log
        await pipeline._generate_audio(llm_response)
    
    @pytest.mark.asyncio
    async def test_generate_audio_sets_speaking_flag(self, pipeline):
        """Test audio generation sets and clears speaking flag."""
        pipeline._tts_service = Mock()
        
        async def mock_stream():
            # Check flag is set during streaming
            assert pipeline._is_agent_speaking
            yield b'audio_chunk'
        
        pipeline._tts_service.synthesize_streaming = Mock(return_value=mock_stream())
        pipeline._tts_service.apply_emotional_context = AsyncMock(return_value="text")
        
        llm_response = LLMResponse(
            text="Test",
            intent=Intent(name="general_inquiry", confidence=0.9),
            confidence=0.9,
        )
        
        assert not pipeline._is_agent_speaking
        await pipeline._generate_audio(llm_response)
        assert not pipeline._is_agent_speaking


class TestVoicePipelineBargeIn:
    """Test voice pipeline barge-in functionality."""
    
    def test_enable_barge_in(self, pipeline):
        """Test enabling barge-in."""
        pipeline.enable_barge_in(True)
        assert pipeline._barge_in_enabled is True
        
        pipeline.enable_barge_in(False)
        assert pipeline._barge_in_enabled is False
    
    @pytest.mark.asyncio
    async def test_handle_barge_in_cancels_tts(self, pipeline):
        """Test barge-in cancels ongoing TTS."""
        # Create mock TTS task
        mock_task = Mock()
        mock_task.done.return_value = False
        mock_task.cancel = Mock()
        
        pipeline._current_tts_task = mock_task
        pipeline._is_agent_speaking = True
        
        await pipeline._handle_barge_in()
        
        mock_task.cancel.assert_called_once()
        assert not pipeline._is_agent_speaking
    
    @pytest.mark.asyncio
    async def test_handle_barge_in_no_active_tts(self, pipeline):
        """Test barge-in with no active TTS is safe."""
        pipeline._current_tts_task = None
        pipeline._is_agent_speaking = False
        
        # Should not raise error
        await pipeline._handle_barge_in()


class TestVoicePipelineShutdown:
    """Test voice pipeline shutdown."""
    
    @pytest.mark.asyncio
    async def test_shutdown_success(self, pipeline):
        """Test successful pipeline shutdown."""
        # Initialize first
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock), \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            await pipeline.initialize()
        
        # Mock disconnect and is_connected property
        with patch.object(pipeline._stt_service, 'disconnect', new_callable=AsyncMock) as mock_disconnect, \
             patch.object(type(pipeline._stt_service), 'is_connected', new_callable=PropertyMock, return_value=True):
            
            await pipeline.shutdown()
            
            assert pipeline.state == PipelineState.IDLE
            mock_disconnect.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_cancels_tts(self, pipeline):
        """Test shutdown cancels ongoing TTS."""
        # Create a real asyncio task that we can cancel
        async def dummy_tts():
            await asyncio.sleep(10)
        
        mock_task = asyncio.create_task(dummy_tts())
        
        pipeline._current_tts_task = mock_task
        pipeline._state = PipelineState.ACTIVE
        
        with patch.object(pipeline._stt_service, 'disconnect', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected', new_callable=PropertyMock, return_value=False):
            
            await pipeline.shutdown()
            
            # Task should be cancelled
            assert mock_task.cancelled()
    
    @pytest.mark.asyncio
    async def test_shutdown_already_idle(self, pipeline):
        """Test shutdown when already idle is safe."""
        assert pipeline.state == PipelineState.IDLE
        
        await pipeline.shutdown()
        
        assert pipeline.state == PipelineState.IDLE


class TestVoicePipelineContext:
    """Test voice pipeline context management."""
    
    def test_get_context(self, pipeline):
        """Test getting conversation context."""
        context = pipeline.get_context()
        
        assert context.session_id == "test_session_123"
        assert context.conversation_history == []
        assert context.extracted_parameters == {}
    
    def test_update_context(self, pipeline):
        """Test updating context fields."""
        pipeline.update_context(
            current_intent="schedule_appointment",
            extracted_parameters={"date": "tomorrow"},
        )
        
        context = pipeline.get_context()
        assert context.current_intent == "schedule_appointment"
        assert context.extracted_parameters == {"date": "tomorrow"}
    
    def test_update_context_invalid_field(self, pipeline):
        """Test updating invalid context field is ignored."""
        # Should not raise error
        pipeline.update_context(invalid_field="value")
        
        context = pipeline.get_context()
        assert not hasattr(context, "invalid_field")


class TestVoicePipelineStreamingOptimizations:
    """Test voice pipeline streaming optimizations."""
    
    def test_streaming_config_defaults(self, pipeline):
        """Test default streaming configuration."""
        assert pipeline._streaming_config.enable_concurrent_processing is True
        assert pipeline._streaming_config.max_audio_buffer_size == 32768
        assert pipeline._streaming_config.audio_chunk_size == 8192
    
    def test_custom_streaming_config(self, pipeline_config):
        """Test custom streaming configuration."""
        streaming_config = StreamingConfig(
            enable_concurrent_processing=False,
            max_audio_buffer_size=16384,
            audio_chunk_size=4096,
        )
        pipeline_config.streaming_config = streaming_config
        
        pipeline = VoicePipeline(pipeline_config)
        
        assert pipeline._streaming_config.enable_concurrent_processing is False
        assert pipeline._streaming_config.max_audio_buffer_size == 16384
        assert pipeline._streaming_config.audio_chunk_size == 4096
    
    def test_latency_metrics_initialization(self, pipeline):
        """Test latency metrics are initialized."""
        metrics = pipeline.get_latency_metrics()
        
        assert isinstance(metrics, LatencyMetrics)
        assert metrics.stt_latency_ms == 0.0
        assert metrics.llm_ttft_ms == 0.0
        assert metrics.llm_total_ms == 0.0
        assert metrics.tts_ttfb_ms == 0.0
        assert metrics.tts_total_ms == 0.0
        assert metrics.end_to_end_ms == 0.0
    
    def test_latency_metrics_to_dict(self):
        """Test latency metrics conversion to dictionary."""
        metrics = LatencyMetrics(
            stt_latency_ms=50.0,
            llm_ttft_ms=100.0,
            llm_total_ms=200.0,
            tts_ttfb_ms=75.0,
            tts_total_ms=150.0,
            end_to_end_ms=500.0,
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["stt_latency_ms"] == 50.0
        assert metrics_dict["llm_ttft_ms"] == 100.0
        assert metrics_dict["llm_total_ms"] == 200.0
        assert metrics_dict["tts_ttfb_ms"] == 75.0
        assert metrics_dict["tts_total_ms"] == 150.0
        assert metrics_dict["end_to_end_ms"] == 500.0
    
    def test_latency_callback_registration(self, pipeline):
        """Test latency callback registration."""
        callback = Mock()
        pipeline.on_latency_metric(callback)
        assert pipeline._latency_callback == callback
    
    def test_start_latency_timer(self, pipeline):
        """Test starting latency timer."""
        pipeline._start_latency_timer("test_stage")
        
        assert "test_stage" in pipeline._stage_timers
        assert isinstance(pipeline._stage_timers["test_stage"], float)
    
    def test_end_latency_timer(self, pipeline):
        """Test ending latency timer."""
        pipeline._start_latency_timer("test_stage")
        
        import time
        time.sleep(0.01)  # Small delay
        
        duration = pipeline._end_latency_timer("test_stage")
        
        assert duration > 0
        assert "test_stage" not in pipeline._stage_timers
    
    def test_end_latency_timer_no_start(self, pipeline):
        """Test ending timer that was never started."""
        duration = pipeline._end_latency_timer("nonexistent")
        
        assert duration == 0.0
    
    def test_record_latency_stt(self, pipeline):
        """Test recording STT latency."""
        pipeline._record_latency("stt", 50.0)
        
        metrics = pipeline.get_latency_metrics()
        assert metrics.stt_latency_ms == 50.0
    
    def test_record_latency_llm(self, pipeline):
        """Test recording LLM latency."""
        pipeline._record_latency("llm_ttft", 100.0)
        pipeline._record_latency("llm_total", 200.0)
        
        metrics = pipeline.get_latency_metrics()
        assert metrics.llm_ttft_ms == 100.0
        assert metrics.llm_total_ms == 200.0
    
    def test_record_latency_tts(self, pipeline):
        """Test recording TTS latency."""
        pipeline._record_latency("tts_ttfb", 75.0)
        pipeline._record_latency("tts_total", 150.0)
        
        metrics = pipeline.get_latency_metrics()
        assert metrics.tts_ttfb_ms == 75.0
        assert metrics.tts_total_ms == 150.0
    
    def test_record_latency_end_to_end(self, pipeline):
        """Test recording end-to-end latency."""
        pipeline._record_latency("end_to_end", 500.0)
        
        metrics = pipeline.get_latency_metrics()
        assert metrics.end_to_end_ms == 500.0
    
    def test_check_backpressure_empty_queue(self, pipeline):
        """Test backpressure check with empty queue."""
        assert not pipeline._check_backpressure()
    
    @pytest.mark.asyncio
    async def test_check_backpressure_threshold(self, pipeline):
        """Test backpressure detection at threshold."""
        # Fill queue to 80% capacity
        max_size = pipeline._audio_queue.maxsize
        threshold_count = int(max_size * 0.8)
        
        for _ in range(threshold_count):
            await pipeline._audio_queue.put(b'data')
        
        assert pipeline._check_backpressure()
        
        # Clean up queue
        while not pipeline._audio_queue.empty():
            await pipeline._audio_queue.get()
    
    @pytest.mark.asyncio
    async def test_audio_buffering_with_chunks(self, pipeline):
        """Test audio buffering processes data in chunks."""
        # Initialize pipeline
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock), \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            await pipeline.initialize()
        
        # Create audio data larger than chunk size
        chunk_size = pipeline._streaming_config.audio_chunk_size
        audio_data = b'\x00\x01' * (chunk_size + 100)
        
        vad_result = VADResult(is_speech=False, confidence=0.5, timestamp=1.0)
        
        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock) as mock_send, \
             patch.object(type(pipeline._stt_service), 'is_connected', new_callable=PropertyMock, return_value=True):
            
            await pipeline.process_audio_chunk(audio_data)
            
            # Should have sent at least one chunk
            assert mock_send.call_count >= 1
            
            # Buffer should contain remaining data
            assert len(pipeline._audio_buffer) < chunk_size
    
    @pytest.mark.asyncio
    async def test_audio_buffer_overflow_protection(self, pipeline):
        """Test audio buffer overflow protection."""
        # Initialize pipeline
        with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock), \
             patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts:
            
            mock_tts.return_value = Mock()
            await pipeline.initialize()
        
        # Fill buffer beyond max size
        max_size = pipeline._streaming_config.max_audio_buffer_size
        large_audio = b'\x00\x01' * (max_size + 1000)
        
        # Add data directly to buffer to simulate overflow
        pipeline._audio_buffer.extend(large_audio)
        
        vad_result = VADResult(is_speech=False, confidence=0.5, timestamp=1.0)
        
        with patch.object(pipeline._vad_module, 'process_audio', return_value=vad_result), \
             patch.object(pipeline._stt_service, 'send_audio', new_callable=AsyncMock), \
             patch.object(type(pipeline._stt_service), 'is_connected', new_callable=PropertyMock, return_value=True):
            
            await pipeline.process_audio_chunk(b'\x00\x01' * 100)
            
            # Buffer should be capped at max size
            assert len(pipeline._audio_buffer) <= max_size
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_semaphores(self, pipeline):
        """Test concurrent processing uses semaphores."""
        assert pipeline._llm_semaphore._value == pipeline._streaming_config.max_concurrent_llm_requests
        assert pipeline._tts_semaphore._value == pipeline._streaming_config.max_concurrent_tts_requests
    
    @pytest.mark.asyncio
    async def test_latency_tracking_in_llm_processing(self, pipeline, llm_service):
        """Test latency tracking during LLM processing."""
        llm_response = LLMResponse(
            text="Test response",
            intent=Intent(name="general_inquiry", confidence=0.9),
            confidence=0.9,
        )
        llm_service.generate_response_mock.return_value = llm_response
        
        with patch.object(pipeline, '_generate_audio', new_callable=AsyncMock):
            await pipeline._process_llm("Test input")
            
            # Check that LLM latency was recorded
            metrics = pipeline.get_latency_metrics()
            assert metrics.llm_ttft_ms > 0
            assert metrics.llm_total_ms > 0
    
    @pytest.mark.asyncio
    async def test_latency_callback_invoked(self, pipeline):
        """Test latency callback is invoked after processing."""
        callback = Mock()
        pipeline.on_latency_metric(callback)
        
        # Initialize TTS service
        pipeline._tts_service = Mock()
        
        async def mock_stream():
            yield b'audio_chunk'
        
        pipeline._tts_service.synthesize_streaming = Mock(return_value=mock_stream())
        pipeline._tts_service.apply_emotional_context = AsyncMock(return_value="text")
        
        # Start end-to-end timer
        pipeline._start_latency_timer("end_to_end")
        
        llm_response = LLMResponse(
            text="Test",
            intent=Intent(name="general_inquiry", confidence=0.9),
            confidence=0.9,
        )
        
        await pipeline._generate_audio(llm_response)
        
        # Callback should have been invoked with metrics
        callback.assert_called_once()
        args = callback.call_args[0]
        assert isinstance(args[0], LatencyMetrics)
