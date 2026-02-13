"""
Property-based tests for Voice Pipeline.

**Validates: Requirements 2.3**

This module contains property-based tests using Hypothesis to verify universal
properties of the voice pipeline, specifically testing that pipeline stages
execute in the correct order: VAD → STT → LLM → TTS.
"""

import pytest
import asyncio
from hypothesis import given, settings, strategies as st
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import List, Dict, Any

from src.voice_agent.core.voice_pipeline import (
    VoicePipeline,
    PipelineConfig,
    PipelineState,
)
from src.voice_agent.services.stt_service import STTConfig, ConnectionState
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


class StageOrderTracker:
    """Tracks the order of pipeline stage executions."""
    
    def __init__(self):
        self.stages: List[str] = []
        self.lock = asyncio.Lock()
    
    async def record_stage(self, stage: str):
        """Record a stage execution."""
        async with self.lock:
            self.stages.append(stage)
    
    def get_stages(self) -> List[str]:
        """Get recorded stages."""
        return self.stages.copy()
    
    def reset(self):
        """Reset recorded stages."""
        self.stages.clear()


# Hypothesis strategies for generating test data

@st.composite
def audio_chunk_strategy(draw):
    """Generate realistic audio chunks (16-bit PCM)."""
    # Audio chunks should be reasonable sizes
    # At 16kHz sample rate, VAD needs at least 512 samples (1024 bytes)
    # Generate chunks between 1KB and 4KB
    size = draw(st.integers(min_value=1024, max_value=4096))
    # Make size even for 16-bit samples
    size = size - (size % 2)
    # Generate random audio data using binary strategy
    return draw(st.binary(min_size=size, max_size=size))


@st.composite
def transcript_strategy(draw):
    """Generate realistic user transcripts."""
    transcript_templates = [
        "I want to schedule an appointment",
        "Can I book an appointment for tomorrow?",
        "I need to reschedule my appointment",
        "Please cancel my appointment",
        "What are your hours?",
        "Schedule me for Monday at 2pm",
        "Hello, I need help",
        "Check my appointments",
    ]
    return draw(st.sampled_from(transcript_templates))


def create_pipeline_config(llm_service: LLMService) -> PipelineConfig:
    """Create pipeline configuration for testing."""
    from src.voice_agent.core.voice_pipeline import StreamingConfig
    
    stt_config = STTConfig(
        api_key="test_api_key",
        model="nova-2",
        language="en-US",
    )
    
    vad_config = VADConfig(
        sample_rate=16000,
        threshold=0.5,
    )
    
    voice_config = VoiceConfig(
        provider="openai",
        voice_id="alloy",
    )
    
    # Use smaller chunk size for testing to ensure audio gets processed
    # But not too small - VAD needs at least 1024 bytes (512 samples at 16kHz)
    streaming_config = StreamingConfig(
        enable_concurrent_processing=True,
        max_audio_buffer_size=16384,
        audio_chunk_size=1024,  # Minimum size for VAD processing
        max_concurrent_llm_requests=1,
        max_concurrent_tts_requests=1,
    )
    
    return PipelineConfig(
        session_id="test_session_123",
        stt_config=stt_config,
        llm_service=llm_service,
        vad_config=vad_config,
        voice_config=voice_config,
        enable_barge_in=True,
        streaming_config=streaming_config,
    )


class TestVoicePipelineStageOrderingProperties:
    """Property-based tests for voice pipeline stage ordering."""
    
    @pytest.mark.property
    @pytest.mark.asyncio
    @settings(max_examples=50, deadline=5000)
    @given(audio_chunk_strategy(), transcript_strategy())
    async def test_pipeline_stage_ordering(self, audio_data, transcript):
        """
        Property 6: Pipeline stage ordering.
        
        For any audio input, pipeline stages should execute in order:
        VAD → STT → LLM → TTS
        
        This property verifies that the voice pipeline processes audio through
        the correct sequence of stages, ensuring proper data flow and
        maintaining the expected processing order.
        
        **Validates: Requirements 2.3**
        """
        # Create tracker to monitor stage execution order
        tracker = StageOrderTracker()
        
        # Create LLM service
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
        )
        llm_service = MockLLMService(llm_config)
        
        # Setup mock LLM response
        llm_response = LLMResponse(
            text="I can help you with that.",
            intent=Intent(name="general_inquiry", confidence=0.9),
            parameters={},
            confidence=0.9,
        )
        llm_service.generate_response_mock.return_value = llm_response
        
        # Create pipeline
        config = create_pipeline_config(llm_service)
        pipeline = VoicePipeline(config)
        
        try:
            # Mock WebSocket connection for STT
            mock_ws = MagicMock()
            mock_ws.send = Mock()
            
            # Mock TTS service
            mock_tts_service = Mock()
            
            async def mock_tts_stream(text):
                await tracker.record_stage("TTS")
                yield b'audio_chunk_1'
                yield b'audio_chunk_2'
            
            mock_tts_service.synthesize_streaming = Mock(return_value=mock_tts_stream("test"))
            mock_tts_service.apply_emotional_context = AsyncMock(return_value="test text")
            
            # Initialize pipeline with mocked components
            with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock) as mock_connect, \
                 patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts_factory:
                
                # Setup mocks
                async def setup_stt_connection(session_id):
                    pipeline._stt_service._connection = mock_ws
                    pipeline._stt_service._state = ConnectionState.CONNECTED
                
                mock_connect.side_effect = setup_stt_connection
                mock_tts_factory.return_value = mock_tts_service
                
                await pipeline.initialize()
            
            # Wrap pipeline methods to track execution order
            original_vad_process = pipeline._vad_module.process_audio
            
            def tracked_vad_process(audio_chunk):
                tracker.stages.append("VAD")
                return original_vad_process(audio_chunk)
            
            original_stt_send = pipeline._stt_service.send_audio
            
            async def tracked_stt_send(audio_chunk):
                await tracker.record_stage("STT")
                # Don't actually send to WebSocket, just record
                return None
            
            original_llm_generate = llm_service.generate_response
            
            async def tracked_llm_generate(transcript, context):
                await tracker.record_stage("LLM")
                return await original_llm_generate(transcript, context)
            
            # Apply tracking wrappers
            pipeline._vad_module.process_audio = tracked_vad_process
            pipeline._stt_service.send_audio = tracked_stt_send
            llm_service.generate_response = tracked_llm_generate
            
            # Process audio chunk
            await pipeline.process_audio_chunk(audio_data)
            
            # Trigger final transcript to start LLM processing
            await pipeline._handle_final_transcript(transcript, {"confidence": 0.9})
            
            # Wait for async processing to complete
            await asyncio.sleep(0.2)
            
            # Wait for any pending tasks
            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)
            
            # Wait for TTS to complete
            await asyncio.sleep(0.2)
            
            # Get recorded stages
            stages = tracker.get_stages()
            
            # Verify stages were executed
            assert len(stages) > 0, "No stages were recorded"
            
            # Find indices of each stage (first occurrence)
            vad_indices = [i for i, s in enumerate(stages) if s == "VAD"]
            stt_indices = [i for i, s in enumerate(stages) if s == "STT"]
            llm_indices = [i for i, s in enumerate(stages) if s == "LLM"]
            tts_indices = [i for i, s in enumerate(stages) if s == "TTS"]
            
            # Verify all stages were executed
            assert len(vad_indices) > 0, f"VAD stage not executed. Stages: {stages}"
            assert len(stt_indices) > 0, f"STT stage not executed. Stages: {stages}"
            assert len(llm_indices) > 0, f"LLM stage not executed. Stages: {stages}"
            assert len(tts_indices) > 0, f"TTS stage not executed. Stages: {stages}"
            
            # Verify ordering: VAD → STT → LLM → TTS
            # VAD should come before STT
            assert min(vad_indices) < min(stt_indices), \
                f"VAD should execute before STT. Stages: {stages}"
            
            # STT should come before LLM
            assert min(stt_indices) < min(llm_indices), \
                f"STT should execute before LLM. Stages: {stages}"
            
            # LLM should come before TTS
            assert min(llm_indices) < min(tts_indices), \
                f"LLM should execute before TTS. Stages: {stages}"
            
            # Overall ordering check
            first_vad = min(vad_indices)
            first_stt = min(stt_indices)
            first_llm = min(llm_indices)
            first_tts = min(tts_indices)
            
            assert first_vad < first_stt < first_llm < first_tts, \
                f"Pipeline stages not in correct order VAD→STT→LLM→TTS. " \
                f"Order: VAD@{first_vad}, STT@{first_stt}, LLM@{first_llm}, TTS@{first_tts}. " \
                f"Full stages: {stages}"
        
        finally:
            # Cleanup
            if pipeline.is_active:
                await pipeline.shutdown()

    
    @pytest.mark.property
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(
        st.lists(audio_chunk_strategy(), min_size=2, max_size=5),
        transcript_strategy()
    )
    async def test_pipeline_stage_ordering_multiple_chunks(self, audio_chunks, transcript):
        """
        Property 6 (extended): Pipeline stage ordering with multiple audio chunks.
        
        For multiple audio inputs, pipeline stages should maintain order:
        VAD → STT → LLM → TTS, even when processing multiple chunks.
        
        This verifies that the pipeline maintains correct ordering even with
        multiple audio chunks being processed.
        
        **Validates: Requirements 2.3**
        """
        # Create tracker to monitor stage execution order
        tracker = StageOrderTracker()
        
        # Create LLM service
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
        )
        llm_service = MockLLMService(llm_config)
        
        # Setup mock LLM response
        llm_response = LLMResponse(
            text="I can help you with that.",
            intent=Intent(name="general_inquiry", confidence=0.9),
            parameters={},
            confidence=0.9,
        )
        llm_service.generate_response_mock.return_value = llm_response
        
        # Create pipeline
        config = create_pipeline_config(llm_service)
        pipeline = VoicePipeline(config)
        
        try:
            # Mock WebSocket connection for STT
            mock_ws = MagicMock()
            mock_ws.send = Mock()
            
            # Mock TTS service
            mock_tts_service = Mock()
            
            async def mock_tts_stream(text):
                await tracker.record_stage("TTS")
                yield b'audio_chunk'
            
            mock_tts_service.synthesize_streaming = Mock(return_value=mock_tts_stream("test"))
            mock_tts_service.apply_emotional_context = AsyncMock(return_value="test text")
            
            # Initialize pipeline with mocked components
            with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock) as mock_connect, \
                 patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts_factory:
                
                # Setup mocks
                async def setup_stt_connection(session_id):
                    pipeline._stt_service._connection = mock_ws
                    pipeline._stt_service._state = ConnectionState.CONNECTED
                
                mock_connect.side_effect = setup_stt_connection
                mock_tts_factory.return_value = mock_tts_service
                
                await pipeline.initialize()
            
            # Wrap pipeline methods to track execution order
            original_vad_process = pipeline._vad_module.process_audio
            
            def tracked_vad_process(audio_chunk):
                tracker.stages.append("VAD")
                return original_vad_process(audio_chunk)
            
            original_stt_send = pipeline._stt_service.send_audio
            
            async def tracked_stt_send(audio_chunk):
                await tracker.record_stage("STT")
                # Don't actually send to WebSocket, just record
                return None
            
            original_llm_generate = llm_service.generate_response
            
            async def tracked_llm_generate(transcript, context):
                await tracker.record_stage("LLM")
                return await original_llm_generate(transcript, context)
            
            # Apply tracking wrappers
            pipeline._vad_module.process_audio = tracked_vad_process
            pipeline._stt_service.send_audio = tracked_stt_send
            llm_service.generate_response = tracked_llm_generate
            
            # Process multiple audio chunks
            for audio_data in audio_chunks:
                await pipeline.process_audio_chunk(audio_data)
                await asyncio.sleep(0.01)  # Small delay between chunks
            
            # Trigger final transcript to start LLM processing
            await pipeline._handle_final_transcript(transcript, {"confidence": 0.9})
            
            # Wait for async processing to complete
            await asyncio.sleep(0.2)
            
            # Wait for any pending tasks
            if pipeline._processing_tasks:
                await asyncio.gather(*pipeline._processing_tasks, return_exceptions=True)
            
            # Wait for TTS to complete
            await asyncio.sleep(0.2)
            
            # Get recorded stages
            stages = tracker.get_stages()
            
            # Verify stages were executed
            assert len(stages) > 0, "No stages were recorded"
            
            # Find indices of each stage
            vad_indices = [i for i, s in enumerate(stages) if s == "VAD"]
            stt_indices = [i for i, s in enumerate(stages) if s == "STT"]
            llm_indices = [i for i, s in enumerate(stages) if s == "LLM"]
            tts_indices = [i for i, s in enumerate(stages) if s == "TTS"]
            
            # Verify all stages were executed
            assert len(vad_indices) > 0, f"VAD stage not executed. Stages: {stages}"
            assert len(stt_indices) > 0, f"STT stage not executed. Stages: {stages}"
            assert len(llm_indices) > 0, f"LLM stage not executed. Stages: {stages}"
            assert len(tts_indices) > 0, f"TTS stage not executed. Stages: {stages}"
            
            # Verify ordering: first VAD before first STT before first LLM before first TTS
            first_vad = min(vad_indices)
            first_stt = min(stt_indices)
            first_llm = min(llm_indices)
            first_tts = min(tts_indices)
            
            assert first_vad < first_stt < first_llm < first_tts, \
                f"Pipeline stages not in correct order VAD→STT→LLM→TTS with multiple chunks. " \
                f"Order: VAD@{first_vad}, STT@{first_stt}, LLM@{first_llm}, TTS@{first_tts}. " \
                f"Full stages: {stages}"
        
        finally:
            # Cleanup
            if pipeline.is_active:
                await pipeline.shutdown()
    
    @pytest.mark.property
    @pytest.mark.asyncio
    @settings(max_examples=30, deadline=5000)
    @given(audio_chunk_strategy())
    async def test_vad_always_processes_before_stt(self, audio_data):
        """
        Property: VAD always processes audio before STT.
        
        For any audio input, VAD should process the audio before it's sent
        to the STT service. This ensures speech detection happens first.
        
        **Validates: Requirements 2.3, 2.4**
        """
        # Create tracker
        tracker = StageOrderTracker()
        
        # Create LLM service
        llm_config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=150,
        )
        llm_service = MockLLMService(llm_config)
        
        # Create pipeline
        config = create_pipeline_config(llm_service)
        pipeline = VoicePipeline(config)
        
        try:
            # Mock WebSocket connection for STT
            mock_ws = MagicMock()
            mock_ws.send = Mock()
            
            # Mock TTS service
            mock_tts_service = Mock()
            mock_tts_service.synthesize_streaming = Mock(return_value=AsyncMock())
            mock_tts_service.apply_emotional_context = AsyncMock(return_value="test text")
            
            # Initialize pipeline
            with patch.object(pipeline._stt_service, 'connect', new_callable=AsyncMock) as mock_connect, \
                 patch.object(pipeline._tts_factory, 'create_tts_service', new_callable=AsyncMock) as mock_tts_factory:
                
                # Setup mocks
                async def setup_stt_connection(session_id):
                    pipeline._stt_service._connection = mock_ws
                    pipeline._stt_service._state = ConnectionState.CONNECTED
                
                mock_connect.side_effect = setup_stt_connection
                mock_tts_factory.return_value = mock_tts_service
                
                await pipeline.initialize()
            
            # Wrap VAD and STT to track execution order
            original_vad_process = pipeline._vad_module.process_audio
            
            def tracked_vad_process(audio_chunk):
                tracker.stages.append("VAD")
                return original_vad_process(audio_chunk)
            
            original_stt_send = pipeline._stt_service.send_audio
            
            async def tracked_stt_send(audio_chunk):
                await tracker.record_stage("STT")
                # Don't actually send to WebSocket, just record
                return None
            
            pipeline._vad_module.process_audio = tracked_vad_process
            pipeline._stt_service.send_audio = tracked_stt_send
            
            # Process audio
            await pipeline.process_audio_chunk(audio_data)
            
            # Wait for processing
            await asyncio.sleep(0.1)
            
            # Get recorded stages
            stages = tracker.get_stages()
            
            # If both stages were executed, VAD should come before STT
            vad_indices = [i for i, s in enumerate(stages) if s == "VAD"]
            stt_indices = [i for i, s in enumerate(stages) if s == "STT"]
            
            if len(vad_indices) > 0 and len(stt_indices) > 0:
                assert min(vad_indices) < min(stt_indices), \
                    f"VAD should always process before STT. Stages: {stages}"
        
        finally:
            # Cleanup
            if pipeline.is_active:
                await pipeline.shutdown()
