"""Unit tests for Twilio-Voice Pipeline Bridge."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from voice_agent.telephony.twilio_voice_bridge import (
    BridgeConfig,
    TwilioVoiceBridge,
)
from voice_agent.telephony.twilio_integration import (
    CallMetadata as TwilioCallMetadata,
    CallStatus,
)
from voice_agent.core.voice_pipeline import LatencyMetrics


@pytest.fixture
def bridge_config():
    """Create a test bridge configuration."""
    return BridgeConfig(
        session_id="test_session_123",
        enable_audio_logging=False,
        audio_buffer_size=8192,
    )


@pytest.fixture
def mock_twilio():
    """Create a mock Twilio integration."""
    twilio = MagicMock()
    twilio.is_connected = True
    twilio.is_active = False
    twilio.on_call_started = MagicMock()
    twilio.on_call_ended = MagicMock()
    twilio.on_audio_received = MagicMock()
    twilio.on_error = MagicMock()
    twilio.send_audio = AsyncMock()
    twilio.get_call_metadata = MagicMock(return_value=None)
    return twilio


@pytest.fixture
def mock_orchestrator():
    """Create a mock Pipecat orchestrator."""
    orchestrator = MagicMock()
    orchestrator.start_session = AsyncMock()
    orchestrator.stop_session = AsyncMock()
    return orchestrator


@pytest.fixture
def mock_pipeline():
    """Create a mock voice pipeline."""
    pipeline = MagicMock()
    pipeline.is_active = True
    pipeline.process_audio_chunk = AsyncMock()
    pipeline.on_audio_generated = MagicMock()
    pipeline.on_latency_metric = MagicMock()
    pipeline.get_latency_metrics = MagicMock(return_value=LatencyMetrics())
    return pipeline


@pytest.fixture
def bridge(bridge_config, mock_twilio, mock_orchestrator):
    """Create a Twilio-Voice Bridge instance."""
    return TwilioVoiceBridge(
        config=bridge_config,
        twilio_integration=mock_twilio,
        orchestrator=mock_orchestrator,
    )


class TestBridgeConfig:
    """Tests for BridgeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BridgeConfig(session_id="test_123")
        
        assert config.session_id == "test_123"
        assert config.enable_audio_logging is False
        assert config.audio_buffer_size == 8192

    def test_custom_config(self):
        """Test custom configuration values."""
        config = BridgeConfig(
            session_id="custom_456",
            enable_audio_logging=True,
            audio_buffer_size=16384,
        )
        
        assert config.session_id == "custom_456"
        assert config.enable_audio_logging is True
        assert config.audio_buffer_size == 16384


class TestTwilioVoiceBridge:
    """Tests for TwilioVoiceBridge."""

    def test_initialization(self, bridge, bridge_config, mock_twilio, mock_orchestrator):
        """Test bridge initialization."""
        assert bridge.config == bridge_config
        assert bridge._twilio == mock_twilio
        assert bridge._orchestrator == mock_orchestrator
        assert not bridge.is_active
        assert bridge.pipeline is None

    @pytest.mark.asyncio
    async def test_start_bridge(self, bridge, mock_twilio):
        """Test starting the bridge."""
        await bridge.start()
        
        assert bridge.is_active
        
        # Verify callbacks were registered
        mock_twilio.on_call_started.assert_called_once()
        mock_twilio.on_call_ended.assert_called_once()
        mock_twilio.on_audio_received.assert_called_once()
        mock_twilio.on_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_bridge_already_active(self, bridge, mock_twilio):
        """Test starting bridge when already active."""
        await bridge.start()
        assert bridge.is_active
        
        # Start again
        await bridge.start()
        
        # Should still be active, callbacks registered only once
        assert bridge.is_active
        assert mock_twilio.on_call_started.call_count == 1

    @pytest.mark.asyncio
    async def test_stop_bridge(self, bridge):
        """Test stopping the bridge."""
        await bridge.start()
        assert bridge.is_active
        
        await bridge.stop()
        
        assert not bridge.is_active

    @pytest.mark.asyncio
    async def test_stop_bridge_with_active_pipeline(self, bridge, mock_orchestrator, mock_pipeline):
        """Test stopping bridge with active pipeline."""
        await bridge.start()
        bridge._pipeline = mock_pipeline
        
        await bridge.stop()
        
        assert not bridge.is_active
        assert bridge.pipeline is None
        mock_orchestrator.stop_session.assert_called_once_with(bridge.config.session_id)

    @pytest.mark.asyncio
    async def test_stop_bridge_already_stopped(self, bridge):
        """Test stopping bridge when already stopped."""
        assert not bridge.is_active
        
        await bridge.stop()
        
        assert not bridge.is_active

    @pytest.mark.asyncio
    async def test_handle_call_started(self, bridge, mock_orchestrator, mock_pipeline):
        """Test handling call started event."""
        await bridge.start()
        
        # Mock orchestrator to return pipeline
        mock_orchestrator.start_session.return_value = mock_pipeline
        
        # Create call metadata
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
            direction="inbound",
            from_city="San Francisco",
            from_state="CA",
            from_country="US",
        )
        
        # Trigger call started
        bridge._handle_call_started(call_metadata)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Verify session was started
        mock_orchestrator.start_session.assert_called_once()
        call_args = mock_orchestrator.start_session.call_args
        assert call_args.kwargs["session_id"] == bridge.config.session_id
        assert call_args.kwargs["caller_id"] == "+15551234567"
        
        # Verify pipeline callbacks were registered
        mock_pipeline.on_audio_generated.assert_called_once()
        mock_pipeline.on_latency_metric.assert_called_once()
        
        # Verify pipeline is stored
        assert bridge.pipeline == mock_pipeline

    @pytest.mark.asyncio
    async def test_handle_call_started_error(self, bridge, mock_orchestrator, mock_twilio):
        """Test handling call started with session creation error."""
        await bridge.start()
        
        # Mock orchestrator to raise error
        mock_orchestrator.start_session.side_effect = Exception("Session creation failed")
        
        # Set up error callback
        error_callback = MagicMock()
        mock_twilio._on_error = error_callback
        
        # Create call metadata
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        
        # Trigger call started
        bridge._handle_call_started(call_metadata)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Verify error callback was called
        error_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_call_ended(self, bridge, mock_orchestrator, mock_pipeline):
        """Test handling call ended event."""
        await bridge.start()
        bridge._pipeline = mock_pipeline
        
        # Mock stop_session to return summary
        mock_orchestrator.stop_session.return_value = {
            "session_id": bridge.config.session_id,
            "turn_count": 5,
            "duration_seconds": 120.5,
        }
        
        # Create call metadata
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
            status=CallStatus.COMPLETED,
            duration_seconds=120.5,
        )
        
        # Trigger call ended
        bridge._handle_call_ended(call_metadata)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Verify session was stopped
        mock_orchestrator.stop_session.assert_called_once_with(
            session_id=bridge.config.session_id,
            persist=True,
        )
        
        # Verify pipeline is cleared
        assert bridge.pipeline is None

    @pytest.mark.asyncio
    async def test_handle_twilio_audio(self, bridge, mock_pipeline):
        """Test handling incoming audio from Twilio."""
        await bridge.start()
        bridge._pipeline = mock_pipeline
        
        # Send audio
        audio_data = b"test_audio_data"
        bridge._handle_twilio_audio(audio_data)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Verify audio was routed to pipeline
        mock_pipeline.process_audio_chunk.assert_called_once_with(audio_data)

    @pytest.mark.asyncio
    async def test_handle_twilio_audio_no_pipeline(self, bridge):
        """Test handling audio when pipeline is not active."""
        await bridge.start()
        
        # Send audio without pipeline
        audio_data = b"test_audio_data"
        bridge._handle_twilio_audio(audio_data)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Should not raise error, just drop audio

    @pytest.mark.asyncio
    async def test_handle_twilio_audio_with_logging(self, bridge_config, mock_twilio, mock_orchestrator, mock_pipeline):
        """Test handling audio with logging enabled."""
        bridge_config.enable_audio_logging = True
        bridge = TwilioVoiceBridge(bridge_config, mock_twilio, mock_orchestrator)
        
        await bridge.start()
        bridge._pipeline = mock_pipeline
        
        # Send audio
        audio_data = b"test_audio_data"
        
        with patch("voice_agent.telephony.twilio_voice_bridge.logger") as mock_logger:
            bridge._handle_twilio_audio(audio_data)
            
            # Give async task time to execute
            await asyncio.sleep(0.1)
            
            # Verify logging occurred
            mock_logger.debug.assert_called()

    @pytest.mark.asyncio
    async def test_handle_pipeline_audio(self, bridge, mock_twilio):
        """Test handling generated audio from pipeline."""
        await bridge.start()
        
        # Generate audio
        audio_data = b"generated_audio_data"
        bridge._handle_pipeline_audio(audio_data)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Verify audio was sent to Twilio
        mock_twilio.send_audio.assert_called_once_with(audio_data)

    @pytest.mark.asyncio
    async def test_handle_pipeline_audio_not_connected(self, bridge, mock_twilio):
        """Test handling pipeline audio when Twilio not connected."""
        await bridge.start()
        mock_twilio.is_connected = False
        
        # Generate audio
        audio_data = b"generated_audio_data"
        bridge._handle_pipeline_audio(audio_data)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Should not send audio
        mock_twilio.send_audio.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_pipeline_audio_with_logging(self, bridge_config, mock_twilio, mock_orchestrator):
        """Test handling pipeline audio with logging enabled."""
        bridge_config.enable_audio_logging = True
        bridge = TwilioVoiceBridge(bridge_config, mock_twilio, mock_orchestrator)
        
        await bridge.start()
        
        # Generate audio
        audio_data = b"generated_audio_data"
        
        with patch("voice_agent.telephony.twilio_voice_bridge.logger") as mock_logger:
            bridge._handle_pipeline_audio(audio_data)
            
            # Give async task time to execute
            await asyncio.sleep(0.1)
            
            # Verify logging occurred
            mock_logger.debug.assert_called()

    def test_handle_latency_metrics(self, bridge):
        """Test handling latency metrics from pipeline."""
        metrics = LatencyMetrics(
            stt_latency_ms=50.0,
            llm_ttft_ms=100.0,
            llm_total_ms=200.0,
            tts_ttfb_ms=75.0,
            tts_total_ms=150.0,
            end_to_end_ms=500.0,
        )
        
        with patch("voice_agent.telephony.twilio_voice_bridge.logger") as mock_logger:
            bridge._handle_latency_metrics(metrics)
            
            # Verify metrics were logged
            mock_logger.info.assert_called_once()
            # Check that the call was made with the expected format string and values
            call_args = mock_logger.info.call_args
            assert call_args is not None
            # The first arg is the format string, subsequent args are the values
            assert "Pipeline latency metrics" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_handle_twilio_error(self, bridge, mock_orchestrator, mock_pipeline):
        """Test handling error from Twilio."""
        await bridge.start()
        bridge._pipeline = mock_pipeline
        
        error = Exception("Twilio connection lost")
        bridge._handle_twilio_error(error)
        
        # Give async task time to execute
        await asyncio.sleep(0.1)
        
        # Verify session stop was attempted
        mock_orchestrator.stop_session.assert_called_once()

    def test_get_session_id(self, bridge, bridge_config):
        """Test getting session ID."""
        assert bridge.get_session_id() == bridge_config.session_id

    def test_get_call_metadata(self, bridge, mock_twilio):
        """Test getting call metadata."""
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        mock_twilio.get_call_metadata.return_value = call_metadata
        
        result = bridge.get_call_metadata()
        
        assert result == call_metadata
        mock_twilio.get_call_metadata.assert_called_once()

    def test_get_pipeline_metrics(self, bridge, mock_pipeline):
        """Test getting pipeline metrics."""
        metrics = LatencyMetrics(end_to_end_ms=500.0)
        mock_pipeline.get_latency_metrics.return_value = metrics
        
        bridge._pipeline = mock_pipeline
        result = bridge.get_pipeline_metrics()
        
        assert result == metrics
        mock_pipeline.get_latency_metrics.assert_called_once()

    def test_get_pipeline_metrics_no_pipeline(self, bridge):
        """Test getting metrics when no pipeline is active."""
        result = bridge.get_pipeline_metrics()
        
        assert result is None


class TestTwilioVoiceBridgeIntegration:
    """Integration tests for Twilio-Voice Bridge."""

    @pytest.mark.asyncio
    async def test_full_call_lifecycle(self, bridge, mock_orchestrator, mock_pipeline, mock_twilio):
        """Test complete call lifecycle from start to end."""
        await bridge.start()
        
        # Mock orchestrator to return pipeline
        mock_orchestrator.start_session.return_value = mock_pipeline
        mock_orchestrator.stop_session.return_value = {
            "session_id": bridge.config.session_id,
            "turn_count": 3,
            "duration_seconds": 60.0,
        }
        
        # 1. Call started
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        bridge._handle_call_started(call_metadata)
        await asyncio.sleep(0.1)
        
        # Verify session started
        assert bridge.pipeline == mock_pipeline
        mock_orchestrator.start_session.assert_called_once()
        
        # 2. Audio exchange
        # Incoming audio from Twilio
        incoming_audio = b"incoming_audio"
        bridge._handle_twilio_audio(incoming_audio)
        await asyncio.sleep(0.1)
        mock_pipeline.process_audio_chunk.assert_called_with(incoming_audio)
        
        # Outgoing audio from pipeline
        outgoing_audio = b"outgoing_audio"
        bridge._handle_pipeline_audio(outgoing_audio)
        await asyncio.sleep(0.1)
        mock_twilio.send_audio.assert_called_with(outgoing_audio)
        
        # 3. Call ended
        call_metadata.status = CallStatus.COMPLETED
        call_metadata.duration_seconds = 60.0
        bridge._handle_call_ended(call_metadata)
        await asyncio.sleep(0.1)
        
        # Verify session stopped
        assert bridge.pipeline is None
        mock_orchestrator.stop_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_bidirectional_audio_streaming(self, bridge, mock_orchestrator, mock_pipeline, mock_twilio):
        """Test bidirectional audio streaming."""
        await bridge.start()
        mock_orchestrator.start_session.return_value = mock_pipeline
        
        # Start call
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        bridge._handle_call_started(call_metadata)
        await asyncio.sleep(0.1)
        
        # Stream multiple audio chunks in both directions
        for i in range(5):
            # Twilio -> Pipeline
            incoming = f"incoming_{i}".encode()
            bridge._handle_twilio_audio(incoming)
            
            # Pipeline -> Twilio
            outgoing = f"outgoing_{i}".encode()
            bridge._handle_pipeline_audio(outgoing)
        
        await asyncio.sleep(0.2)
        
        # Verify all chunks were processed
        assert mock_pipeline.process_audio_chunk.call_count == 5
        assert mock_twilio.send_audio.call_count == 5

    @pytest.mark.asyncio
    async def test_error_recovery(self, bridge, mock_orchestrator, mock_pipeline, mock_twilio):
        """Test error recovery during call."""
        await bridge.start()
        mock_orchestrator.start_session.return_value = mock_pipeline
        
        # Start call
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        bridge._handle_call_started(call_metadata)
        await asyncio.sleep(0.1)
        
        # Simulate error
        error = Exception("Connection error")
        bridge._handle_twilio_error(error)
        await asyncio.sleep(0.1)
        
        # Verify session was stopped
        mock_orchestrator.stop_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_audio_processing(self, bridge, mock_orchestrator, mock_pipeline, mock_twilio):
        """Test concurrent audio processing doesn't cause issues."""
        await bridge.start()
        mock_orchestrator.start_session.return_value = mock_pipeline
        
        # Start call
        call_metadata = TwilioCallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        bridge._handle_call_started(call_metadata)
        await asyncio.sleep(0.1)
        
        # Send audio concurrently
        tasks = []
        for i in range(10):
            audio = f"audio_{i}".encode()
            tasks.append(asyncio.create_task(bridge._route_audio_to_pipeline(audio)))
        
        # Wait for all tasks
        await asyncio.gather(*tasks)
        
        # Verify all audio was processed
        assert mock_pipeline.process_audio_chunk.call_count == 10
