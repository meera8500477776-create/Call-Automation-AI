"""
Unit tests for STT Service.

Tests WebSocket connection management, audio streaming, callback handlers,
and error handling for the Deepgram STT service wrapper.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.voice_agent.services.stt_service import (
    ConnectionState,
    STTConfig,
    STTService,
)


@pytest.fixture
def stt_config():
    """Create test STT configuration."""
    return STTConfig(
        api_key="test_api_key",
        model="nova-2",
        language="en-US",
        interim_results=True,
        punctuate=True,
        smart_format=True,
        diarize=False,
        encoding="linear16",
        sample_rate=16000,
        channels=1,
    )


@pytest.fixture
def stt_service(stt_config):
    """Create STT service instance."""
    return STTService(stt_config)


class TestSTTServiceInitialization:
    """Test STT service initialization."""

    def test_initialization_with_config(self, stt_config):
        """Test service initializes with provided configuration."""
        service = STTService(stt_config)

        assert service.config == stt_config
        assert service.state == ConnectionState.DISCONNECTED
        assert not service.is_connected
        assert service._session_id is None

    def test_default_config_values(self):
        """Test default configuration values are applied."""
        config = STTConfig(api_key="test_key")

        assert config.model == "nova-2"
        assert config.language == "en-US"
        assert config.interim_results is True
        assert config.punctuate is True
        assert config.sample_rate == 16000


class TestCallbackRegistration:
    """Test callback handler registration."""

    def test_register_interim_callback(self, stt_service):
        """Test registering interim result callback."""
        callback = Mock()
        stt_service.on_interim_result(callback)

        assert stt_service._interim_callback == callback

    def test_register_final_callback(self, stt_service):
        """Test registering final result callback."""
        callback = Mock()
        stt_service.on_final_result(callback)

        assert stt_service._final_callback == callback

    def test_register_error_callback(self, stt_service):
        """Test registering error callback."""
        callback = Mock()
        stt_service.on_error(callback)

        assert stt_service._error_callback == callback

    def test_multiple_callback_registrations(self, stt_service):
        """Test registering all callback types."""
        interim_cb = Mock()
        final_cb = Mock()
        error_cb = Mock()

        stt_service.on_interim_result(interim_cb)
        stt_service.on_final_result(final_cb)
        stt_service.on_error(error_cb)

        assert stt_service._interim_callback == interim_cb
        assert stt_service._final_callback == final_cb
        assert stt_service._error_callback == error_cb


class TestConnectionManagement:
    """Test WebSocket connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_success(self, stt_service):
        """Test successful connection to Deepgram."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mocks
            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_connection.__aexit__ = AsyncMock()
            mock_connection.start_listening = AsyncMock()
            mock_connection.on = Mock()

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Connect
            await stt_service.connect("test_session_123")

            # Verify state
            assert stt_service.state == ConnectionState.CONNECTED
            assert stt_service.is_connected
            assert stt_service._session_id == "test_session_123"

            # Verify connection was started
            mock_connection.start_listening.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_with_retry(self, stt_service):
        """Test connection failure triggers retry logic."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mock to fail then succeed
            call_count = 0

            async def mock_aenter_side_effect():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Connection failed")
                return mock_connection

            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(side_effect=mock_aenter_side_effect)
            mock_connection.__aexit__ = AsyncMock()
            mock_connection.start_listening = AsyncMock()
            mock_connection.on = Mock()

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Reduce retry delay for testing
            stt_service._reconnect_delay = 0.01

            # Connect should succeed after retry
            await stt_service.connect("test_session")

            assert stt_service.is_connected

    @pytest.mark.asyncio
    async def test_connect_max_retries_exceeded(self, stt_service):
        """Test connection fails after max retry attempts."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mock to always fail
            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Reduce retry delay for testing
            stt_service._reconnect_delay = 0.01
            stt_service._max_reconnect_attempts = 2

            # Connect should fail after max retries
            with pytest.raises(ConnectionError, match="Failed to connect after"):
                await stt_service.connect("test_session")

            assert stt_service.state == ConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_disconnect_when_connected(self, stt_service):
        """Test disconnecting from active connection."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mocks
            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_connection.__aexit__ = AsyncMock()
            mock_connection.start_listening = AsyncMock()
            mock_connection.on = Mock()

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Connect then disconnect
            await stt_service.connect("test_session")
            await stt_service.disconnect()

            # Verify state
            assert stt_service.state == ConnectionState.DISCONNECTED
            assert not stt_service.is_connected
            assert stt_service._session_id is None

            # Verify cleanup
            mock_connection.__aexit__.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, stt_service):
        """Test disconnecting when not connected is safe."""
        await stt_service.disconnect()

        assert stt_service.state == ConnectionState.DISCONNECTED
        assert not stt_service.is_connected


class TestAudioStreaming:
    """Test audio chunk streaming."""

    @pytest.mark.asyncio
    async def test_send_audio_success(self, stt_service):
        """Test sending audio chunk when connected."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mocks
            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_connection.start_listening = AsyncMock()
            mock_connection.send = Mock()
            mock_connection.on = Mock()

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Connect
            await stt_service.connect("test_session")

            # Send audio
            audio_data = b"fake_audio_data"
            await stt_service.send_audio(audio_data)

            # Verify audio was sent
            mock_connection.send.assert_called_once_with(audio_data)

    @pytest.mark.asyncio
    async def test_send_audio_not_connected(self, stt_service):
        """Test sending audio when not connected raises error."""
        audio_data = b"fake_audio_data"

        with pytest.raises(RuntimeError, match="Not connected"):
            await stt_service.send_audio(audio_data)

    @pytest.mark.asyncio
    async def test_send_empty_audio(self, stt_service):
        """Test sending empty audio chunk raises error."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mocks
            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_connection.start_listening = AsyncMock()
            mock_connection.on = Mock()

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Connect
            await stt_service.connect("test_session")

            # Send empty audio
            with pytest.raises(ValueError, match="cannot be empty"):
                await stt_service.send_audio(b"")


class TestMessageHandling:
    """Test message event handling."""

    def test_interim_message_callback(self, stt_service):
        """Test interim message invokes correct callback."""
        # Register callback
        interim_results = []
        def interim_callback(transcript, metadata):
            interim_results.append((transcript, metadata))

        stt_service.on_interim_result(interim_callback)

        # Create mock message
        mock_alternative = Mock()
        mock_alternative.transcript = "Hello world"
        mock_alternative.confidence = 0.85

        mock_channel = Mock()
        mock_channel.alternatives = [mock_alternative]

        mock_message = Mock()
        mock_message.type = "Results"
        mock_message.channel = mock_channel
        mock_message.is_final = False
        mock_message.speech_final = False
        mock_message.duration = 1.5
        mock_message.start = 0.0

        # Invoke handler
        stt_service._on_message(mock_message)

        # Verify callback was invoked
        assert len(interim_results) == 1
        transcript, metadata = interim_results[0]
        assert transcript == "Hello world"
        assert metadata["confidence"] == 0.85
        assert metadata["is_final"] is False

    def test_final_message_callback(self, stt_service):
        """Test final message invokes correct callback."""
        # Register callback
        final_results = []
        def final_callback(transcript, metadata):
            final_results.append((transcript, metadata))

        stt_service.on_final_result(final_callback)

        # Create mock message
        mock_alternative = Mock()
        mock_alternative.transcript = "Complete sentence."
        mock_alternative.confidence = 0.95

        mock_channel = Mock()
        mock_channel.alternatives = [mock_alternative]

        mock_message = Mock()
        mock_message.type = "Results"
        mock_message.channel = mock_channel
        mock_message.is_final = True
        mock_message.speech_final = True
        mock_message.duration = 2.0
        mock_message.start = 0.0

        # Invoke handler
        stt_service._on_message(mock_message)

        # Verify callback was invoked
        assert len(final_results) == 1
        transcript, metadata = final_results[0]
        assert transcript == "Complete sentence."
        assert metadata["confidence"] == 0.95
        assert metadata["is_final"] is True

    def test_empty_message_ignored(self, stt_service):
        """Test empty messages are ignored."""
        # Register callback
        callback = Mock()
        stt_service.on_final_result(callback)

        # Create mock message with empty transcript
        mock_alternative = Mock()
        mock_alternative.transcript = ""

        mock_channel = Mock()
        mock_channel.alternatives = [mock_alternative]

        mock_message = Mock()
        mock_message.type = "Results"
        mock_message.channel = mock_channel
        mock_message.is_final = True

        # Invoke handler
        stt_service._on_message(mock_message)

        # Verify callback was not invoked
        callback.assert_not_called()


class TestErrorHandling:
    """Test error handling."""

    def test_error_event_callback(self, stt_service):
        """Test error event invokes error callback."""
        # Register callback
        error_callback = Mock()
        stt_service.on_error(error_callback)

        # Trigger error
        error_msg = "Connection timeout"
        stt_service._on_error(error_msg)

        # Verify callback was invoked
        error_callback.assert_called_once()
        assert stt_service.state == ConnectionState.ERROR

    def test_close_event_updates_state(self, stt_service):
        """Test close event updates connection state."""
        # Set initial state
        stt_service._state = ConnectionState.CONNECTED

        # Trigger close
        stt_service._on_close()

        # Verify state updated
        assert stt_service.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_send_audio_error_handling(self, stt_service):
        """Test error handling when sending audio fails."""
        with patch('src.voice_agent.services.stt_service.AsyncDeepgramClient') as mock_client_class:
            # Setup mocks
            mock_connection = AsyncMock()
            mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
            mock_connection.start_listening = AsyncMock()
            mock_connection.send = Mock(side_effect=Exception("Network error"))
            mock_connection.on = Mock()

            mock_connect = Mock(return_value=mock_connection)
            mock_v2 = Mock()
            mock_v2.connect = mock_connect

            mock_listen = Mock()
            mock_listen.v2 = mock_v2

            mock_client = Mock()
            mock_client.listen = mock_listen
            mock_client_class.return_value = mock_client

            # Register error callback
            error_callback = Mock()
            stt_service.on_error(error_callback)

            # Connect
            await stt_service.connect("test_session")

            # Send audio should fail
            audio_data = b"fake_audio_data"
            with pytest.raises(Exception, match="Network error"):
                await stt_service.send_audio(audio_data)

            # Verify error callback was invoked
            error_callback.assert_called_once()
            assert stt_service.state == ConnectionState.ERROR


class TestDegradedAudioHandling:
    """Test handling of degraded audio quality.

    **Validates: Requirements 12.5**
    """

    def test_low_confidence_transcript_handling(self, stt_service):
        """Test handling of low confidence transcripts from degraded audio."""
        # Register callback
        final_results = []
        def final_callback(transcript, metadata):
            final_results.append((transcript, metadata))

        stt_service.on_final_result(final_callback)

        # Create mock message with low confidence (degraded audio)
        mock_alternative = Mock()
        mock_alternative.transcript = "unclear speech"
        mock_alternative.confidence = 0.35  # Low confidence

        mock_channel = Mock()
        mock_channel.alternatives = [mock_alternative]

        mock_message = Mock()
        mock_message.type = "Results"
        mock_message.channel = mock_channel
        mock_message.is_final = True
        mock_message.speech_final = True
        mock_message.duration = 1.0
        mock_message.start = 0.0

        # Invoke handler
        stt_service._on_message(mock_message)

        # Verify callback was invoked with low confidence metadata
        assert len(final_results) == 1
        transcript, metadata = final_results[0]
        assert transcript == "unclear speech"
        assert metadata["confidence"] == 0.35
        assert metadata["is_final"] is True

    def test_multiple_low_confidence_interim_results(self, stt_service):
        """Test handling of multiple interim results with varying confidence."""
        # Register callback
        interim_results = []
        def interim_callback(transcript, metadata):
            interim_results.append((transcript, metadata))

        stt_service.on_interim_result(interim_callback)

        # Simulate multiple interim results with degraded audio
        confidence_levels = [0.25, 0.40, 0.55, 0.70]
        transcripts = ["uh", "uh hello", "hello there", "hello there friend"]

        for confidence, transcript_text in zip(confidence_levels, transcripts):
            mock_alternative = Mock()
            mock_alternative.transcript = transcript_text
            mock_alternative.confidence = confidence

            mock_channel = Mock()
            mock_channel.alternatives = [mock_alternative]

            mock_message = Mock()
            mock_message.type = "Results"
            mock_message.channel = mock_channel
            mock_message.is_final = False
            mock_message.speech_final = False
            mock_message.duration = 0.5
            mock_message.start = 0.0

            stt_service._on_message(mock_message)

        # Verify all interim results were processed
        assert len(interim_results) == 4

        # Verify confidence improves as audio quality improves
        for i, (transcript, metadata) in enumerate(interim_results):
            assert metadata["confidence"] == confidence_levels[i]
            assert transcript == transcripts[i]

    def test_message_processing_error_handling(self, stt_service):
        """Test error handling when message processing fails."""
        # Register callbacks
        error_callback = Mock()
        final_callback = Mock()

        stt_service.on_error(error_callback)
        stt_service.on_final_result(final_callback)

        # Create malformed message (missing required attributes)
        mock_message = Mock()
        mock_message.type = "Results"
        # Intentionally missing channel attribute
        del mock_message.channel

        # Invoke handler - should handle gracefully
        stt_service._on_message(mock_message)

        # Verify final callback was not invoked
        final_callback.assert_not_called()

    def test_no_alternatives_in_message(self, stt_service):
        """Test handling of messages with no alternatives (degraded audio)."""
        # Register callback
        final_callback = Mock()
        stt_service.on_final_result(final_callback)

        # Create message with no alternatives
        mock_channel = Mock()
        mock_channel.alternatives = []

        mock_message = Mock()
        mock_message.type = "Results"
        mock_message.channel = mock_channel
        mock_message.is_final = True

        # Invoke handler
        stt_service._on_message(mock_message)

        # Verify callback was not invoked (no transcript to process)
        final_callback.assert_not_called()

    def test_whitespace_only_transcript(self, stt_service):
        """Test handling of whitespace-only transcripts from degraded audio."""
        # Register callback
        final_callback = Mock()
        stt_service.on_final_result(final_callback)

        # Create message with whitespace-only transcript
        mock_alternative = Mock()
        mock_alternative.transcript = "   "
        mock_alternative.confidence = 0.10

        mock_channel = Mock()
        mock_channel.alternatives = [mock_alternative]

        mock_message = Mock()
        mock_message.type = "Results"
        mock_message.channel = mock_channel
        mock_message.is_final = True
        mock_message.speech_final = True
        mock_message.duration = 0.5
        mock_message.start = 0.0

        # Invoke handler
        stt_service._on_message(mock_message)

        # Verify callback was not invoked (empty transcript)
        final_callback.assert_not_called()
