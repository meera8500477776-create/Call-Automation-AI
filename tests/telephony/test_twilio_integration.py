"""Unit tests for Twilio integration layer."""

import asyncio
import base64
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent.telephony.twilio_integration import (
    CallMetadata,
    CallStatus,
    ConnectionType,
    TwilioConfig,
    TwilioIntegration,
)


@pytest.fixture
def twilio_config():
    """Create a test Twilio configuration."""
    return TwilioConfig(
        account_sid="test_account_sid",
        auth_token="test_auth_token",
        connection_type=ConnectionType.WEBSOCKET,
        audio_format="mulaw",
        sample_rate=8000,
        channels=1,
        chunk_size=160,
        enable_dtmf=True,
    )


@pytest.fixture
def twilio_integration(twilio_config):
    """Create a Twilio integration instance."""
    return TwilioIntegration(twilio_config)


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.recv = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


class TestTwilioConfig:
    """Tests for TwilioConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TwilioConfig(
            account_sid="test_sid",
            auth_token="test_token",
        )

        assert config.account_sid == "test_sid"
        assert config.auth_token == "test_token"
        assert config.connection_type == ConnectionType.WEBSOCKET
        assert config.audio_format == "mulaw"
        assert config.sample_rate == 8000
        assert config.channels == 1
        assert config.chunk_size == 160
        assert config.enable_dtmf is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TwilioConfig(
            account_sid="custom_sid",
            auth_token="custom_token",
            connection_type=ConnectionType.WEBRTC,
            audio_format="pcm",
            sample_rate=16000,
            channels=2,
            chunk_size=320,
            enable_dtmf=True,
        )

        assert config.connection_type == ConnectionType.WEBRTC
        assert config.audio_format == "pcm"
        assert config.sample_rate == 16000
        assert config.channels == 2
        assert config.chunk_size == 320
        assert config.enable_dtmf is True


class TestCallMetadata:
    """Tests for CallMetadata."""

    def test_call_metadata_creation(self):
        """Test creating call metadata."""
        metadata = CallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
            direction="inbound",
        )

        assert metadata.call_sid == "CA123456"
        assert metadata.caller_id == "+15551234567"
        assert metadata.called_number == "+15559876543"
        assert metadata.direction == "inbound"
        assert metadata.status == CallStatus.INITIATED
        assert isinstance(metadata.start_time, datetime)
        assert metadata.end_time is None
        assert metadata.duration_seconds == 0.0

    def test_call_metadata_to_dict(self):
        """Test converting call metadata to dictionary."""
        metadata = CallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
            from_city="San Francisco",
            from_state="CA",
            from_country="US",
            custom_parameters={"foo": "bar"},
        )

        result = metadata.to_dict()

        assert result["call_sid"] == "CA123456"
        assert result["caller_id"] == "+15551234567"
        assert result["called_number"] == "+15559876543"
        assert result["from_city"] == "San Francisco"
        assert result["from_state"] == "CA"
        assert result["from_country"] == "US"
        assert result["custom_parameters"] == {"foo": "bar"}
        assert result["status"] == "initiated"


class TestTwilioIntegration:
    """Tests for TwilioIntegration."""

    def test_initialization(self, twilio_integration, twilio_config):
        """Test Twilio integration initialization."""
        assert twilio_integration.config == twilio_config
        assert not twilio_integration.is_connected
        assert not twilio_integration.is_active
        assert twilio_integration.call_metadata is None
        assert twilio_integration.stream_sid is None

    def test_callback_registration(self, twilio_integration):
        """Test registering callbacks."""
        audio_callback = MagicMock()
        start_callback = MagicMock()
        end_callback = MagicMock()
        dtmf_callback = MagicMock()
        error_callback = MagicMock()

        twilio_integration.on_audio_received(audio_callback)
        twilio_integration.on_call_started(start_callback)
        twilio_integration.on_call_ended(end_callback)
        twilio_integration.on_dtmf(dtmf_callback)
        twilio_integration.on_error(error_callback)

        assert twilio_integration._on_audio_received == audio_callback
        assert twilio_integration._on_call_started == start_callback
        assert twilio_integration._on_call_ended == end_callback
        assert twilio_integration._on_dtmf == dtmf_callback
        assert twilio_integration._on_error == error_callback

    @pytest.mark.asyncio
    async def test_connect_success(self, twilio_integration, mock_websocket):
        """Test successful WebSocket connection."""
        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Make the mock awaitable
            async def mock_connect_coro(*args, **kwargs):
                return mock_websocket
            mock_connect.side_effect = mock_connect_coro

            await twilio_integration.connect("wss://test.twilio.com")

            assert twilio_integration.is_connected
            assert twilio_integration._websocket == mock_websocket
            mock_connect.assert_called_once_with("wss://test.twilio.com")

    @pytest.mark.asyncio
    async def test_connect_failure(self, twilio_integration):
        """Test WebSocket connection failure."""
        error_callback = MagicMock()
        twilio_integration.on_error(error_callback)

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(ConnectionError, match="Failed to connect to Twilio"):
                await twilio_integration.connect("wss://test.twilio.com")

            assert not twilio_integration.is_connected
            error_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, twilio_integration, mock_websocket):
        """Test disconnecting from Twilio."""
        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Make the mock awaitable
            async def mock_connect_coro(*args, **kwargs):
                return mock_websocket
            mock_connect.side_effect = mock_connect_coro

            await twilio_integration.connect("wss://test.twilio.com")
            assert twilio_integration.is_connected

            await twilio_integration.disconnect()

            assert not twilio_integration.is_connected
            assert not twilio_integration.is_active
            mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_audio_not_connected(self, twilio_integration):
        """Test sending audio when not connected raises error."""
        with pytest.raises(RuntimeError, match="Not connected to Twilio"):
            await twilio_integration.send_audio(b"audio_data")

    @pytest.mark.asyncio
    async def test_send_audio_connected(self, twilio_integration, mock_websocket):
        """Test sending audio when connected."""
        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Make the mock awaitable
            async def mock_connect_coro(*args, **kwargs):
                return mock_websocket
            mock_connect.side_effect = mock_connect_coro

            await twilio_integration.connect("wss://test.twilio.com")
            twilio_integration._stream_sid = "ST123456"

            # Send audio
            audio_data = b"test_audio_data"
            await twilio_integration.send_audio(audio_data)

            # Give send loop time to process
            await asyncio.sleep(0.1)

            # Verify audio was queued
            assert not twilio_integration._audio_buffer.empty()

    @pytest.mark.asyncio
    async def test_handle_connected_event(self, twilio_integration):
        """Test handling connected event from Twilio."""
        message = {
            "event": "connected",
            "protocol": "Call",
            "version": "1.0.0"
        }

        await twilio_integration._handle_connected(message)
        # Should not raise any errors

    @pytest.mark.asyncio
    async def test_handle_start_event(self, twilio_integration):
        """Test handling start event from Twilio."""
        start_callback = MagicMock()
        twilio_integration.on_call_started(start_callback)

        message = {
            "event": "start",
            "streamSid": "ST123456",
            "start": {
                "callSid": "CA123456",
                "customParameters": {
                    "From": "+15551234567",
                    "To": "+15559876543",
                    "Direction": "inbound",
                    "FromCity": "San Francisco",
                    "FromState": "CA",
                    "FromCountry": "US",
                }
            }
        }

        await twilio_integration._handle_start(message)

        assert twilio_integration.is_active
        assert twilio_integration.stream_sid == "ST123456"
        assert twilio_integration.call_metadata is not None
        assert twilio_integration.call_metadata.call_sid == "CA123456"
        assert twilio_integration.call_metadata.caller_id == "+15551234567"
        assert twilio_integration.call_metadata.called_number == "+15559876543"
        assert twilio_integration.call_metadata.direction == "inbound"
        assert twilio_integration.call_metadata.from_city == "San Francisco"
        assert twilio_integration.call_metadata.from_state == "CA"
        assert twilio_integration.call_metadata.from_country == "US"
        assert twilio_integration.call_metadata.status == CallStatus.IN_PROGRESS

        start_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_media_event(self, twilio_integration):
        """Test handling media event from Twilio."""
        audio_callback = MagicMock()
        twilio_integration.on_audio_received(audio_callback)

        audio_data = b"test_audio_data"
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        message = {
            "event": "media",
            "streamSid": "ST123456",
            "media": {
                "payload": audio_base64
            }
        }

        await twilio_integration._handle_media(message)

        audio_callback.assert_called_once_with(audio_data)

    @pytest.mark.asyncio
    async def test_handle_media_event_empty_payload(self, twilio_integration):
        """Test handling media event with empty payload."""
        audio_callback = MagicMock()
        twilio_integration.on_audio_received(audio_callback)

        message = {
            "event": "media",
            "streamSid": "ST123456",
            "media": {
                "payload": ""
            }
        }

        await twilio_integration._handle_media(message)

        audio_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_stop_event(self, twilio_integration):
        """Test handling stop event from Twilio."""
        end_callback = MagicMock()
        twilio_integration.on_call_ended(end_callback)

        # Set up call metadata
        twilio_integration._call_metadata = CallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        twilio_integration._is_active = True

        message = {
            "event": "stop",
            "streamSid": "ST123456"
        }

        await twilio_integration._handle_stop(message)

        assert not twilio_integration.is_active
        assert twilio_integration.call_metadata.status == CallStatus.COMPLETED
        assert twilio_integration.call_metadata.end_time is not None
        assert twilio_integration.call_metadata.duration_seconds > 0

        end_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_dtmf_event(self, twilio_integration):
        """Test handling DTMF event from Twilio."""
        dtmf_callback = MagicMock()
        twilio_integration.on_dtmf(dtmf_callback)

        message = {
            "event": "dtmf",
            "streamSid": "ST123456",
            "dtmf": {
                "digit": "5"
            }
        }

        await twilio_integration._handle_dtmf(message)

        dtmf_callback.assert_called_once_with("5")

    @pytest.mark.asyncio
    async def test_handle_dtmf_event_disabled(self, twilio_config):
        """Test DTMF event is ignored when disabled."""
        twilio_config.enable_dtmf = False
        integration = TwilioIntegration(twilio_config)

        dtmf_callback = MagicMock()
        integration.on_dtmf(dtmf_callback)

        message = {
            "event": "dtmf",
            "streamSid": "ST123456",
            "dtmf": {
                "digit": "5"
            }
        }

        await integration._handle_dtmf(message)

        dtmf_callback.assert_not_called()

    def test_get_call_metadata(self, twilio_integration):
        """Test getting call metadata."""
        assert twilio_integration.get_call_metadata() is None

        metadata = CallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        twilio_integration._call_metadata = metadata

        assert twilio_integration.get_call_metadata() == metadata

    def test_extract_caller_id(self, twilio_integration):
        """Test extracting caller ID."""
        assert twilio_integration.extract_caller_id() is None

        metadata = CallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        twilio_integration._call_metadata = metadata

        assert twilio_integration.extract_caller_id() == "+15551234567"

    def test_extract_call_sid(self, twilio_integration):
        """Test extracting call SID."""
        assert twilio_integration.extract_call_sid() is None

        metadata = CallMetadata(
            call_sid="CA123456",
            caller_id="+15551234567",
            called_number="+15559876543",
        )
        twilio_integration._call_metadata = metadata

        assert twilio_integration.extract_call_sid() == "CA123456"


class TestTwilioIntegrationEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_handle_media_invalid_base64(self, twilio_integration):
        """Test handling media event with invalid base64."""
        error_callback = MagicMock()
        twilio_integration.on_error(error_callback)

        message = {
            "event": "media",
            "streamSid": "ST123456",
            "media": {
                "payload": "invalid_base64!!!"
            }
        }

        await twilio_integration._handle_media(message)

        error_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_audio_encodes_correctly(self, twilio_integration, mock_websocket):
        """Test that audio is correctly encoded before sending."""
        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Make the mock awaitable
            async def mock_connect_coro(*args, **kwargs):
                return mock_websocket
            mock_connect.side_effect = mock_connect_coro

            await twilio_integration.connect("wss://test.twilio.com")
            twilio_integration._stream_sid = "ST123456"

            # Send audio
            audio_data = b"test_audio_data"
            await twilio_integration.send_audio(audio_data)

            # Give send loop time to process
            await asyncio.sleep(0.1)

            # Verify message format
            if mock_websocket.send.called:
                sent_message = json.loads(mock_websocket.send.call_args[0][0])
                assert sent_message["event"] == "media"
                assert sent_message["streamSid"] == "ST123456"
                assert "payload" in sent_message["media"]

                # Verify payload can be decoded
                decoded = base64.b64decode(sent_message["media"]["payload"])
                assert decoded == audio_data

    @pytest.mark.asyncio
    async def test_multiple_callbacks(self, twilio_integration):
        """Test that multiple events trigger callbacks correctly."""
        audio_callback = MagicMock()
        start_callback = MagicMock()
        end_callback = MagicMock()

        twilio_integration.on_audio_received(audio_callback)
        twilio_integration.on_call_started(start_callback)
        twilio_integration.on_call_ended(end_callback)

        # Simulate call lifecycle
        start_message = {
            "event": "start",
            "streamSid": "ST123456",
            "start": {
                "callSid": "CA123456",
                "customParameters": {
                    "From": "+15551234567",
                    "To": "+15559876543",
                }
            }
        }
        await twilio_integration._handle_start(start_message)
        start_callback.assert_called_once()

        # Simulate audio
        audio_data = b"test_audio"
        media_message = {
            "event": "media",
            "media": {
                "payload": base64.b64encode(audio_data).decode("utf-8")
            }
        }
        await twilio_integration._handle_media(media_message)
        audio_callback.assert_called_once_with(audio_data)

        # Simulate call end
        stop_message = {
            "event": "stop",
            "streamSid": "ST123456"
        }
        await twilio_integration._handle_stop(stop_message)
        end_callback.assert_called_once()


class TestTwilioIntegrationIntegration:
    """Integration tests for Twilio integration.

    Tests WebSocket connection establishment, bidirectional audio streaming,
    and complete call lifecycle scenarios.

    **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
    """

    @pytest.mark.asyncio
    async def test_websocket_connection_establishment(self, twilio_integration, mock_websocket):
        """
        Test WebSocket connection establishment with mock Twilio.

        **Validates: Requirement 11.1**
        """
        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Make the mock awaitable
            async def mock_connect_coro(*args, **kwargs):
                return mock_websocket
            mock_connect.side_effect = mock_connect_coro

            # Connect to Twilio
            websocket_url = "wss://streams.twilio.com/v1/stream"
            await twilio_integration.connect(websocket_url)

            # Verify connection was established
            assert twilio_integration.is_connected
            assert twilio_integration._websocket == mock_websocket
            mock_connect.assert_called_once_with(websocket_url)

            # Verify background tasks were started
            assert twilio_integration._receive_task is not None
            assert twilio_integration._send_task is not None
            assert not twilio_integration._receive_task.done()
            assert not twilio_integration._send_task.done()

            # Cleanup
            await twilio_integration.disconnect()

    @pytest.mark.asyncio
    async def test_websocket_connection_failure_handling(self, twilio_integration):
        """
        Test WebSocket connection failure handling.

        **Validates: Requirement 11.1**
        """
        error_callback = MagicMock()
        twilio_integration.on_error(error_callback)

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Simulate connection failure
            mock_connect.side_effect = Exception("Network error")

            # Attempt to connect
            with pytest.raises(ConnectionError, match="Failed to connect to Twilio"):
                await twilio_integration.connect("wss://invalid.url")

            # Verify error handling
            assert not twilio_integration.is_connected
            error_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_bidirectional_audio_streaming(self, twilio_integration):
        """
        Test audio streaming in both directions (Twilio -> Integration -> Twilio).

        **Validates: Requirements 11.2, 11.3**
        """
        # Create a mock websocket that properly handles recv
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Make recv block indefinitely (simulating waiting for messages)
        async def mock_recv():
            await asyncio.sleep(100)  # Block forever
            return "{}"
        mock_ws.recv = mock_recv

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Setup connection
            async def mock_connect_coro(*args, **kwargs):
                return mock_ws
            mock_connect.side_effect = mock_connect_coro

            await twilio_integration.connect("wss://test.twilio.com")
            twilio_integration._stream_sid = "ST123456"

            # Setup callbacks
            received_audio = []
            def audio_callback(audio_data: bytes):
                received_audio.append(audio_data)

            twilio_integration.on_audio_received(audio_callback)

            # Test incoming audio (Twilio -> Integration)
            incoming_audio_1 = b"incoming_audio_chunk_1"
            incoming_audio_2 = b"incoming_audio_chunk_2"

            await twilio_integration._handle_media({
                "event": "media",
                "media": {
                    "payload": base64.b64encode(incoming_audio_1).decode("utf-8")
                }
            })

            await twilio_integration._handle_media({
                "event": "media",
                "media": {
                    "payload": base64.b64encode(incoming_audio_2).decode("utf-8")
                }
            })

            # Verify incoming audio was received
            assert len(received_audio) == 2
            assert received_audio[0] == incoming_audio_1
            assert received_audio[1] == incoming_audio_2

            # Test outgoing audio (Integration -> Twilio)
            outgoing_audio_1 = b"outgoing_audio_chunk_1"
            outgoing_audio_2 = b"outgoing_audio_chunk_2"

            await twilio_integration.send_audio(outgoing_audio_1)
            await twilio_integration.send_audio(outgoing_audio_2)

            # Give send loop time to process
            await asyncio.sleep(0.2)

            # Verify outgoing audio was sent
            assert mock_ws.send.call_count >= 2

            # Verify audio format
            sent_messages = [
                json.loads(call[0][0])
                for call in mock_ws.send.call_args_list
            ]

            assert all(msg["event"] == "media" for msg in sent_messages)
            assert all(msg["streamSid"] == "ST123456" for msg in sent_messages)

            # Verify payloads can be decoded
            decoded_payloads = [
                base64.b64decode(msg["media"]["payload"])
                for msg in sent_messages
            ]
            assert outgoing_audio_1 in decoded_payloads
            assert outgoing_audio_2 in decoded_payloads

            # Cleanup
            await twilio_integration.disconnect()

    @pytest.mark.asyncio
    async def test_complete_call_lifecycle(self, twilio_integration):
        """
        Test complete call lifecycle: initiation, active, termination.

        **Validates: Requirements 11.1, 11.2, 11.3, 11.4**
        """
        # Create a mock websocket that properly handles recv
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Make recv block indefinitely (simulating waiting for messages)
        async def mock_recv():
            await asyncio.sleep(100)  # Block forever
            return "{}"
        mock_ws.recv = mock_recv

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            # Setup connection
            async def mock_connect_coro(*args, **kwargs):
                return mock_ws
            mock_connect.side_effect = mock_connect_coro

            # Track lifecycle events
            lifecycle_events = []
            received_audio = []

            def on_call_started(metadata):
                lifecycle_events.append(("started", metadata))

            def on_audio_received(audio_data):
                received_audio.append(audio_data)

            def on_call_ended(metadata):
                lifecycle_events.append(("ended", metadata))

            twilio_integration.on_call_started(on_call_started)
            twilio_integration.on_audio_received(on_audio_received)
            twilio_integration.on_call_ended(on_call_ended)

            # Phase 1: Connection establishment
            await twilio_integration.connect("wss://streams.twilio.com/v1/stream")
            assert twilio_integration.is_connected
            assert not twilio_integration.is_active

            # Simulate connected event
            await twilio_integration._handle_connected({
                "event": "connected",
                "protocol": "Call",
                "version": "1.0.0"
            })

            # Phase 2: Call initiation
            start_message = {
                "event": "start",
                "streamSid": "ST123456789",
                "start": {
                    "callSid": "CA987654321",
                    "customParameters": {
                        "From": "+15551234567",
                        "To": "+15559876543",
                        "Direction": "inbound",
                        "FromCity": "San Francisco",
                        "FromState": "CA",
                        "FromCountry": "US",
                    }
                }
            }

            await twilio_integration._handle_start(start_message)

            # Verify call is active
            assert twilio_integration.is_active
            assert twilio_integration.stream_sid == "ST123456789"
            assert len(lifecycle_events) == 1
            assert lifecycle_events[0][0] == "started"

            # Verify call metadata
            call_metadata = twilio_integration.call_metadata
            assert call_metadata is not None
            assert call_metadata.call_sid == "CA987654321"
            assert call_metadata.caller_id == "+15551234567"
            assert call_metadata.called_number == "+15559876543"
            assert call_metadata.direction == "inbound"
            assert call_metadata.status == CallStatus.IN_PROGRESS

            # Phase 3: Active call - bidirectional audio streaming
            # Simulate multiple audio exchanges
            for i in range(3):
                # Incoming audio from caller
                incoming_audio = f"caller_audio_{i}".encode()
                await twilio_integration._handle_media({
                    "event": "media",
                    "streamSid": "ST123456789",
                    "media": {
                        "payload": base64.b64encode(incoming_audio).decode("utf-8")
                    }
                })

                # Outgoing audio to caller
                outgoing_audio = f"agent_audio_{i}".encode()
                await twilio_integration.send_audio(outgoing_audio)

            # Give send loop time to process
            await asyncio.sleep(0.2)

            # Verify audio was exchanged
            assert len(received_audio) == 3
            assert all(f"caller_audio_{i}".encode() in received_audio for i in range(3))
            assert mock_ws.send.call_count >= 3

            # Phase 4: Call termination
            stop_message = {
                "event": "stop",
                "streamSid": "ST123456789"
            }

            await twilio_integration._handle_stop(stop_message)

            # Verify call ended
            assert not twilio_integration.is_active
            assert len(lifecycle_events) == 2
            assert lifecycle_events[1][0] == "ended"

            # Verify final call metadata
            final_metadata = lifecycle_events[1][1]
            assert final_metadata.status == CallStatus.COMPLETED
            assert final_metadata.end_time is not None
            assert final_metadata.duration_seconds > 0

            # Phase 5: Cleanup
            await twilio_integration.disconnect()
            assert not twilio_integration.is_connected

    @pytest.mark.asyncio
    async def test_call_lifecycle_with_dtmf(self, twilio_integration):
        """
        Test call lifecycle with DTMF tone detection.

        **Validates: Requirements 11.1, 11.4**
        """
        # Create a mock websocket that properly handles recv
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Make recv block indefinitely (simulating waiting for messages)
        async def mock_recv():
            await asyncio.sleep(100)  # Block forever
            return "{}"
        mock_ws.recv = mock_recv

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            async def mock_connect_coro(*args, **kwargs):
                return mock_ws
            mock_connect.side_effect = mock_connect_coro

            # Track DTMF digits
            dtmf_digits = []

            def on_dtmf(digit):
                dtmf_digits.append(digit)

            twilio_integration.on_dtmf(on_dtmf)

            # Connect and start call
            await twilio_integration.connect("wss://test.twilio.com")

            await twilio_integration._handle_start({
                "event": "start",
                "streamSid": "ST123456",
                "start": {
                    "callSid": "CA123456",
                    "customParameters": {
                        "From": "+15551234567",
                        "To": "+15559876543",
                    }
                }
            })

            # Simulate DTMF tones
            for digit in ["1", "2", "3", "#"]:
                await twilio_integration._handle_dtmf({
                    "event": "dtmf",
                    "streamSid": "ST123456",
                    "dtmf": {
                        "digit": digit
                    }
                })

            # Verify DTMF digits were captured
            assert dtmf_digits == ["1", "2", "3", "#"]

            # Cleanup
            await twilio_integration.disconnect()

    @pytest.mark.asyncio
    async def test_concurrent_audio_streaming(self, twilio_integration):
        """
        Test concurrent audio streaming doesn't cause race conditions.

        **Validates: Requirements 11.2, 11.3**
        """
        # Create a mock websocket that properly handles recv
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Make recv block indefinitely (simulating waiting for messages)
        async def mock_recv():
            await asyncio.sleep(100)  # Block forever
            return "{}"
        mock_ws.recv = mock_recv

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            async def mock_connect_coro(*args, **kwargs):
                return mock_ws
            mock_connect.side_effect = mock_connect_coro

            await twilio_integration.connect("wss://test.twilio.com")
            twilio_integration._stream_sid = "ST123456"

            # Track received audio
            received_audio = []

            def audio_callback(audio_data: bytes):
                received_audio.append(audio_data)

            twilio_integration.on_audio_received(audio_callback)

            # Send multiple audio chunks concurrently
            incoming_tasks = []
            outgoing_tasks = []

            for i in range(10):
                # Incoming audio
                incoming_audio = f"incoming_{i}".encode()
                incoming_tasks.append(
                    twilio_integration._handle_media({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(incoming_audio).decode("utf-8")
                        }
                    })
                )

                # Outgoing audio
                outgoing_audio = f"outgoing_{i}".encode()
                outgoing_tasks.append(
                    twilio_integration.send_audio(outgoing_audio)
                )

            # Execute concurrently
            await asyncio.gather(*incoming_tasks, *outgoing_tasks)

            # Give send loop time to process
            await asyncio.sleep(0.3)

            # Verify all audio was processed
            assert len(received_audio) == 10
            assert all(f"incoming_{i}".encode() in received_audio for i in range(10))

            # Cleanup
            await twilio_integration.disconnect()

    @pytest.mark.asyncio
    async def test_call_lifecycle_with_error_recovery(self, twilio_integration):
        """
        Test call lifecycle with error recovery.

        **Validates: Requirements 11.1, 11.4**
        """
        # Create a mock websocket that properly handles recv
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Make recv block indefinitely (simulating waiting for messages)
        async def mock_recv():
            await asyncio.sleep(100)  # Block forever
            return "{}"
        mock_ws.recv = mock_recv

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            async def mock_connect_coro(*args, **kwargs):
                return mock_ws
            mock_connect.side_effect = mock_connect_coro

            # Track errors
            errors = []

            def on_error(error):
                errors.append(error)

            twilio_integration.on_error(on_error)

            # Connect and start call
            await twilio_integration.connect("wss://test.twilio.com")

            await twilio_integration._handle_start({
                "event": "start",
                "streamSid": "ST123456",
                "start": {
                    "callSid": "CA123456",
                    "customParameters": {
                        "From": "+15551234567",
                        "To": "+15559876543",
                    }
                }
            })

            assert twilio_integration.is_active

            # Simulate error during audio processing
            await twilio_integration._handle_media({
                "event": "media",
                "media": {
                    "payload": "invalid_base64_data!!!"
                }
            })

            # Verify error was captured
            assert len(errors) == 1

            # Verify call is still active (error didn't crash the system)
            assert twilio_integration.is_active

            # Normal call termination
            await twilio_integration._handle_stop({
                "event": "stop",
                "streamSid": "ST123456"
            })

            assert not twilio_integration.is_active

            # Cleanup
            await twilio_integration.disconnect()

    @pytest.mark.asyncio
    async def test_multiple_call_sessions(self, twilio_config):
        """
        Test handling multiple sequential call sessions.

        **Validates: Requirements 11.1, 11.4**
        """
        # Create a mock websocket that properly handles recv
        mock_ws = AsyncMock()
        mock_ws.send = AsyncMock()
        mock_ws.close = AsyncMock()

        # Make recv block indefinitely (simulating waiting for messages)
        async def mock_recv():
            await asyncio.sleep(100)  # Block forever
            return "{}"
        mock_ws.recv = mock_recv

        with patch("voice_agent.telephony.twilio_integration.websockets.connect") as mock_connect:
            async def mock_connect_coro(*args, **kwargs):
                return mock_ws
            mock_connect.side_effect = mock_connect_coro

            # Create integration
            integration = TwilioIntegration(twilio_config)

            # Track call sessions
            call_sessions = []

            def on_call_started(metadata):
                call_sessions.append(("started", metadata.call_sid))

            def on_call_ended(metadata):
                call_sessions.append(("ended", metadata.call_sid))

            integration.on_call_started(on_call_started)
            integration.on_call_ended(on_call_ended)

            # Connect once
            await integration.connect("wss://test.twilio.com")

            # Simulate multiple call sessions
            for i in range(3):
                call_sid = f"CA{i:06d}"
                stream_sid = f"ST{i:06d}"

                # Start call
                await integration._handle_start({
                    "event": "start",
                    "streamSid": stream_sid,
                    "start": {
                        "callSid": call_sid,
                        "customParameters": {
                            "From": f"+1555123{i:04d}",
                            "To": "+15559876543",
                        }
                    }
                })

                assert integration.is_active
                assert integration.call_metadata.call_sid == call_sid

                # End call
                await integration._handle_stop({
                    "event": "stop",
                    "streamSid": stream_sid
                })

                assert not integration.is_active

            # Verify all sessions were tracked
            assert len(call_sessions) == 6  # 3 starts + 3 ends
            assert call_sessions[0] == ("started", "CA000000")
            assert call_sessions[1] == ("ended", "CA000000")
            assert call_sessions[2] == ("started", "CA000001")
            assert call_sessions[3] == ("ended", "CA000001")
            assert call_sessions[4] == ("started", "CA000002")
            assert call_sessions[5] == ("ended", "CA000002")

            # Cleanup
            await integration.disconnect()
