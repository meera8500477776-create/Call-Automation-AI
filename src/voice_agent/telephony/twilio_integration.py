"""Twilio telephony integration for voice agent platform.

This module provides WebSocket/WebRTC connection handling for Twilio,
audio stream processing for incoming and outgoing calls, and call
lifecycle management.
"""

import asyncio
import base64
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

import websockets
from websockets.client import WebSocketClientProtocol

logger = logging.getLogger(__name__)


class ConnectionType(Enum):
    """Type of connection to Twilio."""
    WEBSOCKET = "websocket"
    WEBRTC = "webrtc"


class CallStatus(Enum):
    """Status of a call."""
    INITIATED = "initiated"
    RINGING = "ringing"
    IN_PROGRESS = "in-progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BUSY = "busy"
    NO_ANSWER = "no-answer"


@dataclass
class TwilioConfig:
    """Configuration for Twilio integration.
    
    Attributes:
        account_sid: Twilio account SID
        auth_token: Twilio authentication token
        connection_type: Type of connection (WebSocket or WebRTC)
        audio_format: Audio format (e.g., "mulaw", "pcm")
        sample_rate: Audio sample rate in Hz
        channels: Number of audio channels (1 for mono, 2 for stereo)
        chunk_size: Size of audio chunks in bytes
        enable_dtmf: Enable DTMF tone detection
    """
    account_sid: str
    auth_token: str
    connection_type: ConnectionType = ConnectionType.WEBSOCKET
    audio_format: str = "mulaw"
    sample_rate: int = 8000
    channels: int = 1
    chunk_size: int = 160  # 20ms at 8kHz
    enable_dtmf: bool = False


@dataclass
class CallMetadata:
    """Metadata extracted from a Twilio call.
    
    Attributes:
        call_sid: Unique identifier for the call
        caller_id: Phone number of the caller (E.164 format)
        called_number: Phone number that was called
        direction: Call direction ("inbound" or "outbound")
        status: Current status of the call
        start_time: When the call was initiated
        end_time: When the call ended (if completed)
        duration_seconds: Duration of the call in seconds
        from_city: City of the caller (if available)
        from_state: State of the caller (if available)
        from_country: Country of the caller (if available)
        custom_parameters: Additional custom parameters from the call
    """
    call_sid: str
    caller_id: str
    called_number: str
    direction: str = "inbound"
    status: CallStatus = CallStatus.INITIATED
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    from_city: Optional[str] = None
    from_state: Optional[str] = None
    from_country: Optional[str] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "call_sid": self.call_sid,
            "caller_id": self.caller_id,
            "called_number": self.called_number,
            "direction": self.direction,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "from_city": self.from_city,
            "from_state": self.from_state,
            "from_country": self.from_country,
            "custom_parameters": self.custom_parameters,
        }


class TwilioIntegration:
    """Twilio telephony integration layer.
    
    Handles WebSocket/WebRTC connections to Twilio, processes incoming
    and outgoing audio streams, and manages call lifecycle.
    """

    def __init__(self, config: TwilioConfig):
        """Initialize Twilio integration.
        
        Args:
            config: Twilio configuration
        """
        self.config = config
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._call_metadata: Optional[CallMetadata] = None
        self._is_connected = False
        self._is_active = False
        self._stream_sid: Optional[str] = None
        
        # Callbacks
        self._on_audio_received: Optional[Callable[[bytes], None]] = None
        self._on_call_started: Optional[Callable[[CallMetadata], None]] = None
        self._on_call_ended: Optional[Callable[[CallMetadata], None]] = None
        self._on_dtmf: Optional[Callable[[str], None]] = None
        self._on_error: Optional[Callable[[Exception], None]] = None
        
        # Audio buffering
        self._audio_buffer: asyncio.Queue = asyncio.Queue()
        self._receive_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        
        logger.info(
            f"Initialized Twilio integration with {config.connection_type.value} "
            f"connection, {config.audio_format} format at {config.sample_rate}Hz"
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to Twilio."""
        return self._is_connected

    @property
    def is_active(self) -> bool:
        """Check if call is active."""
        return self._is_active

    @property
    def call_metadata(self) -> Optional[CallMetadata]:
        """Get current call metadata."""
        return self._call_metadata

    @property
    def stream_sid(self) -> Optional[str]:
        """Get current stream SID."""
        return self._stream_sid

    def on_audio_received(self, callback: Callable[[bytes], None]) -> None:
        """Register callback for incoming audio data.
        
        Args:
            callback: Function to call with audio data (bytes)
        """
        self._on_audio_received = callback
        logger.debug("Registered audio received callback")

    def on_call_started(self, callback: Callable[[CallMetadata], None]) -> None:
        """Register callback for call start event.
        
        Args:
            callback: Function to call with call metadata
        """
        self._on_call_started = callback
        logger.debug("Registered call started callback")

    def on_call_ended(self, callback: Callable[[CallMetadata], None]) -> None:
        """Register callback for call end event.
        
        Args:
            callback: Function to call with call metadata
        """
        self._on_call_ended = callback
        logger.debug("Registered call ended callback")

    def on_dtmf(self, callback: Callable[[str], None]) -> None:
        """Register callback for DTMF tone detection.
        
        Args:
            callback: Function to call with DTMF digit
        """
        self._on_dtmf = callback
        logger.debug("Registered DTMF callback")

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Register callback for error events.
        
        Args:
            callback: Function to call with exception
        """
        self._on_error = callback
        logger.debug("Registered error callback")

    async def connect(self, websocket_url: str) -> None:
        """Connect to Twilio via WebSocket.
        
        Args:
            websocket_url: WebSocket URL provided by Twilio
            
        Raises:
            ConnectionError: If connection fails
        """
        try:
            logger.info(f"Connecting to Twilio WebSocket: {websocket_url}")
            self._websocket = await websockets.connect(websocket_url)
            self._is_connected = True
            
            # Start receive task
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._send_task = asyncio.create_task(self._send_loop())
            
            logger.info("Successfully connected to Twilio")
            
        except Exception as e:
            logger.error(f"Failed to connect to Twilio: {e}")
            self._is_connected = False
            if self._on_error:
                self._on_error(e)
            raise ConnectionError(f"Failed to connect to Twilio: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from Twilio and cleanup resources."""
        logger.info("Disconnecting from Twilio")
        
        self._is_connected = False
        self._is_active = False
        
        # Cancel tasks
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._send_task and not self._send_task.done():
            self._send_task.cancel()
            try:
                await self._send_task
            except asyncio.CancelledError:
                pass
        
        # Close WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        logger.info("Disconnected from Twilio")

    async def send_audio(self, audio_data: bytes) -> None:
        """Send audio data to Twilio.
        
        Args:
            audio_data: Raw audio bytes to send
            
        Raises:
            RuntimeError: If not connected
        """
        if not self._is_connected or not self._websocket:
            raise RuntimeError("Not connected to Twilio")
        
        # Add to send queue
        await self._audio_buffer.put(audio_data)

    async def _send_loop(self) -> None:
        """Background task to send audio from buffer to Twilio."""
        try:
            while self._is_connected:
                # Get audio from buffer
                audio_data = await self._audio_buffer.get()
                
                if not self._websocket or not self._stream_sid:
                    continue
                
                # Encode audio as base64
                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                
                # Create media message
                message = {
                    "event": "media",
                    "streamSid": self._stream_sid,
                    "media": {
                        "payload": audio_base64
                    }
                }
                
                # Send to Twilio
                await self._websocket.send(json.dumps(message))
                
        except asyncio.CancelledError:
            logger.debug("Send loop cancelled")
        except Exception as e:
            logger.error(f"Error in send loop: {e}")
            if self._on_error:
                self._on_error(e)

    async def _receive_loop(self) -> None:
        """Background task to receive messages from Twilio."""
        try:
            while self._is_connected and self._websocket:
                # Receive message
                message_str = await self._websocket.recv()
                message = json.loads(message_str)
                
                # Handle message based on event type
                event = message.get("event")
                
                if event == "connected":
                    await self._handle_connected(message)
                elif event == "start":
                    await self._handle_start(message)
                elif event == "media":
                    await self._handle_media(message)
                elif event == "stop":
                    await self._handle_stop(message)
                elif event == "dtmf":
                    await self._handle_dtmf(message)
                else:
                    logger.warning(f"Unknown event type: {event}")
                    
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
            self._is_connected = False
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
            self._is_connected = False
            if self._on_error:
                self._on_error(e)

    async def _handle_connected(self, message: Dict[str, Any]) -> None:
        """Handle connected event from Twilio.
        
        Args:
            message: Connected event message
        """
        logger.info("Received connected event from Twilio")
        logger.debug(f"Connected message: {message}")

    async def _handle_start(self, message: Dict[str, Any]) -> None:
        """Handle start event from Twilio.
        
        Args:
            message: Start event message with call metadata
        """
        logger.info("Received start event from Twilio")
        
        # Extract stream SID
        self._stream_sid = message.get("streamSid")
        
        # Extract call metadata
        start_data = message.get("start", {})
        call_sid = start_data.get("callSid", "")
        
        # Extract custom parameters
        custom_params = start_data.get("customParameters", {})
        
        # Create call metadata
        self._call_metadata = CallMetadata(
            call_sid=call_sid,
            caller_id=custom_params.get("From", ""),
            called_number=custom_params.get("To", ""),
            direction=custom_params.get("Direction", "inbound"),
            status=CallStatus.IN_PROGRESS,
            from_city=custom_params.get("FromCity"),
            from_state=custom_params.get("FromState"),
            from_country=custom_params.get("FromCountry"),
            custom_parameters=custom_params,
        )
        
        self._is_active = True
        
        logger.info(
            f"Call started: {call_sid} from {self._call_metadata.caller_id} "
            f"to {self._call_metadata.called_number}"
        )
        
        # Notify callback
        if self._on_call_started:
            self._on_call_started(self._call_metadata)

    async def _handle_media(self, message: Dict[str, Any]) -> None:
        """Handle media event from Twilio.
        
        Args:
            message: Media event message with audio data
        """
        media_data = message.get("media", {})
        payload = media_data.get("payload", "")
        
        if not payload:
            return
        
        # Decode base64 audio
        try:
            audio_data = base64.b64decode(payload)
            
            # Notify callback
            if self._on_audio_received:
                self._on_audio_received(audio_data)
                
        except Exception as e:
            logger.error(f"Error decoding audio data: {e}")
            if self._on_error:
                self._on_error(e)

    async def _handle_stop(self, message: Dict[str, Any]) -> None:
        """Handle stop event from Twilio.
        
        Args:
            message: Stop event message
        """
        logger.info("Received stop event from Twilio")
        
        self._is_active = False
        
        # Update call metadata
        if self._call_metadata:
            self._call_metadata.status = CallStatus.COMPLETED
            self._call_metadata.end_time = datetime.now()
            
            # Calculate duration
            duration = (
                self._call_metadata.end_time - self._call_metadata.start_time
            ).total_seconds()
            self._call_metadata.duration_seconds = duration
            
            logger.info(
                f"Call ended: {self._call_metadata.call_sid}, "
                f"duration: {duration:.2f}s"
            )
            
            # Notify callback
            if self._on_call_ended:
                self._on_call_ended(self._call_metadata)

    async def _handle_dtmf(self, message: Dict[str, Any]) -> None:
        """Handle DTMF event from Twilio.
        
        Args:
            message: DTMF event message
        """
        if not self.config.enable_dtmf:
            return
        
        dtmf_data = message.get("dtmf", {})
        digit = dtmf_data.get("digit", "")
        
        if digit:
            logger.debug(f"Received DTMF digit: {digit}")
            
            # Notify callback
            if self._on_dtmf:
                self._on_dtmf(digit)

    def get_call_metadata(self) -> Optional[CallMetadata]:
        """Get current call metadata.
        
        Returns:
            Call metadata if call is active, None otherwise
        """
        return self._call_metadata

    def extract_caller_id(self) -> Optional[str]:
        """Extract caller ID from current call.
        
        Returns:
            Caller ID (phone number) if available, None otherwise
        """
        if self._call_metadata:
            return self._call_metadata.caller_id
        return None

    def extract_call_sid(self) -> Optional[str]:
        """Extract call SID from current call.
        
        Returns:
            Call SID if available, None otherwise
        """
        if self._call_metadata:
            return self._call_metadata.call_sid
        return None
