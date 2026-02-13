"""
STT Service wrapper for Deepgram Nova.

This module provides a streaming Speech-to-Text service using Deepgram's Nova model.
It handles WebSocket connections, audio streaming, and transcript callbacks.
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """WebSocket connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class STTConfig:
    """Configuration for STT service."""
    api_key: str
    model: str = "nova-2"
    language: str = "en-US"
    interim_results: bool = True
    punctuate: bool = True
    smart_format: bool = True
    diarize: bool = False
    encoding: str = "linear16"
    sample_rate: int = 16000
    channels: int = 1


class STTService:
    """
    Speech-to-Text service wrapper for Deepgram Nova.

    Provides streaming transcription with WebSocket connection management,
    audio chunk processing, and callback handlers for interim and final results.
    """

    def __init__(self, config: STTConfig):
        """
        Initialize STT service.

        Args:
            config: STT configuration including API key and model settings
        """
        self.config = config
        self._state = ConnectionState.DISCONNECTED
        self._session_id: str | None = None

        # Deepgram client and connection
        self._client: AsyncDeepgramClient | None = None
        self._connection: Any | None = None

        # Callback handlers
        self._interim_callback: Callable[[str, dict[str, Any]], None] | None = None
        self._final_callback: Callable[[str, dict[str, Any]], None] | None = None
        self._error_callback: Callable[[Exception], None] | None = None

        # Connection management
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._reconnect_delay = 1.0  # seconds

        logger.info("STT service initialized with model: %s", config.model)

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._state == ConnectionState.CONNECTED

    def on_interim_result(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """
        Register callback for interim transcription results.

        Args:
            callback: Function to call with (transcript, metadata) when interim result arrives
        """
        self._interim_callback = callback
        logger.debug("Interim result callback registered")

    def on_final_result(self, callback: Callable[[str, dict[str, Any]], None]) -> None:
        """
        Register callback for final transcription results.

        Args:
            callback: Function to call with (transcript, metadata) when final result arrives
        """
        self._final_callback = callback
        logger.debug("Final result callback registered")

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """
        Register callback for error events.

        Args:
            callback: Function to call with exception when error occurs
        """
        self._error_callback = callback
        logger.debug("Error callback registered")

    async def connect(self, session_id: str) -> None:
        """
        Establish WebSocket connection to Deepgram.

        Args:
            session_id: Unique identifier for this transcription session

        Raises:
            ConnectionError: If connection fails after max retry attempts
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()

        self._session_id = session_id
        self._state = ConnectionState.CONNECTING

        try:
            # Initialize Deepgram client
            self._client = AsyncDeepgramClient(api_key=self.config.api_key)

            # Create live transcription connection using v2 API
            self._connection = await self._client.listen.v2.connect(
                model=self.config.model,
                language=self.config.language,
                interim_results=self.config.interim_results,
                punctuate=self.config.punctuate,
                smart_format=self.config.smart_format,
                diarize=self.config.diarize,
                encoding=self.config.encoding,
                sample_rate=str(self.config.sample_rate),
                channels=self.config.channels,
            ).__aenter__()

            # Register event handlers
            self._connection.on(EventType.OPEN, self._on_open)
            self._connection.on(EventType.MESSAGE, self._on_message)
            self._connection.on(EventType.ERROR, self._on_error)
            self._connection.on(EventType.CLOSE, self._on_close)

            # Start listening
            await self._connection.start_listening()

            self._state = ConnectionState.CONNECTED
            self._reconnect_attempts = 0

            logger.info(
                "Connected to Deepgram for session %s (model: %s, language: %s)",
                session_id,
                self.config.model,
                self.config.language,
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            logger.error("Failed to connect to Deepgram: %s", str(e))

            # Attempt reconnection if within retry limit
            if self._reconnect_attempts < self._max_reconnect_attempts:
                self._reconnect_attempts += 1
                logger.info(
                    "Attempting reconnection %d/%d in %.1f seconds",
                    self._reconnect_attempts,
                    self._max_reconnect_attempts,
                    self._reconnect_delay,
                )
                await asyncio.sleep(self._reconnect_delay)
                await self.connect(session_id)
            else:
                error_msg = f"Failed to connect after {self._max_reconnect_attempts} attempts"
                logger.error(error_msg)
                if self._error_callback:
                    self._error_callback(ConnectionError(error_msg))
                raise ConnectionError(error_msg) from e

    async def send_audio(self, audio_chunk: bytes) -> None:
        """
        Send audio chunk for transcription.

        Args:
            audio_chunk: Raw audio data bytes

        Raises:
            RuntimeError: If not connected
            ValueError: If audio chunk is invalid
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to Deepgram")

        if not audio_chunk:
            raise ValueError("Audio chunk cannot be empty")

        try:
            self._connection.send(audio_chunk)
            logger.debug("Sent audio chunk: %d bytes", len(audio_chunk))

        except Exception as e:
            logger.error("Failed to send audio chunk: %s", str(e))
            self._state = ConnectionState.ERROR
            if self._error_callback:
                self._error_callback(e)
            raise

    async def disconnect(self) -> None:
        """
        Close WebSocket connection and cleanup resources.
        """
        if self._state == ConnectionState.DISCONNECTED:
            logger.debug("Already disconnected")
            return

        try:
            if self._connection:
                await self._connection.__aexit__(None, None, None)
                self._connection = None

            self._client = None
            self._state = ConnectionState.DISCONNECTED

            logger.info("Disconnected from Deepgram (session: %s)", self._session_id)

        except Exception as e:
            logger.error("Error during disconnect: %s", str(e))
            self._state = ConnectionState.ERROR
            raise
        finally:
            self._session_id = None

    def _on_open(self, *args, **kwargs) -> None:
        """
        Internal handler for connection open events from Deepgram.
        """
        logger.info("Deepgram connection opened")

    def _on_message(self, *args, **kwargs) -> None:
        """
        Internal handler for message events from Deepgram.

        Processes both interim and final transcripts and invokes appropriate callbacks.
        """
        try:
            # Extract message from args
            if not args:
                logger.warning("Received message event with no data")
                return

            message = args[0]

            # Check if this is a transcript message
            if not hasattr(message, 'type') or message.type != 'Results':
                logger.debug("Received non-transcript message: %s", getattr(message, 'type', 'unknown'))
                return

            # Extract transcript data
            if not hasattr(message, 'channel'):
                logger.warning("Message has no channel data")
                return

            channel = message.channel
            if not hasattr(channel, 'alternatives') or not channel.alternatives:
                logger.debug("No alternatives in message")
                return

            # Get best alternative
            alternative = channel.alternatives[0]
            transcript = alternative.transcript if hasattr(alternative, 'transcript') else ""

            if not transcript or transcript.strip() == "":
                logger.debug("Empty transcript received")
                return

            # Extract metadata
            metadata = {
                "confidence": getattr(alternative, 'confidence', 0.0),
                "is_final": getattr(message, 'is_final', False),
                "speech_final": getattr(message, 'speech_final', False),
                "duration": getattr(message, 'duration', 0.0),
                "start": getattr(message, 'start', 0.0),
            }

            # Invoke appropriate callback
            if metadata["is_final"]:
                logger.info("Final transcript: %s (confidence: %.2f)", transcript, metadata["confidence"])
                if self._final_callback:
                    self._final_callback(transcript, metadata)
            else:
                logger.debug("Interim transcript: %s", transcript)
                if self._interim_callback:
                    self._interim_callback(transcript, metadata)

        except Exception as e:
            logger.error("Error processing message: %s", str(e))
            if self._error_callback:
                self._error_callback(e)

    def _on_error(self, *args, **kwargs) -> None:
        """
        Internal handler for error events from Deepgram.
        """
        try:
            error = args[0] if args else "Unknown error"
            logger.error("Deepgram error: %s", error)

            self._state = ConnectionState.ERROR

            if self._error_callback:
                self._error_callback(Exception(str(error)))

        except Exception as e:
            logger.error("Error in error handler: %s", str(e))

    def _on_close(self, *args, **kwargs) -> None:
        """
        Internal handler for connection close events from Deepgram.
        """
        logger.info("Deepgram connection closed")
        self._state = ConnectionState.DISCONNECTED
