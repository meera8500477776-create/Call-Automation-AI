"""
VAD Module wrapper for Silero VAD.

This module provides Voice Activity Detection using Silero VAD for speech boundary
detection, turn-taking, and barge-in capability in voice conversations.
"""

import logging
import torch
import numpy as np
from typing import Callable, Optional, List
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class SpeechState(Enum):
    """Speech detection states."""
    SILENCE = "silence"
    SPEECH = "speech"
    UNKNOWN = "unknown"


@dataclass
class VADResult:
    """
    Result of VAD processing.
    
    Attributes:
        is_speech: Whether speech was detected in the audio chunk
        confidence: Detection confidence score (0.0 to 1.0)
        timestamp: Audio timestamp in seconds
    """
    is_speech: bool
    confidence: float
    timestamp: float


@dataclass
class VADConfig:
    """
    Configuration for VAD module.
    
    Attributes:
        sample_rate: Audio sample rate in Hz (8000 or 16000)
        threshold: Speech detection threshold (0.0 to 1.0, default 0.5)
        min_speech_duration_ms: Minimum speech duration to trigger speech_start (milliseconds)
        min_silence_duration_ms: Minimum silence duration to trigger speech_end (milliseconds)
        window_size_samples: Number of samples per processing window
    """
    sample_rate: int = 16000
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    window_size_samples: int = 512


class VADModule:
    """
    Voice Activity Detection module using Silero VAD.
    
    Provides real-time speech detection for turn-taking and barge-in functionality.
    Detects speech boundaries and invokes callbacks for speech start/end events.
    """

    def __init__(self, config: VADConfig):
        """
        Initialize VAD module.
        
        Args:
            config: VAD configuration including sample rate and thresholds
        """
        self.config = config
        self._state = SpeechState.SILENCE
        self._current_timestamp = 0.0
        
        # Callback handlers
        self._speech_start_callback: Optional[Callable[[float], None]] = None
        self._speech_end_callback: Optional[Callable[[float], None]] = None
        
        # Speech/silence duration tracking
        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_start_time: Optional[float] = None
        
        # Load Silero VAD model
        self._model = None
        self._load_model()
        
        logger.info(
            "VAD module initialized (sample_rate: %d, threshold: %.2f)",
            config.sample_rate,
            config.threshold,
        )

    def _load_model(self) -> None:
        """
        Load Silero VAD model from torch hub.
        
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            # Load Silero VAD model
            self._model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            
            # Set model to evaluation mode
            self._model.eval()
            
            logger.info("Silero VAD model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load Silero VAD model: %s", str(e))
            raise RuntimeError(f"Failed to load Silero VAD model: {e}") from e

    def on_speech_start(self, callback: Callable[[float], None]) -> None:
        """
        Register callback for speech start events.
        
        Args:
            callback: Function to call with timestamp when speech starts
        """
        self._speech_start_callback = callback
        logger.debug("Speech start callback registered")

    def on_speech_end(self, callback: Callable[[float], None]) -> None:
        """
        Register callback for speech end events.
        
        Args:
            callback: Function to call with timestamp when speech ends
        """
        self._speech_end_callback = callback
        logger.debug("Speech end callback registered")

    def set_sensitivity(self, threshold: float) -> None:
        """
        Adjust speech detection sensitivity.
        
        Args:
            threshold: Detection threshold (0.0 to 1.0)
                      Lower values = more sensitive (detect more speech)
                      Higher values = less sensitive (require clearer speech)
                      
        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        
        self.config.threshold = threshold
        logger.info("VAD sensitivity updated: threshold=%.2f", threshold)

    def process_audio(self, audio_chunk: bytes) -> VADResult:
        """
        Process audio chunk and detect speech activity.
        
        Args:
            audio_chunk: Raw audio data bytes (16-bit PCM)
            
        Returns:
            VADResult containing detection result and metadata
            
        Raises:
            ValueError: If audio chunk is invalid
            RuntimeError: If model is not loaded
        """
        if not audio_chunk:
            raise ValueError("Audio chunk cannot be empty")
        
        if self._model is None:
            raise RuntimeError("VAD model not loaded")
        
        try:
            # Convert bytes to numpy array (16-bit PCM)
            audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Convert to float32 normalized to [-1, 1]
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_float32)
            
            # Get speech probability from model
            with torch.no_grad():
                speech_prob = self._model(audio_tensor, self.config.sample_rate).item()
            
            # Determine if speech is detected
            is_speech = speech_prob >= self.config.threshold
            
            # Update timestamp
            chunk_duration = len(audio_int16) / self.config.sample_rate
            self._current_timestamp += chunk_duration
            
            # Create result
            result = VADResult(
                is_speech=is_speech,
                confidence=speech_prob,
                timestamp=self._current_timestamp,
            )
            
            # Update state and trigger callbacks
            self._update_state(result)
            
            logger.debug(
                "VAD processed: is_speech=%s, confidence=%.3f, timestamp=%.3f",
                is_speech,
                speech_prob,
                self._current_timestamp,
            )
            
            return result
            
        except Exception as e:
            logger.error("Error processing audio chunk: %s", str(e))
            raise

    def _update_state(self, result: VADResult) -> None:
        """
        Update internal state and trigger callbacks based on VAD result.
        
        Args:
            result: VAD processing result
        """
        # Calculate actual chunk duration in milliseconds
        # We track based on actual audio processed, not fixed window sizes
        
        if result.is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            
            # Transition from silence to speech
            if self._state == SpeechState.SILENCE:
                # Check if we've accumulated enough speech
                # Since we don't know chunk size in advance, we use a simple counter
                # Assuming typical chunk is ~500ms, we need at least 1 chunk for 250ms threshold
                if self._speech_frames >= 1:
                    self._state = SpeechState.SPEECH
                    self._speech_start_time = result.timestamp
                    
                    logger.info("Speech started at %.3f seconds", result.timestamp)
                    
                    if self._speech_start_callback:
                        self._speech_start_callback(result.timestamp)
        else:
            self._silence_frames += 1
            self._speech_frames = 0
            
            # Transition from speech to silence
            if self._state == SpeechState.SPEECH:
                # Similar logic for silence - need at least 1 chunk
                if self._silence_frames >= 1:
                    self._state = SpeechState.SILENCE
                    
                    speech_duration = result.timestamp - (self._speech_start_time or 0.0)
                    logger.info(
                        "Speech ended at %.3f seconds (duration: %.3f seconds)",
                        result.timestamp,
                        speech_duration,
                    )
                    
                    if self._speech_end_callback:
                        self._speech_end_callback(result.timestamp)
                    
                    self._speech_start_time = None

    def reset(self) -> None:
        """
        Reset VAD state and counters.
        
        Useful when starting a new conversation or after a long pause.
        """
        self._state = SpeechState.SILENCE
        self._current_timestamp = 0.0
        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_start_time = None
        
        logger.info("VAD state reset")

    @property
    def current_state(self) -> SpeechState:
        """Get current speech detection state."""
        return self._state

    @property
    def is_speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self._state == SpeechState.SPEECH
