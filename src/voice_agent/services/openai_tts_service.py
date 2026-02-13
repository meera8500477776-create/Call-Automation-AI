"""
OpenAI TTS Service implementation.

This module provides an OpenAI-based Text-to-Speech service with streaming
synthesis and consistent interface with ElevenLabs provider for use as a
fallback TTS provider.

**Validates: Requirements 5.6**
"""

import logging
from typing import AsyncIterator, Optional, List
from dataclasses import dataclass

from openai import AsyncOpenAI

from .elevenlabs_tts_service import VoiceConfig, VoiceSettings


logger = logging.getLogger(__name__)


@dataclass
class OpenAITTSConfig:
    """
    Configuration for OpenAI TTS service.
    
    Attributes:
        api_key: OpenAI API key
        default_voice_config: Default voice configuration
        model: OpenAI TTS model (default: "tts-1" for speed, "tts-1-hd" for quality)
        chunk_size: Audio chunk size for streaming (bytes)
    """
    api_key: str
    default_voice_config: VoiceConfig
    model: str = "tts-1"
    chunk_size: int = 4096


class OpenAITTSService:
    """
    OpenAI Text-to-Speech service wrapper.
    
    Provides streaming synthesis with consistent interface to ElevenLabs
    for use as a fallback TTS provider. Supports OpenAI's TTS models
    with natural-sounding voices.
    """

    # OpenAI voice mapping
    VOICE_MAPPING = {
        "alloy": "alloy",
        "echo": "echo",
        "fable": "fable",
        "onyx": "onyx",
        "nova": "nova",
        "shimmer": "shimmer",
    }

    def __init__(self, config: OpenAITTSConfig):
        """
        Initialize OpenAI TTS service.
        
        Args:
            config: TTS configuration including API key and voice settings
        """
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key)
        
        logger.info(
            "OpenAI TTS service initialized (voice: %s, model: %s)",
            config.default_voice_config.voice_id,
            config.model,
        )

    async def synthesize_streaming(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech with streaming output.
        
        Generates audio chunks as they become available for minimal latency.
        Note: OpenAI TTS does not support SSML or emotional tags like ElevenLabs.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration (uses default if None)
            
        Yields:
            Audio data chunks as bytes
            
        Raises:
            RuntimeError: If synthesis fails
            ValueError: If text is empty or voice is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        config = voice_config or self.config.default_voice_config
        
        if config.provider != "openai":
            raise ValueError(f"OpenAI service requires provider='openai', got '{config.provider}'")
        
        # Map voice_id to OpenAI voice name
        voice_name = self._get_openai_voice(config.voice_id)
        
        try:
            logger.info(
                "Starting streaming synthesis (voice: %s, model: %s, length: %d chars)",
                voice_name,
                self.config.model,
                len(text),
            )
            
            # Stream audio generation
            response = await self.client.audio.speech.create(
                model=self.config.model,
                voice=voice_name,
                input=text,
                response_format="mp3",
            )
            
            # Stream audio chunks
            chunk_count = 0
            total_bytes = 0
            
            async for chunk in response.iter_bytes(chunk_size=self.config.chunk_size):
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    yield chunk
            
            logger.info(
                "Streaming synthesis complete (chunks: %d, total bytes: %d)",
                chunk_count,
                total_bytes,
            )
            
        except Exception as e:
            logger.error("Failed to synthesize speech: %s", str(e))
            raise RuntimeError(f"Speech synthesis failed: {str(e)}") from e

    def _get_openai_voice(self, voice_id: str) -> str:
        """
        Map voice_id to OpenAI voice name.
        
        Args:
            voice_id: Voice identifier (can be OpenAI voice name or custom mapping)
            
        Returns:
            OpenAI voice name
            
        Raises:
            ValueError: If voice is not supported
        """
        # Check if it's a direct OpenAI voice name
        if voice_id in self.VOICE_MAPPING:
            return self.VOICE_MAPPING[voice_id]
        
        # Default to "alloy" if not found
        logger.warning(
            "Voice '%s' not found in OpenAI voices, defaulting to 'alloy'",
            voice_id,
        )
        return "alloy"

    def apply_emotional_context(self, text: str, tags: List[str]) -> str:
        """
        Apply emotional context to text.
        
        Note: OpenAI TTS does not support audio tags or SSML like ElevenLabs.
        This method is provided for interface consistency but returns text unchanged.
        
        Args:
            text: Original text
            tags: List of emotional tags (ignored for OpenAI)
            
        Returns:
            Original text unchanged
        """
        if tags:
            logger.debug(
                "OpenAI TTS does not support emotional tags, ignoring: %s",
                tags,
            )
        return text

    async def clone_voice(
        self,
        audio_samples: List[bytes],
        voice_name: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Clone a voice from audio samples.
        
        Note: OpenAI TTS does not support voice cloning.
        This method is provided for interface consistency.
        
        Args:
            audio_samples: List of audio file data (bytes)
            voice_name: Name for the cloned voice
            description: Optional description of the voice
            
        Returns:
            Voice ID of the cloned voice
            
        Raises:
            NotImplementedError: OpenAI TTS does not support voice cloning
        """
        raise NotImplementedError(
            "OpenAI TTS does not support voice cloning. "
            "Use ElevenLabs for voice cloning capabilities."
        )

    async def get_voice_settings(self, voice_id: str) -> VoiceSettings:
        """
        Retrieve voice settings for a specific voice.
        
        Note: OpenAI TTS does not have configurable voice settings like ElevenLabs.
        This method returns default settings for interface consistency.
        
        Args:
            voice_id: OpenAI voice identifier
            
        Returns:
            Default voice settings
        """
        logger.debug(
            "OpenAI TTS does not have configurable voice settings, returning defaults for: %s",
            voice_id,
        )
        
        # Return default settings for consistency
        return VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.0,
            use_speaker_boost=False,
        )

    async def list_available_voices(self) -> List[dict]:
        """
        List all available voices from OpenAI.
        
        Returns:
            List of voice information dictionaries
        """
        logger.info("Listing available OpenAI voices")
        
        voices = [
            {
                "voice_id": "alloy",
                "name": "Alloy",
                "category": "standard",
                "description": "Neutral and balanced voice",
                "labels": {},
            },
            {
                "voice_id": "echo",
                "name": "Echo",
                "category": "standard",
                "description": "Male voice with clarity",
                "labels": {},
            },
            {
                "voice_id": "fable",
                "name": "Fable",
                "category": "standard",
                "description": "Expressive British accent",
                "labels": {},
            },
            {
                "voice_id": "onyx",
                "name": "Onyx",
                "category": "standard",
                "description": "Deep male voice",
                "labels": {},
            },
            {
                "voice_id": "nova",
                "name": "Nova",
                "category": "standard",
                "description": "Energetic female voice",
                "labels": {},
            },
            {
                "voice_id": "shimmer",
                "name": "Shimmer",
                "category": "standard",
                "description": "Warm female voice",
                "labels": {},
            },
        ]
        
        logger.info("Retrieved %d available voices", len(voices))
        
        return voices
