"""
ElevenLabs TTS Service implementation.

This module provides an ElevenLabs-based Text-to-Speech service with streaming
synthesis, emotional context support via audio tags, voice cloning, and
configurable voice settings for natural, empathetic speech generation.

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**
"""

import logging
import re
from typing import AsyncIterator, List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from elevenlabs import AsyncElevenLabs
from elevenlabs.types import VoiceSettings as ElevenLabsVoiceSettings


logger = logging.getLogger(__name__)


class AudioTag(Enum):
    """Supported ElevenLabs audio tags for emotional context."""
    SIGH = "[sigh]"
    LAUGHS = "[laughs]"
    EXCITED = "<excited>"
    EMPATHETIC = "<empathetic>"
    QUESTIONING = "<questioning>"
    UNDERSTANDING = "<understanding>"


@dataclass
class VoiceConfig:
    """
    Configuration for TTS voice synthesis.
    
    Attributes:
        provider: TTS provider ("elevenlabs" or "openai")
        voice_id: Provider-specific voice identifier
        stability: Voice stability (0.0-1.0, ElevenLabs only)
        similarity_boost: Similarity boost (0.0-1.0, ElevenLabs only)
        style: Style exaggeration (0.0-1.0, ElevenLabs only)
        use_speaker_boost: Enable speaker boost (ElevenLabs only)
        model_id: Model identifier (default: "eleven_flash_v2_5")
    """
    provider: str
    voice_id: str
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True
    model_id: str = "eleven_flash_v2_5"

    def __post_init__(self):
        """Validate configuration."""
        if self.provider not in ["elevenlabs", "openai"]:
            raise ValueError(f"Invalid provider: {self.provider}")
        if not 0.0 <= self.stability <= 1.0:
            raise ValueError(f"Stability must be between 0.0 and 1.0, got {self.stability}")
        if not 0.0 <= self.similarity_boost <= 1.0:
            raise ValueError(f"Similarity boost must be between 0.0 and 1.0, got {self.similarity_boost}")
        if not 0.0 <= self.style <= 1.0:
            raise ValueError(f"Style must be between 0.0 and 1.0, got {self.style}")


@dataclass
class VoiceSettings:
    """
    Voice settings retrieved from ElevenLabs.
    
    Attributes:
        stability: Voice stability setting
        similarity_boost: Similarity boost setting
        style: Style exaggeration setting
        use_speaker_boost: Speaker boost enabled
    """
    stability: float
    similarity_boost: float
    style: float = 0.0
    use_speaker_boost: bool = True


@dataclass
class TTSConfig:
    """
    Configuration for TTS service.
    
    Attributes:
        api_key: ElevenLabs API key
        default_voice_config: Default voice configuration
        chunk_size: Audio chunk size for streaming (bytes)
        optimize_streaming_latency: Optimize for low latency
        output_format: Audio output format
    """
    api_key: str
    default_voice_config: VoiceConfig
    chunk_size: int = 2048
    optimize_streaming_latency: int = 3
    output_format: str = "mp3_44100_128"


class ElevenLabsTTSService:
    """
    ElevenLabs Text-to-Speech service wrapper.
    
    Provides streaming synthesis with emotional context support,
    voice cloning capabilities, and configurable voice settings
    for natural, empathetic speech generation.
    """

    def __init__(self, config: TTSConfig):
        """
        Initialize ElevenLabs TTS service.
        
        Args:
            config: TTS configuration including API key and voice settings
        """
        self.config = config
        self.client = AsyncElevenLabs(api_key=config.api_key)
        
        # Cache for voice settings
        self._voice_settings_cache: Dict[str, VoiceSettings] = {}
        
        logger.info(
            "ElevenLabs TTS service initialized (voice: %s, model: %s)",
            config.default_voice_config.voice_id,
            config.default_voice_config.model_id,
        )

    async def synthesize_streaming(
        self,
        text: str,
        voice_config: Optional[VoiceConfig] = None,
    ) -> AsyncIterator[bytes]:
        """
        Synthesize text to speech with streaming output.
        
        Generates audio chunks as they become available for minimal latency.
        Supports ElevenLabs audio tags for emotional context.
        
        Args:
            text: Text to synthesize
            voice_config: Voice configuration (uses default if None)
            
        Yields:
            Audio data chunks as bytes
            
        Raises:
            RuntimeError: If synthesis fails
            ValueError: If text is empty
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        config = voice_config or self.config.default_voice_config
        
        if config.provider != "elevenlabs":
            raise ValueError(f"ElevenLabs service requires provider='elevenlabs', got '{config.provider}'")
        
        try:
            # Create voice settings
            voice_settings = ElevenLabsVoiceSettings(
                stability=config.stability,
                similarity_boost=config.similarity_boost,
                style=config.style,
                use_speaker_boost=config.use_speaker_boost,
            )
            
            logger.info(
                "Starting streaming synthesis (voice: %s, model: %s, length: %d chars)",
                config.voice_id,
                config.model_id,
                len(text),
            )
            
            # Stream audio generation
            audio_stream = await self.client.text_to_speech.convert(
                voice_id=config.voice_id,
                text=text,
                model_id=config.model_id,
                voice_settings=voice_settings,
                optimize_streaming_latency=self.config.optimize_streaming_latency,
                output_format=self.config.output_format,
            )
            
            # Yield audio chunks
            chunk_count = 0
            total_bytes = 0
            
            async for chunk in audio_stream:
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

    def apply_emotional_context(self, text: str, tags: List[str]) -> str:
        """
        Apply emotional context to text using audio tags.
        
        Inserts ElevenLabs audio tags into text to modulate tone and
        emotional delivery. Supports tags like [sigh], [laughs], <excited>.
        
        Args:
            text: Original text
            tags: List of emotional tags to apply
            
        Returns:
            Text with audio tags inserted
        """
        # Start with original text
        modified_text = text
        
        # Start with original text
        modified_text = text
        
        # Map common emotional indicators to audio tags
        tag_mapping = {
            "empathetic": AudioTag.EMPATHETIC.value,
            "enthusiastic": AudioTag.EXCITED.value,
            "understanding": AudioTag.UNDERSTANDING.value,
            "questioning": AudioTag.QUESTIONING.value,
        }
        
        # Apply tags based on content and emotional indicators (only if tags provided)
        if tags:
            for tag in tags:
                audio_tag = tag_mapping.get(tag.lower())
                if not audio_tag:
                    continue
                
                # Insert tag at appropriate position
                if tag.lower() == "empathetic":
                    # Add empathetic tag at the beginning
                    if not modified_text.startswith("<"):
                        modified_text = f"{audio_tag} {modified_text}"
                elif tag.lower() == "enthusiastic":
                    # Add excited tag around positive statements
                    if any(word in modified_text.lower() for word in ["great", "wonderful", "excellent", "perfect"]):
                        # Wrap the positive word
                        for word in ["great", "wonderful", "excellent", "perfect"]:
                            pattern = re.compile(rf'\b({word})\b', re.IGNORECASE)
                            modified_text = pattern.sub(rf'{audio_tag}\1{audio_tag.replace("<", "</")}', modified_text, count=1)
                            if audio_tag in modified_text:
                                break
                elif tag.lower() == "understanding":
                    # Add understanding tag before acknowledgment phrases
                    for phrase in ["I understand", "I see", "I hear you"]:
                        if phrase.lower() in modified_text.lower():
                            pattern = re.compile(rf'({re.escape(phrase)})', re.IGNORECASE)
                            modified_text = pattern.sub(rf'{audio_tag}\1{audio_tag.replace("<", "</")}', modified_text, count=1)
                            break
        
        # Add special audio effects for specific phrases (always check, regardless of tags)
        if any(word in text.lower() for word in ["sorry", "apologize", "unfortunately"]):
            # Add sigh before apologies
            if AudioTag.SIGH.value not in modified_text:
                modified_text = f"{AudioTag.SIGH.value} {modified_text}"
        
        logger.debug("Applied emotional tags: %s -> %s", tags, modified_text)
        
        return modified_text

    async def clone_voice(
        self,
        audio_samples: List[bytes],
        voice_name: str,
        description: Optional[str] = None,
    ) -> str:
        """
        Clone a voice from audio samples.
        
        Creates a custom voice in ElevenLabs from provided audio samples.
        Requires at least 1 minute of clear audio for best results.
        
        Args:
            audio_samples: List of audio file data (bytes)
            voice_name: Name for the cloned voice
            description: Optional description of the voice
            
        Returns:
            Voice ID of the cloned voice
            
        Raises:
            RuntimeError: If voice cloning fails
            ValueError: If audio samples are invalid
        """
        if not audio_samples:
            raise ValueError("At least one audio sample is required")
        
        if not voice_name or not voice_name.strip():
            raise ValueError("Voice name cannot be empty")
        
        try:
            logger.info(
                "Starting voice cloning (name: %s, samples: %d)",
                voice_name,
                len(audio_samples),
            )
            
            # Note: The actual ElevenLabs API for voice cloning requires file uploads
            # This is a simplified interface - in production, you'd need to:
            # 1. Save audio samples to temporary files
            # 2. Use the voice cloning API endpoint
            # 3. Clean up temporary files
            
            # For now, we'll raise NotImplementedError with guidance
            raise NotImplementedError(
                "Voice cloning requires file upload implementation. "
                "Please use the ElevenLabs web interface or implement file handling."
            )
            
        except NotImplementedError:
            raise
        except Exception as e:
            logger.error("Failed to clone voice: %s", str(e))
            raise RuntimeError(f"Voice cloning failed: {str(e)}") from e

    async def get_voice_settings(self, voice_id: str) -> VoiceSettings:
        """
        Retrieve voice settings for a specific voice.
        
        Gets the current configuration for a voice, including stability,
        similarity boost, and style settings. Results are cached.
        
        Args:
            voice_id: ElevenLabs voice identifier
            
        Returns:
            Voice settings configuration
            
        Raises:
            RuntimeError: If retrieval fails
        """
        # Check cache first
        if voice_id in self._voice_settings_cache:
            logger.debug("Returning cached voice settings for: %s", voice_id)
            return self._voice_settings_cache[voice_id]
        
        try:
            logger.info("Retrieving voice settings for: %s", voice_id)
            
            # Get voice details from ElevenLabs
            voice = await self.client.voices.get(voice_id=voice_id)
            
            # Extract settings
            settings = VoiceSettings(
                stability=voice.settings.stability if voice.settings else 0.5,
                similarity_boost=voice.settings.similarity_boost if voice.settings else 0.75,
                style=voice.settings.style if voice.settings else 0.0,
                use_speaker_boost=voice.settings.use_speaker_boost if voice.settings else True,
            )
            
            # Cache the settings
            self._voice_settings_cache[voice_id] = settings
            
            logger.info(
                "Retrieved voice settings (stability: %.2f, similarity: %.2f, style: %.2f)",
                settings.stability,
                settings.similarity_boost,
                settings.style,
            )
            
            return settings
            
        except Exception as e:
            logger.error("Failed to retrieve voice settings: %s", str(e))
            raise RuntimeError(f"Failed to get voice settings: {str(e)}") from e

    async def list_available_voices(self) -> List[Dict[str, Any]]:
        """
        List all available voices from ElevenLabs.
        
        Returns:
            List of voice information dictionaries
            
        Raises:
            RuntimeError: If listing fails
        """
        try:
            logger.info("Listing available voices")
            
            voices_response = await self.client.voices.get_all()
            
            voices = []
            for voice in voices_response.voices:
                voices.append({
                    "voice_id": voice.voice_id,
                    "name": voice.name,
                    "category": voice.category if hasattr(voice, "category") else None,
                    "description": voice.description if hasattr(voice, "description") else None,
                    "labels": voice.labels if hasattr(voice, "labels") else {},
                })
            
            logger.info("Retrieved %d available voices", len(voices))
            
            return voices
            
        except Exception as e:
            logger.error("Failed to list voices: %s", str(e))
            raise RuntimeError(f"Failed to list voices: {str(e)}") from e
