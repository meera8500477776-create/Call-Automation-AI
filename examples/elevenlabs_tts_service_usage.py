"""
Example usage of ElevenLabs TTS Service.

Demonstrates streaming synthesis, emotional context application,
voice settings retrieval, and voice listing.
"""

import asyncio
import os
from pathlib import Path

from src.voice_agent.services.elevenlabs_tts_service import (
    ElevenLabsTTSService,
    TTSConfig,
    VoiceConfig,
)


async def main():
    """Demonstrate ElevenLabs TTS service usage."""
    
    # Get API key from environment
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("Error: ELEVENLABS_API_KEY environment variable not set")
        print("Please set it with: export ELEVENLABS_API_KEY='your-api-key'")
        return
    
    # Create voice configuration
    voice_config = VoiceConfig(
        provider="elevenlabs",
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice (default)
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True,
        model_id="eleven_flash_v2_5",
    )
    
    # Create TTS configuration
    tts_config = TTSConfig(
        api_key=api_key,
        default_voice_config=voice_config,
        chunk_size=2048,
        optimize_streaming_latency=3,
        output_format="mp3_44100_128",
    )
    
    # Initialize TTS service
    tts_service = ElevenLabsTTSService(tts_config)
    print("✓ ElevenLabs TTS service initialized")
    
    # Example 1: List available voices
    print("\n=== Example 1: List Available Voices ===")
    try:
        voices = await tts_service.list_available_voices()
        print(f"Found {len(voices)} available voices:")
        for voice in voices[:5]:  # Show first 5
            print(f"  - {voice['name']} (ID: {voice['voice_id']})")
    except Exception as e:
        print(f"Error listing voices: {e}")
    
    # Example 2: Get voice settings
    print("\n=== Example 2: Get Voice Settings ===")
    try:
        settings = await tts_service.get_voice_settings(voice_config.voice_id)
        print(f"Voice settings for {voice_config.voice_id}:")
        print(f"  - Stability: {settings.stability}")
        print(f"  - Similarity Boost: {settings.similarity_boost}")
        print(f"  - Style: {settings.style}")
        print(f"  - Speaker Boost: {settings.use_speaker_boost}")
    except Exception as e:
        print(f"Error getting voice settings: {e}")
    
    # Example 3: Apply emotional context
    print("\n=== Example 3: Apply Emotional Context ===")
    
    # Test empathetic tag
    text1 = "I understand your concern about the appointment."
    tagged1 = tts_service.apply_emotional_context(text1, ["empathetic"])
    print(f"Original: {text1}")
    print(f"With empathetic tag: {tagged1}")
    
    # Test enthusiastic tag
    text2 = "That's wonderful news! I'm so glad to hear it."
    tagged2 = tts_service.apply_emotional_context(text2, ["enthusiastic"])
    print(f"\nOriginal: {text2}")
    print(f"With enthusiastic tag: {tagged2}")
    
    # Test automatic sigh for apology
    text3 = "I'm sorry about the confusion."
    tagged3 = tts_service.apply_emotional_context(text3, [])
    print(f"\nOriginal: {text3}")
    print(f"With automatic sigh: {tagged3}")
    
    # Example 4: Streaming synthesis
    print("\n=== Example 4: Streaming Synthesis ===")
    
    # Text to synthesize
    text = "Hello! I'm here to help you schedule your appointment. What day works best for you?"
    
    # Apply emotional context
    text_with_emotion = tts_service.apply_emotional_context(text, ["enthusiastic"])
    print(f"Synthesizing: {text_with_emotion}")
    
    try:
        # Create output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        # Stream audio to file
        output_file = output_dir / "tts_example.mp3"
        
        print(f"Streaming audio to: {output_file}")
        
        chunk_count = 0
        total_bytes = 0
        
        with open(output_file, "wb") as f:
            async for chunk in tts_service.synthesize_streaming(text_with_emotion):
                f.write(chunk)
                chunk_count += 1
                total_bytes += len(chunk)
        
        print(f"✓ Synthesis complete!")
        print(f"  - Chunks received: {chunk_count}")
        print(f"  - Total bytes: {total_bytes}")
        print(f"  - Output file: {output_file}")
        
    except Exception as e:
        print(f"Error during synthesis: {e}")
    
    # Example 5: Custom voice configuration
    print("\n=== Example 5: Custom Voice Configuration ===")
    
    # Create custom voice config with different settings
    custom_voice_config = VoiceConfig(
        provider="elevenlabs",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        stability=0.7,  # More stable
        similarity_boost=0.9,  # Higher similarity
        style=0.3,  # Some style exaggeration
        use_speaker_boost=True,
    )
    
    text = "This is a test with custom voice settings."
    
    try:
        output_file = output_dir / "tts_custom_voice.mp3"
        print(f"Synthesizing with custom settings to: {output_file}")
        
        with open(output_file, "wb") as f:
            async for chunk in tts_service.synthesize_streaming(text, custom_voice_config):
                f.write(chunk)
        
        print(f"✓ Custom voice synthesis complete!")
        
    except Exception as e:
        print(f"Error during custom synthesis: {e}")
    
    print("\n=== All Examples Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
