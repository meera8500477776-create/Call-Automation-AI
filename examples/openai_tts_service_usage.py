"""
Example usage of OpenAI TTS Service.

Demonstrates streaming synthesis, voice selection, and fallback usage
for the OpenAI TTS service wrapper.

**Validates: Requirements 5.6**
"""

import asyncio
import os
from pathlib import Path

from src.voice_agent.services.openai_tts_service import (
    OpenAITTSService,
    OpenAITTSConfig,
)
from src.voice_agent.services.elevenlabs_tts_service import VoiceConfig


async def example_basic_synthesis():
    """Example: Basic text-to-speech synthesis."""
    print("\n=== Basic Synthesis ===")
    
    # Configure OpenAI TTS
    voice_config = VoiceConfig(
        provider="openai",
        voice_id="alloy",
    )
    
    config = OpenAITTSConfig(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        default_voice_config=voice_config,
        model="tts-1",  # Use "tts-1-hd" for higher quality
    )
    
    service = OpenAITTSService(config)
    
    # Synthesize text
    text = "Hello! This is a test of the OpenAI text-to-speech service."
    
    print(f"Synthesizing: {text}")
    
    # Collect audio chunks
    audio_chunks = []
    async for chunk in service.synthesize_streaming(text):
        audio_chunks.append(chunk)
        print(f"  Received chunk: {len(chunk)} bytes")
    
    print(f"Total chunks: {len(audio_chunks)}")
    print(f"Total audio size: {sum(len(c) for c in audio_chunks)} bytes")


async def example_different_voices():
    """Example: Using different OpenAI voices."""
    print("\n=== Different Voices ===")
    
    # Available OpenAI voices
    voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    config = OpenAITTSConfig(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        default_voice_config=VoiceConfig(provider="openai", voice_id="alloy"),
    )
    
    service = OpenAITTSService(config)
    
    text = "This is a test with different voices."
    
    for voice_id in voices:
        print(f"\nVoice: {voice_id}")
        
        voice_config = VoiceConfig(
            provider="openai",
            voice_id=voice_id,
        )
        
        chunk_count = 0
        async for chunk in service.synthesize_streaming(text, voice_config):
            chunk_count += 1
        
        print(f"  Generated {chunk_count} chunks")


async def example_list_voices():
    """Example: List available voices."""
    print("\n=== Available Voices ===")
    
    config = OpenAITTSConfig(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        default_voice_config=VoiceConfig(provider="openai", voice_id="alloy"),
    )
    
    service = OpenAITTSService(config)
    
    voices = await service.list_available_voices()
    
    print(f"Found {len(voices)} voices:")
    for voice in voices:
        print(f"  - {voice['name']} ({voice['voice_id']}): {voice['description']}")


async def example_fallback_usage():
    """Example: Using OpenAI TTS as fallback for ElevenLabs."""
    print("\n=== Fallback Usage ===")
    
    # Primary: ElevenLabs configuration
    elevenlabs_config = VoiceConfig(
        provider="elevenlabs",
        voice_id="some_elevenlabs_voice",
    )
    
    # Fallback: OpenAI configuration
    openai_config = VoiceConfig(
        provider="openai",
        voice_id="nova",
    )
    
    # In production, you would try ElevenLabs first, then fall back to OpenAI
    print("Primary provider: ElevenLabs")
    print("Fallback provider: OpenAI")
    
    # Simulate fallback scenario
    use_fallback = True  # Set to True to simulate ElevenLabs failure
    
    if use_fallback:
        print("\nUsing fallback provider (OpenAI)...")
        
        config = OpenAITTSConfig(
            api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
            default_voice_config=openai_config,
        )
        
        service = OpenAITTSService(config)
        
        text = "This is using the fallback OpenAI TTS provider."
        
        chunk_count = 0
        async for chunk in service.synthesize_streaming(text):
            chunk_count += 1
        
        print(f"Successfully generated {chunk_count} chunks with fallback provider")


async def example_emotional_context():
    """Example: Emotional context (note: not supported by OpenAI)."""
    print("\n=== Emotional Context ===")
    
    config = OpenAITTSConfig(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        default_voice_config=VoiceConfig(provider="openai", voice_id="nova"),
    )
    
    service = OpenAITTSService(config)
    
    text = "I understand your concern and I'm here to help."
    tags = ["empathetic", "understanding"]
    
    # Note: OpenAI TTS doesn't support emotional tags
    modified_text = service.apply_emotional_context(text, tags)
    
    print(f"Original text: {text}")
    print(f"Tags: {tags}")
    print(f"Modified text: {modified_text}")
    print("Note: OpenAI TTS does not support emotional tags, text is unchanged")


async def example_save_to_file():
    """Example: Save synthesized audio to file."""
    print("\n=== Save to File ===")
    
    config = OpenAITTSConfig(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        default_voice_config=VoiceConfig(provider="openai", voice_id="shimmer"),
    )
    
    service = OpenAITTSService(config)
    
    text = "This audio will be saved to a file."
    output_file = Path("output_openai_tts.mp3")
    
    print(f"Synthesizing and saving to: {output_file}")
    
    # Collect and save audio
    with open(output_file, "wb") as f:
        async for chunk in service.synthesize_streaming(text):
            f.write(chunk)
    
    print(f"Audio saved successfully ({output_file.stat().st_size} bytes)")


async def example_high_quality_synthesis():
    """Example: High-quality synthesis with tts-1-hd model."""
    print("\n=== High-Quality Synthesis ===")
    
    config = OpenAITTSConfig(
        api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
        default_voice_config=VoiceConfig(provider="openai", voice_id="fable"),
        model="tts-1-hd",  # Higher quality model
    )
    
    service = OpenAITTSService(config)
    
    text = "This uses the high-quality TTS model for better audio fidelity."
    
    print(f"Model: {config.model}")
    print(f"Synthesizing: {text}")
    
    chunk_count = 0
    total_bytes = 0
    async for chunk in service.synthesize_streaming(text):
        chunk_count += 1
        total_bytes += len(chunk)
    
    print(f"Generated {chunk_count} chunks ({total_bytes} bytes)")


async def main():
    """Run all examples."""
    print("OpenAI TTS Service Examples")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY not set in environment")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("\nRunning examples with placeholder key (will fail on actual API calls)")
    
    try:
        await example_basic_synthesis()
        await example_different_voices()
        await example_list_voices()
        await example_fallback_usage()
        await example_emotional_context()
        await example_save_to_file()
        await example_high_quality_synthesis()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure OPENAI_API_KEY is set correctly")


if __name__ == "__main__":
    asyncio.run(main())
