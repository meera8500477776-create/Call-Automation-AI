"""
Example usage of TTS Provider Factory.

Demonstrates how to use the TTS factory to create TTS service instances
with automatic provider selection and failover capabilities.

**Validates: Requirements 5.6**
"""

import asyncio
import os
from dotenv import load_dotenv

from src.voice_agent.services.tts_factory import TTSProviderFactory
from src.voice_agent.services.elevenlabs_tts_service import VoiceConfig


async def main():
    """Demonstrate TTS factory usage."""
    # Load environment variables
    load_dotenv()
    
    # Create factory with API keys
    factory = TTSProviderFactory(
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        enable_failover=True,
    )
    
    print("=== TTS Provider Factory Demo ===\n")
    
    # Example 1: Create ElevenLabs TTS service
    print("1. Creating ElevenLabs TTS service...")
    elevenlabs_config = VoiceConfig(
        provider="elevenlabs",
        voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
        stability=0.5,
        similarity_boost=0.75,
        style=0.0,
        use_speaker_boost=True,
    )
    
    try:
        elevenlabs_service = await factory.create_tts_service(elevenlabs_config)
        print(f"   ✓ Created: {type(elevenlabs_service).__name__}")
        
        # Test synthesis
        text = "Hello! This is a test of the ElevenLabs text-to-speech service."
        print(f"   Synthesizing: '{text}'")
        
        chunk_count = 0
        async for chunk in elevenlabs_service.synthesize_streaming(text):
            chunk_count += 1
        
        print(f"   ✓ Generated {chunk_count} audio chunks\n")
        
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Example 2: Create OpenAI TTS service
    print("2. Creating OpenAI TTS service...")
    openai_config = VoiceConfig(
        provider="openai",
        voice_id="alloy",
    )
    
    try:
        openai_service = await factory.create_tts_service(openai_config)
        print(f"   ✓ Created: {type(openai_service).__name__}")
        
        # Test synthesis
        text = "Hello! This is a test of the OpenAI text-to-speech service."
        print(f"   Synthesizing: '{text}'")
        
        chunk_count = 0
        async for chunk in openai_service.synthesize_streaming(text):
            chunk_count += 1
        
        print(f"   ✓ Generated {chunk_count} audio chunks\n")
        
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Example 3: Check provider health
    print("3. Checking provider health...")
    health_status = await factory.get_all_provider_health()
    
    for provider, health in health_status.items():
        print(f"   {provider}: {health.value}")
    
    print()
    
    # Example 4: Demonstrate failover
    print("4. Testing failover (ElevenLabs → OpenAI)...")
    print("   Note: Failover occurs automatically if ElevenLabs is unavailable")
    
    # Create service with failover enabled
    service = await factory.create_tts_service(
        elevenlabs_config,
        provider_config={
            "fallback_voice": "nova",  # Use Nova voice for OpenAI fallback
        }
    )
    
    print(f"   ✓ Service created: {type(service).__name__}")
    print("   (Will automatically failover to OpenAI if ElevenLabs fails)\n")
    
    # Example 5: Custom provider configuration
    print("5. Creating service with custom configuration...")
    custom_config = VoiceConfig(
        provider="elevenlabs",
        voice_id="21m00Tcm4TlvDq8ikWAM",
        stability=0.7,  # More stable
        similarity_boost=0.8,  # Higher similarity
        style=0.2,  # Slight style exaggeration
        use_speaker_boost=True,
    )
    
    provider_config = {
        "chunk_size": 4096,  # Larger chunks
        "optimize_streaming_latency": 4,  # Maximum optimization
        "output_format": "mp3_44100_128",
    }
    
    try:
        custom_service = await factory.create_tts_service(
            custom_config,
            provider_config=provider_config,
        )
        print(f"   ✓ Created with custom settings: {type(custom_service).__name__}")
        print(f"   Chunk size: {custom_service.config.chunk_size}")
        print(f"   Latency optimization: {custom_service.config.optimize_streaming_latency}")
        print(f"   Output format: {custom_service.config.output_format}\n")
        
    except Exception as e:
        print(f"   ✗ Error: {e}\n")
    
    # Example 6: Disable failover
    print("6. Creating factory with failover disabled...")
    no_failover_factory = TTSProviderFactory(
        elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        enable_failover=False,
    )
    
    print("   ✓ Factory created with failover disabled")
    print("   (Will raise error if primary provider fails)\n")
    
    print("=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
