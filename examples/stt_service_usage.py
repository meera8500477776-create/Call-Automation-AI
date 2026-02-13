"""
Example usage of STT Service with Deepgram.

This example demonstrates how to use the STT service for real-time
speech-to-text transcription with WebSocket streaming.
"""

import asyncio
import os
from src.voice_agent.services import STTService, STTConfig


async def main():
    """Demonstrate STT service usage."""
    
    # Get API key from environment
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("Error: DEEPGRAM_API_KEY environment variable not set")
        return
    
    # Create STT configuration
    config = STTConfig(
        api_key=api_key,
        model="nova-2",
        language="en-US",
        interim_results=True,
        punctuate=True,
        smart_format=True,
    )
    
    # Create STT service
    stt_service = STTService(config)
    
    # Register callbacks
    def on_interim(transcript: str, metadata: dict):
        print(f"[INTERIM] {transcript}")
    
    def on_final(transcript: str, metadata: dict):
        print(f"[FINAL] {transcript} (confidence: {metadata['confidence']:.2f})")
    
    def on_error(error: Exception):
        print(f"[ERROR] {error}")
    
    stt_service.on_interim_result(on_interim)
    stt_service.on_final_result(on_final)
    stt_service.on_error(on_error)
    
    try:
        # Connect to Deepgram
        print("Connecting to Deepgram...")
        await stt_service.connect("example_session_001")
        print(f"Connected! State: {stt_service.state}")
        
        # In a real application, you would stream audio chunks here
        # For example:
        # while audio_stream.is_active():
        #     audio_chunk = audio_stream.read()
        #     await stt_service.send_audio(audio_chunk)
        
        print("\nSTT service is ready to receive audio chunks.")
        print("In a real application, you would stream audio data using:")
        print("  await stt_service.send_audio(audio_chunk)")
        
        # Keep connection alive for demonstration
        await asyncio.sleep(2)
        
    finally:
        # Disconnect
        print("\nDisconnecting...")
        await stt_service.disconnect()
        print(f"Disconnected! State: {stt_service.state}")


if __name__ == "__main__":
    asyncio.run(main())
