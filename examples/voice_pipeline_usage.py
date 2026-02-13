"""
Example usage of Voice Pipeline orchestrator.

This example demonstrates how to create and use the VoicePipeline to coordinate
STT, LLM, TTS, and VAD components for real-time voice processing with barge-in support.
"""

import asyncio
import os
from src.voice_agent.core.voice_pipeline import (
    VoicePipeline,
    PipelineConfig,
    PipelineState,
)
from src.voice_agent.services.stt_service import STTConfig
from src.voice_agent.services.llm_service import LLMConfig
from src.voice_agent.services.openai_llm_service import OpenAILLMService
from src.voice_agent.services.vad_module import VADConfig
from src.voice_agent.services.elevenlabs_tts_service import VoiceConfig


async def main():
    """Demonstrate Voice Pipeline usage."""
    
    # Configure STT service (Deepgram)
    stt_config = STTConfig(
        api_key=os.getenv("DEEPGRAM_API_KEY", "your_deepgram_api_key"),
        model="nova-2",
        language="en-US",
        interim_results=True,
        punctuate=True,
    )
    
    # Configure LLM service (OpenAI)
    llm_config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        max_tokens=150,
        system_prompt="You are a helpful voice assistant. Keep responses brief and conversational.",
    )
    llm_service = OpenAILLMService(llm_config)
    
    # Configure VAD module (Silero VAD)
    vad_config = VADConfig(
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )
    
    # Configure TTS voice (OpenAI TTS)
    voice_config = VoiceConfig(
        provider="openai",
        voice_id="alloy",
    )
    
    # Create pipeline configuration
    pipeline_config = PipelineConfig(
        session_id="demo_session_001",
        stt_config=stt_config,
        llm_service=llm_service,
        vad_config=vad_config,
        voice_config=voice_config,
        enable_barge_in=True,
        enable_interim_processing=False,
    )
    
    # Create voice pipeline
    pipeline = VoicePipeline(pipeline_config)
    
    # Register callbacks
    def on_transcript(transcript: str, is_final: bool):
        """Handle transcript events."""
        status = "FINAL" if is_final else "INTERIM"
        print(f"[{status}] User: {transcript}")
    
    def on_llm_response(response):
        """Handle LLM response events."""
        print(f"[LLM] Intent: {response.intent.name} (confidence: {response.intent.confidence:.2f})")
        print(f"[LLM] Agent: {response.text}")
        if response.parameters:
            print(f"[LLM] Parameters: {response.parameters}")
    
    def on_audio_generated(audio_chunk: bytes):
        """Handle TTS audio generation events."""
        print(f"[TTS] Generated audio chunk: {len(audio_chunk)} bytes")
        # In a real application, you would send this to the audio output device
    
    pipeline.on_transcript(on_transcript)
    pipeline.on_llm_response(on_llm_response)
    pipeline.on_audio_generated(on_audio_generated)
    
    try:
        # Initialize pipeline
        print("Initializing voice pipeline...")
        await pipeline.initialize()
        print(f"Pipeline state: {pipeline.state.value}")
        print(f"Barge-in enabled: {pipeline._barge_in_enabled}")
        
        # Simulate audio processing
        # In a real application, you would receive audio from a microphone or telephony system
        print("\nSimulating audio processing...")
        print("(In a real application, audio chunks would come from microphone/telephony)")
        
        # Example: Process a few audio chunks
        # Note: This is just a demonstration - real audio data would be needed
        # for actual transcription
        sample_audio = b'\x00\x01' * 8000  # 16-bit PCM, 0.5 seconds at 16kHz
        
        print("\nProcessing audio chunk 1...")
        # await pipeline.process_audio_chunk(sample_audio)
        
        # Demonstrate barge-in control
        print("\nDemonstrating barge-in control...")
        print(f"Barge-in currently: {'enabled' if pipeline._barge_in_enabled else 'disabled'}")
        
        pipeline.enable_barge_in(False)
        print(f"Barge-in now: {'enabled' if pipeline._barge_in_enabled else 'disabled'}")
        
        pipeline.enable_barge_in(True)
        print(f"Barge-in now: {'enabled' if pipeline._barge_in_enabled else 'disabled'}")
        
        # Get conversation context
        print("\nConversation context:")
        context = pipeline.get_context()
        print(f"  Session ID: {context.session_id}")
        print(f"  History length: {len(context.conversation_history)}")
        print(f"  Extracted parameters: {context.extracted_parameters}")
        
        # Update context
        pipeline.update_context(
            current_intent="schedule_appointment",
            extracted_parameters={"date": "tomorrow", "time": "2pm"},
        )
        
        context = pipeline.get_context()
        print(f"\nUpdated context:")
        print(f"  Current intent: {context.current_intent}")
        print(f"  Extracted parameters: {context.extracted_parameters}")
        
    finally:
        # Shutdown pipeline
        print("\nShutting down pipeline...")
        await pipeline.shutdown()
        print(f"Pipeline state: {pipeline.state.value}")


if __name__ == "__main__":
    print("=" * 60)
    print("Voice Pipeline Orchestrator Example")
    print("=" * 60)
    print()
    
    # Run the example
    asyncio.run(main())
    
    print()
    print("=" * 60)
    print("Example completed!")
    print("=" * 60)
