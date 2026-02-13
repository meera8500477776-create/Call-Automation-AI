"""Example usage of Twilio integration layer.

This example demonstrates how to use the TwilioIntegration class to:
1. Connect to Twilio via WebSocket
2. Handle incoming audio from calls
3. Send audio back to Twilio
4. Extract call metadata (caller ID, call SID)
5. Handle call lifecycle events
"""

import asyncio
import logging
import os

from voice_agent.telephony.twilio_integration import (
    CallMetadata,
    ConnectionType,
    TwilioConfig,
    TwilioIntegration,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Main example function."""
    
    # Create Twilio configuration
    config = TwilioConfig(
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", "test_account_sid"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", "test_auth_token"),
        connection_type=ConnectionType.WEBSOCKET,
        audio_format="mulaw",
        sample_rate=8000,
        channels=1,
        chunk_size=160,
        enable_dtmf=True,
    )
    
    # Create Twilio integration
    twilio = TwilioIntegration(config)
    
    # Register callbacks
    def on_audio_received(audio_data: bytes):
        """Handle incoming audio from Twilio."""
        logger.info(f"Received audio chunk: {len(audio_data)} bytes")
        # Process audio here (e.g., send to voice pipeline)
    
    def on_call_started(metadata: CallMetadata):
        """Handle call start event."""
        logger.info(f"Call started: {metadata.call_sid}")
        logger.info(f"  Caller ID: {metadata.caller_id}")
        logger.info(f"  Called number: {metadata.called_number}")
        logger.info(f"  Direction: {metadata.direction}")
        logger.info(f"  From: {metadata.from_city}, {metadata.from_state}, {metadata.from_country}")
    
    def on_call_ended(metadata: CallMetadata):
        """Handle call end event."""
        logger.info(f"Call ended: {metadata.call_sid}")
        logger.info(f"  Duration: {metadata.duration_seconds:.2f} seconds")
        logger.info(f"  Status: {metadata.status.value}")
    
    def on_dtmf(digit: str):
        """Handle DTMF tone detection."""
        logger.info(f"DTMF digit pressed: {digit}")
    
    def on_error(error: Exception):
        """Handle errors."""
        logger.error(f"Error occurred: {error}")
    
    # Register all callbacks
    twilio.on_audio_received(on_audio_received)
    twilio.on_call_started(on_call_started)
    twilio.on_call_ended(on_call_ended)
    twilio.on_dtmf(on_dtmf)
    twilio.on_error(on_error)
    
    try:
        # Connect to Twilio
        # In production, this URL would come from Twilio's webhook
        websocket_url = os.getenv(
            "TWILIO_WEBSOCKET_URL",
            "wss://streams.twilio.com/v1/stream"
        )
        
        logger.info(f"Connecting to Twilio: {websocket_url}")
        await twilio.connect(websocket_url)
        
        logger.info("Connected to Twilio successfully")
        logger.info(f"Connection status: {twilio.is_connected}")
        
        # Simulate sending audio back to Twilio
        # In production, this would come from your TTS service
        sample_audio = b"\x00" * 160  # 20ms of silence at 8kHz mulaw
        
        logger.info("Sending audio to Twilio...")
        for i in range(5):
            await twilio.send_audio(sample_audio)
            await asyncio.sleep(0.02)  # 20ms between chunks
        
        logger.info("Audio sent successfully")
        
        # Extract call metadata
        if twilio.call_metadata:
            logger.info("\nCall Metadata:")
            logger.info(f"  Call SID: {twilio.extract_call_sid()}")
            logger.info(f"  Caller ID: {twilio.extract_caller_id()}")
            logger.info(f"  Stream SID: {twilio.stream_sid}")
        
        # Keep connection alive for a while
        logger.info("\nKeeping connection alive for 10 seconds...")
        await asyncio.sleep(10)
        
    except ConnectionError as e:
        logger.error(f"Failed to connect: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        # Disconnect
        logger.info("\nDisconnecting from Twilio...")
        await twilio.disconnect()
        logger.info("Disconnected successfully")


async def example_with_voice_pipeline():
    """Example integrating Twilio with voice pipeline.
    
    This demonstrates how to connect Twilio audio streams to the
    voice pipeline for real-time processing.
    """
    from voice_agent.core.voice_pipeline import VoicePipeline, PipelineConfig
    from voice_agent.services.stt_service import STTConfig
    from voice_agent.services.llm_service import LLMConfig
    from voice_agent.services.tts_factory import TTSConfig
    
    # Create Twilio integration
    twilio_config = TwilioConfig(
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", "test_account_sid"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", "test_auth_token"),
    )
    twilio = TwilioIntegration(twilio_config)
    
    # Create voice pipeline
    pipeline_config = PipelineConfig(
        stt_config=STTConfig(
            api_key=os.getenv("DEEPGRAM_API_KEY", "test_key"),
            model="nova-2",
            language="en-US",
        ),
        llm_config=LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "test_key"),
        ),
        tts_config=TTSConfig(
            provider="elevenlabs",
            api_key=os.getenv("ELEVENLABS_API_KEY", "test_key"),
            voice_id="default",
        ),
    )
    pipeline = VoicePipeline(pipeline_config)
    
    # Connect Twilio audio to pipeline
    def on_audio_received(audio_data: bytes):
        """Forward Twilio audio to voice pipeline."""
        asyncio.create_task(pipeline.process_audio_chunk(audio_data))
    
    # Connect pipeline audio to Twilio
    def on_audio_generated(audio_data: bytes):
        """Forward pipeline audio to Twilio."""
        asyncio.create_task(twilio.send_audio(audio_data))
    
    # Register callbacks
    twilio.on_audio_received(on_audio_received)
    pipeline.on_audio_generated(on_audio_generated)
    
    # Handle call lifecycle
    def on_call_started(metadata: CallMetadata):
        """Initialize pipeline when call starts."""
        logger.info(f"Call started from {metadata.caller_id}")
        asyncio.create_task(pipeline.initialize())
    
    def on_call_ended(metadata: CallMetadata):
        """Cleanup pipeline when call ends."""
        logger.info(f"Call ended, duration: {metadata.duration_seconds:.2f}s")
        asyncio.create_task(pipeline.shutdown())
    
    twilio.on_call_started(on_call_started)
    twilio.on_call_ended(on_call_ended)
    
    try:
        # Connect and process
        websocket_url = os.getenv("TWILIO_WEBSOCKET_URL")
        if not websocket_url:
            logger.error("TWILIO_WEBSOCKET_URL not set")
            return
        
        await twilio.connect(websocket_url)
        logger.info("Connected and processing calls...")
        
        # Keep running until interrupted
        await asyncio.Event().wait()
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await twilio.disconnect()
        await pipeline.shutdown()


if __name__ == "__main__":
    # Run basic example
    print("=" * 60)
    print("Twilio Integration Example")
    print("=" * 60)
    print()
    
    # Uncomment to run the example
    # asyncio.run(main())
    
    # Or run the voice pipeline integration example
    # asyncio.run(example_with_voice_pipeline())
    
    print("Example code ready to run.")
    print("Set environment variables and uncomment asyncio.run() to execute.")
