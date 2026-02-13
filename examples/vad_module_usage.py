"""
Example usage of VAD Module with Silero VAD.

This example demonstrates how to use the VAD module for speech detection
in a voice conversation system.
"""

import numpy as np
import logging
from src.voice_agent.services.vad_module import VADModule, VADConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def on_speech_start(timestamp: float) -> None:
    """Callback when speech starts."""
    logger.info(f"🎤 Speech started at {timestamp:.3f} seconds")


def on_speech_end(timestamp: float) -> None:
    """Callback when speech ends."""
    logger.info(f"🔇 Speech ended at {timestamp:.3f} seconds")


def main():
    """Demonstrate VAD module usage."""
    logger.info("=== VAD Module Example ===\n")
    
    # 1. Create VAD configuration
    logger.info("1. Creating VAD configuration...")
    config = VADConfig(
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )
    logger.info(f"   Sample rate: {config.sample_rate} Hz")
    logger.info(f"   Threshold: {config.threshold}")
    logger.info(f"   Min speech duration: {config.min_speech_duration_ms} ms\n")
    
    # 2. Initialize VAD module
    logger.info("2. Initializing VAD module...")
    vad = VADModule(config)
    logger.info(f"   Initial state: {vad.current_state.value}\n")
    
    # 3. Register callbacks
    logger.info("3. Registering callbacks...")
    vad.on_speech_start(on_speech_start)
    vad.on_speech_end(on_speech_end)
    logger.info("   Callbacks registered\n")
    
    # 4. Simulate audio processing
    logger.info("4. Processing audio chunks...\n")
    
    # Simulate 5 seconds of audio with speech in the middle
    chunk_duration = 0.5  # 500ms chunks
    sample_rate = 16000
    chunk_size = int(chunk_duration * sample_rate)
    
    for i in range(10):
        # Generate synthetic audio (random noise)
        audio_samples = np.random.randint(-32768, 32767, size=chunk_size, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()
        
        # Process audio
        result = vad.process_audio(audio_bytes)
        
        logger.info(
            f"   Chunk {i+1}: is_speech={result.is_speech}, "
            f"confidence={result.confidence:.3f}, "
            f"timestamp={result.timestamp:.3f}s"
        )
    
    logger.info(f"\n   Final state: {vad.current_state.value}")
    
    # 5. Adjust sensitivity
    logger.info("\n5. Adjusting sensitivity...")
    logger.info("   Setting threshold to 0.7 (less sensitive)")
    vad.set_sensitivity(0.7)
    logger.info(f"   New threshold: {vad.config.threshold}\n")
    
    # 6. Reset VAD state
    logger.info("6. Resetting VAD state...")
    vad.reset()
    logger.info(f"   State after reset: {vad.current_state.value}")
    logger.info(f"   Timestamp after reset: {vad._current_timestamp}\n")
    
    logger.info("=== Example Complete ===")


if __name__ == "__main__":
    main()
