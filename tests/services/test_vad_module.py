"""
Unit tests for VAD Module.

Tests speech detection, callback handling, and sensitivity configuration.
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from src.voice_agent.services.vad_module import (
    SpeechState,
    VADConfig,
    VADModule,
    VADResult,
)


@pytest.fixture
def vad_config():
    """Create default VAD configuration for testing."""
    return VADConfig(
        sample_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
        window_size_samples=512,
    )


@pytest.fixture
def mock_silero_model():
    """Create mock Silero VAD model."""
    model = MagicMock()
    model.eval = MagicMock()

    # Default: return speech probability of 0.8
    model.return_value = torch.tensor([0.8])

    return model


@pytest.fixture
def vad_module(vad_config, mock_silero_model):
    """Create VAD module with mocked Silero model."""
    with patch('torch.hub.load') as mock_load:
        mock_load.return_value = (mock_silero_model, {})
        module = VADModule(vad_config)
        return module


class TestVADModuleInitialization:
    """Test VAD module initialization."""

    def test_initialization_with_default_config(self, mock_silero_model):
        """Test VAD module initializes with default configuration."""
        with patch('torch.hub.load') as mock_load:
            mock_load.return_value = (mock_silero_model, {})

            config = VADConfig()
            module = VADModule(config)

            assert module.config.sample_rate == 16000
            assert module.config.threshold == 0.5
            assert module.current_state == SpeechState.SILENCE
            assert not module.is_speech_active

    def test_initialization_with_custom_config(self, mock_silero_model):
        """Test VAD module initializes with custom configuration."""
        with patch('torch.hub.load') as mock_load:
            mock_load.return_value = (mock_silero_model, {})

            config = VADConfig(
                sample_rate=8000,
                threshold=0.7,
                min_speech_duration_ms=500,
            )
            module = VADModule(config)

            assert module.config.sample_rate == 8000
            assert module.config.threshold == 0.7
            assert module.config.min_speech_duration_ms == 500

    def test_model_loading_failure(self):
        """Test VAD module handles model loading failure."""
        with patch('torch.hub.load') as mock_load:
            mock_load.side_effect = RuntimeError("Model download failed")

            config = VADConfig()
            with pytest.raises(RuntimeError, match="Failed to load Silero VAD model"):
                VADModule(config)


class TestSpeechDetection:
    """Test speech detection functionality."""

    def test_process_audio_with_speech(self, vad_module, mock_silero_model):
        """Test processing audio chunk containing speech."""
        # Create synthetic audio (1 second of random audio)
        audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Mock model to return high speech probability
        mock_silero_model.return_value = torch.tensor([0.9])

        result = vad_module.process_audio(audio_bytes)

        assert isinstance(result, VADResult)
        assert result.is_speech is True
        assert pytest.approx(result.confidence, 0.01) == 0.9
        assert result.timestamp > 0.0

    def test_process_audio_with_silence(self, vad_module, mock_silero_model):
        """Test processing audio chunk containing silence."""
        # Create synthetic audio
        audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Mock model to return low speech probability
        mock_silero_model.return_value = torch.tensor([0.1])

        result = vad_module.process_audio(audio_bytes)

        assert isinstance(result, VADResult)
        assert result.is_speech is False
        assert pytest.approx(result.confidence, 0.01) == 0.1

    def test_process_audio_at_threshold(self, vad_module, mock_silero_model):
        """Test processing audio at exact threshold boundary."""
        audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Mock model to return probability exactly at threshold
        mock_silero_model.return_value = torch.tensor([0.5])

        result = vad_module.process_audio(audio_bytes)

        # At threshold, should be detected as speech
        assert result.is_speech is True
        assert result.confidence == 0.5

    def test_process_empty_audio_raises_error(self, vad_module):
        """Test processing empty audio chunk raises ValueError."""
        with pytest.raises(ValueError, match="Audio chunk cannot be empty"):
            vad_module.process_audio(b"")

    def test_timestamp_increments_correctly(self, vad_module, mock_silero_model):
        """Test timestamp increments correctly with each audio chunk."""
        # Create 0.5 second audio chunks
        chunk_size = 8000  # 0.5 seconds at 16kHz
        audio_samples = np.random.randint(-32768, 32767, size=chunk_size, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        mock_silero_model.return_value = torch.tensor([0.8])

        result1 = vad_module.process_audio(audio_bytes)
        result2 = vad_module.process_audio(audio_bytes)
        result3 = vad_module.process_audio(audio_bytes)

        # Each chunk is 0.5 seconds
        assert pytest.approx(result1.timestamp, 0.01) == 0.5
        assert pytest.approx(result2.timestamp, 0.01) == 1.0
        assert pytest.approx(result3.timestamp, 0.01) == 1.5


class TestSensitivityConfiguration:
    """Test sensitivity threshold configuration."""

    def test_set_sensitivity_valid_threshold(self, vad_module):
        """Test setting valid sensitivity threshold."""
        vad_module.set_sensitivity(0.7)
        assert vad_module.config.threshold == 0.7

        vad_module.set_sensitivity(0.3)
        assert vad_module.config.threshold == 0.3

        vad_module.set_sensitivity(0.0)
        assert vad_module.config.threshold == 0.0

        vad_module.set_sensitivity(1.0)
        assert vad_module.config.threshold == 1.0

    def test_set_sensitivity_invalid_threshold_too_low(self, vad_module):
        """Test setting threshold below 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            vad_module.set_sensitivity(-0.1)

    def test_set_sensitivity_invalid_threshold_too_high(self, vad_module):
        """Test setting threshold above 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="Threshold must be between 0.0 and 1.0"):
            vad_module.set_sensitivity(1.5)

    def test_sensitivity_affects_detection(self, vad_module, mock_silero_model):
        """Test that sensitivity threshold affects speech detection."""
        audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Mock model to return probability of 0.6
        mock_silero_model.return_value = torch.tensor([0.6])

        # With threshold 0.5, should detect speech
        vad_module.set_sensitivity(0.5)
        result1 = vad_module.process_audio(audio_bytes)
        assert result1.is_speech is True

        # Reset timestamp for next test
        vad_module.reset()

        # With threshold 0.7, should not detect speech
        vad_module.set_sensitivity(0.7)
        result2 = vad_module.process_audio(audio_bytes)
        assert result2.is_speech is False


class TestCallbackHandlers:
    """Test callback handler functionality."""

    def test_register_speech_start_callback(self, vad_module):
        """Test registering speech start callback."""
        callback = Mock()
        vad_module.on_speech_start(callback)

        assert vad_module._speech_start_callback is callback

    def test_register_speech_end_callback(self, vad_module):
        """Test registering speech end callback."""
        callback = Mock()
        vad_module.on_speech_end(callback)

        assert vad_module._speech_end_callback is callback

    def test_speech_start_callback_triggered(self, vad_module, mock_silero_model):
        """Test speech start callback is triggered when speech begins."""
        callback = Mock()
        vad_module.on_speech_start(callback)

        # Create audio chunks
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Mock high speech probability
        mock_silero_model.return_value = torch.tensor([0.9])

        # Process multiple chunks to exceed min_speech_duration
        # min_speech_duration_ms = 250ms, each chunk is 0.5s = 500ms
        vad_module.process_audio(audio_bytes)

        # Callback should be triggered after sufficient speech duration
        callback.assert_called_once()
        assert callback.call_args[0][0] > 0.0  # timestamp argument

    def test_speech_end_callback_triggered(self, vad_module, mock_silero_model):
        """Test speech end callback is triggered when speech ends."""
        start_callback = Mock()
        end_callback = Mock()
        vad_module.on_speech_start(start_callback)
        vad_module.on_speech_end(end_callback)

        # Create audio chunks
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # First, trigger speech start with high probability
        mock_silero_model.return_value = torch.tensor([0.9])
        vad_module.process_audio(audio_bytes)

        assert start_callback.called

        # Then trigger speech end with low probability
        mock_silero_model.return_value = torch.tensor([0.1])
        vad_module.process_audio(audio_bytes)

        # End callback should be triggered after sufficient silence
        assert end_callback.called
        assert end_callback.call_args[0][0] > 0.0  # timestamp argument

    def test_callbacks_not_triggered_without_registration(self, vad_module, mock_silero_model):
        """Test that unregistered callbacks don't cause errors."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        mock_silero_model.return_value = torch.tensor([0.9])

        # Should not raise error even without callbacks registered
        result = vad_module.process_audio(audio_bytes)
        assert result.is_speech is True


class TestStateManagement:
    """Test VAD state management."""

    def test_initial_state_is_silence(self, vad_module):
        """Test initial state is silence."""
        assert vad_module.current_state == SpeechState.SILENCE
        assert not vad_module.is_speech_active

    def test_state_transitions_to_speech(self, vad_module, mock_silero_model):
        """Test state transitions from silence to speech."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        mock_silero_model.return_value = torch.tensor([0.9])

        # Process enough audio to trigger speech state
        vad_module.process_audio(audio_bytes)

        assert vad_module.current_state == SpeechState.SPEECH
        assert vad_module.is_speech_active

    def test_state_transitions_to_silence(self, vad_module, mock_silero_model):
        """Test state transitions from speech back to silence."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # First trigger speech
        mock_silero_model.return_value = torch.tensor([0.9])
        vad_module.process_audio(audio_bytes)
        assert vad_module.is_speech_active

        # Then trigger silence
        mock_silero_model.return_value = torch.tensor([0.1])
        vad_module.process_audio(audio_bytes)

        assert vad_module.current_state == SpeechState.SILENCE
        assert not vad_module.is_speech_active

    def test_reset_clears_state(self, vad_module, mock_silero_model):
        """Test reset clears all state and counters."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Process some audio to change state
        mock_silero_model.return_value = torch.tensor([0.9])
        vad_module.process_audio(audio_bytes)

        # Reset
        vad_module.reset()

        assert vad_module.current_state == SpeechState.SILENCE
        assert vad_module._current_timestamp == 0.0
        assert vad_module._speech_frames == 0
        assert vad_module._silence_frames == 0
        assert vad_module._speech_start_time is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_process_audio_without_model(self, vad_config):
        """Test processing audio when model is not loaded raises RuntimeError."""
        with patch('torch.hub.load') as mock_load:
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, {})

            module = VADModule(vad_config)
            # Manually set model to None to simulate failure
            module._model = None

            audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
            audio_bytes = audio_samples.tobytes()

            with pytest.raises(RuntimeError, match="VAD model not loaded"):
                module.process_audio(audio_bytes)

    def test_process_audio_with_invalid_data(self, vad_module, mock_silero_model):
        """Test processing audio when model raises exception."""
        audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        # Make the model raise an exception during processing
        mock_silero_model.side_effect = RuntimeError("Model inference failed")

        # This should raise an exception during processing
        with pytest.raises(RuntimeError, match="Model inference failed"):
            vad_module.process_audio(audio_bytes)

    def test_process_very_short_audio(self, vad_module, mock_silero_model):
        """Test processing very short audio chunks."""
        # Create 10ms of audio (160 samples at 16kHz)
        audio_samples = np.random.randint(-32768, 32767, size=160, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        mock_silero_model.return_value = torch.tensor([0.8])

        result = vad_module.process_audio(audio_bytes)

        assert isinstance(result, VADResult)
        assert result.timestamp > 0.0

    def test_process_very_long_audio(self, vad_module, mock_silero_model):
        """Test processing very long audio chunks."""
        # Create 5 seconds of audio
        audio_samples = np.random.randint(-32768, 32767, size=80000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        mock_silero_model.return_value = torch.tensor([0.8])

        result = vad_module.process_audio(audio_bytes)

        assert isinstance(result, VADResult)
        assert pytest.approx(result.timestamp, 0.1) == 5.0

    def test_alternating_speech_and_silence(self, vad_module, mock_silero_model):
        """Test handling alternating speech and silence."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        start_callback = Mock()
        end_callback = Mock()
        vad_module.on_speech_start(start_callback)
        vad_module.on_speech_end(end_callback)

        # Alternate between speech and silence
        for i in range(10):
            if i % 2 == 0:
                mock_silero_model.return_value = torch.tensor([0.9])  # Speech
            else:
                mock_silero_model.return_value = torch.tensor([0.1])  # Silence

            vad_module.process_audio(audio_bytes)

        # Should have multiple start/end events
        assert start_callback.call_count >= 1
        assert end_callback.call_count >= 1

    def test_continuous_speech(self, vad_module, mock_silero_model):
        """Test handling continuous speech without silence."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        start_callback = Mock()
        end_callback = Mock()
        vad_module.on_speech_start(start_callback)
        vad_module.on_speech_end(end_callback)

        # Continuous speech
        mock_silero_model.return_value = torch.tensor([0.9])

        for _ in range(10):
            vad_module.process_audio(audio_bytes)

        # Should trigger start once, but not end
        assert start_callback.call_count == 1
        assert end_callback.call_count == 0
        assert vad_module.is_speech_active

    def test_continuous_silence(self, vad_module, mock_silero_model):
        """Test handling continuous silence without speech."""
        audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
        audio_bytes = audio_samples.tobytes()

        start_callback = Mock()
        end_callback = Mock()
        vad_module.on_speech_start(start_callback)
        vad_module.on_speech_end(end_callback)

        # Continuous silence
        mock_silero_model.return_value = torch.tensor([0.1])

        for _ in range(10):
            vad_module.process_audio(audio_bytes)

        # Should not trigger any callbacks
        assert start_callback.call_count == 0
        assert end_callback.call_count == 0
        assert not vad_module.is_speech_active


class TestDifferentSampleRates:
    """Test VAD with different audio sample rates."""

    def test_8khz_sample_rate(self, mock_silero_model):
        """Test VAD with 8kHz audio."""
        with patch('torch.hub.load') as mock_load:
            mock_load.return_value = (mock_silero_model, {})

            config = VADConfig(sample_rate=8000)
            module = VADModule(config)

            # Create 1 second of 8kHz audio
            audio_samples = np.random.randint(-32768, 32767, size=8000, dtype=np.int16)
            audio_bytes = audio_samples.tobytes()

            mock_silero_model.return_value = torch.tensor([0.8])

            result = module.process_audio(audio_bytes)

            assert isinstance(result, VADResult)
            assert pytest.approx(result.timestamp, 0.01) == 1.0

    def test_16khz_sample_rate(self, mock_silero_model):
        """Test VAD with 16kHz audio."""
        with patch('torch.hub.load') as mock_load:
            mock_load.return_value = (mock_silero_model, {})

            config = VADConfig(sample_rate=16000)
            module = VADModule(config)

            # Create 1 second of 16kHz audio
            audio_samples = np.random.randint(-32768, 32767, size=16000, dtype=np.int16)
            audio_bytes = audio_samples.tobytes()

            mock_silero_model.return_value = torch.tensor([0.8])

            result = module.process_audio(audio_bytes)

            assert isinstance(result, VADResult)
            assert pytest.approx(result.timestamp, 0.01) == 1.0
