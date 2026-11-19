"""Regression tests for provider/Twilio PCM format conversion."""

import numpy as np

from voice_agent.services.audio_utils import (
    mulaw_to_pcm16,
    pcm16_to_mulaw,
    pcm16_to_twilio_mulaw,
    resample_pcm16,
    twilio_mulaw_to_pcm16k,
)


def test_mulaw_silence_round_trip():
    silence = np.zeros(160, dtype="<i2").tobytes()
    encoded = pcm16_to_mulaw(silence)

    assert encoded == b"\xff" * 160
    assert mulaw_to_pcm16(encoded) == silence


def test_twilio_media_resampling_uses_expected_pipeline_rates():
    twilio_silence = b"\xff" * 160
    pipeline_audio = twilio_mulaw_to_pcm16k(twilio_silence)
    assert len(pipeline_audio) == 640
    assert pipeline_audio == b"\x00" * 640

    provider_audio = b"\x00" * 640
    twilio_audio = pcm16_to_twilio_mulaw(provider_audio, source_rate=16000)
    assert len(twilio_audio) == 160
    assert twilio_audio == b"\xff" * 160


def test_resample_pcm16_is_identity_for_matching_rates():
    audio = np.arange(80, dtype="<i2").tobytes()
    assert resample_pcm16(audio, 16000, 16000) == audio
