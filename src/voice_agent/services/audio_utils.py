"""Small PCM and G.711 helpers for microphone and Twilio media streams."""

from __future__ import annotations

import numpy as np


def resample_pcm16(audio: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample little-endian mono PCM16 between sample rates."""
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if len(audio) % 2:
        raise ValueError("PCM16 audio must contain complete samples")
    samples = np.frombuffer(audio, dtype="<i2")
    if source_rate == target_rate or samples.size == 0:
        return audio
    target_count = max(1, round(samples.size * target_rate / source_rate))
    positions = np.linspace(0, samples.size - 1, target_count)
    output = np.interp(positions, np.arange(samples.size), samples.astype(np.float64))
    return np.clip(np.rint(output), -32768, 32767).astype("<i2").tobytes()


def mulaw_to_pcm16(audio: bytes) -> bytes:
    """Decode G.711 μ-law bytes to little-endian PCM16 samples."""
    decoded = []
    for byte in audio:
        value = (~byte) & 0xFF
        magnitude = (((value & 0x0F) << 3) + 0x84) << ((value & 0x70) >> 4)
        sample = magnitude - 0x84
        decoded.append(-sample if value & 0x80 else sample)
    return np.asarray(decoded, dtype="<i2").tobytes()


def pcm16_to_mulaw(audio: bytes) -> bytes:
    """Encode little-endian PCM16 samples as G.711 μ-law."""
    if len(audio) % 2:
        raise ValueError("PCM16 audio must contain complete samples")
    encoded = bytearray()
    for sample in np.frombuffer(audio, dtype="<i2"):
        value = int(sample)
        sign = 0x80 if value < 0 else 0
        if value < 0:
            value = -value
        value = min(value, 32635) + 0x84
        exponent = 7
        mask = 0x4000
        while exponent > 0 and not (value & mask):
            exponent -= 1
            mask >>= 1
        mantissa = (value >> (exponent + 3)) & 0x0F
        encoded.append((~(sign | (exponent << 4) | mantissa)) & 0xFF)
    return bytes(encoded)


def twilio_mulaw_to_pcm16k(audio: bytes) -> bytes:
    """Convert Twilio's 8 kHz μ-law media payload to pipeline PCM16/16 kHz."""
    return resample_pcm16(mulaw_to_pcm16(audio), 8000, 16000)


def pcm16_to_twilio_mulaw(audio: bytes, source_rate: int) -> bytes:
    """Convert provider PCM16 audio to Twilio's 8 kHz μ-law media format."""
    return pcm16_to_mulaw(resample_pcm16(audio, source_rate, 8000))
