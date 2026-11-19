import base64
import io
import wave
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from voice_agent.services.elevenlabs_tts_service import VoiceConfig
from voice_agent.services.gemini_tts_service import GeminiTTSConfig, GeminiTTSService


def build_wav(rate=24000, channels=1, sample_width=2):
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x01\x00" * 80)
    return stream.getvalue()


class EventStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for event in self.events:
            yield event

    async def aclose(self):
        self.closed = True


@pytest.fixture
def service_fixture():
    interactions = SimpleNamespace(create=AsyncMock())
    client = SimpleNamespace(aio=SimpleNamespace(interactions=interactions, aclose=AsyncMock()))
    config = GeminiTTSConfig(
        api_key="test-gemini-key",
        default_voice_config=VoiceConfig(provider="gemini", voice_id="Kore"),
    )
    with patch("voice_agent.services.gemini_tts_service.genai.Client", return_value=client):
        service = GeminiTTSService(config)
    return service, interactions, client


@pytest.mark.asyncio
async def test_synthesizes_and_chunks_real_pcm_payload_shape(service_fixture):
    service, interactions, _ = service_fixture
    pcm = b"\x01\x00" * 40
    stream = EventStream([
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="audio", data=base64.b64encode(pcm).decode()),
        )
    ])
    interactions.create.return_value = stream

    chunks = [chunk async for chunk in service.synthesize_streaming("Hello there.")]

    assert b"".join(chunks) == pcm
    assert all(len(chunk) <= service.config.chunk_size for chunk in chunks)
    request = interactions.create.await_args.kwargs
    assert request["model"] == "gemini-3.8-flash-tts"
    assert request["generation_config"]["speech_config"] == [{"voice": "Kore"}]
    assert request["response_format"] == {
        "type": "audio",
        "mime_type": "audio/l16",
        "sample_rate": 24000,
    }
    assert request["stream"] is True
    assert request["input"][0]["content"][0]["text"] == "Hello there."
    assert stream.closed


@pytest.mark.asyncio
async def test_strips_supported_wav_container(service_fixture):
    service, _, _ = service_fixture
    audio = build_wav()
    result = service._normalize_audio(audio)

    assert result == b"\x01\x00" * 80


@pytest.mark.asyncio
async def test_rejects_empty_audio_response(service_fixture):
    service, interactions, _ = service_fixture
    interactions.create.return_value = EventStream([])

    with pytest.raises(RuntimeError, match="speech generation failed"):
        _ = [chunk async for chunk in service.synthesize_streaming("Hello")]


@pytest.mark.asyncio
async def test_closes_async_sdk_client(service_fixture):
    service, _, client = service_fixture

    await service.aclose()

    client.aio.aclose.assert_awaited_once()


def test_gemini_service_requires_key():
    config = GeminiTTSConfig(
        api_key="",
        default_voice_config=VoiceConfig(provider="gemini", voice_id="Kore"),
    )
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiTTSService(config)


def test_normalizer_rejects_non_mono_wav(service_fixture):
    service, _, _ = service_fixture

    with pytest.raises(ValueError, match="mono"):
        service._normalize_audio(build_wav(channels=2))
