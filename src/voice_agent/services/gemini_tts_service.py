"""Gemini Flash TTS adapter using the official Google Gen AI SDK."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass

from google import genai

from .elevenlabs_tts_service import VoiceConfig

logger = logging.getLogger(__name__)


@dataclass
class GeminiTTSConfig:
    """Runtime settings for Gemini speech generation."""

    api_key: str
    default_voice_config: VoiceConfig
    model: str = "gemini-3.8-flash-tts"
    style_prompt: str = "Warm, friendly, natural customer-service conversation."
    timeout: float = 45.0
    chunk_size: int = 4096
    sample_rate: int = 24000


class GeminiTTSService:
    """Synthesize exact response text as mono 24 kHz PCM for the pipeline."""

    def __init__(self, config: GeminiTTSConfig):
        if not config.api_key.strip():
            raise ValueError("GEMINI_API_KEY is required when TTS_PROVIDER=gemini")
        if config.default_voice_config.provider != "gemini":
            raise ValueError("Gemini TTS requires a Gemini voice configuration")
        if not config.default_voice_config.voice_id.strip():
            raise ValueError("A Gemini prebuilt voice name must be configured")
        self.config = config
        self.client = genai.Client(
            api_key=config.api_key,
            http_options={"api_version": "v1beta", "timeout": int(config.timeout * 1000)},
        )
        self._closed = False

    async def aclose(self) -> None:
        """Close SDK-owned HTTP transports once per service instance."""
        if not self._closed:
            await self.client.aio.aclose()
            self._closed = True

    async def apply_emotional_context(self, text: str, tags: list[str]) -> str:
        """Keep the transcript verbatim; style is supplied separately to Gemini."""
        return text

    async def synthesize_streaming(
        self,
        text: str,
        voice_config: VoiceConfig | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream raw PCM chunks for low-latency speaker playback."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        voice = voice_config or self.config.default_voice_config
        if voice.provider != "gemini":
            raise ValueError("Gemini TTS requires provider='gemini'")

        try:
            stream = await asyncio.wait_for(
                self.client.aio.interactions.create(
                    model=self.config.model,
                    input=[
                        {
                            "type": "user_input",
                            "content": [
                                {
                                    "type": "text",
                                    "text": text.strip(),
                                    "annotations": [
                                        {"type": "speech_metadata", "style": self.config.style_prompt}
                                    ],
                                }
                            ],
                        }
                    ],
                    response_format={
                        "type": "audio",
                        "mime_type": "audio/l16",
                        "sample_rate": self.config.sample_rate,
                    },
                    generation_config={"speech_config": [{"voice": voice.voice_id}]},
                    stream=True,
                    timeout=self.config.timeout,
                ),
                timeout=self.config.timeout + 1,
            )
            iterator = stream.__aiter__()
            deadline = asyncio.get_running_loop().time() + self.config.timeout
            received_audio = False
            try:
                while True:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError("Gemini TTS streaming timed out")
                    try:
                        event = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
                    except StopAsyncIteration:
                        break
                    delta = getattr(event, "delta", None)
                    if getattr(event, "event_type", None) != "step.delta":
                        continue
                    if getattr(delta, "type", None) != "audio":
                        continue
                    encoded = getattr(delta, "data", None)
                    if not encoded:
                        continue
                    audio = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
                    audio = self._normalize_audio(audio)
                    if audio:
                        received_audio = True
                        for offset in range(0, len(audio), self.config.chunk_size):
                            yield audio[offset : offset + self.config.chunk_size]
            finally:
                close_stream = getattr(stream, "aclose", None)
                if callable(close_stream):
                    await close_stream()
            if not received_audio:
                raise RuntimeError("Gemini returned no audio output")
        except Exception as exc:
            logger.error("Gemini TTS failed (%s)", type(exc).__name__)
            raise RuntimeError(f"Gemini speech generation failed ({type(exc).__name__})") from exc

    def _normalize_audio(self, audio: bytes) -> bytes:
        """Strip a WAV wrapper if returned and enforce mono 24 kHz 16-bit PCM."""
        if not audio.startswith(b"RIFF"):
            if len(audio) % 2:
                raise ValueError("Gemini PCM output has an incomplete 16-bit sample")
            return audio
        with wave.open(io.BytesIO(audio), "rb") as wav_file:
            if wav_file.getnchannels() != 1 or wav_file.getsampwidth() != 2:
                raise ValueError("Gemini WAV output must be mono 16-bit PCM")
            if wav_file.getframerate() != self.config.sample_rate:
                raise ValueError(
                    f"Gemini WAV sample rate {wav_file.getframerate()} does not match "
                    f"configured {self.config.sample_rate} Hz"
                )
            return wav_file.readframes(wav_file.getnframes())
