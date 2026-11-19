"""Run a configured tenant through the computer microphone and speakers."""

from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import threading
import uuid

from dotenv import load_dotenv

load_dotenv()

from .config.registry import TenantRegistry  # noqa: E402
from .config.settings import Settings  # noqa: E402
from .core.pipecat_orchestrator import CallMetadata, PipecatOrchestrator  # noqa: E402

logger = logging.getLogger("voice_agent.local")


async def run_local_agent(tenant_id: str | None = None) -> None:
    """Capture 16 kHz mono PCM from the default mic and play provider PCM."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "Install local audio support with: pip install -e .[audio]"
        ) from exc

    configured = Settings()
    selected_tenant = tenant_id or configured.voice_agent_tenant_id
    if not selected_tenant:
        raise RuntimeError("Pass --tenant or set VOICE_AGENT_TENANT_ID in .env")

    registry = TenantRegistry(configured.tenant_config_dir, {"local-user": selected_tenant})
    tenant = registry.load_tenant_config(selected_tenant)
    provider = getattr(tenant.tts.provider, "value", tenant.tts.provider)
    if provider == "elevenlabs":
        tts_provider_config = {"output_format": "pcm_16000"}
        output_rate = 16000
    elif provider == "openai":
        tts_provider_config = {"response_format": "pcm"}
        output_rate = 24000
    else:
        raise RuntimeError(f"Unsupported local TTS provider: {provider}")

    orchestrator = PipecatOrchestrator(registry)
    session_id = f"local-{uuid.uuid4().hex[:12]}"
    pipeline = await orchestrator.start_session(
        session_id=session_id,
        caller_id="local-user",
        call_metadata=CallMetadata(caller_id="local-user", call_sid=session_id),
        tts_provider_config=tts_provider_config,
    )

    audio_input: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
    playback: queue.Queue[bytes] = queue.Queue(maxsize=64)
    loop = asyncio.get_running_loop()
    stop_playback = threading.Event()

    def on_transcript(text: str, final: bool) -> None:
        if final and text.strip():
            print(f"You: {text}", flush=True)

    def on_response(response) -> None:
        text = getattr(response, "text", "")
        if text:
            print(f"Agent: {text}", flush=True)

    def on_audio(data: bytes) -> None:
        if len(data) % 2:
            logger.warning("Ignoring a malformed PCM chunk with an odd byte count")
            return
        try:
            playback.put_nowait(data)
        except queue.Full:
            logger.warning("Speaker queue full; dropping a TTS audio chunk")

    def interrupt_playback() -> None:
        while True:
            try:
                playback.get_nowait()
            except queue.Empty:
                break

    pipeline.on_transcript(on_transcript)
    pipeline.on_llm_response(on_response)
    pipeline.on_audio_generated(on_audio)
    pipeline.on_interruption(interrupt_playback)

    def play_audio() -> None:
        with sd.RawOutputStream(samplerate=output_rate, channels=1, dtype="int16") as output:
            while not stop_playback.is_set():
                try:
                    chunk = playback.get(timeout=0.1)
                except queue.Empty:
                    continue
                output.write(chunk)

    def mic_callback(indata, _frames, _time_info, status) -> None:
        if status:
            logger.warning("Microphone status: %s", status)
        data = bytes(indata)

        def enqueue() -> None:
            if audio_input.full():
                try:
                    audio_input.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            audio_input.put_nowait(data)

        loop.call_soon_threadsafe(enqueue)

    speaker_thread = threading.Thread(target=play_audio, name="voice-agent-speaker", daemon=True)
    speaker_thread.start()
    print("Voice agent listening. Press Ctrl+C to stop.", flush=True)
    try:
        with sd.RawInputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=4096,
            callback=mic_callback,
        ):
            while True:
                chunk = await audio_input.get()
                await pipeline.process_audio_chunk(chunk)
    finally:
        stop_playback.set()
        speaker_thread.join(timeout=2)
        await orchestrator.stop_session(session_id, persist=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the voice agent on the local microphone")
    parser.add_argument("--tenant", help="Tenant config name under TENANT_CONFIG_DIR")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(run_local_agent(args.tenant))
    except KeyboardInterrupt:
        print("Voice agent stopped.", flush=True)


if __name__ == "__main__":
    main()
