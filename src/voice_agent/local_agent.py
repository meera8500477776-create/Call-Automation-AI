"""Run a configured tenant through the computer microphone and speakers."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import queue
import threading
import time
import uuid

from dotenv import load_dotenv

from .config.settings import ENV_FILE  # noqa: E402

load_dotenv(ENV_FILE, override=False)

from .config.registry import TenantRegistry  # noqa: E402
from .config.settings import Settings  # noqa: E402
from .core.pipecat_orchestrator import CallMetadata, PipecatOrchestrator  # noqa: E402

logger = logging.getLogger("voice_agent.local")


def _pcm16_level(pcm: bytes) -> tuple[float, int]:
    """Return RMS dBFS and absolute peak for signed 16-bit PCM bytes."""
    import numpy as np

    if not pcm or len(pcm) % 2:
        raise ValueError("PCM input must contain complete 16-bit samples")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    peak = int(np.max(np.abs(samples)))
    dbfs = 20 * math.log10(max(rms, 1e-9) / 32768)
    return dbfs, peak


def _pcm16_near_clip_percent(pcm: bytes) -> float:
    """Report the percent of samples within about 0.2% of int16 full scale."""
    import numpy as np

    if not pcm or len(pcm) % 2:
        raise ValueError("PCM input must contain complete 16-bit samples")
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.int32)
    return float(np.count_nonzero(np.abs(samples) >= 32700) * 100 / samples.size)


def list_input_devices() -> None:
    """Print input-capable device indexes without reading or printing secrets."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("Install local audio support with: pip install -e .[audio]") from exc

    default_input = sd.default.device[0]
    for index, device in enumerate(sd.query_devices()):
        if device["max_input_channels"]:
            marker = " (default)" if index == default_input else ""
            print(
                f"{index}{marker}: {device['name']} | "
                f"inputs={device['max_input_channels']} | "
                f"native_rate={device['default_samplerate']} Hz"
            )


def diagnose_microphone(
    duration_seconds: int = 8,
    device: int | None = None,
    vad_threshold: float = 0.5,
    transcribe: bool = False,
) -> None:
    """Show mic level and Silero frames; optionally stream captured audio to Deepgram."""
    import sounddevice as sd

    from .services.vad_module import VADConfig, VADModule

    if not 1 <= duration_seconds <= 60:
        raise ValueError("Diagnostic duration must be between 1 and 60 seconds")
    if not 0.0 < vad_threshold < 1.0:
        raise ValueError("VAD threshold must be greater than 0 and less than 1")

    selected_device = sd.default.device[0] if device is None else device
    try:
        info = sd.query_devices(selected_device, "input")
        sd.check_input_settings(
            device=selected_device,
            samplerate=16000,
            channels=1,
            dtype="int16",
        )
    except Exception as exc:
        print(f"microphone_open=FAIL {type(exc).__name__}: {str(exc)[:180]}")
        return
    print(
        f"microphone={info['name']} device_id={selected_device} "
        f"native_rate={info['default_samplerate']} Hz capture=16000 Hz mono PCM16"
    )
    print(f"vad_threshold={vad_threshold:.2f}; audio is analyzed in memory and not saved")

    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=128)
    callback_status: list[str] = []
    dropped_chunks = 0

    def capture_callback(indata, _frames, _time_info, status) -> None:
        nonlocal dropped_chunks
        if status:
            callback_status.append(str(status))
        try:
            audio_queue.put_nowait(bytes(indata))
        except queue.Full:
            dropped_chunks += 1

    vad = VADModule(VADConfig(sample_rate=16000, threshold=vad_threshold))
    total_frames = 0
    speech_frames = 0
    all_pcm = bytearray()
    vad_buffer = bytearray()
    level_pcm = bytearray()
    level_bytes_per_second = 16000 * 2
    next_level_report = time.monotonic() + 1
    deadline = time.monotonic() + duration_seconds

    try:
        with sd.RawInputStream(
            device=selected_device,
            samplerate=16000,
            channels=1,
            dtype="int16",
            blocksize=512,
            callback=capture_callback,
        ):
            print(f"capturing for {duration_seconds} seconds; speak at a normal volume now", flush=True)
            while time.monotonic() < deadline:
                try:
                    pcm = audio_queue.get(timeout=min(0.25, max(0.01, deadline - time.monotonic())))
                except queue.Empty:
                    continue
                all_pcm.extend(pcm)
                vad_buffer.extend(pcm)
                level_pcm.extend(pcm)
                while len(vad_buffer) >= 1024:
                    frame = bytes(vad_buffer[:1024])
                    del vad_buffer[:1024]
                    result = vad.process_audio(frame)
                    total_frames += 1
                    speech_frames += int(result.is_speech)
                now = time.monotonic()
                if len(level_pcm) >= level_bytes_per_second or now >= next_level_report:
                    complete_level_bytes = len(level_pcm) - (len(level_pcm) % 2)
                    if complete_level_bytes:
                        dbfs, peak = _pcm16_level(bytes(level_pcm[:complete_level_bytes]))
                        near_clip = _pcm16_near_clip_percent(bytes(level_pcm[:complete_level_bytes]))
                        print(f"mic_level_rms={dbfs:.1f} dBFS peak={peak}/32768 near_clip={near_clip:.2f}%")
                        del level_pcm[:complete_level_bytes]
                    next_level_report = now + 1
    except Exception as exc:
        print(f"microphone_open=FAIL {type(exc).__name__}: {str(exc)[:180]}")
        return

    if level_pcm:
        complete_level_bytes = len(level_pcm) - (len(level_pcm) % 2)
        if complete_level_bytes:
            dbfs, peak = _pcm16_level(bytes(level_pcm[:complete_level_bytes]))
            near_clip = _pcm16_near_clip_percent(bytes(level_pcm[:complete_level_bytes]))
            print(f"mic_level_rms={dbfs:.1f} dBFS peak={peak}/32768 near_clip={near_clip:.2f}%")
    duration = len(all_pcm) / (16000 * 2)
    print(
        f"microphone_open=PASS captured={duration:.2f}s "
        f"speech_frames={speech_frames}/{total_frames} "
        f"speech_audio={speech_frames * 512 / 16000:.2f}s "
        f"dropped_chunks={dropped_chunks} callback_status_events={len(callback_status)}"
    )
    if transcribe and all_pcm:
        print("deepgram_audio_transmission=enabled (captured mic audio is sent to Deepgram)")
        asyncio.run(_transcribe_diagnostic_audio(bytes(all_pcm)))


async def _transcribe_diagnostic_audio(pcm: bytes) -> None:
    """Stream an in-memory mic capture at real-time pacing and report final STT."""
    from .services.stt_service import STTConfig, STTService

    api_key = Settings().deepgram_api_key
    if not api_key:
        print("deepgram_transcription=BLOCKED missing Deepgram API key")
        return

    service = STTService(
        STTConfig(
            api_key=api_key,
            model="nova-2",
            language="en-US",
            interim_results=True,
            encoding="linear16",
            sample_rate=16000,
            channels=1,
            endpointing_ms=300,
            utterance_end_ms=1000,
        )
    )
    finals: list[str] = []
    interims: list[str] = []
    errors: list[str] = []
    final_event = asyncio.Event()

    def on_final(text: str, _metadata: dict[str, object]) -> None:
        if text.strip():
            finals.append(text.strip())
            final_event.set()

    service.on_final_result(on_final)
    service.on_interim_result(lambda text, _metadata: interims.append(text.strip()))
    service.on_error(lambda error: errors.append(type(error).__name__))
    try:
        await service.connect("microphone-diagnostic")
        frame_bytes = 3200  # 100 ms mono PCM16 at 16 kHz
        for offset in range(0, len(pcm), frame_bytes):
            frame = pcm[offset : offset + frame_bytes]
            if len(frame) < frame_bytes:
                frame += b"\x00" * (frame_bytes - len(frame))
            await service.send_audio(frame)
            await asyncio.sleep(0.1)
        for _ in range(12):
            await service.send_audio(b"\x00" * frame_bytes)
            await asyncio.sleep(0.1)
        await service.finalize()
        if not finals:
            try:
                await asyncio.wait_for(final_event.wait(), timeout=2.5)
            except TimeoutError:
                pass
    except Exception as exc:
        print(f"deepgram_transcription=FAIL {type(exc).__name__}: {str(exc)[:180]}")
        return
    finally:
        await service.disconnect()

    if finals:
        for transcript in finals:
            print(f"deepgram_final_transcript={transcript}")
        print("deepgram_transcription=PASS")
    else:
        last_interim = interims[-1] if interims else ""
        print(
            f"deepgram_transcription=NO_FINAL_TRANSCRIPT "
            f"interim_count={len(interims)} last_interim={last_interim!r} errors={errors}"
        )


async def run_local_agent(tenant_id: str | None = None, input_device: int | None = None) -> None:
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
    elif provider == "gemini":
        tts_provider_config = {
            "model": tenant.tts.model or "gemini-3.8-flash-tts",
            "style_prompt": tenant.tts.style_prompt
            or "Warm, friendly, natural customer-service conversation.",
        }
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
            device=input_device,
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
    parser.add_argument("--device", type=int, help="Input device index from --list-input-devices")
    parser.add_argument("--list-input-devices", action="store_true", help="List input device indexes and sample rates")
    parser.add_argument("--diagnose-microphone", action="store_true", help="Capture microphone levels and Silero speech frames")
    parser.add_argument("--duration", type=int, default=8, help="Diagnostic capture duration in seconds (1-60)")
    parser.add_argument("--vad-threshold", type=float, default=0.5, help="Diagnostic-only Silero threshold; default is 0.5")
    parser.add_argument("--transcribe", action="store_true", help="Stream diagnostic audio to Deepgram and print final transcript")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.list_input_devices:
        list_input_devices()
        return
    if args.diagnose_microphone:
        diagnose_microphone(args.duration, args.device, args.vad_threshold, args.transcribe)
        return
    try:
        asyncio.run(run_local_agent(args.tenant, args.device))
    except KeyboardInterrupt:
        print("Voice agent stopped.", flush=True)


if __name__ == "__main__":
    main()
