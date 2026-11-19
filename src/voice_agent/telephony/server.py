"""FastAPI endpoints for Twilio Programmable Voice Media Streams."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from urllib.parse import parse_qs
from xml.etree.ElementTree import Element, SubElement, tostring

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from twilio.request_validator import RequestValidator

from ..config.settings import ENV_FILE  # noqa: E402

load_dotenv(ENV_FILE, override=False)

from ..config.registry import TenantRegistry  # noqa: E402
from ..config.settings import Settings  # noqa: E402
from ..core.pipecat_orchestrator import PipecatOrchestrator  # noqa: E402
from .twilio_integration import ConnectionType, TwilioConfig, TwilioIntegration  # noqa: E402
from .twilio_voice_bridge import BridgeConfig, TwilioVoiceBridge  # noqa: E402

logger = logging.getLogger(__name__)


class _TwilioWebSocketAdapter:
    """Adapt FastAPI's server WebSocket to the integration's socket protocol."""

    def __init__(self, websocket: WebSocket, call_contexts: dict[str, dict[str, str]]):
        self._websocket = websocket
        self._call_contexts = call_contexts

    async def recv(self) -> str | None:
        try:
            message = json.loads(await self._websocket.receive_text())
        except WebSocketDisconnect:
            return None

        if message.get("event") == "start":
            start = message.setdefault("start", {})
            parameters = start.setdefault("customParameters", {})
            call_sid = start.get("callSid") or parameters.get("CallSid", "")
            call_context = self._call_contexts.get(call_sid)
            if not call_context:
                await self._websocket.close(code=1008, reason="Unknown call context")
                return None
            for name in ("From", "To", "Direction", "TenantId"):
                if call_context.get(name):
                    parameters.setdefault(name, call_context[name])
        return json.dumps(message)

    async def send(self, message: str) -> None:
        await self._websocket.send_text(message)

    async def close(self) -> None:
        try:
            await self._websocket.close()
        except RuntimeError:
            # The peer may already have closed during normal stream teardown.
            pass


def create_app(
    settings: Settings | None = None,
    tenant_registry: TenantRegistry | None = None,
    orchestrator: PipecatOrchestrator | None = None,
) -> FastAPI:
    """Create the Twilio webhook/WebSocket application.

    Dependencies can be injected for tests. Production startup loads the
    existing tenant registry and Pipecat orchestrator from the configured
    tenant directory.
    """
    configured = settings or Settings()
    supplied_registry = tenant_registry
    supplied_orchestrator = orchestrator

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not configured.voice_agent_tenant_id and supplied_registry is None:
            raise RuntimeError("VOICE_AGENT_TENANT_ID must name a tenant config")
        registry = supplied_registry or TenantRegistry(configured.tenant_config_dir)
        tenant_config = None
        if configured.voice_agent_tenant_id:
            tenant_config = registry.load_tenant_config(configured.voice_agent_tenant_id)
        app.state.tenant_registry = registry
        app.state.orchestrator = supplied_orchestrator or PipecatOrchestrator(registry)
        app.state.call_contexts = {}
        provider = getattr(getattr(getattr(tenant_config, "tts", None), "provider", None), "value", None)
        if provider == "elevenlabs":
            app.state.tts_provider_config = {"output_format": "pcm_16000"}
            app.state.tts_sample_rate = 16000
        elif provider == "openai":
            app.state.tts_provider_config = {"response_format": "pcm"}
            app.state.tts_sample_rate = 24000
        else:
            app.state.tts_provider_config = None
            app.state.tts_sample_rate = 16000
        yield

    app = FastAPI(title="Call Automation Voice Agent", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/twilio/voice")
    async def twilio_voice(request: Request) -> Response:
        if not configured.twilio_public_url.startswith("https://"):
            raise HTTPException(503, "TWILIO_PUBLIC_URL must be an HTTPS tunnel or deployment URL")
        if not configured.voice_agent_tenant_id:
            raise HTTPException(503, "VOICE_AGENT_TENANT_ID is not configured")

        body = (await request.body()).decode("utf-8")
        parsed = parse_qs(body, keep_blank_values=True)
        parameters = {key: values[-1] for key, values in parsed.items()}
        if configured.twilio_validate_signature:
            token = configured.twilio_auth_token
            signature = request.headers.get("X-Twilio-Signature", "")
            validator = RequestValidator(token)
            signature_url = f"{configured.twilio_public_url.rstrip('/')}{request.url.path}"
            if request.url.query:
                signature_url = f"{signature_url}?{request.url.query}"
            if not token or not validator.validate(signature_url, parameters, signature):
                raise HTTPException(403, "Invalid Twilio request signature")

        call_sid = parameters.get("CallSid", "")
        caller = parameters.get("From", "")
        called = parameters.get("To", "")
        if not call_sid or not caller or not called:
            raise HTTPException(400, "Twilio request is missing CallSid, From, or To")

        app.state.tenant_registry.register_caller(caller, configured.voice_agent_tenant_id)
        app.state.call_contexts[call_sid] = {
            "From": caller,
            "To": called,
            "Direction": parameters.get("Direction", "inbound"),
            "TenantId": configured.voice_agent_tenant_id,
        }

        stream_url = configured.twilio_public_url.replace("https://", "wss://", 1)
        stream_url = f"{stream_url.rstrip('/')}/twilio/stream"
        response = Element("Response")
        connect = SubElement(response, "Connect")
        stream = SubElement(connect, "Stream", {"url": stream_url})
        for name, value in (("CallSid", call_sid), ("TenantId", configured.voice_agent_tenant_id)):
            SubElement(stream, "Parameter", {"name": name, "value": value})
        xml = tostring(response, encoding="unicode")
        return Response(content=f'<?xml version="1.0" encoding="UTF-8"?>{xml}', media_type="application/xml")

    @app.websocket("/twilio/stream")
    async def twilio_stream(websocket: WebSocket) -> None:
        call_sid = websocket.query_params.get("call_sid", "")
        if configured.twilio_validate_signature:
            token = configured.twilio_auth_token
            signature = websocket.headers.get("X-Twilio-Signature", "")
            signature_url = f"{configured.twilio_public_url.rstrip('/')}/twilio/stream"
            validator = RequestValidator(token)
            valid = bool(token) and (
                validator.validate(signature_url, {}, signature)
                or validator.validate(f"{signature_url}/", {}, signature)
            )
            if not valid:
                await websocket.close(code=1008, reason="Invalid Twilio signature")
                return

        await websocket.accept()
        integration = TwilioIntegration(
            TwilioConfig(
                account_sid=configured.twilio_account_sid,
                auth_token=configured.twilio_auth_token,
                connection_type=ConnectionType.WEBSOCKET,
            )
        )
        bridge = TwilioVoiceBridge(
            BridgeConfig(
                session_id=call_sid,
                convert_twilio_audio=True,
                tts_sample_rate=app.state.tts_sample_rate,
                tts_provider_config=app.state.tts_provider_config,
            ),
            integration,
            app.state.orchestrator,
        )
        await bridge.start()
        try:
            await integration.serve(_TwilioWebSocketAdapter(websocket, app.state.call_contexts))
        except Exception:
            logger.exception("Twilio stream failed for call %s", call_sid)
            raise
        finally:
            await bridge.stop()
            app.state.call_contexts.pop(call_sid, None)

    return app


app = create_app()


def main() -> None:
    """Start the HTTP/WebSocket server for local or hosted operation."""
    import uvicorn

    configured = Settings()
    uvicorn.run(
        "voice_agent.telephony.server:app",
        host=configured.server_host,
        port=configured.server_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
