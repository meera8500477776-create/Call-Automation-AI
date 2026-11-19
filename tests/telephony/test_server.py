"""Tests for the FastAPI Twilio webhook and stream lifecycle."""

import json
from xml.etree import ElementTree

import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from voice_agent.config.settings import Settings
from voice_agent.telephony.server import create_app


class StubRegistry:
    def __init__(self):
        self.callers = {}

    def register_caller(self, caller, tenant):
        self.callers[caller] = tenant

    def load_tenant_config(self, _tenant):
        return object()


class StubPipeline:
    is_active = True

    def on_audio_generated(self, _callback):
        pass

    def on_latency_metric(self, _callback):
        pass

    def on_interruption(self, _callback):
        pass

    async def process_audio_chunk(self, audio):
        self.last_audio = audio


class StubOrchestrator:
    def __init__(self):
        self.started = []
        self.stopped = []
        self.pipeline = StubPipeline()

    async def start_session(self, **kwargs):
        self.started.append(kwargs)
        return self.pipeline

    async def stop_session(self, session_id, persist=True):
        self.stopped.append((session_id, persist))
        return {"turn_count": 0}


def make_client(validate_signature=False):
    registry = StubRegistry()
    orchestrator = StubOrchestrator()
    settings = Settings(
        twilio_account_sid="test-account",
        twilio_auth_token="test-token",
        twilio_public_url="https://voice.example.test",
        voice_agent_tenant_id="customer_service_001",
        twilio_validate_signature=validate_signature,
    )
    app = create_app(settings, registry, orchestrator)
    return TestClient(app), registry, orchestrator


def test_health_and_twilio_webhook_builds_secure_stream_twiml():
    client, registry, _orchestrator = make_client()
    with client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post(
            "/twilio/voice",
            data={
                "CallSid": "CA123",
                "From": "+15551234567",
                "To": "+15557654321",
                "Direction": "inbound",
            },
        )
    assert response.status_code == 200
    root = ElementTree.fromstring(response.text)
    assert root.find("./Connect/Stream").attrib["url"] == "wss://voice.example.test/twilio/stream"
    assert root.find("./Connect/Stream/Parameter[@name='CallSid']").attrib["value"] == "CA123"
    assert registry.callers["+15551234567"] == "customer_service_001"


def test_twilio_webhook_rejects_invalid_signature():
    client, _registry, _orchestrator = make_client(validate_signature=True)
    with client:
        response = client.post(
            "/twilio/voice",
            data={"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321"},
        )
    assert response.status_code == 403


def test_twilio_websocket_rejects_invalid_signature():
    client, _registry, _orchestrator = make_client(validate_signature=True)
    with client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/twilio/stream"):
                pass


def test_twilio_websocket_runs_call_start_stop_and_audio_lifecycle():
    client, _registry, orchestrator = make_client()
    with client:
        client.post(
            "/twilio/voice",
            data={"CallSid": "CA123", "From": "+15551234567", "To": "+15557654321"},
        )
        with client.websocket_connect("/twilio/stream") as websocket:
            websocket.send_text(json.dumps({"event": "connected"}))
            websocket.send_text(
                json.dumps(
                    {
                        "event": "start",
                        "streamSid": "MZ123",
                        "start": {"callSid": "CA123", "customParameters": {"CallSid": "CA123"}},
                    }
                )
            )
            websocket.send_text(
                json.dumps(
                    {
                        "event": "media",
                        "media": {"payload": "AQID"},
                    }
                )
            )
            websocket.send_text(json.dumps({"event": "stop", "streamSid": "MZ123"}))

    assert orchestrator.started
    assert orchestrator.started[0]["call_metadata"].caller_id == "+15551234567"
    assert orchestrator.stopped == [("CA123", True)]
    assert len(orchestrator.pipeline.last_audio) == 12
