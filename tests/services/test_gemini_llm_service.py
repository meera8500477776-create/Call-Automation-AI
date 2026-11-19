from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from voice_agent.config.schema import LLMConfig as TenantLLMConfig
from voice_agent.services.gemini_llm_service import GeminiLLMService
from voice_agent.services.llm_service import ConversationContext, LLMConfig


class AsyncModels:
    def __init__(self):
        self.list = AsyncMock(return_value=AsyncItems([SimpleNamespace(name="models/gemini-3.8-flash")]))


class AsyncInteractions:
    def __init__(self):
        self.create = AsyncMock()


class AsyncItems:
    def __init__(self, items):
        self.items = items

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for item in self.items:
            yield item


class AsyncResponses:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def setup_gemini():
    models = AsyncModels()
    interactions = AsyncInteractions()
    client = SimpleNamespace(aio=SimpleNamespace(models=models, interactions=interactions, aclose=AsyncMock()))
    with patch("voice_agent.services.gemini_llm_service.genai.Client", return_value=client):
        service = GeminiLLMService(
            LLMConfig(provider="gemini", model="gemini-3.8-flash", system_prompt="Be Alex."),
            api_key="test-gemini-key",
        )
    return service, models, interactions, client


@pytest.fixture
def conversation():
    return ConversationContext(
        session_id="session-1",
        conversation_history=[
            {"role": "user", "content": "I need help with my order."},
            {"role": "assistant", "content": "I can help. What is your order number?"},
            {"role": "user", "content": "It is 123."},
        ],
    )


@pytest.mark.asyncio
async def test_streamed_response_preserves_context_and_does_not_duplicate_current_turn(setup_gemini, conversation):
    service, models, interactions, _ = setup_gemini
    interactions.create.return_value = AsyncResponses(
        [
            SimpleNamespace(event_type="interaction.created", interaction=SimpleNamespace(id="int-1", usage=None)),
            SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text="Thanks, ")),
            SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text="I found your order. Is there anything else?")),
            SimpleNamespace(event_type="interaction.completed", interaction=SimpleNamespace(id="int-1", usage=SimpleNamespace(input_tokens=10, output_tokens=8, total_tokens=18))),
        ]
    )

    response = await service.generate_response("It is 123.", conversation)

    assert response.text.startswith("Thanks, I found your order.")
    assert response.metadata["provider"] == "gemini"
    assert models.list.await_count == 0
    request = interactions.create.await_args.kwargs
    assert request["input"] == "It is 123."
    assert request["previous_interaction_id"] is None
    assert request["system_instruction"].startswith("Be Alex.")
    assert service._session_interactions["session-1"] == "int-1"


@pytest.mark.asyncio
async def test_second_turn_continues_gemini_interaction(setup_gemini, conversation):
    service, _, interactions, _ = setup_gemini
    interactions.create.side_effect = [
        AsyncResponses([
            SimpleNamespace(event_type="interaction.created", interaction=SimpleNamespace(id="int-1", usage=None)),
            SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text="Your order is on its way.")),
        ]),
        AsyncResponses([
            SimpleNamespace(event_type="interaction.created", interaction=SimpleNamespace(id="int-2", usage=None)),
            SimpleNamespace(event_type="step.delta", delta=SimpleNamespace(type="text", text="I will keep it concise.")),
        ]),
    ]

    await service.generate_response("Where is my order?", conversation)
    response = await service.generate_response("Please remember I like short answers.", conversation)

    assert response.text == "I will keep it concise."
    assert interactions.create.await_args_list[1].kwargs["previous_interaction_id"] == "int-1"
    assert service._session_interactions["session-1"] == "int-2"


@pytest.mark.asyncio
async def test_rejects_model_not_discovered(setup_gemini, conversation):
    service, models, interactions, _ = setup_gemini
    models.list.return_value = AsyncItems([SimpleNamespace(name="models/gemini-3.8-pro")])

    with pytest.raises(RuntimeError, match="not available"):
        await service.validate_model()

    interactions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_closes_async_sdk_client(setup_gemini):
    service, _, _, client = setup_gemini
    service._session_interactions["session-1"] = "interaction-1"

    await service.aclose()

    client.aio.aclose.assert_awaited_once()
    assert service._session_interactions == {}


def test_gemini_service_requires_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiLLMService(LLMConfig(provider="gemini", model="gemini-3.8-flash"), api_key="")


def test_tenant_schema_config_can_initialize_gemini_service():
    client = SimpleNamespace(aio=SimpleNamespace(aclose=AsyncMock()))
    with patch("voice_agent.services.gemini_llm_service.genai.Client", return_value=client):
        service = GeminiLLMService(
            TenantLLMConfig(provider="gemini", model="gemini-3.8-flash"),
            api_key="test-gemini-key",
        )
    assert service.config.timeout == 30.0
