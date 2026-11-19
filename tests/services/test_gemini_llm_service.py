from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from voice_agent.config.schema import LLMConfig as TenantLLMConfig
from voice_agent.services.gemini_llm_service import GeminiLLMService
from voice_agent.services.llm_service import ConversationContext, LLMConfig


class AsyncModels:
    def __init__(self):
        self.list = AsyncMock(return_value=AsyncItems([SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["generateContent"])]))
        self.generate_content_stream = AsyncMock()


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
    client = SimpleNamespace(aio=SimpleNamespace(models=models, aclose=AsyncMock()))
    with patch("voice_agent.services.gemini_llm_service.genai.Client", return_value=client):
        service = GeminiLLMService(
            LLMConfig(provider="gemini", model="gemini-3.8-flash", system_prompt="Be Alex."),
            api_key="test-gemini-key",
        )
    return service, models, client


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
    service, models, _ = setup_gemini
    models.generate_content_stream.return_value = AsyncResponses(
        [
            SimpleNamespace(text="Thanks, ", usage_metadata=None),
            SimpleNamespace(text="I found your order. Is there anything else?", usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=8, total_token_count=18)),
        ]
    )

    response = await service.generate_response("It is 123.", conversation)

    assert response.text.startswith("Thanks, I found your order.")
    assert response.metadata["provider"] == "gemini"
    assert models.list.await_count == 0
    request = models.generate_content_stream.await_args.kwargs
    assert [item.role for item in request["contents"]] == ["user", "model", "user"]
    assert request["contents"][-1].parts[0].text == "It is 123."
    assert request["config"].system_instruction.startswith("Be Alex.")


@pytest.mark.asyncio
async def test_second_turn_continues_gemini_interaction(setup_gemini, conversation):
    service, models, _ = setup_gemini
    models.generate_content_stream.side_effect = [
        AsyncResponses([SimpleNamespace(text="Your order is on its way.", usage_metadata=None)]),
        AsyncResponses([SimpleNamespace(text="I will keep it concise.", usage_metadata=None)]),
    ]

    await service.generate_response("Where is my order?", conversation)
    response = await service.generate_response("Please remember I like short answers.", conversation)

    assert response.text == "I will keep it concise."
    second_contents = models.generate_content_stream.await_args_list[1].kwargs["contents"]
    assert [item.role for item in second_contents] == ["user", "model", "user", "user"]
    assert second_contents[1].parts[0].text == "I can help. What is your order number?"
    assert second_contents[-1].parts[0].text == "Please remember I like short answers."


@pytest.mark.asyncio
async def test_rejects_model_not_discovered(setup_gemini, conversation):
    service, models, _ = setup_gemini
    models.list.return_value = AsyncItems([SimpleNamespace(name="models/gemini-3.8-pro", supported_actions=["generateContent"])])

    with pytest.raises(RuntimeError, match="not available"):
        await service.validate_model()

    models.generate_content_stream.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejects_model_without_text_generation_action(setup_gemini):
    service, models, _ = setup_gemini
    models.list.return_value = AsyncItems([SimpleNamespace(name="models/gemini-3.8-flash", supported_actions=["embedContent"])])

    with pytest.raises(RuntimeError, match="does not support generateContent"):
        await service.validate_model()


@pytest.mark.asyncio
async def test_closes_async_sdk_client(setup_gemini):
    service, _, client = setup_gemini

    await service.aclose()

    client.aio.aclose.assert_awaited_once()


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


def test_high_demand_503_is_reported_and_retryable(setup_gemini):
    service, _, _ = setup_gemini
    error = RuntimeError("503 UNAVAILABLE: This model is currently experiencing high demand.")

    assert service._is_retryable(error)
    assert service._error_summary(error) == (
        "Google Gemini HTTP 503 UNAVAILABLE: the selected model is experiencing high demand"
    )


def test_deadline_504_is_reported_without_exposing_a_key(setup_gemini):
    service, _, _ = setup_gemini
    error = RuntimeError("504 DEADLINE_EXCEEDED while using key=sk-test-sensitive-value")
    service._api_key = "sk-test-sensitive-value"

    assert service._is_retryable(error)
    summary = service._error_summary(error)
    assert summary.startswith("Google Gemini HTTP 504 DEADLINE_EXCEEDED:")
    assert "sk-test-sensitive-value" not in summary
