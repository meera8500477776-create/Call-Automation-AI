"""
Unit tests for Pipecat Orchestrator.

Tests the orchestrator's ability to route calls, initialize pipelines,
and manage session lifecycle.

**Validates: Requirements 1.3, 1.4, 11.5**
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.voice_agent.core.pipecat_orchestrator import (
    PipecatOrchestrator,
    CallMetadata,
)
from src.voice_agent.config.registry import TenantRegistry
from src.voice_agent.config.schema import (
    TenantConfig,
    PersonaConfig,
    LLMConfig,
    TTSConfig,
)
from src.voice_agent.core.session_manager import SessionManager
from src.voice_agent.core.voice_pipeline import VoicePipeline


class MockSTTConfig:
    """Mock STT config for testing."""
    provider = "deepgram"
    model = "nova-2"
    language = "en-US"
    api_key = ""
    interim_results = True
    punctuate = True


@pytest.fixture
def mock_tenant_config():
    """Create a mock tenant configuration."""
    config = Mock()
    config.tenant_id = "test_tenant"
    config.persona = Mock()
    config.persona.name = "TestBot"
    config.persona.voice_id = "test_voice_123"
    config.persona.system_prompt = "You are a helpful assistant."
    config.llm = Mock()
    config.llm.provider = "openai"
    config.llm.model = "gpt-4"
    config.llm.temperature = 0.7
    config.llm.max_tokens = 150
    config.stt = Mock()
    config.stt.provider = "deepgram"
    config.stt.model = "nova-2"
    config.stt.language = "en-US"
    config.stt.api_key = ""
    config.stt.interim_results = True
    config.stt.punctuate = True
    config.tts = Mock()
    config.tts.provider = "elevenlabs"
    config.tts.voice_id = "test_voice_123"
    config.tts.stability = 0.5
    config.tts.similarity_boost = 0.75
    config.tts.style = 0.0
    config.tts.use_speaker_boost = True
    config.tools = []
    config.knowledge_bases = []
    return config


@pytest.fixture
def mock_tenant_registry(mock_tenant_config, tmp_path):
    """Create a mock tenant registry."""
    registry = Mock(spec=TenantRegistry)
    registry.get_tenant_by_caller = Mock(return_value="test_tenant")
    registry.load_tenant_config = Mock(return_value=mock_tenant_config)
    return registry


@pytest.fixture
def mock_session_manager():
    """Create a mock session manager."""
    manager = Mock(spec=SessionManager)
    manager.create_session = Mock(return_value=Mock(session_id="test_session"))
    manager.cleanup_session = Mock(return_value={
        "session_id": "test_session",
        "duration_seconds": 120.5,
    })
    manager.delete_session = Mock(return_value=True)
    return manager


@pytest.fixture
def orchestrator(mock_tenant_registry, mock_session_manager):
    """Create orchestrator instance with mocked dependencies."""
    return PipecatOrchestrator(
        tenant_registry=mock_tenant_registry,
        session_manager=mock_session_manager,
    )


@pytest.fixture
def call_metadata():
    """Create sample call metadata."""
    return CallMetadata(
        caller_id="+15551234567",
        call_sid="CA1234567890abcdef",
        from_number="+15551234567",
        to_number="+15559876543",
        additional_data={"source": "twilio"},
    )


class TestPipecatOrchestrator:
    """Test suite for PipecatOrchestrator."""

    def test_initialization(self, mock_tenant_registry):
        """Test orchestrator initialization."""
        orchestrator = PipecatOrchestrator(tenant_registry=mock_tenant_registry)
        
        assert orchestrator._tenant_registry == mock_tenant_registry
        assert orchestrator._session_manager is not None
        assert orchestrator._llm_factory is not None
        assert len(orchestrator._active_pipelines) == 0

    def test_initialization_with_custom_session_manager(
        self, mock_tenant_registry, mock_session_manager
    ):
        """Test orchestrator initialization with custom session manager."""
        orchestrator = PipecatOrchestrator(
            tenant_registry=mock_tenant_registry,
            session_manager=mock_session_manager,
        )
        
        assert orchestrator._session_manager == mock_session_manager

    def test_route_call_success(
        self, orchestrator, call_metadata, mock_tenant_config
    ):
        """Test successful call routing to tenant."""
        tenant_config = orchestrator.route_call(
            caller_id=call_metadata.caller_id,
            call_metadata=call_metadata,
        )
        
        assert tenant_config == mock_tenant_config
        orchestrator._tenant_registry.get_tenant_by_caller.assert_called_once_with(
            call_metadata.caller_id
        )
        orchestrator._tenant_registry.load_tenant_config.assert_called_once_with(
            "test_tenant"
        )

    def test_route_call_unknown_caller(self, orchestrator, call_metadata):
        """Test call routing with unknown caller ID."""
        orchestrator._tenant_registry.get_tenant_by_caller.side_effect = KeyError(
            "No tenant mapping found"
        )
        
        with pytest.raises(KeyError, match="No tenant mapping found"):
            orchestrator.route_call(
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

    def test_route_call_missing_config(self, orchestrator, call_metadata):
        """Test call routing with missing tenant configuration."""
        orchestrator._tenant_registry.load_tenant_config.side_effect = FileNotFoundError(
            "Configuration file not found"
        )
        
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            orchestrator.route_call(
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )

    @patch('src.voice_agent.core.pipecat_orchestrator.VoicePipeline')
    def test_initialize_pipeline(
        self, mock_pipeline_class, orchestrator, mock_tenant_config
    ):
        """Test pipeline initialization from tenant configuration."""
        mock_pipeline = Mock(spec=VoicePipeline)
        mock_pipeline_class.return_value = mock_pipeline
        
        pipeline = orchestrator.initialize_pipeline(
            session_id="test_session",
            tenant_config=mock_tenant_config,
        )
        
        assert pipeline == mock_pipeline
        mock_pipeline_class.assert_called_once()
        
        # Verify pipeline config was created with correct parameters
        call_args = mock_pipeline_class.call_args
        pipeline_config = call_args[0][0]
        
        assert pipeline_config.session_id == "test_session"
        assert pipeline_config.llm_service is not None
        assert pipeline_config.stt_config is not None
        assert pipeline_config.vad_config is not None
        assert pipeline_config.voice_config is not None

    @pytest.mark.asyncio
    @patch('src.voice_agent.core.pipecat_orchestrator.VoicePipeline')
    async def test_start_session_success(
        self, mock_pipeline_class, orchestrator, call_metadata, mock_tenant_config
    ):
        """Test successful session start."""
        mock_pipeline = AsyncMock(spec=VoicePipeline)
        mock_pipeline.initialize = AsyncMock()
        mock_pipeline_class.return_value = mock_pipeline
        
        pipeline = await orchestrator.start_session(
            session_id="test_session",
            caller_id=call_metadata.caller_id,
            call_metadata=call_metadata,
        )
        
        assert pipeline == mock_pipeline
        
        # Verify session was created
        orchestrator._session_manager.create_session.assert_called_once()
        create_call = orchestrator._session_manager.create_session.call_args
        assert create_call[1]["tenant_id"] == "test_tenant"
        assert create_call[1]["caller_id"] == call_metadata.caller_id
        assert create_call[1]["session_id"] == "test_session"
        
        # Verify pipeline was initialized
        mock_pipeline.initialize.assert_called_once()
        
        # Verify pipeline was stored
        assert "test_session" in orchestrator._active_pipelines
        assert orchestrator._active_pipelines["test_session"] == mock_pipeline

    @pytest.mark.asyncio
    @patch('src.voice_agent.core.pipecat_orchestrator.VoicePipeline')
    async def test_start_session_failure_cleanup(
        self, mock_pipeline_class, orchestrator, call_metadata
    ):
        """Test session cleanup on start failure."""
        mock_pipeline = AsyncMock(spec=VoicePipeline)
        mock_pipeline.initialize = AsyncMock(side_effect=RuntimeError("Init failed"))
        mock_pipeline_class.return_value = mock_pipeline
        
        with pytest.raises(RuntimeError, match="Init failed"):
            await orchestrator.start_session(
                session_id="test_session",
                caller_id=call_metadata.caller_id,
                call_metadata=call_metadata,
            )
        
        # Verify session was deleted on failure
        orchestrator._session_manager.delete_session.assert_called_once_with("test_session")

    @pytest.mark.asyncio
    async def test_stop_session_success(self, orchestrator):
        """Test successful session stop."""
        # Create mock pipeline
        mock_pipeline = AsyncMock(spec=VoicePipeline)
        mock_pipeline.shutdown = AsyncMock()
        orchestrator._active_pipelines["test_session"] = mock_pipeline
        
        summary = await orchestrator.stop_session("test_session", persist=True)
        
        # Verify pipeline was shutdown
        mock_pipeline.shutdown.assert_called_once()
        
        # Verify session was cleaned up
        orchestrator._session_manager.cleanup_session.assert_called_once_with(
            "test_session", persist=True
        )
        
        # Verify pipeline was removed
        assert "test_session" not in orchestrator._active_pipelines
        
        # Verify summary was returned
        assert summary["session_id"] == "test_session"
        assert summary["duration_seconds"] == 120.5

    @pytest.mark.asyncio
    async def test_stop_session_no_pipeline(self, orchestrator):
        """Test stopping session with no active pipeline."""
        summary = await orchestrator.stop_session("nonexistent_session")
        
        # Should still cleanup session
        orchestrator._session_manager.cleanup_session.assert_called_once_with(
            "nonexistent_session", persist=True
        )

    @pytest.mark.asyncio
    async def test_stop_session_pipeline_error(self, orchestrator):
        """Test session stop with pipeline shutdown error."""
        mock_pipeline = AsyncMock(spec=VoicePipeline)
        mock_pipeline.shutdown = AsyncMock(side_effect=RuntimeError("Shutdown failed"))
        orchestrator._active_pipelines["test_session"] = mock_pipeline
        
        # Should not raise, but log error
        summary = await orchestrator.stop_session("test_session")
        
        # Verify pipeline was still removed
        assert "test_session" not in orchestrator._active_pipelines
        
        # Verify session was still cleaned up
        orchestrator._session_manager.cleanup_session.assert_called_once()

    def test_get_active_pipeline(self, orchestrator):
        """Test retrieving active pipeline."""
        mock_pipeline = Mock(spec=VoicePipeline)
        orchestrator._active_pipelines["test_session"] = mock_pipeline
        
        pipeline = orchestrator.get_active_pipeline("test_session")
        assert pipeline == mock_pipeline
        
        # Test nonexistent session
        pipeline = orchestrator.get_active_pipeline("nonexistent")
        assert pipeline is None

    def test_get_active_session_ids(self, orchestrator):
        """Test retrieving active session IDs."""
        orchestrator._active_pipelines["session1"] = Mock()
        orchestrator._active_pipelines["session2"] = Mock()
        orchestrator._active_pipelines["session3"] = Mock()
        
        session_ids = orchestrator.get_active_session_ids()
        
        assert len(session_ids) == 3
        assert "session1" in session_ids
        assert "session2" in session_ids
        assert "session3" in session_ids

    def test_get_session_count(self, orchestrator):
        """Test getting active session count."""
        assert orchestrator.get_session_count() == 0
        
        orchestrator._active_pipelines["session1"] = Mock()
        assert orchestrator.get_session_count() == 1
        
        orchestrator._active_pipelines["session2"] = Mock()
        orchestrator._active_pipelines["session3"] = Mock()
        assert orchestrator.get_session_count() == 3

    def test_build_stt_config(self, orchestrator, mock_tenant_config):
        """Test STT configuration building."""
        stt_config = orchestrator._build_stt_config(mock_tenant_config)
        
        assert stt_config.model == "nova-2"
        assert stt_config.language == "en-US"
        assert stt_config.enable_interim_results is True
        assert stt_config.enable_punctuation is True

    def test_build_vad_config(self, orchestrator, mock_tenant_config):
        """Test VAD configuration building."""
        vad_config = orchestrator._build_vad_config(mock_tenant_config)
        
        # Should use defaults when not specified
        assert vad_config.threshold == 0.5
        assert vad_config.min_speech_duration_ms == 250
        assert vad_config.min_silence_duration_ms == 500

    def test_build_voice_config(self, orchestrator, mock_tenant_config):
        """Test voice configuration building."""
        voice_config = orchestrator._build_voice_config(mock_tenant_config)
        
        assert voice_config.provider == "elevenlabs"
        assert voice_config.voice_id == "test_voice_123"
        assert voice_config.stability == 0.5
        assert voice_config.similarity_boost == 0.75
