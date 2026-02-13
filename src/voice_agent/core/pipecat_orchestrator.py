"""
Pipecat Orchestrator for managing voice agent pipeline lifecycle.

This module provides the PipecatOrchestrator class that integrates all components
of the voice agent platform: Tenant Registry, Session Manager, Voice Pipeline,
and handles call routing and session lifecycle management.

**Validates: Requirements 1.3, 1.4, 11.5**
"""

import logging
from dataclasses import dataclass
from typing import Any

from ..config.registry import TenantRegistry
from ..config.schema import TenantConfig
from ..services.elevenlabs_tts_service import VoiceConfig
from ..services.llm_factory import LLMProviderFactory
from ..services.stt_service import STTConfig
from ..services.vad_module import VADConfig
from .session_manager import SessionManager
from .voice_pipeline import PipelineConfig, StreamingConfig, VoicePipeline

logger = logging.getLogger(__name__)


@dataclass
class CallMetadata:
    """Metadata for an incoming call."""
    caller_id: str
    call_sid: str
    from_number: str | None = None
    to_number: str | None = None
    additional_data: dict[str, Any] | None = None


class PipecatOrchestrator:
    """
    Pipecat Orchestrator integrating all voice agent components.
    
    **Validates: Requirements 1.3, 1.4, 11.5**
    """

    def __init__(
        self,
        tenant_registry: TenantRegistry,
        session_manager: SessionManager | None = None,
    ):
        """Initialize the Pipecat Orchestrator."""
        self._tenant_registry = tenant_registry
        self._session_manager = session_manager or SessionManager()
        self._llm_factory = LLMProviderFactory()
        self._active_pipelines: dict[str, VoicePipeline] = {}
        logger.info("Pipecat Orchestrator initialized")

    def route_call(self, caller_id: str, call_metadata: CallMetadata) -> TenantConfig:
        """Route incoming call to appropriate tenant configuration."""
        logger.info("Routing call from caller_id=%s", caller_id)
        tenant_id = self._tenant_registry.get_tenant_by_caller(caller_id)
        tenant_config = self._tenant_registry.load_tenant_config(tenant_id)
        logger.info("Routed to tenant: %s", tenant_id)
        return tenant_config

    async def initialize_pipeline(
        self,
        session_id: str,
        tenant_config: TenantConfig,
    ) -> VoicePipeline:
        """Initialize voice pipeline from tenant configuration."""
        logger.info("Initializing pipeline for session=%s, tenant=%s", session_id, tenant_config.tenant_id)

        # Create LLM service
        llm_service = await self._llm_factory.create_llm_service(tenant_config.llm)

        # Build STT configuration
        stt_config = self._build_stt_config(tenant_config)

        # Build VAD configuration
        vad_config = self._build_vad_config(tenant_config)

        # Build voice configuration
        voice_config = self._build_voice_config(tenant_config)

        # Build streaming configuration
        streaming_config = StreamingConfig(
            enable_concurrent_processing=True,
            enable_early_llm_start=False,
        )

        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            session_id=session_id,
            stt_config=stt_config,
            llm_service=llm_service,
            vad_config=vad_config,
            voice_config=voice_config,
            enable_barge_in=True,
            enable_interim_processing=False,
            streaming_config=streaming_config,
        )

        # Create and return pipeline
        pipeline = VoicePipeline(pipeline_config)
        logger.info("Pipeline initialized for session: %s", session_id)
        return pipeline

    def _build_stt_config(self, tenant_config: TenantConfig) -> STTConfig:
        """Build STT configuration from tenant config."""
        stt = getattr(tenant_config, 'stt', None)
        return STTConfig(
            api_key=getattr(stt, 'api_key', None) or "" if stt else "",
            model=getattr(stt, 'model', "nova-2") if stt else "nova-2",
            language=getattr(stt, 'language', "en-US") if stt else "en-US",
            interim_results=getattr(stt, 'interim_results', True) if stt else True,
            punctuate=getattr(stt, 'punctuate', True) if stt else True,
        )

    def _build_vad_config(self, tenant_config: TenantConfig) -> VADConfig:
        """Build VAD configuration from tenant config."""
        vad = getattr(tenant_config, 'vad', None)
        return VADConfig(
            threshold=getattr(vad, 'threshold', 0.5) if vad else 0.5,
            min_speech_duration_ms=getattr(vad, 'min_speech_duration_ms', 250) if vad else 250,
            min_silence_duration_ms=getattr(vad, 'min_silence_duration_ms', 500) if vad else 500,
        )

    def _build_voice_config(self, tenant_config: TenantConfig) -> VoiceConfig:
        """Build voice configuration from tenant config."""
        return VoiceConfig(
            provider=getattr(tenant_config.tts, 'provider', "elevenlabs"),
            voice_id=tenant_config.persona.voice_id,
            stability=getattr(tenant_config.tts, 'stability', 0.5),
            similarity_boost=getattr(tenant_config.tts, 'similarity_boost', 0.75),
            style=getattr(tenant_config.tts, 'style', 0.0),
            use_speaker_boost=getattr(tenant_config.tts, 'use_speaker_boost', True),
        )

    async def start_session(
        self,
        session_id: str,
        caller_id: str,
        call_metadata: CallMetadata,
    ) -> VoicePipeline:
        """
        Start a new voice session.
        
        Creates session, routes to tenant, initializes pipeline, and starts it.
        
        Args:
            session_id: Unique session identifier
            caller_id: Caller identification
            call_metadata: Call metadata
            
        Returns:
            Initialized and started VoicePipeline
            
        Raises:
            KeyError: If caller cannot be routed to tenant
            RuntimeError: If pipeline initialization fails
            
        **Validates: Requirements 1.3, 1.4, 11.5**
        """
        logger.info("Starting session: %s for caller: %s", session_id, caller_id)

        try:
            # Route call to tenant
            tenant_config = self.route_call(caller_id, call_metadata)

            # Create session
            session = self._session_manager.create_session(
                tenant_id=tenant_config.tenant_id,
                caller_id=caller_id,
                session_id=session_id,
                metadata={
                    "call_sid": call_metadata.call_sid,
                    "from_number": call_metadata.from_number,
                    "to_number": call_metadata.to_number,
                    "additional_data": call_metadata.additional_data,
                },
            )
            logger.info("Created session: %s for tenant: %s", session_id, tenant_config.tenant_id)

            # Initialize pipeline
            pipeline = await self.initialize_pipeline(session_id, tenant_config)

            # Initialize pipeline components
            await pipeline.initialize()

            # Store active pipeline
            self._active_pipelines[session_id] = pipeline

            logger.info("Session started successfully: %s", session_id)
            return pipeline

        except Exception as e:
            logger.error("Failed to start session %s: %s", session_id, str(e))
            # Cleanup on failure
            self._session_manager.delete_session(session_id)
            raise

    async def stop_session(
        self,
        session_id: str,
        persist: bool = True,
    ) -> dict[str, Any] | None:
        """
        Stop a voice session and cleanup resources.
        
        Shuts down pipeline, ends session, and optionally persists logs.
        
        Args:
            session_id: Session identifier to stop
            persist: Whether to persist session data (default: True)
            
        Returns:
            Session summary dictionary, or None if session not found
            
        **Validates: Requirements 11.5, 11.6**
        """
        logger.info("Stopping session: %s (persist=%s)", session_id, persist)

        # Get pipeline
        pipeline = self._active_pipelines.get(session_id)

        if pipeline:
            try:
                # Shutdown pipeline
                await pipeline.shutdown()
                logger.info("Pipeline shutdown complete for session: %s", session_id)
            except Exception as e:
                logger.error("Error shutting down pipeline for session %s: %s", session_id, str(e))
            finally:
                # Remove from active pipelines
                del self._active_pipelines[session_id]
        else:
            logger.warning("No active pipeline found for session: %s", session_id)

        # Cleanup session
        summary = self._session_manager.cleanup_session(session_id, persist=persist)

        if summary:
            logger.info("Session stopped: %s (duration: %.2fs)",
                       session_id,
                       summary.get("duration_seconds", 0))
        else:
            logger.warning("Session not found: %s", session_id)

        return summary

    def get_active_pipeline(self, session_id: str) -> VoicePipeline | None:
        """
        Get active pipeline for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            VoicePipeline if active, None otherwise
        """
        return self._active_pipelines.get(session_id)

    def get_active_session_ids(self) -> list[str]:
        """
        Get list of active session IDs.
        
        Returns:
            List of active session identifiers
        """
        return list(self._active_pipelines.keys())

    def get_session_count(self) -> int:
        """
        Get count of active sessions.
        
        Returns:
            Number of active sessions
        """
        return len(self._active_pipelines)
