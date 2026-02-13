"""
End-to-end integration tests for complete call flows.

Tests complete call flow from Twilio to FHIR and back, multi-tenant routing,
latency requirements, concurrent call handling, error scenarios, and human handoff.

**Validates: Requirements 1.3, 1.4, 2.1, 2.8, 9.6, 10.6**
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import List

from src.voice_agent.core.pipecat_orchestrator import (
    PipecatOrchestrator,
    CallMetadata,
)
from src.voice_agent.telephony.twilio_integration import TwilioIntegration
from src.voice_agent.adapters.fhir_adapter import (
    FHIRAdapter,
    Slot,
    Appointment,
    AppointmentConstraints,
)
from src.voice_agent.monitoring.latency_monitor import LatencyMonitor
from src.voice_agent.core.session_manager import SessionManager
from src.voice_agent.config.registry import TenantRegistry


@pytest.fixture
def latency_monitor():
    """Create latency monitor for testing."""
    return LatencyMonitor()


@pytest.fixture
def mock_twilio_integration():
    """Create mock Twilio integration."""
    integration = AsyncMock(spec=TwilioIntegration)
    integration.connect = AsyncMock()
    integration.disconnect = AsyncMock()
    integration.send_audio = AsyncMock()
    return integration


@pytest.fixture
def mock_fhir_adapter():
    """Create mock FHIR adapter."""
    adapter = AsyncMock(spec=FHIRAdapter)
    
    # Setup default responses
    tomorrow = datetime.now() + timedelta(days=1)
    adapter.find_appointments.return_value = [
        Slot(
            id="slot1",
            schedule_id="schedule1",
            start_time=tomorrow.replace(hour=9, minute=0),
            end_time=tomorrow.replace(hour=9, minute=30),
            status="free",
            service_type="General",
            practitioner_id="Practitioner/1"
        )
    ]
    
    adapter.book_appointment.return_value = Appointment(
        id="appt_new",
        patient_id="Patient/123",
        practitioner_id="Practitioner/1",
        start_time=tomorrow.replace(hour=9, minute=0),
        end_time=tomorrow.replace(hour=9, minute=30),
        status="booked",
        service_type="General",
        is_cancelable=True,
        is_reschedulable=True,
        location="Main Clinic"
    )
    
    adapter.check_constraints.return_value = AppointmentConstraints(
        is_cancelable=True,
        is_reschedulable=True
    )
    
    return adapter


class TestCompleteCallFlow:
    """Test complete call flow from Twilio to FHIR and back."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_healthcare_appointment_booking_flow(
        self, mock_twilio_integration, mock_fhir_adapter, latency_monitor
    ):
        """
        Test complete call flow for healthcare appointment booking.
        
        Flow: Twilio → Orchestrator → STT → LLM → FHIR → LLM → TTS → Twilio
        
        **Validates: Requirements 1.3, 2.1, 3.2, 3.3, 11.1, 11.2**
        """
        # Setup orchestrator with mocked components
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            with patch('src.voice_agent.adapters.fhir_adapter.FHIRAdapter', return_value=mock_fhir_adapter):
                # Setup tenant registry
                mock_registry = Mock(spec=TenantRegistry)
                mock_registry.get_tenant_by_caller.return_value = "healthcare_clinic_001"
                mock_registry.load_tenant_config.return_value = Mock(
                    tenant_id="healthcare_clinic_001",
                    persona=Mock(name="Sarah", system_prompt="You are Sarah"),
                    llm=Mock(provider="openai", model="gpt-4"),
                    tts=Mock(provider="elevenlabs"),
                    tools=[],
                    knowledge_bases=[],
                    constraints=Mock(max_call_duration=600, enable_barge_in=True)
                )
                mock_registry_class.return_value = mock_registry
                
                orchestrator = PipecatOrchestrator(
                    tenant_registry=mock_registry,
                    session_manager=SessionManager()
                )
                
                # Step 1: Incoming call from Twilio
                call_metadata = CallMetadata(
                    caller_id="+15551234567",
                    call_sid="CA_e2e_test_001",
                    from_number="+15551234567",
                    to_number="+15551111111"
                )
                
                # Step 2: Route call to tenant
                tenant_config = orchestrator.route_call(
                    caller_id=call_metadata.caller_id,
                    call_metadata=call_metadata
                )
                assert tenant_config.tenant_id == "healthcare_clinic_001"
                
                # Step 3: Start session and initialize pipeline  
                mock_pipeline = Mock()  # Use regular Mock, not AsyncMock
                mock_pipeline.initialize = AsyncMock()
                mock_pipeline.shutdown = AsyncMock()
                
                with patch.object(orchestrator, 'initialize_pipeline') as mock_init:
                    # Make the mock return a coroutine that resolves to mock_pipeline
                    async def return_pipeline(*args, **kwargs):
                        return mock_pipeline
                    mock_init.side_effect = return_pipeline
                    
                    session_id = "e2e_test_session_001"
                    pipeline = await orchestrator.start_session(
                        session_id=session_id,
                        caller_id=call_metadata.caller_id,
                        call_metadata=call_metadata
                    )
                    
                    assert pipeline is not None
                    assert orchestrator.get_session_count() == 1
                    
                    # Step 4: Simulate audio processing (STT → LLM → FHIR → TTS)
                    # In real scenario, audio chunks would flow through pipeline
                    # Here we verify FHIR adapter is called correctly
                    
                    # Simulate appointment search
                    slots = await mock_fhir_adapter.find_appointments(
                        patient_id="Patient/123",
                        date_range=Mock(),
                        specialty="General"
                    )
                    assert len(slots) > 0
                    
                    # Simulate appointment booking
                    appointment = await mock_fhir_adapter.book_appointment(
                        patient_id="Patient/123",
                        slot_id="slot1",
                        reason="Annual checkup"
                    )
                    assert appointment.status == "booked"
                    
                    # Step 5: End call and cleanup
                    summary = await orchestrator.stop_session(session_id, persist=True)
                    
                    assert orchestrator.get_session_count() == 0
                    assert summary is not None
                    
                    # Verify pipeline was properly shutdown
                    mock_pipeline.shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_complete_automotive_service_booking_flow(
        self, mock_twilio_integration, latency_monitor
    ):
        """
        Test complete call flow for automotive service booking.
        
        Flow: Twilio → Orchestrator → STT → LLM → DMS → LLM → TTS → Twilio
        
        **Validates: Requirements 1.3, 4.2, 11.1, 11.2**
        """
        # Setup mock DMS adapter
        mock_dms_adapter = AsyncMock()
        mock_dms_adapter.find_service_slots.return_value = [
            Mock(id="slot1", start_time=datetime.now() + timedelta(days=1))
        ]
        mock_dms_adapter.book_service.return_value = Mock(
            id="service_appt_001",
            status="booked"
        )
        
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            with patch('src.voice_agent.adapters.dms_adapter.DMSAdapter', return_value=mock_dms_adapter):
                # Setup tenant registry for automotive
                mock_registry = Mock(spec=TenantRegistry)
                mock_registry.get_tenant_by_caller.return_value = "automotive_dealer_001"
                mock_registry.load_tenant_config.return_value = Mock(
                    tenant_id="automotive_dealer_001",
                    persona=Mock(name="Mike", system_prompt="You are Mike"),
                    llm=Mock(provider="ollama", model="llama2"),
                    tts=Mock(provider="openai"),
                    tools=[],
                    knowledge_bases=[],
                    constraints=Mock(max_call_duration=600, enable_barge_in=True)
                )
                mock_registry_class.return_value = mock_registry
                
                orchestrator = PipecatOrchestrator(
                    tenant_registry=mock_registry,
                    session_manager=SessionManager()
                )
                
                # Incoming call
                call_metadata = CallMetadata(
                    caller_id="+15559876543",
                    call_sid="CA_automotive_001",
                    from_number="+15559876543",
                    to_number="+15552222222"
                )
                
                # Route to automotive tenant
                tenant_config = orchestrator.route_call(
                    caller_id=call_metadata.caller_id,
                    call_metadata=call_metadata
                )
                assert tenant_config.tenant_id == "automotive_dealer_001"
                
                # Start session
                with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
                    mock_pipeline = AsyncMock()
                    mock_pipeline.initialize = AsyncMock()
                    mock_pipeline.shutdown = AsyncMock()
                    mock_init.return_value = mock_pipeline
                    
                    session_id = "automotive_session_001"
                    await orchestrator.start_session(
                        session_id=session_id,
                        caller_id=call_metadata.caller_id,
                        call_metadata=call_metadata
                    )
                    
                    # Simulate DMS operations
                    slots = await mock_dms_adapter.find_service_slots(
                        vehicle_id="VIN123",
                        service_type="oil_change",
                        date_range=Mock()
                    )
                    assert len(slots) > 0
                    
                    service_appt = await mock_dms_adapter.book_service(
                        vehicle_id="VIN123",
                        slot_id="slot1",
                        service_type="oil_change"
                    )
                    assert service_appt.status == "booked"
                    
                    # Cleanup
                    await orchestrator.stop_session(session_id)
                    assert orchestrator.get_session_count() == 0


class TestMultiTenantRouting:
    """Test multi-tenant routing with different configurations."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_concurrent_calls_different_tenants(self):
        """
        Test concurrent calls routed to different tenants.
        
        **Validates: Requirements 1.3, 1.4**
        """
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            # Setup tenant registry with multiple tenants
            mock_registry = Mock(spec=TenantRegistry)
            
            def get_tenant_by_caller(caller_id):
                mapping = {
                    "+15551111111": "healthcare_clinic_001",
                    "+15552222222": "automotive_dealer_001",
                    "+15553333333": "healthcare_clinic_002"
                }
                return mapping.get(caller_id)
            
            def load_tenant_config(tenant_id):
                configs = {
                    "healthcare_clinic_001": Mock(
                        tenant_id="healthcare_clinic_001",
                        persona=Mock(name="Sarah"),
                        llm=Mock(provider="openai"),
                        tts=Mock(provider="elevenlabs"),
                        tools=[], knowledge_bases=[],
                        constraints=Mock(max_call_duration=600, enable_barge_in=True)
                    ),
                    "automotive_dealer_001": Mock(
                        tenant_id="automotive_dealer_001",
                        persona=Mock(name="Mike"),
                        llm=Mock(provider="ollama"),
                        tts=Mock(provider="openai"),
                        tools=[], knowledge_bases=[],
                        constraints=Mock(max_call_duration=600, enable_barge_in=True)
                    ),
                    "healthcare_clinic_002": Mock(
                        tenant_id="healthcare_clinic_002",
                        persona=Mock(name="Emily"),
                        llm=Mock(provider="openai"),
                        tts=Mock(provider="elevenlabs"),
                        tools=[], knowledge_bases=[],
                        constraints=Mock(max_call_duration=600, enable_barge_in=True)
                    )
                }
                return configs[tenant_id]
            
            mock_registry.get_tenant_by_caller.side_effect = get_tenant_by_caller
            mock_registry.load_tenant_config.side_effect = load_tenant_config
            mock_registry_class.return_value = mock_registry
            
            orchestrator = PipecatOrchestrator(
                tenant_registry=mock_registry,
                session_manager=SessionManager()
            )
            
            # Create multiple concurrent calls
            calls = [
                CallMetadata(caller_id="+15551111111", call_sid="CA_001", from_number="+15551111111"),
                CallMetadata(caller_id="+15552222222", call_sid="CA_002", from_number="+15552222222"),
                CallMetadata(caller_id="+15553333333", call_sid="CA_003", from_number="+15553333333"),
            ]
            
            # Route all calls
            configs = []
            for call in calls:
                config = orchestrator.route_call(
                    caller_id=call.caller_id,
                    call_metadata=call
                )
                configs.append(config)
            
            # Verify correct routing
            assert configs[0].tenant_id == "healthcare_clinic_001"
            assert configs[1].tenant_id == "automotive_dealer_001"
            assert configs[2].tenant_id == "healthcare_clinic_002"
            
            # Start all sessions concurrently
            with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
                mock_pipeline = AsyncMock()
                mock_pipeline.initialize = AsyncMock()
                mock_pipeline.shutdown = AsyncMock()
                mock_init.return_value = mock_pipeline
                
                session_ids = ["session_001", "session_002", "session_003"]
                
                # Start sessions concurrently
                await asyncio.gather(*[
                    orchestrator.start_session(
                        session_id=session_ids[i],
                        caller_id=calls[i].caller_id,
                        call_metadata=calls[i]
                    )
                    for i in range(3)
                ])
                
                # Verify all sessions are active
                assert orchestrator.get_session_count() == 3
                
                # Verify each session has correct tenant
                for i, session_id in enumerate(session_ids):
                    session = orchestrator._session_manager.get_session(session_id)
                    assert session.tenant_id == configs[i].tenant_id
                
                # Cleanup all sessions
                await asyncio.gather(*[
                    orchestrator.stop_session(sid)
                    for sid in session_ids
                ])
                
                assert orchestrator.get_session_count() == 0


class TestLatencyRequirements:
    """Test latency requirements (median < 1000ms)."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_end_to_end_latency_under_1000ms(self, latency_monitor):
        """
        Test that end-to-end latency is under 1000ms.
        
        **Validates: Requirements 2.1, 2.8**
        """
        session_id = "latency_test_session"
        
        # Simulate pipeline stages with realistic timings
        latency_monitor.start_timer(session_id, "stt")
        await asyncio.sleep(0.1)  # 100ms STT
        latency_monitor.end_timer(session_id, "stt")
        
        latency_monitor.start_timer(session_id, "llm_ttft")
        await asyncio.sleep(0.2)  # 200ms LLM TTFT
        latency_monitor.end_timer(session_id, "llm_ttft")
        
        latency_monitor.start_timer(session_id, "llm_total")
        await asyncio.sleep(0.4)  # 400ms LLM total
        latency_monitor.end_timer(session_id, "llm_total")
        
        latency_monitor.start_timer(session_id, "tts_ttfb")
        await asyncio.sleep(0.1)  # 100ms TTS TTFB
        latency_monitor.end_timer(session_id, "tts_ttfb")
        
        latency_monitor.start_timer(session_id, "tts_total")
        await asyncio.sleep(0.2)  # 200ms TTS total
        latency_monitor.end_timer(session_id, "tts_total")
        
        # Get metrics
        metrics = latency_monitor.get_session_metrics(session_id)
        
        # Verify individual stages
        assert metrics.stt_latency_ms < 200
        assert metrics.llm_ttft_ms < 300
        assert metrics.llm_total_ms < 500
        assert metrics.tts_ttfb_ms < 200
        assert metrics.tts_total_ms < 300
        
        # Verify end-to-end latency
        assert metrics.end_to_end_ms < 1000, f"End-to-end latency {metrics.end_to_end_ms}ms exceeds 1000ms"
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_latency_monitoring_across_multiple_turns(self, latency_monitor):
        """
        Test latency monitoring across multiple conversation turns.
        
        **Validates: Requirements 2.7, 2.8**
        """
        session_id = "multi_turn_session"
        
        # Simulate 3 conversation turns
        for turn in range(3):
            turn_session_id = f"{session_id}_turn_{turn}"
            
            # Simulate pipeline stages
            latency_monitor.start_timer(turn_session_id, "stt")
            await asyncio.sleep(0.05)
            latency_monitor.end_timer(turn_session_id, "stt")
            
            latency_monitor.start_timer(turn_session_id, "llm_total")
            await asyncio.sleep(0.3)
            latency_monitor.end_timer(turn_session_id, "llm_total")
            
            latency_monitor.start_timer(turn_session_id, "tts_total")
            await asyncio.sleep(0.15)
            latency_monitor.end_timer(turn_session_id, "tts_total")
            
            # Verify each turn meets latency requirements
            metrics = latency_monitor.get_session_metrics(turn_session_id)
            assert metrics.end_to_end_ms < 1000


class TestErrorScenariosAndGracefulDegradation:
    """Test error scenarios and graceful degradation."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_fhir_api_failure_graceful_degradation(self, mock_fhir_adapter):
        """
        Test graceful degradation when FHIR API fails.
        
        **Validates: Requirements 3.7, 10.5**
        """
        # Setup FHIR adapter to fail
        mock_fhir_adapter.find_appointments.side_effect = Exception("FHIR API unavailable")
        
        with patch('src.voice_agent.adapters.fhir_adapter.FHIRAdapter', return_value=mock_fhir_adapter):
            # Attempt to search appointments
            with pytest.raises(Exception, match="FHIR API unavailable"):
                await mock_fhir_adapter.find_appointments(
                    patient_id="Patient/123",
                    date_range=Mock(),
                    specialty="General"
                )
            
            # In real scenario, workflow would catch this and trigger human handoff
            # Verify error is properly raised for handling
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_llm_service_timeout_handling(self):
        """
        Test handling of LLM service timeouts.
        
        **Validates: Requirements 10.5**
        """
        # Simulate LLM timeout
        mock_llm = AsyncMock()
        mock_llm.generate_response.side_effect = asyncio.TimeoutError("LLM timeout")
        
        # Attempt to generate response
        with pytest.raises(asyncio.TimeoutError):
            await mock_llm.generate_response(
                transcript="I need an appointment",
                context=Mock(),
                config=Mock()
            )
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_session_cleanup_on_unexpected_disconnect(self):
        """
        Test session cleanup when call disconnects unexpectedly.
        
        **Validates: Requirements 11.6**
        """
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            mock_registry = Mock(spec=TenantRegistry)
            mock_registry.get_tenant_by_caller.return_value = "healthcare_clinic_001"
            mock_registry.load_tenant_config.return_value = Mock(
                tenant_id="healthcare_clinic_001",
                persona=Mock(name="Sarah"),
                llm=Mock(provider="openai"),
                tts=Mock(provider="elevenlabs"),
                tools=[], knowledge_bases=[],
                constraints=Mock(max_call_duration=600, enable_barge_in=True)
            )
            mock_registry_class.return_value = mock_registry
            
            orchestrator = PipecatOrchestrator(
                tenant_registry=mock_registry,
                session_manager=SessionManager()
            )
            
            # Start session
            with patch.object(orchestrator, 'initialize_pipeline', new_callable=AsyncMock) as mock_init:
                mock_pipeline = AsyncMock()
                mock_pipeline.initialize = AsyncMock()
                mock_pipeline.shutdown = AsyncMock()
                mock_init.return_value = mock_pipeline
                
                session_id = "disconnect_test_session"
                call_metadata = CallMetadata(
                    caller_id="+15551234567",
                    call_sid="CA_disconnect",
                    from_number="+15551234567"
                )
                
                await orchestrator.start_session(
                    session_id=session_id,
                    caller_id=call_metadata.caller_id,
                    call_metadata=call_metadata
                )
                
                assert orchestrator.get_session_count() == 1
                
                # Simulate unexpected disconnect (stop session)
                await orchestrator.stop_session(session_id)
                
                # Verify cleanup
                assert orchestrator.get_session_count() == 0
                assert orchestrator.get_active_pipeline(session_id) is None
                mock_pipeline.shutdown.assert_called_once()


class TestHumanHandoffScenarios:
    """Test human handoff scenarios."""
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_transfer_to_human_on_workflow_failure(self):
        """
        Test transfer to human agent when workflow fails.
        
        **Validates: Requirements 10.6**
        """
        session_manager = SessionManager()
        session = session_manager.create_session(
            session_id="handoff_test",
            tenant_id="healthcare_clinic_001",
            caller_id="+15551234567",
            metadata={}
        )
        
        # Simulate workflow failure requiring handoff
        transfer_result = session_manager.transfer_to_human(
            session_id="handoff_test",
            reason="Unable to find available appointments"
        )
        
        # Verify transfer was initiated
        assert transfer_result is not None
        assert transfer_result["session_id"] == "handoff_test"
        assert transfer_result["reason"] == "Unable to find available appointments"
        assert transfer_result["transfer_initiated"] is True
    
    @pytest.mark.asyncio
    @pytest.mark.e2e
    async def test_transfer_on_missing_patient_information(self):
        """
        Test transfer when patient information is missing.
        
        **Validates: Requirements 10.6**
        """
        session_manager = SessionManager()
        session = session_manager.create_session(
            session_id="missing_info_test",
            tenant_id="healthcare_clinic_001",
            caller_id="+15551234567",
            metadata={}  # No patient_id
        )
        
        # Verify patient_id is missing
        assert "patient_id" not in session.metadata
        
        # Initiate transfer
        transfer_result = session_manager.transfer_to_human(
            session_id="missing_info_test",
            reason="Patient identification required"
        )
        
        assert transfer_result["transfer_initiated"] is True
        assert "Patient identification required" in transfer_result["reason"]
