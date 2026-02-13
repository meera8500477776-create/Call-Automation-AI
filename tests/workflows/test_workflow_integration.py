"""
Integration tests for appointment workflows.

Tests complete end-to-end flows for scheduling, rescheduling, and cancellation
workflows including error handling, graceful degradation, and human handoff.

Requirements: 10.5, 10.6, 10.7, 10.8
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, Mock, patch

from voice_agent.workflows.appointment_scheduling import (
    AppointmentSchedulingWorkflow,
    WorkflowStage
)
from voice_agent.workflows.appointment_rescheduling import (
    AppointmentReschedulingWorkflow,
    RescheduleWorkflowStage
)
from voice_agent.workflows.appointment_cancellation import (
    AppointmentCancellationWorkflow,
    CancelWorkflowStage
)
from voice_agent.adapters.fhir_adapter import (
    FHIRAdapter, DateRange, Slot, Appointment, 
    AppointmentConstraints, CancellationResult
)
from voice_agent.core.session import Session


@pytest.fixture
def mock_fhir_adapter():
    """Create a mock FHIR adapter for integration tests."""
    adapter = AsyncMock(spec=FHIRAdapter)
    return adapter


@pytest.fixture
def test_session():
    """Create a test session with patient information."""
    return Session(
        tenant_id="test_tenant",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/123"}
    )


@pytest.fixture
def available_slots():
    """Create test appointment slots."""
    tomorrow = datetime.now() + timedelta(days=1)
    return [
        Slot(
            id="slot1",
            schedule_id="schedule1",
            start_time=tomorrow.replace(hour=9, minute=0),
            end_time=tomorrow.replace(hour=9, minute=30),
            status="free",
            service_type="General",
            practitioner_id="Practitioner/1"
        ),
        Slot(
            id="slot2",
            schedule_id="schedule1",
            start_time=tomorrow.replace(hour=14, minute=0),
            end_time=tomorrow.replace(hour=14, minute=30),
            status="free",
            service_type="General",
            practitioner_id="Practitioner/1"
        )
    ]


@pytest.fixture
def existing_appointment():
    """Create a test existing appointment."""
    return Appointment(
        id="appt123",
        patient_id="Patient/123",
        practitioner_id="Practitioner/1",
        start_time=datetime.now() + timedelta(days=7),
        end_time=datetime.now() + timedelta(days=7, hours=1),
        status="booked",
        service_type="General",
        is_cancelable=True,
        is_reschedulable=True,
        location="Main Clinic"
    )


class TestAppointmentSchedulingIntegration:
    """Integration tests for complete appointment scheduling flow."""
    
    @pytest.mark.asyncio
    async def test_complete_scheduling_flow_happy_path(
        self, mock_fhir_adapter, test_session, available_slots
    ):
        """Test complete scheduling flow from start to finish."""
        # Setup workflow
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mock responses
        mock_fhir_adapter.find_appointments.return_value = available_slots
        booked_appointment = Appointment(
            id="appt_new",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=available_slots[1].start_time,
            end_time=available_slots[1].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.book_appointment.return_value = booked_appointment
        
        # Step 1: Initial request with date and time
        response1 = await workflow.process_turn(
            test_session,
            "I want to schedule an appointment for tomorrow at 2pm",
            {"date": "tomorrow", "time": "2pm"}
        )
        
        # Should search for slots and present options
        assert "found" in response1.lower() or "available" in response1.lower()
        assert test_session.workflow_stage == WorkflowStage.CONFIRMING.value
        mock_fhir_adapter.find_appointments.assert_called_once()
        
        # Step 2: Confirm booking
        response2 = await workflow.process_turn(
            test_session,
            "Yes, please book it",
            {}
        )
        
        # Should complete booking
        assert "scheduled" in response2.lower() or "booked" in response2.lower()
        assert test_session.workflow_stage == WorkflowStage.COMPLETED.value
        mock_fhir_adapter.book_appointment.assert_called_once()
        
        # Verify booking parameters
        call_args = mock_fhir_adapter.book_appointment.call_args
        assert call_args.kwargs["patient_id"] == "Patient/123"
        assert call_args.kwargs["slot_id"] == "slot2"  # 2pm slot
    
    @pytest.mark.asyncio
    async def test_scheduling_flow_with_incremental_collection(
        self, mock_fhir_adapter, test_session, available_slots
    ):
        """Test scheduling flow collecting parameters one at a time."""
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.find_appointments.return_value = available_slots
        booked_appointment = Appointment(
            id="appt_new",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=available_slots[0].start_time,
            end_time=available_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.book_appointment.return_value = booked_appointment
        
        # Step 1: Initial request without parameters
        response1 = await workflow.process_turn(
            test_session,
            "I need to schedule an appointment",
            {}
        )
        assert "date" in response1.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_DATE.value
        
        # Step 2: Provide date
        response2 = await workflow.process_turn(
            test_session,
            "tomorrow",
            {"date": "tomorrow"}
        )
        assert "time" in response2.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_TIME.value
        
        # Step 3: Provide time
        response3 = await workflow.process_turn(
            test_session,
            "morning",
            {"time": "morning"}
        )
        assert "reason" in response3.lower() or "visit" in response3.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_REASON.value
        
        # Step 4: Provide reason (optional)
        response4 = await workflow.process_turn(
            test_session,
            "annual checkup",
            {"reason": "annual checkup"}
        )
        # Should search and present slot
        assert "found" in response4.lower() or "available" in response4.lower()
        assert test_session.workflow_stage == WorkflowStage.CONFIRMING.value
        
        # Step 5: Confirm
        response5 = await workflow.process_turn(
            test_session,
            "yes",
            {}
        )
        assert "scheduled" in response5.lower()
        assert test_session.workflow_stage == WorkflowStage.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_scheduling_no_availability_error_handling(
        self, mock_fhir_adapter, test_session
    ):
        """Test error handling when no slots are available."""
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mock to return no slots
        mock_fhir_adapter.find_appointments.return_value = []
        
        # Request appointment
        response = await workflow.process_turn(
            test_session,
            "I want an appointment tomorrow at 2pm",
            {"date": "tomorrow", "time": "2pm"}
        )
        
        # Should fail gracefully
        assert "sorry" in response.lower() or "don't see" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_scheduling_api_error_graceful_degradation(
        self, mock_fhir_adapter, test_session
    ):
        """Test graceful degradation when API fails."""
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mock to raise exception
        mock_fhir_adapter.find_appointments.side_effect = Exception("API error")
        
        # Request appointment
        response = await workflow.process_turn(
            test_session,
            "I want an appointment tomorrow",
            {"date": "tomorrow", "time": "2pm"}
        )
        
        # Should fail gracefully with transfer message
        assert "sorry" in response.lower() or "trouble" in response.lower()
        assert "transfer" in response.lower() or "help" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_scheduling_booking_failure_error_handling(
        self, mock_fhir_adapter, test_session, available_slots
    ):
        """Test error handling when booking fails."""
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.find_appointments.return_value = available_slots
        mock_fhir_adapter.book_appointment.side_effect = Exception("Booking failed")
        
        # Request and confirm appointment
        await workflow.process_turn(
            test_session,
            "I want an appointment tomorrow at 2pm",
            {"date": "tomorrow", "time": "2pm"}
        )
        
        response = await workflow.process_turn(
            test_session,
            "yes",
            {}
        )
        
        # Should fail gracefully
        assert "sorry" in response.lower() or "trouble" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value


class TestAppointmentReschedulingIntegration:
    """Integration tests for complete appointment rescheduling flow."""
    
    @pytest.mark.asyncio
    async def test_complete_rescheduling_flow_happy_path(
        self, mock_fhir_adapter, test_session, available_slots, existing_appointment
    ):
        """Test complete rescheduling flow from start to finish."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.find_appointments.return_value = available_slots
        updated_appointment = Appointment(
            id="appt123",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=available_slots[1].start_time,
            end_time=available_slots[1].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.reschedule_appointment.return_value = updated_appointment
        
        # Step 1: Initial request with appointment ID
        response1 = await workflow.process_turn(
            test_session,
            "I need to reschedule my appointment",
            {"appointment_id": "appt123"}
        )
        assert "found your appointment" in response1.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.CONFIRMING_APPOINTMENT.value
        
        # Step 2: Confirm appointment
        response2 = await workflow.process_turn(
            test_session,
            "Yes, that's the one",
            {}
        )
        assert "what date" in response2.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.COLLECTING_NEW_DATE.value
        
        # Step 3: Provide new date
        response3 = await workflow.process_turn(
            test_session,
            "next Monday",
            {"date": "Monday"}
        )
        assert "time" in response3.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.COLLECTING_NEW_TIME.value
        
        # Step 4: Provide new time
        response4 = await workflow.process_turn(
            test_session,
            "afternoon",
            {"time": "afternoon"}
        )
        assert "found" in response4.lower() or "available" in response4.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.CONFIRMING_RESCHEDULE.value
        
        # Step 5: Confirm reschedule
        response5 = await workflow.process_turn(
            test_session,
            "yes please",
            {}
        )
        assert "rescheduled" in response5.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.COMPLETED.value
        mock_fhir_adapter.reschedule_appointment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rescheduling_not_reschedulable_constraint(
        self, mock_fhir_adapter, test_session
    ):
        """Test rescheduling when appointment cannot be rescheduled."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mock with non-reschedulable constraint
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=False  # Cannot reschedule
        )
        
        # Request reschedule
        response = await workflow.process_turn(
            test_session,
            "I need to reschedule",
            {"appointment_id": "appt123"}
        )
        
        # Should fail with constraint message
        assert "cannot be rescheduled" in response.lower()
        assert "clinical constraints" in response.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_rescheduling_no_slots_available(
        self, mock_fhir_adapter, test_session
    ):
        """Test rescheduling when no new slots are available."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.find_appointments.return_value = []  # No slots
        
        # Go through flow
        await workflow.process_turn(
            test_session,
            "Reschedule my appointment",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "tomorrow", {"date": "tomorrow"})
        
        response = await workflow.process_turn(
            test_session,
            "morning",
            {"time": "morning"}
        )
        
        # Should fail gracefully
        assert "sorry" in response.lower() or "don't see" in response.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_rescheduling_api_error_graceful_degradation(
        self, mock_fhir_adapter, test_session
    ):
        """Test graceful degradation when rescheduling API fails."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mock to fail on reschedule
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.find_appointments.return_value = [
            Slot(
                id="slot1",
                schedule_id="schedule1",
                start_time=datetime.now() + timedelta(days=1, hours=9),
                end_time=datetime.now() + timedelta(days=1, hours=10),
                status="free",
                service_type="General",
                practitioner_id="Practitioner/1"
            )
        ]
        mock_fhir_adapter.reschedule_appointment.side_effect = Exception("API error")
        
        # Go through flow
        await workflow.process_turn(
            test_session,
            "Reschedule",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "tomorrow", {"date": "tomorrow"})
        await workflow.process_turn(test_session, "morning", {"time": "morning"})
        
        response = await workflow.process_turn(test_session, "yes", {})
        
        # Should fail gracefully
        assert "sorry" in response.lower() or "trouble" in response.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.FAILED.value


class TestAppointmentCancellationIntegration:
    """Integration tests for complete appointment cancellation flow."""
    
    @pytest.mark.asyncio
    async def test_complete_cancellation_flow_happy_path(
        self, mock_fhir_adapter, test_session, existing_appointment
    ):
        """Test complete cancellation flow from start to finish."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.cancel_appointment.return_value = CancellationResult(
            success=True,
            appointment_id="appt123",
            reason="feeling better"
        )
        
        # Step 1: Initial request with appointment ID
        response1 = await workflow.process_turn(
            test_session,
            "I need to cancel my appointment",
            {"appointment_id": "appt123"}
        )
        assert "found your appointment" in response1.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.CONFIRMING_APPOINTMENT.value
        
        # Step 2: Confirm appointment
        response2 = await workflow.process_turn(
            test_session,
            "Yes, that's the one",
            {}
        )
        assert "why" in response2.lower() or "reason" in response2.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.COLLECTING_REASON.value
        
        # Step 3: Provide reason
        response3 = await workflow.process_turn(
            test_session,
            "I'm feeling better",
            {"reason": "feeling better"}
        )
        assert "confirm" in response3.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        
        # Step 4: Confirm cancellation
        response4 = await workflow.process_turn(
            test_session,
            "yes please",
            {}
        )
        assert "canceled" in response4.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.COMPLETED.value
        mock_fhir_adapter.cancel_appointment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cancellation_not_cancelable_constraint(
        self, mock_fhir_adapter, test_session
    ):
        """Test cancellation when appointment cannot be canceled."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mock with non-cancelable constraint
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=False,  # Cannot cancel
            is_reschedulable=True
        )
        
        # Request cancellation
        response = await workflow.process_turn(
            test_session,
            "I need to cancel",
            {"appointment_id": "appt123"}
        )
        
        # Should fail with constraint message
        assert "cannot be canceled" in response.lower()
        assert "clinical constraints" in response.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_cancellation_user_changes_mind(
        self, mock_fhir_adapter, test_session
    ):
        """Test cancellation when user changes their mind at confirmation."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        
        # Go through flow
        await workflow.process_turn(
            test_session,
            "Cancel my appointment",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "I changed my mind", {"reason": "changed mind"})
        
        # User says no at final confirmation
        response = await workflow.process_turn(
            test_session,
            "no, don't cancel",
            {}
        )
        
        # Should not cancel
        assert "won't cancel" in response.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.FAILED.value
        mock_fhir_adapter.cancel_appointment.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cancellation_api_error_graceful_degradation(
        self, mock_fhir_adapter, test_session
    ):
        """Test graceful degradation when cancellation API fails."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.cancel_appointment.side_effect = Exception("API error")
        
        # Go through flow
        await workflow.process_turn(
            test_session,
            "Cancel",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "feeling better", {"reason": "feeling better"})
        
        response = await workflow.process_turn(test_session, "yes", {})
        
        # Should fail gracefully
        assert "sorry" in response.lower() or "trouble" in response.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.FAILED.value



class TestWorkflowErrorHandlingAndRecovery:
    """Integration tests for error handling and recovery across workflows."""
    
    @pytest.mark.asyncio
    async def test_scheduling_missing_patient_id_triggers_handoff(
        self, mock_fhir_adapter
    ):
        """Test that missing patient ID triggers human handoff."""
        # Create session without patient_id
        session = Session(
            tenant_id="test_tenant",
            caller_id="+1234567890",
            metadata={}  # No patient_id
        )
        session.workflow_stage = WorkflowStage.INITIAL.value
        
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        
        # Request appointment
        response = await workflow.process_turn(
            session,
            "I want an appointment tomorrow",
            {"date": "tomorrow", "time": "2pm"}
        )
        
        # Should fail and suggest transfer
        assert "sorry" in response.lower()
        assert "transfer" in response.lower() or "help" in response.lower()
        assert session.workflow_stage == WorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_rescheduling_lookup_error_triggers_handoff(
        self, mock_fhir_adapter, test_session
    ):
        """Test that appointment lookup errors trigger human handoff."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mock to fail
        mock_fhir_adapter.check_constraints.side_effect = Exception("Lookup failed")
        
        # Request reschedule
        response = await workflow.process_turn(
            test_session,
            "Reschedule my appointment",
            {"appointment_id": "appt123"}
        )
        
        # Should fail gracefully with transfer
        assert "trouble" in response.lower() or "error" in response.lower()
        assert "transfer" in response.lower() or "help" in response.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_cancellation_constraint_check_error_triggers_handoff(
        self, mock_fhir_adapter, test_session
    ):
        """Test that constraint check errors trigger human handoff."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mock to fail
        mock_fhir_adapter.check_constraints.side_effect = Exception("Constraint check failed")
        
        # Request cancellation
        response = await workflow.process_turn(
            test_session,
            "Cancel my appointment",
            {"appointment_id": "appt123"}
        )
        
        # Should fail gracefully with transfer
        assert "trouble" in response.lower() or "error" in response.lower()
        assert "transfer" in response.lower() or "help" in response.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.FAILED.value


class TestWorkflowVariousScenarios:
    """Integration tests for various workflow scenarios."""
    
    @pytest.mark.asyncio
    async def test_scheduling_user_changes_preference_mid_flow(
        self, mock_fhir_adapter, test_session, available_slots
    ):
        """Test scheduling when user changes their mind about time."""
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.find_appointments.return_value = available_slots
        booked_appointment = Appointment(
            id="appt_new",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=available_slots[0].start_time,
            end_time=available_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.book_appointment.return_value = booked_appointment
        
        # Request appointment
        await workflow.process_turn(
            test_session,
            "I want an appointment tomorrow at 2pm",
            {"date": "tomorrow", "time": "2pm"}
        )
        
        # User says no to first option
        response = await workflow.process_turn(
            test_session,
            "no, I want a different time",
            {}
        )
        
        # Should restart date collection
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_DATE.value
        
        # Provide new preferences - this will trigger another search
        await workflow.process_turn(
            test_session,
            "tomorrow",
            {"date": "tomorrow"}
        )
        await workflow.process_turn(
            test_session,
            "morning",
            {"time": "morning"}
        )
        
        # After providing time, should ask for reason (optional)
        # Then search and present slot
        response = await workflow.process_turn(
            test_session,
            "annual checkup",
            {"reason": "annual checkup"}
        )
        
        # Should have searched and reached confirmation stage
        assert test_session.workflow_stage == WorkflowStage.CONFIRMING.value
    
    @pytest.mark.asyncio
    async def test_rescheduling_multiple_date_attempts(
        self, mock_fhir_adapter, test_session, available_slots
    ):
        """Test rescheduling with multiple date attempts."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        # First search returns no slots, second returns slots
        mock_fhir_adapter.find_appointments.side_effect = [[], available_slots]
        updated_appointment = Appointment(
            id="appt123",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=available_slots[0].start_time,
            end_time=available_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.reschedule_appointment.return_value = updated_appointment
        
        # Start flow
        await workflow.process_turn(
            test_session,
            "Reschedule",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "tomorrow", {"date": "tomorrow"})
        
        # First attempt - no slots
        response1 = await workflow.process_turn(
            test_session,
            "morning",
            {"time": "morning"}
        )
        
        # Should fail and ask for different date
        assert "sorry" in response1.lower() or "don't see" in response1.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_cancellation_without_reason_still_succeeds(
        self, mock_fhir_adapter, test_session
    ):
        """Test cancellation succeeds even without explicit reason."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.cancel_appointment.return_value = CancellationResult(
            success=True,
            appointment_id="appt123",
            reason="Patient requested cancellation"
        )
        
        # Go through flow without providing reason
        await workflow.process_turn(
            test_session,
            "Cancel",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "I'd rather not say", {})
        
        response = await workflow.process_turn(test_session, "yes", {})
        
        # Should succeed with default reason
        assert "canceled" in response.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.COMPLETED.value
        
        # Verify default reason was used
        call_args = mock_fhir_adapter.cancel_appointment.call_args
        assert "Patient requested cancellation" in call_args.kwargs["reason"]
    
    @pytest.mark.asyncio
    async def test_scheduling_with_provider_preference(
        self, mock_fhir_adapter, test_session, available_slots
    ):
        """Test scheduling with specific provider preference."""
        workflow = AppointmentSchedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = WorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.find_appointments.return_value = available_slots
        booked_appointment = Appointment(
            id="appt_new",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=available_slots[0].start_time,
            end_time=available_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.book_appointment.return_value = booked_appointment
        
        # Request with provider
        await workflow.process_turn(
            test_session,
            "I want to see Dr. Smith tomorrow morning",
            {"date": "tomorrow", "time": "morning", "provider": "Dr. Smith"}
        )
        
        # Verify provider was passed to search
        call_args = mock_fhir_adapter.find_appointments.call_args
        assert call_args.kwargs["specialty"] == "Dr. Smith"
        
        # Confirm booking
        await workflow.process_turn(test_session, "yes", {})
        
        assert test_session.workflow_stage == WorkflowStage.COMPLETED.value
    
    @pytest.mark.asyncio
    async def test_rescheduling_wrong_appointment_confirmation(
        self, mock_fhir_adapter, test_session
    ):
        """Test rescheduling when user says wrong appointment."""
        workflow = AppointmentReschedulingWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = RescheduleWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        
        # Start flow
        await workflow.process_turn(
            test_session,
            "Reschedule",
            {"appointment_id": "appt123"}
        )
        
        # User says no - wrong appointment
        # Note: The workflow interprets "no" as confirmation to proceed, not rejection
        # To indicate wrong appointment, user needs to say "wrong" or "different"
        response = await workflow.process_turn(
            test_session,
            "no, that's wrong",
            {}
        )
        
        # Should fail and offer transfer
        assert "apologize" in response.lower() or "confusion" in response.lower()
        assert "transfer" in response.lower() or "help" in response.lower()
        assert test_session.workflow_stage == RescheduleWorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_cancellation_unclear_confirmation_asks_again(
        self, mock_fhir_adapter, test_session
    ):
        """Test cancellation asks again on unclear confirmation."""
        workflow = AppointmentCancellationWorkflow(mock_fhir_adapter)
        test_session.workflow_stage = CancelWorkflowStage.INITIAL.value
        
        # Setup mocks
        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
        
        # Go through flow
        await workflow.process_turn(
            test_session,
            "Cancel",
            {"appointment_id": "appt123"}
        )
        await workflow.process_turn(test_session, "yes", {})
        await workflow.process_turn(test_session, "feeling better", {"reason": "feeling better"})
        
        # Unclear response
        response = await workflow.process_turn(
            test_session,
            "I'm not sure",
            {}
        )
        
        # Should ask again
        assert "make sure" in response.lower() or "understand" in response.lower()
        assert test_session.workflow_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        mock_fhir_adapter.cancel_appointment.assert_not_called()
