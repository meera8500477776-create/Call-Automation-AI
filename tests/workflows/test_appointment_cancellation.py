"""
Unit tests for appointment cancellation workflow.

Tests the multi-turn conversation flow for canceling appointments,
including appointment lookup, constraint checking, and FHIR integration.

Requirements: 10.7, 3.5
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from voice_agent.adapters.fhir_adapter import (
    Appointment,
    AppointmentConstraints,
    CancellationResult,
    FHIRAdapter,
)
from voice_agent.core.session import Session
from voice_agent.workflows.appointment_cancellation import (
    AppointmentCancellationWorkflow,
    CancelWorkflowParameters,
    CancelWorkflowStage,
)


@pytest.fixture
def mock_fhir_adapter():
    """Create a mock FHIR adapter."""
    adapter = AsyncMock(spec=FHIRAdapter)
    return adapter


@pytest.fixture
def workflow(mock_fhir_adapter):
    """Create a cancellation workflow instance."""
    return AppointmentCancellationWorkflow(mock_fhir_adapter)


@pytest.fixture
def session():
    """Create a test session."""
    return Session(
        session_id="test_session",
        tenant_id="test_tenant",
        caller_id="+1234567890",
        metadata={"patient_id": "patient123"},
        workflow_stage=CancelWorkflowStage.INITIAL.value
    )


@pytest.fixture
def existing_appointment():
    """Create a test existing appointment."""
    return Appointment(
        id="appt123",
        patient_id="patient123",
        practitioner_id="dr_smith",
        start_time=datetime.now() + timedelta(days=7),
        end_time=datetime.now() + timedelta(days=7, hours=1),
        status="booked",
        service_type="General",
        is_cancelable=True,
        is_reschedulable=True,
        location="Main Clinic"
    )


class TestCancelWorkflowParameters:
    """Test CancelWorkflowParameters data model."""

    def test_initialization(self):
        """Test parameter initialization."""
        params = CancelWorkflowParameters(patient_id="patient123")
        assert params.patient_id == "patient123"
        assert params.existing_appointment_id is None
        assert params.cancellation_reason is None
        assert params.existing_appointment is None
        assert params.constraints is None


class TestAppointmentCancellationWorkflow:
    """Test AppointmentCancellationWorkflow class."""

    @pytest.mark.asyncio
    async def test_initial_stage_without_patient_id(self, workflow):
        """Test initial stage fails without patient ID."""
        session = Session(
            session_id="test",
            tenant_id="test",
            caller_id="+1234567890",
            metadata={},  # No patient_id
            workflow_stage=CancelWorkflowStage.INITIAL.value
        )

        response = await workflow.process_turn(session, "", {})

        assert "patient information" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_lookup_appointment_with_id(
        self, workflow, mock_fhir_adapter, session
    ):
        """Test looking up appointment with specific ID."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "found your appointment" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_APPOINTMENT.value
        mock_fhir_adapter.check_constraints.assert_called_once_with("appt123")

    @pytest.mark.asyncio
    async def test_lookup_appointment_not_cancelable(
        self, workflow, mock_fhir_adapter, session
    ):
        """Test looking up appointment that cannot be canceled."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=False,  # Cannot cancel
            is_reschedulable=True
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "cannot be canceled" in response.lower()
        assert "clinical constraints" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_lookup_appointment_without_id(self, workflow, session):
        """Test looking up appointment without specific ID."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123"
            # No appointment_id
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "need more information" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_confirming_appointment_yes(self, workflow, session):
        """Test confirming appointment with yes response."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "Yes, that's the one", {})

        # Verify
        assert "why" in response.lower() or "reason" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.COLLECTING_REASON.value

    @pytest.mark.asyncio
    async def test_confirming_appointment_no(self, workflow, session):
        """Test confirming appointment with no response."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "No, that's not it", {})

        # Verify
        assert "transfer" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_confirming_appointment_unclear(self, workflow, session):
        """Test confirming appointment with unclear response."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "Maybe", {})

        # Verify
        assert "make sure" in response.lower() or "right appointment" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_APPOINTMENT.value

    @pytest.mark.asyncio
    async def test_collecting_reason_with_reason(self, workflow, session):
        """Test collecting reason when reason is provided."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.COLLECTING_REASON.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute with reason parameter
        response = await workflow.process_turn(
            session, "I'm feeling better", {"reason": "feeling better"}
        )

        # Verify
        assert "confirm" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        assert session.metadata["workflow_params"].cancellation_reason == "feeling better"

    @pytest.mark.asyncio
    async def test_collecting_reason_without_reason(self, workflow, session):
        """Test collecting reason when no reason is provided."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.COLLECTING_REASON.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute without reason parameter
        response = await workflow.process_turn(session, "I'd rather not say", {})

        # Verify - should still proceed to confirmation
        assert "confirm" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION.value

    @pytest.mark.asyncio
    async def test_confirming_cancellation_yes(
        self, workflow, mock_fhir_adapter, session, existing_appointment
    ):
        """Test confirming cancellation with yes response."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            existing_appointment=existing_appointment,
            cancellation_reason="feeling better"
        )

        mock_fhir_adapter.cancel_appointment.return_value = CancellationResult(
            success=True,
            appointment_id="appt123",
            reason="feeling better"
        )

        # Execute
        response = await workflow.process_turn(session, "Yes, please cancel it", {})

        # Verify
        assert "canceled" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.COMPLETED.value
        mock_fhir_adapter.cancel_appointment.assert_called_once_with(
            appointment_id="appt123",
            reason="feeling better"
        )

    @pytest.mark.asyncio
    async def test_confirming_cancellation_no(self, workflow, session, mock_fhir_adapter):
        """Test confirming cancellation with no response."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "No, don't cancel", {})

        # Verify
        assert "won't cancel" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value
        # Ensure cancel_appointment was NOT called
        mock_fhir_adapter.cancel_appointment.assert_not_called()

    @pytest.mark.asyncio
    async def test_confirming_cancellation_unclear(self, workflow, session, mock_fhir_adapter):
        """Test confirming cancellation with unclear response."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "I'm not sure", {})

        # Verify
        assert "make sure" in response.lower() or "understand" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        # Ensure cancel_appointment was NOT called
        mock_fhir_adapter.cancel_appointment.assert_not_called()

    @pytest.mark.asyncio
    async def test_canceling_success(
        self, workflow, mock_fhir_adapter, session, existing_appointment
    ):
        """Test successful appointment cancellation."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CANCELING.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            existing_appointment=existing_appointment,
            cancellation_reason="feeling better"
        )

        mock_fhir_adapter.cancel_appointment.return_value = CancellationResult(
            success=True,
            appointment_id="appt123",
            reason="feeling better"
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "canceled" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.COMPLETED.value

        # Verify cancellation was called with correct parameters
        call_args = mock_fhir_adapter.cancel_appointment.call_args
        assert call_args.kwargs["appointment_id"] == "appt123"
        assert call_args.kwargs["reason"] == "feeling better"

    @pytest.mark.asyncio
    async def test_canceling_without_reason(
        self, workflow, mock_fhir_adapter, session, existing_appointment
    ):
        """Test cancellation without explicit reason uses default."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CANCELING.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            existing_appointment=existing_appointment
            # No cancellation_reason
        )

        mock_fhir_adapter.cancel_appointment.return_value = CancellationResult(
            success=True,
            appointment_id="appt123",
            reason="Patient requested cancellation"
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "canceled" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.COMPLETED.value

        # Verify default reason was used
        call_args = mock_fhir_adapter.cancel_appointment.call_args
        assert call_args.kwargs["reason"] == "Patient requested cancellation"

    @pytest.mark.asyncio
    async def test_canceling_without_appointment_id(self, workflow, session):
        """Test canceling without appointment ID fails gracefully."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CANCELING.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123"
            # No existing_appointment_id
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "lost track" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_canceling_constraint_violation(
        self, workflow, mock_fhir_adapter, session
    ):
        """Test canceling with constraint violation."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CANCELING.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.cancel_appointment.side_effect = ValueError(
            "Appointment cannot be canceled"
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "couldn't cancel" in response.lower()
        assert "cannot be canceled" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_canceling_api_error(
        self, workflow, mock_fhir_adapter, session
    ):
        """Test canceling handles API errors gracefully."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.CANCELING.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.cancel_appointment.side_effect = Exception("Network error")

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "trouble" in response.lower() or "error" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_error_handling_in_lookup(self, workflow, mock_fhir_adapter, session):
        """Test error handling during appointment lookup."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.check_constraints.side_effect = Exception("Network error")

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "trouble" in response.lower() or "error" in response.lower()
        assert session.workflow_stage == CancelWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_completed_stage(self, workflow, session):
        """Test completed stage returns appropriate message."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.COMPLETED.value

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "canceled" in response.lower()
        assert "anything else" in response.lower()

    @pytest.mark.asyncio
    async def test_failed_stage(self, workflow, session):
        """Test failed stage returns appropriate message."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.FAILED.value

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "wasn't able" in response.lower() or "sorry" in response.lower()

    @pytest.mark.asyncio
    async def test_parameter_extraction_appointment_id(self, workflow, session):
        """Test that appointment_id is extracted from parameters."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.INITIAL.value

        # Execute with appointment_id in extracted params
        await workflow.process_turn(
            session,
            "I want to cancel appointment appt456",
            {"appointment_id": "appt456"}
        )

        # Verify
        workflow_params = session.metadata["workflow_params"]
        assert workflow_params.existing_appointment_id == "appt456"
        assert session.extracted_parameters["appointment_id"] == "appt456"

    @pytest.mark.asyncio
    async def test_parameter_extraction_reason(self, workflow, session):
        """Test that cancellation reason is extracted from parameters."""
        # Setup
        session.workflow_stage = CancelWorkflowStage.COLLECTING_REASON.value
        session.metadata["workflow_params"] = CancelWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute with reason in extracted params
        await workflow.process_turn(
            session,
            "I'm feeling better now",
            {"reason": "feeling better"}
        )

        # Verify
        workflow_params = session.metadata["workflow_params"]
        assert workflow_params.cancellation_reason == "feeling better"
        assert session.extracted_parameters["cancellation_reason"] == "feeling better"

    @pytest.mark.asyncio
    async def test_full_workflow_happy_path(
        self, workflow, mock_fhir_adapter, session, existing_appointment
    ):
        """Test complete workflow from start to finish."""
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

        # Step 1: Initial - should move to lookup
        response1 = await workflow.process_turn(
            session, "I want to cancel my appointment", {"appointment_id": "appt123"}
        )
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_APPOINTMENT.value

        # Step 2: Confirm appointment
        response2 = await workflow.process_turn(session, "Yes, that's the one", {})
        assert session.workflow_stage == CancelWorkflowStage.COLLECTING_REASON.value

        # Step 3: Provide reason
        response3 = await workflow.process_turn(
            session, "I'm feeling better", {"reason": "feeling better"}
        )
        assert session.workflow_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION.value

        # Step 4: Confirm cancellation
        response4 = await workflow.process_turn(session, "Yes, cancel it", {})
        assert session.workflow_stage == CancelWorkflowStage.COMPLETED.value
        assert "canceled" in response4.lower()
