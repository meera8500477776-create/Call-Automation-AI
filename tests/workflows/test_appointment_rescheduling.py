"""
Unit tests for appointment rescheduling workflow.

Tests the multi-turn conversation flow for rescheduling appointments,
including appointment lookup, constraint checking, and FHIR integration.
"""

from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from voice_agent.adapters.fhir_adapter import (
    Appointment,
    AppointmentConstraints,
    FHIRAdapter,
    Slot,
)
from voice_agent.core.session import Session
from voice_agent.workflows.appointment_rescheduling import (
    AppointmentReschedulingWorkflow,
    RescheduleWorkflowParameters,
    RescheduleWorkflowStage,
)


@pytest.fixture
def mock_fhir_adapter():
    """Create a mock FHIR adapter."""
    adapter = AsyncMock(spec=FHIRAdapter)
    return adapter


@pytest.fixture
def workflow(mock_fhir_adapter):
    """Create a rescheduling workflow instance."""
    return AppointmentReschedulingWorkflow(mock_fhir_adapter)


@pytest.fixture
def session():
    """Create a test session."""
    return Session(
        session_id="test_session",
        tenant_id="test_tenant",
        caller_id="+1234567890",
        metadata={"patient_id": "patient123"},
        workflow_stage=RescheduleWorkflowStage.INITIAL.value
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


@pytest.fixture
def available_slots():
    """Create test available slots."""
    base_time = datetime.now() + timedelta(days=14)
    return [
        Slot(
            id="slot1",
            schedule_id="schedule1",
            start_time=base_time.replace(hour=9, minute=0),
            end_time=base_time.replace(hour=10, minute=0),
            status="free",
            service_type="General",
            practitioner_id="dr_smith"
        ),
        Slot(
            id="slot2",
            schedule_id="schedule1",
            start_time=base_time.replace(hour=14, minute=0),
            end_time=base_time.replace(hour=15, minute=0),
            status="free",
            service_type="General",
            practitioner_id="dr_smith"
        )
    ]


class TestRescheduleWorkflowParameters:
    """Test RescheduleWorkflowParameters data model."""

    def test_initialization(self):
        """Test parameter initialization."""
        params = RescheduleWorkflowParameters(patient_id="patient123")
        assert params.patient_id == "patient123"
        assert params.existing_appointment_id is None
        assert params.new_date is None
        assert params.new_time is None
        assert params.available_slots == []

    def test_post_init_creates_empty_list(self):
        """Test that __post_init__ creates empty list for available_slots."""
        params = RescheduleWorkflowParameters()
        assert params.available_slots is not None
        assert isinstance(params.available_slots, list)


class TestAppointmentReschedulingWorkflow:
    """Test AppointmentReschedulingWorkflow class."""

    @pytest.mark.asyncio
    async def test_initial_stage_without_patient_id(self, workflow, mock_fhir_adapter):
        """Test initial stage fails without patient ID."""
        session = Session(
            session_id="test",
            tenant_id="test",
            caller_id="+1234567890",
            metadata={},  # No patient_id
            workflow_stage=RescheduleWorkflowStage.INITIAL.value
        )

        response = await workflow.process_turn(session, "", {})

        assert "patient information" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_lookup_appointment_with_id(
        self, workflow, mock_fhir_adapter, session, existing_appointment
    ):
        """Test looking up appointment with specific ID."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
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
        assert session.workflow_stage == RescheduleWorkflowStage.CONFIRMING_APPOINTMENT.value
        mock_fhir_adapter.check_constraints.assert_called_once_with("appt123")

    @pytest.mark.asyncio
    async def test_lookup_appointment_not_reschedulable(
        self, workflow, mock_fhir_adapter, session
    ):
        """Test looking up appointment that cannot be rescheduled."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.check_constraints.return_value = AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=False  # Cannot reschedule
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "cannot be rescheduled" in response.lower()
        assert "clinical constraints" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_confirming_appointment_yes(self, workflow, session):
        """Test confirming appointment with yes response."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.CONFIRMING_APPOINTMENT.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "Yes, that's the one", {})

        # Verify
        assert "what date" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.COLLECTING_NEW_DATE.value

    @pytest.mark.asyncio
    async def test_confirming_appointment_no(self, workflow, session):
        """Test confirming appointment with no response."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.CONFIRMING_APPOINTMENT.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute
        response = await workflow.process_turn(session, "No, that's not it", {})

        # Verify
        assert "transfer" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_collecting_new_date(self, workflow, session):
        """Test collecting new date."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.COLLECTING_NEW_DATE.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        # Execute with date parameter
        response = await workflow.process_turn(
            session, "Tomorrow", {"date": "tomorrow"}
        )

        # Verify
        assert "what time" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.COLLECTING_NEW_TIME.value
        assert session.metadata["workflow_params"].new_date == "tomorrow"

    @pytest.mark.asyncio
    async def test_collecting_new_time(self, workflow, mock_fhir_adapter, session, available_slots):
        """Test collecting new time."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.COLLECTING_NEW_TIME.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            new_date="tomorrow"
        )

        mock_fhir_adapter.find_appointments.return_value = available_slots

        # Execute with time parameter
        response = await workflow.process_turn(
            session, "2pm", {"time": "2pm"}
        )

        # Verify
        assert "found an available appointment" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.CONFIRMING_RESCHEDULE.value
        assert session.metadata["workflow_params"].new_time == "2pm"

    @pytest.mark.asyncio
    async def test_searching_slots_no_availability(
        self, workflow, mock_fhir_adapter, session
    ):
        """Test searching slots when none are available."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.SEARCHING_SLOTS.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            new_date="tomorrow",
            new_time="2pm"
        )

        mock_fhir_adapter.find_appointments.return_value = []  # No slots

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "don't see any available" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_confirming_reschedule_yes(
        self, workflow, mock_fhir_adapter, session, available_slots, existing_appointment
    ):
        """Test confirming reschedule with yes response."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.CONFIRMING_RESCHEDULE.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            new_date="tomorrow",
            new_time="2pm",
            selected_slot=available_slots[0]
        )

        updated_appointment = Appointment(
            id="appt123",
            patient_id="patient123",
            practitioner_id="dr_smith",
            start_time=available_slots[0].start_time,
            end_time=available_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True,
            location="Main Clinic"
        )
        mock_fhir_adapter.reschedule_appointment.return_value = updated_appointment

        # Execute
        response = await workflow.process_turn(session, "Yes, please", {})

        # Verify
        assert "rescheduled your appointment" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.COMPLETED.value
        mock_fhir_adapter.reschedule_appointment.assert_called_once_with(
            appointment_id="appt123",
            new_slot_id="slot1"
        )

    @pytest.mark.asyncio
    async def test_confirming_reschedule_no(self, workflow, session, available_slots):
        """Test confirming reschedule with no response."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.CONFIRMING_RESCHEDULE.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            new_date="tomorrow",
            new_time="2pm",
            selected_slot=available_slots[0]
        )

        # Execute
        response = await workflow.process_turn(session, "No, different time", {})

        # Verify
        assert "what date and time" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.COLLECTING_NEW_DATE.value
        # Parameters should be reset
        assert session.metadata["workflow_params"].new_date is None
        assert session.metadata["workflow_params"].new_time is None
        assert session.metadata["workflow_params"].selected_slot is None

    @pytest.mark.asyncio
    async def test_rescheduling_constraint_violation(
        self, workflow, mock_fhir_adapter, session, available_slots
    ):
        """Test rescheduling with constraint violation."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.RESCHEDULING.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123",
            selected_slot=available_slots[0]
        )

        mock_fhir_adapter.reschedule_appointment.side_effect = ValueError(
            "Appointment cannot be rescheduled"
        )

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "couldn't reschedule" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.FAILED.value

    @pytest.mark.asyncio
    async def test_error_handling(self, workflow, mock_fhir_adapter, session):
        """Test error handling in workflow."""
        # Setup
        session.workflow_stage = RescheduleWorkflowStage.LOOKING_UP_APPOINTMENT.value
        session.metadata["workflow_params"] = RescheduleWorkflowParameters(
            patient_id="patient123",
            existing_appointment_id="appt123"
        )

        mock_fhir_adapter.check_constraints.side_effect = Exception("Network error")

        # Execute
        response = await workflow.process_turn(session, "", {})

        # Verify
        assert "trouble" in response.lower() or "error" in response.lower()
        assert session.workflow_stage == RescheduleWorkflowStage.FAILED.value

    def test_parse_date_to_range_today(self, workflow):
        """Test parsing 'today' to date range."""
        date_range = workflow._parse_date_to_range("today")
        assert date_range.start == date.today()
        assert date_range.end == date.today()

    def test_parse_date_to_range_tomorrow(self, workflow):
        """Test parsing 'tomorrow' to date range."""
        date_range = workflow._parse_date_to_range("tomorrow")
        expected = date.today() + timedelta(days=1)
        assert date_range.start == expected
        assert date_range.end == expected

    def test_parse_date_to_range_next_week(self, workflow):
        """Test parsing 'next week' to date range."""
        date_range = workflow._parse_date_to_range("next week")
        expected_start = date.today() + timedelta(days=7)
        expected_end = expected_start + timedelta(days=6)
        assert date_range.start == expected_start
        assert date_range.end == expected_end

    def test_parse_date_to_range_weekday(self, workflow):
        """Test parsing weekday name to date range."""
        date_range = workflow._parse_date_to_range("Monday")
        # Should be next Monday
        assert date_range.start.weekday() == 0  # Monday
        assert date_range.start > date.today()

    def test_find_best_slot_no_preference(self, workflow, available_slots):
        """Test finding best slot without preference."""
        slot = workflow._find_best_slot(available_slots, None)
        assert slot == available_slots[0]

    def test_find_best_slot_morning(self, workflow, available_slots):
        """Test finding best slot for morning preference."""
        slot = workflow._find_best_slot(available_slots, "morning")
        assert slot.start_time.hour < 12

    def test_find_best_slot_afternoon(self, workflow, available_slots):
        """Test finding best slot for afternoon preference."""
        slot = workflow._find_best_slot(available_slots, "afternoon")
        assert 12 <= slot.start_time.hour < 17

    def test_find_best_slot_specific_time(self, workflow, available_slots):
        """Test finding best slot for specific time."""
        slot = workflow._find_best_slot(available_slots, "2pm")
        # Should pick the 2pm slot (slot2)
        assert slot.start_time.hour == 14

    def test_find_best_slot_empty_list(self, workflow):
        """Test finding best slot with empty list raises error."""
        with pytest.raises(ValueError, match="No slots available"):
            workflow._find_best_slot([], "morning")
