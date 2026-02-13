"""
Unit tests for appointment scheduling workflow.

Tests multi-turn conversation flow, parameter extraction, slot search,
confirmation, and booking with error handling.

Requirements: 10.7, 10.8, 3.2, 3.3
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, Mock, patch

from voice_agent.workflows.appointment_scheduling import (
    AppointmentSchedulingWorkflow,
    WorkflowStage,
    WorkflowParameters
)
from voice_agent.adapters.fhir_adapter import Slot, Appointment, DateRange
from voice_agent.core.session import Session


@pytest.fixture
def mock_fhir_adapter():
    """Create a mock FHIR adapter."""
    adapter = AsyncMock()
    return adapter


@pytest.fixture
def workflow(mock_fhir_adapter):
    """Create an appointment scheduling workflow instance."""
    return AppointmentSchedulingWorkflow(fhir_adapter=mock_fhir_adapter)


@pytest.fixture
def test_session():
    """Create a test session."""
    session = Session(
        tenant_id="test_tenant",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/123"}
    )
    session.workflow_stage = WorkflowStage.INITIAL.value
    return session


@pytest.fixture
def test_slots():
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
        ),
        Slot(
            id="slot3",
            schedule_id="schedule1",
            start_time=tomorrow.replace(hour=17, minute=0),
            end_time=tomorrow.replace(hour=17, minute=30),
            status="free",
            service_type="General",
            practitioner_id="Practitioner/1"
        )
    ]


class TestWorkflowParameters:
    """Test WorkflowParameters data class."""
    
    def test_is_complete_with_all_params(self):
        """Test is_complete returns True when all required params present."""
        params = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123"
        )
        assert params.is_complete()
    
    def test_is_complete_missing_date(self):
        """Test is_complete returns False when date missing."""
        params = WorkflowParameters(
            time="2pm",
            patient_id="Patient/123"
        )
        assert not params.is_complete()
    
    def test_is_complete_missing_time(self):
        """Test is_complete returns False when time missing."""
        params = WorkflowParameters(
            date="tomorrow",
            patient_id="Patient/123"
        )
        assert not params.is_complete()
    
    def test_is_complete_missing_patient_id(self):
        """Test is_complete returns False when patient_id missing."""
        params = WorkflowParameters(
            date="tomorrow",
            time="2pm"
        )
        assert not params.is_complete()
    
    def test_get_missing_parameters(self):
        """Test get_missing_parameters returns correct list."""
        params = WorkflowParameters(date="tomorrow")
        missing = params.get_missing_parameters()
        assert "time" in missing
        assert "patient_id" in missing
        assert "date" not in missing


class TestInitialStage:
    """Test initial stage of workflow."""
    
    @pytest.mark.asyncio
    async def test_initial_stage_asks_for_date(self, workflow, test_session):
        """Test initial stage asks for date when no parameters."""
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="I want to schedule an appointment",
            extracted_params={}
        )
        
        assert "date" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_DATE.value
    
    @pytest.mark.asyncio
    async def test_initial_stage_with_date_asks_for_time(self, workflow, test_session):
        """Test initial stage asks for time when date provided."""
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="I want to schedule an appointment for tomorrow",
            extracted_params={"date": "tomorrow"}
        )
        
        assert "time" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_TIME.value


class TestDateCollection:
    """Test date collection stage."""
    
    @pytest.mark.asyncio
    async def test_collecting_date_moves_to_time(self, workflow, test_session):
        """Test collecting date stage moves to time collection."""
        test_session.workflow_stage = WorkflowStage.COLLECTING_DATE.value
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="tomorrow",
            extracted_params={"date": "tomorrow"}
        )
        
        assert "time" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_TIME.value
    
    @pytest.mark.asyncio
    async def test_collecting_date_without_date_asks_again(self, workflow, test_session):
        """Test collecting date without date parameter asks again."""
        test_session.workflow_stage = WorkflowStage.COLLECTING_DATE.value
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="I'm not sure",
            extracted_params={}
        )
        
        assert "date" in response.lower() or "day" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_DATE.value


class TestTimeCollection:
    """Test time collection stage."""
    
    @pytest.mark.asyncio
    async def test_collecting_time_moves_to_reason(self, workflow, test_session):
        """Test collecting time stage moves to reason collection."""
        test_session.workflow_stage = WorkflowStage.COLLECTING_TIME.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            patient_id="Patient/123"
        )
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="2pm",
            extracted_params={"time": "2pm"}
        )
        
        assert "reason" in response.lower() or "visit" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_REASON.value
    
    @pytest.mark.asyncio
    async def test_collecting_time_without_time_asks_again(self, workflow, test_session):
        """Test collecting time without time parameter asks again."""
        test_session.workflow_stage = WorkflowStage.COLLECTING_TIME.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            patient_id="Patient/123"
        )
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="I don't know",
            extracted_params={}
        )
        
        assert "time" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_TIME.value


class TestSlotSearch:
    """Test slot search stage."""
    
    @pytest.mark.asyncio
    async def test_searching_slots_finds_appointments(
        self,
        workflow,
        test_session,
        mock_fhir_adapter,
        test_slots
    ):
        """Test searching slots finds and presents available appointments."""
        test_session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123"
        )
        
        mock_fhir_adapter.find_appointments.return_value = test_slots
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "found" in response.lower() or "available" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.CONFIRMING.value
        mock_fhir_adapter.find_appointments.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_searching_slots_no_availability(
        self,
        workflow,
        test_session,
        mock_fhir_adapter
    ):
        """Test searching slots handles no availability."""
        test_session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123"
        )
        
        mock_fhir_adapter.find_appointments.return_value = []
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "sorry" in response.lower() or "don't see" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_searching_slots_without_patient_id_fails(
        self,
        workflow,
        test_session
    ):
        """Test searching slots without patient_id fails gracefully."""
        test_session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id=None  # Missing patient ID
        )
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "sorry" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value


class TestConfirmation:
    """Test confirmation stage."""
    
    @pytest.mark.asyncio
    async def test_confirming_with_yes_books_appointment(
        self,
        workflow,
        test_session,
        mock_fhir_adapter,
        test_slots
    ):
        """Test confirmation with 'yes' proceeds to booking."""
        test_session.workflow_stage = WorkflowStage.CONFIRMING.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123",
            selected_slot=test_slots[0]
        )
        
        mock_appointment = Appointment(
            id="appt1",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=test_slots[0].start_time,
            end_time=test_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.book_appointment.return_value = mock_appointment
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="yes",
            extracted_params={}
        )
        
        assert "scheduled" in response.lower() or "booked" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COMPLETED.value
        mock_fhir_adapter.book_appointment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_confirming_with_no_restarts(
        self,
        workflow,
        test_session,
        test_slots
    ):
        """Test confirmation with 'no' restarts date collection."""
        test_session.workflow_stage = WorkflowStage.CONFIRMING.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123",
            selected_slot=test_slots[0]
        )
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="no, I want a different time",
            extracted_params={}
        )
        
        assert test_session.workflow_stage == WorkflowStage.COLLECTING_DATE.value
        # Parameters should be reset
        workflow_params = test_session.metadata["workflow_params"]
        assert workflow_params.date is None
        assert workflow_params.time is None
    
    @pytest.mark.asyncio
    async def test_confirming_with_unclear_response_asks_again(
        self,
        workflow,
        test_session,
        test_slots
    ):
        """Test confirmation with unclear response asks again."""
        test_session.workflow_stage = WorkflowStage.CONFIRMING.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123",
            selected_slot=test_slots[0]
        )
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="maybe",
            extracted_params={}
        )
        
        assert "book" in response.lower() or "understand" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.CONFIRMING.value


class TestBooking:
    """Test booking stage."""
    
    @pytest.mark.asyncio
    async def test_booking_success(
        self,
        workflow,
        test_session,
        mock_fhir_adapter,
        test_slots
    ):
        """Test successful appointment booking."""
        test_session.workflow_stage = WorkflowStage.BOOKING.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            reason="checkup",
            patient_id="Patient/123",
            selected_slot=test_slots[0]
        )
        
        mock_appointment = Appointment(
            id="appt1",
            patient_id="Patient/123",
            practitioner_id="Practitioner/1",
            start_time=test_slots[0].start_time,
            end_time=test_slots[0].end_time,
            status="booked",
            service_type="General",
            is_cancelable=True,
            is_reschedulable=True
        )
        mock_fhir_adapter.book_appointment.return_value = mock_appointment
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "scheduled" in response.lower() or "booked" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.COMPLETED.value
        
        # Verify booking was called with correct parameters
        call_args = mock_fhir_adapter.book_appointment.call_args
        assert call_args.kwargs["patient_id"] == "Patient/123"
        assert call_args.kwargs["slot_id"] == "slot1"
        assert call_args.kwargs["reason"] == "checkup"
    
    @pytest.mark.asyncio
    async def test_booking_without_slot_fails(
        self,
        workflow,
        test_session
    ):
        """Test booking without selected slot fails gracefully."""
        test_session.workflow_stage = WorkflowStage.BOOKING.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123",
            selected_slot=None  # No slot selected
        )
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "sorry" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value
    
    @pytest.mark.asyncio
    async def test_booking_api_error_fails_gracefully(
        self,
        workflow,
        test_session,
        mock_fhir_adapter,
        test_slots
    ):
        """Test booking handles API errors gracefully."""
        test_session.workflow_stage = WorkflowStage.BOOKING.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123",
            selected_slot=test_slots[0]
        )
        
        mock_fhir_adapter.book_appointment.side_effect = Exception("API error")
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "sorry" in response.lower() or "trouble" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value


class TestDateParsing:
    """Test date parsing utilities."""
    
    def test_parse_today(self, workflow):
        """Test parsing 'today' to date range."""
        date_range = workflow._parse_date_to_range("today")
        assert date_range.start == date.today()
        assert date_range.end == date.today()
    
    def test_parse_tomorrow(self, workflow):
        """Test parsing 'tomorrow' to date range."""
        date_range = workflow._parse_date_to_range("tomorrow")
        tomorrow = date.today() + timedelta(days=1)
        assert date_range.start == tomorrow
        assert date_range.end == tomorrow
    
    def test_parse_next_week(self, workflow):
        """Test parsing 'next week' to date range."""
        date_range = workflow._parse_date_to_range("next week")
        assert date_range.end > date_range.start
        assert (date_range.end - date_range.start).days == 6
    
    def test_parse_weekday(self, workflow):
        """Test parsing weekday name to date range."""
        date_range = workflow._parse_date_to_range("Monday")
        # Should be a Monday
        assert date_range.start.weekday() == 0
        # Should be in the future
        assert date_range.start >= date.today()


class TestSlotMatching:
    """Test slot matching utilities."""
    
    def test_find_best_slot_morning(self, workflow, test_slots):
        """Test finding best slot for morning preference."""
        best_slot = workflow._find_best_slot(test_slots, "morning")
        assert best_slot.start_time.hour < 12
    
    def test_find_best_slot_afternoon(self, workflow, test_slots):
        """Test finding best slot for afternoon preference."""
        best_slot = workflow._find_best_slot(test_slots, "afternoon")
        assert 12 <= best_slot.start_time.hour < 17
    
    def test_find_best_slot_evening(self, workflow, test_slots):
        """Test finding best slot for evening preference."""
        best_slot = workflow._find_best_slot(test_slots, "evening")
        assert best_slot.start_time.hour >= 17
    
    def test_find_best_slot_specific_time(self, workflow, test_slots):
        """Test finding best slot for specific time."""
        best_slot = workflow._find_best_slot(test_slots, "2pm")
        # Should match the 2pm slot (14:00)
        assert best_slot.start_time.hour == 14
    
    def test_find_best_slot_no_preference(self, workflow, test_slots):
        """Test finding best slot with no preference returns first."""
        best_slot = workflow._find_best_slot(test_slots, None)
        assert best_slot == test_slots[0]


class TestErrorHandling:
    """Test error handling throughout workflow."""
    
    @pytest.mark.asyncio
    async def test_exception_during_processing_fails_gracefully(
        self,
        workflow,
        test_session,
        mock_fhir_adapter
    ):
        """Test that exceptions during processing are handled gracefully."""
        test_session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
        test_session.metadata["workflow_params"] = WorkflowParameters(
            date="tomorrow",
            time="2pm",
            patient_id="Patient/123"
        )
        
        # Make FHIR adapter raise an exception
        mock_fhir_adapter.find_appointments.side_effect = Exception("Network error")
        
        response = await workflow.process_turn(
            session=test_session,
            user_transcript="",
            extracted_params={}
        )
        
        assert "sorry" in response.lower() or "trouble" in response.lower()
        assert test_session.workflow_stage == WorkflowStage.FAILED.value
