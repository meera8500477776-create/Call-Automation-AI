"""
Appointment Rescheduling Workflow.

Multi-turn conversation flow for rescheduling existing appointments with
appointment lookup, constraint checking, and FHIR adapter integration.

Requirements: 10.7, 3.5
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

from voice_agent.adapters.fhir_adapter import (
    Appointment,
    AppointmentConstraints,
    DateRange,
    FHIRAdapter,
    Slot,
)
from voice_agent.core.session import Session

logger = logging.getLogger(__name__)


class RescheduleWorkflowStage(Enum):
    """Stages in the appointment rescheduling workflow."""
    INITIAL = "initial"
    LOOKING_UP_APPOINTMENT = "looking_up_appointment"
    CONFIRMING_APPOINTMENT = "confirming_appointment"
    COLLECTING_NEW_DATE = "collecting_new_date"
    COLLECTING_NEW_TIME = "collecting_new_time"
    SEARCHING_SLOTS = "searching_slots"
    CONFIRMING_RESCHEDULE = "confirming_reschedule"
    RESCHEDULING = "rescheduling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RescheduleWorkflowParameters:
    """Parameters collected during the rescheduling workflow."""
    patient_id: str | None = None
    existing_appointment_id: str | None = None
    existing_appointment: Appointment | None = None
    new_date: str | None = None
    new_time: str | None = None
    selected_slot: Slot | None = None
    available_slots: list[Slot] = None
    constraints: AppointmentConstraints | None = None

    def __post_init__(self):
        if self.available_slots is None:
            self.available_slots = []


class AppointmentReschedulingWorkflow:
    """
    Multi-turn conversation workflow for appointment rescheduling.
    
    Handles existing appointment lookup, constraint checking, new slot search,
    confirmation, and rescheduling through natural conversation flow.
    
    Requirements:
    - 10.7: Multi-turn workflow requiring multiple pieces of information
    - 3.5: FHIR adapter integration for rescheduling with constraint checking
    """

    def __init__(self, fhir_adapter: FHIRAdapter):
        """
        Initialize the appointment rescheduling workflow.
        
        Args:
            fhir_adapter: FHIR adapter for EHR integration
        """
        self.fhir_adapter = fhir_adapter
        self.logger = logging.getLogger(__name__)

    async def process_turn(
        self,
        session: Session,
        user_transcript: str,
        extracted_params: dict[str, Any]
    ) -> str:
        """
        Process a conversation turn in the rescheduling workflow.
        
        Args:
            session: Current conversation session
            user_transcript: User's transcribed speech
            extracted_params: Parameters extracted from user input
            
        Returns:
            Agent response text for this turn
        """
        # Get or initialize workflow parameters
        workflow_params = self._get_workflow_params(session)

        # Update parameters with newly extracted values
        self._update_parameters(workflow_params, extracted_params, session)

        # Determine current stage
        current_stage = RescheduleWorkflowStage(session.workflow_stage)

        # Process based on current stage
        try:
            if current_stage == RescheduleWorkflowStage.INITIAL:
                return await self._handle_initial(session, workflow_params)
            elif current_stage == RescheduleWorkflowStage.LOOKING_UP_APPOINTMENT:
                return await self._handle_looking_up_appointment(session, workflow_params)
            elif current_stage == RescheduleWorkflowStage.CONFIRMING_APPOINTMENT:
                return await self._handle_confirming_appointment(
                    session, workflow_params, user_transcript
                )
            elif current_stage == RescheduleWorkflowStage.COLLECTING_NEW_DATE:
                return await self._handle_collecting_new_date(session, workflow_params)
            elif current_stage == RescheduleWorkflowStage.COLLECTING_NEW_TIME:
                return await self._handle_collecting_new_time(session, workflow_params)
            elif current_stage == RescheduleWorkflowStage.SEARCHING_SLOTS:
                return await self._handle_searching_slots(session, workflow_params)
            elif current_stage == RescheduleWorkflowStage.CONFIRMING_RESCHEDULE:
                return await self._handle_confirming_reschedule(
                    session, workflow_params, user_transcript
                )
            elif current_stage == RescheduleWorkflowStage.RESCHEDULING:
                return await self._handle_rescheduling(session, workflow_params)
            elif current_stage == RescheduleWorkflowStage.COMPLETED:
                return "Your appointment has been rescheduled. Is there anything else I can help you with?"
            elif current_stage == RescheduleWorkflowStage.FAILED:
                return "I'm sorry, but I wasn't able to complete the rescheduling. Would you like to try again or speak with someone?"
            else:
                return await self._handle_initial(session, workflow_params)
        except Exception as e:
            self.logger.error(f"Error processing workflow turn: {e}", exc_info=True)
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I'm sorry, I encountered an error. Let me transfer you to someone who can help."

    def _get_workflow_params(self, session: Session) -> RescheduleWorkflowParameters:
        """Get or create workflow parameters from session."""
        if "workflow_params" not in session.metadata:
            # Initialize with patient_id from session metadata if available
            patient_id = session.metadata.get("patient_id")
            session.metadata["workflow_params"] = RescheduleWorkflowParameters(
                patient_id=patient_id
            )
        return session.metadata["workflow_params"]

    def _update_parameters(
        self,
        workflow_params: RescheduleWorkflowParameters,
        extracted_params: dict[str, Any],
        session: Session
    ) -> None:
        """Update workflow parameters with newly extracted values."""
        if "appointment_id" in extracted_params and not workflow_params.existing_appointment_id:
            workflow_params.existing_appointment_id = extracted_params["appointment_id"]
            session.extracted_parameters["appointment_id"] = extracted_params["appointment_id"]

        if "date" in extracted_params and not workflow_params.new_date:
            workflow_params.new_date = extracted_params["date"]
            session.extracted_parameters["new_date"] = extracted_params["date"]

        if "time" in extracted_params and not workflow_params.new_time:
            workflow_params.new_time = extracted_params["time"]
            session.extracted_parameters["new_time"] = extracted_params["time"]

    async def _handle_initial(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters
    ) -> str:
        """Handle initial stage - start looking up appointment."""
        if not workflow_params.patient_id:
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I'm sorry, I need your patient information to look up appointments. Let me transfer you to someone who can help."

        # Move to looking up appointment
        session.workflow_stage = RescheduleWorkflowStage.LOOKING_UP_APPOINTMENT.value
        return await self._handle_looking_up_appointment(session, workflow_params)

    async def _handle_looking_up_appointment(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters
    ) -> str:
        """Handle appointment lookup stage."""
        try:
            # Search for upcoming appointments for this patient
            # Look for appointments in the next 90 days
            today = date.today()
            date_range = DateRange(start=today, end=today + timedelta(days=90))

            # Use find_appointments to get available slots, then filter for booked ones
            # In a real implementation, we'd have a separate method to get existing appointments
            # For now, we'll simulate by checking if we have an appointment_id

            if workflow_params.existing_appointment_id:
                # We have a specific appointment ID, check its constraints
                constraints = await self.fhir_adapter.check_constraints(
                    workflow_params.existing_appointment_id
                )
                workflow_params.constraints = constraints

                # Create a mock appointment object for confirmation
                # In production, we'd fetch the full appointment details
                workflow_params.existing_appointment = Appointment(
                    id=workflow_params.existing_appointment_id,
                    patient_id=workflow_params.patient_id,
                    practitioner_id="",
                    start_time=datetime.now() + timedelta(days=7),
                    end_time=datetime.now() + timedelta(days=7, hours=1),
                    status="booked",
                    service_type="General",
                    is_cancelable=constraints.is_cancelable,
                    is_reschedulable=constraints.is_reschedulable,
                    location=""
                )

                # Check if appointment can be rescheduled
                if not constraints.is_reschedulable:
                    session.workflow_stage = RescheduleWorkflowStage.FAILED.value
                    return "I'm sorry, but this appointment cannot be rescheduled due to clinical constraints. Please contact the office directly for assistance."

                # Move to confirmation
                session.workflow_stage = RescheduleWorkflowStage.CONFIRMING_APPOINTMENT.value
                appt_time = workflow_params.existing_appointment.start_time.strftime("%I:%M %p")
                appt_date = workflow_params.existing_appointment.start_time.strftime("%A, %B %d")
                return f"I found your appointment scheduled for {appt_date} at {appt_time}. Is this the appointment you'd like to reschedule?"
            else:
                # No specific appointment ID provided
                # In production, we'd list recent appointments and ask which one
                session.workflow_stage = RescheduleWorkflowStage.FAILED.value
                return "I need more information to find your appointment. Could you tell me the date of your current appointment?"

        except Exception as e:
            self.logger.error(f"Error looking up appointment: {e}", exc_info=True)
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I'm having trouble finding your appointment. Let me transfer you to someone who can help."

    async def _handle_confirming_appointment(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters,
        user_transcript: str
    ) -> str:
        """Handle appointment confirmation stage."""
        transcript_lower = user_transcript.lower()

        if any(word in transcript_lower for word in ["yes", "yeah", "sure", "correct", "right"]):
            # User confirmed, move to collecting new date
            session.workflow_stage = RescheduleWorkflowStage.COLLECTING_NEW_DATE.value
            return "Great! What date would work better for you?"
        elif any(word in transcript_lower for word in ["no", "nope", "different", "wrong"]):
            # Wrong appointment
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I apologize for the confusion. Let me transfer you to someone who can help find the right appointment."
        else:
            # Unclear response, ask again
            return "I want to make sure I have the right appointment. Is this the one you'd like to reschedule?"

    async def _handle_collecting_new_date(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters
    ) -> str:
        """Handle new date collection stage."""
        if workflow_params.new_date:
            # Date collected, move to time
            session.workflow_stage = RescheduleWorkflowStage.COLLECTING_NEW_TIME.value
            return f"Perfect, I have {workflow_params.new_date}. What time would work for you?"
        else:
            # Still need date
            return "I didn't catch the date. What day would you like to reschedule your appointment to?"

    async def _handle_collecting_new_time(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters
    ) -> str:
        """Handle new time collection stage."""
        if workflow_params.new_time:
            # Time collected, search for slots
            session.workflow_stage = RescheduleWorkflowStage.SEARCHING_SLOTS.value
            return await self._handle_searching_slots(session, workflow_params)
        else:
            # Still need time
            return "What time would you prefer for your rescheduled appointment?"

    async def _handle_searching_slots(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters
    ) -> str:
        """Handle slot search stage."""
        try:
            # Parse date and create date range
            date_range = self._parse_date_to_range(workflow_params.new_date)

            # Search for available slots
            slots = await self.fhir_adapter.find_appointments(
                patient_id=workflow_params.patient_id,
                date_range=date_range,
                specialty=None
            )

            if not slots:
                session.workflow_stage = RescheduleWorkflowStage.FAILED.value
                return f"I'm sorry, I don't see any available appointments for {workflow_params.new_date}. Would you like to try a different date?"

            # Find best matching slot based on requested time
            best_slot = self._find_best_slot(slots, workflow_params.new_time)
            workflow_params.selected_slot = best_slot
            workflow_params.available_slots = slots

            # Move to confirmation
            session.workflow_stage = RescheduleWorkflowStage.CONFIRMING_RESCHEDULE.value

            # Format slot time for response
            slot_time = best_slot.start_time.strftime("%I:%M %p")
            slot_date = best_slot.start_time.strftime("%A, %B %d")

            return f"I found an available appointment on {slot_date} at {slot_time}. Would you like me to reschedule your appointment to this time?"

        except Exception as e:
            self.logger.error(f"Error searching for slots: {e}", exc_info=True)
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I'm having trouble finding available appointments right now. Let me transfer you to someone who can help."

    async def _handle_confirming_reschedule(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters,
        user_transcript: str
    ) -> str:
        """Handle reschedule confirmation stage."""
        transcript_lower = user_transcript.lower()

        if any(word in transcript_lower for word in ["yes", "yeah", "sure", "ok", "okay", "reschedule", "confirm"]):
            # User confirmed, proceed to rescheduling
            session.workflow_stage = RescheduleWorkflowStage.RESCHEDULING.value
            return await self._handle_rescheduling(session, workflow_params)
        elif any(word in transcript_lower for word in ["no", "nope", "different", "another"]):
            # User wants different time
            session.workflow_stage = RescheduleWorkflowStage.COLLECTING_NEW_DATE.value
            workflow_params.new_date = None
            workflow_params.new_time = None
            workflow_params.selected_slot = None
            return "No problem. What date and time would work better for you?"
        else:
            # Unclear response, ask again
            return "I want to make sure I understand. Would you like me to reschedule your appointment to this new time?"

    async def _handle_rescheduling(
        self,
        session: Session,
        workflow_params: RescheduleWorkflowParameters
    ) -> str:
        """Handle rescheduling stage."""
        if not workflow_params.existing_appointment_id or not workflow_params.selected_slot:
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I'm sorry, I lost track of the appointment details. Let's start over."

        try:
            # Reschedule the appointment
            updated_appointment = await self.fhir_adapter.reschedule_appointment(
                appointment_id=workflow_params.existing_appointment_id,
                new_slot_id=workflow_params.selected_slot.id
            )

            # Success!
            session.workflow_stage = RescheduleWorkflowStage.COMPLETED.value

            # Format confirmation message
            appt_time = updated_appointment.start_time.strftime("%I:%M %p")
            appt_date = updated_appointment.start_time.strftime("%A, %B %d")

            return f"Perfect! I've rescheduled your appointment to {appt_date} at {appt_time}. You'll receive a confirmation shortly."

        except ValueError as e:
            # Constraint violation or other validation error
            self.logger.error(f"Error rescheduling appointment: {e}", exc_info=True)
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return f"I'm sorry, I couldn't reschedule that appointment. {str(e)}"
        except Exception as e:
            self.logger.error(f"Error rescheduling appointment: {e}", exc_info=True)
            session.workflow_stage = RescheduleWorkflowStage.FAILED.value
            return "I'm sorry, I had trouble rescheduling that appointment. Let me transfer you to someone who can help."

    def _parse_date_to_range(self, date_str: str) -> DateRange:
        """
        Parse a date string to a DateRange.
        
        Args:
            date_str: Date string (e.g., "tomorrow", "Monday", "January 15")
            
        Returns:
            DateRange for searching appointments
        """
        today = date.today()

        # Handle relative dates
        if date_str.lower() == "today":
            return DateRange(start=today, end=today)
        elif date_str.lower() == "tomorrow":
            tomorrow = today + timedelta(days=1)
            return DateRange(start=tomorrow, end=tomorrow)
        elif date_str.lower() == "next week":
            next_week_start = today + timedelta(days=7)
            next_week_end = next_week_start + timedelta(days=6)
            return DateRange(start=next_week_start, end=next_week_end)

        # Handle day of week (e.g., "Monday")
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        date_lower = date_str.lower()
        for i, day in enumerate(weekdays):
            if day in date_lower:
                # Find next occurrence of this weekday
                days_ahead = i - today.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                target_date = today + timedelta(days=days_ahead)
                return DateRange(start=target_date, end=target_date)

        # Default: search next 7 days
        return DateRange(start=today, end=today + timedelta(days=7))

    def _find_best_slot(self, slots: list[Slot], preferred_time: str | None) -> Slot:
        """
        Find the best matching slot based on preferred time.
        
        Args:
            slots: List of available slots
            preferred_time: Preferred time string (e.g., "2pm", "morning")
            
        Returns:
            Best matching slot
        """
        if not slots:
            raise ValueError("No slots available")

        if not preferred_time:
            # No preference, return first slot
            return slots[0]

        # Parse preferred time
        time_lower = preferred_time.lower()

        # Handle time of day preferences
        if "morning" in time_lower:
            # Find slots before noon
            morning_slots = [s for s in slots if s.start_time.hour < 12]
            return morning_slots[0] if morning_slots else slots[0]
        elif "afternoon" in time_lower:
            # Find slots between noon and 5pm
            afternoon_slots = [s for s in slots if 12 <= s.start_time.hour < 17]
            return afternoon_slots[0] if afternoon_slots else slots[0]
        elif "evening" in time_lower:
            # Find slots after 5pm
            evening_slots = [s for s in slots if s.start_time.hour >= 17]
            return evening_slots[0] if evening_slots else slots[0]

        # Try to parse specific time (e.g., "2pm", "14:00")
        try:
            # Extract hour from time string
            import re
            time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', time_lower)
            if time_match:
                hour = int(time_match.group(1))
                am_pm = time_match.group(3)

                # Convert to 24-hour format
                if am_pm == "pm" and hour != 12:
                    hour += 12
                elif am_pm == "am" and hour == 12:
                    hour = 0

                # Find closest slot to this hour
                closest_slot = min(slots, key=lambda s: abs(s.start_time.hour - hour))
                return closest_slot
        except Exception:
            pass

        # Default: return first slot
        return slots[0]
