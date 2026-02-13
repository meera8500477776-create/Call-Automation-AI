"""
Appointment Scheduling Workflow.

Multi-turn conversation flow for scheduling appointments with parameter extraction,
FHIR adapter integration, confirmation, and error handling.

Requirements: 10.7, 10.8, 3.2, 3.3
"""

import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum

from voice_agent.adapters.fhir_adapter import FHIRAdapter, DateRange, Slot, Appointment
from voice_agent.core.session import Session, ConversationTurn
from voice_agent.services.llm_service import IntentType


logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages in the appointment scheduling workflow."""
    INITIAL = "initial"
    COLLECTING_DATE = "collecting_date"
    COLLECTING_TIME = "collecting_time"
    COLLECTING_REASON = "collecting_reason"
    COLLECTING_PROVIDER = "collecting_provider"
    SEARCHING_SLOTS = "searching_slots"
    CONFIRMING = "confirming"
    BOOKING = "booking"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowParameters:
    """Parameters collected during the scheduling workflow."""
    date: Optional[str] = None
    time: Optional[str] = None
    reason: Optional[str] = None
    provider: Optional[str] = None
    patient_id: Optional[str] = None
    selected_slot: Optional[Slot] = None
    available_slots: List[Slot] = None
    
    def __post_init__(self):
        if self.available_slots is None:
            self.available_slots = []
    
    def is_complete(self) -> bool:
        """Check if all required parameters are collected."""
        return all([
            self.date is not None,
            self.time is not None,
            self.patient_id is not None
        ])
    
    def get_missing_parameters(self) -> List[str]:
        """Get list of missing required parameters."""
        missing = []
        if not self.date:
            missing.append("date")
        if not self.time:
            missing.append("time")
        if not self.patient_id:
            missing.append("patient_id")
        return missing


class AppointmentSchedulingWorkflow:
    """
    Multi-turn conversation workflow for appointment scheduling.
    
    Handles parameter extraction, slot search, confirmation, and booking
    through natural conversation flow with error handling and recovery.
    
    Requirements:
    - 10.7: Multi-turn workflow requiring multiple pieces of information
    - 10.8: Appointment scheduling workflow
    - 3.2: FHIR adapter integration for slot search
    - 3.3: FHIR adapter integration for booking
    """
    
    def __init__(self, fhir_adapter: FHIRAdapter):
        """
        Initialize the appointment scheduling workflow.
        
        Args:
            fhir_adapter: FHIR adapter for EHR integration
        """
        self.fhir_adapter = fhir_adapter
        self.logger = logging.getLogger(__name__)
    
    async def process_turn(
        self,
        session: Session,
        user_transcript: str,
        extracted_params: Dict[str, Any]
    ) -> str:
        """
        Process a conversation turn in the scheduling workflow.
        
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
        current_stage = WorkflowStage(session.workflow_stage)
        
        # Process based on current stage
        try:
            if current_stage == WorkflowStage.INITIAL:
                return await self._handle_initial(session, workflow_params)
            elif current_stage == WorkflowStage.COLLECTING_DATE:
                return await self._handle_collecting_date(session, workflow_params)
            elif current_stage == WorkflowStage.COLLECTING_TIME:
                return await self._handle_collecting_time(session, workflow_params)
            elif current_stage == WorkflowStage.COLLECTING_REASON:
                return await self._handle_collecting_reason(session, workflow_params)
            elif current_stage == WorkflowStage.SEARCHING_SLOTS:
                return await self._handle_searching_slots(session, workflow_params)
            elif current_stage == WorkflowStage.CONFIRMING:
                return await self._handle_confirming(session, workflow_params, user_transcript)
            elif current_stage == WorkflowStage.BOOKING:
                return await self._handle_booking(session, workflow_params)
            elif current_stage == WorkflowStage.COMPLETED:
                return "Your appointment has been scheduled. Is there anything else I can help you with?"
            elif current_stage == WorkflowStage.FAILED:
                return "I'm sorry, but I wasn't able to complete the scheduling. Would you like to try again or speak with someone?"
            else:
                return await self._handle_initial(session, workflow_params)
        except Exception as e:
            self.logger.error(f"Error processing workflow turn: {e}", exc_info=True)
            session.workflow_stage = WorkflowStage.FAILED.value
            return "I'm sorry, I encountered an error. Let me transfer you to someone who can help."
    
    def _get_workflow_params(self, session: Session) -> WorkflowParameters:
        """Get or create workflow parameters from session."""
        if "workflow_params" not in session.metadata:
            # Initialize with patient_id from session metadata if available
            patient_id = session.metadata.get("patient_id")
            session.metadata["workflow_params"] = WorkflowParameters(patient_id=patient_id)
        return session.metadata["workflow_params"]
    
    def _update_parameters(
        self,
        workflow_params: WorkflowParameters,
        extracted_params: Dict[str, Any],
        session: Session
    ) -> None:
        """Update workflow parameters with newly extracted values."""
        if "date" in extracted_params and not workflow_params.date:
            workflow_params.date = extracted_params["date"]
            session.extracted_parameters["date"] = extracted_params["date"]
        
        if "time" in extracted_params and not workflow_params.time:
            workflow_params.time = extracted_params["time"]
            session.extracted_parameters["time"] = extracted_params["time"]
        
        if "reason" in extracted_params and not workflow_params.reason:
            workflow_params.reason = extracted_params["reason"]
            session.extracted_parameters["reason"] = extracted_params["reason"]
        
        if "provider" in extracted_params and not workflow_params.provider:
            workflow_params.provider = extracted_params["provider"]
            session.extracted_parameters["provider"] = extracted_params["provider"]
    
    async def _handle_initial(
        self,
        session: Session,
        workflow_params: WorkflowParameters
    ) -> str:
        """Handle initial stage - start collecting parameters."""
        # Check what we already have
        missing = workflow_params.get_missing_parameters()
        
        if not workflow_params.date:
            session.workflow_stage = WorkflowStage.COLLECTING_DATE.value
            return "I'd be happy to help you schedule an appointment. What date works best for you?"
        elif not workflow_params.time:
            session.workflow_stage = WorkflowStage.COLLECTING_TIME.value
            return f"Great, I have {workflow_params.date}. What time would you prefer?"
        else:
            # We have date and time, move to searching
            session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
            return await self._handle_searching_slots(session, workflow_params)
    
    async def _handle_collecting_date(
        self,
        session: Session,
        workflow_params: WorkflowParameters
    ) -> str:
        """Handle date collection stage."""
        if workflow_params.date:
            # Date collected, move to time
            session.workflow_stage = WorkflowStage.COLLECTING_TIME.value
            return f"Perfect, I have {workflow_params.date}. What time would work for you?"
        else:
            # Still need date
            return "I didn't catch the date. Could you tell me what day you'd like to schedule your appointment?"
    
    async def _handle_collecting_time(
        self,
        session: Session,
        workflow_params: WorkflowParameters
    ) -> str:
        """Handle time collection stage."""
        if workflow_params.time:
            # Time collected, optionally collect reason
            if not workflow_params.reason:
                session.workflow_stage = WorkflowStage.COLLECTING_REASON.value
                return f"Got it, {workflow_params.date} at {workflow_params.time}. What's the reason for your visit?"
            else:
                # Have date, time, and reason - search for slots
                session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
                return await self._handle_searching_slots(session, workflow_params)
        else:
            # Still need time
            return "What time would you prefer for your appointment?"
    
    async def _handle_collecting_reason(
        self,
        session: Session,
        workflow_params: WorkflowParameters
    ) -> str:
        """Handle reason collection stage."""
        if workflow_params.reason:
            # Reason collected, search for slots
            session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
            return await self._handle_searching_slots(session, workflow_params)
        else:
            # Reason is optional, can proceed without it
            session.workflow_stage = WorkflowStage.SEARCHING_SLOTS.value
            return await self._handle_searching_slots(session, workflow_params)
    
    async def _handle_searching_slots(
        self,
        session: Session,
        workflow_params: WorkflowParameters
    ) -> str:
        """Handle slot search stage."""
        if not workflow_params.patient_id:
            session.workflow_stage = WorkflowStage.FAILED.value
            return "I'm sorry, I need your patient information to search for appointments. Let me transfer you to someone who can help."
        
        try:
            # Parse date and create date range
            date_range = self._parse_date_to_range(workflow_params.date)
            
            # Search for available slots
            slots = await self.fhir_adapter.find_appointments(
                patient_id=workflow_params.patient_id,
                date_range=date_range,
                specialty=workflow_params.provider
            )
            
            if not slots:
                session.workflow_stage = WorkflowStage.FAILED.value
                return f"I'm sorry, I don't see any available appointments for {workflow_params.date}. Would you like to try a different date?"
            
            # Find best matching slot based on requested time
            best_slot = self._find_best_slot(slots, workflow_params.time)
            workflow_params.selected_slot = best_slot
            workflow_params.available_slots = slots
            
            # Move to confirmation
            session.workflow_stage = WorkflowStage.CONFIRMING.value
            
            # Format slot time for response
            slot_time = best_slot.start_time.strftime("%I:%M %p")
            slot_date = best_slot.start_time.strftime("%A, %B %d")
            
            return f"I found an available appointment on {slot_date} at {slot_time}. Would you like me to book that for you?"
            
        except Exception as e:
            self.logger.error(f"Error searching for slots: {e}", exc_info=True)
            session.workflow_stage = WorkflowStage.FAILED.value
            return "I'm having trouble finding available appointments right now. Let me transfer you to someone who can help."
    
    async def _handle_confirming(
        self,
        session: Session,
        workflow_params: WorkflowParameters,
        user_transcript: str
    ) -> str:
        """Handle confirmation stage."""
        # Check for confirmation
        transcript_lower = user_transcript.lower()
        
        if any(word in transcript_lower for word in ["yes", "yeah", "sure", "ok", "okay", "book it", "confirm"]):
            # User confirmed, proceed to booking
            session.workflow_stage = WorkflowStage.BOOKING.value
            return await self._handle_booking(session, workflow_params)
        elif any(word in transcript_lower for word in ["no", "nope", "different", "another"]):
            # User wants different time
            session.workflow_stage = WorkflowStage.COLLECTING_DATE.value
            workflow_params.date = None
            workflow_params.time = None
            workflow_params.selected_slot = None
            return "No problem. What date and time would work better for you?"
        else:
            # Unclear response, ask again
            return "I want to make sure I understand. Would you like me to book this appointment?"
    
    async def _handle_booking(
        self,
        session: Session,
        workflow_params: WorkflowParameters
    ) -> str:
        """Handle booking stage."""
        if not workflow_params.selected_slot:
            session.workflow_stage = WorkflowStage.FAILED.value
            return "I'm sorry, I lost track of the appointment details. Let's start over."
        
        try:
            # Book the appointment
            appointment = await self.fhir_adapter.book_appointment(
                patient_id=workflow_params.patient_id,
                slot_id=workflow_params.selected_slot.id,
                reason=workflow_params.reason or "General appointment",
                practitioner_id=workflow_params.selected_slot.practitioner_id
            )
            
            # Success!
            session.workflow_stage = WorkflowStage.COMPLETED.value
            
            # Format confirmation message
            appt_time = appointment.start_time.strftime("%I:%M %p")
            appt_date = appointment.start_time.strftime("%A, %B %d")
            
            return f"Perfect! I've scheduled your appointment for {appt_date} at {appt_time}. You'll receive a confirmation shortly."
            
        except Exception as e:
            self.logger.error(f"Error booking appointment: {e}", exc_info=True)
            session.workflow_stage = WorkflowStage.FAILED.value
            return "I'm sorry, I had trouble booking that appointment. Let me transfer you to someone who can help."
    
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
    
    def _find_best_slot(self, slots: List[Slot], preferred_time: Optional[str]) -> Slot:
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
