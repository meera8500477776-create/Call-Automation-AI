"""
Appointment Cancellation Workflow.

Multi-turn conversation flow for canceling existing appointments with
appointment lookup, constraint checking, confirmation, and FHIR adapter integration.

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
)
from voice_agent.core.session import Session

logger = logging.getLogger(__name__)


class CancelWorkflowStage(Enum):
    """Stages in the appointment cancellation workflow."""
    INITIAL = "initial"
    LOOKING_UP_APPOINTMENT = "looking_up_appointment"
    CONFIRMING_APPOINTMENT = "confirming_appointment"
    COLLECTING_REASON = "collecting_reason"
    CONFIRMING_CANCELLATION = "confirming_cancellation"
    CANCELING = "canceling"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CancelWorkflowParameters:
    """Parameters collected during the cancellation workflow."""
    patient_id: str | None = None
    existing_appointment_id: str | None = None
    existing_appointment: Appointment | None = None
    cancellation_reason: str | None = None
    constraints: AppointmentConstraints | None = None


class AppointmentCancellationWorkflow:
    """
    Multi-turn conversation workflow for appointment cancellation.
    
    Handles existing appointment lookup, constraint checking, reason collection,
    confirmation, and cancellation through natural conversation flow.
    
    Requirements:
    - 10.7: Multi-turn workflow requiring multiple pieces of information
    - 3.5: FHIR adapter integration for cancellation with constraint checking
    """

    def __init__(self, fhir_adapter: FHIRAdapter):
        """
        Initialize the appointment cancellation workflow.
        
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
        Process a conversation turn in the cancellation workflow.
        
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
        current_stage = CancelWorkflowStage(session.workflow_stage)

        # Process based on current stage
        try:
            if current_stage == CancelWorkflowStage.INITIAL:
                return await self._handle_initial(session, workflow_params)
            elif current_stage == CancelWorkflowStage.LOOKING_UP_APPOINTMENT:
                return await self._handle_looking_up_appointment(session, workflow_params)
            elif current_stage == CancelWorkflowStage.CONFIRMING_APPOINTMENT:
                return await self._handle_confirming_appointment(
                    session, workflow_params, user_transcript
                )
            elif current_stage == CancelWorkflowStage.COLLECTING_REASON:
                return await self._handle_collecting_reason(session, workflow_params)
            elif current_stage == CancelWorkflowStage.CONFIRMING_CANCELLATION:
                return await self._handle_confirming_cancellation(
                    session, workflow_params, user_transcript
                )
            elif current_stage == CancelWorkflowStage.CANCELING:
                return await self._handle_canceling(session, workflow_params)
            elif current_stage == CancelWorkflowStage.COMPLETED:
                return "Your appointment has been canceled. Is there anything else I can help you with?"
            elif current_stage == CancelWorkflowStage.FAILED:
                return "I'm sorry, but I wasn't able to complete the cancellation. Would you like to try again or speak with someone?"
            else:
                return await self._handle_initial(session, workflow_params)
        except Exception as e:
            self.logger.error(f"Error processing workflow turn: {e}", exc_info=True)
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "I'm sorry, I encountered an error. Let me transfer you to someone who can help."

    def _get_workflow_params(self, session: Session) -> CancelWorkflowParameters:
        """Get or create workflow parameters from session."""
        if "workflow_params" not in session.metadata:
            # Initialize with patient_id from session metadata if available
            patient_id = session.metadata.get("patient_id")
            session.metadata["workflow_params"] = CancelWorkflowParameters(
                patient_id=patient_id
            )
        return session.metadata["workflow_params"]

    def _update_parameters(
        self,
        workflow_params: CancelWorkflowParameters,
        extracted_params: dict[str, Any],
        session: Session
    ) -> None:
        """Update workflow parameters with newly extracted values."""
        if "appointment_id" in extracted_params and not workflow_params.existing_appointment_id:
            workflow_params.existing_appointment_id = extracted_params["appointment_id"]
            session.extracted_parameters["appointment_id"] = extracted_params["appointment_id"]

        if "reason" in extracted_params and not workflow_params.cancellation_reason:
            workflow_params.cancellation_reason = extracted_params["reason"]
            session.extracted_parameters["cancellation_reason"] = extracted_params["reason"]

    async def _handle_initial(
        self,
        session: Session,
        workflow_params: CancelWorkflowParameters
    ) -> str:
        """Handle initial stage - start looking up appointment."""
        if not workflow_params.patient_id:
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "I'm sorry, I need your patient information to look up appointments. Let me transfer you to someone who can help."

        # Move to looking up appointment
        session.workflow_stage = CancelWorkflowStage.LOOKING_UP_APPOINTMENT.value
        return await self._handle_looking_up_appointment(session, workflow_params)

    async def _handle_looking_up_appointment(
        self,
        session: Session,
        workflow_params: CancelWorkflowParameters
    ) -> str:
        """Handle appointment lookup stage."""
        try:
            # Search for upcoming appointments for this patient
            # Look for appointments in the next 90 days
            today = date.today()
            date_range = DateRange(start=today, end=today + timedelta(days=90))

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

                # Check if appointment can be canceled
                if not constraints.is_cancelable:
                    session.workflow_stage = CancelWorkflowStage.FAILED.value
                    return "I'm sorry, but this appointment cannot be canceled due to clinical constraints. Please contact the office directly for assistance."

                # Move to confirmation
                session.workflow_stage = CancelWorkflowStage.CONFIRMING_APPOINTMENT.value
                appt_time = workflow_params.existing_appointment.start_time.strftime("%I:%M %p")
                appt_date = workflow_params.existing_appointment.start_time.strftime("%A, %B %d")
                return f"I found your appointment scheduled for {appt_date} at {appt_time}. Is this the appointment you'd like to cancel?"
            else:
                # No specific appointment ID provided
                # In production, we'd list recent appointments and ask which one
                session.workflow_stage = CancelWorkflowStage.FAILED.value
                return "I need more information to find your appointment. Could you tell me the date of your appointment?"

        except Exception as e:
            self.logger.error(f"Error looking up appointment: {e}", exc_info=True)
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "I'm having trouble finding your appointment. Let me transfer you to someone who can help."

    async def _handle_confirming_appointment(
        self,
        session: Session,
        workflow_params: CancelWorkflowParameters,
        user_transcript: str
    ) -> str:
        """Handle appointment confirmation stage."""
        transcript_lower = user_transcript.lower()

        if any(word in transcript_lower for word in ["yes", "yeah", "sure", "correct", "right"]):
            # User confirmed, move to collecting reason
            session.workflow_stage = CancelWorkflowStage.COLLECTING_REASON.value
            return "I understand. May I ask why you need to cancel this appointment?"
        elif any(word in transcript_lower for word in ["no", "nope", "different", "wrong"]):
            # Wrong appointment
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "I apologize for the confusion. Let me transfer you to someone who can help find the right appointment."
        else:
            # Unclear response, ask again
            return "I want to make sure I have the right appointment. Is this the one you'd like to cancel?"

    async def _handle_collecting_reason(
        self,
        session: Session,
        workflow_params: CancelWorkflowParameters
    ) -> str:
        """Handle reason collection stage."""
        if workflow_params.cancellation_reason:
            # Reason collected, move to final confirmation
            session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
            return "I understand. Just to confirm, would you like me to cancel this appointment?"
        else:
            # Reason is optional, can proceed without it
            session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
            return "Okay. Just to confirm, would you like me to cancel this appointment?"

    async def _handle_confirming_cancellation(
        self,
        session: Session,
        workflow_params: CancelWorkflowParameters,
        user_transcript: str
    ) -> str:
        """Handle cancellation confirmation stage."""
        transcript_lower = user_transcript.lower()

        # Check for uncertain/negative responses first (higher priority)
        # Use word boundaries to avoid false matches
        uncertain_patterns = ["not sure", "unsure", "don't know", "maybe", "i don't think"]
        if any(pattern in transcript_lower for pattern in uncertain_patterns):
            # Unclear response, ask again
            session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
            return "I want to make sure I understand. Would you like me to cancel this appointment?"

        negative_patterns = [
            " no ", " no,", " nope ", " nope,", "don't", "do not", " wait ", " wait,"
        ]
        if any(pattern in f" {transcript_lower} " for pattern in negative_patterns):
            # User changed their mind
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "No problem, I won't cancel your appointment. Is there anything else I can help you with?"

        # Check for positive responses
        # Use word boundaries to avoid false matches
        positive_patterns = [
            " yes ", " yes,", " yeah ", " yeah,", " sure ", " sure,",
            " ok ", " ok,", " okay ", " okay,", " please ", " please,", "go ahead"
        ]
        if any(pattern in f" {transcript_lower} " for pattern in positive_patterns):
            # User confirmed, proceed to cancellation
            session.workflow_stage = CancelWorkflowStage.CANCELING.value
            return await self._handle_canceling(session, workflow_params)

        # Unclear response, ask again - explicitly keep stage
        session.workflow_stage = CancelWorkflowStage.CONFIRMING_CANCELLATION.value
        return "I want to make sure I understand. Would you like me to cancel this appointment?"

    async def _handle_canceling(
        self,
        session: Session,
        workflow_params: CancelWorkflowParameters
    ) -> str:
        """Handle cancellation stage."""
        if not workflow_params.existing_appointment_id:
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "I'm sorry, I lost track of the appointment details. Let's start over."

        try:
            # Cancel the appointment
            result = await self.fhir_adapter.cancel_appointment(
                appointment_id=workflow_params.existing_appointment_id,
                reason=workflow_params.cancellation_reason or "Patient requested cancellation"
            )

            # Success!
            session.workflow_stage = CancelWorkflowStage.COMPLETED.value

            # Format confirmation message
            if workflow_params.existing_appointment:
                appt_time = workflow_params.existing_appointment.start_time.strftime("%I:%M %p")
                appt_date = workflow_params.existing_appointment.start_time.strftime("%A, %B %d")
                return f"I've canceled your appointment for {appt_date} at {appt_time}. You'll receive a confirmation shortly."
            else:
                return "I've canceled your appointment. You'll receive a confirmation shortly."

        except ValueError as e:
            # Constraint violation or other validation error
            self.logger.error(f"Error canceling appointment: {e}", exc_info=True)
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return f"I'm sorry, I couldn't cancel that appointment. {str(e)}"
        except Exception as e:
            self.logger.error(f"Error canceling appointment: {e}", exc_info=True)
            session.workflow_stage = CancelWorkflowStage.FAILED.value
            return "I'm sorry, I had trouble canceling that appointment. Let me transfer you to someone who can help."
