"""
Example usage of the Appointment Cancellation Workflow.

This example demonstrates how to use the AppointmentCancellationWorkflow
to handle multi-turn conversations for canceling appointments.

Requirements: 10.7, 3.5
"""

import asyncio
from datetime import datetime, timedelta

from voice_agent.workflows.appointment_cancellation import (
    AppointmentCancellationWorkflow,
    CancelWorkflowStage
)
from voice_agent.adapters.fhir_adapter import (
    FHIRAdapter, Appointment, AppointmentConstraints, CancellationResult
)
from voice_agent.core.session import Session


class MockFHIRAdapter:
    """Mock FHIR adapter for demonstration purposes."""
    
    async def check_constraints(self, appointment_id: str) -> AppointmentConstraints:
        """Mock constraint checking."""
        print(f"[FHIR] Checking constraints for appointment {appointment_id}")
        return AppointmentConstraints(
            is_cancelable=True,
            is_reschedulable=True
        )
    
    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: str
    ) -> CancellationResult:
        """Mock appointment cancellation."""
        print(f"[FHIR] Canceling appointment {appointment_id}")
        print(f"[FHIR] Reason: {reason}")
        return CancellationResult(
            success=True,
            appointment_id=appointment_id,
            reason=reason
        )


async def example_cancellation_workflow():
    """
    Example of a complete appointment cancellation workflow.
    
    This demonstrates the multi-turn conversation flow:
    1. User requests to cancel appointment
    2. System looks up appointment and checks constraints
    3. System confirms which appointment to cancel
    4. System asks for cancellation reason
    5. System confirms cancellation
    6. System cancels the appointment
    """
    print("=" * 60)
    print("Appointment Cancellation Workflow Example")
    print("=" * 60)
    print()
    
    # Create mock FHIR adapter
    fhir_adapter = MockFHIRAdapter()
    
    # Create workflow
    workflow = AppointmentCancellationWorkflow(fhir_adapter)
    
    # Create session
    session = Session(
        session_id="example_session",
        tenant_id="healthcare_clinic",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/12345"}
    )
    session.workflow_stage = CancelWorkflowStage.INITIAL.value
    
    print("Session created:")
    print(f"  Patient ID: {session.metadata['patient_id']}")
    print(f"  Session ID: {session.session_id}")
    print()
    
    # Turn 1: User wants to cancel appointment
    print("Turn 1: User initiates cancellation")
    print("-" * 60)
    user_input_1 = "I need to cancel my appointment"
    extracted_params_1 = {"appointment_id": "Appointment/67890"}
    print(f"User: {user_input_1}")
    print(f"Extracted params: {extracted_params_1}")
    
    response_1 = await workflow.process_turn(
        session=session,
        user_transcript=user_input_1,
        extracted_params=extracted_params_1
    )
    print(f"Agent: {response_1}")
    print(f"Stage: {session.workflow_stage}")
    print()
    
    # Turn 2: User confirms the appointment
    print("Turn 2: User confirms appointment")
    print("-" * 60)
    user_input_2 = "Yes, that's the one"
    extracted_params_2 = {}
    print(f"User: {user_input_2}")
    
    response_2 = await workflow.process_turn(
        session=session,
        user_transcript=user_input_2,
        extracted_params=extracted_params_2
    )
    print(f"Agent: {response_2}")
    print(f"Stage: {session.workflow_stage}")
    print()
    
    # Turn 3: User provides reason
    print("Turn 3: User provides cancellation reason")
    print("-" * 60)
    user_input_3 = "I'm feeling much better now"
    extracted_params_3 = {"reason": "feeling better"}
    print(f"User: {user_input_3}")
    print(f"Extracted params: {extracted_params_3}")
    
    response_3 = await workflow.process_turn(
        session=session,
        user_transcript=user_input_3,
        extracted_params=extracted_params_3
    )
    print(f"Agent: {response_3}")
    print(f"Stage: {session.workflow_stage}")
    print()
    
    # Turn 4: User confirms cancellation
    print("Turn 4: User confirms cancellation")
    print("-" * 60)
    user_input_4 = "Yes, please cancel it"
    extracted_params_4 = {}
    print(f"User: {user_input_4}")
    
    response_4 = await workflow.process_turn(
        session=session,
        user_transcript=user_input_4,
        extracted_params=extracted_params_4
    )
    print(f"Agent: {response_4}")
    print(f"Stage: {session.workflow_stage}")
    print()
    
    print("=" * 60)
    print("Workflow completed successfully!")
    print("=" * 60)


async def example_cancellation_not_allowed():
    """
    Example of cancellation workflow when appointment cannot be canceled.
    
    This demonstrates constraint checking and graceful failure.
    """
    print("\n\n")
    print("=" * 60)
    print("Cancellation Not Allowed Example")
    print("=" * 60)
    print()
    
    # Create mock FHIR adapter that returns non-cancelable constraint
    class NonCancelableFHIRAdapter:
        async def check_constraints(self, appointment_id: str) -> AppointmentConstraints:
            print(f"[FHIR] Checking constraints for appointment {appointment_id}")
            return AppointmentConstraints(
                is_cancelable=False,  # Cannot cancel
                is_reschedulable=True
            )
    
    fhir_adapter = NonCancelableFHIRAdapter()
    workflow = AppointmentCancellationWorkflow(fhir_adapter)
    
    # Create session
    session = Session(
        session_id="example_session_2",
        tenant_id="healthcare_clinic",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/12345"}
    )
    session.workflow_stage = CancelWorkflowStage.INITIAL.value
    
    # User tries to cancel
    print("User tries to cancel appointment")
    print("-" * 60)
    user_input = "I need to cancel my appointment"
    extracted_params = {"appointment_id": "Appointment/99999"}
    print(f"User: {user_input}")
    print(f"Extracted params: {extracted_params}")
    
    response = await workflow.process_turn(
        session=session,
        user_transcript=user_input,
        extracted_params=extracted_params
    )
    print(f"Agent: {response}")
    print(f"Stage: {session.workflow_stage}")
    print()
    
    print("=" * 60)
    print("Workflow failed due to constraints (expected behavior)")
    print("=" * 60)


async def example_user_changes_mind():
    """
    Example of cancellation workflow when user changes their mind.
    
    This demonstrates handling user hesitation and cancellation of the cancellation.
    """
    print("\n\n")
    print("=" * 60)
    print("User Changes Mind Example")
    print("=" * 60)
    print()
    
    fhir_adapter = MockFHIRAdapter()
    workflow = AppointmentCancellationWorkflow(fhir_adapter)
    
    # Create session
    session = Session(
        session_id="example_session_3",
        tenant_id="healthcare_clinic",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/12345"}
    )
    session.workflow_stage = CancelWorkflowStage.INITIAL.value
    
    # Turn 1: User wants to cancel
    print("Turn 1: User initiates cancellation")
    print("-" * 60)
    response_1 = await workflow.process_turn(
        session=session,
        user_transcript="I want to cancel my appointment",
        extracted_params={"appointment_id": "Appointment/11111"}
    )
    print(f"Agent: {response_1}")
    print()
    
    # Turn 2: User confirms appointment
    print("Turn 2: User confirms appointment")
    print("-" * 60)
    response_2 = await workflow.process_turn(
        session=session,
        user_transcript="Yes",
        extracted_params={}
    )
    print(f"Agent: {response_2}")
    print()
    
    # Turn 3: User provides reason
    print("Turn 3: User provides reason")
    print("-" * 60)
    response_3 = await workflow.process_turn(
        session=session,
        user_transcript="I have a conflict",
        extracted_params={"reason": "scheduling conflict"}
    )
    print(f"Agent: {response_3}")
    print()
    
    # Turn 4: User changes mind
    print("Turn 4: User changes mind")
    print("-" * 60)
    user_input_4 = "Actually, no, don't cancel it"
    print(f"User: {user_input_4}")
    response_4 = await workflow.process_turn(
        session=session,
        user_transcript=user_input_4,
        extracted_params={}
    )
    print(f"Agent: {response_4}")
    print(f"Stage: {session.workflow_stage}")
    print()
    
    print("=" * 60)
    print("Workflow stopped - appointment NOT canceled (expected behavior)")
    print("=" * 60)


async def main():
    """Run all examples."""
    await example_cancellation_workflow()
    await example_cancellation_not_allowed()
    await example_user_changes_mind()


if __name__ == "__main__":
    asyncio.run(main())
