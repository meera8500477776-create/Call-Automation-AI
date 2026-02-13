"""
Example usage of the Appointment Rescheduling Workflow.

Demonstrates how to use the rescheduling workflow for multi-turn
conversations about rescheduling existing appointments.
"""

import asyncio
from datetime import datetime, date, timedelta

from voice_agent.workflows.appointment_rescheduling import (
    AppointmentReschedulingWorkflow,
    RescheduleWorkflowStage
)
from voice_agent.adapters.fhir_adapter import (
    FHIRAdapter, DateRange, Slot, Appointment, AppointmentConstraints
)
from voice_agent.core.session import Session
from voice_agent.security.authentication import AuthenticationService


async def main():
    """Demonstrate appointment rescheduling workflow."""
    
    # Initialize FHIR adapter
    auth_service = AuthenticationService()
    fhir_adapter = FHIRAdapter(
        base_url="https://fhir.epic.com/api/FHIR/R4",
        client_id="your_client_id",
        client_secret="your_client_secret",
        auth_service=auth_service
    )
    
    # Create workflow
    workflow = AppointmentReschedulingWorkflow(fhir_adapter)
    
    # Create session with patient information
    session = Session(
        session_id="session_001",
        tenant_id="healthcare_clinic_001",
        caller_id="+15551234567",
        metadata={
            "patient_id": "patient_12345",
            "appointment_id": "appt_67890"  # Existing appointment
        },
        workflow_stage=RescheduleWorkflowStage.INITIAL.value
    )
    
    print("=== Appointment Rescheduling Workflow Demo ===\n")
    
    # Turn 1: Initial request
    print("User: I need to reschedule my appointment")
    response = await workflow.process_turn(
        session=session,
        user_transcript="I need to reschedule my appointment",
        extracted_params={}
    )
    print(f"Agent: {response}\n")
    
    # Turn 2: Confirm appointment
    print("User: Yes, that's the one")
    response = await workflow.process_turn(
        session=session,
        user_transcript="Yes, that's the one",
        extracted_params={}
    )
    print(f"Agent: {response}\n")
    
    # Turn 3: Provide new date
    print("User: How about next Monday?")
    response = await workflow.process_turn(
        session=session,
        user_transcript="How about next Monday?",
        extracted_params={"date": "Monday"}
    )
    print(f"Agent: {response}\n")
    
    # Turn 4: Provide new time
    print("User: 2pm would be great")
    response = await workflow.process_turn(
        session=session,
        user_transcript="2pm would be great",
        extracted_params={"time": "2pm"}
    )
    print(f"Agent: {response}\n")
    
    # Turn 5: Confirm reschedule
    print("User: Yes, please book that")
    response = await workflow.process_turn(
        session=session,
        user_transcript="Yes, please book that",
        extracted_params={}
    )
    print(f"Agent: {response}\n")
    
    # Check final state
    print(f"Final workflow stage: {session.workflow_stage}")
    print(f"Extracted parameters: {session.extracted_parameters}")


async def demo_constraint_violation():
    """Demonstrate handling of non-reschedulable appointment."""
    
    print("\n=== Constraint Violation Demo ===\n")
    
    # Initialize components
    auth_service = AuthenticationService()
    fhir_adapter = FHIRAdapter(
        base_url="https://fhir.epic.com/api/FHIR/R4",
        client_id="your_client_id",
        client_secret="your_client_secret",
        auth_service=auth_service
    )
    
    workflow = AppointmentReschedulingWorkflow(fhir_adapter)
    
    # Create session with non-reschedulable appointment
    session = Session(
        session_id="session_002",
        tenant_id="healthcare_clinic_001",
        caller_id="+15551234567",
        metadata={
            "patient_id": "patient_12345",
            "appointment_id": "appt_non_reschedulable"
        },
        workflow_stage=RescheduleWorkflowStage.INITIAL.value
    )
    
    print("User: I need to reschedule my surgery appointment")
    response = await workflow.process_turn(
        session=session,
        user_transcript="I need to reschedule my surgery appointment",
        extracted_params={}
    )
    print(f"Agent: {response}\n")
    print(f"Workflow stage: {session.workflow_stage}")


async def demo_no_availability():
    """Demonstrate handling when no slots are available."""
    
    print("\n=== No Availability Demo ===\n")
    
    # Initialize components
    auth_service = AuthenticationService()
    fhir_adapter = FHIRAdapter(
        base_url="https://fhir.epic.com/api/FHIR/R4",
        client_id="your_client_id",
        client_secret="your_client_secret",
        auth_service=auth_service
    )
    
    workflow = AppointmentReschedulingWorkflow(fhir_adapter)
    
    # Create session
    session = Session(
        session_id="session_003",
        tenant_id="healthcare_clinic_001",
        caller_id="+15551234567",
        metadata={
            "patient_id": "patient_12345",
            "appointment_id": "appt_67890"
        },
        workflow_stage=RescheduleWorkflowStage.INITIAL.value
    )
    
    # Simulate conversation through to slot search
    print("User: I need to reschedule")
    await workflow.process_turn(session, "I need to reschedule", {})
    
    print("User: Yes")
    await workflow.process_turn(session, "Yes", {})
    
    print("User: Tomorrow")
    await workflow.process_turn(session, "Tomorrow", {"date": "tomorrow"})
    
    print("User: 3am")  # Unrealistic time with no availability
    response = await workflow.process_turn(
        session, "3am", {"time": "3am"}
    )
    print(f"Agent: {response}\n")
    print(f"Workflow stage: {session.workflow_stage}")


if __name__ == "__main__":
    # Run main demo
    asyncio.run(main())
    
    # Run constraint violation demo
    asyncio.run(demo_constraint_violation())
    
    # Run no availability demo
    asyncio.run(demo_no_availability())
