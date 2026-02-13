"""
Example usage of the Appointment Scheduling Workflow.

This example demonstrates how to use the AppointmentSchedulingWorkflow
to handle multi-turn conversations for scheduling appointments.

Requirements: 10.7, 10.8, 3.2, 3.3
"""

import asyncio
from datetime import datetime

from voice_agent.workflows.appointment_scheduling import AppointmentSchedulingWorkflow
from voice_agent.adapters.fhir_adapter import FHIRAdapter
from voice_agent.security.authentication import AuthenticationService
from voice_agent.core.session import Session


async def main():
    """Demonstrate appointment scheduling workflow."""
    
    # Initialize FHIR adapter
    auth_service = AuthenticationService()
    fhir_adapter = FHIRAdapter(
        base_url="https://fhir.example.com/api/FHIR/R4",
        auth_service=auth_service,
        client_id="example_client_id",
        client_secret="example_client_secret"
    )
    
    # Initialize workflow
    workflow = AppointmentSchedulingWorkflow(fhir_adapter=fhir_adapter)
    
    # Create a session
    session = Session(
        tenant_id="healthcare_clinic_001",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/12345"}
    )
    
    print("=== Appointment Scheduling Workflow Demo ===\n")
    
    # Turn 1: User initiates scheduling
    print("User: I'd like to schedule an appointment")
    response = await workflow.process_turn(
        session=session,
        user_transcript="I'd like to schedule an appointment",
        extracted_params={}
    )
    print(f"Agent: {response}\n")
    
    # Turn 2: User provides date
    print("User: Tomorrow")
    response = await workflow.process_turn(
        session=session,
        user_transcript="Tomorrow",
        extracted_params={"date": "tomorrow"}
    )
    print(f"Agent: {response}\n")
    
    # Turn 3: User provides time
    print("User: 2pm")
    response = await workflow.process_turn(
        session=session,
        user_transcript="2pm",
        extracted_params={"time": "2pm"}
    )
    print(f"Agent: {response}\n")
    
    # Turn 4: User provides reason
    print("User: Annual checkup")
    response = await workflow.process_turn(
        session=session,
        user_transcript="Annual checkup",
        extracted_params={"reason": "annual checkup"}
    )
    print(f"Agent: {response}\n")
    
    # Note: In a real scenario, the workflow would search for slots and present options
    # The user would then confirm, and the appointment would be booked
    
    print("\n=== Workflow State ===")
    print(f"Current stage: {session.workflow_stage}")
    print(f"Extracted parameters: {session.extracted_parameters}")
    print(f"Turn count: {session.get_turn_count()}")


async def demo_complete_flow():
    """Demonstrate a complete appointment scheduling flow with mock data."""
    
    print("\n=== Complete Flow Demo (with mocked FHIR adapter) ===\n")
    
    # Create a mock FHIR adapter for demonstration
    from unittest.mock import AsyncMock
    from voice_agent.adapters.fhir_adapter import Slot, Appointment
    from datetime import timedelta
    
    mock_fhir_adapter = AsyncMock()
    
    # Mock available slots
    tomorrow = datetime.now() + timedelta(days=1)
    mock_slots = [
        Slot(
            id="slot1",
            schedule_id="schedule1",
            start_time=tomorrow.replace(hour=14, minute=0),
            end_time=tomorrow.replace(hour=14, minute=30),
            status="free",
            service_type="General",
            practitioner_id="Practitioner/1"
        )
    ]
    mock_fhir_adapter.find_appointments.return_value = mock_slots
    
    # Mock successful booking
    mock_appointment = Appointment(
        id="appt123",
        patient_id="Patient/12345",
        practitioner_id="Practitioner/1",
        start_time=mock_slots[0].start_time,
        end_time=mock_slots[0].end_time,
        status="booked",
        service_type="General",
        is_cancelable=True,
        is_reschedulable=True,
        reason="Annual checkup"
    )
    mock_fhir_adapter.book_appointment.return_value = mock_appointment
    
    # Initialize workflow with mock adapter
    workflow = AppointmentSchedulingWorkflow(fhir_adapter=mock_fhir_adapter)
    
    # Create session
    session = Session(
        tenant_id="healthcare_clinic_001",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/12345"}
    )
    
    # Simulate complete conversation
    conversation = [
        ("I need to book an appointment", {}),
        ("Tomorrow", {"date": "tomorrow"}),
        ("2pm", {"time": "2pm"}),
        ("Annual checkup", {"reason": "annual checkup"}),
        ("Yes, please book it", {})
    ]
    
    for i, (user_input, params) in enumerate(conversation, 1):
        print(f"Turn {i}")
        print(f"User: {user_input}")
        response = await workflow.process_turn(
            session=session,
            user_transcript=user_input,
            extracted_params=params
        )
        print(f"Agent: {response}")
        print(f"Stage: {session.workflow_stage}\n")
    
    print("=== Final Session State ===")
    print(f"Workflow stage: {session.workflow_stage}")
    print(f"Extracted parameters: {session.extracted_parameters}")
    print(f"Total turns: {session.get_turn_count()}")


async def demo_error_handling():
    """Demonstrate error handling in the workflow."""
    
    print("\n=== Error Handling Demo ===\n")
    
    from unittest.mock import AsyncMock
    
    # Create a mock FHIR adapter that returns no slots
    mock_fhir_adapter = AsyncMock()
    mock_fhir_adapter.find_appointments.return_value = []  # No available slots
    
    workflow = AppointmentSchedulingWorkflow(fhir_adapter=mock_fhir_adapter)
    
    session = Session(
        tenant_id="healthcare_clinic_001",
        caller_id="+1234567890",
        metadata={"patient_id": "Patient/12345"}
    )
    
    # Fast-forward to searching stage
    session.workflow_stage = "searching_slots"
    session.metadata["workflow_params"] = type('obj', (object,), {
        'date': 'tomorrow',
        'time': '2pm',
        'patient_id': 'Patient/12345',
        'reason': None,
        'provider': None,
        'selected_slot': None,
        'available_slots': []
    })()
    
    print("User: (workflow searching for slots...)")
    response = await workflow.process_turn(
        session=session,
        user_transcript="",
        extracted_params={}
    )
    print(f"Agent: {response}")
    print(f"Stage: {session.workflow_stage}")
    print("\nNote: Workflow gracefully handles no availability scenario")


if __name__ == "__main__":
    # Run the basic demo
    asyncio.run(main())
    
    # Run the complete flow demo
    asyncio.run(demo_complete_flow())
    
    # Run the error handling demo
    asyncio.run(demo_error_handling())
