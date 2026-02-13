"""Industry-specific adapters (FHIR, DMS)."""

from voice_agent.adapters.fhir_adapter import (
    FHIRAdapter,
    Slot,
    Appointment,
    AppointmentConstraints,
    CancellationResult,
    DateRange
)

from voice_agent.adapters.dms_adapter import (
    DMSAdapter,
    ServiceSlot,
    ServiceAppointment
)

__all__ = [
    "FHIRAdapter",
    "Slot",
    "Appointment",
    "AppointmentConstraints",
    "CancellationResult",
    "DateRange",
    "DMSAdapter",
    "ServiceSlot",
    "ServiceAppointment"
]
