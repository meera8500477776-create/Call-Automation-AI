"""Industry-specific adapters (FHIR, DMS)."""

from voice_agent.adapters.dms_adapter import DMSAdapter, ServiceAppointment, ServiceSlot
from voice_agent.adapters.fhir_adapter import (
    Appointment,
    AppointmentConstraints,
    CancellationResult,
    DateRange,
    FHIRAdapter,
    Slot,
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
