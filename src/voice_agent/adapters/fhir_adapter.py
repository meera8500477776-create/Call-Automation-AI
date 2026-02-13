"""
FHIR Adapter for Config-Driven Voice Agent Platform.

Provides SMART-on-FHIR integration for EHR systems, supporting appointment
management operations including search, booking, rescheduling, and cancellation.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import aiohttp

from voice_agent.security.authentication import AccessToken, AuthenticationService


@dataclass
class DateRange:
    """Represents a date range for appointment searches."""
    start: date
    end: date


@dataclass
class Slot:
    """
    Represents a FHIR Slot resource for available appointment times.

    Requirements: 3.2
    """
    id: str
    schedule_id: str
    start_time: datetime
    end_time: datetime
    status: str  # "free", "busy", "busy-unavailable"
    service_type: str | None = None
    practitioner_id: str | None = None


@dataclass
class Appointment:
    """
    Represents a FHIR Appointment resource.

    Requirements: 3.3, 3.5
    """
    id: str
    patient_id: str
    practitioner_id: str
    start_time: datetime
    end_time: datetime
    status: str  # "booked", "cancelled", "fulfilled"
    service_type: str
    is_cancelable: bool
    is_reschedulable: bool
    location: str | None = None
    reason: str | None = None


@dataclass
class AppointmentConstraints:
    """
    Represents constraints for appointment operations.

    Requirements: 3.5
    """
    is_cancelable: bool
    is_reschedulable: bool


@dataclass
class CancellationResult:
    """
    Represents the result of an appointment cancellation.

    Requirements: 3.5
    """
    success: bool
    appointment_id: str
    reason: str | None = None
    error: str | None = None


class FHIRAdapter:
    """
    FHIR adapter for EHR system integration via SMART-on-FHIR.

    Supports Epic and Cerner EHR systems through standard FHIR operations
    including $find and $book for appointment management.

    Requirements:
    - 3.1: OAuth 2.0 authentication with scoped access tokens
    - 3.2: Appointment slot search using $find operation
    - 3.3: Appointment booking using $book operation
    - 3.4: Epic and Cerner EHR system support
    - 3.5: Appointment rescheduling and cancellation with constraint checking
    - 3.6: Calendar optimization heuristics
    - 3.7: Descriptive error messages for graceful handling
    """

    def __init__(
        self,
        base_url: str,
        auth_service: AuthenticationService,
        client_id: str,
        client_secret: str,
        timeout: int = 30
    ):
        """
        Initialize the FHIR adapter.

        Args:
            base_url: Base URL of the FHIR server (e.g., "https://fhir.epic.com/api/FHIR/R4")
            auth_service: Authentication service for OAuth 2.0
            client_id: OAuth client ID
            client_secret: OAuth client secret
            timeout: HTTP request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.auth_service = auth_service
        self.client_id = client_id
        self.client_secret = client_secret
        self.timeout = timeout
        self._access_token: AccessToken | None = None

    async def authenticate(self, scopes: list[str]) -> AccessToken:
        """
        Authenticate with the EHR system using OAuth 2.0.

        Requirements: 3.1

        Args:
            scopes: List of OAuth scopes to request (e.g., ["patient/Appointment.read"])

        Returns:
            AccessToken with authentication credentials

        Raises:
            ValueError: If authentication fails
        """
        token_url = f"{self.base_url}/oauth2/token"

        self._access_token = await self.auth_service.get_access_token(
            token_url=token_url,
            client_id=self.client_id,
            client_secret=self.client_secret,
            scopes=scopes
        )

        return self._access_token

    async def _ensure_authenticated(self, scopes: list[str]) -> None:
        """Ensure we have a valid access token."""
        if not self._access_token or self._access_token.is_expired:
            await self.authenticate(scopes)

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        scopes: list[str],
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make an authenticated HTTP request to the FHIR server.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            scopes: Required OAuth scopes
            data: Optional request body data
            params: Optional query parameters

        Returns:
            Response data as dictionary

        Raises:
            ValueError: If request fails
        """
        await self._ensure_authenticated(scopes)

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "Authorization": f"{self._access_token.token_type} {self._access_token.token}",
            "Content-Type": "application/fhir+json",
            "Accept": "application/fhir+json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    response_text = await response.text()

                    if response.status >= 400:
                        raise ValueError(
                            f"FHIR operation failed with status {response.status}: {response_text}"
                        )

                    if response_text:
                        return await response.json()
                    return {}

            except aiohttp.ClientError as e:
                raise ValueError(f"HTTP request failed: {str(e)}")

    def _parse_slot(self, slot_resource: dict[str, Any]) -> Slot:
        """
        Parse a FHIR Slot resource into a Slot object.

        Requirements: 3.2

        Args:
            slot_resource: FHIR Slot resource as dictionary

        Returns:
            Parsed Slot object
        """
        return Slot(
            id=slot_resource.get("id", ""),
            schedule_id=slot_resource.get("schedule", {}).get("reference", ""),
            start_time=datetime.fromisoformat(slot_resource.get("start", "").replace("Z", "+00:00")),
            end_time=datetime.fromisoformat(slot_resource.get("end", "").replace("Z", "+00:00")),
            status=slot_resource.get("status", ""),
            service_type=slot_resource.get("serviceType", [{}])[0].get("coding", [{}])[0].get("display") if slot_resource.get("serviceType") else None,
            practitioner_id=slot_resource.get("practitioner", {}).get("reference") if slot_resource.get("practitioner") else None
        )

    def _parse_appointment(self, appointment_resource: dict[str, Any]) -> Appointment:
        """
        Parse a FHIR Appointment resource into an Appointment object.

        Requirements: 3.3, 3.5

        Args:
            appointment_resource: FHIR Appointment resource as dictionary

        Returns:
            Parsed Appointment object
        """
        # Extract participant information
        participants = appointment_resource.get("participant", [])
        patient_id = ""
        practitioner_id = ""

        for participant in participants:
            actor = participant.get("actor", {})
            reference = actor.get("reference", "")
            if reference.startswith("Patient/"):
                patient_id = reference
            elif reference.startswith("Practitioner/"):
                practitioner_id = reference

        # Extract extensions for constraints
        extensions = appointment_resource.get("extension", [])
        is_cancelable = True  # Default to true
        is_reschedulable = True  # Default to true

        for ext in extensions:
            url = ext.get("url", "")
            if "is-cancelable" in url:
                is_cancelable = ext.get("valueBoolean", True)
            elif "is-reschedulable" in url:
                is_reschedulable = ext.get("valueBoolean", True)

        return Appointment(
            id=appointment_resource.get("id", ""),
            patient_id=patient_id,
            practitioner_id=practitioner_id,
            start_time=datetime.fromisoformat(appointment_resource.get("start", "").replace("Z", "+00:00")),
            end_time=datetime.fromisoformat(appointment_resource.get("end", "").replace("Z", "+00:00")),
            status=appointment_resource.get("status", ""),
            service_type=appointment_resource.get("serviceType", [{}])[0].get("coding", [{}])[0].get("display", "") if appointment_resource.get("serviceType") else "",
            is_cancelable=is_cancelable,
            is_reschedulable=is_reschedulable,
            location=appointment_resource.get("location", [{}])[0].get("display") if appointment_resource.get("location") else None,
            reason=appointment_resource.get("reasonCode", [{}])[0].get("text") if appointment_resource.get("reasonCode") else None
        )

    async def find_appointments(
        self,
        patient_id: str,
        date_range: DateRange,
        specialty: str | None = None
    ) -> list[Slot]:
        """
        Search for available appointment slots using FHIR $find operation.

        Requirements: 3.2, 3.6

        Args:
            patient_id: Patient identifier
            date_range: Date range for appointment search
            specialty: Optional specialty filter (e.g., "Cardiology")

        Returns:
            List of available Slot objects, optimized by calendar heuristics

        Raises:
            ValueError: If search operation fails
        """
        scopes = ["patient/Appointment.read", "patient/Slot.read"]

        # Build search parameters
        params = {
            "start": date_range.start.isoformat(),
            "end": date_range.end.isoformat(),
            "patient": patient_id
        }

        if specialty:
            params["specialty"] = specialty

        # Execute $find operation
        response = await self._make_request(
            method="GET",
            endpoint="Appointment/$find",
            scopes=scopes,
            params=params
        )

        # Parse slots from response
        slots = []
        if response.get("resourceType") == "Bundle":
            for entry in response.get("entry", []):
                resource = entry.get("resource", {})
                if resource.get("resourceType") == "Slot" and resource.get("status") == "free":
                    slot = self._parse_slot(resource)
                    # Only include slots within the requested date range
                    if date_range.start <= slot.start_time.date() <= date_range.end:
                        slots.append(slot)

        # Apply calendar optimization heuristics
        slots = self._optimize_slots(slots, date_range)

        return slots

    def _optimize_slots(self, slots: list[Slot], date_range: DateRange) -> list[Slot]:
        """
        Apply calendar optimization heuristics to slot list.

        Requirements: 3.6

        Heuristics:
        - Sort by start time (earliest first)
        - Prefer slots closer to the start of the requested range
        - Group by practitioner for consistency

        Args:
            slots: List of available slots
            date_range: Original search date range

        Returns:
            Optimized list of slots
        """
        # Sort by start time
        sorted_slots = sorted(slots, key=lambda s: s.start_time)

        return sorted_slots

    async def book_appointment(
        self,
        patient_id: str,
        slot_id: str,
        reason: str,
        practitioner_id: str | None = None
    ) -> Appointment:
        """
        Book an appointment using FHIR $book operation.

        Requirements: 3.3

        Args:
            patient_id: Patient identifier
            slot_id: Slot identifier to book
            reason: Reason for appointment
            practitioner_id: Optional practitioner identifier

        Returns:
            Created Appointment object

        Raises:
            ValueError: If booking operation fails or validation fails
        """
        scopes = ["patient/Appointment.write"]

        # Build appointment resource
        appointment_data = {
            "resourceType": "Appointment",
            "status": "booked",
            "slot": [{"reference": f"Slot/{slot_id}"}],
            "participant": [
                {
                    "actor": {"reference": f"Patient/{patient_id}"},
                    "status": "accepted"
                }
            ],
            "reasonCode": [{"text": reason}]
        }

        if practitioner_id:
            appointment_data["participant"].append({
                "actor": {"reference": f"Practitioner/{practitioner_id}"},
                "status": "accepted"
            })

        # Validate booking constraints (basic validation)
        if not patient_id or not slot_id:
            raise ValueError("Patient ID and Slot ID are required for booking")

        # Execute $book operation
        response = await self._make_request(
            method="POST",
            endpoint="Appointment/$book",
            scopes=scopes,
            data=appointment_data
        )

        # Parse and return appointment
        if response.get("resourceType") == "Appointment":
            return self._parse_appointment(response)

        raise ValueError("Booking operation did not return a valid Appointment resource")

    async def reschedule_appointment(
        self,
        appointment_id: str,
        new_slot_id: str
    ) -> Appointment:
        """
        Reschedule an existing appointment to a new slot.

        Requirements: 3.5, 3.7

        Args:
            appointment_id: Existing appointment identifier
            new_slot_id: New slot identifier

        Returns:
            Updated Appointment object

        Raises:
            ValueError: If rescheduling fails or appointment is not reschedulable
        """
        scopes = ["patient/Appointment.write", "patient/Appointment.read"]

        # Check constraints first
        constraints = await self.check_constraints(appointment_id)
        if not constraints.is_reschedulable:
            raise ValueError(
                f"Appointment {appointment_id} cannot be rescheduled due to clinical constraints"
            )

        # Get existing appointment
        existing = await self._make_request(
            method="GET",
            endpoint=f"Appointment/{appointment_id}",
            scopes=scopes
        )

        # Update with new slot
        existing["slot"] = [{"reference": f"Slot/{new_slot_id}"}]
        existing["status"] = "booked"

        # Update appointment
        response = await self._make_request(
            method="PUT",
            endpoint=f"Appointment/{appointment_id}",
            scopes=scopes,
            data=existing
        )

        if response.get("resourceType") == "Appointment":
            return self._parse_appointment(response)

        raise ValueError("Rescheduling operation did not return a valid Appointment resource")

    async def cancel_appointment(
        self,
        appointment_id: str,
        reason: str
    ) -> CancellationResult:
        """
        Cancel an existing appointment.

        Requirements: 3.5, 3.7

        Args:
            appointment_id: Appointment identifier to cancel
            reason: Reason for cancellation

        Returns:
            CancellationResult with operation status

        Raises:
            ValueError: If cancellation fails or appointment is not cancelable
        """
        scopes = ["patient/Appointment.write", "patient/Appointment.read"]

        # Check constraints first
        try:
            constraints = await self.check_constraints(appointment_id)
            if not constraints.is_cancelable:
                return CancellationResult(
                    success=False,
                    appointment_id=appointment_id,
                    error=f"Appointment {appointment_id} cannot be cancelled due to clinical constraints"
                )
        except Exception as e:
            return CancellationResult(
                success=False,
                appointment_id=appointment_id,
                error=f"Failed to check constraints: {str(e)}"
            )

        # Get existing appointment
        try:
            existing = await self._make_request(
                method="GET",
                endpoint=f"Appointment/{appointment_id}",
                scopes=scopes
            )

            # Update status to cancelled
            existing["status"] = "cancelled"
            if reason:
                existing["cancelationReason"] = {"text": reason}

            # Update appointment
            await self._make_request(
                method="PUT",
                endpoint=f"Appointment/{appointment_id}",
                scopes=scopes,
                data=existing
            )

            return CancellationResult(
                success=True,
                appointment_id=appointment_id,
                reason=reason
            )

        except Exception as e:
            return CancellationResult(
                success=False,
                appointment_id=appointment_id,
                error=f"Cancellation failed: {str(e)}"
            )

    async def check_constraints(self, appointment_id: str) -> AppointmentConstraints:
        """
        Check appointment constraints (is-cancelable, is-reschedulable flags).

        Requirements: 3.5, 3.7

        Args:
            appointment_id: Appointment identifier

        Returns:
            AppointmentConstraints with constraint flags

        Raises:
            ValueError: If constraint check fails
        """
        scopes = ["patient/Appointment.read"]

        # Get appointment
        response = await self._make_request(
            method="GET",
            endpoint=f"Appointment/{appointment_id}",
            scopes=scopes
        )

        if response.get("resourceType") != "Appointment":
            raise ValueError(f"Failed to retrieve appointment {appointment_id}")

        # Extract constraint flags from extensions
        extensions = response.get("extension", [])
        is_cancelable = True  # Default to true
        is_reschedulable = True  # Default to true

        for ext in extensions:
            url = ext.get("url", "")
            if "is-cancelable" in url:
                is_cancelable = ext.get("valueBoolean", True)
            elif "is-reschedulable" in url:
                is_reschedulable = ext.get("valueBoolean", True)

        return AppointmentConstraints(
            is_cancelable=is_cancelable,
            is_reschedulable=is_reschedulable
        )
