"""
DMS Adapter for Config-Driven Voice Agent Platform.

Provides Dealership Management System integration for automotive use cases,
supporting service appointment management including slot search and booking.

Requirements: 4.2, 4.6
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass
class DateRange:
    """Represents a date range for service appointment searches."""
    start: date
    end: date


@dataclass
class ServiceSlot:
    """
    Represents an available service appointment slot.
    
    Requirements: 4.6
    """
    id: str
    dealer_id: str
    start_time: datetime
    end_time: datetime
    service_type: str
    status: str  # "available", "booked", "unavailable"
    bay_number: str | None = None
    technician_id: str | None = None


@dataclass
class ServiceAppointment:
    """
    Represents a booked service appointment.
    
    Requirements: 4.6
    """
    id: str
    dealer_id: str
    vehicle_id: str
    vin: str
    start_time: datetime
    end_time: datetime
    service_type: str
    status: str  # "scheduled", "in_progress", "completed", "cancelled"
    bay_number: str | None = None
    technician_id: str | None = None
    notes: str | None = None


@dataclass
class MaintenanceQuote:
    """
    Represents a maintenance service quote.
    
    Requirements: 4.6
    """
    quote_id: str
    vehicle_id: str
    service_type: str
    labor_cost: float
    parts_cost: float
    total_cost: float
    estimated_duration_minutes: int
    parts_available: bool
    valid_until: datetime
    notes: str | None = None


@dataclass
class PartAvailability:
    """
    Represents parts availability information.
    
    Requirements: 4.6
    """
    part_number: str
    part_name: str
    quantity_available: int
    quantity_needed: int
    in_stock: bool
    estimated_arrival: datetime | None = None


@dataclass
class TriageResult:
    """
    Represents emergency triage assessment result.
    
    Requirements: 4.6
    """
    severity: str  # "critical", "urgent", "moderate", "low"
    safe_to_drive: bool
    recommended_action: str
    estimated_repair_time: int | None = None  # minutes
    requires_tow: bool = False
    priority_score: int = 0  # 0-100, higher is more urgent


class DMSAdapter:
    """
    DMS adapter for Dealership Management System integration.
    
    Supports automotive service scheduling including slot search, booking,
    VIN-based vehicle identification, maintenance quotes, emergency triage,
    and parts availability checking.
    
    Requirements:
    - 4.2: DMS integration for automotive industry
    - 4.6: Service scheduling, maintenance quotes, and emergency triage
    """

    # Service type classifications
    SERVICE_TYPES = {
        "oil_change": {"category": "maintenance", "typical_duration": 30},
        "tire_rotation": {"category": "maintenance", "typical_duration": 45},
        "diagnostics": {"category": "inspection", "typical_duration": 60},
        "brake_service": {"category": "repair", "typical_duration": 120},
        "transmission_service": {"category": "maintenance", "typical_duration": 90},
        "engine_repair": {"category": "repair", "typical_duration": 240},
        "battery_replacement": {"category": "maintenance", "typical_duration": 30},
        "alignment": {"category": "maintenance", "typical_duration": 60},
    }

    def __init__(
        self,
        base_url: str,
        dealer_id: str,
        api_key: str,
        timeout: int = 30
    ):
        """
        Initialize the DMS adapter.
        
        Args:
            base_url: Base URL of the DMS API (e.g., "https://api.dealership.com/v1")
            dealer_id: Dealership identifier
            api_key: API key for authentication
            timeout: HTTP request timeout in seconds (default: 30)
        """
        self.base_url = base_url.rstrip('/')
        self.dealer_id = dealer_id
        self.api_key = api_key
        self.timeout = timeout
        self._authenticated = False

    async def authenticate(self) -> bool:
        """
        Authenticate with the DMS API using API key.
        
        Requirements: 4.2
        
        Returns:
            True if authentication successful
            
        Raises:
            ValueError: If authentication fails
        """
        import aiohttp

        url = f"{self.base_url}/auth/validate"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url=url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status == 200:
                        self._authenticated = True
                        return True
                    else:
                        response_text = await response.text()
                        raise ValueError(
                            f"Authentication failed with status {response.status}: {response_text}"
                        )
            except ValueError:
                raise
            except aiohttp.ClientError as e:
                raise ValueError(f"Authentication request failed: {str(e)}")

    async def _ensure_authenticated(self) -> None:
        """Ensure we are authenticated before making requests."""
        if not self._authenticated:
            await self.authenticate()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Make an authenticated HTTP request to the DMS API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Optional request body data
            params: Optional query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            ValueError: If request fails
        """
        import aiohttp

        await self._ensure_authenticated()

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json"
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
                            f"DMS operation failed with status {response.status}: {response_text}"
                        )

                    if response_text:
                        return await response.json()
                    return {}
            except ValueError:
                raise
            except aiohttp.ClientError as e:
                raise ValueError(f"HTTP request failed: {str(e)}")

    def _parse_service_slot(self, slot_data: dict[str, Any]) -> ServiceSlot:
        """
        Parse a service slot from DMS API response.
        
        Requirements: 4.6
        
        Args:
            slot_data: Service slot data as dictionary
            
        Returns:
            Parsed ServiceSlot object
        """
        return ServiceSlot(
            id=slot_data.get("id", ""),
            dealer_id=slot_data.get("dealer_id", self.dealer_id),
            start_time=datetime.fromisoformat(slot_data.get("start_time", "").replace("Z", "+00:00")),
            end_time=datetime.fromisoformat(slot_data.get("end_time", "").replace("Z", "+00:00")),
            service_type=slot_data.get("service_type", ""),
            status=slot_data.get("status", "available"),
            bay_number=slot_data.get("bay_number"),
            technician_id=slot_data.get("technician_id")
        )

    def _parse_service_appointment(self, appointment_data: dict[str, Any]) -> ServiceAppointment:
        """
        Parse a service appointment from DMS API response.
        
        Requirements: 4.6
        
        Args:
            appointment_data: Service appointment data as dictionary
            
        Returns:
            Parsed ServiceAppointment object
        """
        return ServiceAppointment(
            id=appointment_data.get("id", ""),
            dealer_id=appointment_data.get("dealer_id", self.dealer_id),
            vehicle_id=appointment_data.get("vehicle_id", ""),
            vin=appointment_data.get("vin", ""),
            start_time=datetime.fromisoformat(appointment_data.get("start_time", "").replace("Z", "+00:00")),
            end_time=datetime.fromisoformat(appointment_data.get("end_time", "").replace("Z", "+00:00")),
            service_type=appointment_data.get("service_type", ""),
            status=appointment_data.get("status", "scheduled"),
            bay_number=appointment_data.get("bay_number"),
            technician_id=appointment_data.get("technician_id"),
            notes=appointment_data.get("notes")
        )

    async def find_service_slots(
        self,
        vehicle_id: str,
        service_type: str,
        date_range: DateRange
    ) -> list[ServiceSlot]:
        """
        Search for available service appointment slots.
        
        Requirements: 4.6
        
        Args:
            vehicle_id: Vehicle identifier (VIN or internal ID)
            service_type: Type of service (e.g., "oil_change", "tire_rotation", "diagnostics")
            date_range: Date range for service appointment search
            
        Returns:
            List of available ServiceSlot objects
            
        Raises:
            ValueError: If search operation fails
        """
        # Build search parameters
        params = {
            "dealer_id": self.dealer_id,
            "vehicle_id": vehicle_id,
            "service_type": service_type,
            "start_date": date_range.start.isoformat(),
            "end_date": date_range.end.isoformat(),
            "status": "available"
        }

        # Execute search request
        response = await self._make_request(
            method="GET",
            endpoint="service/slots",
            params=params
        )

        # Parse slots from response
        slots = []
        if "slots" in response:
            for slot_data in response["slots"]:
                slot = self._parse_service_slot(slot_data)
                # Only include slots within the requested date range
                if date_range.start <= slot.start_time.date() <= date_range.end:
                    slots.append(slot)

        # Sort by start time
        slots.sort(key=lambda s: s.start_time)

        return slots

    async def book_service(
        self,
        vehicle_id: str,
        slot_id: str,
        service_type: str,
        vin: str | None = None,
        notes: str | None = None
    ) -> ServiceAppointment:
        """
        Book a service appointment.
        
        Requirements: 4.6
        
        Args:
            vehicle_id: Vehicle identifier
            slot_id: Service slot identifier to book
            service_type: Type of service
            vin: Vehicle Identification Number (optional, for VIN-based identification)
            notes: Optional notes for the service appointment
            
        Returns:
            Created ServiceAppointment object
            
        Raises:
            ValueError: If booking operation fails or validation fails
        """
        # Validate required fields
        if not vehicle_id or not slot_id or not service_type:
            raise ValueError("Vehicle ID, Slot ID, and Service Type are required for booking")

        # Build appointment data
        appointment_data = {
            "dealer_id": self.dealer_id,
            "vehicle_id": vehicle_id,
            "slot_id": slot_id,
            "service_type": service_type,
            "status": "scheduled"
        }

        if vin:
            appointment_data["vin"] = vin

        if notes:
            appointment_data["notes"] = notes

        # Execute booking request
        response = await self._make_request(
            method="POST",
            endpoint="service/appointments",
            data=appointment_data
        )

        # Parse and return appointment
        if "appointment" in response:
            return self._parse_service_appointment(response["appointment"])

        raise ValueError("Booking operation did not return a valid service appointment")

    def classify_service_type(self, service_type: str) -> dict[str, Any]:
        """
        Classify a service type and return its metadata.
        
        Requirements: 4.6
        
        Args:
            service_type: Type of service to classify
            
        Returns:
            Dictionary with service classification metadata including category and typical duration
        """
        if service_type in self.SERVICE_TYPES:
            return self.SERVICE_TYPES[service_type].copy()

        # Default classification for unknown service types
        return {
            "category": "other",
            "typical_duration": 60
        }

    async def get_maintenance_quote(
        self,
        vehicle_id: str,
        service_type: str,
        vin: str | None = None
    ) -> MaintenanceQuote:
        """
        Retrieve a maintenance service quote with pricing and parts availability.
        
        Requirements: 4.6
        
        Args:
            vehicle_id: Vehicle identifier
            service_type: Type of service (e.g., "oil_change", "tire_rotation", "diagnostics")
            vin: Vehicle Identification Number (optional)
            
        Returns:
            MaintenanceQuote object with pricing and availability information
            
        Raises:
            ValueError: If quote retrieval fails
        """
        # Build quote request parameters
        params = {
            "dealer_id": self.dealer_id,
            "vehicle_id": vehicle_id,
            "service_type": service_type
        }

        if vin:
            params["vin"] = vin

        # Execute quote request
        response = await self._make_request(
            method="GET",
            endpoint="service/quotes",
            params=params
        )

        # Parse and return quote
        if "quote" in response:
            quote_data = response["quote"]
            return MaintenanceQuote(
                quote_id=quote_data.get("quote_id", ""),
                vehicle_id=quote_data.get("vehicle_id", vehicle_id),
                service_type=quote_data.get("service_type", service_type),
                labor_cost=float(quote_data.get("labor_cost", 0.0)),
                parts_cost=float(quote_data.get("parts_cost", 0.0)),
                total_cost=float(quote_data.get("total_cost", 0.0)),
                estimated_duration_minutes=int(quote_data.get("estimated_duration_minutes", 60)),
                parts_available=bool(quote_data.get("parts_available", True)),
                valid_until=datetime.fromisoformat(
                    quote_data.get("valid_until", "").replace("Z", "+00:00")
                ),
                notes=quote_data.get("notes")
            )

        raise ValueError("Quote retrieval did not return a valid maintenance quote")

    async def check_parts_availability(
        self,
        vehicle_id: str,
        service_type: str,
        vin: str | None = None
    ) -> list[PartAvailability]:
        """
        Check parts availability for a specific service.
        
        Requirements: 4.6
        
        Args:
            vehicle_id: Vehicle identifier
            service_type: Type of service
            vin: Vehicle Identification Number (optional)
            
        Returns:
            List of PartAvailability objects
            
        Raises:
            ValueError: If parts check fails
        """
        # Build parts check parameters
        params = {
            "dealer_id": self.dealer_id,
            "vehicle_id": vehicle_id,
            "service_type": service_type
        }

        if vin:
            params["vin"] = vin

        # Execute parts availability request
        response = await self._make_request(
            method="GET",
            endpoint="service/parts/availability",
            params=params
        )

        # Parse parts availability
        parts = []
        if "parts" in response:
            for part_data in response["parts"]:
                part = PartAvailability(
                    part_number=part_data.get("part_number", ""),
                    part_name=part_data.get("part_name", ""),
                    quantity_available=int(part_data.get("quantity_available", 0)),
                    quantity_needed=int(part_data.get("quantity_needed", 1)),
                    in_stock=bool(part_data.get("in_stock", False)),
                    estimated_arrival=(
                        datetime.fromisoformat(part_data.get("estimated_arrival", "").replace("Z", "+00:00"))
                        if part_data.get("estimated_arrival")
                        else None
                    )
                )
                parts.append(part)

        return parts

    async def triage_emergency(
        self,
        vehicle_id: str,
        issue_description: str,
        vin: str | None = None
    ) -> TriageResult:
        """
        Assess the severity of a vehicle issue and provide emergency triage.
        
        Requirements: 4.6
        
        Args:
            vehicle_id: Vehicle identifier
            issue_description: Description of the vehicle issue
            vin: Vehicle Identification Number (optional)
            
        Returns:
            TriageResult object with severity assessment and recommendations
            
        Raises:
            ValueError: If triage assessment fails
        """
        # Build triage request data
        triage_data = {
            "dealer_id": self.dealer_id,
            "vehicle_id": vehicle_id,
            "issue_description": issue_description
        }

        if vin:
            triage_data["vin"] = vin

        # Execute triage request
        response = await self._make_request(
            method="POST",
            endpoint="service/triage",
            data=triage_data
        )

        # Parse and return triage result
        if "triage" in response:
            triage_data = response["triage"]
            return TriageResult(
                severity=triage_data.get("severity", "moderate"),
                safe_to_drive=bool(triage_data.get("safe_to_drive", True)),
                recommended_action=triage_data.get("recommended_action", "Schedule service appointment"),
                estimated_repair_time=triage_data.get("estimated_repair_time"),
                requires_tow=bool(triage_data.get("requires_tow", False)),
                priority_score=int(triage_data.get("priority_score", 50))
            )

        raise ValueError("Triage assessment did not return a valid result")
