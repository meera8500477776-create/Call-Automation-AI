"""
Unit tests for DMS Adapter.

Tests API key authentication, service slot search, service booking,
and VIN-based vehicle identification.

Requirements: 4.2, 4.6
"""

import pytest
from datetime import datetime, date, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from voice_agent.adapters.dms_adapter import (
    DMSAdapter,
    ServiceSlot,
    ServiceAppointment,
    DateRange,
    MaintenanceQuote,
    PartAvailability,
    TriageResult
)


def create_mock_aiohttp_session(responses):
    """
    Helper to create a properly mocked aiohttp session.
    
    Args:
        responses: List of (status, json_data, text_data) tuples or a single tuple
    """
    if not isinstance(responses, list):
        responses = [responses]
    
    # Track which response we're on
    response_index = [0]
    
    def get_next_response(*args, **kwargs):
        """Get the next response from the list."""
        idx = response_index[0]
        if idx >= len(responses):
            # Reuse last response if we run out
            idx = len(responses) - 1
        else:
            response_index[0] += 1
        
        status, json_data, text_data = responses[idx]
        
        mock_response = AsyncMock()
        mock_response.status = status
        
        # For error responses (status >= 400), text() is called
        # For success responses, json() is called
        if status >= 400:
            # Ensure text_data is a string
            text_str = text_data if isinstance(text_data, str) else str(text_data)
            mock_response.text = AsyncMock(return_value=text_str)
            mock_response.json = AsyncMock(return_value={})
        else:
            # Success response - return json_data
            mock_response.json = AsyncMock(return_value=json_data if json_data is not None else {})
            mock_response.text = AsyncMock(return_value=str(json_data) if json_data else "")
        
        # Create context manager for the response
        # __aexit__ must return False or None to not swallow exceptions
        async def aexit_func(*args):
            return False
        
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_response
        mock_context.__aexit__ = aexit_func
        return mock_context
    
    # Create the session mock
    async def session_aexit(*args):
        return False
    
    mock_session = MagicMock()
    mock_session.request.side_effect = get_next_response
    mock_session.get.side_effect = get_next_response
    mock_session.post.side_effect = get_next_response
    
    # Make the session itself a context manager
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__ = session_aexit
    
    return mock_session


@pytest.fixture
def dms_adapter():
    """Create a DMS adapter instance."""
    return DMSAdapter(
        base_url="https://api.dealership.com/v1",
        dealer_id="dealer-123",
        api_key="test_api_key",
        timeout=30
    )


@pytest.fixture
def mock_service_slots_response():
    """Create a mock DMS service slots response."""
    return {
        "slots": [
            {
                "id": "slot-001",
                "dealer_id": "dealer-123",
                "start_time": "2024-03-15T09:00:00Z",
                "end_time": "2024-03-15T10:00:00Z",
                "service_type": "oil_change",
                "status": "available",
                "bay_number": "Bay-1",
                "technician_id": "tech-456"
            },
            {
                "id": "slot-002",
                "dealer_id": "dealer-123",
                "start_time": "2024-03-15T14:00:00Z",
                "end_time": "2024-03-15T15:00:00Z",
                "service_type": "oil_change",
                "status": "available",
                "bay_number": "Bay-2",
                "technician_id": "tech-789"
            }
        ]
    }


@pytest.fixture
def mock_service_appointment_response():
    """Create a mock DMS service appointment response."""
    return {
        "appointment": {
            "id": "appt-001",
            "dealer_id": "dealer-123",
            "vehicle_id": "vehicle-456",
            "vin": "1HGBH41JXMN109186",
            "start_time": "2024-03-15T09:00:00Z",
            "end_time": "2024-03-15T10:00:00Z",
            "service_type": "oil_change",
            "status": "scheduled",
            "bay_number": "Bay-1",
            "technician_id": "tech-456",
            "notes": "Customer requested synthetic oil"
        }
    }


class TestDMSAdapterAuthentication:
    """Test API key authentication flow."""
    
    @pytest.mark.asyncio
    async def test_authenticate_success(self, dms_adapter):
        """Test successful API key authentication."""
        # Arrange
        mock_session = create_mock_aiohttp_session([(200, {}, "json")])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            result = await dms_adapter.authenticate()
        
        # Assert
        assert result is True
        assert dms_adapter._authenticated is True
    
    @pytest.mark.asyncio
    async def test_authenticate_failure(self, dms_adapter):
        """Test authentication failure handling."""
        # Arrange - create a fresh mock for this test
        mock_session = create_mock_aiohttp_session([(401, {}, "Unauthorized")])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act & Assert
            try:
                result = await dms_adapter.authenticate()
                # If we get here, check what happened
                pytest.fail(f"Expected ValueError but got result: {result}, authenticated: {dms_adapter._authenticated}")
            except ValueError as e:
                # This is expected
                assert "Authentication failed" in str(e)
                assert dms_adapter._authenticated is False
            except Exception as e:
                pytest.fail(f"Got unexpected exception: {type(e).__name__}: {e}")
    
    @pytest.mark.asyncio
    async def test_ensure_authenticated_calls_authenticate(self, dms_adapter):
        """Test that _ensure_authenticated calls authenticate when not authenticated."""
        # Arrange
        assert dms_adapter._authenticated is False
        mock_session = create_mock_aiohttp_session([(200, {}, "json")])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            await dms_adapter._ensure_authenticated()
        
        # Assert
        assert dms_adapter._authenticated is True


class TestDMSAdapterFindServiceSlots:
    """Test service slot search functionality."""
    
    @pytest.mark.asyncio
    async def test_find_service_slots_success(
        self,
        dms_adapter,
        mock_service_slots_response
    ):
        """Test successful service slot search."""
        # Arrange
        date_range = DateRange(
            start=date(2024, 3, 15),
            end=date(2024, 3, 20)
        )
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_service_slots_response, "json")  # find_service_slots
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            slots = await dms_adapter.find_service_slots(
                vehicle_id="vehicle-456",
                service_type="oil_change",
                date_range=date_range
            )
        
        # Assert
        assert len(slots) == 2
        assert all(isinstance(slot, ServiceSlot) for slot in slots)
        assert slots[0].id == "slot-001"
        assert slots[0].service_type == "oil_change"
        assert slots[0].status == "available"
        assert slots[0].bay_number == "Bay-1"
        assert slots[1].id == "slot-002"
    
    @pytest.mark.asyncio
    async def test_find_service_slots_with_date_filtering(self, dms_adapter):
        """Test that returned slots are within requested date range."""
        # Arrange
        date_range = DateRange(
            start=date(2024, 3, 15),
            end=date(2024, 3, 16)
        )
        
        # Response includes slots both inside and outside the range
        response_with_out_of_range = {
            "slots": [
                {
                    "id": "slot-in-range",
                    "dealer_id": "dealer-123",
                    "start_time": "2024-03-15T09:00:00Z",
                    "end_time": "2024-03-15T10:00:00Z",
                    "service_type": "oil_change",
                    "status": "available"
                },
                {
                    "id": "slot-out-of-range",
                    "dealer_id": "dealer-123",
                    "start_time": "2024-03-20T09:00:00Z",
                    "end_time": "2024-03-20T10:00:00Z",
                    "service_type": "oil_change",
                    "status": "available"
                }
            ]
        }
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, response_with_out_of_range, "json")  # find_service_slots
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            slots = await dms_adapter.find_service_slots(
                vehicle_id="vehicle-456",
                service_type="oil_change",
                date_range=date_range
            )
        
        # Assert - only slot within range should be returned
        assert len(slots) == 1
        assert slots[0].id == "slot-in-range"
    
    @pytest.mark.asyncio
    async def test_find_service_slots_sorted_by_time(self, dms_adapter):
        """Test that returned slots are sorted by start time."""
        # Arrange
        date_range = DateRange(
            start=date(2024, 3, 15),
            end=date(2024, 3, 20)
        )
        
        # Response with slots in non-chronological order
        unsorted_response = {
            "slots": [
                {
                    "id": "slot-afternoon",
                    "dealer_id": "dealer-123",
                    "start_time": "2024-03-15T14:00:00Z",
                    "end_time": "2024-03-15T15:00:00Z",
                    "service_type": "oil_change",
                    "status": "available"
                },
                {
                    "id": "slot-morning",
                    "dealer_id": "dealer-123",
                    "start_time": "2024-03-15T09:00:00Z",
                    "end_time": "2024-03-15T10:00:00Z",
                    "service_type": "oil_change",
                    "status": "available"
                }
            ]
        }
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, unsorted_response, "json")  # find_service_slots
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            slots = await dms_adapter.find_service_slots(
                vehicle_id="vehicle-456",
                service_type="oil_change",
                date_range=date_range
            )
        
        # Assert - slots should be sorted by start time
        assert len(slots) == 2
        assert slots[0].id == "slot-morning"
        assert slots[1].id == "slot-afternoon"
    
    @pytest.mark.asyncio
    async def test_find_service_slots_error_handling(self, dms_adapter):
        """Test error handling for failed search operations."""
        # Arrange
        date_range = DateRange(
            start=date(2024, 3, 15),
            end=date(2024, 3, 20)
        )
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (500, {}, "Internal server error")  # find_service_slots fails
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act & Assert
            with pytest.raises(ValueError, match="DMS operation failed"):
                await dms_adapter.find_service_slots(
                    vehicle_id="vehicle-456",
                    service_type="oil_change",
                    date_range=date_range
                )


class TestDMSAdapterBookService:
    """Test service booking functionality."""
    
    @pytest.mark.asyncio
    async def test_book_service_success(
        self,
        dms_adapter,
        mock_service_appointment_response
    ):
        """Test successful service booking."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_service_appointment_response, "json")  # book_service
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            appointment = await dms_adapter.book_service(
                vehicle_id="vehicle-456",
                slot_id="slot-001",
                service_type="oil_change",
                vin="1HGBH41JXMN109186",
                notes="Customer requested synthetic oil"
            )
        
        # Assert
        assert isinstance(appointment, ServiceAppointment)
        assert appointment.id == "appt-001"
        assert appointment.vehicle_id == "vehicle-456"
        assert appointment.vin == "1HGBH41JXMN109186"
        assert appointment.service_type == "oil_change"
        assert appointment.status == "scheduled"
        assert appointment.notes == "Customer requested synthetic oil"
    
    @pytest.mark.asyncio
    async def test_book_service_with_vin(
        self,
        dms_adapter,
        mock_service_appointment_response
    ):
        """Test service booking with VIN-based vehicle identification."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_service_appointment_response, "json")  # book_service
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            appointment = await dms_adapter.book_service(
                vehicle_id="vehicle-456",
                slot_id="slot-001",
                service_type="oil_change",
                vin="1HGBH41JXMN109186"
            )
        
        # Assert
        assert appointment.vin == "1HGBH41JXMN109186"
    
    @pytest.mark.asyncio
    async def test_book_service_validation_error(self, dms_adapter):
        """Test booking validation for missing required fields."""
        # Act & Assert - missing vehicle_id
        with pytest.raises(ValueError, match="Vehicle ID, Slot ID, and Service Type are required"):
            await dms_adapter.book_service(
                vehicle_id="",
                slot_id="slot-001",
                service_type="oil_change"
            )
        
        # Act & Assert - missing slot_id
        with pytest.raises(ValueError, match="Vehicle ID, Slot ID, and Service Type are required"):
            await dms_adapter.book_service(
                vehicle_id="vehicle-456",
                slot_id="",
                service_type="oil_change"
            )
        
        # Act & Assert - missing service_type
        with pytest.raises(ValueError, match="Vehicle ID, Slot ID, and Service Type are required"):
            await dms_adapter.book_service(
                vehicle_id="vehicle-456",
                slot_id="slot-001",
                service_type=""
            )
    
    @pytest.mark.asyncio
    async def test_book_service_error_handling(self, dms_adapter):
        """Test error handling for failed booking operations."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (400, {}, "Slot not available")  # book_service fails
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act & Assert
            with pytest.raises(ValueError, match="DMS operation failed"):
                await dms_adapter.book_service(
                    vehicle_id="vehicle-456",
                    slot_id="slot-001",
                    service_type="oil_change"
                )
    
    @pytest.mark.asyncio
    async def test_book_service_invalid_response(self, dms_adapter):
        """Test handling of invalid booking response."""
        # Arrange - response without 'appointment' key
        invalid_response = {"error": "Something went wrong"}
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, invalid_response, "json")  # book_service with invalid response
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act & Assert
            with pytest.raises(ValueError, match="Booking operation did not return a valid service appointment"):
                await dms_adapter.book_service(
                    vehicle_id="vehicle-456",
                    slot_id="slot-001",
                    service_type="oil_change"
                )


class TestDMSAdapterServiceTypes:
    """Test various service type scenarios."""
    
    @pytest.mark.asyncio
    async def test_different_service_types(self, dms_adapter):
        """Test booking different types of services."""
        service_types = ["oil_change", "tire_rotation", "diagnostics", "brake_service"]
        
        for service_type in service_types:
            # Arrange - create a fresh adapter and mock for each service type
            adapter = DMSAdapter(
                base_url="https://api.dealership.com/v1",
                dealer_id="dealer-123",
                api_key="test_api_key",
                timeout=30
            )
            
            response = {
                "appointment": {
                    "id": f"appt-{service_type}",
                    "dealer_id": "dealer-123",
                    "vehicle_id": "vehicle-456",
                    "vin": "1HGBH41JXMN109186",
                    "start_time": "2024-03-15T09:00:00Z",
                    "end_time": "2024-03-15T10:00:00Z",
                    "service_type": service_type,
                    "status": "scheduled"
                }
            }
            
            mock_session = create_mock_aiohttp_session([
                (200, {}, "json"),  # authenticate
                (200, response, "json")  # book_service
            ])
            
            with patch("aiohttp.ClientSession", return_value=mock_session):
                # Act
                appointment = await adapter.book_service(
                    vehicle_id="vehicle-456",
                    slot_id="slot-001",
                    service_type=service_type
                )
            
            # Assert
            assert appointment.service_type == service_type



class TestDMSAdapterServiceTypeClassification:
    """Test service type classification functionality."""
    
    def test_classify_known_service_types(self, dms_adapter):
        """Test classification of known service types."""
        # Test oil change
        oil_change = dms_adapter.classify_service_type("oil_change")
        assert oil_change["category"] == "maintenance"
        assert oil_change["typical_duration"] == 30
        
        # Test tire rotation
        tire_rotation = dms_adapter.classify_service_type("tire_rotation")
        assert tire_rotation["category"] == "maintenance"
        assert tire_rotation["typical_duration"] == 45
        
        # Test diagnostics
        diagnostics = dms_adapter.classify_service_type("diagnostics")
        assert diagnostics["category"] == "inspection"
        assert diagnostics["typical_duration"] == 60
        
        # Test brake service
        brake_service = dms_adapter.classify_service_type("brake_service")
        assert brake_service["category"] == "repair"
        assert brake_service["typical_duration"] == 120
    
    def test_classify_unknown_service_type(self, dms_adapter):
        """Test classification of unknown service types returns default."""
        unknown = dms_adapter.classify_service_type("unknown_service")
        assert unknown["category"] == "other"
        assert unknown["typical_duration"] == 60


class TestDMSAdapterMaintenanceQuote:
    """Test maintenance quote retrieval functionality."""
    
    @pytest.fixture
    def mock_quote_response(self):
        """Create a mock maintenance quote response."""
        return {
            "quote": {
                "quote_id": "quote-001",
                "vehicle_id": "vehicle-456",
                "service_type": "oil_change",
                "labor_cost": 45.00,
                "parts_cost": 35.00,
                "total_cost": 80.00,
                "estimated_duration_minutes": 30,
                "parts_available": True,
                "valid_until": "2024-03-20T23:59:59Z",
                "notes": "Includes synthetic oil"
            }
        }
    
    @pytest.mark.asyncio
    async def test_get_maintenance_quote_success(self, dms_adapter, mock_quote_response):
        """Test successful maintenance quote retrieval."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_quote_response, "json")  # get_maintenance_quote
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            quote = await dms_adapter.get_maintenance_quote(
                vehicle_id="vehicle-456",
                service_type="oil_change",
                vin="1HGBH41JXMN109186"
            )
        
        # Assert
        assert isinstance(quote, MaintenanceQuote)
        assert quote.quote_id == "quote-001"
        assert quote.vehicle_id == "vehicle-456"
        assert quote.service_type == "oil_change"
        assert quote.labor_cost == 45.00
        assert quote.parts_cost == 35.00
        assert quote.total_cost == 80.00
        assert quote.estimated_duration_minutes == 30
        assert quote.parts_available is True
        assert quote.notes == "Includes synthetic oil"
    
    @pytest.mark.asyncio
    async def test_get_maintenance_quote_without_vin(self, dms_adapter, mock_quote_response):
        """Test maintenance quote retrieval without VIN."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_quote_response, "json")  # get_maintenance_quote
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            quote = await dms_adapter.get_maintenance_quote(
                vehicle_id="vehicle-456",
                service_type="oil_change"
            )
        
        # Assert
        assert isinstance(quote, MaintenanceQuote)
        assert quote.quote_id == "quote-001"
    
    @pytest.mark.asyncio
    async def test_get_maintenance_quote_invalid_response(self, dms_adapter):
        """Test handling of invalid quote response."""
        # Arrange - response without 'quote' key
        invalid_response = {"error": "Quote not available"}
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, invalid_response, "json")  # get_maintenance_quote with invalid response
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act & Assert
            with pytest.raises(ValueError, match="Quote retrieval did not return a valid maintenance quote"):
                await dms_adapter.get_maintenance_quote(
                    vehicle_id="vehicle-456",
                    service_type="oil_change"
                )


class TestDMSAdapterPartsAvailability:
    """Test parts availability checking functionality."""
    
    @pytest.fixture
    def mock_parts_response(self):
        """Create a mock parts availability response."""
        return {
            "parts": [
                {
                    "part_number": "OIL-FILTER-001",
                    "part_name": "Oil Filter",
                    "quantity_available": 5,
                    "quantity_needed": 1,
                    "in_stock": True
                },
                {
                    "part_number": "OIL-5W30-001",
                    "part_name": "Synthetic Oil 5W-30",
                    "quantity_available": 10,
                    "quantity_needed": 5,
                    "in_stock": True
                },
                {
                    "part_number": "GASKET-001",
                    "part_name": "Oil Pan Gasket",
                    "quantity_available": 0,
                    "quantity_needed": 1,
                    "in_stock": False,
                    "estimated_arrival": "2024-03-18T10:00:00Z"
                }
            ]
        }
    
    @pytest.mark.asyncio
    async def test_check_parts_availability_success(self, dms_adapter, mock_parts_response):
        """Test successful parts availability check."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_parts_response, "json")  # check_parts_availability
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            parts = await dms_adapter.check_parts_availability(
                vehicle_id="vehicle-456",
                service_type="oil_change",
                vin="1HGBH41JXMN109186"
            )
        
        # Assert
        assert len(parts) == 3
        assert all(isinstance(part, PartAvailability) for part in parts)
        
        # Check first part (in stock)
        assert parts[0].part_number == "OIL-FILTER-001"
        assert parts[0].part_name == "Oil Filter"
        assert parts[0].quantity_available == 5
        assert parts[0].quantity_needed == 1
        assert parts[0].in_stock is True
        assert parts[0].estimated_arrival is None
        
        # Check third part (out of stock with estimated arrival)
        assert parts[2].part_number == "GASKET-001"
        assert parts[2].in_stock is False
        assert parts[2].estimated_arrival is not None
    
    @pytest.mark.asyncio
    async def test_check_parts_availability_empty_response(self, dms_adapter):
        """Test parts availability check with no parts."""
        # Arrange
        empty_response = {"parts": []}
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, empty_response, "json")  # check_parts_availability
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            parts = await dms_adapter.check_parts_availability(
                vehicle_id="vehicle-456",
                service_type="oil_change"
            )
        
        # Assert
        assert len(parts) == 0


class TestDMSAdapterEmergencyTriage:
    """Test emergency triage functionality."""
    
    @pytest.fixture
    def mock_triage_critical_response(self):
        """Create a mock critical emergency triage response."""
        return {
            "triage": {
                "severity": "critical",
                "safe_to_drive": False,
                "recommended_action": "Do not drive. Call for tow truck immediately.",
                "estimated_repair_time": 180,
                "requires_tow": True,
                "priority_score": 95
            }
        }
    
    @pytest.fixture
    def mock_triage_low_response(self):
        """Create a mock low severity triage response."""
        return {
            "triage": {
                "severity": "low",
                "safe_to_drive": True,
                "recommended_action": "Schedule service appointment within the next week.",
                "estimated_repair_time": 60,
                "requires_tow": False,
                "priority_score": 25
            }
        }
    
    @pytest.mark.asyncio
    async def test_triage_emergency_critical(self, dms_adapter, mock_triage_critical_response):
        """Test emergency triage for critical issue."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_triage_critical_response, "json")  # triage_emergency
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            result = await dms_adapter.triage_emergency(
                vehicle_id="vehicle-456",
                issue_description="Engine overheating, smoke coming from hood",
                vin="1HGBH41JXMN109186"
            )
        
        # Assert
        assert isinstance(result, TriageResult)
        assert result.severity == "critical"
        assert result.safe_to_drive is False
        assert result.requires_tow is True
        assert result.priority_score == 95
        assert "tow truck" in result.recommended_action.lower()
    
    @pytest.mark.asyncio
    async def test_triage_emergency_low_severity(self, dms_adapter, mock_triage_low_response):
        """Test emergency triage for low severity issue."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_triage_low_response, "json")  # triage_emergency
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            result = await dms_adapter.triage_emergency(
                vehicle_id="vehicle-456",
                issue_description="Check engine light on, no performance issues"
            )
        
        # Assert
        assert isinstance(result, TriageResult)
        assert result.severity == "low"
        assert result.safe_to_drive is True
        assert result.requires_tow is False
        assert result.priority_score == 25
    
    @pytest.mark.asyncio
    async def test_triage_emergency_with_vin(self, dms_adapter, mock_triage_low_response):
        """Test emergency triage with VIN provided."""
        # Arrange
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, mock_triage_low_response, "json")  # triage_emergency
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act
            result = await dms_adapter.triage_emergency(
                vehicle_id="vehicle-456",
                issue_description="Strange noise from brakes",
                vin="1HGBH41JXMN109186"
            )
        
        # Assert
        assert isinstance(result, TriageResult)
    
    @pytest.mark.asyncio
    async def test_triage_emergency_invalid_response(self, dms_adapter):
        """Test handling of invalid triage response."""
        # Arrange - response without 'triage' key
        invalid_response = {"error": "Triage failed"}
        
        mock_session = create_mock_aiohttp_session([
            (200, {}, "json"),  # authenticate
            (200, invalid_response, "json")  # triage_emergency with invalid response
        ])
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Act & Assert
            with pytest.raises(ValueError, match="Triage assessment did not return a valid result"):
                await dms_adapter.triage_emergency(
                    vehicle_id="vehicle-456",
                    issue_description="Unknown issue"
                )
