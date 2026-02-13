"""
Example usage of DMS Adapter for automotive service scheduling.

This example demonstrates:
- API key authentication with DMS
- Searching for available service slots
- Booking service appointments with VIN-based vehicle identification
- Getting maintenance quotes with pricing
- Checking parts availability
- Emergency triage for vehicle issues
- Service type classification

Requirements: 4.2, 4.6
"""

import asyncio
from datetime import date, timedelta
from voice_agent.adapters.dms_adapter import DMSAdapter, DateRange


async def main():
    """Demonstrate DMS adapter usage for automotive service scheduling."""
    
    # Initialize DMS adapter
    adapter = DMSAdapter(
        base_url="https://api.dealership.com/v1",
        dealer_id="dealer-123",
        api_key="your_api_key_here",
        timeout=30
    )
    
    print("=== DMS Adapter Example ===\n")
    
    # Step 1: Authenticate
    print("1. Authenticating with DMS API...")
    try:
        authenticated = await adapter.authenticate()
        print(f"   ✓ Authentication successful: {authenticated}\n")
    except ValueError as e:
        print(f"   ✗ Authentication failed: {e}\n")
        return
    
    # Step 2: Search for available service slots
    print("2. Searching for available service slots...")
    date_range = DateRange(
        start=date.today(),
        end=date.today() + timedelta(days=7)
    )
    
    try:
        slots = await adapter.find_service_slots(
            vehicle_id="vehicle-456",
            service_type="oil_change",
            date_range=date_range
        )
        
        print(f"   ✓ Found {len(slots)} available slots:")
        for slot in slots[:3]:  # Show first 3 slots
            print(f"     - Slot {slot.id}: {slot.start_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"       Service: {slot.service_type}, Bay: {slot.bay_number}")
        print()
    except ValueError as e:
        print(f"   ✗ Slot search failed: {e}\n")
        return
    
    # Step 3: Book a service appointment
    if slots:
        print("3. Booking service appointment...")
        try:
            appointment = await adapter.book_service(
                vehicle_id="vehicle-456",
                slot_id=slots[0].id,
                service_type="oil_change",
                vin="1HGBH41JXMN109186",
                notes="Customer requested synthetic oil"
            )
            
            print(f"   ✓ Appointment booked successfully!")
            print(f"     - Appointment ID: {appointment.id}")
            print(f"     - VIN: {appointment.vin}")
            print(f"     - Time: {appointment.start_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"     - Service: {appointment.service_type}")
            print(f"     - Status: {appointment.status}")
            print(f"     - Bay: {appointment.bay_number}")
            if appointment.notes:
                print(f"     - Notes: {appointment.notes}")
            print()
        except ValueError as e:
            print(f"   ✗ Booking failed: {e}\n")
    
    # Step 4: Classify service types
    print("4. Classifying service types...")
    service_types = ["oil_change", "tire_rotation", "diagnostics", "brake_service"]
    for service_type in service_types:
        classification = adapter.classify_service_type(service_type)
        print(f"   - {service_type}: {classification['category']} ({classification['typical_duration']} min)")
    print()
    
    # Step 5: Get maintenance quote
    print("5. Getting maintenance quote...")
    try:
        quote = await adapter.get_maintenance_quote(
            vehicle_id="vehicle-456",
            service_type="oil_change",
            vin="1HGBH41JXMN109186"
        )
        
        print(f"   ✓ Quote retrieved successfully!")
        print(f"     - Quote ID: {quote.quote_id}")
        print(f"     - Service: {quote.service_type}")
        print(f"     - Labor Cost: ${quote.labor_cost:.2f}")
        print(f"     - Parts Cost: ${quote.parts_cost:.2f}")
        print(f"     - Total Cost: ${quote.total_cost:.2f}")
        print(f"     - Estimated Duration: {quote.estimated_duration_minutes} minutes")
        print(f"     - Parts Available: {quote.parts_available}")
        print(f"     - Valid Until: {quote.valid_until.strftime('%Y-%m-%d %H:%M')}")
        if quote.notes:
            print(f"     - Notes: {quote.notes}")
        print()
    except ValueError as e:
        print(f"   ✗ Quote retrieval failed: {e}\n")
    
    # Step 6: Check parts availability
    print("6. Checking parts availability...")
    try:
        parts = await adapter.check_parts_availability(
            vehicle_id="vehicle-456",
            service_type="oil_change",
            vin="1HGBH41JXMN109186"
        )
        
        print(f"   ✓ Found {len(parts)} parts:")
        for part in parts:
            status = "✓ In Stock" if part.in_stock else "✗ Out of Stock"
            print(f"     - {part.part_name} ({part.part_number}): {status}")
            print(f"       Available: {part.quantity_available}/{part.quantity_needed}")
            if not part.in_stock and part.estimated_arrival:
                print(f"       Estimated Arrival: {part.estimated_arrival.strftime('%Y-%m-%d')}")
        print()
    except ValueError as e:
        print(f"   ✗ Parts check failed: {e}\n")
    
    # Step 7: Emergency triage
    print("7. Performing emergency triage...")
    
    # Example 1: Critical issue
    print("   Example 1: Critical Issue")
    try:
        triage = await adapter.triage_emergency(
            vehicle_id="vehicle-456",
            issue_description="Engine overheating, smoke coming from hood",
            vin="1HGBH41JXMN109186"
        )
        
        print(f"     - Severity: {triage.severity.upper()}")
        print(f"     - Safe to Drive: {'Yes' if triage.safe_to_drive else 'No'}")
        print(f"     - Requires Tow: {'Yes' if triage.requires_tow else 'No'}")
        print(f"     - Priority Score: {triage.priority_score}/100")
        print(f"     - Recommended Action: {triage.recommended_action}")
        if triage.estimated_repair_time:
            print(f"     - Estimated Repair Time: {triage.estimated_repair_time} minutes")
        print()
    except ValueError as e:
        print(f"     ✗ Triage failed: {e}\n")
    
    # Example 2: Low severity issue
    print("   Example 2: Low Severity Issue")
    try:
        triage = await adapter.triage_emergency(
            vehicle_id="vehicle-456",
            issue_description="Check engine light on, no performance issues",
            vin="1HGBH41JXMN109186"
        )
        
        print(f"     - Severity: {triage.severity.upper()}")
        print(f"     - Safe to Drive: {'Yes' if triage.safe_to_drive else 'No'}")
        print(f"     - Priority Score: {triage.priority_score}/100")
        print(f"     - Recommended Action: {triage.recommended_action}")
        print()
    except ValueError as e:
        print(f"     ✗ Triage failed: {e}\n")
    
    print("=== Example Complete ===")


if __name__ == "__main__":
    # Note: This example requires a valid DMS API endpoint and credentials
    # Replace the base_url and api_key with actual values to run
    print("Note: This is a demonstration. Replace API credentials with actual values to run.\n")
    
    # Uncomment to run with actual credentials:
    # asyncio.run(main())
