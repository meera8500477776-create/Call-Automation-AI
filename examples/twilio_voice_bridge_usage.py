"""
Example usage of Twilio-Voice Pipeline Bridge.

This example demonstrates how to wire Twilio telephony integration
to the Voice Pipeline for handling real-time voice calls.
"""

import asyncio
import os
from voice_agent.telephony.twilio_integration import TwilioIntegration, TwilioConfig, ConnectionType
from voice_agent.telephony.twilio_voice_bridge import TwilioVoiceBridge, BridgeConfig
from voice_agent.core.pipecat_orchestrator import PipecatOrchestrator
from voice_agent.config.registry import TenantRegistry


async def main():
    """Main example function."""
    
    # 1. Set up Twilio configuration
    twilio_config = TwilioConfig(
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", "test_account_sid"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", "test_auth_token"),
        connection_type=ConnectionType.WEBSOCKET,
        audio_format="mulaw",
        sample_rate=8000,
        channels=1,
        chunk_size=160,
        enable_dtmf=True,
    )
    
    # 2. Create Twilio integration
    twilio = TwilioIntegration(twilio_config)
    
    # 3. Set up tenant registry and orchestrator
    tenant_registry = TenantRegistry()
    orchestrator = PipecatOrchestrator(tenant_registry)
    
    # 4. Create bridge configuration
    bridge_config = BridgeConfig(
        session_id="example_session_123",
        enable_audio_logging=True,  # Enable for debugging
        audio_buffer_size=8192,
    )
    
    # 5. Create the bridge
    bridge = TwilioVoiceBridge(
        config=bridge_config,
        twilio_integration=twilio,
        orchestrator=orchestrator,
    )
    
    # 6. Start the bridge
    print("Starting Twilio-Voice Bridge...")
    await bridge.start()
    print("Bridge started successfully!")
    
    # 7. Connect to Twilio WebSocket
    # In production, this URL comes from Twilio's webhook
    websocket_url = os.getenv("TWILIO_WEBSOCKET_URL", "wss://example.twilio.com/stream")
    
    try:
        print(f"Connecting to Twilio WebSocket: {websocket_url}")
        await twilio.connect(websocket_url)
        print("Connected to Twilio!")
        
        # The bridge will now automatically:
        # - Handle call started events and create sessions
        # - Route audio from Twilio to Voice Pipeline
        # - Route generated audio from Pipeline to Twilio
        # - Handle call ended events and cleanup sessions
        
        # Keep the connection alive
        print("Bridge is active. Press Ctrl+C to stop...")
        while bridge.is_active:
            await asyncio.sleep(1)
            
            # Optionally check metrics
            if bridge.pipeline:
                metrics = bridge.get_pipeline_metrics()
                if metrics and metrics.end_to_end_ms > 0:
                    print(f"Pipeline E2E latency: {metrics.end_to_end_ms:.2f}ms")
    
    except KeyboardInterrupt:
        print("\nStopping bridge...")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # 8. Cleanup
        print("Cleaning up...")
        await bridge.stop()
        await twilio.disconnect()
        print("Bridge stopped successfully!")


async def example_with_callbacks():
    """Example with custom callbacks for monitoring."""
    
    # Set up components (same as main example)
    twilio_config = TwilioConfig(
        account_sid="test_account_sid",
        auth_token="test_auth_token",
    )
    twilio = TwilioIntegration(twilio_config)
    
    tenant_registry = TenantRegistry()
    orchestrator = PipecatOrchestrator(tenant_registry)
    
    bridge_config = BridgeConfig(session_id="callback_example_123")
    bridge = TwilioVoiceBridge(bridge_config, twilio, orchestrator)
    
    # Define custom callbacks
    def on_call_started(call_metadata):
        print(f"📞 Call started: {call_metadata.call_sid}")
        print(f"   From: {call_metadata.caller_id}")
        print(f"   To: {call_metadata.called_number}")
    
    def on_call_ended(call_metadata):
        print(f"📞 Call ended: {call_metadata.call_sid}")
        print(f"   Duration: {call_metadata.duration_seconds:.2f}s")
    
    def on_audio_received(audio_data):
        print(f"🎤 Received audio: {len(audio_data)} bytes")
    
    def on_error(error):
        print(f"❌ Error: {error}")
    
    # Register callbacks
    twilio.on_call_started(on_call_started)
    twilio.on_call_ended(on_call_ended)
    twilio.on_audio_received(on_audio_received)
    twilio.on_error(on_error)
    
    # Start bridge
    await bridge.start()
    
    # Connect and run
    try:
        websocket_url = "wss://example.twilio.com/stream"
        await twilio.connect(websocket_url)
        
        print("Bridge running with callbacks...")
        await asyncio.sleep(10)  # Run for 10 seconds
    
    finally:
        await bridge.stop()
        await twilio.disconnect()


async def example_production_setup():
    """Example production setup with error handling and monitoring."""
    
    # Production configuration
    twilio_config = TwilioConfig(
        account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        connection_type=ConnectionType.WEBSOCKET,
        audio_format="mulaw",
        sample_rate=8000,
        enable_dtmf=True,
    )
    
    # Validate configuration
    if not twilio_config.account_sid or not twilio_config.auth_token:
        raise ValueError("TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN must be set")
    
    # Create components
    twilio = TwilioIntegration(twilio_config)
    tenant_registry = TenantRegistry()
    orchestrator = PipecatOrchestrator(tenant_registry)
    
    # Production bridge configuration
    bridge_config = BridgeConfig(
        session_id=f"prod_session_{os.getpid()}",
        enable_audio_logging=False,  # Disable in production for performance
        audio_buffer_size=16384,  # Larger buffer for production
    )
    
    bridge = TwilioVoiceBridge(bridge_config, twilio, orchestrator)
    
    # Error handling callback
    def handle_error(error):
        print(f"Production error: {error}")
        # In production, send to monitoring system
        # e.g., sentry_sdk.capture_exception(error)
    
    twilio.on_error(handle_error)
    
    try:
        # Start bridge
        await bridge.start()
        print("Production bridge started")
        
        # Connect to Twilio
        websocket_url = os.getenv("TWILIO_WEBSOCKET_URL")
        if not websocket_url:
            raise ValueError("TWILIO_WEBSOCKET_URL must be set")
        
        await twilio.connect(websocket_url)
        print("Connected to Twilio in production mode")
        
        # Monitor bridge health
        while bridge.is_active:
            await asyncio.sleep(5)
            
            # Check call metadata
            call_metadata = bridge.get_call_metadata()
            if call_metadata:
                print(f"Active call: {call_metadata.call_sid}")
            
            # Check pipeline metrics
            metrics = bridge.get_pipeline_metrics()
            if metrics and metrics.end_to_end_ms > 0:
                # Alert if latency is too high
                if metrics.end_to_end_ms > 1000:
                    print(f"⚠️  High latency detected: {metrics.end_to_end_ms:.2f}ms")
    
    except Exception as e:
        print(f"Production error: {e}")
        raise
    
    finally:
        # Graceful shutdown
        print("Shutting down production bridge...")
        await bridge.stop()
        await twilio.disconnect()
        print("Production bridge stopped")


if __name__ == "__main__":
    print("Twilio-Voice Bridge Examples")
    print("=" * 50)
    print()
    
    # Run the main example
    print("Running main example...")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    
    print()
    print("=" * 50)
    print("Example completed!")
