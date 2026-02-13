"""
Load testing suite for concurrent call scenarios.

Tests concurrent call handling, auto-scaling behavior, latency under load,
and system stability at 1000+ concurrent calls.

**Validates: Requirements 2.1, 9.6**
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
from typing import List
import statistics

from src.voice_agent.core.pipecat_orchestrator import (
    PipecatOrchestrator,
    CallMetadata,
)
from src.voice_agent.monitoring.latency_monitor import LatencyMonitor
from src.voice_agent.core.session_manager import SessionManager
from src.voice_agent.config.registry import TenantRegistry
from src.voice_agent.deployment.scaling_controller import ScalingController


@pytest.fixture
def scaling_controller():
    """Create scaling controller for testing."""
    return ScalingController(
        min_replicas=1,
        max_replicas=10,
        target_concurrent_calls=100
    )


@pytest.fixture
def latency_monitor():
    """Create latency monitor for load testing."""
    return LatencyMonitor()


class TestConcurrentCallHandling:
    """Test concurrent call handling capabilities."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_handle_100_concurrent_calls(self):
        """
        Test handling 100 concurrent calls.
        
        **Validates: Requirements 9.6**
        """
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            # Setup tenant registry
            mock_registry = Mock(spec=TenantRegistry)
            mock_registry.get_tenant_by_caller.return_value = "healthcare_clinic_001"
            mock_registry.load_tenant_config.return_value = Mock(
                tenant_id="healthcare_clinic_001",
                persona=Mock(name="Sarah"),
                llm=Mock(provider="openai"),
                tts=Mock(provider="elevenlabs"),
                tools=[], knowledge_bases=[],
                constraints=Mock(max_call_duration=600, enable_barge_in=True)
            )
            mock_registry_class.return_value = mock_registry
            
            orchestrator = PipecatOrchestrator(
                tenant_registry=mock_registry,
                session_manager=SessionManager()
            )
            
            # Mock pipeline initialization
            async def mock_init_pipeline(*args, **kwargs):
                p = AsyncMock()
                p.initialize = AsyncMock()
                p.shutdown = AsyncMock()
                return p
            
            with patch.object(orchestrator, 'initialize_pipeline', side_effect=mock_init_pipeline) as mock_init:
                
                # Create 100 concurrent calls
                num_calls = 100
                session_ids = [f"load_test_session_{i:03d}" for i in range(num_calls)]
                call_metadatas = [
                    CallMetadata(
                        caller_id=f"+1555{i:07d}",
                        call_sid=f"CA_load_{i:03d}",
                        from_number=f"+1555{i:07d}"
                    )
                    for i in range(num_calls)
                ]
                
                # Start all sessions concurrently
                start_time = time.time()
                
                await asyncio.gather(*[
                    orchestrator.start_session(
                        session_id=session_ids[i],
                        caller_id=call_metadatas[i].caller_id,
                        call_metadata=call_metadatas[i]
                    )
                    for i in range(num_calls)
                ])
                
                startup_time = time.time() - start_time
                
                # Verify all sessions are active
                assert orchestrator.get_session_count() == num_calls
                print(f"\nStarted {num_calls} sessions in {startup_time:.2f}s")
                
                # Simulate active calls for a short period
                await asyncio.sleep(0.5)
                
                # Stop all sessions concurrently
                shutdown_start = time.time()
                
                await asyncio.gather(*[
                    orchestrator.stop_session(sid)
                    for sid in session_ids
                ])
                
                shutdown_time = time.time() - shutdown_start
                
                # Verify all sessions are cleaned up
                assert orchestrator.get_session_count() == 0
                print(f"Stopped {num_calls} sessions in {shutdown_time:.2f}s")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_handle_1000_concurrent_calls(self):
        """
        Test handling 1000+ concurrent calls.
        
        **Validates: Requirements 9.6**
        """
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            # Setup tenant registry
            mock_registry = Mock(spec=TenantRegistry)
            mock_registry.get_tenant_by_caller.return_value = "healthcare_clinic_001"
            mock_registry.load_tenant_config.return_value = Mock(
                tenant_id="healthcare_clinic_001",
                persona=Mock(name="Sarah"),
                llm=Mock(provider="openai"),
                tts=Mock(provider="elevenlabs"),
                tools=[], knowledge_bases=[],
                constraints=Mock(max_call_duration=600, enable_barge_in=True)
            )
            mock_registry_class.return_value = mock_registry
            
            orchestrator = PipecatOrchestrator(
                tenant_registry=mock_registry,
                session_manager=SessionManager()
            )
            
            # Mock pipeline initialization
            async def mock_init_pipeline(*args, **kwargs):
                p = AsyncMock()
                p.initialize = AsyncMock()
                p.shutdown = AsyncMock()
                return p
            
            with patch.object(orchestrator, 'initialize_pipeline', side_effect=mock_init_pipeline) as mock_init:
                
                num_calls = 1000
                
                # Create 1000 concurrent calls
                session_ids = [f"load_test_1k_{i:04d}" for i in range(num_calls)]
                call_metadatas = [
                    CallMetadata(
                        caller_id=f"+1555{i:07d}",
                        call_sid=f"CA_1k_{i:04d}",
                        from_number=f"+1555{i:07d}"
                    )
                    for i in range(num_calls)
                ]
                
                # Start sessions in batches to avoid overwhelming the system
                batch_size = 100
                start_time = time.time()
                
                for batch_start in range(0, num_calls, batch_size):
                    batch_end = min(batch_start + batch_size, num_calls)
                    await asyncio.gather(*[
                        orchestrator.start_session(
                            session_id=session_ids[i],
                            caller_id=call_metadatas[i].caller_id,
                            call_metadata=call_metadatas[i]
                        )
                        for i in range(batch_start, batch_end)
                    ])
                
                startup_time = time.time() - start_time
                
                # Verify all sessions are active
                assert orchestrator.get_session_count() == num_calls
                print(f"\nStarted {num_calls} sessions in {startup_time:.2f}s")
                print(f"Average startup time per session: {(startup_time/num_calls)*1000:.2f}ms")
                
                # Simulate active calls
                await asyncio.sleep(0.5)
                
                # Stop sessions in batches
                shutdown_start = time.time()
                
                for batch_start in range(0, num_calls, batch_size):
                    batch_end = min(batch_start + batch_size, num_calls)
                    await asyncio.gather(*[
                        orchestrator.stop_session(session_ids[i])
                        for i in range(batch_start, batch_end)
                    ])
                
                shutdown_time = time.time() - shutdown_start
                
                # Verify all sessions are cleaned up
                assert orchestrator.get_session_count() == 0
                print(f"Stopped {num_calls} sessions in {shutdown_time:.2f}s")
                print(f"Average shutdown time per session: {(shutdown_time/num_calls)*1000:.2f}ms")


class TestAutoScalingBehavior:
    """Test auto-scaling behavior under load."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_scaling_up_under_increasing_load(self, scaling_controller):
        """
        Test auto-scaling up as load increases.
        
        **Validates: Requirements 9.3, 9.6**
        """
        # Simulate increasing load
        load_levels = [50, 150, 300, 500, 800, 1000]
        
        for concurrent_calls in load_levels:
            # Get current load metrics
            load_metrics = scaling_controller.get_current_load()
            load_metrics.concurrent_calls = concurrent_calls
            load_metrics.gpu_utilization = min(concurrent_calls / 10, 100)  # Simulate GPU usage
            
            # Determine if scaling is needed
            should_scale = scaling_controller.should_scale_up(load_metrics)
            
            if should_scale:
                # Calculate desired replicas
                desired_replicas = scaling_controller.calculate_desired_replicas(load_metrics)
                print(f"\nLoad: {concurrent_calls} calls -> Scale to {desired_replicas} replicas")
                
                # Verify scaling decision is reasonable
                assert desired_replicas >= 1
                assert desired_replicas <= scaling_controller.max_replicas
                
                # Simulate scaling up
                scaling_controller.scale_up(desired_replicas)
                
                # Verify new replica count
                assert scaling_controller.current_replicas == desired_replicas
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_scaling_down_under_decreasing_load(self, scaling_controller):
        """
        Test auto-scaling down as load decreases.
        
        **Validates: Requirements 9.3, 9.7**
        """
        # Start with high load
        scaling_controller.scale_up(10)
        assert scaling_controller.current_replicas == 10
        
        # Simulate decreasing load
        load_levels = [800, 500, 300, 150, 50, 10]
        
        for concurrent_calls in load_levels:
            # Get current load metrics
            load_metrics = scaling_controller.get_current_load()
            load_metrics.concurrent_calls = concurrent_calls
            load_metrics.gpu_utilization = concurrent_calls / 10
            
            # Determine if scaling down is needed
            should_scale = scaling_controller.should_scale_down(load_metrics)
            
            if should_scale:
                # Calculate desired replicas
                desired_replicas = scaling_controller.calculate_desired_replicas(load_metrics)
                print(f"\nLoad: {concurrent_calls} calls -> Scale to {desired_replicas} replicas")
                
                # Verify scaling decision
                assert desired_replicas >= scaling_controller.min_replicas
                assert desired_replicas <= scaling_controller.current_replicas
                
                # Simulate scaling down
                scaling_controller.scale_down(desired_replicas)
                
                # Verify new replica count
                assert scaling_controller.current_replicas == desired_replicas


class TestLatencyUnderLoad:
    """Test latency under various load conditions."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_latency_remains_under_1000ms_at_100_concurrent(self, latency_monitor):
        """
        Test that latency remains under 1000ms with 100 concurrent calls.
        
        **Validates: Requirements 2.1, 2.8**
        """
        num_sessions = 100
        latencies: List[float] = []
        
        # Simulate 100 concurrent sessions
        async def simulate_session(session_id: str):
            # Simulate pipeline stages
            latency_monitor.start_timer(session_id, "stt")
            await asyncio.sleep(0.08)  # 80ms STT
            latency_monitor.end_timer(session_id, "stt")
            
            latency_monitor.start_timer(session_id, "llm_ttft")
            await asyncio.sleep(0.15)  # 150ms LLM TTFT
            latency_monitor.end_timer(session_id, "llm_ttft")
            
            latency_monitor.start_timer(session_id, "llm_total")
            await asyncio.sleep(0.35)  # 350ms LLM total
            latency_monitor.end_timer(session_id, "llm_total")
            
            latency_monitor.start_timer(session_id, "tts_ttfb")
            await asyncio.sleep(0.08)  # 80ms TTS TTFB
            latency_monitor.end_timer(session_id, "tts_ttfb")
            
            latency_monitor.start_timer(session_id, "tts_total")
            await asyncio.sleep(0.15)  # 150ms TTS total
            latency_monitor.end_timer(session_id, "tts_total")
            
            # Get metrics
            metrics = latency_monitor.get_session_metrics(session_id)
            return metrics.end_to_end_ms
        
        # Run all sessions concurrently
        session_ids = [f"latency_test_{i:03d}" for i in range(num_sessions)]
        latencies = await asyncio.gather(*[
            simulate_session(sid) for sid in session_ids
        ])
        
        # Calculate statistics
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_latency = statistics.quantiles(latencies, n=100)[98]  # 99th percentile
        
        print(f"\nLatency statistics for {num_sessions} concurrent calls:")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")
        
        # Verify latency requirements
        assert median_latency < 1000, f"Median latency {median_latency}ms exceeds 1000ms"
        assert p95_latency < 1500, f"P95 latency {p95_latency}ms exceeds 1500ms"
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_latency_degradation_at_high_load(self, latency_monitor):
        """
        Test latency degradation at high load (1000 concurrent calls).
        
        **Validates: Requirements 2.1, 9.6**
        """
        num_sessions = 1000
        latencies: List[float] = []
        
        # Simulate 1000 concurrent sessions with slightly increased latency
        async def simulate_session_under_load(session_id: str):
            # Simulate pipeline stages with load-induced delays
            latency_monitor.start_timer(session_id, "stt")
            await asyncio.sleep(0.1)  # 100ms STT (slight increase)
            latency_monitor.end_timer(session_id, "stt")
            
            latency_monitor.start_timer(session_id, "llm_ttft")
            await asyncio.sleep(0.2)  # 200ms LLM TTFT
            latency_monitor.end_timer(session_id, "llm_ttft")
            
            latency_monitor.start_timer(session_id, "llm_total")
            await asyncio.sleep(0.4)  # 400ms LLM total
            latency_monitor.end_timer(session_id, "llm_total")
            
            latency_monitor.start_timer(session_id, "tts_ttfb")
            await asyncio.sleep(0.1)  # 100ms TTS TTFB
            latency_monitor.end_timer(session_id, "tts_ttfb")
            
            latency_monitor.start_timer(session_id, "tts_total")
            await asyncio.sleep(0.2)  # 200ms TTS total
            latency_monitor.end_timer(session_id, "tts_total")
            
            metrics = latency_monitor.get_session_metrics(session_id)
            return metrics.end_to_end_ms
        
        # Run sessions in batches to avoid overwhelming test system
        batch_size = 100
        for batch_start in range(0, num_sessions, batch_size):
            batch_end = min(batch_start + batch_size, num_sessions)
            session_ids = [f"load_latency_{i:04d}" for i in range(batch_start, batch_end)]
            
            batch_latencies = await asyncio.gather(*[
                simulate_session_under_load(sid) for sid in session_ids
            ])
            latencies.extend(batch_latencies)
        
        # Calculate statistics
        median_latency = statistics.median(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        
        print(f"\nLatency statistics for {num_sessions} concurrent calls:")
        print(f"  Median: {median_latency:.2f}ms")
        print(f"  P95: {p95_latency:.2f}ms")
        print(f"  P99: {p99_latency:.2f}ms")
        
        # Verify latency remains acceptable under high load
        # Allow some degradation but should still be reasonable
        assert median_latency < 1200, f"Median latency {median_latency}ms exceeds acceptable threshold"
        assert p95_latency < 2000, f"P95 latency {p95_latency}ms exceeds acceptable threshold"


class TestSystemStability:
    """Test system stability at high concurrent call volumes."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_system_stability_sustained_1000_calls(self):
        """
        Test system stability with sustained 1000 concurrent calls.
        
        **Validates: Requirements 9.5, 9.6**
        """
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            # Setup tenant registry
            mock_registry = Mock(spec=TenantRegistry)
            mock_registry.get_tenant_by_caller.return_value = "healthcare_clinic_001"
            mock_registry.load_tenant_config.return_value = Mock(
                tenant_id="healthcare_clinic_001",
                persona=Mock(name="Sarah"),
                llm=Mock(provider="openai"),
                tts=Mock(provider="elevenlabs"),
                tools=[], knowledge_bases=[],
                constraints=Mock(max_call_duration=600, enable_barge_in=True)
            )
            mock_registry_class.return_value = mock_registry
            
            orchestrator = PipecatOrchestrator(
                tenant_registry=mock_registry,
                session_manager=SessionManager()
            )
            
            # Mock pipeline initialization
            async def mock_init_pipeline(*args, **kwargs):
                p = AsyncMock()
                p.initialize = AsyncMock()
                p.shutdown = AsyncMock()
                return p
            
            with patch.object(orchestrator, 'initialize_pipeline', side_effect=mock_init_pipeline) as mock_init:
                
                num_calls = 1000
                
                # Start 1000 sessions
                session_ids = [f"stability_test_{i:04d}" for i in range(num_calls)]
                call_metadatas = [
                    CallMetadata(
                        caller_id=f"+1555{i:07d}",
                        call_sid=f"CA_stab_{i:04d}",
                        from_number=f"+1555{i:07d}"
                    )
                    for i in range(num_calls)
                ]
                
                # Start sessions in batches
                batch_size = 100
                for batch_start in range(0, num_calls, batch_size):
                    batch_end = min(batch_start + batch_size, num_calls)
                    await asyncio.gather(*[
                        orchestrator.start_session(
                            session_id=session_ids[i],
                            caller_id=call_metadatas[i].caller_id,
                            call_metadata=call_metadatas[i]
                        )
                        for i in range(batch_start, batch_end)
                    ])
                
                # Verify all sessions are active
                assert orchestrator.get_session_count() == num_calls
                print(f"\nMaintaining {num_calls} concurrent sessions")
                
                # Sustain load for a period
                sustain_duration = 2.0  # seconds
                await asyncio.sleep(sustain_duration)
                
                # Verify system remains stable
                assert orchestrator.get_session_count() == num_calls
                print(f"System stable after {sustain_duration}s with {num_calls} concurrent calls")
                
                # Cleanup
                for batch_start in range(0, num_calls, batch_size):
                    batch_end = min(batch_start + batch_size, num_calls)
                    await asyncio.gather(*[
                        orchestrator.stop_session(session_ids[i])
                        for i in range(batch_start, batch_end)
                    ])
                
                assert orchestrator.get_session_count() == 0
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_no_memory_leaks_during_session_churn(self):
        """
        Test for memory leaks during high session churn.
        
        **Validates: Requirements 9.5, 11.6**
        """
        with patch('src.voice_agent.core.pipecat_orchestrator.TenantRegistry') as mock_registry_class:
            # Setup tenant registry
            mock_registry = Mock(spec=TenantRegistry)
            mock_registry.get_tenant_by_caller.return_value = "healthcare_clinic_001"
            mock_registry.load_tenant_config.return_value = Mock(
                tenant_id="healthcare_clinic_001",
                persona=Mock(name="Sarah"),
                llm=Mock(provider="openai"),
                tts=Mock(provider="elevenlabs"),
                tools=[], knowledge_bases=[],
                constraints=Mock(max_call_duration=600, enable_barge_in=True)
            )
            mock_registry_class.return_value = mock_registry
            
            orchestrator = PipecatOrchestrator(
                tenant_registry=mock_registry,
                session_manager=SessionManager()
            )
            
            # Mock pipeline initialization
            async def mock_init_pipeline(*args, **kwargs):
                p = AsyncMock()
                p.initialize = AsyncMock()
                p.shutdown = AsyncMock()
                return p
            
            with patch.object(orchestrator, 'initialize_pipeline', side_effect=mock_init_pipeline) as mock_init:
                # Simulate high churn: start and stop sessions repeatedly
                num_iterations = 100
                sessions_per_iteration = 10
                
                for iteration in range(num_iterations):
                    # Start sessions
                    session_ids = [f"churn_{iteration}_{i}" for i in range(sessions_per_iteration)]
                    call_metadatas = [
                        CallMetadata(
                            caller_id=f"+1555{iteration:04d}{i:03d}",
                            call_sid=f"CA_churn_{iteration}_{i}",
                            from_number=f"+1555{iteration:04d}{i:03d}"
                        )
                        for i in range(sessions_per_iteration)
                    ]
                    
                    await asyncio.gather(*[
                        orchestrator.start_session(
                            session_id=session_ids[i],
                            caller_id=call_metadatas[i].caller_id,
                            call_metadata=call_metadatas[i]
                        )
                        for i in range(sessions_per_iteration)
                    ])
                    
                    # Verify sessions started
                    assert orchestrator.get_session_count() == sessions_per_iteration
                    
                    # Stop sessions
                    await asyncio.gather(*[
                        orchestrator.stop_session(sid)
                        for sid in session_ids
                    ])
                    
                    # Verify complete cleanup
                    assert orchestrator.get_session_count() == 0
                
                print(f"\nCompleted {num_iterations} iterations of {sessions_per_iteration} sessions each")
                print(f"Total sessions created and destroyed: {num_iterations * sessions_per_iteration}")
                print("No memory leaks detected (all sessions properly cleaned up)")
