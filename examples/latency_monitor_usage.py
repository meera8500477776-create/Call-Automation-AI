"""Example usage of LatencyMonitor for tracking voice pipeline latency."""

import time

from voice_agent.monitoring import LatencyMonitor


def simulate_voice_pipeline():
    """Simulate a voice pipeline with latency tracking."""
    monitor = LatencyMonitor()
    session_id = "example-session-001"

    print("=== Voice Pipeline Latency Monitoring Example ===\n")

    # Simulate STT processing
    print("1. Processing Speech-to-Text...")
    monitor.start_timer(session_id, LatencyMonitor.STAGE_STT)
    time.sleep(0.12)  # Simulate 120ms STT processing
    stt_duration = monitor.end_timer(session_id, LatencyMonitor.STAGE_STT)
    print(f"   STT completed in {stt_duration:.2f}ms\n")

    # Simulate LLM processing
    print("2. Processing LLM response...")
    monitor.start_timer(session_id, LatencyMonitor.STAGE_LLM_TTFT)
    time.sleep(0.18)  # Simulate 180ms to first token
    llm_ttft = monitor.end_timer(session_id, LatencyMonitor.STAGE_LLM_TTFT)
    print(f"   LLM Time-to-First-Token: {llm_ttft:.2f}ms")

    monitor.start_timer(session_id, LatencyMonitor.STAGE_LLM_TOTAL)
    time.sleep(0.15)  # Additional 150ms for complete response
    llm_total = monitor.end_timer(session_id, LatencyMonitor.STAGE_LLM_TOTAL)
    print(f"   LLM Total: {llm_total:.2f}ms\n")

    # Simulate TTS processing
    print("3. Processing Text-to-Speech...")
    monitor.start_timer(session_id, LatencyMonitor.STAGE_TTS_TTFB)
    time.sleep(0.09)  # Simulate 90ms to first audio byte
    tts_ttfb = monitor.end_timer(session_id, LatencyMonitor.STAGE_TTS_TTFB)
    print(f"   TTS Time-to-First-Byte: {tts_ttfb:.2f}ms")

    monitor.start_timer(session_id, LatencyMonitor.STAGE_TTS_TOTAL)
    time.sleep(0.11)  # Additional 110ms for complete audio
    tts_total = monitor.end_timer(session_id, LatencyMonitor.STAGE_TTS_TOTAL)
    print(f"   TTS Total: {tts_total:.2f}ms\n")

    # Record end-to-end latency
    end_to_end = stt_duration + llm_total + tts_total
    monitor.record_latency(session_id, LatencyMonitor.STAGE_END_TO_END, end_to_end)
    print(f"4. End-to-End Latency: {end_to_end:.2f}ms\n")

    # Get session metrics
    print("=== Session Metrics ===")
    metrics = monitor.get_session_metrics(session_id)
    if metrics:
        print(f"STT Latency:        {metrics.stt_latency_ms:.2f}ms")
        print(f"LLM TTFT:           {metrics.llm_ttft_ms:.2f}ms")
        print(f"LLM Total:          {metrics.llm_total_ms:.2f}ms")
        print(f"TTS TTFB:           {metrics.tts_ttfb_ms:.2f}ms")
        print(f"TTS Total:          {metrics.tts_total_ms:.2f}ms")
        print(f"End-to-End:         {metrics.end_to_end_ms:.2f}ms")
        print()

    return monitor


def simulate_multiple_sessions():
    """Simulate multiple sessions to demonstrate aggregate metrics."""
    monitor = LatencyMonitor()

    print("=== Simulating Multiple Sessions ===\n")

    # Simulate 10 sessions with varying latencies
    for i in range(10):
        session_id = f"session-{i:03d}"

        # Vary latencies to simulate real-world conditions
        stt_latency = 100.0 + (i * 10)
        llm_latency = 200.0 + (i * 15)
        tts_latency = 150.0 + (i * 8)

        monitor.record_latency(session_id, LatencyMonitor.STAGE_STT, stt_latency)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_LLM_TOTAL, llm_latency)
        monitor.record_latency(session_id, LatencyMonitor.STAGE_TTS_TOTAL, tts_latency)

        print(f"Session {i}: STT={stt_latency:.0f}ms, LLM={llm_latency:.0f}ms, TTS={tts_latency:.0f}ms")

    print("\n=== Aggregate Metrics ===\n")

    # Get aggregate metrics for all stages
    agg_metrics = monitor.get_aggregate_metrics()

    for stage_name, metrics in agg_metrics.items():
        print(f"{stage_name.upper()}:")
        print(f"  Count:      {metrics.count}")
        print(f"  Mean:       {metrics.mean_ms:.2f}ms")
        print(f"  Median:     {metrics.median_ms:.2f}ms")
        print(f"  P95:        {metrics.p95_ms:.2f}ms")
        print(f"  P99:        {metrics.p99_ms:.2f}ms")
        print(f"  Min:        {metrics.min_ms:.2f}ms")
        print(f"  Max:        {metrics.max_ms:.2f}ms")
        print()


def demonstrate_threshold_alerts():
    """Demonstrate latency threshold alerting."""
    monitor = LatencyMonitor()

    print("=== Threshold Alerting Demo ===\n")

    # Good latency (< 500ms target)
    print("1. Recording good latency (300ms)...")
    monitor.record_latency("session-good", LatencyMonitor.STAGE_END_TO_END, 300.0)
    print("   ✓ No alerts (below 500ms target)\n")

    # Above target but below warning (500-1000ms)
    print("2. Recording above-target latency (750ms)...")
    monitor.record_latency("session-target", LatencyMonitor.STAGE_END_TO_END, 750.0)
    print("   ⚠ Info alert (above 500ms target)\n")

    # Above warning threshold (> 1000ms)
    print("3. Recording high latency (1500ms)...")
    monitor.record_latency("session-warning", LatencyMonitor.STAGE_END_TO_END, 1500.0)
    print("   ⚠️ Warning alert (above 1000ms threshold)\n")


def demonstrate_session_cleanup():
    """Demonstrate session cleanup."""
    monitor = LatencyMonitor()
    session_id = "cleanup-demo"

    print("=== Session Cleanup Demo ===\n")

    # Create session with metrics
    monitor.start_timer(session_id, LatencyMonitor.STAGE_STT)
    monitor.record_latency(session_id, LatencyMonitor.STAGE_LLM_TOTAL, 200.0)

    print(f"Session '{session_id}' created with metrics")
    print(f"Metrics exist: {monitor.get_session_metrics(session_id) is not None}")
    print(f"Active timers: {session_id in monitor._timers}\n")

    # Clean up session
    monitor.clear_session(session_id)

    print(f"Session '{session_id}' cleared")
    print(f"Metrics exist: {monitor.get_session_metrics(session_id) is not None}")
    print(f"Active timers: {session_id in monitor._timers}\n")


if __name__ == "__main__":
    # Run all examples
    simulate_voice_pipeline()
    print("\n" + "=" * 60 + "\n")

    simulate_multiple_sessions()
    print("\n" + "=" * 60 + "\n")

    demonstrate_threshold_alerts()
    print("\n" + "=" * 60 + "\n")

    demonstrate_session_cleanup()
