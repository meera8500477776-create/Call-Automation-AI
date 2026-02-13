"""
Unit tests for ScalingController.

Tests scaling decisions, policy configuration, and cost optimization logic.
"""

import pytest
from voice_agent.deployment import ScalingController, LoadMetrics, ScalingPolicy


@pytest.fixture
def default_policy():
    """Create a default scaling policy for testing."""
    return ScalingPolicy(
        min_replicas=1,
        max_replicas=10,
        target_concurrent_calls_per_replica=50,
        target_gpu_utilization_percent=70.0,
        scale_up_threshold_percent=80.0,
        scale_down_threshold_percent=30.0,
        cooldown_period_seconds=300
    )


@pytest.fixture
def controller(default_policy):
    """Create a scaling controller instance."""
    return ScalingController(default_policy)


class TestScalingController:
    """Test suite for ScalingController."""
    
    def test_initialization(self, default_policy):
        """Test controller initialization with policy."""
        controller = ScalingController(default_policy)
        assert controller.policy == default_policy
        assert controller._last_scale_time is None
    
    def test_get_current_load(self, controller):
        """Test retrieving current load metrics."""
        metrics = controller.get_current_load()
        assert isinstance(metrics, LoadMetrics)
        assert metrics.concurrent_calls >= 0
        assert metrics.gpu_utilization_percent >= 0.0
        assert metrics.request_queue_depth >= 0
        assert metrics.avg_response_latency_ms >= 0.0
    
    def test_calculate_desired_replicas_low_load(self, controller):
        """Test replica calculation with low load."""
        metrics = LoadMetrics(
            concurrent_calls=10,
            gpu_utilization_percent=20.0,
            request_queue_depth=5,
            avg_response_latency_ms=500.0
        )
        desired = controller.calculate_desired_replicas(
            current_replicas=3,
            metrics=metrics
        )
        # Should respect min_replicas
        assert desired == 1
    
    def test_calculate_desired_replicas_high_load(self, controller):
        """Test replica calculation with high load."""
        metrics = LoadMetrics(
            concurrent_calls=500,
            gpu_utilization_percent=90.0,
            request_queue_depth=200,
            avg_response_latency_ms=1200.0
        )
        desired = controller.calculate_desired_replicas(
            current_replicas=3,
            metrics=metrics
        )
        # Should scale up based on concurrent calls (500/50 = 10)
        assert desired == 10
    
    def test_calculate_desired_replicas_respects_max(self, controller):
        """Test that desired replicas respects max limit."""
        metrics = LoadMetrics(
            concurrent_calls=1000,
            gpu_utilization_percent=95.0,
            request_queue_depth=500,
            avg_response_latency_ms=1500.0
        )
        desired = controller.calculate_desired_replicas(
            current_replicas=5,
            metrics=metrics
        )
        # Should not exceed max_replicas (10)
        assert desired <= controller.policy.max_replicas
    
    def test_calculate_desired_replicas_respects_min(self, controller):
        """Test that desired replicas respects min limit."""
        metrics = LoadMetrics(
            concurrent_calls=0,
            gpu_utilization_percent=0.0,
            request_queue_depth=0,
            avg_response_latency_ms=0.0
        )
        desired = controller.calculate_desired_replicas(
            current_replicas=5,
            metrics=metrics
        )
        # Should not go below min_replicas (1)
        assert desired >= controller.policy.min_replicas
    
    def test_should_scale_up_high_utilization(self, controller):
        """Test scale up decision with high GPU utilization."""
        metrics = LoadMetrics(
            concurrent_calls=200,
            gpu_utilization_percent=85.0,  # Above scale_up_threshold (80%)
            request_queue_depth=100,
            avg_response_latency_ms=900.0
        )
        should_scale = controller.should_scale(
            current_replicas=2,
            desired_replicas=4,
            metrics=metrics
        )
        assert should_scale is True
    
    def test_should_scale_down_low_utilization(self, controller):
        """Test scale down decision with low GPU utilization."""
        metrics = LoadMetrics(
            concurrent_calls=20,
            gpu_utilization_percent=25.0,  # Below scale_down_threshold (30%)
            request_queue_depth=5,
            avg_response_latency_ms=400.0
        )
        should_scale = controller.should_scale(
            current_replicas=5,
            desired_replicas=2,
            metrics=metrics
        )
        assert should_scale is True
    
    def test_should_not_scale_same_replicas(self, controller):
        """Test that scaling is not triggered when replicas match."""
        metrics = LoadMetrics(
            concurrent_calls=100,
            gpu_utilization_percent=70.0,
            request_queue_depth=50,
            avg_response_latency_ms=600.0
        )
        should_scale = controller.should_scale(
            current_replicas=3,
            desired_replicas=3,
            metrics=metrics
        )
        assert should_scale is False
    
    def test_should_not_scale_moderate_utilization(self, controller):
        """Test that scaling is not triggered with moderate utilization."""
        metrics = LoadMetrics(
            concurrent_calls=150,
            gpu_utilization_percent=60.0,  # Between thresholds
            request_queue_depth=75,
            avg_response_latency_ms=700.0
        )
        should_scale = controller.should_scale(
            current_replicas=3,
            desired_replicas=4,
            metrics=metrics
        )
        assert should_scale is False
    
    def test_scale_up(self, controller):
        """Test scale up operation."""
        # Should not raise exception
        controller.scale_up(5)
    
    def test_scale_down(self, controller):
        """Test scale down operation."""
        # Should not raise exception
        controller.scale_down(2)
    
    def test_configure_scaling_policy(self, controller):
        """Test updating scaling policy."""
        new_policy = ScalingPolicy(
            min_replicas=2,
            max_replicas=20,
            target_concurrent_calls_per_replica=100,
            target_gpu_utilization_percent=80.0,
            scale_up_threshold_percent=85.0,
            scale_down_threshold_percent=40.0,
            cooldown_period_seconds=600
        )
        controller.configure_scaling_policy(new_policy)
        assert controller.policy == new_policy
        assert controller.policy.min_replicas == 2
        assert controller.policy.max_replicas == 20
    
    def test_optimize_cost_prefer_cloud(self, controller):
        """Test cost optimization preferring cloud APIs."""
        metrics = LoadMetrics(
            concurrent_calls=10,
            gpu_utilization_percent=15.0,
            request_queue_depth=5,
            avg_response_latency_ms=500.0
        )
        strategy = controller.optimize_cost(
            metrics=metrics,
            cloud_cost_per_call=0.01,
            local_cost_per_call=0.05
        )
        assert strategy == "cloud"
    
    def test_optimize_cost_prefer_local(self, controller):
        """Test cost optimization preferring local inference."""
        metrics = LoadMetrics(
            concurrent_calls=1000,
            gpu_utilization_percent=85.0,
            request_queue_depth=200,
            avg_response_latency_ms=800.0
        )
        strategy = controller.optimize_cost(
            metrics=metrics,
            cloud_cost_per_call=0.05,
            local_cost_per_call=0.01
        )
        assert strategy == "local"
    
    def test_optimize_cost_no_calls(self, controller):
        """Test cost optimization with no active calls."""
        metrics = LoadMetrics(
            concurrent_calls=0,
            gpu_utilization_percent=0.0,
            request_queue_depth=0,
            avg_response_latency_ms=0.0
        )
        strategy = controller.optimize_cost(
            metrics=metrics,
            cloud_cost_per_call=0.01,
            local_cost_per_call=0.01
        )
        # Should prefer cloud when no load (no GPU needed)
        assert strategy == "cloud"
    
    def test_optimize_cost_hybrid(self, controller):
        """Test cost optimization recommending hybrid approach."""
        metrics = LoadMetrics(
            concurrent_calls=100,
            gpu_utilization_percent=50.0,
            request_queue_depth=50,
            avg_response_latency_ms=600.0
        )
        # With equal costs and fixed GPU cost, the ratio will be > 1
        # So we need to adjust the test to create a true hybrid scenario
        strategy = controller.optimize_cost(
            metrics=metrics,
            cloud_cost_per_call=0.03,  # Cloud: 100 * 0.03 = 3.0
            local_cost_per_call=0.02   # Local: 100 * 0.02 + 1.0 = 3.0
        )
        # When costs are similar (ratio = 1.0), should recommend hybrid
        assert strategy == "hybrid"


class TestLoadMetrics:
    """Test suite for LoadMetrics dataclass."""
    
    def test_load_metrics_creation(self):
        """Test creating LoadMetrics instance."""
        metrics = LoadMetrics(
            concurrent_calls=100,
            gpu_utilization_percent=75.5,
            request_queue_depth=50,
            avg_response_latency_ms=650.0
        )
        assert metrics.concurrent_calls == 100
        assert metrics.gpu_utilization_percent == 75.5
        assert metrics.request_queue_depth == 50
        assert metrics.avg_response_latency_ms == 650.0


class TestScalingPolicy:
    """Test suite for ScalingPolicy dataclass."""
    
    def test_scaling_policy_creation(self):
        """Test creating ScalingPolicy instance."""
        policy = ScalingPolicy(
            min_replicas=1,
            max_replicas=10,
            target_concurrent_calls_per_replica=50,
            target_gpu_utilization_percent=70.0,
            scale_up_threshold_percent=80.0,
            scale_down_threshold_percent=30.0,
            cooldown_period_seconds=300
        )
        assert policy.min_replicas == 1
        assert policy.max_replicas == 10
        assert policy.target_concurrent_calls_per_replica == 50
        assert policy.target_gpu_utilization_percent == 70.0
        assert policy.scale_up_threshold_percent == 80.0
        assert policy.scale_down_threshold_percent == 30.0
        assert policy.cooldown_period_seconds == 300
