"""
Scaling Controller for KEDA-based GPU auto-scaling.

This module provides the ScalingController class that manages auto-scaling
decisions for GPU resources based on system load metrics.
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoadMetrics:
    """Current system load metrics."""
    concurrent_calls: int
    gpu_utilization_percent: float
    request_queue_depth: int
    avg_response_latency_ms: float


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_replicas: int
    max_replicas: int
    target_concurrent_calls_per_replica: int
    target_gpu_utilization_percent: float
    scale_up_threshold_percent: float
    scale_down_threshold_percent: float
    cooldown_period_seconds: int


class ScalingController:
    """
    Controller for managing KEDA-based auto-scaling of GPU resources.
    
    This controller analyzes system load metrics and makes scaling decisions
    to optimize resource utilization and cost while maintaining performance.
    """
    
    def __init__(self, policy: ScalingPolicy):
        """
        Initialize the scaling controller.
        
        Args:
            policy: Scaling policy configuration
        """
        self.policy = policy
        self._last_scale_time: Optional[float] = None
        logger.info(f"Initialized ScalingController with policy: {policy}")
    
    def get_current_load(self) -> LoadMetrics:
        """
        Retrieve current system load metrics.
        
        In production, this would query Prometheus or other monitoring systems.
        For now, this is a placeholder that returns mock data.
        
        Returns:
            Current load metrics
        """
        # TODO: Implement actual metrics collection from Prometheus
        return LoadMetrics(
            concurrent_calls=0,
            gpu_utilization_percent=0.0,
            request_queue_depth=0,
            avg_response_latency_ms=0.0
        )
    
    def calculate_desired_replicas(self, current_replicas: int, metrics: LoadMetrics) -> int:
        """
        Calculate desired number of replicas based on current metrics.
        
        Args:
            current_replicas: Current number of replicas
            metrics: Current load metrics
            
        Returns:
            Desired number of replicas
        """
        # Calculate based on concurrent calls
        calls_based_replicas = max(
            1,
            int(metrics.concurrent_calls / self.policy.target_concurrent_calls_per_replica)
        )
        
        # Calculate based on GPU utilization
        if metrics.gpu_utilization_percent > 0:
            gpu_based_replicas = max(
                1,
                int(current_replicas * metrics.gpu_utilization_percent / 
                    self.policy.target_gpu_utilization_percent)
            )
        else:
            gpu_based_replicas = current_replicas
        
        # Take the maximum to ensure we have enough capacity
        desired_replicas = max(calls_based_replicas, gpu_based_replicas)
        
        # Apply min/max constraints
        desired_replicas = max(self.policy.min_replicas, 
                              min(self.policy.max_replicas, desired_replicas))
        
        logger.debug(
            f"Calculated desired replicas: {desired_replicas} "
            f"(calls: {calls_based_replicas}, gpu: {gpu_based_replicas})"
        )
        
        return desired_replicas
    
    def should_scale(self, current_replicas: int, desired_replicas: int, 
                    metrics: LoadMetrics) -> bool:
        """
        Determine if scaling action should be taken.
        
        Args:
            current_replicas: Current number of replicas
            desired_replicas: Desired number of replicas
            metrics: Current load metrics
            
        Returns:
            True if scaling should occur, False otherwise
        """
        if current_replicas == desired_replicas:
            return False
        
        # Check if we're in cooldown period
        # TODO: Implement cooldown period checking
        
        # Scale up if utilization exceeds threshold
        if desired_replicas > current_replicas:
            utilization = metrics.gpu_utilization_percent
            if utilization >= self.policy.scale_up_threshold_percent:
                logger.info(
                    f"Scale up triggered: utilization {utilization}% >= "
                    f"threshold {self.policy.scale_up_threshold_percent}%"
                )
                return True
        
        # Scale down if utilization is below threshold
        if desired_replicas < current_replicas:
            utilization = metrics.gpu_utilization_percent
            if utilization <= self.policy.scale_down_threshold_percent:
                logger.info(
                    f"Scale down triggered: utilization {utilization}% <= "
                    f"threshold {self.policy.scale_down_threshold_percent}%"
                )
                return True
        
        return False
    
    def scale_up(self, replicas: int) -> None:
        """
        Increase GPU pod count.
        
        In production, this would interact with Kubernetes API or KEDA.
        
        Args:
            replicas: Target number of replicas
        """
        logger.info(f"Scaling up to {replicas} replicas")
        # TODO: Implement actual scaling via Kubernetes API
    
    def scale_down(self, replicas: int) -> None:
        """
        Decrease GPU pod count.
        
        In production, this would interact with Kubernetes API or KEDA.
        
        Args:
            replicas: Target number of replicas
        """
        logger.info(f"Scaling down to {replicas} replicas")
        # TODO: Implement actual scaling via Kubernetes API
    
    def configure_scaling_policy(self, policy: ScalingPolicy) -> None:
        """
        Update the scaling policy configuration.
        
        Args:
            policy: New scaling policy
        """
        self.policy = policy
        logger.info(f"Updated scaling policy: {policy}")
    
    def optimize_cost(self, metrics: LoadMetrics, cloud_cost_per_call: float,
                     local_cost_per_call: float) -> str:
        """
        Determine optimal deployment strategy based on cost.
        
        Args:
            metrics: Current load metrics
            cloud_cost_per_call: Cost per call using cloud LLM APIs
            local_cost_per_call: Cost per call using local GPU inference
            
        Returns:
            Recommended strategy: "cloud", "local", or "hybrid"
        """
        # Simple cost optimization logic
        if metrics.concurrent_calls == 0:
            return "cloud"  # No need for GPU resources
        
        # Calculate costs
        cloud_total_cost = metrics.concurrent_calls * cloud_cost_per_call
        local_total_cost = metrics.concurrent_calls * local_cost_per_call
        
        # Add fixed GPU cost (amortized) - only for local
        gpu_fixed_cost = 1.0  # Placeholder
        local_total_cost_with_fixed = local_total_cost + gpu_fixed_cost
        
        # Calculate cost difference ratio
        if cloud_total_cost > 0:
            local_ratio = local_total_cost_with_fixed / cloud_total_cost
        else:
            local_ratio = float('inf')
        
        if local_ratio < 0.7:  # Local is 30%+ cheaper
            logger.info(
                f"Cost optimization: local inference recommended "
                f"(${local_total_cost_with_fixed:.2f} vs ${cloud_total_cost:.2f})"
            )
            return "local"
        elif local_ratio > 1.43:  # Cloud is 30%+ cheaper (1/0.7 ≈ 1.43)
            logger.info(
                f"Cost optimization: cloud APIs recommended "
                f"(${cloud_total_cost:.2f} vs ${local_total_cost_with_fixed:.2f})"
            )
            return "cloud"
        else:
            logger.info("Cost optimization: hybrid approach recommended")
            return "hybrid"
