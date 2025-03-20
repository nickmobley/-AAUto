"""
Core monitoring system for tracking performance, adaptation effectiveness, 
system health, anomalies, and analytics reporting.
"""
import asyncio
import logging
import time
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Assuming these modules exist in our architecture
from src.core.uncertainty import UncertaintyEstimator
from src.core.validation import ValidationSystem
from src.core.decision import DecisionFramework
from src.core.events import EventBus
from src.core.config import ConfigManager
from src.core.logging import AdaptiveLogger

logger = AdaptiveLogger(__name__)

class MetricType(Enum):
    """Types of metrics being monitored."""
    PERFORMANCE = "performance"
    ADAPTATION = "adaptation"
    HEALTH = "health"
    RESOURCE = "resource"
    ANOMALY = "anomaly"
    UNCERTAINTY = "uncertainty"
    DECISION = "decision"
    VALIDATION = "validation"

@dataclass
class Metric:
    """Data class for storing metric information."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    type: MetricType = MetricType.PERFORMANCE
    context: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[float] = None

class PerformanceTracker:
    """Tracks performance metrics across the system."""
    
    def __init__(self, window_size: int = 1000):
        self.metrics: Dict[str, List[Metric]] = {}
        self.window_size = window_size
        
    async def record_metric(self, metric: Metric) -> None:
        """Record a new metric value."""
        if metric.name not in self.metrics:
            self.metrics[metric.name] = []
            
        self.metrics[metric.name].append(metric)
        
        # Maintain window size
        if len(self.metrics[metric.name]) > self.window_size:
            self.metrics[metric.name].pop(0)
            
    def get_metrics(self, metric_name: str, limit: int = 100) -> List[Metric]:
        """Get recent metrics for a given name."""
        if metric_name not in self.metrics:
            return []
        
        return self.metrics[metric_name][-limit:]
    
    def get_average(self, metric_name: str, window: int = 100) -> Optional[float]:
        """Calculate average value for a metric over a window."""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = [m.value for m in self.metrics[metric_name][-window:]]
        return sum(values) / len(values) if values else None
    
    def get_trend(self, metric_name: str, window: int = 100) -> Optional[float]:
        """Calculate the trend (slope) of a metric over time."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < 2:
            return None
        
        metrics = self.metrics[metric_name][-window:]
        if len(metrics) < 2:
            return 0.0
            
        x = np.array([(m.timestamp - metrics[0].timestamp).total_seconds() for m in metrics])
        y = np.array([m.value for m in metrics])
        
        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope

class AdaptationTracker:
    """Tracks the effectiveness of system adaptations."""
    
    def __init__(self):
        self.adaptations: Dict[str, List[Dict[str, Any]]] = {}
        
    async def record_adaptation(
        self, 
        component: str, 
        parameters: Dict[str, Any], 
        performance_before: Dict[str, float],
        expected_improvement: Dict[str, float]
    ) -> str:
        """Record a new adaptation."""
        adaptation_id = f"{component}_{time.time()}"
        
        if component not in self.adaptations:
            self.adaptations[component] = []
            
        adaptation = {
            "id": adaptation_id,
            "timestamp": datetime.now(),
            "parameters": parameters,
            "performance_before": performance_before,
            "expected_improvement": expected_improvement,
            "performance_after": {},
            "actual_improvement": {},
            "effective": None
        }
        
        self.adaptations[component].append(adaptation)
        return adaptation_id
    
    async def update_adaptation_result(
        self,
        adaptation_id: str,
        performance_after: Dict[str, float]
    ) -> None:
        """Update an adaptation with actual performance results."""
        for component, adaptations in self.adaptations.items():
            for adaptation in adaptations:
                if adaptation["id"] == adaptation_id:
                    adaptation["performance_after"] = performance_after
                    
                    # Calculate actual improvement
                    actual_improvement = {}
                    effective_count = 0
                    total_metrics = 0
                    
                    for metric, before_value in adaptation["performance_before"].items():
                        if metric in performance_after:
                            improvement = performance_after[metric] - before_value
                            actual_improvement[metric] = improvement
                            
                            # Check if improvement matches or exceeds expectation
                            if metric in adaptation["expected_improvement"]:
                                expected = adaptation["expected_improvement"][metric]
                                if (expected >= 0 and improvement >= expected * 0.8) or \
                                   (expected < 0 and improvement <= expected * 0.8):
                                    effective_count += 1
                            total_metrics += 1
                    
                    adaptation["actual_improvement"] = actual_improvement
                    
                    # Determine overall effectiveness
                    if total_metrics > 0:
                        effectiveness_ratio = effective_count / total_metrics
                        adaptation["effective"] = effectiveness_ratio >= 0.7
                    
                    return
                    
        logger.warning(f"Adaptation ID {adaptation_id} not found for update")
        
    def get_effectiveness_rate(self, component: str, window: int = 10) -> Optional[float]:
        """Get the rate of effective adaptations for a component."""
        if component not in self.adaptations:
            return None
            
        recent_adaptations = self.adaptations[component][-window:]
        if not recent_adaptations:
            return None
            
        effective_count = sum(1 for a in recent_adaptations if a.get("effective") is True)
        return effective_count / len(recent_adaptations)

class HealthMonitor:
    """Monitors system health across components."""
    
    def __init__(self, thresholds: Dict[str, Dict[str, float]] = None):
        self.component_health: Dict[str, Dict[str, Any]] = {}
        self.thresholds = thresholds or {}
        self.checks: Dict[str, Callable] = {}
        
    def register_health_check(self, component: str, check_func: Callable) -> None:
        """Register a health check function for a component."""
        self.checks[component] = check_func
        
    async def check_component_health(self, component: str) -> Dict[str, Any]:
        """Run health check for a specific component."""
        if component not in self.checks:
            return {"status": "unknown", "reason": "No health check registered"}
            
        try:
            result = await self.checks[component]()
            self.component_health[component] = {
                "status": result.get("status", "unknown"),
                "metrics": result.get("metrics", {}),
                "timestamp": datetime.now(),
                "details": result.get("details", {})
            }
        except Exception as e:
            self.component_health[component] = {
                "status": "error",
                "reason": str(e),
                "timestamp": datetime.now()
            }
            
        return self.component_health[component]
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check health of all registered components."""
        results = {}
        for component in self.checks:
            results[component] = await self.check_component_health(component)
            
        # Determine overall system health
        error_components = [c for c, r in results.items() if r["status"] == "error"]
        warning_components = [c for c, r in results.items() if r["status"] == "warning"]
        
        if error_components:
            system_status = "error"
            reason = f"Errors in components: {', '.join(error_components)}"
        elif warning_components:
            system_status = "warning"
            reason = f"Warnings in components: {', '.join(warning_components)}"
        else:
            system_status = "healthy"
            reason = "All components healthy"
            
        return {
            "status": system_status,
            "reason": reason,
            "timestamp": datetime.now(),
            "components": results
        }

class AnomalyDetector:
    """Detects anomalies in system behavior and metrics."""
    
    def __init__(self, sensitivity: float = 3.0):
        self.metric_history: Dict[str, List[float]] = {}
        self.sensitivity = sensitivity  # Number of standard deviations for anomaly detection
        self.anomalies: List[Dict[str, Any]] = []
        
    async def update_metric(self, metric_name: str, value: float) -> bool:
        """Update a metric and check for anomalies."""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = []
            
        self.metric_history[metric_name].append(value)
        
        # Keep last 1000 values
        if len(self.metric_history[metric_name]) > 1000:
            self.metric_history[metric_name].pop(0)
            
        # Need at least 30 data points for statistical significance
        if len(self.metric_history[metric_name]) < 30:
            return False
            
        return await self._check_anomaly(metric_name, value)
    
    async def _check_anomaly(self, metric_name: str, current_value: float) -> bool:
        """Check if the current value is anomalous."""
        history = self.metric_history[metric_name]
        mean = sum(history[:-1]) / len(history[:-1])
        std_dev = np.std(history[:-1])
        
        if std_dev == 0:
            return False
            
        z_score = abs(current_value - mean) / std_dev
        
        is_anomaly = z_score > self.sensitivity
        
        if is_anomaly:
            anomaly = {
                "metric": metric_name,
                "value": current_value,
                "mean": mean,
                "std_dev": std_dev,
                "z_score": z_score,
                "timestamp": datetime.now()
            }
            self.anomalies.append(anomaly)
            logger.warning(f"Anomaly detected: {anomaly}")
            
        return is_anomaly
    
    def get_recent_anomalies(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get the most recent anomalies."""
        return self.anomalies[-limit:]
    
    def get_anomaly_frequency(self, metric_name: str = None, days: int = 1) -> Dict[str, int]:
        """Get frequency of anomalies by metric."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_anomalies = [a for a in self.anomalies 
                           if a["timestamp"].timestamp() > cutoff]
        
        if metric_name:
            recent_anomalies = [a for a in recent_anomalies if a["metric"] == metric_name]
            
        frequency = {}
        for anomaly in recent_anomalies:
            metric = anomaly["metric"]
            if metric not in frequency:
                frequency[metric] = 0
            frequency[metric] += 1
            
        return frequency

class MonitoringSystem:
    """Core monitoring system that integrates all monitoring capabilities."""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
        validation_system: Optional[ValidationSystem] = None,
        decision_framework: Optional[DecisionFramework] = None,
        event_bus: Optional[EventBus] = None
    ):
        self.performance_tracker = PerformanceTracker()
        self.adaptation_tracker = AdaptationTracker()
        self.health_monitor = HealthMonitor()
        self.anomaly_detector = AnomalyDetector()
        
        self.config_manager = config_manager
        self.uncertainty_estimator = uncertainty_estimator
        self.validation_system = validation_system
        self.decision_framework = decision_framework
        self.event_bus = event_bus
        
        self.running = False
        self.monitoring_task = None
        
    async def start(self) -> None:
        """Start the monitoring system."""
        if self.running:
            return
            
        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring system started")
        
    async def stop(self) -> None:
        """Stop the monitoring system."""
        if not self.running:
            return
            
        self.running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            
        logger.info("Monitoring system stopped")
        
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop that collects metrics and performs analysis."""
        while self.running:
            try:
                await self._collect_metrics()
                await self._check_health()
                await self._analyze_performance()
                await asyncio.sleep(1)  # Adjust based on required monitoring frequency
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Back off on errors
                
    async def _collect_metrics(self) -> None:
        """Collect metrics from all system components."""
        # Implement logic to collect metrics from components
        pass
        
    async def _check_health(self) -> None:
        """Check system health."""
        health_status = await self.health_monitor.check_system_health()
        if health_status["status"] != "healthy":
            logger.warning(f"System health issue: {health_status['reason']}")
            
    async def _analyze_performance(self) -> None:
        """Analyze system performance and detect anomalies."""
        # Implement performance analysis logic
        pass
        
    async def record_metric(
        self, 
        name: str, 
        value: float, 
        metric_type: MetricType = MetricType.PERFORMANCE,
        context: Dict[str, Any] = None
    ) -> None:
        """Record a metric and check for anomalies."""
        context = context or {}
        
        # Add uncertainty if available
        uncertainty = None
        if self.uncertainty_estimator and metric_type in [MetricType.PERFORMANCE, MetricType.ADAPTATION]:
            try:
                uncertainty = await self.uncertainty_estimator.estimate_uncertainty(
                    

