import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Assuming these imports work with the rest of the project structure
from src.core import AdaptiveSystem
from src.analysis.market import MarketAnalyzer
from src.risk.portfolio import AdaptiveRiskManager
from src.execution.optimization import ExecutionOptimizer
from src.strategy import StrategyManager


@dataclass
class PerformanceMetric:
    """Represents a single performance metric with metadata."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    category: str = "general"
    tags: Set[str] = field(default_factory=set)
    threshold: Optional[float] = None
    is_critical: bool = False
    
    def is_below_threshold(self) -> bool:
        """Check if the metric is below its threshold, if one exists."""
        return self.threshold is not None and self.value < self.threshold


@dataclass
class AdaptationEvent:
    """Tracks a single adaptation event in the system."""
    component: str
    adaptation_type: str
    timestamp: datetime = field(default_factory=datetime.now)
    previous_state: Any = None
    new_state: Any = None
    trigger_metrics: List[PerformanceMetric] = field(default_factory=list)
    success_rating: Optional[float] = None
    
    def mark_success(self, rating: float) -> None:
        """Rate the success of this adaptation from 0.0 to 1.0."""
        self.success_rating = max(0.0, min(1.0, rating))


class PerformanceTracker:
    """Tracks and stores performance metrics across the system."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.adaptation_events: List[AdaptationEvent] = []
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.callbacks: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
    
    def record_metric(self, metric: PerformanceMetric) -> None:
        """Record a new performance metric."""
        self.metrics.append(metric)
        
        # Update metric history
        if metric.name not in self.metric_history:
            self.metric_history[metric.name] = []
        self.metric_history[metric.name].append((metric.timestamp, metric.value))
        
        # Check thresholds and trigger callbacks
        if metric.is_below_threshold() and metric.is_critical:
            self.logger.warning(f"Critical metric {metric.name} below threshold: {metric.value} < {metric.threshold}")
            self._trigger_callbacks("threshold_breach", metric)
    
    def record_adaptation(self, event: AdaptationEvent) -> None:
        """Record an adaptation event."""
        self.adaptation_events.append(event)
        self._trigger_callbacks("adaptation", event)
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for specific event types."""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, data: Any) -> None:
        """Trigger all registered callbacks for an event type."""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Callback error: {str(e)}")
    
    def get_metrics_by_category(self, category: str) -> List[PerformanceMetric]:
        """Get all metrics for a specific category."""
        return [m for m in self.metrics if m.category == category]
    
    def get_metrics_by_tag(self, tag: str) -> List[PerformanceMetric]:
        """Get all metrics with a specific tag."""
        return [m for m in self.metrics if tag in m.tags]
    
    def get_metric_trend(self, metric_name: str, window: int = 10) -> Optional[float]:
        """Calculate the trend (slope) of a metric over the last window values."""
        if metric_name not in self.metric_history:
            return None
            
        history = self.metric_history[metric_name]
        if len(history) < 2:
            return 0.0
            
        recent = history[-window:] if len(history) >= window else history
        x = np.array([(entry[0] - recent[0][0]).total_seconds() for entry in recent])
        y = np.array([entry[1] for entry in recent])
        
        # Simple linear regression to get slope
        slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0.0
        return slope


class StrategyAnalyzer:
    """Analyzes the effectiveness of strategy adaptations."""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.strategy_performance: Dict[str, Dict[str, List[float]]] = {}
        self.adaptation_success_rates: Dict[str, List[float]] = {}
        
    def register_strategy(self, strategy_id: str) -> None:
        """Register a new strategy for tracking."""
        if strategy_id not in self.strategy_performance:
            self.strategy_performance[strategy_id] = {
                "returns": [],
                "drawdowns": [],
                "sharpe": [],
                "adaptation_count": []
            }
            
    def update_strategy_metrics(self, strategy_id: str, metrics: Dict[str, float]) -> None:
        """Update performance metrics for a strategy."""
        if strategy_id not in self.strategy_performance:
            self.register_strategy(strategy_id)
            
        for metric_name, value in metrics.items():
            if metric_name in self.strategy_performance[strategy_id]:
                self.strategy_performance[strategy_id][metric_name].append(value)
    
    def evaluate_adaptation(self, strategy_id: str, adaptation_event: AdaptationEvent, 
                           before_metrics: Dict[str, float], after_metrics: Dict[str, float]) -> float:
        """Evaluate how effective a strategy adaptation was based on before/after metrics."""
        improvement_scores = []
        
        for metric_name, before_value in before_metrics.items():
            if metric_name in after_metrics:
                after_value = after_metrics[metric_name]
                
                # Calculate percent improvement (positive is better)
                if before_value != 0:
                    pct_change = (after_value - before_value) / abs(before_value)
                    
                    # Convert to a 0-1 score (sigmoid-like)
                    # Positive changes map to >0.5, negative to <0.5
                    score = 0.5 + (1 / (1 + np.exp(-pct_change * 5))) / 2
                    improvement_scores.append(score)
        
        # Overall improvement score (average)
        avg_score = np.mean(improvement_scores) if improvement_scores else 0.5
        
        # Update adaptation event with success rating
        adaptation_event.mark_success(avg_score)
        
        # Update adaptation success rates
        adaptation_type = adaptation_event.adaptation_type
        if adaptation_type not in self.adaptation_success_rates:
            self.adaptation_success_rates[adaptation_type] = []
        self.adaptation_success_rates[adaptation_type].append(avg_score)
        
        return avg_score
        
    def get_adaptation_effectiveness(self, adaptation_type: Optional[str] = None, 
                                   window: int = 10) -> Dict[str, float]:
        """Get the effectiveness of adaptations, optionally filtered by type."""
        results = {}
        
        if adaptation_type:
            if adaptation_type in self.adaptation_success_rates:
                rates = self.adaptation_success_rates[adaptation_type]
                recent = rates[-window:] if len(rates) >= window else rates
                results[adaptation_type] = np.mean(recent) if recent else 0.0
        else:
            # Get effectiveness for all adaptation types
            for adapt_type, rates in self.adaptation_success_rates.items():
                recent = rates[-window:] if len(rates) >= window else rates
                results[adapt_type] = np.mean(recent) if recent else 0.0
                
        return results
    
    def get_best_performing_strategies(self, metric: str = "sharpe", top_n: int = 3) -> List[str]:
        """Get the top N best performing strategies based on a metric."""
        strategies = []
        
        for strategy_id, metrics in self.strategy_performance.items():
            if metric in metrics and metrics[metric]:
                # Use the most recent value
                strategies.append((strategy_id, metrics[metric][-1]))
                
        # Sort by metric value (descending) and take top_n
        strategies.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in strategies[:top_n]]


class RiskMonitor:
    """Monitors risk management adaptation and effectiveness."""
    
    def __init__(self, tracker: PerformanceTracker):
        self.tracker = tracker
        self.risk_metrics: Dict[str, List[float]] = {
            "var": [],          # Value at Risk
            "cvar": [],         # Conditional VaR
            "drawdown": [],     # Maximum drawdown
            "leverage": [],     # Leverage used
            "concentration": [] # Position concentration
        }
        self.breaches: List[Dict[str, Any]] = []
        self.adaptation_responses: List[Dict[str, Any]] = []
        
    def record_risk_metrics(self, metrics: Dict[str, float]) -> None:
        """Record current risk metrics."""
        for metric_name, value in metrics.items():
            if metric_name in self.risk_metrics:
                self.risk_metrics[metric_name].append(value)
                
                # Create a PerformanceMetric for the tracker
                perf_metric = PerformanceMetric(
                    name=f"risk_{metric_name}",
                    value=value,
                    category="risk",
                    tags={"risk", metric_name}
                )
                self.tracker.record_metric(perf_metric)
    
    def record_risk_breach(self, metric_name: str, threshold: float, 
                          actual: float, timestamp: datetime = None) -> None:
        """Record a risk threshold breach."""
        if timestamp is None:
            timestamp = datetime.now()
            
        breach = {
            "metric": metric_name,
            "threshold": threshold,
            "actual": actual,
            "timestamp": timestamp,
            "resolved": False,
            "resolution_time": None
        }
        self.breaches.append(breach)
        
        # Log the breach
        logging.warning(f"Risk breach: {metric_name} = {actual}, threshold = {threshold}")
    
    def record_adaptation_response(self, breach_idx: int, adaptation_event: AdaptationEvent) -> None:
        """Record an adaptation made in response to a risk breach."""
        if 0 <= breach_idx < len(self.breaches):
            response = {
                "breach_idx": breach_idx,
                "adaptation_event": adaptation_event,
                "timestamp": datetime.now(),
                "effectiveness": None  # To be filled later
            }
            self.adaptation_responses.append(response)
            
            # Mark the breach as being addressed
            self.breaches[breach_idx]["resolution_action"] = True
            
    def evaluate_risk_adaptations(self, window: int = 10) -> Dict[str, float]:
        """Evaluate the effectiveness of risk adaptations."""
        results = {}
        
        # Group adaptations by risk metric
        adaptations_by_metric = {}
        for response in self.adaptation_responses[-window:]:
            breach_idx = response["breach_idx"]
            if breach_idx < len(self.breaches):
                metric_name = self.breaches[breach_idx]["metric"]
                
                if metric_name not in adaptations_by_metric:
                    adaptations_by_metric[metric_name] = []
                    
                # Check if the adaptation has a success rating
                adaptation = response["adaptation_event"]
                if adaptation.success_rating is not None:
                    adaptations_by_metric[metric_name].append(adaptation.success_rating)
        
        # Calculate average success by metric
        for metric, ratings in adaptations_by_metric.items():
            if ratings:
                results[metric] = np.mean(ratings)
                
        return results
    
    def get_risk_trends(self) -> Dict[str, float]:
        """Get the trend direction for each risk metric."""
        trends = {}
        window = 10  # Look at the last 10 values
        
        for metric_name, values in self.risk_metrics.items():
            if len(values) < 2:
                trends[metric_name] = 0.0
                continue
                
            recent = values[-window:] if len(values) >= window else values
            x = np.arange(len(recent))
            y = np.array(recent)
            
            # Calculate slope of the trend
            slope = np.polyfit(x, y, 1)[0] if len(y) > 1 else 0.0
            trends[metric_name] = slope
            
        return trends


class OptimizationFeedback:
    """Provides real-time feedback for system optimization."""
    
    def __init__(self, tracker: PerformanceTracker, strategy_analyzer: StrategyAnalyzer, 
                risk_monitor: RiskMonitor):
        self.tracker = tracker
        self.strategy_analyzer = strategy_analyzer
        self.risk_monitor = risk_monitor
        self.optimization_suggestions: List[Dict[str, Any]] = []
        self.applied_optimizations: List[Dict[str, Any]] = []
        
    async def generate_insights(self) -> Dict[str, Any]:
        """Generate performance insights and optimization suggestions."""
        insights = {
            "metrics": self._analyze_metrics(),
            "strategies": self._analyze_strategies(),
            "risk": self._analyze_risk(),
            "adaptations": self._analyze_adaptations(),
            "suggestions": self._generate_suggestions()
        }
        return insights
    
    def _analyze_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics for insights."""
        analysis = {
            "critical_breaches": [],
            "improving_metrics": [],
            "declining_metrics": []
        }
        
        # Find critical breaches
        critical_metrics = [m for m in self.tracker.metrics if m.is_critical]
        breaches = [m for m in critical_metrics if m.is_below_threshold()]
        analysis["critical_breaches"] = breaches
        
        # Analyze trends for all metrics
        for name in self.tracker.metric_history:
            trend = self.tracker.get_metric_trend(name)
            if trend is not None:
                if trend > 0.01:  # Positive trend
                    analysis["improving_metrics"].append((name, trend))
                elif trend < -0.01:  # Negative trend
                    analysis["declining_metrics"].

