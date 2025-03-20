"""
Core adaptive system for self-improvement and optimization.

This module provides the foundation for a self-improving AI system with:
1. Performance tracking and optimization
2. System state management
3. Integration hooks for all subsystems
4. Automatic learning and adaptation
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Set, Tuple, Type, TypeVar, Union, cast
import uuid
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

# Type definitions
T = TypeVar('T')
MetricValue = Union[float, int, bool, str]
MetricsDict = Dict[str, MetricValue]
StateDict = Dict[str, Any]
HookFunction = Callable[..., Any]


class SystemState(Enum):
    """System operational states."""
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    OPTIMIZING = auto()
    PAUSED = auto()
    SHUTTING_DOWN = auto()
    ERROR = auto()


class OptimizationStrategy(Enum):
    """Available optimization strategies."""
    GRADIENT_DESCENT = auto()
    BAYESIAN = auto()
    EVOLUTIONARY = auto()
    REINFORCEMENT_LEARNING = auto()
    MANUAL = auto()


@dataclass
class PerformanceMetrics:
    """Container for system performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: float = 0.0
    memory_usage: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    throughput: float = 0.0
    latency: float = 0.0
    custom_metrics: MetricsDict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        result = {
            "timestamp": self.timestamp.isoformat(),
            "execution_time": self.execution_time,
            "memory_usage": self.memory_usage,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "throughput": self.throughput,
            "latency": self.latency,
        }
        result.update(self.custom_metrics)
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create metrics from dictionary."""
        custom_metrics = {k: v for k, v in data.items() 
                         if k not in ["timestamp", "execution_time", "memory_usage", 
                                     "success_rate", "error_count", "throughput", "latency"]}
        
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            execution_time=float(data["execution_time"]),
            memory_usage=float(data["memory_usage"]),
            success_rate=float(data["success_rate"]),
            error_count=int(data["error_count"]),
            throughput=float(data["throughput"]),
            latency=float(data["latency"]),
            custom_metrics=custom_metrics
        )


class Subsystem(Protocol):
    """Protocol defining the interface for subsystems."""
    
    @property
    def name(self) -> str:
        """Get the subsystem name."""
        ...
    
    async def initialize(self) -> bool:
        """Initialize the subsystem."""
        ...
    
    async def shutdown(self) -> bool:
        """Shutdown the subsystem."""
        ...
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get performance metrics for this subsystem."""
        ...
    
    def get_state(self) -> StateDict:
        """Get the current state."""
        ...
    
    def set_state(self, state: StateDict) -> bool:
        """Set the current state."""
        ...


class OptimizationResult:
    """Results of an optimization run."""
    
    def __init__(self, 
                 parameters: Dict[str, Any],
                 metrics_before: PerformanceMetrics,
                 metrics_after: PerformanceMetrics,
                 timestamp: datetime = None):
        self.parameters = parameters
        self.metrics_before = metrics_before
        self.metrics_after = metrics_after
        self.timestamp = timestamp or datetime.now()
        self.improvement_percentage: Dict[str, float] = self._calculate_improvement()
    
    def _calculate_improvement(self) -> Dict[str, float]:
        """Calculate percentage improvement for each metric."""
        improvements = {}
        
        before_dict = self.metrics_before.to_dict()
        after_dict = self.metrics_after.to_dict()
        
        for key, after_value in after_dict.items():
            if key == "timestamp":
                continue
                
            before_value = before_dict.get(key, 0)
            
            # Skip division by zero
            if before_value == 0:
                improvements[key] = 100.0 if after_value > 0 else 0.0
                continue
                
            # For error_count, lower is better
            if key == "error_count":
                improvements[key] = ((before_value - after_value) / before_value) * 100
            else:
                # For other metrics, higher is better
                improvements[key] = ((after_value - before_value) / before_value) * 100
                
        return improvements
    
    def is_improvement(self) -> bool:
        """Determine if this optimization improved the system."""
        # Consider it an improvement if more metrics improved than degraded
        improved_count = sum(1 for value in self.improvement_percentage.values() if value > 0)
        degraded_count = sum(1 for value in self.improvement_percentage.values() if value < 0)
        return improved_count > degraded_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for storage."""
        return {
            "parameters": self.parameters,
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "improvement_percentage": self.improvement_percentage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationResult':
        """Create result from dictionary."""
        result = cls(
            parameters=data["parameters"],
            metrics_before=PerformanceMetrics.from_dict(data["metrics_before"]),
            metrics_after=PerformanceMetrics.from_dict(data["metrics_after"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )
        result.improvement_percentage = data["improvement_percentage"]
        return result


class Event:
    """Base event class for the event system."""
    
    def __init__(self, event_type: str, source: str, data: Any = None):
        self.id = str(uuid.uuid4())
        self.event_type = event_type
        self.source = source
        self.data = data
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp.isoformat()
        }


class EventBus:
    """Event bus for system-wide event distribution."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], Any]]] = {}
        self._history: List[Event] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        
        # Get subscribers for this event type
        event_subscribers = self._subscribers.get(event.event_type, [])
        all_subscribers = self._subscribers.get("*", [])
        
        # Execute all subscriber callbacks
        for callback in event_subscribers + all_subscribers:
            try:
                # Handle both sync and async callbacks
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logging.error(f"Error in event subscriber: {e}")
    
    def subscribe(self, event_type: str, callback: Callable[[Event], Any]) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], Any]) -> None:
        """Unsubscribe from events of a specific type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]
    
    async def get_history(self, event_type: Optional[str] = None, 
                         limit: int = 100) -> List[Event]:
        """Get event history, optionally filtered by type."""
        async with self._lock:
            if event_type is None:
                return self._history[-limit:]
            return [e for e in self._history if e.event_type == event_type][-limit:]


class AdaptiveSystem:
    """Core adaptive system with self-improvement capabilities."""
    
    def __init__(self, 
                 name: str,
                 config_path: Optional[Path] = None,
                 optimization_strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN,
                 auto_optimize_interval: Optional[int] = None):
        self.name = name
        self.id = str(uuid.uuid4())
        self.state = SystemState.INITIALIZING
        self.started_at = datetime.now()
        self.config_path = config_path or Path("./config")
        
        # System components
        self.event_bus = EventBus()
        self.subsystems: Dict[str, Subsystem] = {}
        self.optimization_strategy = optimization_strategy
        self.auto_optimize_interval = auto_optimize_interval
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_history: List[OptimizationResult] = []
        self._optimization_task: Optional[asyncio.Task] = None
        
        # Hooks for extensibility
        self._before_start_hooks: List[HookFunction] = []
        self._after_start_hooks: List[HookFunction] = []
        self._before_shutdown_hooks: List[HookFunction] = []
        self._after_shutdown_hooks: List[HookFunction] = []
        
        # State management
        self._state_data: StateDict = {}
        self._last_metrics = PerformanceMetrics()
        self._logger = logging.getLogger(f"{self.name}.adaptive_system")
    
    async def initialize(self) -> bool:
        """Initialize the adaptive system and all registered subsystems."""
        try:
            self._logger.info(f"Initializing adaptive system: {self.name}")
            
            # Create config directory if it doesn't exist
            self.config_path.mkdir(parents=True, exist_ok=True)
            
            # Load previous state if available
            await self._load_state()
            
            # Initialize all subsystems
            for name, subsystem in self.subsystems.items():
                try:
                    self._logger.info(f"Initializing subsystem: {name}")
                    success = await subsystem.initialize()
                    if not success:
                        self._logger.error(f"Failed to initialize subsystem: {name}")
                except Exception as e:
                    self._logger.error(f"Error initializing subsystem {name}: {e}")
            
            self.state = SystemState.READY
            await self.event_bus.publish(Event("system.initialized", self.name))
            return True
        except Exception as e:
            self._logger.error(f"Failed to initialize adaptive system: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def start(self) -> bool:
        """Start the adaptive system."""
        if self.state != SystemState.READY:
            self._logger.error(f"Cannot start system: Invalid state: {self.state}")
            return False
        
        try:
            # Execute all pre-start hooks
            for hook in self._before_start_hooks:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            
            self.state = SystemState.RUNNING
            await self.event_bus.publish(Event("system.started", self.name))
            
            # Start auto-optimization if configured
            if self.auto_optimize_interval:
                self._optimization_task = asyncio.create_task(self._auto_optimize_loop())
            
            # Execute all post-start hooks
            for hook in self._after_start_hooks:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            
            return True
        except Exception as e:
            self._logger.error(f"Failed to start adaptive system: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def shutdown(self) -> bool:
        """Shutdown the adaptive system."""
        try:
            # Execute all pre-shutdown hooks
            for hook in self._before_shutdown_hooks:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            
            self.state = SystemState.SHUTTING_DOWN
            await self.event_bus.publish(Event("system.shutting_down", self.name))
            
            # Cancel auto-optimization if running
            if self._optimization_task:
                self._optimization_task.cancel()
                try:
                    await self._optimization_task
                except asyncio.CancelledError:
                    pass
            
            # Shutdown all subsystems
            for name, subsystem in self.subsystems.items():
                try:
                    self._logger.info(f"Shutting down subsystem: {name}")
                    await subsystem.shutdown()
                except Exception as e:
                    self._logger.error(f"Error shutting down subsystem {name}: {e}")
            
            # Save current state
            await self._save_state()
            
            # Execute all post-shutdown hooks
            for hook in self._after_shutdown_hooks:
                if asyncio.iscoroutinefunction(hook):
                    await hook(self)
                else:
                    hook(self)
            
            return True
        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")
            return False
    
    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics from all subsystems."""
        system_metrics = PerformanceMetrics(timestamp=datetime.now())
        
        # Collect

