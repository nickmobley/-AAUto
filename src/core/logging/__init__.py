import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import inspect
import threading
import contextlib
from datetime import datetime

# Define log levels
class LogLevel(Enum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class LogEvent:
    """A structured log event with rich metadata for analysis."""
    timestamp: float = field(default_factory=time.time)
    level: LogLevel = LogLevel.INFO
    message: str = ""
    component: str = ""
    correlation_id: str = ""
    execution_time: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    exception: Optional[Exception] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert log event to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = datetime.fromtimestamp(data['timestamp']).isoformat()
        data['level'] = data['level'].name
        data['exception'] = str(data['exception']) if data['exception'] else None
        data['tags'] = list(data['tags'])
        return data

    def to_json(self) -> str:
        """Convert log event to JSON string."""
        return json.dumps(self.to_dict())


class PerformanceTracker:
    """Tracks component performance for dynamic log level adjustment."""
    
    def __init__(self):
        self._performance_data: Dict[str, List[Tuple[float, float]]] = {}
        self._lock = threading.Lock()
        
    def record_execution(self, component: str, execution_time: float) -> None:
        """Record execution time for a component."""
        with self._lock:
            if component not in self._performance_data:
                self._performance_data[component] = []
            
            # Keep last 100 measurements per component
            if len(self._performance_data[component]) >= 100:
                self._performance_data[component].pop(0)
            
            self._performance_data[component].append((time.time(), execution_time))
    
    def get_average_execution_time(self, component: str) -> Optional[float]:
        """Get average execution time for a component over the last N records."""
        with self._lock:
            if component not in self._performance_data or not self._performance_data[component]:
                return None
            
            total_time = sum(exec_time for _, exec_time in self._performance_data[component])
            return total_time / len(self._performance_data[component])

    def is_performance_degrading(self, component: str) -> Optional[bool]:
        """Check if component performance is degrading over time."""
        with self._lock:
            if component not in self._performance_data or len(self._performance_data[component]) < 10:
                return None
            
            # Compare recent vs earlier performance
            recent = self._performance_data[component][-5:]
            earlier = self._performance_data[component][-10:-5]
            
            recent_avg = sum(exec_time for _, exec_time in recent) / len(recent)
            earlier_avg = sum(exec_time for _, exec_time in earlier) / len(earlier)
            
            # Consider degrading if recent average is 20% worse than earlier
            return recent_avg > earlier_avg * 1.2


class EventDetector:
    """Detects important system events based on log patterns."""
    
    def __init__(self, importance_threshold: float = 0.7):
        self._recent_events: List[LogEvent] = []
        self._known_patterns: Dict[str, float] = {
            "timeout": 0.8,
            "exception": 0.9,
            "critical": 1.0,
            "failed": 0.7,
            "memory": 0.8,
            "performance": 0.7,
            "latency": 0.75,
            "error rate": 0.85
        }
        self._importance_threshold = importance_threshold
        self._lock = threading.Lock()
        
    def process_event(self, event: LogEvent) -> Optional[Tuple[str, float]]:
        """Process a log event and detect important patterns."""
        with self._lock:
            # Keep last 1000 events for pattern recognition
            if len(self._recent_events) >= 1000:
                self._recent_events.pop(0)
            self._recent_events.append(event)
            
            # Check for known patterns in the message
            for pattern, importance in self._known_patterns.items():
                if pattern.lower() in event.message.lower() and importance >= self._importance_threshold:
                    return pattern, importance
            
            # Check for sudden increase in errors
            if event.level in (LogLevel.ERROR, LogLevel.CRITICAL):
                error_count = sum(1 for e in self._recent_events[-20:] 
                                  if e.level in (LogLevel.ERROR, LogLevel.CRITICAL))
                if error_count >= 5:  # 25% of recent events are errors
                    return "error_surge", 0.9
            
            return None
            
    def update_pattern_importance(self, pattern: str, importance: float) -> None:
        """Update the importance of a known pattern."""
        with self._lock:
            self._known_patterns[pattern] = importance


class CorrelationTracker:
    """Tracks correlations between components and events."""
    
    def __init__(self, correlation_window: int = 100):
        self._correlation_window = correlation_window
        self._events_by_correlation_id: Dict[str, List[LogEvent]] = {}
        self._component_interactions: Dict[Tuple[str, str], int] = {}
        self._lock = threading.Lock()
        
    def add_event(self, event: LogEvent) -> None:
        """Add an event to correlation tracking."""
        if not event.correlation_id:
            return
            
        with self._lock:
            if event.correlation_id not in self._events_by_correlation_id:
                self._events_by_correlation_id[event.correlation_id] = []
            
            # Add the new event
            self._events_by_correlation_id[event.correlation_id].append(event)
            
            # Update component interactions
            current_events = self._events_by_correlation_id[event.correlation_id]
            if len(current_events) >= 2:
                previous_event = current_events[-2]
                if previous_event.component != event.component:
                    interaction = (previous_event.component, event.component)
                    self._component_interactions[interaction] = self._component_interactions.get(interaction, 0) + 1
            
            # Clean up old correlation ids to prevent memory bloat
            self._cleanup_old_correlations()
    
    def _cleanup_old_correlations(self) -> None:
        """Remove old correlation data to prevent memory issues."""
        now = time.time()
        expired_ids = []
        
        for corr_id, events in self._events_by_correlation_id.items():
            oldest_timestamp = min(event.timestamp for event in events)
            if now - oldest_timestamp > 3600:  # Keep data for up to 1 hour
                expired_ids.append(corr_id)
        
        for corr_id in expired_ids:
            del self._events_by_correlation_id[corr_id]
    
    def get_related_events(self, correlation_id: str) -> List[LogEvent]:
        """Get all events related to a specific correlation ID."""
        with self._lock:
            return self._events_by_correlation_id.get(correlation_id, []).copy()
    
    def get_frequent_interactions(self, min_count: int = 5) -> List[Tuple[Tuple[str, str], int]]:
        """Get component interactions that occur frequently."""
        with self._lock:
            return [(interaction, count) for interaction, count in 
                   self._component_interactions.items() if count >= min_count]


class AdaptiveLogger:
    """A self-adjusting logger with dynamic log levels based on component performance."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AdaptiveLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if AdaptiveLogger._initialized:
            return
            
        # Initialize standard Python logger
        self._logger = logging.getLogger("adaptive_system")
        self._handler = logging.StreamHandler()
        self._formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self._handler.setFormatter(self._formatter)
        self._logger.addHandler(self._handler)
        self._logger.setLevel(logging.DEBUG)
        
        # Register TRACE log level with the logging module
        logging.addLevelName(LogLevel.TRACE.value, LogLevel.TRACE.name)
        
        # Set up dynamic level adjustment
        self._component_log_levels: Dict[str, LogLevel] = {}
        self._default_level = LogLevel.INFO
        
        # Create supporting systems
        self._performance_tracker = PerformanceTracker()
        self._event_detector = EventDetector()
        self._correlation_tracker = CorrelationTracker()
        
        # For analytics integration
        self._log_sink: List[Callable[[LogEvent], None]] = []
        self._analytics_queue = asyncio.Queue()
        self._processing_task = None
        
        # Contextual information
        self._context_var = threading.local()
        self._context_var.values = {}
        self._context_var.correlation_id = ""
        
        # Throttling configuration to prevent log storms
        self._throttle_counters: Dict[str, Tuple[int, float]] = {}  # {message_hash: (count, first_seen)}
        self._throttle_window = 60.0  # seconds
        self._throttle_threshold = 10  # occurrences
        
        AdaptiveLogger._initialized = True

    def start_processing(self) -> None:
        """Start asynchronous processing of log events."""
        if self._processing_task is None or self._processing_task.done():
            # Initialize the event loop for the current thread if not already present
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            self._processing_task = asyncio.create_task(self._process_analytics_queue())

    async def _process_analytics_queue(self) -> None:
        """Process log events for analytics."""
        while True:
            try:
                event = await self._analytics_queue.get()
                for sink in self._log_sink:
                    try:
                        # Call synchronous sinks directly
                        if not inspect.iscoroutinefunction(sink):
                            sink(event)
                        else:
                            # Schedule asynchronous sinks
                            asyncio.create_task(sink(event))
                    except Exception as e:
                        # Don't use self.log here to avoid recursion
                        self._logger.error(f"Error in log sink: {e}")
                self._analytics_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error processing analytics queue: {e}")
                await asyncio.sleep(1)  # Prevent tight loop in case of repeated errors

    def stop_processing(self) -> None:
        """Stop asynchronous processing of log events."""
        if self._processing_task is not None and not self._processing_task.done():
            self._processing_task.cancel()

    def add_sink(self, sink: Callable[[LogEvent], None]) -> None:
        """Add a sink for log events for external processing."""
        self._log_sink.append(sink)

    def remove_sink(self, sink: Callable[[LogEvent], None]) -> None:
        """Remove a sink for log events."""
        if sink in self._log_sink:
            self._log_sink.remove(sink)

    @contextlib.contextmanager
    def correlation_context(self, correlation_id: Optional[str] = None) -> None:
        """Set a correlation ID for the current context."""
        old_correlation_id = getattr(self._context_var, 'correlation_id', '')
        self._context_var.correlation_id = correlation_id or str(uuid.uuid4())
        try:
            yield self._context_var.correlation_id
        finally:
            self._context_var.correlation_id = old_correlation_id

    @contextlib.contextmanager
    def context(self, **kwargs) -> None:
        """Add context information to all logs in this context."""
        old_values = getattr(self._context_var, 'values', {}).copy()
        if not hasattr(self._context_var, 'values'):
            self._context_var.values = {}
        self._context_var.values.update(kwargs)
        try:
            yield
        finally:
            self._context_var.values = old_values

    def _should_log(self, component: str, level: LogLevel) -> bool:
        """Determine if a message should be logged based on dynamic levels."""
        component_level = self._component_log_levels.get(component, self._default_level)
        return level.value >= component_level.value

    def _create_log_event(self, level: LogLevel, message: str, component: str, 
                          execution_time: Optional[float] = None, tags: Optional[Set[str]] = None,
                          exception: Optional[Exception] = None, **kwargs) -> LogEvent:
        """Create a structured log event."""
        context = getattr(self._context_var, 'values', {}).copy()
        context.update(kwargs)
        
        return LogEvent(
            timestamp=time.time(),
            level=level,
            message=message,
            component=component,
            correlation_id=getattr(self._context_var, 'correlation_id', ''),
            execution_time=execution_time,
            context=context,
            tags=tags or set(),
            exception=exception
        )

    def _update_dynamic_levels(self) -> None:
        """Update dynamic log levels based on component performance."""
        for component in list(self._performance_tracker._performance_data.keys()):
            is_degrading = self._performance_tracker.is_performance_degrading(component)
            avg_time = self._performance_tracker.get_average_execution_time(component)
            
            if is_degrading is True and avg_time is not None:
                # Increase verbosity for components with degrading performance
                self._component_log_levels[component] = LogLevel.DEBUG
            elif is_degrading is False and component in self._component_log_levels:
                # Reset to default for components with stable performance
                self._component_log_levels[component] = self._default_level

    def _apply_throttling(self, message: str, component: str) -> bool:
        """Apply throttling to prevent log storms."""
        now = time.time()
        # Create a simple hash of the message and component

