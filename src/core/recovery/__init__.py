"""
Recovery system for handling failures and implementing resilient recovery strategies.

This module implements comprehensive failure detection, recovery strategies, state restoration,
recovery tracking, and cascading failure prevention mechanisms.
"""

import asyncio
import logging
import time
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, Protocol
import traceback
import uuid

# For integration with other core systems
from ..monitoring import SystemMonitor
from ..orchestration import Orchestrator
from ..config import ConfigManager
from ..adaptation import AdaptationManager


class FailureType(Enum):
    """Types of failures that can occur in the system."""
    COMPONENT_CRASH = auto()  # Complete component failure
    PERFORMANCE_DEGRADATION = auto()  # Component still running but performance degraded
    RESOURCE_EXHAUSTION = auto()  # Out of memory, CPU, etc.
    EXTERNAL_DEPENDENCY_FAILURE = auto()  # External API/service failure
    DATA_CORRUPTION = auto()  # Corrupted state or data
    NETWORK_FAILURE = auto()  # Network connectivity issues
    TIMEOUT = auto()  # Operation took too long
    CONFIGURATION_ERROR = auto()  # Misconfiguration
    UNKNOWN = auto()  # Unclassified failure


class FailureSeverity(Enum):
    """Severity levels for failures."""
    CRITICAL = auto()  # Requires immediate attention and recovery
    HIGH = auto()  # Severe impact but system can continue partially
    MEDIUM = auto()  # Noticeable impact but most functionality remains
    LOW = auto()  # Minor impact, can be addressed later
    INFO = auto()  # Informational only, no impact


class RecoveryStrategy(Enum):
    """Available strategies for recovering from failures."""
    RESTART_COMPONENT = auto()  # Restart only the failed component
    RELOAD_CONFIGURATION = auto()  # Reload configuration and restart
    ROLLBACK_STATE = auto()  # Rollback to a previous known good state
    FAILOVER = auto()  # Switch to a backup component/system
    GRACEFUL_DEGRADATION = auto()  # Continue with reduced functionality
    RETRY_OPERATION = auto()  # Retry the failed operation
    MANUAL_INTERVENTION = auto()  # Require manual intervention
    CIRCUIT_BREAKER = auto()  # Stop operations to prevent cascading failures


class FailureEvent:
    """Represents a failure event in the system."""
    
    def __init__(
        self,
        component_id: str,
        failure_type: FailureType,
        severity: FailureSeverity,
        timestamp: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None,
        stack_trace: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.component_id = component_id
        self.failure_type = failure_type
        self.severity = severity
        self.timestamp = timestamp or datetime.now()
        self.details = details or {}
        self.exception = exception
        self.stack_trace = stack_trace or (
            "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            )) if exception else None
        )
        self.resolved = False
        self.resolution_timestamp: Optional[datetime] = None
        self.recovery_strategy: Optional[RecoveryStrategy] = None
        self.recovery_attempts: int = 0
        self.cascaded_from: Optional[str] = None  # ID of parent failure that caused this
        
    def resolve(self, strategy: RecoveryStrategy) -> None:
        """Mark the failure as resolved with the given strategy."""
        self.resolved = True
        self.resolution_timestamp = datetime.now()
        self.recovery_strategy = strategy
        
    def add_attempt(self) -> None:
        """Increment the recovery attempt counter."""
        self.recovery_attempts += 1
        
    def set_cascaded_from(self, parent_failure_id: str) -> None:
        """Set this failure as cascaded from another failure."""
        self.cascaded_from = parent_failure_id
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "component_id": self.component_id,
            "failure_type": self.failure_type.name,
            "severity": self.severity.name,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
            "exception_type": type(self.exception).__name__ if self.exception else None,
            "stack_trace": self.stack_trace,
            "resolved": self.resolved,
            "resolution_timestamp": self.resolution_timestamp.isoformat() if self.resolution_timestamp else None,
            "recovery_strategy": self.recovery_strategy.name if self.recovery_strategy else None,
            "recovery_attempts": self.recovery_attempts,
            "cascaded_from": self.cascaded_from
        }


class ComponentStateSnapshot:
    """Represents a snapshot of a component's state for restoration."""
    
    def __init__(
        self,
        component_id: str,
        state_data: Dict[str, Any],
        timestamp: Optional[datetime] = None,
        version: str = "1.0",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid.uuid4())
        self.component_id = component_id
        self.state_data = state_data
        self.timestamp = timestamp or datetime.now()
        self.version = version
        self.metadata = metadata or {}
        self.verified = False
        
    def verify(self) -> bool:
        """Verify the integrity of the snapshot."""
        # Implementation would depend on your verification mechanism
        # For example, checksum validation, schema validation, etc.
        self.verified = True
        return self.verified
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "component_id": self.component_id,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
            "verified": self.verified,
            # State data might need special serialization depending on content
            "state_data": self.state_data
        }


class RecoveryAction:
    """Represents a recovery action to be taken for a failure."""
    
    def __init__(
        self,
        failure_event: FailureEvent,
        strategy: RecoveryStrategy,
        action_handler: Callable[["RecoveryAction"], Any],
        priority: int = 0,
        timeout_seconds: float = 60.0,
        max_attempts: int = 3,
        dependencies: Optional[List[str]] = None  # IDs of other actions this depends on
    ):
        self.id = str(uuid.uuid4())
        self.failure_event = failure_event
        self.strategy = strategy
        self.action_handler = action_handler
        self.priority = priority
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.dependencies = dependencies or []
        self.attempts = 0
        self.successful = False
        self.last_attempt_timestamp: Optional[datetime] = None
        self.completion_timestamp: Optional[datetime] = None
        self.result: Any = None
        self.error: Optional[Exception] = None
        
    async def execute(self) -> bool:
        """Execute the recovery action."""
        if self.attempts >= self.max_attempts:
            return False
            
        self.attempts += 1
        self.last_attempt_timestamp = datetime.now()
        self.failure_event.add_attempt()
        
        try:
            self.result = await asyncio.wait_for(
                asyncio.coroutine(self.action_handler)(self),
                timeout=self.timeout_seconds
            )
            self.successful = True
            self.completion_timestamp = datetime.now()
            return True
        except Exception as e:
            self.error = e
            return False


class StateManager:
    """Manages component state snapshots for recovery purposes."""
    
    def __init__(self):
        self.snapshots: Dict[str, List[ComponentStateSnapshot]] = {}
        self.snapshot_frequency: Dict[str, float] = {}  # component_id -> seconds
        self.max_snapshots_per_component: int = 10
        self.last_snapshot_time: Dict[str, datetime] = {}
        
    async def create_snapshot(self, component_id: str, state_data: Dict[str, Any],
                        metadata: Optional[Dict[str, Any]] = None) -> ComponentStateSnapshot:
        """Create and store a new state snapshot for a component."""
        snapshot = ComponentStateSnapshot(
            component_id=component_id,
            state_data=state_data,
            metadata=metadata
        )
        
        if component_id not in self.snapshots:
            self.snapshots[component_id] = []
            
        self.snapshots[component_id].append(snapshot)
        self.last_snapshot_time[component_id] = datetime.now()
        
        # Limit the number of snapshots per component
        if len(self.snapshots[component_id]) > self.max_snapshots_per_component:
            self.snapshots[component_id].pop(0)  # Remove oldest
            
        return snapshot
        
    def get_latest_snapshot(self, component_id: str) -> Optional[ComponentStateSnapshot]:
        """Get the most recent snapshot for a component."""
        if component_id not in self.snapshots or not self.snapshots[component_id]:
            return None
            
        return self.snapshots[component_id][-1]
        
    def get_snapshot_before(self, component_id: str, timestamp: datetime) -> Optional[ComponentStateSnapshot]:
        """Get the most recent snapshot before the given timestamp."""
        if component_id not in self.snapshots:
            return None
            
        valid_snapshots = [s for s in self.snapshots[component_id] if s.timestamp < timestamp]
        if not valid_snapshots:
            return None
            
        return max(valid_snapshots, key=lambda s: s.timestamp)
        
    def set_snapshot_frequency(self, component_id: str, seconds: float) -> None:
        """Set the frequency for automatic snapshots of a component."""
        self.snapshot_frequency[component_id] = seconds
        
    def should_create_snapshot(self, component_id: str) -> bool:
        """Check if it's time to create a new snapshot based on frequency."""
        if component_id not in self.snapshot_frequency:
            return False
            
        if component_id not in self.last_snapshot_time:
            return True
            
        elapsed = (datetime.now() - self.last_snapshot_time[component_id]).total_seconds()
        return elapsed >= self.snapshot_frequency[component_id]
        
    def prune_old_snapshots(self, max_age_days: int = 7) -> int:
        """Remove snapshots older than the specified age."""
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        count = 0
        
        for component_id in self.snapshots:
            original_len = len(self.snapshots[component_id])
            self.snapshots[component_id] = [
                s for s in self.snapshots[component_id] 
                if s.timestamp.timestamp() > cutoff
            ]
            count += original_len - len(self.snapshots[component_id])
            
        return count


class FailureDetector:
    """Detects component failures using various mechanisms."""
    
    def __init__(self, monitor: Optional[SystemMonitor] = None):
        self.monitor = monitor
        self.component_health_checks: Dict[str, Callable[[], bool]] = {}
        self.performance_thresholds: Dict[str, Dict[str, float]] = {}
        self.resource_thresholds: Dict[str, Dict[str, float]] = {}
        self.failure_patterns: List[Dict[str, Any]] = []
        self.failure_listeners: List[Callable[[FailureEvent], None]] = []
        
    def register_health_check(self, component_id: str, check_func: Callable[[], bool]) -> None:
        """Register a health check function for a component."""
        self.component_health_checks[component_id] = check_func
        
    def set_performance_threshold(self, component_id: str, metric: str, threshold: float) -> None:
        """Set a performance threshold for a component."""
        if component_id not in self.performance_thresholds:
            self.performance_thresholds[component_id] = {}
            
        self.performance_thresholds[component_id][metric] = threshold
        
    def set_resource_threshold(self, component_id: str, resource: str, threshold: float) -> None:
        """Set a resource usage threshold for a component."""
        if component_id not in self.resource_thresholds:
            self.resource_thresholds[component_id] = {}
            
        self.resource_thresholds[component_id][resource] = threshold
        
    def add_failure_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add a pattern that indicates a failure condition."""
        self.failure_patterns.append(pattern)
        
    def add_failure_listener(self, listener: Callable[[FailureEvent], None]) -> None:
        """Add a listener to be notified when failures are detected."""
        self.failure_listeners.append(listener)
        
    async def check_component_health(self, component_id: str) -> Optional[FailureEvent]:
        """Check the health of a specific component."""
        if component_id not in self.component_health_checks:
            return None
            
        try:
            if not self.component_health_checks[component_id]():
                failure = FailureEvent(
                    component_id=component_id,
                    failure_type=FailureType.COMPONENT_CRASH,
                    severity=FailureSeverity.HIGH,
                    details={"source": "health_check"}
                )
                self._notify_failure(failure)
                return failure
        except Exception as e:
            failure = FailureEvent(
                component_id=component_id,
                failure_type=FailureType.UNKNOWN,
                severity=FailureSeverity.HIGH,
                details={"source": "health_check_exception"},
                exception=e
            )
            self._notify_failure(failure)
            return failure
            
        return None
        
    async def check_performance_metrics(self, component_id: str) -> List[FailureEvent]:
        """Check performance metrics against thresholds."""
        if not self.monitor or component_id not in self.performance_thresholds:
            return []
            
        failures = []
        component_metrics = await self.monitor.get_component_metrics(component_id)
        
        for metric, threshold in self.performance_thresholds[component_id].items():
            if metric in component_metrics and component_metrics[metric] > threshold:
                failure = FailureEvent(
                    component_id=component_id,
                    failure_type=FailureType.PERFORMANCE_DEGRADATION,
                    severity=FailureSeverity.MEDIUM,
                    details={
                        "metric": metric,
                        "value": component_metrics[metric],
                        "threshold": threshold
                    }

