"""
Coordination system responsible for orchestrating all components of the trading platform.

This module provides:
1. Component orchestration with lifecycle management
2. State management and synchronization
3. Resource allocation and optimization
4. Event coordination and prioritization
5. Synchronization mechanisms for system-wide consistency
"""

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast
from uuid import UUID, uuid4

# For type annotations
T = TypeVar('T')
ComponentType = TypeVar('ComponentType', bound='Component')
ResourceType = TypeVar('ResourceType')
EventType = TypeVar('EventType', bound='Event')

# Setup logging
logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """States a component can be in during its lifecycle."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


class ComponentPriority(Enum):
    """Priority levels for component execution and resource allocation."""
    CRITICAL = 0  # Highest priority, must never be starved
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4  # Lowest priority, can be preempted


class EventPriority(Enum):
    """Priority levels for event processing."""
    EMERGENCY = 0  # Immediate processing required (e.g., stop-loss triggered)
    HIGH = 1       # Urgent market events
    NORMAL = 2     # Regular operation events
    LOW = 3        # Background processing events
    AUDIT = 4      # Logging and audit events


class Event:
    """Base class for all system events."""
    
    def __init__(
        self, 
        event_type: str,
        source: Optional[Union[str, 'Component']] = None,
        priority: EventPriority = EventPriority.NORMAL,
        data: Optional[Dict[str, Any]] = None
    ):
        self.id = uuid4()
        self.event_type = event_type
        self.source = source
        self.priority = priority
        self.data = data or {}
        self.timestamp = time.time()
        self.processed = False
        self.processing_time: Optional[float] = None

    def __repr__(self) -> str:
        return f"Event(type={self.event_type}, source={self.source}, priority={self.priority})"


class Component:
    """Base class for all system components with lifecycle management."""
    
    def __init__(
        self,
        name: str,
        priority: ComponentPriority = ComponentPriority.NORMAL,
        dependencies: Optional[List[Type['Component']]] = None
    ):
        self.id = uuid4()
        self.name = name
        self.priority = priority
        self.dependencies = dependencies or []
        self.state = ComponentState.UNINITIALIZED
        self.error: Optional[Exception] = None
        self.last_active = time.time()
        self.metrics: Dict[str, Any] = {}
        
    async def initialize(self) -> bool:
        """Initialize the component. Return True on success, False on failure."""
        self.state = ComponentState.INITIALIZING
        try:
            success = await self._initialize()
            self.state = ComponentState.READY if success else ComponentState.ERROR
            return success
        except Exception as e:
            self.error = e
            self.state = ComponentState.ERROR
            logger.exception(f"Error initializing component {self.name}: {e}")
            return False
    
    async def _initialize(self) -> bool:
        """Implement component-specific initialization logic."""
        return True
    
    async def start(self) -> bool:
        """Start the component. Return True on success, False on failure."""
        if self.state != ComponentState.READY:
            logger.error(f"Cannot start component {self.name}: not in READY state")
            return False
        
        try:
            success = await self._start()
            self.state = ComponentState.RUNNING if success else ComponentState.ERROR
            return success
        except Exception as e:
            self.error = e
            self.state = ComponentState.ERROR
            logger.exception(f"Error starting component {self.name}: {e}")
            return False
    
    async def _start(self) -> bool:
        """Implement component-specific start logic."""
        return True
    
    async def pause(self) -> bool:
        """Pause the component. Return True on success, False on failure."""
        if self.state != ComponentState.RUNNING:
            logger.error(f"Cannot pause component {self.name}: not in RUNNING state")
            return False
        
        try:
            success = await self._pause()
            self.state = ComponentState.PAUSED if success else ComponentState.ERROR
            return success
        except Exception as e:
            self.error = e
            self.state = ComponentState.ERROR
            logger.exception(f"Error pausing component {self.name}: {e}")
            return False
    
    async def _pause(self) -> bool:
        """Implement component-specific pause logic."""
        return True
    
    async def resume(self) -> bool:
        """Resume the component. Return True on success, False on failure."""
        if self.state != ComponentState.PAUSED:
            logger.error(f"Cannot resume component {self.name}: not in PAUSED state")
            return False
        
        try:
            success = await self._resume()
            self.state = ComponentState.RUNNING if success else ComponentState.ERROR
            return success
        except Exception as e:
            self.error = e
            self.state = ComponentState.ERROR
            logger.exception(f"Error resuming component {self.name}: {e}")
            return False
    
    async def _resume(self) -> bool:
        """Implement component-specific resume logic."""
        return True
    
    async def stop(self) -> bool:
        """Stop the component. Return True on success, False on failure."""
        if self.state not in (ComponentState.RUNNING, ComponentState.PAUSED, ComponentState.ERROR):
            logger.error(f"Cannot stop component {self.name}: not in a running state")
            return False
        
        self.state = ComponentState.STOPPING
        try:
            success = await self._stop()
            self.state = ComponentState.STOPPED if success else ComponentState.ERROR
            return success
        except Exception as e:
            self.error = e
            self.state = ComponentState.ERROR
            logger.exception(f"Error stopping component {self.name}: {e}")
            return False
    
    async def _stop(self) -> bool:
        """Implement component-specific stop logic."""
        return True
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update component metrics."""
        self.metrics.update(metrics)
        self.last_active = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status including state and metrics."""
        return {
            "id": str(self.id),
            "name": self.name,
            "state": self.state.name,
            "priority": self.priority.name,
            "error": str(self.error) if self.error else None,
            "last_active": self.last_active,
            "metrics": self.metrics
        }
    
    def __repr__(self) -> str:
        return f"Component(name={self.name}, state={self.state.name})"


class Resource:
    """Base class for system resources that can be allocated to components."""
    
    def __init__(self, name: str, capacity: float):
        self.id = uuid4()
        self.name = name
        self.capacity = capacity
        self.available = capacity
        self.allocations: Dict[UUID, float] = {}
        self.last_updated = time.time()
    
    def allocate(self, component_id: UUID, amount: float) -> bool:
        """Allocate resource to a component. Return True on success."""
        if amount <= 0:
            return False
        
        if amount > self.available:
            return False
        
        self.allocations[component_id] = self.allocations.get(component_id, 0) + amount
        self.available -= amount
        self.last_updated = time.time()
        return True
    
    def release(self, component_id: UUID, amount: Optional[float] = None) -> float:
        """Release resource allocation. Return amount released."""
        if component_id not in self.allocations:
            return 0.0
        
        if amount is None:
            # Release all resources allocated to this component
            released = self.allocations[component_id]
            self.available += released
            del self.allocations[component_id]
        else:
            # Release specific amount
            current = self.allocations[component_id]
            released = min(current, amount)
            self.allocations[component_id] = current - released
            self.available += released
            
            if self.allocations[component_id] <= 0:
                del self.allocations[component_id]
        
        self.last_updated = time.time()
        return released
    
    def get_usage(self) -> Dict[str, Any]:
        """Get resource usage statistics."""
        return {
            "id": str(self.id),
            "name": self.name,
            "capacity": self.capacity,
            "available": self.available,
            "utilization": (self.capacity - self.available) / self.capacity,
            "allocations": {str(k): v for k, v in self.allocations.items()},
            "last_updated": self.last_updated
        }


class SystemState:
    """Maintains the global state of the system."""
    
    def __init__(self):
        self.components: Dict[UUID, Component] = {}
        self.resources: Dict[UUID, Resource] = {}
        self.events: List[Event] = []
        self.event_handlers: Dict[str, List[Callable[[Event], Any]]] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self.start_time = time.time()
        self.snapshots: List[Dict[str, Any]] = []
    
    def register_component(self, component: Component) -> None:
        """Register a component with the system state."""
        self.components[component.id] = component
    
    def unregister_component(self, component_id: UUID) -> bool:
        """Unregister a component from the system state."""
        if component_id in self.components:
            del self.components[component_id]
            return True
        return False
    
    def register_resource(self, resource: Resource) -> None:
        """Register a resource with the system state."""
        self.resources[resource.id] = resource
    
    def unregister_resource(self, resource_id: UUID) -> bool:
        """Unregister a resource from the system state."""
        if resource_id in self.resources:
            del self.resources[resource_id]
            return True
        return False
    
    def add_event(self, event: Event) -> None:
        """Add an event to the system state."""
        self.events.append(event)
        # Limit event history to prevent memory issues
        if len(self.events) > 10000:
            self.events = self.events[-5000:]
    
    def register_event_handler(self, event_type: str, handler: Callable[[Event], Any]) -> None:
        """Register a handler for an event type."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def unregister_event_handler(self, event_type: str, handler: Callable[[Event], Any]) -> bool:
        """Unregister a handler for an event type."""
        if event_type in self.event_handlers and handler in self.event_handlers[event_type]:
            self.event_handlers[event_type].remove(handler)
            return True
        return False
    
    def get_lock(self, name: str) -> asyncio.Lock:
        """Get or create a named lock."""
        if name not in self.locks:
            self.locks[name] = asyncio.Lock()
        return self.locks[name]
    
    def create_snapshot(self) -> Dict[str, Any]:
        """Create a snapshot of the current system state."""
        snapshot = {
            "timestamp": time.time(),
            "uptime": time.time() - self.start_time,
            "components": {str(k): v.get_status() for k, v in self.components.items()},
            "resources": {str(k): v.get_usage() for k, v in self.resources.items()},
            "event_count": len(self.events),
            "recent_events": [
                {
                    "id": str(e.id),
                    "type": e.event_type,
                    "source": str(e.source) if hasattr(e.source, "id") else e.source,
                    "priority": e.priority.name,
                    "timestamp": e.timestamp,
                    "processed": e.processed
                }
                for e in self.events[-10:]
            ]
        }
        self.snapshots.append(snapshot)
        # Limit snapshot history
        if len(self.snapshots) > 100:
            self.snapshots = self.snapshots[-50:]
        return snapshot


class Coordinator:
    """Central coordination system that orchestrates all components and manages system state."""
    
    def __init__(self):
        self.state = SystemState()
        self.event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.running = False
        self.event_task: Optional[asyncio.Task] = None
        
    async def initialize(self) -> bool:
        """Initialize the coordinator and all registered components."""
        logger.info("Initializing coordinator")
        
        # Create standard system resources
        cpu_resource = Resource("CPU", 100.0)  # 100% capacity
        memory_resource = Resource("Memory", 100.0)  # 100% capacity
        io_resource = Resource("IO", 100.0)  # 100% capacity
        network_resource = Resource("Network", 100.0)  # 100% capacity
        
        self.state.register_resource(cpu_resource)
        self.state.register_resource(memory_resource)
        self.state.register_resource(io_resource)
        self.state.register_resource(network_resource)
        
        return True
    
    async def start(self) -> bool:
        """Start the coordinator and all registered components."""
        if self.running:
            logger.warning("Coordinator already running")
            return False
        
        logger.info("Starting coordinator")
        
        # Start event processing
        self.running = True
        self.event_task = asyncio.create_task(self._process_events())
        
        # Get all components sorted by priority (highest priority first)
        components = sorted(
            self.state.components.values(),
            key=lambda c: c.priority.value
        )
        
        # Initialize components in dependency order
        for component in components:
            # Check if all dependencies are initialized
            deps_ready = all(
                dep_id in self.state.components and 
                self.state.components[dep_id].state in (ComponentState.READY, ComponentState.RUNNING)
                for dep_id in [c.id for c in component.dependencies]
            )
            
            if not deps_ready:
                logger.error(f"Cannot initialize

