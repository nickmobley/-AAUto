"""
Orchestration Module for System Lifecycle Management.

This module provides the core orchestration capabilities for managing component lifecycle,
dependency resolution, state transitions, health monitoring, and system recovery.
"""

import asyncio
import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Type, TypeVar, Generic, Callable, Any, Awaitable
import uuid
import importlib
import inspect
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import signal
import traceback

from ..config import ConfigurationManager
from ..coordination import Coordinator
from ..monitoring import HealthMonitor, SystemMetrics
from ..validation import ValidationResult, Validator

# Type definitions
T = TypeVar('T')
ComponentT = TypeVar('ComponentT', bound='Component')

class SystemState(Enum):
    """System state enumeration."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSING = auto()
    PAUSED = auto()
    RESUMING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()
    RECOVERING = auto()


class DependencyType(Enum):
    """Types of dependencies between components."""
    REQUIRED = auto()  # Component won't start without this dependency
    OPTIONAL = auto()  # Component can start without this dependency
    RUNTIME = auto()   # Component needs this dependency at runtime, not startup


@dataclass
class ComponentState:
    """Component state tracking."""
    state: SystemState = SystemState.UNINITIALIZED
    health_score: float = 1.0
    last_health_check: float = 0.0
    start_time: Optional[float] = None
    stop_time: Optional[float] = None
    error_count: int = 0
    last_error: Optional[Exception] = None
    recovery_attempts: int = 0
    dependencies_satisfied: bool = False


@dataclass
class ComponentDependency:
    """Defines a dependency between components."""
    component_type: Type
    dependency_type: DependencyType
    timeout: float = 30.0  # Timeout in seconds


class Component:
    """Base component interface that all system components must implement."""
    
    def __init__(self, component_id: Optional[str] = None):
        self.component_id = component_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.state = ComponentState()
        self._dependencies: List[ComponentDependency] = []
        self._dependents: Set[str] = set()
        
    def requires(self, component_type: Type, 
                dependency_type: DependencyType = DependencyType.REQUIRED, 
                timeout: float = 30.0) -> None:
        """Register a dependency on another component."""
        self._dependencies.append(ComponentDependency(
            component_type=component_type,
            dependency_type=dependency_type,
            timeout=timeout
        ))
    
    async def initialize(self) -> None:
        """Initialize the component. Called before starting."""
        pass
    
    async def start(self) -> None:
        """Start the component. Must be implemented by subclasses."""
        raise NotImplementedError("Component must implement start()")
    
    async def pause(self) -> None:
        """Pause the component. Optional."""
        pass
    
    async def resume(self) -> None:
        """Resume the component. Optional."""
        pass
    
    async def stop(self) -> None:
        """Stop the component. Must be implemented by subclasses."""
        raise NotImplementedError("Component must implement stop()")
    
    async def health_check(self) -> float:
        """Return health score from 0.0 (failed) to 1.0 (perfect)."""
        return 1.0
    
    async def validate(self) -> ValidationResult:
        """Validate component state and configuration."""
        return ValidationResult(valid=True)
    
    async def recover(self, error: Exception) -> bool:
        """Attempt to recover from error. Return True if successful."""
        return False


@dataclass
class DependencyGraph:
    """Manages component dependencies and resolution order."""
    
    components: Dict[str, Component] = field(default_factory=dict)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    dependents: Dict[str, Set[str]] = field(default_factory=dict)
    
    def add_component(self, component: Component) -> None:
        """Add a component to the dependency graph."""
        self.components[component.component_id] = component
        if component.component_id not in self.dependencies:
            self.dependencies[component.component_id] = set()
        if component.component_id not in self.dependents:
            self.dependents[component.component_id] = set()
    
    def add_dependency(self, dependent_id: str, dependency_id: str) -> None:
        """Add a dependency relationship between components."""
        if dependent_id not in self.dependencies:
            self.dependencies[dependent_id] = set()
        self.dependencies[dependent_id].add(dependency_id)
        
        if dependency_id not in self.dependents:
            self.dependents[dependency_id] = set()
        self.dependents[dependency_id].add(dependent_id)
    
    def get_start_order(self) -> List[str]:
        """Return component IDs in dependency-resolved start order."""
        visited: Set[str] = set()
        start_order: List[str] = []
        
        def visit(component_id: str) -> None:
            if component_id in visited:
                return
            visited.add(component_id)
            
            for dependency_id in self.dependencies.get(component_id, set()):
                visit(dependency_id)
            
            start_order.append(component_id)
        
        for component_id in self.components:
            visit(component_id)
            
        return start_order
    
    def get_stop_order(self) -> List[str]:
        """Return component IDs in reverse dependency order for shutdown."""
        return list(reversed(self.get_start_order()))


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


class ComponentRegistry:
    """Registry of all system components."""
    
    def __init__(self):
        self._components: Dict[str, Component] = {}
        self._component_types: Dict[Type, List[Component]] = {}
        self._dependency_graph = DependencyGraph()
        
    def register(self, component: Component) -> None:
        """Register a component with the system."""
        self._components[component.component_id] = component
        
        component_type = type(component)
        if component_type not in self._component_types:
            self._component_types[component_type] = []
        self._component_types[component_type].append(component)
        
        self._dependency_graph.add_component(component)
    
    def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return self._components.get(component_id)
    
    def get_components_by_type(self, component_type: Type) -> List[Component]:
        """Get all components of a specific type."""
        return self._component_types.get(component_type, [])
    
    def build_dependency_graph(self) -> None:
        """Build the dependency graph from component dependencies."""
        for component in self._components.values():
            for dependency in component._dependencies:
                # Find all instances of the required component type
                dependency_components = self.get_components_by_type(dependency.component_type)
                
                if not dependency_components and dependency.dependency_type == DependencyType.REQUIRED:
                    raise ValueError(
                        f"Component {component.component_id} requires {dependency.component_type.__name__}, "
                        f"but no instances are registered"
                    )
                
                # Add dependencies to the graph
                for dep_component in dependency_components:
                    self._dependency_graph.add_dependency(
                        component.component_id, dep_component.component_id
                    )
    
    def get_start_order(self) -> List[str]:
        """Get dependency-resolved component start order."""
        try:
            return self._dependency_graph.get_start_order()
        except RecursionError:
            raise CircularDependencyError("Circular dependency detected in component graph")
    
    def get_stop_order(self) -> List[str]:
        """Get dependency-resolved component stop order."""
        return self._dependency_graph.get_stop_order()


class StateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
    pass


class Orchestrator:
    """
    System orchestrator that manages component lifecycle, state transitions,
    health monitoring, and recovery.
    """
    
    def __init__(self, 
                config_manager: Optional[ConfigurationManager] = None,
                coordinator: Optional[Coordinator] = None,
                health_monitor: Optional[HealthMonitor] = None):
        self.registry = ComponentRegistry()
        self.config_manager = config_manager
        self.coordinator = coordinator
        self.health_monitor = health_monitor
        
        self.state = SystemState.UNINITIALIZED
        self._component_states: Dict[str, ComponentState] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._shutdown_event = asyncio.Event()
        self._logger = logging.getLogger(__name__)
    
    def register_component(self, component: Component) -> None:
        """Register a component with the orchestrator."""
        self.registry.register(component)
        self._component_states[component.component_id] = component.state
    
    async def initialize(self) -> None:
        """Initialize the orchestration system and all components."""
        if self.state != SystemState.UNINITIALIZED:
            raise StateTransitionError(f"Cannot initialize from state {self.state}")
        
        self.state = SystemState.INITIALIZING
        self._logger.info("Initializing orchestration system")
        
        # Register signal handlers
        self._setup_signal_handlers()
        
        # Build dependency graph
        try:
            self.registry.build_dependency_graph()
        except Exception as e:
            self._logger.error(f"Failed to build dependency graph: {e}")
            self.state = SystemState.FAILED
            raise
        
        # Initialize components in dependency order
        for component_id in self.registry.get_start_order():
            component = self.registry.get_component(component_id)
            if not component:
                continue
            
            try:
                self._logger.debug(f"Initializing component {component_id}")
                await component.initialize()
                component.state.state = SystemState.INITIALIZED
            except Exception as e:
                self._logger.error(f"Failed to initialize component {component_id}: {e}")
                component.state.state = SystemState.FAILED
                component.state.last_error = e
                self.state = SystemState.FAILED
                raise
        
        self.state = SystemState.INITIALIZED
        self._logger.info("Orchestration system initialized")
    
    async def start(self) -> None:
        """Start all components in dependency order."""
        if self.state != SystemState.INITIALIZED:
            raise StateTransitionError(f"Cannot start from state {self.state}")
        
        self.state = SystemState.STARTING
        self._logger.info("Starting components in dependency order")
        
        start_order = self.registry.get_start_order()
        for component_id in start_order:
            component = self.registry.get_component(component_id)
            if not component:
                continue
            
            try:
                self._logger.debug(f"Starting component {component_id}")
                component.state.state = SystemState.STARTING
                await component.start()
                component.state.state = SystemState.RUNNING
                component.state.start_time = time.time()
                
                # Start health check task for this component
                self._tasks[f"health_{component_id}"] = asyncio.create_task(
                    self._monitor_component_health(component)
                )
            except Exception as e:
                self._logger.error(f"Failed to start component {component_id}: {e}")
                component.state.state = SystemState.FAILED
                component.state.last_error = e
                component.state.error_count += 1
                
                # Try to recover the component
                await self._attempt_recovery(component)
                
                # If it's a required component, fail the system
                # TODO: Implement more nuanced dependency handling
                self.state = SystemState.FAILED
                raise
        
        self.state = SystemState.RUNNING
        self._logger.info("All components started successfully")
    
    async def stop(self) -> None:
        """Stop all components in reverse dependency order."""
        if self.state in {SystemState.UNINITIALIZED, SystemState.STOPPED}:
            return
        
        self.state = SystemState.STOPPING
        self._logger.info("Stopping components in reverse dependency order")
        
        # Signal all tasks to stop
        self._shutdown_event.set()
        
        # Cancel health monitoring tasks
        for task_name, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        
        # Stop components in reverse order
        stop_order = self.registry.get_stop_order()
        for component_id in stop_order:
            component = self.registry.get_component(component_id)
            if not component:
                continue
            
            try:
                if component.state.state in {SystemState.RUNNING, SystemState.PAUSED}:
                    self._logger.debug(f"Stopping component {component_id}")
                    component.state.state = SystemState.STOPPING
                    await component.stop()
                    component.state.state = SystemState.STOPPED
                    component.state.stop_time = time.time()
            except Exception as e:
                self._logger.error(f"Error stopping component {component_id}: {e}")
                component.state.last_error = e
                component.state.error_count += 1
        
        self.state = SystemState.STOPPED
        self._logger.info("All components stopped")
    
    async def pause(self) -> None:
        """Pause all components."""
        if self.state != SystemState.RUNNING:
            raise StateTransitionError(f"Cannot pause from state {self.state}")
        
        self.state = SystemState.PAUSING
        self._logger.info("Pausing all components")
        
        # Pause in reverse order to respect dependencies
        for component_id in self.registry.get_stop_order():
            component = self.registry.get_component(component_id)
            if not component or component.state.state != SystemState.RUNNING:
                continue
            
            try:
                self._logger.debug(f"Pausing component {component_id}")
                component.state.state = SystemState.PAUSING
                await component.pause()
                component.state.state = SystemState.PAUSED
            except Exception as e:
                self._logger.error(f"Error pausing component {component_id}: {e}")
                component.state.last_error = e
                component.state.error_count += 1
        
        self.state = SystemState.

