"""
Adaptation System Module

This module provides a comprehensive framework for managing adaptations in the system.
It handles adaptation strategies, safe adaptation mechanisms, verification,
rollback capabilities, and adaptation tracking.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast

# Import core modules for integration
from src.core.uncertainty import UncertaintyEstimator
from src.core.monitoring import SystemMonitor
from src.analytics import AnalyticsEngine
from src.core.validation import ValidationSystem

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')
AdaptationID = str
ComponentID = str
ParameterName = str
ParameterValue = Any


class AdaptationState(Enum):
    """States for the adaptation lifecycle."""
    PROPOSED = auto()  # Initial state
    VALIDATED = auto()  # Passed validation checks
    STAGED = auto()    # Ready for deployment
    APPLIED = auto()   # Currently active
    VERIFIED = auto()  # Confirmed effective
    REJECTED = auto()  # Failed validation or verification
    ROLLED_BACK = auto()  # Reversed due to issues


class AdaptationPriority(Enum):
    """Priority levels for adaptations."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class AdaptationType(Enum):
    """Types of adaptations that can be performed."""
    PARAMETER_ADJUSTMENT = auto()  # Change numerical parameters
    STRATEGY_SWITCH = auto()       # Switch between strategies
    MODEL_UPDATE = auto()          # Update ML model
    THRESHOLD_CHANGE = auto()      # Modify decision thresholds
    FEATURE_TOGGLE = auto()        # Enable/disable features
    ALGORITHM_CHANGE = auto()      # Change core algorithms
    COMPONENT_RECONFIGURATION = auto()  # Reconfigure component architecture


@dataclass
class AdaptationTarget:
    """Represents the target of an adaptation."""
    component_id: ComponentID
    parameter_name: Optional[ParameterName] = None
    current_value: Optional[ParameterValue] = None
    
    def __str__(self) -> str:
        if self.parameter_name:
            return f"{self.component_id}.{self.parameter_name}"
        return self.component_id


@dataclass
class AdaptationChange:
    """Represents a specific change to be applied."""
    target: AdaptationTarget
    new_value: ParameterValue
    reversible: bool = True
    
    def __str__(self) -> str:
        return f"{self.target} -> {self.new_value}"


@dataclass
class AdaptationMetrics:
    """Metrics for measuring adaptation effectiveness."""
    pre_adaptation_metrics: Dict[str, float] = field(default_factory=dict)
    post_adaptation_metrics: Dict[str, float] = field(default_factory=dict)
    improvement_percentage: Dict[str, float] = field(default_factory=dict)
    confidence_level: float = 0.0
    uncertainty: float = 1.0
    
    def calculate_improvement(self) -> None:
        """Calculate improvement percentages for all metrics."""
        for key, post_value in self.post_adaptation_metrics.items():
            if key in self.pre_adaptation_metrics and self.pre_adaptation_metrics[key] != 0:
                pre_value = self.pre_adaptation_metrics[key]
                improvement = (post_value - pre_value) / abs(pre_value) * 100
                self.improvement_percentage[key] = improvement


@dataclass
class AdaptationVerificationResult:
    """Results of adaptation verification."""
    passed: bool
    metrics: AdaptationMetrics
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Adaptation:
    """Represents a complete adaptation record."""
    id: AdaptationID
    name: str
    description: str
    changes: List[AdaptationChange]
    type: AdaptationType
    priority: AdaptationPriority
    state: AdaptationState
    proposed_time: datetime
    applied_time: Optional[datetime] = None
    verified_time: Optional[datetime] = None
    rollback_time: Optional[datetime] = None
    verification_result: Optional[AdaptationVerificationResult] = None
    proposed_by: Optional[str] = None  # Component or user that proposed the adaptation
    verified_by: Optional[str] = None  # Component or user that verified the adaptation
    dependencies: Set[AdaptationID] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """Get the age of the adaptation in seconds."""
        return (datetime.now() - self.proposed_time).total_seconds()
    
    @property
    def is_reversible(self) -> bool:
        """Check if the adaptation is reversible."""
        return all(change.reversible for change in self.changes)
    
    @property
    def is_verified(self) -> bool:
        """Check if the adaptation has been verified."""
        return self.state == AdaptationState.VERIFIED
    
    @property
    def is_pending(self) -> bool:
        """Check if the adaptation is pending application."""
        return self.state in (AdaptationState.PROPOSED, AdaptationState.VALIDATED, AdaptationState.STAGED)
    
    @property
    def affected_components(self) -> Set[ComponentID]:
        """Get the set of components affected by this adaptation."""
        return {change.target.component_id for change in self.changes}


class AdaptationStrategy:
    """Base class for adaptation strategies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    async def generate_adaptations(
        self, 
        uncertainty_estimator: UncertaintyEstimator,
        system_monitor: SystemMonitor,
        analytics_engine: AnalyticsEngine
    ) -> List[Adaptation]:
        """Generate adaptation proposals based on system state."""
        raise NotImplementedError("Subclasses must implement generate_adaptations method")
    
    async def evaluate_adaptations(
        self, 
        adaptations: List[Adaptation],
        uncertainty_estimator: UncertaintyEstimator
    ) -> List[Tuple[Adaptation, float]]:
        """Evaluate and rank adaptations by expected benefit."""
        raise NotImplementedError("Subclasses must implement evaluate_adaptations method")


class SafeAdaptationManager:
    """
    Manages the safe application of adaptations, including validation,
    staged rollout, and rollback capabilities.
    """
    
    def __init__(
        self,
        validation_system: ValidationSystem,
        uncertainty_estimator: UncertaintyEstimator,
        system_monitor: SystemMonitor
    ):
        self.validation_system = validation_system
        self.uncertainty_estimator = uncertainty_estimator
        self.system_monitor = system_monitor
        self.adaptation_registry: Dict[AdaptationID, Adaptation] = {}
        self.active_adaptations: List[AdaptationID] = []
        self.rollback_history: List[Tuple[Adaptation, str, datetime]] = []
        self._component_adapters: Dict[ComponentID, Callable[[AdaptationChange], bool]] = {}
        self._component_rollbackers: Dict[ComponentID, Callable[[AdaptationChange], bool]] = {}
        
    def register_component_adapter(
        self, 
        component_id: ComponentID, 
        adapter: Callable[[AdaptationChange], bool],
        rollbacker: Callable[[AdaptationChange], bool]
    ) -> None:
        """
        Register functions to apply and rollback adaptations for a component.
        
        Args:
            component_id: Unique identifier for the component
            adapter: Function that applies an adaptation to the component
            rollbacker: Function that rolls back an adaptation from the component
        """
        self._component_adapters[component_id] = adapter
        self._component_rollbackers[component_id] = rollbacker
        
    async def validate_adaptation(self, adaptation: Adaptation) -> bool:
        """
        Validate an adaptation before it is applied.
        
        Args:
            adaptation: The adaptation to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        logger.info(f"Validating adaptation {adaptation.id}: {adaptation.name}")
        
        # Check for missing component adapters
        missing_adapters = [
            component_id for component_id in adaptation.affected_components
            if component_id not in self._component_adapters
        ]
        
        if missing_adapters:
            logger.error(f"Missing adapters for components: {missing_adapters}")
            return False
        
        # Check for non-reversible changes when rollback might be needed
        if not adaptation.is_reversible:
            logger.warning(f"Adaptation {adaptation.id} contains non-reversible changes")
        
        # Check for dependencies
        for dep_id in adaptation.dependencies:
            if dep_id not in self.adaptation_registry:
                logger.error(f"Dependency {dep_id} not found for adaptation {adaptation.id}")
                return False
            
            dep = self.adaptation_registry[dep_id]
            if dep.state not in (AdaptationState.APPLIED, AdaptationState.VERIFIED):
                logger.error(f"Dependency {dep_id} is not applied yet for adaptation {adaptation.id}")
                return False
        
        # Use the validation system for deep validation
        validation_result = await self.validation_system.validate_adaptation(adaptation)
        if not validation_result.valid:
            logger.error(f"Validation failed for adaptation {adaptation.id}: {validation_result.issues}")
            adaptation.state = AdaptationState.REJECTED
            return False
        
        # Update the adaptation state
        adaptation.state = AdaptationState.VALIDATED
        return True
        
    async def stage_adaptation(self, adaptation: Adaptation) -> bool:
        """
        Stage an adaptation for application.
        
        Args:
            adaptation: The adaptation to stage
            
        Returns:
            True if staging succeeds, False otherwise
        """
        # Ensure the adaptation is validated
        if adaptation.state != AdaptationState.VALIDATED:
            logger.error(f"Cannot stage unvalidated adaptation {adaptation.id}")
            return False
            
        logger.info(f"Staging adaptation {adaptation.id}: {adaptation.name}")
        
        # Calculate pre-adaptation metrics for later comparison
        metrics = AdaptationMetrics()
        metrics.pre_adaptation_metrics = await self.system_monitor.get_component_metrics(
            list(adaptation.affected_components)
        )
        
        # Estimate uncertainty
        uncertainty = await self.uncertainty_estimator.estimate_adaptation_uncertainty(adaptation)
        metrics.uncertainty = uncertainty
        
        # Set confidence level based on uncertainty (inverse relationship)
        metrics.confidence_level = max(0.0, 1.0 - uncertainty)
        
        # Store metrics in the adaptation
        adaptation.verification_result = AdaptationVerificationResult(
            passed=False,  # Will be updated after verification
            metrics=metrics
        )
        
        # Update the adaptation state
        adaptation.state = AdaptationState.STAGED
        return True
        
    async def apply_adaptation(self, adaptation: Adaptation) -> bool:
        """
        Apply a staged adaptation to the system.
        
        Args:
            adaptation: The adaptation to apply
            
        Returns:
            True if application succeeds, False otherwise
        """
        # Ensure the adaptation is staged
        if adaptation.state != AdaptationState.STAGED:
            logger.error(f"Cannot apply unstaged adaptation {adaptation.id}")
            return False
            
        logger.info(f"Applying adaptation {adaptation.id}: {adaptation.name}")
        
        try:
            # Apply all changes
            for change in adaptation.changes:
                component_id = change.target.component_id
                adapter = self._component_adapters[component_id]
                
                success = adapter(change)
                if not success:
                    logger.error(f"Failed to apply change {change} in adaptation {adaptation.id}")
                    await self._rollback_changes(adaptation, change)
                    adaptation.state = AdaptationState.REJECTED
                    return False
            
            # Update adaptation state and times
            adaptation.state = AdaptationState.APPLIED
            adaptation.applied_time = datetime.now()
            
            # Add to active adaptations
            self.active_adaptations.append(adaptation.id)
            
            return True
            
        except Exception as e:
            logger.exception(f"Error applying adaptation {adaptation.id}: {e}")
            await self._rollback_changes(adaptation)
            adaptation.state = AdaptationState.REJECTED
            return False
    
    async def _rollback_changes(
        self, 
        adaptation: Adaptation, 
        failed_change: Optional[AdaptationChange] = None
    ) -> None:
        """
        Roll back changes that have been applied.
        
        Args:
            adaptation: The adaptation to roll back
            failed_change: The change that failed, if any
        """
        logger.warning(f"Rolling back adaptation {adaptation.id}: {adaptation.name}")
        
        # Get the index of the failed change
        if failed_change:
            failed_idx = adaptation.changes.index(failed_change)
        else:
            failed_idx = len(adaptation.changes)
        
        # Roll back changes in reverse order up to the failed change
        for i in range(failed_idx - 1, -1, -1):
            change = adaptation.changes[i]
            
            if not change.reversible:
                logger.error(f"Cannot rollback non-reversible change {change}")
                continue
                
            component_id = change.target.component_id
            rollbacker = self._component_rollbackers.get(component_id)
            
            if not rollbacker:
                logger.error(f"No rollbacker found for component {component_id}")
                continue
                
            try:
                success = rollbacker(change)
                if not success:
                    logger.error(f"Failed to rollback change {change}")
            except Exception as e:
                logger.exception(f"Error rolling back change {change}: {e}")
    
    async def verify_adaptation(self, adaptation_id: AdaptationID) -> Optional[AdaptationVerificationResult]:
        """
        Verify the effectiveness of an applied adaptation.
        
        Args:
            adaptation_id: ID of the adaptation to verify
            
        Returns:
            Verification result or None if verification failed
        """
        if adaptation_id not in self.adaptation_registry:
            logger.error(f"Adaptation {adaptation_id} not found")
            return None
            
        adaptation = self.adaptation_registry[adaptation_id]
        
        if adaptation.state != AdaptationState.APPLIED:
            logger.error(f"Cannot verify unapplied adaptation {adaptation_id}")
            return None
            
        logger.info(f"Verifying adaptation {adaptation_id}: {adaptation.name}")
        
        # Get updated metrics
        if not adaptation.verification_result:
            logger.error(f"No metrics found for adaptation {adaptation_id}")
            return None
            
        metrics = adaptation.verification_result.metrics
        metrics.post_adaptation_metrics = await self.system_monitor.get

