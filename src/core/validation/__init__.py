"""
Validation Module for AAUto Trading System.

This module provides comprehensive validation capabilities including:
1. Component-level validation
2. System-wide integrity checks
3. Adaptation validation mechanisms
4. Performance validation
5. Regulatory compliance validation

It integrates with uncertainty quantification, calibration, and decision systems
to ensure reliable, compliant, and optimal system operation.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import numpy as np
from pydantic import BaseModel, Field, validator

from src.core.calibration import CalibrationMonitor, CalibrationResult
from src.core.decision import DecisionFramework, DecisionOutcome
from src.core.metadata import MetadataManager
from src.core.uncertainty import UncertaintyEstimator, UncertaintyLevel

# Type definitions
T = TypeVar('T')
ValidationFunction = Callable[..., "ValidationResult"]
ComponentId = str
ValidationId = str


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = auto()
    WARNING = auto()
    FAILED = auto()
    UNKNOWN = auto()


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    INFO = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class ValidationDomain(Enum):
    """Domains for validation checks."""
    COMPONENT = auto()
    SYSTEM = auto()
    ADAPTATION = auto()
    PERFORMANCE = auto()
    REGULATORY = auto()


@dataclass
class ValidationThresholds:
    """Configurable thresholds for validation checks."""
    warning_threshold: float = 0.8
    failure_threshold: float = 0.6
    min_confidence: float = 0.9
    max_uncertainty: float = 0.2
    min_calibration_score: float = 0.85
    max_drift: float = 0.15


class ValidationResult(BaseModel):
    """Result of a validation check."""
    id: ValidationId
    status: ValidationStatus
    domain: ValidationDomain
    severity: ValidationSeverity
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0)
    message: str
    component_id: Optional[ComponentId] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if validation result indicates validity."""
        return self.status in (ValidationStatus.PASSED, ValidationStatus.WARNING)
    
    @property
    def requires_attention(self) -> bool:
        """Check if validation result requires attention."""
        return (self.status == ValidationStatus.FAILED or 
                (self.status == ValidationStatus.WARNING and 
                 self.severity in (ValidationSeverity.HIGH, ValidationSeverity.CRITICAL)))


class ValidationReport(BaseModel):
    """Comprehensive report of validation results."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    results: List[ValidationResult] = Field(default_factory=list)
    overall_status: ValidationStatus = ValidationStatus.UNKNOWN
    summary: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def passed_count(self) -> int:
        """Count of passed validations."""
        return sum(1 for result in self.results if result.status == ValidationStatus.PASSED)
    
    @property
    def warning_count(self) -> int:
        """Count of validation warnings."""
        return sum(1 for result in self.results if result.status == ValidationStatus.WARNING)
    
    @property
    def failed_count(self) -> int:
        """Count of failed validations."""
        return sum(1 for result in self.results if result.status == ValidationStatus.FAILED)
    
    @property
    def critical_failures(self) -> List[ValidationResult]:
        """List critical validation failures."""
        return [r for r in self.results if r.status == ValidationStatus.FAILED 
                and r.severity == ValidationSeverity.CRITICAL]
    
    @property
    def is_valid(self) -> bool:
        """Check if overall validation indicates validity."""
        return (self.overall_status in (ValidationStatus.PASSED, ValidationStatus.WARNING) 
                and not self.critical_failures)


class ComponentValidator:
    """Validates individual system components."""
    
    def __init__(self, uncertainty_estimator: UncertaintyEstimator):
        """Initialize component validator.
        
        Args:
            uncertainty_estimator: Estimator for validation uncertainty
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.thresholds = ValidationThresholds()
        self.registered_validations: Dict[ComponentId, List[ValidationFunction]] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_validation(self, component_id: ComponentId, validation_fn: ValidationFunction) -> None:
        """Register a validation function for a component.
        
        Args:
            component_id: Unique identifier for the component
            validation_fn: Function that performs validation
        """
        if component_id not in self.registered_validations:
            self.registered_validations[component_id] = []
        self.registered_validations[component_id].append(validation_fn)
        self.logger.debug(f"Registered validation for component {component_id}")
    
    async def validate_component(self, component_id: ComponentId, 
                                component: Any) -> List[ValidationResult]:
        """Validate a specific component.
        
        Args:
            component_id: Unique identifier for the component
            component: The component instance to validate
            
        Returns:
            List of validation results
        """
        results = []
        
        if component_id not in self.registered_validations:
            self.logger.warning(f"No validations registered for component {component_id}")
            return results
        
        for validation_fn in self.registered_validations[component_id]:
            try:
                result = await asyncio.coroutine(validation_fn)(component)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Validation error for component {component_id}: {str(e)}")
                # Create a failure result for the failed validation
                results.append(ValidationResult(
                    id=f"{component_id}_validation_error",
                    status=ValidationStatus.FAILED,
                    domain=ValidationDomain.COMPONENT,
                    severity=ValidationSeverity.HIGH,
                    score=0.0,
                    confidence=1.0,  # High confidence that validation failed
                    uncertainty=0.0,
                    message=f"Validation function raised an exception: {str(e)}",
                    component_id=component_id,
                    metadata={"error_type": type(e).__name__}
                ))
        
        return results


class SystemIntegrityValidator:
    """Validates system-wide integrity and interactions between components."""
    
    def __init__(self, uncertainty_estimator: UncertaintyEstimator):
        """Initialize system integrity validator.
        
        Args:
            uncertainty_estimator: Estimator for validation uncertainty
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.thresholds = ValidationThresholds()
        self.integrity_checks: List[Tuple[ValidationFunction, List[ComponentId]]] = []
        self.logger = logging.getLogger(__name__)
    
    def register_integrity_check(self, validation_fn: ValidationFunction, 
                                component_ids: List[ComponentId]) -> None:
        """Register a system integrity check.
        
        Args:
            validation_fn: Function that performs validation
            component_ids: List of component IDs involved in the check
        """
        self.integrity_checks.append((validation_fn, component_ids))
        self.logger.debug(f"Registered integrity check for components {component_ids}")
    
    async def validate_system_integrity(self, 
                                       components: Dict[ComponentId, Any]) -> List[ValidationResult]:
        """Validate system-wide integrity.
        
        Args:
            components: Dictionary of component instances by ID
            
        Returns:
            List of validation results
        """
        results = []
        
        for validation_fn, component_ids in self.integrity_checks:
            # Check if all required components are available
            missing_components = [cid for cid in component_ids if cid not in components]
            if missing_components:
                self.logger.warning(f"Cannot perform integrity check, missing components: {missing_components}")
                results.append(ValidationResult(
                    id=f"integrity_missing_components",
                    status=ValidationStatus.FAILED,
                    domain=ValidationDomain.SYSTEM,
                    severity=ValidationSeverity.HIGH,
                    score=0.0,
                    confidence=1.0,
                    uncertainty=0.0,
                    message=f"Cannot perform integrity check, missing components: {missing_components}",
                    metadata={"missing_components": missing_components}
                ))
                continue
            
            try:
                # Extract the relevant components
                relevant_components = {cid: components[cid] for cid in component_ids}
                result = await asyncio.coroutine(validation_fn)(relevant_components)
                results.append(result)
            except Exception as e:
                self.logger.error(f"System integrity validation error: {str(e)}")
                results.append(ValidationResult(
                    id=f"integrity_validation_error",
                    status=ValidationStatus.FAILED,
                    domain=ValidationDomain.SYSTEM,
                    severity=ValidationSeverity.HIGH,
                    score=0.0,
                    confidence=1.0,
                    uncertainty=0.0,
                    message=f"System integrity validation raised an exception: {str(e)}",
                    metadata={"error_type": type(e).__name__, "component_ids": component_ids}
                ))
        
        return results


class AdaptationValidator:
    """Validates adaptation mechanisms and changes."""
    
    def __init__(self, uncertainty_estimator: UncertaintyEstimator, 
                 calibration_monitor: CalibrationMonitor):
        """Initialize adaptation validator.
        
        Args:
            uncertainty_estimator: Estimator for validation uncertainty
            calibration_monitor: Monitor for calibration assessment
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.calibration_monitor = calibration_monitor
        self.thresholds = ValidationThresholds()
        self.adaptation_validations: Dict[str, ValidationFunction] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_adaptation_validation(self, adaptation_id: str, 
                                      validation_fn: ValidationFunction) -> None:
        """Register an adaptation validation function.
        
        Args:
            adaptation_id: Identifier for the adaptation mechanism
            validation_fn: Function that performs validation
        """
        self.adaptation_validations[adaptation_id] = validation_fn
        self.logger.debug(f"Registered validation for adaptation {adaptation_id}")
    
    async def validate_adaptation(self, adaptation_id: str, 
                                 before_state: Any, after_state: Any, 
                                 parameters: Dict[str, Any]) -> ValidationResult:
        """Validate an adaptation based on before and after states.
        
        Args:
            adaptation_id: Identifier for the adaptation mechanism
            before_state: System state before adaptation
            after_state: System state after adaptation
            parameters: Parameters used for the adaptation
            
        Returns:
            Validation result
        """
        if adaptation_id not in self.adaptation_validations:
            self.logger.warning(f"No validation registered for adaptation {adaptation_id}")
            return ValidationResult(
                id=f"{adaptation_id}_no_validation",
                status=ValidationStatus.UNKNOWN,
                domain=ValidationDomain.ADAPTATION,
                severity=ValidationSeverity.MEDIUM,
                score=0.5,
                confidence=0.5,
                uncertainty=0.5,
                message=f"No validation registered for adaptation {adaptation_id}",
                metadata={"adaptation_id": adaptation_id}
            )
        
        try:
            validation_fn = self.adaptation_validations[adaptation_id]
            result = await asyncio.coroutine(validation_fn)(before_state, after_state, parameters)
            return result
        except Exception as e:
            self.logger.error(f"Adaptation validation error for {adaptation_id}: {str(e)}")
            return ValidationResult(
                id=f"{adaptation_id}_validation_error",
                status=ValidationStatus.FAILED,
                domain=ValidationDomain.ADAPTATION,
                severity=ValidationSeverity.HIGH,
                score=0.0,
                confidence=1.0,
                uncertainty=0.0,
                message=f"Adaptation validation raised an exception: {str(e)}",
                metadata={"error_type": type(e).__name__, "adaptation_id": adaptation_id}
            )


class PerformanceValidator:
    """Validates system performance against benchmarks and requirements."""
    
    def __init__(self, uncertainty_estimator: UncertaintyEstimator):
        """Initialize performance validator.
        
        Args:
            uncertainty_estimator: Estimator for validation uncertainty
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.thresholds = ValidationThresholds()
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.performance_validations: Dict[str, ValidationFunction] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_performance_metric(self, metric_id: str, 
                                   expected_range: Tuple[float, float],
                                   critical_threshold: Optional[float] = None) -> None:
        """Register a performance metric with expected values.
        
        Args:
            metric_id: Identifier for the performance metric
            expected_range: Tuple of (min, max) expected values
            critical_threshold: Optional threshold for critical failures
        """
        self.performance_metrics[metric_id] = {
            "expected_range": expected_range,
            "critical_threshold": critical_threshold
        }
        self.logger.debug(f"Registered performance metric {metric_id}")
    
    def register_performance_validation(self, validation_id: str, 
                                       validation_fn: ValidationFunction) -> None:
        """Register a performance validation function.
        
        Args:
            validation_id: Identifier for the validation
            validation_fn: Function that performs validation
        """
        self.performance_validations[validation_id] = validation_fn
        self.logger.debug(f"Registered performance validation {validation_id}")
    
    async def validate_metric(self, metric_id: str, value: float) -> ValidationResult:
        """Validate a performance metric against expected values.
        
        Args:
            metric_id: Identifier for the performance metric
            value: Measured value of the metric
            
        Returns:
            Validation result
        """
        if metric_id not in self.performance_metrics:
            self.logger.warning(f"No expected values registered for metric {metric_id}")
            return ValidationResult(
                id=f"{metric_id}_no_expectations",
                status=ValidationStatus.UNKNOWN,
                domain=ValidationDomain.PERFORMANCE,
                severity=ValidationSeverity.MEDIUM,
                score=0.5,
                confidence=0.5,
                uncertainty=0.5,
                message=f"No expected values registered for

