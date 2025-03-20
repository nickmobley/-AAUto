"""
Decision Framework for uncertainty-aware and risk-adjusted decision making.

This module provides a comprehensive framework for making decisions under uncertainty,
with built-in risk adjustment, outcome tracking, and explanation generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, Set, Tuple, TypeVar, Union
import uuid

from ..uncertainty import UncertaintyEstimator, UncertaintyLevel, UncertaintyMetrics
from ..calibration import CalibrationMonitor, CalibrationStatus

# Type definitions
T = TypeVar('T')  # Decision input type
R = TypeVar('R')  # Decision result type

logger = logging.getLogger(__name__)


class DecisionOutcome(Enum):
    """Possible outcomes of a decision after evaluation."""
    SUCCESSFUL = auto()
    SUBOPTIMAL = auto()
    FAILED = auto()
    INCONCLUSIVE = auto()


class DecisionStatus(Enum):
    """Status of a decision in its lifecycle."""
    PENDING = auto()
    EXECUTED = auto()
    EVALUATED = auto()
    ARCHIVED = auto()


@dataclass
class RiskProfile:
    """Defines the risk tolerance parameters for decision making."""
    max_acceptable_uncertainty: float = 0.3
    min_required_confidence: float = 0.7
    max_downside_exposure: float = 0.05
    volatility_tolerance: float = 0.2
    loss_aversion_factor: float = 2.0
    adaptive_threshold: bool = True
    
    def is_within_risk_tolerance(self, uncertainty: UncertaintyMetrics) -> bool:
        """Determine if the given uncertainty is within risk tolerance."""
        if uncertainty.combined_uncertainty > self.max_acceptable_uncertainty:
            return False
        if uncertainty.confidence < self.min_required_confidence:
            return False
        if uncertainty.downside_risk > self.max_downside_exposure:
            return False
        if uncertainty.volatility > self.volatility_tolerance:
            return False
        return True
    
    def adjust_for_market_regime(self, volatility_index: float, correlation_matrix: Any = None) -> 'RiskProfile':
        """Create a new risk profile adjusted for current market conditions."""
        new_profile = RiskProfile(
            max_acceptable_uncertainty=self.max_acceptable_uncertainty * (1.0 - 0.5 * volatility_index),
            min_required_confidence=self.min_required_confidence + (0.1 * volatility_index),
            max_downside_exposure=self.max_downside_exposure * (1.0 - 0.7 * volatility_index),
            volatility_tolerance=self.volatility_tolerance * (1.0 + 0.3 * volatility_index),
            loss_aversion_factor=self.loss_aversion_factor * (1.0 + 0.5 * volatility_index),
            adaptive_threshold=self.adaptive_threshold
        )
        return new_profile


@dataclass
class DecisionMetadata:
    """Metadata associated with a decision."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    source_system: str = ""
    user_id: Optional[str] = None
    parent_decision_id: Optional[str] = None
    
    def add_context(self, key: str, value: Any) -> None:
        """Add additional context to the decision."""
        self.context[key] = value
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the decision."""
        self.tags.add(tag)


class DecisionEvaluator(Protocol, Generic[R]):
    """Protocol for evaluating the outcome of a decision."""
    
    def evaluate(self, result: R, context: Dict[str, Any]) -> DecisionOutcome:
        """Evaluate the outcome of a decision based on its result."""
        ...


@dataclass
class Decision(Generic[T, R]):
    """Represents a decision with uncertainty and risk information."""
    input_data: T
    alternatives: List[T]
    metadata: DecisionMetadata
    uncertainty: UncertaintyMetrics
    risk_profile: RiskProfile
    calibration_status: CalibrationStatus
    selected_option: Optional[T] = None
    result: Optional[R] = None
    status: DecisionStatus = DecisionStatus.PENDING
    explanation: str = ""
    outcome: Optional[DecisionOutcome] = None
    
    def with_selection(self, selected: T) -> 'Decision[T, R]':
        """Create a new decision with the selected option."""
        return Decision(
            input_data=self.input_data,
            alternatives=self.alternatives,
            metadata=self.metadata,
            uncertainty=self.uncertainty,
            risk_profile=self.risk_profile,
            calibration_status=self.calibration_status,
            selected_option=selected,
            status=self.status,
            explanation=self.explanation
        )
    
    def with_result(self, result: R) -> 'Decision[T, R]':
        """Create a new decision with the execution result."""
        return Decision(
            input_data=self.input_data,
            alternatives=self.alternatives,
            metadata=self.metadata,
            uncertainty=self.uncertainty,
            risk_profile=self.risk_profile,
            calibration_status=self.calibration_status,
            selected_option=self.selected_option,
            result=result,
            status=DecisionStatus.EXECUTED,
            explanation=self.explanation
        )
    
    def with_evaluation(self, outcome: DecisionOutcome) -> 'Decision[T, R]':
        """Create a new decision with the evaluation outcome."""
        return Decision(
            input_data=self.input_data,
            alternatives=self.alternatives,
            metadata=self.metadata,
            uncertainty=self.uncertainty,
            risk_profile=self.risk_profile,
            calibration_status=self.calibration_status,
            selected_option=self.selected_option,
            result=self.result,
            status=DecisionStatus.EVALUATED,
            explanation=self.explanation,
            outcome=outcome
        )
    
    def with_explanation(self, explanation: str) -> 'Decision[T, R]':
        """Create a new decision with an explanation."""
        return Decision(
            input_data=self.input_data,
            alternatives=self.alternatives,
            metadata=self.metadata,
            uncertainty=self.uncertainty,
            risk_profile=self.risk_profile,
            calibration_status=self.calibration_status,
            selected_option=self.selected_option,
            result=self.result,
            status=self.status,
            explanation=explanation,
            outcome=self.outcome
        )


class DecisionTracker:
    """Tracks decisions and their outcomes for analysis and improvement."""
    
    def __init__(self):
        self.decisions: Dict[str, Decision] = {}
        self.pending_decisions: Set[str] = set()
        self.successful_decisions: Set[str] = set()
        self.failed_decisions: Set[str] = set()
    
    async def track_decision(self, decision: Decision) -> None:
        """Track a new decision."""
        self.decisions[decision.metadata.decision_id] = decision
        if decision.status == DecisionStatus.PENDING:
            self.pending_decisions.add(decision.metadata.decision_id)
        
        # Log the decision
        logger.info(
            f"Decision {decision.metadata.decision_id} tracked: "
            f"status={decision.status.name}, "
            f"uncertainty={decision.uncertainty.combined_uncertainty:.2f}"
        )
    
    async def update_decision(self, decision: Decision) -> None:
        """Update an existing decision."""
        decision_id = decision.metadata.decision_id
        
        if decision_id not in self.decisions:
            raise ValueError(f"Decision {decision_id} not found in tracker")
        
        old_status = self.decisions[decision_id].status
        self.decisions[decision_id] = decision
        
        # Update status tracking sets
        if old_status == DecisionStatus.PENDING and decision.status != DecisionStatus.PENDING:
            self.pending_decisions.remove(decision_id)
        
        if decision.status == DecisionStatus.EVALUATED:
            if decision.outcome == DecisionOutcome.SUCCESSFUL:
                self.successful_decisions.add(decision_id)
            elif decision.outcome == DecisionOutcome.FAILED:
                self.failed_decisions.add(decision_id)
        
        # Log the update
        logger.info(
            f"Decision {decision_id} updated: "
            f"status={decision.status.name}, "
            f"outcome={decision.outcome.name if decision.outcome else 'None'}"
        )
    
    def get_decision(self, decision_id: str) -> Optional[Decision]:
        """Retrieve a decision by ID."""
        return self.decisions.get(decision_id)
    
    def get_decisions_by_tag(self, tag: str) -> List[Decision]:
        """Retrieve all decisions with the given tag."""
        return [
            decision for decision in self.decisions.values()
            if tag in decision.metadata.tags
        ]
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of evaluated decisions."""
        evaluated = len(self.successful_decisions) + len(self.failed_decisions)
        if evaluated == 0:
            return 0.0
        return len(self.successful_decisions) / evaluated
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """Generate a report on decision performance."""
        total = len(self.decisions)
        pending = len(self.pending_decisions)
        successful = len(self.successful_decisions)
        failed = len(self.failed_decisions)
        
        success_rate = self.get_success_rate()
        
        # Calculate average uncertainty for successful vs failed decisions
        avg_uncertainty_success = 0.0
        avg_uncertainty_failed = 0.0
        
        if successful > 0:
            avg_uncertainty_success = sum(
                self.decisions[d_id].uncertainty.combined_uncertainty
                for d_id in self.successful_decisions
            ) / successful
        
        if failed > 0:
            avg_uncertainty_failed = sum(
                self.decisions[d_id].uncertainty.combined_uncertainty
                for d_id in self.failed_decisions
            ) / failed
        
        return {
            "total_decisions": total,
            "pending_decisions": pending,
            "successful_decisions": successful,
            "failed_decisions": failed,
            "success_rate": success_rate,
            "avg_uncertainty_success": avg_uncertainty_success,
            "avg_uncertainty_failed": avg_uncertainty_failed,
            "timestamp": datetime.now()
        }


class AdaptiveThresholdManager:
    """Manages thresholds that adapt based on historical performance and uncertainty."""
    
    def __init__(self, decision_tracker: DecisionTracker, initial_threshold: float = 0.5):
        self.decision_tracker = decision_tracker
        self.base_threshold = initial_threshold
        self.thresholds: Dict[str, float] = {}
        self.adjustment_factor = 0.1
        self.min_samples = 10
        self.learning_rate = 0.05
    
    async def get_threshold(self, context_key: str) -> float:
        """Get the current threshold for a specific context."""
        if context_key not in self.thresholds:
            return self.base_threshold
        return self.thresholds[context_key]
    
    async def update_thresholds(self) -> None:
        """Update thresholds based on decision outcomes."""
        # Get all decisions
        all_decisions = list(self.decision_tracker.decisions.values())
        evaluated_decisions = [d for d in all_decisions if d.status == DecisionStatus.EVALUATED]
        
        # Group decisions by context
        decisions_by_context: Dict[str, List[Decision]] = {}
        
        for decision in evaluated_decisions:
            for key in decision.metadata.context:
                if key not in decisions_by_context:
                    decisions_by_context[key] = []
                decisions_by_context[key].append(decision)
        
        # Update thresholds for each context
        for context_key, decisions in decisions_by_context.items():
            if len(decisions) < self.min_samples:
                continue
            
            # Calculate success rate for different uncertainty levels
            decisions_by_uncertainty: Dict[float, List[Tuple[Decision, bool]]] = {}
            
            for decision in decisions:
                # Round uncertainty to nearest 0.1
                uncertainty_level = round(decision.uncertainty.combined_uncertainty * 10) / 10
                if uncertainty_level not in decisions_by_uncertainty:
                    decisions_by_uncertainty[uncertainty_level] = []
                
                success = decision.outcome == DecisionOutcome.SUCCESSFUL
                decisions_by_uncertainty[uncertainty_level].append((decision, success))
            
            # Find the uncertainty level with the best success rate
            best_uncertainty = 0.0
            best_success_rate = 0.0
            
            for uncertainty, decision_results in decisions_by_uncertainty.items():
                if len(decision_results) < 3:  # Need enough samples at this level
                    continue
                
                success_count = sum(1 for _, success in decision_results if success)
                success_rate = success_count / len(decision_results)
                
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_uncertainty = uncertainty
            
            # Update threshold with exponential moving average
            current = self.thresholds.get(context_key, self.base_threshold)
            updated = current * (1 - self.learning_rate) + best_uncertainty * self.learning_rate
            self.thresholds[context_key] = updated
            
            logger.info(
                f"Updated threshold for {context_key}: {current:.2f} -> {updated:.2f} "
                f"(best_uncertainty={best_uncertainty:.2f}, success_rate={best_success_rate:.2f})"
            )
    
    async def adjust_threshold_for_market(
        self,
        context_key: str,
        market_volatility: float,
        risk_aversion: float
    ) -> float:
        """Adjust threshold based on current market conditions."""
        base = await self.get_threshold(context_key)
        
        # Reduce threshold (accept more uncertainty) in low volatility
        # Increase threshold (require more certainty) in high volatility
        volatility_adjustment = market_volatility * self.adjustment_factor
        
        # Risk aversion factor (higher = more conservative)
        risk_adjustment = risk_aversion * self.adjustment_factor
        
        adjusted = base + volatility_adjustment + risk_adjustment
        
        # Ensure the threshold is between 0 and 1
        adjusted = max(0.1, min(0.9, adjusted))
        
        return adjusted


class ExplanationGenerator:
    """Generates human-readable explanations for decisions."""
    
    def generate_decision_explanation(self, decision: Decision) -> str:
        

