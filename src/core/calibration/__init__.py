"""
Calibration Module for Uncertainty Quantification

This module provides tools for monitoring, assessing, and improving the
calibration of probabilistic predictions. It integrates with the uncertainty
quantification system to ensure reliable uncertainty estimates.

Features:
- Online calibration monitoring
- Multi-metric calibration assessment 
- Adaptive recalibration mechanisms
- Performance tracking and validation
- Feedback loops for system adjustment
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

try:
    from ..uncertainty import UncertaintyEstimator
except ImportError:
    # For when the module is imported standalone
    class UncertaintyEstimator:
        pass


class CalibrationMetric(Enum):
    """Metrics used to assess calibration quality."""
    EXPECTED_CALIBRATION_ERROR = "ece"
    MAXIMUM_CALIBRATION_ERROR = "mce"
    BRIER_SCORE = "brier"
    LOG_LOSS = "log_loss"
    SHARPNESS = "sharpness"
    CONFIDENCE_HISTOGRAM_DISPERSION = "confidence_histogram_dispersion"
    RELIABILITY_DIAGRAM_SLOPE = "reliability_diagram_slope"


@dataclass
class CalibrationResult:
    """Results from a calibration assessment."""
    metrics: Dict[CalibrationMetric, float]
    timestamp: float = field(default_factory=time.time)
    calibration_curve: Optional[List[Tuple[float, float]]] = None
    confidence_histogram: Optional[List[int]] = None
    is_well_calibrated: bool = False
    needs_recalibration: bool = False


@dataclass
class RecalibrationConfig:
    """Configuration for recalibration operations."""
    method: str = "temperature_scaling"
    window_size: int = 1000  # Number of samples to consider
    min_samples_required: int = 200
    recalibration_threshold: float = 0.1  # ECE threshold to trigger recalibration
    max_iterations: int = 100  # For iterative methods
    learning_rate: float = 0.01  # For gradient-based methods
    regularization_strength: float = 0.001


class CalibrationMonitor:
    """
    Monitors the calibration of probabilistic predictions in real-time.
    
    This class tracks prediction probabilities and actual outcomes to assess
    whether a model is well-calibrated and triggers recalibration when needed.
    """
    
    def __init__(
        self,
        window_size: int = 1000,
        assessment_frequency: int = 100,
        metrics: List[CalibrationMetric] = None,
        recalibration_config: Optional[RecalibrationConfig] = None,
        uncertainty_estimator: Optional[UncertaintyEstimator] = None,
    ):
        """
        Initialize the calibration monitor.
        
        Args:
            window_size: Number of recent predictions to maintain for calibration assessment
            assessment_frequency: Frequency of calibration assessment (in number of predictions)
            metrics: List of calibration metrics to compute
            recalibration_config: Configuration for recalibration operations
            uncertainty_estimator: Optional uncertainty estimator to integrate with
        """
        self.window_size = window_size
        self.assessment_frequency = assessment_frequency
        self.metrics = metrics or [
            CalibrationMetric.EXPECTED_CALIBRATION_ERROR,
            CalibrationMetric.BRIER_SCORE,
            CalibrationMetric.SHARPNESS
        ]
        
        self.recalibration_config = recalibration_config or RecalibrationConfig()
        self.uncertainty_estimator = uncertainty_estimator
        
        # Storage for recent predictions and outcomes
        self.confidences = deque(maxlen=window_size)
        self.outcomes = deque(maxlen=window_size)
        self.prediction_times = deque(maxlen=window_size)
        
        # Performance tracking
        self.calibration_history = []
        self.recalibration_history = []
        self.current_calibration = None
        
        # For calibration curve computation
        self.num_bins = 10
        
        # Counters
        self.total_predictions = 0
        self.assessment_counter = 0
        
        self.logger = logging.getLogger(__name__)
        self.is_active = True
    
    async def add_prediction(self, confidence: float, outcome: bool) -> None:
        """
        Add a new prediction and its outcome to the monitor.
        
        Args:
            confidence: Predicted probability or confidence score [0,1]
            outcome: Actual outcome (True/False or 1/0)
        """
        self.confidences.append(confidence)
        self.outcomes.append(1 if outcome else 0)
        self.prediction_times.append(time.time())
        self.total_predictions += 1
        self.assessment_counter += 1
        
        # Check if we should perform assessment
        if self.assessment_counter >= self.assessment_frequency and len(self.confidences) >= self.recalibration_config.min_samples_required:
            self.assessment_counter = 0
            await self.assess_calibration()
    
    async def assess_calibration(self) -> CalibrationResult:
        """
        Assess the current calibration of the model.
        
        Returns:
            CalibrationResult: Results of calibration assessment
        """
        confidences = np.array(self.confidences)
        outcomes = np.array(self.outcomes)
        
        if len(confidences) < self.recalibration_config.min_samples_required:
            self.logger.info(f"Not enough samples for calibration assessment. Have {len(confidences)}, need {self.recalibration_config.min_samples_required}")
            return None
        
        result = CalibrationResult(metrics={})
        
        # Compute calibration curve (reliability diagram)
        bin_edges = np.linspace(0, 1, self.num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Calculate calibration curve
        calibration_curve = []
        confidence_histogram = [0] * self.num_bins
        
        for bin_idx in range(self.num_bins):
            mask = bin_indices == bin_idx
            bin_count = np.sum(mask)
            confidence_histogram[bin_idx] = bin_count
            
            if bin_count > 0:
                avg_confidence = np.mean(confidences[mask])
                avg_accuracy = np.mean(outcomes[mask])
                calibration_curve.append((avg_confidence, avg_accuracy))
            else:
                # Use bin midpoint as placeholder
                bin_midpoint = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                calibration_curve.append((bin_midpoint, np.nan))
        
        result.calibration_curve = calibration_curve
        result.confidence_histogram = confidence_histogram
        
        # Calculate requested metrics
        for metric in self.metrics:
            if metric == CalibrationMetric.EXPECTED_CALIBRATION_ERROR:
                result.metrics[metric] = self._calculate_ece(confidences, outcomes, self.num_bins)
            elif metric == CalibrationMetric.MAXIMUM_CALIBRATION_ERROR:
                result.metrics[metric] = self._calculate_mce(confidences, outcomes, self.num_bins)
            elif metric == CalibrationMetric.BRIER_SCORE:
                result.metrics[metric] = np.mean((confidences - outcomes) ** 2)
            elif metric == CalibrationMetric.LOG_LOSS:
                epsilon = 1e-15  # To avoid log(0)
                log_loss = -np.mean(
                    outcomes * np.log(np.clip(confidences, epsilon, 1 - epsilon)) +
                    (1 - outcomes) * np.log(np.clip(1 - confidences, epsilon, 1 - epsilon))
                )
                result.metrics[metric] = log_loss
            elif metric == CalibrationMetric.SHARPNESS:
                result.metrics[metric] = np.var(confidences)
        
        # Determine if calibration is sufficient or recalibration is needed
        if CalibrationMetric.EXPECTED_CALIBRATION_ERROR in result.metrics:
            ece = result.metrics[CalibrationMetric.EXPECTED_CALIBRATION_ERROR]
            result.is_well_calibrated = ece < self.recalibration_config.recalibration_threshold
            result.needs_recalibration = ece >= self.recalibration_config.recalibration_threshold
        
        # Store result in history
        self.current_calibration = result
        self.calibration_history.append(result)
        
        # Trigger recalibration if needed
        if result.needs_recalibration:
            self.logger.info(f"Recalibration needed. ECE: {ece:.4f} (threshold: {self.recalibration_config.recalibration_threshold})")
            await self.recalibrate()
        
        return result
    
    async def recalibrate(self) -> bool:
        """
        Recalibrate the model based on recent predictions.
        
        Returns:
            bool: True if recalibration was successful
        """
        if not self.is_active or len(self.confidences) < self.recalibration_config.min_samples_required:
            return False
        
        confidences = np.array(self.confidences)
        outcomes = np.array(self.outcomes)
        
        method = self.recalibration_config.method
        self.logger.info(f"Starting recalibration using {method} method")
        
        try:
            if method == "temperature_scaling":
                parameters = await self._temperature_scaling(confidences, outcomes)
                success = True
            elif method == "isotonic_regression":
                parameters = await self._isotonic_regression(confidences, outcomes)
                success = True
            elif method == "beta_calibration":
                parameters = await self._beta_calibration(confidences, outcomes)
                success = True
            else:
                self.logger.error(f"Unknown recalibration method: {method}")
                return False
            
            # Record recalibration event
            self.recalibration_history.append({
                "timestamp": time.time(),
                "method": method,
                "parameters": parameters,
                "samples_used": len(confidences),
                "pre_calibration_metrics": self.current_calibration.metrics
            })
            
            # Update uncertainty estimator if available
            if self.uncertainty_estimator:
                await self._update_uncertainty_estimator(method, parameters)
            
            # Verify improvement
            await self.assess_calibration()
            
            self.logger.info(f"Recalibration complete. New ECE: {self.current_calibration.metrics.get(CalibrationMetric.EXPECTED_CALIBRATION_ERROR, 'N/A')}")
            return success
            
        except Exception as e:
            self.logger.error(f"Recalibration failed: {str(e)}")
            return False
    
    def _calculate_ece(self, confidences: np.ndarray, outcomes: np.ndarray, num_bins: int) -> float:
        """
        Calculate Expected Calibration Error.
        
        Args:
            confidences: Predicted confidence scores
            outcomes: Actual outcomes
            num_bins: Number of bins for discretization
            
        Returns:
            float: Expected Calibration Error
        """
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        ece = 0.0
        total_samples = len(confidences)
        
        for bin_idx in range(num_bins):
            mask = bin_indices == bin_idx
            bin_count = np.sum(mask)
            
            if bin_count > 0:
                bin_conf = np.mean(confidences[mask])
                bin_acc = np.mean(outcomes[mask])
                ece += (bin_count / total_samples) * np.abs(bin_conf - bin_acc)
        
        return float(ece)
    
    def _calculate_mce(self, confidences: np.ndarray, outcomes: np.ndarray, num_bins: int) -> float:
        """
        Calculate Maximum Calibration Error.
        
        Args:
            confidences: Predicted confidence scores
            outcomes: Actual outcomes
            num_bins: Number of bins for discretization
            
        Returns:
            float: Maximum Calibration Error
        """
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        max_ce = 0.0
        
        for bin_idx in range(num_bins):
            mask = bin_indices == bin_idx
            bin_count = np.sum(mask)
            
            if bin_count > 0:
                bin_conf = np.mean(confidences[mask])
                bin_acc = np.mean(outcomes[mask])
                ce = np.abs(bin_conf - bin_acc)
                max_ce = max(max_ce, ce)
        
        return float(max_ce)
    
    async def _temperature_scaling(self, confidences: np.ndarray, outcomes: np.ndarray) -> Dict:
        """
        Recalibrate using Temperature Scaling method.
        
        Args:
            confidences: Raw confidence scores
            outcomes: Actual outcomes
            
        Returns:
            Dict: Calibration parameters (temperature)
        """
        from scipy.optimize import minimize
        
        # Define the negative log likelihood loss for temperature scaling
        def nll_loss(T):
            # Apply temperature scaling
            scaled_confidences = 1.0 / (1.0 + np.exp(-(np.log(confidences / (1 - confidences)) / T)))
            
            # Calculate negative log likelihood
            epsilon = 1e-15
            scaled_confidences = np.clip(scaled_confidences, epsilon, 1 - epsilon)
            nll = -np.mean(
                outcomes * np.log(scaled_confidences) +
                (1 - outcomes) * np.log(1 - scaled_confidences)
            )
            return nll
        
        # Initial temperature
        T_init = 1.0
        
        # Optimize to find the best temperature
        result = minimize(
            nll_loss, 
            x0=np.array([T_init]), 
            method='BFGS', 
            options={'gtol': 1e-4, 'disp': False}
        )
        
        temperature = result.x[0]
        return {"temperature": float(temperature)}
    
    async def _isotonic_regression(self, confidences: np.ndarray, outcomes: np.ndarray) -> Dict:
        """
        Recalibrate using Isotonic Regression.
        
        Args:
            confidences: Raw confidence scores
            outcomes

