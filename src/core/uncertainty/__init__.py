"""
Uncertainty Quantification Framework

This module provides comprehensive uncertainty estimation capabilities for the trading system,
including Bayesian uncertainty estimation, probabilistic forecasting, uncertainty propagation,
calibration monitoring, and uncertainty-aware decision making.
"""

import asyncio
import logging
import numpy as np
import typing as tp
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import scipy.stats
from sklearn.calibration import CalibrationDisplay, calibration_curve

# Type aliases for improved readability
PredictionType = np.ndarray
UncertaintyType = np.ndarray
ProbabilisticForecast = Tuple[PredictionType, UncertaintyType]
CalibrationMetrics = Dict[str, float]


class UncertaintyMethod(Enum):
    """Methods for uncertainty estimation."""
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"
    BOOTSTRAP = "bootstrap"
    QUANTILE = "quantile"
    MONTE_CARLO_DROPOUT = "mc_dropout"
    CONFORMAL = "conformal"


class ScoringRule(Enum):
    """Proper scoring rules for evaluating probabilistic forecasts."""
    LOG_SCORE = "log_score"
    CRPS = "continuous_ranked_probability_score"
    BRIER_SCORE = "brier_score"
    INTERVAL_SCORE = "interval_score"
    ENERGY_SCORE = "energy_score"


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimation methods."""
    method: UncertaintyMethod = UncertaintyMethod.BAYESIAN
    num_samples: int = 100
    confidence_level: float = 0.95
    calibration_window: int = 1000
    recalibration_frequency: int = 100
    enable_adaptive_recalibration: bool = True
    propagation_method: str = "monte_carlo"
    decision_threshold: float = 0.7
    cached_calibration_path: Optional[Path] = None


@dataclass
class CalibrationResult:
    """Results from calibration evaluation."""
    timestamp: datetime = field(default_factory=datetime.now)
    expected_vs_observed: Dict[str, np.ndarray] = field(default_factory=dict)
    sharpness: float = 0.0
    resolution: float = 0.0
    calibration_error: float = 0.0
    is_well_calibrated: bool = False
    calibration_scores: Dict[str, float] = field(default_factory=dict)
    recalibration_map: Optional[Callable] = None


class BayesianEstimator:
    """Bayesian methods for uncertainty estimation."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    async def estimate_uncertainty(
        self, 
        model: Any, 
        inputs: np.ndarray,
        prior: Optional[Dict[str, Any]] = None
    ) -> ProbabilisticForecast:
        """
        Estimate prediction uncertainty using Bayesian methods.
        
        Args:
            model: The prediction model
            inputs: Input data for prediction
            prior: Prior distributions for Bayesian inference
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.logger.debug(f"Estimating Bayesian uncertainty for {len(inputs)} inputs")
        
        # Use Monte Carlo sampling for approximate Bayesian inference
        predictions = []
        
        # Run sampling in parallel tasks
        sampling_tasks = []
        for _ in range(self.config.num_samples):
            task = asyncio.create_task(self._sample_posterior(model, inputs, prior))
            sampling_tasks.append(task)
        
        sample_results = await asyncio.gather(*sampling_tasks)
        predictions = np.array(sample_results)
        
        # Calculate mean prediction and uncertainty
        mean_prediction = np.mean(predictions, axis=0)
        
        # Different uncertainty metrics
        std_dev = np.std(predictions, axis=0)
        conf_intervals = self._compute_confidence_intervals(predictions)
        
        # Package uncertainty information
        uncertainty_info = {
            'std_dev': std_dev,
            'conf_intervals': conf_intervals,
            'samples': predictions,
            'variance': np.var(predictions, axis=0),
            'quantiles': self._compute_quantiles(predictions)
        }
        
        return mean_prediction, uncertainty_info
    
    async def _sample_posterior(
        self, 
        model: Any, 
        inputs: np.ndarray,
        prior: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Sample from the posterior distribution."""
        # This would be implemented according to the specific model type
        # For example, different approaches for neural networks vs. Gaussian processes
        
        # For demonstration, we're using a simple approach
        # In practice, this would be more sophisticated
        
        # Add noise to model weights or use dropout for approximate posterior sampling
        return await asyncio.to_thread(self._predict_with_noise, model, inputs)
    
    def _predict_with_noise(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """Make prediction with noise injection for uncertainty sampling."""
        # This is a placeholder - actual implementation depends on model type
        # For neural networks: apply dropout or weight perturbation
        # For Gaussian processes: sample from posterior directly
        # For ensemble methods: sample from ensemble members
        
        # Simplified example
        if hasattr(model, 'predict'):
            predictions = model.predict(inputs)
            # Add noise scaled by model complexity or data uncertainty
            noise_scale = 0.01  # This would be determined by model/data characteristics
            noise = np.random.normal(0, noise_scale, size=predictions.shape)
            return predictions + noise
        else:
            raise TypeError("Model must have a predict method")
    
    def _compute_confidence_intervals(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute confidence intervals from posterior samples."""
        alpha = 1 - self.config.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(samples, lower_percentile, axis=0)
        upper_bound = np.percentile(samples, upper_percentile, axis=0)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence_level': self.config.confidence_level
        }
    
    def _compute_quantiles(self, samples: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various quantiles for detailed uncertainty representation."""
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        result = {}
        
        for q in quantiles:
            result[f'q{int(q*100)}'] = np.percentile(samples, q*100, axis=0)
            
        return result


class EnsembleEstimator:
    """Uncertainty estimation using ensemble methods."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    async def estimate_uncertainty(
        self, 
        models: List[Any], 
        inputs: np.ndarray
    ) -> ProbabilisticForecast:
        """
        Estimate prediction uncertainty using ensemble methods.
        
        Args:
            models: List of ensemble model members
            inputs: Input data for prediction
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.logger.debug(f"Estimating ensemble uncertainty with {len(models)} models")
        
        # Run predictions for each model in parallel
        prediction_tasks = []
        for model in models:
            task = asyncio.create_task(self._predict_with_model(model, inputs))
            prediction_tasks.append(task)
        
        ensemble_predictions = await asyncio.gather(*prediction_tasks)
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate ensemble statistics
        mean_prediction = np.mean(ensemble_predictions, axis=0)
        std_dev = np.std(ensemble_predictions, axis=0)
        
        # Calculate additional uncertainty metrics
        conf_intervals = self._compute_confidence_intervals(ensemble_predictions)
        
        # Package uncertainty information
        uncertainty_info = {
            'std_dev': std_dev,
            'conf_intervals': conf_intervals,
            'ensemble_members': ensemble_predictions,
            'variance': np.var(ensemble_predictions, axis=0),
            'quantiles': self._compute_quantiles(ensemble_predictions)
        }
        
        return mean_prediction, uncertainty_info
    
    async def _predict_with_model(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """Make prediction with a single ensemble member."""
        # Convert to thread to avoid blocking the event loop for CPU-bound operations
        return await asyncio.to_thread(self._predict, model, inputs)
    
    def _predict(self, model: Any, inputs: np.ndarray) -> np.ndarray:
        """Synchronous prediction method."""
        if hasattr(model, 'predict'):
            return model.predict(inputs)
        else:
            raise TypeError("Model must have a predict method")
    
    def _compute_confidence_intervals(self, ensemble_predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute confidence intervals from ensemble predictions."""
        alpha = 1 - self.config.confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(ensemble_predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(ensemble_predictions, upper_percentile, axis=0)
        
        return {
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence_level': self.config.confidence_level
        }
    
    def _compute_quantiles(self, ensemble_predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute various quantiles for detailed uncertainty representation."""
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        result = {}
        
        for q in quantiles:
            result[f'q{int(q*100)}'] = np.percentile(ensemble_predictions, q*100, axis=0)
            
        return result


class ProbabilisticForecaster:
    """Implements probabilistic forecasting with proper scoring rules."""
    
    def __init__(self, config: UncertaintyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.bayesian_estimator = BayesianEstimator(config)
        self.ensemble_estimator = EnsembleEstimator(config)
        
    async def forecast(
        self,
        model: Any,
        inputs: np.ndarray,
        method: UncertaintyMethod = UncertaintyMethod.BAYESIAN,
        forecast_horizon: int = 1,
        prediction_interval: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate probabilistic forecasts for the given inputs.
        
        Args:
            model: Prediction model or ensemble
            inputs: Input data
            method: Uncertainty estimation method
            forecast_horizon: Number of steps to forecast
            prediction_interval: Confidence level for prediction intervals
            
        Returns:
            Dictionary containing forecast distributions and metrics
        """
        self.logger.info(f"Generating probabilistic forecast with {method.value} method")
        
        # Select appropriate uncertainty estimation method
        if method == UncertaintyMethod.BAYESIAN:
            mean, uncertainty = await self.bayesian_estimator.estimate_uncertainty(model, inputs)
        elif method == UncertaintyMethod.ENSEMBLE:
            # Assuming model is a list of models for ensemble method
            if not isinstance(model, list):
                raise TypeError("For ensemble method, model must be a list of models")
            mean, uncertainty = await self.ensemble_estimator.estimate_uncertainty(model, inputs)
        else:
            raise ValueError(f"Unsupported uncertainty method: {method}")
        
        # Calculate prediction intervals
        alpha = 1 - prediction_interval
        lower_bound = mean - scipy.stats.norm.ppf(1 - alpha/2) * uncertainty['std_dev']
        upper_bound = mean + scipy.stats.norm.ppf(1 - alpha/2) * uncertainty['std_dev']
        
        # Package forecast results
        forecast_result = {
            'mean': mean,
            'std_dev': uncertainty['std_dev'],
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'prediction_interval': prediction_interval,
            'uncertainty_metrics': uncertainty,
            'forecast_horizon': forecast_horizon,
            'method': method.value
        }
        
        return forecast_result
    
    def evaluate_forecast(
        self,
        forecasts: Dict[str, Any],
        actual_values: np.ndarray,
        scoring_rule: ScoringRule = ScoringRule.CRPS
    ) -> Dict[str, float]:
        """
        Evaluate probabilistic forecasts using proper scoring rules.
        
        Args:
            forecasts: Probabilistic forecast output
            actual_values: Observed values to compare against
            scoring_rule: Scoring rule to use for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.debug(f"Evaluating forecast with {scoring_rule.value}")
        
        mean = forecasts['mean']
        std_dev = forecasts['std_dev']
        
        # Calculate selected scoring rule
        if scoring_rule == ScoringRule.LOG_SCORE:
            score = self._calculate_log_score(mean, std_dev, actual_values)
        elif scoring_rule == ScoringRule.CRPS:
            score = self._calculate_crps(mean, std_dev, actual_values)
        elif scoring_rule == ScoringRule.BRIER_SCORE:
            # For continuous outcomes, we need to binarize
            # This is a simplified approach for demonstration
            threshold = np.median(actual_values)
            binary_forecast = (mean > threshold).astype(float)
            binary_actual = (actual_values > threshold).astype(float)
            score = self._calculate_brier_score(binary_forecast, binary_actual)
        elif scoring_rule == ScoringRule.INTERVAL_SCORE:
            score = self._calculate_interval_score(
                forecasts['lower_bound'], 
                forecasts['upper_bound'], 
                actual_values, 
                forecasts['prediction_interval']
            )
        else:
            raise ValueError(f"Unsupported scoring rule: {scoring_rule}")
        
        # Calculate additional evaluation metrics
        coverage = self._calculate_coverage(
            forecasts['lower_bound'],
            forecasts['upper_bound'],
            actual_values
        )
        
        sharpness = self._calculate_sharpness(std_dev)
        
        return {
            'score': score,
            'coverage': coverage,
            'sharpness': sharpness,
            'scoring_rule': scoring_rule.value
        }
    
    def _calculate_log_score(
        self,
        mean: np.ndarray,
        std_dev:

