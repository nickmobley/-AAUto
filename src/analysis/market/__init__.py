from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray

# Assuming these modules exist in the project structure
from src.core import AdaptiveSystem
from src.core.events import Event, EventType
from src.data.storage import DataStore


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    MEAN_REVERTING = auto()
    VOLATILE = auto()
    CHOPPY = auto()
    UNKNOWN = auto()


@dataclass
class MarketFeature:
    """A market feature with associated importance and adaptability metrics."""
    name: str
    value: float
    importance: float = 0.0
    adaptation_rate: float = 0.01
    history: List[float] = field(default_factory=list)
    
    def update(self, new_value: float, performance_impact: float) -> None:
        """Update feature value and adjust importance based on performance impact."""
        self.history.append(self.value)
        self.value = new_value
        # Adjust importance based on how much this feature affected performance
        self.importance += self.adaptation_rate * performance_impact
        # Ensure importance stays in reasonable bounds
        self.importance = max(0.0, min(1.0, self.importance))


class MarketAnalyzer:
    """
    Self-improving market analysis system that detects market regimes,
    adapts thresholds based on performance, and ranks feature importance.
    """
    
    def __init__(
        self, 
        adaptive_system: AdaptiveSystem,
        data_store: DataStore,
        initial_features: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.01,
        history_window: int = 100,
        confidence_threshold: float = 0.65
    ):
        """
        Initialize the MarketAnalyzer with connections to the adaptive system.
        
        Args:
            adaptive_system: Core adaptive system to connect with
            data_store: Data storage system for market data
            initial_features: Initial feature set with starting values
            learning_rate: Rate at which parameters adapt
            history_window: Number of historical data points to consider
            confidence_threshold: Minimum confidence for regime detection
        """
        self.adaptive_system = adaptive_system
        self.data_store = data_store
        self.logger = logging.getLogger(__name__)
        
        # Register with the adaptive system for performance feedback
        self.adaptive_system.register_component(
            "market_analyzer", 
            self.receive_feedback
        )
        
        # Feature management
        self.features: Dict[str, MarketFeature] = {}
        if initial_features:
            for name, value in initial_features.items():
                self.features[name] = MarketFeature(name=name, value=value)
        
        # Adaptive parameters
        self.learning_rate = learning_rate
        self.history_window = history_window
        self.confidence_threshold = confidence_threshold
        
        # Performance tracking
        self.detection_accuracy: List[float] = []
        self.regime_history: List[Tuple[MarketRegime, float]] = []
        
        # Adaptive thresholds for regime detection
        self.thresholds = {
            "trend_strength": 0.5,
            "volatility": 0.3,
            "mean_reversion": 0.4,
            "momentum": 0.6,
        }
        
        self.logger.info("MarketAnalyzer initialized with adaptive capabilities")
    
    async def analyze(self, market_data: Dict[str, Any]) -> MarketRegime:
        """
        Analyze market data to detect the current market regime.
        
        Args:
            market_data: Dictionary containing market data
            
        Returns:
            Detected market regime
        """
        self.logger.debug(f"Analyzing market data: {len(market_data)} features")
        
        # Extract relevant metrics from market data
        try:
            # Calculate key indicators with weighted importance
            trend_strength = await self._calculate_trend_strength(market_data)
            volatility = await self._calculate_volatility(market_data)
            mean_reversion = await self._calculate_mean_reversion(market_data)
            momentum = await self._calculate_momentum(market_data)
            
            # Determine market regime with adaptive thresholds
            regime, confidence = await self._detect_regime(
                trend_strength, volatility, mean_reversion, momentum
            )
            
            # Store result for later adaptation
            self.regime_history.append((regime, confidence))
            if len(self.regime_history) > self.history_window:
                self.regime_history.pop(0)
                
            # Emit event for other system components
            await self.adaptive_system.emit_event(
                Event(
                    type=EventType.MARKET_REGIME_DETECTED,
                    data={
                        "regime": regime,
                        "confidence": confidence,
                        "features": {
                            "trend_strength": trend_strength,
                            "volatility": volatility,
                            "mean_reversion": mean_reversion,
                            "momentum": momentum
                        }
                    }
                )
            )
            
            return regime
            
        except Exception as e:
            self.logger.error(f"Error in market regime analysis: {e}")
            return MarketRegime.UNKNOWN
    
    async def _calculate_trend_strength(self, market_data: Dict[str, Any]) -> float:
        """Calculate trend strength with weighted feature importance."""
        # Simplified implementation - in practice would use actual trend calculation
        prices = market_data.get("prices", [])
        if not prices or len(prices) < 2:
            return 0.0
            
        # Basic trend calculation
        price_changes = np.diff(prices)
        positive_changes = np.sum(price_changes > 0)
        negative_changes = np.sum(price_changes < 0)
        
        if positive_changes + negative_changes == 0:
            return 0.0
            
        directional_strength = abs(positive_changes - negative_changes) / (positive_changes + negative_changes)
        
        # Weight by feature importance
        trend_feature = self.features.get("trend_direction", MarketFeature(name="trend_direction", value=directional_strength))
        self.features["trend_direction"] = trend_feature
        
        return directional_strength * (0.5 + 0.5 * trend_feature.importance)
    
    async def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """Calculate market volatility with adaptive thresholds."""
        prices = market_data.get("prices", [])
        if not prices or len(prices) < 2:
            return 0.0
            
        # Calculate volatility as normalized standard deviation
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Normalize with adaptive scaling
        vol_feature = self.features.get("volatility", MarketFeature(name="volatility", value=volatility))
        self.features["volatility"] = vol_feature
        
        return min(1.0, volatility / (0.01 + 0.05 * vol_feature.importance))
    
    async def _calculate_mean_reversion(self, market_data: Dict[str, Any]) -> float:
        """Calculate mean reversion tendency."""
        prices = market_data.get("prices", [])
        if not prices or len(prices) < 10:
            return 0.0
            
        # Calculate autocorrelation as mean reversion indicator
        returns = np.diff(prices) / prices[:-1]
        if len(returns) < 2:
            return 0.0
            
        # Negative autocorrelation suggests mean reversion
        autocorr = np.correlate(returns[:-1], returns[1:], mode='valid')[0]
        mean_reversion_strength = max(0.0, -autocorr)
        
        mr_feature = self.features.get("mean_reversion", MarketFeature(name="mean_reversion", value=mean_reversion_strength))
        self.features["mean_reversion"] = mr_feature
        
        return mean_reversion_strength * (0.5 + 0.5 * mr_feature.importance)
    
    async def _calculate_momentum(self, market_data: Dict[str, Any]) -> float:
        """Calculate price momentum with adaptive lookback."""
        prices = market_data.get("prices", [])
        if not prices or len(prices) < 10:
            return 0.0
            
        # Calculate momentum as rate of change
        momentum_feature = self.features.get("momentum", MarketFeature(name="momentum", value=0.0))
        
        # Adapt lookback period based on feature importance
        lookback = max(5, int(20 * (1 - momentum_feature.importance) + 5))
        if len(prices) > lookback:
            momentum = (prices[-1] / prices[-lookback] - 1.0)
            normalized_momentum = min(1.0, max(0.0, abs(momentum) / 0.1))
            momentum_feature.value = normalized_momentum
            
        self.features["momentum"] = momentum_feature
        return momentum_feature.value
    
    async def _detect_regime(
        self, 
        trend_strength: float, 
        volatility: float, 
        mean_reversion: float, 
        momentum: float
    ) -> Tuple[MarketRegime, float]:
        """
        Detect market regime based on calculated metrics using adaptive thresholds.
        
        Returns:
            Tuple of (detected regime, confidence level)
        """
        # Apply adaptive thresholds
        regime_scores = {
            MarketRegime.TRENDING_UP: (
                trend_strength if momentum > 0 else 0.0
            ) * (1.0 - mean_reversion * 0.5),
            
            MarketRegime.TRENDING_DOWN: (
                trend_strength if momentum < 0 else 0.0
            ) * (1.0 - mean_reversion * 0.5),
            
            MarketRegime.MEAN_REVERTING: mean_reversion * (1.0 - trend_strength * 0.7),
            
            MarketRegime.VOLATILE: volatility * (1.0 - mean_reversion * 0.3),
            
            MarketRegime.CHOPPY: (1.0 - trend_strength) * (1.0 - mean_reversion) * volatility * 0.5,
        }
        
        # Find regime with highest score
        best_regime = max(regime_scores.items(), key=lambda x: x[1])
        
        # Calculate confidence based on difference from second best
        scores = sorted(regime_scores.values(), reverse=True)
        confidence = scores[0] if len(scores) == 1 else scores[0] - scores[1]
        
        # Apply adaptive confidence threshold
        if confidence < self.confidence_threshold:
            return MarketRegime.UNKNOWN, confidence
            
        return best_regime[0], confidence
    
    async def receive_feedback(self, performance_data: Dict[str, Any]) -> None:
        """
        Receive performance feedback from the adaptive system and adjust parameters.
        
        Args:
            performance_data: Dictionary containing performance metrics
        """
        self.logger.debug(f"Received performance feedback: {performance_data}")
        
        # Extract relevant metrics
        prediction_accuracy = performance_data.get("prediction_accuracy", 0.0)
        pnl_impact = performance_data.get("pnl_impact", 0.0)
        
        # Track accuracy for self-improvement
        self.detection_accuracy.append(prediction_accuracy)
        if len(self.detection_accuracy) > self.history_window:
            self.detection_accuracy.pop(0)
        
        # Adjust learning rate based on recent performance
        if len(self.detection_accuracy) > 10:
            recent_trend = np.mean(self.detection_accuracy[-10:]) - np.mean(self.detection_accuracy[-20:-10])
            # If improving, accelerate learning; if degrading, slow down
            self.learning_rate = max(0.001, min(0.1, self.learning_rate * (1.0 + 0.1 * np.sign(recent_trend))))
        
        # Adapt thresholds based on performance
        await self._adapt_thresholds(prediction_accuracy, pnl_impact)
        
        # Update feature importance based on performance impact
        feature_impacts = performance_data.get("feature_impacts", {})
        for feature_name, impact in feature_impacts.items():
            if feature_name in self.features:
                self.features[feature_name].update(
                    self.features[feature_name].value, 
                    impact
                )
        
        # Log adaptation progress
        self.logger.info(f"Adapted thresholds, current learning rate: {self.learning_rate:.4f}")
        
    async def _adapt_thresholds(self, accuracy: float, pnl_impact: float) -> None:
        """Adapt thresholds based on performance metrics."""
        # Don't adapt if accuracy is too low (could be unstable)
        if accuracy < 0.3:
            return
            
        # Calculate adaptation direction and strength
        adaptation_strength = self.learning_rate * pnl_impact * (accuracy - 0.5) * 2.0
        
        # Adapt each threshold
        for threshold_name, value in self.thresholds.items():
            # More complex logic could be used here to selectively adapt thresholds
            # based on which regimes were detected and their performance
            new_value = value * (1.0 + adaptation_strength)
            # Keep thresholds in reasonable bounds
            self.thresholds[threshold_name] = max(0.1, min(0.9, new_value))
    
    async def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """
        Get features ranked by their importance.
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        return sorted(
            [(name, feat.importance) for name, feat in self.features.items()],
            key=lambda x: x[1],
            reverse=True
        )
    
    async def save_state(self) -> Dict[str, Any]:
        """Save the current state for persistence."""
        return {
            "features": {name: {"value": f.value, "importance": f.importance} 
                        for name, f in self.features.items()},
            "thresholds": self.thresholds,
            "learning_rate": self.learning_rate,
            "confidence_threshold": self.confidence_threshold,
        }
    
    async def load_state(self, state: Dict[str, Any]) -> None:
        """Load a previously saved state."""
        if "features" in state:
            for name, data in state["features"].items():
                if name in self.features:
                    self.features[name].value = data["value"]
                    self.features[name].importance = data["importance"]
                else:
                    self.features[name] = MarketFeature(
                        name=name, 
                        value=data["value"],
                        importance=data["importance"]
                    )
        
        if "thresholds" in state:
            self.thresholds.update

