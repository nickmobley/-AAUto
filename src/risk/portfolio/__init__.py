#!/usr/bin/env python3
"""
Adaptive Portfolio Risk Management System

This module provides a comprehensive risk management system that dynamically
adjusts position sizing, risk limits, and portfolio allocations based on
market conditions and performance feedback.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union, cast
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from collections import deque

# Type variables for generic functions
T = TypeVar('T')
PositionId = str
AssetId = str
PositionSize = float
RiskScore = float
MarketRegime = Enum('MarketRegime', ['LOW_VOLATILITY', 'NORMAL', 'HIGH_VOLATILITY', 'CRISIS'])


class MarketRegimeDetector:
    """Detects current market regime based on volatility and other market factors."""
    
    def __init__(self, lookback_period: int = 30, volatility_threshold_low: float = 0.1,
                 volatility_threshold_high: float = 0.2, crisis_threshold: float = 0.3):
        self.lookback_period = lookback_period
        self.volatility_threshold_low = volatility_threshold_low
        self.volatility_threshold_high = volatility_threshold_high
        self.crisis_threshold = crisis_threshold
        self.volatility_history: deque = deque(maxlen=lookback_period)
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []
        
    async def update(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Update market regime detection based on new market data.
        
        Args:
            market_data: DataFrame containing market prices and indicators
            
        Returns:
            Current market regime
        """
        # Calculate volatility (e.g., 5-day rolling standard deviation of returns)
        returns = market_data['close'].pct_change().dropna()
        volatility = returns.rolling(5).std().iloc[-1] if len(returns) >= 5 else 0
        
        self.volatility_history.append(volatility)
        
        # Determine current regime
        if volatility < self.volatility_threshold_low:
            regime = MarketRegime.LOW_VOLATILITY
        elif volatility < self.volatility_threshold_high:
            regime = MarketRegime.NORMAL
        elif volatility < self.crisis_threshold:
            regime = MarketRegime.HIGH_VOLATILITY
        else:
            regime = MarketRegime.CRISIS
        
        # Record the regime change
        self.regime_history.append((datetime.now(), regime))
        
        return regime
    
    def adaptive_update(self, performance_metrics: Dict[str, float]) -> None:
        """
        Adapt regime thresholds based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Example: Adjust thresholds based on Sharpe ratio
        if 'sharpe_ratio' in performance_metrics:
            sharpe = performance_metrics['sharpe_ratio']
            if sharpe < 0.5:  # Poor performance
                # Make thresholds more conservative
                self.volatility_threshold_low *= 0.95
                self.volatility_threshold_high *= 0.95
            elif sharpe > 2.0:  # Excellent performance
                # Allow more risk
                self.volatility_threshold_low *= 1.05
                self.volatility_threshold_high *= 1.05


@dataclass
class RiskParameters:
    """Parameters for risk management that can be dynamically adjusted."""
    
    max_portfolio_var: float = 0.015  # Maximum portfolio variance
    max_position_size: float = 0.25   # Maximum position size as fraction of portfolio
    risk_free_rate: float = 0.03      # Current risk-free rate
    max_drawdown_limit: float = 0.15  # Maximum acceptable drawdown
    correlation_threshold: float = 0.7  # Correlation threshold for diversification
    stop_loss_std_multiplier: float = 2.0  # Stop loss as multiple of std dev
    
    # Parameters that change per market regime
    regime_adjustments: Dict[MarketRegime, Dict[str, float]] = field(default_factory=lambda: {
        MarketRegime.LOW_VOLATILITY: {
            'max_position_size': 0.25,
            'stop_loss_std_multiplier': 2.5
        },
        MarketRegime.NORMAL: {
            'max_position_size': 0.20,
            'stop_loss_std_multiplier': 2.0
        },
        MarketRegime.HIGH_VOLATILITY: {
            'max_position_size': 0.15,
            'stop_loss_std_multiplier': 1.5
        },
        MarketRegime.CRISIS: {
            'max_position_size': 0.10,
            'stop_loss_std_multiplier': 1.0
        }
    })
    
    def apply_regime_adjustment(self, regime: MarketRegime) -> None:
        """
        Adjust risk parameters based on current market regime.
        
        Args:
            regime: The current market regime
        """
        if regime in self.regime_adjustments:
            adjustments = self.regime_adjustments[regime]
            for param, value in adjustments.items():
                if hasattr(self, param):
                    setattr(self, param, value)


class PositionSizer:
    """Dynamically sizes positions based on asset volatility and correlation."""
    
    def __init__(self, risk_params: RiskParameters):
        self.risk_params = risk_params
        self.volatility_lookback: int = 20
        self.position_history: Dict[AssetId, List[Tuple[datetime, PositionSize]]] = {}
        
    async def calculate_position_size(
        self, 
        asset_id: AssetId, 
        portfolio_value: float,
        asset_volatility: float,
        signal_strength: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> PositionSize:
        """
        Calculate optimal position size based on Kelly Criterion and other factors.
        
        Args:
            asset_id: Identifier for the asset
            portfolio_value: Current portfolio value
            asset_volatility: Asset volatility (standard deviation)
            signal_strength: Signal strength from -1.0 to 1.0
            correlation_matrix: Correlation matrix for all portfolio assets
            
        Returns:
            Optimal position size in currency units
        """
        # Base Kelly position size
        win_probability = (signal_strength + 1) / 2  # Convert [-1, 1] to [0, 1]
        win_loss_ratio = 1.5  # Example, could be derived from historical data
        
        # Kelly formula: f* = p - (1-p)/r where p=win probability, r=win/loss ratio
        kelly_fraction = win_probability - (1 - win_probability) / win_loss_ratio
        
        # Limit position size based on volatility
        volatility_adjustment = 1.0 / (1.0 + asset_volatility * 10)
        
        # Correlation adjustment if correlation matrix is provided
        correlation_adjustment = 1.0
        if correlation_matrix is not None and asset_id in correlation_matrix.index:
            # Get average correlation with existing positions
            asset_correlations = correlation_matrix.loc[asset_id].drop(asset_id)
            if not asset_correlations.empty:
                avg_correlation = asset_correlations.abs().mean()
                # Reduce position size for highly correlated assets
                if avg_correlation > self.risk_params.correlation_threshold:
                    correlation_adjustment = (1.0 - avg_correlation) * 2  # Linear scaling
        
        # Calculate final position size with all adjustments
        raw_position_size = kelly_fraction * self.risk_params.max_position_size * volatility_adjustment * correlation_adjustment
        
        # Ensure position size isn't negative and doesn't exceed maximum
        position_size = max(0.0, min(raw_position_size, self.risk_params.max_position_size))
        
        # Convert to currency units
        position_value = position_size * portfolio_value
        
        # Record the position size history
        if asset_id not in self.position_history:
            self.position_history[asset_id] = []
        self.position_history[asset_id].append((datetime.now(), position_size))
        
        return position_value
    
    def adaptive_update(self, performance_metrics: Dict[str, float]) -> None:
        """
        Adapt position sizing parameters based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Example: If Sharpe ratio is low, reduce max position size
        if 'sharpe_ratio' in performance_metrics:
            sharpe = performance_metrics['sharpe_ratio']
            if sharpe < 1.0:
                self.risk_params.max_position_size *= 0.95  # Reduce by 5%
            elif sharpe > 2.0:
                self.risk_params.max_position_size = min(
                    self.risk_params.max_position_size * 1.05,  # Increase by 5%
                    0.25  # Hard cap at 25%
                )


class PortfolioOptimizer:
    """Optimizes portfolio allocations based on modern portfolio theory and risk metrics."""
    
    def __init__(self, risk_params: RiskParameters):
        self.risk_params = risk_params
        self.optimization_history: List[Tuple[datetime, Dict[str, float]]] = []
        
    async def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        current_positions: Dict[AssetId, float]
    ) -> Dict[AssetId, float]:
        """
        Perform mean-variance optimization to find optimal portfolio weights.
        
        Args:
            expected_returns: Series of expected returns for each asset
            covariance_matrix: Covariance matrix of asset returns
            current_positions: Current position sizes
            
        Returns:
            Dictionary of optimal position weights
        """
        n_assets = len(expected_returns)
        if n_assets == 0:
            return {}
        
        # Efficient Frontier optimization
        try:
            # Using quadratic programming to solve for optimal weights
            # This is a simplified version - in practice you'd use libraries like PyPortfolioOpt
            weights = await self._calculate_optimal_weights(expected_returns, covariance_matrix)
            
            # Apply position limits and other constraints
            weights = {
                asset: min(weight, self.risk_params.max_position_size)
                for asset, weight in weights.items()
            }
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {asset: weight / total_weight for asset, weight in weights.items()}
            
            # Record optimization history
            self.optimization_history.append((datetime.now(), weights))
            
            return weights
            
        except Exception as e:
            logging.error(f"Portfolio optimization failed: {e}")
            # Return current positions normalized if optimization fails
            total = sum(current_positions.values())
            if total > 0:
                return {k: v / total for k, v in current_positions.items()}
            return {k: 1.0 / len(current_positions) for k in current_positions} if current_positions else {}
    
    async def _calculate_optimal_weights(
        self, 
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ) -> Dict[AssetId, float]:
        """
        Calculate optimal weights using the Sharpe ratio maximization approach.
        
        This is a simplified implementation using NumPy. In production, use specialized libraries.
        
        Args:
            expected_returns: Series of expected returns
            covariance_matrix: Covariance matrix
            
        Returns:
            Dictionary of optimal weights by asset
        """
        # Convert to numpy arrays for calculations
        mu = expected_returns.values
        sigma = covariance_matrix.values
        
        # Number of assets
        n = len(mu)
        
        # Create constraints matrices for optimization
        # This is a simplified version - in practice you'd use cvxpy or similar
        
        # For demonstration, we'll use a simple inverse volatility weighting
        vols = np.sqrt(np.diag(sigma))
        inv_vols = 1.0 / vols
        weights = inv_vols / np.sum(inv_vols)
        
        # Convert back to dictionary
        return {expected_returns.index[i]: weights[i] for i in range(n)}
    
    def adaptive_update(self, performance_metrics: Dict[str, float]) -> None:
        """
        Adapt optimization parameters based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        # Example: Adjust correlation threshold based on portfolio diversification score
        if 'diversification_score' in performance_metrics:
            div_score = performance_metrics['diversification_score']
            if div_score < 0.3:  # Poor diversification
                self.risk_params.correlation_threshold *= 0.9  # Reduce threshold to enforce diversification
            elif div_score > 0.7:  # Good diversification
                self.risk_params.correlation_threshold = min(
                    self.risk_params.correlation_threshold * 1.1,
                    0.8  # Cap at 0.8
                )


class AdaptiveRiskManager:
    """
    Comprehensive risk management system that adapts to market conditions 
    and performance feedback.
    """
    
    def __init__(self):
        self.risk_params = RiskParameters()
        self.regime_detector = MarketRegimeDetector()
        self.position_sizer = PositionSizer(self.risk_params)
        self.portfolio_optimizer = PortfolioOptimizer(self.risk_params)
        self.current_regime = MarketRegime.NORMAL
        self.performance_history: Dict[str, List[float]] = {
            'sharpe_ratio': [],
            'sortino_ratio': [],
            'max_drawdown': [],
            'diversification_score': []
        }
        self.last_optimization: Optional[datetime] = None
        self.optimization_frequency = 24  # Hours
    
    async def update_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Update the current market regime assessment.
        
        Args:
            market_data: DataFrame with market price data
            
        Returns:
            Current market regime
        """
        self.current_regime = await self.regime_detector.update(market_data)
        self.risk_params.apply_regime_adjustment(self.current_regime)
        return self.current_regime
    
    async def calculate_position_size(
        self,
        asset_id: AssetId,
        portfolio_value: float,
        market_data: pd.DataFrame,
        signal_strength: float,
        correlation_matrix: Optional[pd.DataFrame] = None
    ) -> PositionSize:
        """
        Calculate optimal position size for a given asset.
        
        Args:
            asset_id: Asset identifier
            portfolio_value: Current portfolio value
            market_data: Market data for

