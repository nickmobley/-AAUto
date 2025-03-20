#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import logging
import numpy as np
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Generic, Union, Any, Callable
from collections import deque

# Internal imports
from src.core import AdaptiveSystem, AdaptiveComponent
from src.analysis.market import MarketAnalyzer, MarketRegime
from src.risk.portfolio import PortfolioRiskManager
from src.execution.optimization import ExecutionOptimizer


class StrategyType(Enum):
    """Enumeration of different strategy types."""
    MEAN_REVERSION = "mean_reversion"
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    ARBITRAGE = "arbitrage"
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"
    CUSTOM = "custom"


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy."""
    strategy_id: str
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0
    trades_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_hold_time: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_from_trade(self, pnl: float, win: bool, hold_time: float) -> None:
        """Update performance metrics with a new trade."""
        self.total_pnl += pnl
        self.trades_count += 1
        
        if win:
            self.win_count += 1
            self.avg_win = ((self.avg_win * (self.win_count - 1)) + pnl) / self.win_count
        else:
            self.loss_count += 1
            self.avg_loss = ((self.avg_loss * (self.loss_count - 1)) + pnl) / self.loss_count
            
        self.win_rate = self.win_count / self.trades_count if self.trades_count > 0 else 0.0
        self.avg_hold_time = ((self.avg_hold_time * (self.trades_count - 1)) + hold_time) / self.trades_count
        self.last_updated = datetime.now()
    
    def calculate_metrics(self, returns: List[float], benchmark_returns: List[float]) -> None:
        """Calculate advanced performance metrics from return series."""
        if not returns:
            return
            
        self.volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        
        if self.volatility > 0:
            self.sharpe_ratio = (np.mean(returns) * 252) / self.volatility
        
        # Calculate max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        self.max_drawdown = np.max(drawdown)
        
        # Calculate beta and alpha if benchmark data is available
        if benchmark_returns and len(benchmark_returns) == len(returns):
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_var = np.var(benchmark_returns)
            if benchmark_var > 0:
                self.beta = covariance / benchmark_var
                self.alpha = np.mean(returns) - self.beta * np.mean(benchmark_returns)


class Strategy(ABC, AdaptiveComponent):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 strategy_type: StrategyType = StrategyType.CUSTOM,
                 description: str = "",
                 parameters: Dict[str, Any] = None):
        """Initialize a strategy with basic parameters."""
        super().__init__()
        self.strategy_id = strategy_id or str(uuid.uuid4())
        self.name = name or f"Strategy-{self.strategy_id[:8]}"
        self.strategy_type = strategy_type
        self.description = description
        self.parameters = parameters or {}
        self.performance = StrategyPerformance(strategy_id=self.strategy_id)
        self.enabled = True
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self._signals_history = deque(maxlen=100)
        self.logger = logging.getLogger(f"strategy.{self.name}")
    
    @abstractmethod
    async def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on market data."""
        pass
    
    @abstractmethod
    async def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update strategy parameters."""
        pass
    
    @abstractmethod
    async def adapt(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt strategy based on performance metrics."""
        pass
    
    async def on_trade_completed(self, trade_result: Dict[str, Any]) -> None:
        """Process completed trade results to update performance metrics."""
        pnl = trade_result.get('pnl', 0.0)
        win = pnl > 0
        hold_time = (trade_result.get('close_time', datetime.now()) - 
                     trade_result.get('open_time', datetime.now())).total_seconds() / 3600  # hours
        
        self.performance.update_from_trade(pnl, win, hold_time)
        
        # Log the trade result
        outcome = "PROFIT" if win else "LOSS"
        self.logger.info(f"Trade completed: {outcome} ${pnl:.2f} - {self.name} - {trade_result.get('symbol')}")
        
        # Trigger adaptation if needed
        await self.adapt({'pnl': pnl, 'win': win, 'hold_time': hold_time})
    
    def disable(self) -> None:
        """Disable the strategy."""
        self.enabled = False
        self.logger.info(f"Strategy {self.name} disabled")
    
    def enable(self) -> None:
        """Enable the strategy."""
        self.enabled = True
        self.logger.info(f"Strategy {self.name} enabled")
    
    def __str__(self) -> str:
        return f"{self.name} ({self.strategy_type.value})"


class OnlineLearningStrategy(Strategy):
    """Strategy that implements online learning to adapt in real-time."""
    
    def __init__(self, 
                 strategy_id: str = None, 
                 name: str = None,
                 strategy_type: StrategyType = StrategyType.MACHINE_LEARNING,
                 description: str = "Online learning strategy",
                 parameters: Dict[str, Any] = None,
                 learning_rate: float = 0.01,
                 exploration_rate: float = 0.1,
                 feature_names: List[str] = None):
        """Initialize an online learning strategy."""
        parameters = parameters or {}
        parameters.update({
            'learning_rate': learning_rate,
            'exploration_rate': exploration_rate
        })
        super().__init__(strategy_id, name, strategy_type, description, parameters)
        
        self.feature_names = feature_names or []
        self.weights = np.random.normal(0, 0.1, len(self.feature_names)) if self.feature_names else np.array([])
        self.recent_updates = deque(maxlen=50)  # Track recent weight updates
        
    async def generate_signal(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on market data using current weights."""
        if not self.feature_names or not self.enabled:
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Strategy not ready or disabled'}
        
        # Extract features from market data
        features = self._extract_features(market_data)
        if len(features) != len(self.weights):
            self.logger.warning(f"Feature dimension mismatch: {len(features)} vs weights {len(self.weights)}")
            return {'action': 'HOLD', 'confidence': 0.0, 'reason': 'Feature dimension mismatch'}
        
        # Calculate prediction score
        score = np.dot(features, self.weights)
        
        # Determine action based on score
        action, confidence = self._score_to_action(score)
        
        # Random exploration with probability exploration_rate
        if np.random.random() < self.parameters['exploration_rate']:
            actions = ['BUY', 'SELL', 'HOLD']
            exploration_action = np.random.choice(actions)
            if exploration_action != action:
                self.logger.debug(f"Exploration override: {action} -> {exploration_action}")
                action = exploration_action
                confidence *= 0.5  # Reduce confidence during exploration
        
        signal = {
            'action': action,
            'confidence': confidence,
            'reason': f"Online model prediction: {score:.4f}",
            'features': dict(zip(self.feature_names, features)),
            'timestamp': datetime.now()
        }
        
        self._signals_history.append(signal)
        return signal
    
    async def update_parameters(self, new_parameters: Dict[str, Any]) -> None:
        """Update strategy parameters, merging with existing ones."""
        self.parameters.update(new_parameters)
        self.logger.info(f"Updated parameters: {new_parameters}")
        self.last_updated = datetime.now()
    
    async def adapt(self, performance_metrics: Dict[str, float]) -> None:
        """Adapt strategy weights based on performance metrics."""
        if not self.feature_names or len(self._signals_history) == 0:
            return
            
        # Get the last signal that led to this performance
        last_signal = self._signals_history[-1]
        features = np.array([last_signal['features'].get(f, 0.0) for f in self.feature_names])
        
        # Calculate the reward signal
        pnl = performance_metrics.get('pnl', 0.0)
        reward = np.sign(pnl) * (abs(pnl) ** 0.5)  # Using square root to dampen large values
        
        # Update weights with reward signal and learning rate
        learning_rate = self.parameters['learning_rate']
        direction = 1 if last_signal['action'] == 'BUY' else -1 if last_signal['action'] == 'SELL' else 0
        
        if direction != 0:
            # Only update weights if we took a directional position
            update = learning_rate * reward * direction * features
            self.weights += update
            
            # Record the update
            self.recent_updates.append({
                'timestamp': datetime.now(),
                'update': update.tolist(),
                'reward': reward,
                'pnl': pnl
            })
            
            # Log significant updates
            if abs(reward) > 1.0:
                self.logger.info(f"Significant weight update: reward={reward:.2f}, pnl=${pnl:.2f}")
        
        # Adjust exploration rate based on performance volatility
        if len(self.recent_updates) > 10:
            recent_rewards = [u['reward'] for u in self.recent_updates]
            reward_volatility = np.std(recent_rewards)
            
            # Increase exploration when volatility is high, decrease when low
            self.parameters['exploration_rate'] = min(0.3, max(0.01, 0.1 * reward_volatility))
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from market data."""
        return np.array([market_data.get(feature, 0.0) for feature in self.feature_names])
    
    def _score_to_action(self, score: float) -> Tuple[str, float]:
        """Convert a prediction score to an action and confidence."""
        confidence = min(1.0, abs(score) / 2.0)  # Scale confidence
        
        if score > 0.5:
            return 'BUY', confidence
        elif score < -0.5:
            return 'SELL', confidence
        else:
            return 'HOLD', confidence


class StrategyPortfolio(AdaptiveComponent):
    """Manages a portfolio of strategies with reinforcement learning for selection."""
    
    def __init__(self, 
                 market_analyzer: Optional[MarketAnalyzer] = None,
                 risk_manager: Optional[PortfolioRiskManager] = None,
                 execution_optimizer: Optional[ExecutionOptimizer] = None):
        """Initialize the strategy portfolio manager."""
        super().__init__()
        self.strategies: Dict[str, Strategy] = {}
        self.active_strategies: Set[str] = set()
        self.performance_history: Dict[str, List[StrategyPerformance]] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.last_selection_time = datetime.now()
        self.selection_interval = 3600  # seconds between strategy re-weighting
        
        # References to other system components
        self.market_analyzer = market_analyzer
        self.risk_manager = risk_manager
        self.execution_optimizer = execution_optimizer
        
        # Reinforcement learning parameters
        self.learning_rate = 0.05
        self.reward_decay = 0.9  # Discount factor for future rewards
        self.state_history = deque(maxlen=100)
        self.logger = logging.getLogger("strategy.portfolio")
        
    async def register_strategy(self, strategy: Strategy) -> None:
        """Register a new strategy with the portfolio."""
        self.strategies[strategy.strategy_id] = strategy
        self.performance_history[strategy.strategy_id] = []
        
        # Initialize with equal weights
        total_strategies = len(self.strategies)
        weight = 1.0 / total_strategies if total_strategies > 0 else 0.0
        self.strategy_weights = {sid: weight for sid in self.strategies}
        
        self.logger.info(f"Registered strategy: {strategy.name} (ID: {strategy.strategy_id})")
    
    async def unregister_strategy(self, strategy_id: str) -> None:
        """Unregister a strategy from the portfolio."""
        if strategy

