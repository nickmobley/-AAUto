"""
Strategy Manager module for loading, combining, and orchestrating trading strategies.

This module provides a central manager for all trading strategies, offering
functionality to:
1. Load and instantiate strategies
2. Combine signals from multiple strategies
3. Manage strategy weights and priorities
4. Provide a unified interface to the trading system
"""

import importlib
import logging
import os
from typing import Dict, List, Optional, Type, Union, Any, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from src.strategies.base import Strategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy

# Setup logger
logger = logging.getLogger(__name__)


class SignalCombinationMethod(Enum):
    """Enumeration of methods to combine signals from multiple strategies."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAJORITY_VOTE = "majority_vote"
    HIGHEST_CONFIDENCE = "highest_confidence"
    UNANIMOUS = "unanimous"
    CUSTOM = "custom"


class StrategyManager:
    """
    Manager class for handling strategy instantiation, combination, and orchestration.
    
    This class serves as the central point for managing all trading strategies,
    providing a unified interface to the trading system and handling the complexity
    of strategy coordination.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the StrategyManager.
        
        Args:
            config: Dictionary containing configuration for all strategies
                   and the manager itself.
        
        Raises:
            ValueError: If the configuration is invalid.
        """
        self.config = config
        self.strategies: Dict[str, Strategy] = {}
        self.strategy_weights: Dict[str, float] = {}
        self.combination_method = SignalCombinationMethod.WEIGHTED_AVERAGE
        
        # Extract manager configuration
        manager_config = config.get("strategy_manager", {})
        self.combination_method = SignalCombinationMethod(
            manager_config.get("combination_method", "weighted_average")
        )
        
        # Initialize strategies
        try:
            self._initialize_strategies(config.get("strategies", {}))
            logger.info(f"StrategyManager initialized with {len(self.strategies)} strategies")
        except Exception as e:
            logger.error(f"Failed to initialize StrategyManager: {str(e)}")
            raise ValueError(f"Strategy manager initialization failed: {str(e)}")

    def _initialize_strategies(self, strategies_config: Dict[str, Dict]) -> None:
        """
        Initialize all strategies based on the configuration.
        
        Args:
            strategies_config: Dictionary mapping strategy names to their configurations.
        
        Raises:
            ValueError: If a strategy fails to initialize.
        """
        # Dictionary mapping strategy type to class
        strategy_classes = {
            "momentum": MomentumStrategy,
            "mean_reversion": MeanReversionStrategy,
            "trend_following": TrendFollowingStrategy
        }
        
        # Load and initialize each strategy
        for strategy_id, strategy_config in strategies_config.items():
            try:
                strategy_type = strategy_config.get("type", "").lower()
                
                # Check if strategy type exists
                if strategy_type not in strategy_classes:
                    logger.warning(f"Unknown strategy type: {strategy_type}. Skipping.")
                    continue
                
                # Get strategy class and initialize
                strategy_class = strategy_classes[strategy_type]
                strategy = strategy_class(strategy_config)
                
                # Store strategy and its weight
                self.strategies[strategy_id] = strategy
                self.strategy_weights[strategy_id] = strategy_config.get("weight", 1.0)
                
                logger.info(f"Initialized strategy: {strategy_id} ({strategy_type})")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy_id}: {str(e)}")
                raise ValueError(f"Strategy initialization failed for {strategy_id}: {str(e)}")
    
    def load_custom_strategy(self, strategy_path: str, strategy_id: str, 
                            strategy_config: Dict[str, Any]) -> bool:
        """
        Load a custom strategy from a Python module.
        
        Args:
            strategy_path: Path to the strategy module (dot notation)
            strategy_id: Unique identifier for the strategy
            strategy_config: Configuration for the strategy
            
        Returns:
            bool: True if the strategy was loaded successfully, False otherwise
            
        Raises:
            ImportError: If the module cannot be imported
        """
        try:
            # Import the module dynamically
            module_parts = strategy_path.split('.')
            module_name = '.'.join(module_parts[:-1])
            class_name = module_parts[-1]
            
            module = importlib.import_module(module_name)
            strategy_class = getattr(module, class_name)
            
            # Validate that it's a Strategy subclass
            if not issubclass(strategy_class, Strategy):
                logger.error(f"Class {class_name} is not a subclass of Strategy")
                return False
            
            # Initialize and store the strategy
            strategy = strategy_class(strategy_config)
            self.strategies[strategy_id] = strategy
            self.strategy_weights[strategy_id] = strategy_config.get("weight", 1.0)
            
            logger.info(f"Successfully loaded custom strategy: {strategy_id}")
            return True
            
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"Failed to load custom strategy {strategy_path}: {str(e)}")
            return False
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """
        Get a strategy by its ID.
        
        Args:
            strategy_id: The unique identifier of the strategy.
            
        Returns:
            The strategy object if found, None otherwise.
        """
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return None
        
        return self.strategies[strategy_id]
    
    def update_strategy_weight(self, strategy_id: str, weight: float) -> bool:
        """
        Update the weight of a strategy for signal combination.
        
        Args:
            strategy_id: The unique identifier of the strategy.
            weight: The new weight to assign (0.0 to 1.0).
            
        Returns:
            bool: True if the weight was updated, False otherwise.
            
        Raises:
            ValueError: If the weight is not in the valid range [0.0, 1.0].
        """
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"Strategy weight must be between 0.0 and 1.0, got {weight}")
            
        if strategy_id not in self.strategies:
            logger.warning(f"Cannot update weight: Strategy {strategy_id} not found")
            return False
        
        self.strategy_weights[strategy_id] = weight
        logger.info(f"Updated weight for strategy {strategy_id} to {weight}")
        return True
    
    def set_combination_method(self, method: Union[str, SignalCombinationMethod]) -> None:
        """
        Set the method for combining signals from multiple strategies.
        
        Args:
            method: The combination method to use.
            
        Raises:
            ValueError: If the method is not valid.
        """
        if isinstance(method, str):
            try:
                method = SignalCombinationMethod(method)
            except ValueError:
                valid_methods = [m.value for m in SignalCombinationMethod]
                raise ValueError(f"Invalid combination method: {method}. "
                                f"Valid methods are: {', '.join(valid_methods)}")
        
        self.combination_method = method
        logger.info(f"Set signal combination method to {method.value}")
    
    def get_combined_signals(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Get combined trading signals from all active strategies.
        
        Args:
            market_data: DataFrame containing market data.
            
        Returns:
            DataFrame with combined signals.
            
        Raises:
            ValueError: If no strategies are available or if the market data is invalid.
        """
        if not self.strategies:
            raise ValueError("No strategies available to generate signals")
        
        if market_data is None or market_data.empty:
            raise ValueError("Invalid market data provided for signal generation")
        
        try:
            # Collect signals from all strategies
            strategy_signals = {}
            for strategy_id, strategy in self.strategies.items():
                if self.strategy_weights.get(strategy_id, 0) > 0:
                    signals = strategy.generate_signals(market_data)
                    strategy_signals[strategy_id] = signals
            
            # Check if we have any signals
            if not strategy_signals:
                logger.warning("No signals generated by any strategy")
                return pd.DataFrame()
            
            # Combine signals based on the selected method
            combined_signals = self._combine_signals(strategy_signals, market_data)
            logger.info(f"Generated combined signals using {self.combination_method.value} method")
            
            return combined_signals
        
        except Exception as e:
            logger.error(f"Error generating combined signals: {str(e)}")
            raise
    
    def _combine_signals(self, strategy_signals: Dict[str, pd.DataFrame], 
                         market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine signals from multiple strategies based on the selected method.
        
        Args:
            strategy_signals: Dictionary mapping strategy IDs to their signal DataFrames.
            market_data: Original market data DataFrame.
            
        Returns:
            DataFrame containing the combined signals.
        """
        # Initialize the result DataFrame with the same index as market data
        result = pd.DataFrame(index=market_data.index)
        result['signal'] = 0.0
        result['confidence'] = 0.0
        
        # If only one strategy provided signals, return its signals directly
        if len(strategy_signals) == 1:
            strategy_id = list(strategy_signals.keys())[0]
            signals = strategy_signals[strategy_id]
            return signals
        
        # Apply the selected combination method
        if self.combination_method == SignalCombinationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_combination(strategy_signals)
        
        elif self.combination_method == SignalCombinationMethod.MAJORITY_VOTE:
            return self._majority_vote_combination(strategy_signals)
        
        elif self.combination_method == SignalCombinationMethod.HIGHEST_CONFIDENCE:
            return self._highest_confidence_combination(strategy_signals)
        
        elif self.combination_method == SignalCombinationMethod.UNANIMOUS:
            return self._unanimous_combination(strategy_signals)
        
        elif self.combination_method == SignalCombinationMethod.CUSTOM:
            # Custom combination method should be implemented in a subclass
            logger.warning("Custom combination method not implemented")
            return pd.DataFrame()
        
        # Fallback to weighted average if combination method is not recognized
        return self._weighted_average_combination(strategy_signals)
    
    def _weighted_average_combination(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals using weighted average of all strategies.
        
        Args:
            strategy_signals: Dictionary mapping strategy IDs to their signal DataFrames.
            
        Returns:
            DataFrame with weighted average signals.
        """
        # Initialize with the index from the first strategy's signals
        first_strategy_id = next(iter(strategy_signals))
        result = pd.DataFrame(index=strategy_signals[first_strategy_id].index)
        result['signal'] = 0.0
        result['confidence'] = 0.0
        
        # Calculate the sum of weights for active strategies
        total_weight = sum(self.strategy_weights[sid] for sid in strategy_signals)
        
        if total_weight == 0:
            logger.warning("Total strategy weight is zero, using equal weights")
            total_weight = len(strategy_signals)
            weights = {sid: 1.0 for sid in strategy_signals}
        else:
            weights = {sid: self.strategy_weights[sid] for sid in strategy_signals}
        
        # Calculate weighted average
        for strategy_id, signals in strategy_signals.items():
            weight = weights[strategy_id] / total_weight
            result['signal'] += signals['signal'] * weight
            result['confidence'] += signals['confidence'] * weight
        
        logger.debug(f"Combined signals using weighted average from {len(strategy_signals)} strategies")
        return result
    
    def _majority_vote_combination(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals using majority vote among strategies.
        
        Args:
            strategy_signals: Dictionary mapping strategy IDs to their signal DataFrames.
            
        Returns:
            DataFrame with majority vote signals.
        """
        # Initialize with the index from the first strategy's signals
        first_strategy_id = next(iter(strategy_signals))
        result = pd.DataFrame(index=strategy_signals[first_strategy_id].index)
        
        # Count positive and negative signals for each timestamp
        signal_counts = pd.DataFrame(index=result.index)
        signal_counts['positive'] = 0
        signal_counts['negative'] = 0
        signal_counts['neutral'] = 0
        signal_counts['total_confidence'] = 0.0
        
        for strategy_id, signals in strategy_signals.items():
            # Convert continuous signals to categorical (-1, 0, 1)
            categorical_signals = signals['signal'].apply(lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0))
            
            # Count signals
            for idx in signal_counts.index:
                if idx in categorical_signals.index:
                    signal = categorical_signals.loc[idx]
                    confidence = signals.loc[idx, 'confidence'] if 'confidence' in signals.columns else 1.0
                    
                    if signal > 0:
                        signal_counts.loc[idx, 'positive'] += 1
                    elif signal < 0:
                        signal_counts.loc[idx, 'negative'] += 1
                    else:
                        signal_counts.loc[idx, 'neutral'] += 1
                    
                    signal_counts.loc[idx, 'total_confidence'] += confidence
        
        # Determine the majority signal
        result['signal'] = 0.0
        result['confidence'] = 0.0
        
        for idx in result.index:
            total_votes = signal_counts.loc[idx, 'positive'] + signal_counts.loc[idx, 'negative'] + signal_counts.loc[idx, 'neutral']
            
            if total_votes > 0:
                # Determine signal direction by majority
                if signal_counts.loc[idx, 'positive'] > signal_counts.loc[idx, 'negative']:
                    result.loc[idx, 'signal'] = 1.0
                elif signal_counts.loc[idx, 'negative'] > signal_counts.loc[idx, 'positive']:
                    result.loc[idx, 'signal'] = -1.0
                
                # Scale by proportion of agreeing strategies
                if result.loc[idx, 'signal'] > 0:
                    majority_count = signal_counts.loc[idx, 'positive']
                    result.loc[idx, 'confidence'] = (majority_count / total_votes) * (signal_counts.loc[idx, 'total_confidence'] / total_votes)
                elif result.loc[idx, 'signal'] < 0:
                    majority_count = signal_counts.loc[idx, 'negative']
                    result.loc[idx, 'confidence'] = (majority_count / total_votes) * (signal_counts.loc[idx, 'total_confidence'] / total_votes)
                else:
                    result.loc[idx, 'confidence'] = 0.0

        logger.debug(f"Combined signals using majority vote from {len(strategy_signals)} strategies")
        return result

    def _highest_confidence_combination(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals by selecting the signal with the highest confidence at each timestamp.
        
        Args:
            strategy_signals: Dictionary mapping strategy IDs to their signal DataFrames.
            
        Returns:
            DataFrame with signals from the highest confidence strategy at each point.
        """
        # Initialize with the index from the first strategy's signals
        first_strategy_id = next(iter(strategy_signals))
        result = pd.DataFrame(index=strategy_signals[first_strategy_id].index)
        result['signal'] = 0.0
        result['confidence'] = 0.0
        result['source_strategy'] = ""
        
        # For each timestamp, find the strategy with the highest confidence
        for idx in result.index:
            max_confidence = 0.0
            best_signal = 0.0
            best_strategy = ""
            
            for strategy_id, signals in strategy_signals.items():
                if idx in signals.index:
                    confidence = signals.loc[idx, 'confidence'] if 'confidence' in signals.columns else 0.0
                    
                    # Only consider non-neutral signals with confidence above zero
                    signal_value = signals.loc[idx, 'signal']
                    if abs(signal_value) > 0.1 and confidence > max_confidence:
                        max_confidence = confidence
                        best_signal = signal_value
                        best_strategy = strategy_id
            
            result.loc[idx, 'signal'] = best_signal
            result.loc[idx, 'confidence'] = max_confidence
            result.loc[idx, 'source_strategy'] = best_strategy
        
        logger.debug(f"Combined signals using highest confidence method from {len(strategy_signals)} strategies")
        return result
    
    def _unanimous_combination(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals by requiring unanimous agreement among all strategies.
        
        Only generates a signal when all strategies agree on the direction (bullish or bearish).
        The confidence is the average of all strategy confidences.
        
        Args:
            strategy_signals: Dictionary mapping strategy IDs to their signal DataFrames.
            
        Returns:
            DataFrame with unanimous signals.
        """
        # Initialize with the index from the first strategy's signals
        first_strategy_id = next(iter(strategy_signals))
        result = pd.DataFrame(index=strategy_signals[first_strategy_id].index)
        result['signal'] = 0.0
        result['confidence'] = 0.0
        
        # Track the agreement for each timestamp
        for idx in result.index:
            signals_at_timestamp = []
            confidences_at_timestamp = []
            
            # Collect signals and confidences from all strategies at this timestamp
            for strategy_id, signals_df in strategy_signals.items():
                if idx in signals_df.index:
                    signal_value = signals_df.loc[idx, 'signal']
                    confidence = signals_df.loc[idx, 'confidence'] if 'confidence' in signals_df.columns else 0.0
                    
                    # Convert to categorical signal for agreement check
                    categorical_signal = 1 if signal_value > 0.1 else (-1 if signal_value < -0.1 else 0)
                    
                    signals_at_timestamp.append(categorical_signal)
                    confidences_at_timestamp.append(confidence)
            
            # Check for unanimous agreement (all signals must be the same and non-zero)
            if signals_at_timestamp and all(s == signals_at_timestamp[0] for s in signals_at_timestamp) and signals_at_timestamp[0] != 0:
                # All strategies agree on a non-neutral signal
                result.loc[idx, 'signal'] = signals_at_timestamp[0]  # Use the agreed direction
                
                # Use average confidence, but boost it slightly to reflect unanimity
                avg_confidence = sum(confidences_at_timestamp) / len(confidences_at_timestamp)
                result.loc[idx, 'confidence'] = min(1.0, avg_confidence * 1.2)  # Boost confidence, capped at 1.0
            else:
                # No unanimous agreement or all signals are neutral
                result.loc[idx, 'signal'] = 0.0
                result.loc[idx, 'confidence'] = 0.0
        
        logger.debug(f"Combined signals using unanimous method from {len(strategy_signals)} strategies")
        return result
