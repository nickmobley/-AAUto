"""
Mean reversion trading strategy implementation.

This module provides an implementation of a mean reversion strategy that
looks for price deviations from historical means and generates signals
based on the expectation that prices will revert to their mean over time.
"""
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import Strategy, Signal, SignalType


class MeanReversionStrategy(Strategy):
    """
    Mean reversion trading strategy.
    
    This strategy identifies assets that have deviated significantly from their
    historical mean and generates signals based on the expectation that they will
    revert back to the mean. It can use different types of means (simple, exponential,
    bollinger bands) and deviation thresholds.
    """
    
    # Default parameters if none provided
    DEFAULT_PARAMETERS = {
        "lookback_period": 20,      # Period for mean calculation
        "entry_std_dev": 2.0,       # Standard deviations from mean for entry
        "exit_std_dev": 0.5,        # Standard deviations from mean for exit
        "mean_type": "sma",         # Type of mean (sma, ema, bollinger)
        "min_history": 30,          # Minimum historical data points required
        "bollinger_period": 20,     # Period for Bollinger Bands calculation
        "use_atr": True,            # Use ATR for position sizing
        "atr_period": 14,           # Period for ATR calculation
        "atr_multiplier": 2.0,      # Multiplier for ATR-based stops
        "volume_filter": True,      # Filter using volume
        "volume_min_percentile": 40 # Minimum volume percentile for valid signals
    }
    
    def __init__(self, name: str, symbols: List[str], parameters: Dict[str, Any] = None):
        """
        Initialize the mean reversion strategy.
        
        Args:
            name: Unique name for this strategy instance
            symbols: List of trading symbols to analyze
            parameters: Dictionary of strategy parameters, will use defaults for any missing
        """
        # Merge provided parameters with defaults
        if parameters:
            merged_params = self.DEFAULT_PARAMETERS.copy()
            merged_params.update(parameters)
            parameters = merged_params
        else:
            parameters = self.DEFAULT_PARAMETERS.copy()
            
        super().__init__(name, symbols, parameters)
    
    def _validate_parameters(self) -> None:
        """
        Validate the strategy parameters.
        
        Checks:
        - Periods are positive integers
        - Standard deviation values are positive
        - Mean type is valid
        - ATR parameters are valid if ATR is used
        """
        # Check that periods are positive integers
        for period_name in ["lookback_period", "min_history", "bollinger_period", "atr_period"]:
            if period_name not in self.parameters:
                self._validation_errors.append(f"Missing required parameter: {period_name}")
                continue
                
            period = self.parameters[period_name]
            if not isinstance(period, int) or period <= 0:
                self._validation_errors.append(
                    f"{period_name} must be a positive integer, got {period}"
                )
        
        # Check that standard deviation values are positive
        for std_param in ["entry_std_dev", "exit_std_dev"]:
            if std_param not in self.parameters:
                self._validation_errors.append(f"Missing required parameter: {std_param}")
                continue
                
            std_value = self.parameters[std_param]
            if not isinstance(std_value, (int, float)) or std_value <= 0:
                self._validation_errors.append(
                    f"{std_param} must be a positive number, got {std_value}"
                )
        
        # Validate mean type
        valid_mean_types = ["sma", "ema", "bollinger"]
        if "mean_type" in self.parameters and self.parameters["mean_type"] not in valid_mean_types:
            self._validation_errors.append(
                f"Mean type must be one of {valid_mean_types}, got {self.parameters['mean_type']}"
            )
        
        # Validate exit std is less than entry std
        if ("entry_std_dev" in self.parameters and "exit_std_dev" in self.parameters and
                self.parameters["exit_std_dev"] > self.parameters["entry_std_dev"]):
            self._validation_errors.append(
                f"Exit standard deviation ({self.parameters['exit_std_dev']}) must be less than "
                f"entry standard deviation ({self.parameters['entry_std_dev']})"
            )
        
        # Validate ATR parameters if ATR is used
        if self.parameters.get("use_atr", False):
            if "atr_multiplier" in self.parameters and self.parameters["atr_multiplier"] <= 0:
                self._validation_errors.append(
                    f"ATR multiplier must be positive, got {self.parameters['atr_multiplier']}"
                )
    
    def get_required_data(self) -> Dict[str, Any]:
        """
        Get the data requirements for the mean reversion strategy.
        
        Returns:
            Dictionary with required data specifications
        """
        # Determine the longest lookback period needed
        min_history = self.parameters["min_history"]
        lookback_period = self.parameters["lookback_period"]
        bollinger_period = self.parameters["bollinger_period"]
        atr_period = self.parameters["atr_period"]
        
        max_period = max(min_history, lookback_period, bollinger_period, atr_period)
        
        # Add buffer for calculations
        lookback_bars = max_period * 3
        
        return {
            "timeframe": "1d",  # Daily data
            "bar_count": lookback_bars,
            "columns": ["open", "high", "low", "close", "volume"],
            "indicators": ["atr", "bollinger", "volume_sma"],
            "adjustments": ["splits", "dividends"]
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals based on mean reversion indicators.
        
        Args:
            data: Dictionary of DataFrames with OHLCV data for each symbol
            
        Returns:
            List of Signal objects with buy/sell recommendations
            
        Raises:
            ValueError: If data is missing required columns
            KeyError: If symbols are missing from the data
        """
        signals = []
        
        # Verify that all symbols are in the data
        missing_symbols = set(self.symbols) - set(data.keys())
        if missing_symbols:
            raise KeyError(f"Missing data for symbols: {missing_symbols}")
        
        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue
                
            try:
                # Verify required columns exist
                required_columns = ["open", "high", "low", "close", "volume"]
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Calculate mean reversion indicators
                mean_reversion_signals = self._calculate_mean_reversion_signals(df)
                
                # Generate signals based on mean reversion analysis
                for signal_type, price, timestamp, confidence, metadata in mean_reversion_signals:
                    signals.append(
                        Signal(
                            type=signal_type,
                            symbol=symbol,
                            timestamp=timestamp,
                            price=price,
                            confidence=confidence,
                            metadata=metadata
                        )
                    )
            except Exception as e:
                logger.error(f"Error generating signals for {symbol}: {str(e)}")
                # Continue with other symbols even if one fails
        
        return signals
    
    def _calculate_mean_reversion_signals(self, df: pd.DataFrame) -> List[Tuple[SignalType, float, datetime, float, Dict]]:
        """
        Calculate mean reversion indicators and generate signals.
        
        Args:
            df: DataFrame with market data for a symbol
            
        Returns:
            List of signal tuples (signal_type, price, timestamp, confidence, metadata)
        """
        signals = []
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Get parameters
            lookback_period = self.parameters["lookback_period"]
            entry_std_dev = self.parameters["entry_std_dev"]
            exit_std_dev = self.parameters["exit_std_dev"]
            mean_type = self.parameters["mean_type"]
            use_volume_filter = self.parameters.get("volume_filter", True)
            volume_min_percentile = self.parameters.get("volume_min_percentile", 40)
            
            # Calculate the appropriate mean based on selected type
            if mean_type == "sma":
                data["mean"] = data["close"].rolling(window=lookback_period).mean()
                data["std"] = data["close"].rolling(window=lookback_period).std()
            elif mean_type == "ema":
                data["mean"] = data["close"].ewm(span=lookback_period, adjust=False).mean()
                data["std"] = data["close"].rolling(window=lookback_period).std()
            elif mean_type == "bollinger":
                bollinger_period = self.parameters["bollinger_period"]
                data["mean"] = data["close"].rolling(window=bollinger_period).mean()
                data["std"] = data["close"].rolling(window=bollinger_period).std()
            
            # Calculate deviation from mean in terms of standard deviations
            data["z_score"] = (data["close"] - data["mean"]) / data["std"]
            
            # Calculate ATR if used
            if self.parameters.get("use_atr", True):
                atr_period = self.parameters["atr_period"]
                data["atr"] = self._calculate_atr(data, atr_period)
            
            # Calculate volume metrics for filtering
            if use_volume_filter:
                data["volume_pct"] = data["volume"].rolling(window=lookback_period).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100
                )
            
            # Drop NaN values from calculations
            data.dropna(inplace=True)
            
            # No data to analyze after dropping NaNs
            if len(data) == 0:
                return signals
            
            # Process each row for signals
            for idx, row in data.iloc[-5:].iterrows():  # Look at most recent 5 bars
                close_price = row["close"]
                timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
                
                # Initialize signal metadata
                metadata = {
                    "mean": row["mean"],
                    "std": row["std"],
                    "z_score": row["z_score"],
                    "price_deviation_pct": (row["close"] - row["mean"]) / row["mean"] * 100
                }
                
                if self.parameters.get("use_atr", True):
                    metadata["atr"] = row["atr"]
                    metadata["atr_stop"] = row["atr"] * self.parameters["atr_multiplier"]
                
                # Volume filtering
                if use_volume_filter and row["volume_pct"] < volume_min_percentile:
                    # Skip if volume is too low
                    continue
                
                # Generate Buy signals - when price is significantly below the mean
                if row["z_score"] <= -entry_std_dev:
                    # Calculate confidence based on the deviation from mean
                    # Higher deviation means higher confidence (up to a point)
                    confidence = min(0.95, 0.5 + abs(row["z_score"]) / 10.0)
                    
                    # Additional confidence boost for higher volume
                    if use_volume_filter and row["volume_pct"] > 70:
                        confidence = min(0.98, confidence + 0.1)
                    
                    # Add ATR-based stop loss to metadata
                    if self.parameters.get("use_atr", True):
                        metadata["stop_loss"] = close_price - metadata["atr_stop"]
                        metadata["take_profit"] = row["mean"]
                    
                    signals.append((
                        SignalType.BUY,
                        close_price,
                        timestamp,
                        confidence,
                        metadata
                    ))
                
                # Generate Sell signals - when price is significantly above the mean
                elif row["z_score"] >= entry_std_dev:
                    # Calculate confidence based on the deviation from mean
                    confidence = min(0.95, 0.5 + abs(row["z_score"]) / 10.0)
                    
                    # Additional confidence boost for higher volume
                    if use_volume_filter and row["volume_pct"] > 70:
                        confidence = min(0.98, confidence + 0.1)
                    
                    # Add ATR-based stop loss to metadata
                    if self.parameters.get("use_atr", True):
                        metadata["stop_loss"] = close_price + metadata["atr_stop"]
                        metadata["take_profit"] = row["mean"]
                    
                    signals.append((
                        SignalType.SELL,
                        close_price,
                        timestamp,
                        confidence,
                        metadata
                    ))
                
                # Generate Exit signals for existing positions
                # Exit long positions when price returns to the mean (within exit_std_dev)
                elif -exit_std_dev <= row["z_score"] <= 0 and abs(row["z_score"]) < abs(entry_std_dev):
                    # Lower confidence for exit signals
                    confidence = 0.6 + (exit_std_dev - abs(row["z_score"])) / exit_std_dev * 0.2
                    
                    signals.append((
                        SignalType.EXIT_LONG,
                        close_price,
                        timestamp,
                        confidence,
                        metadata
                    ))
                
                # Exit short positions when price returns to the mean (within exit_std_dev)
                elif 0 <= row["z_score"] <= exit_std_dev and abs(row["z_score"]) < abs(entry_std_dev):
                    # Lower confidence for exit signals
                    confidence = 0.6 + (exit_std_dev - abs(row["z_score"])) / exit_std_dev * 0.2
                    
                    signals.append((
                        SignalType.EXIT_SHORT,
                        close_price,
                        timestamp,
                        confidence,
                        metadata
                    ))
            
            return signals
        
        except Exception as e:
            logger.error(f"Error calculating mean reversion signals: {str(e)}")
            return []
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Average True Range (ATR) for volatility measurement.
        
        Args:
            data: DataFrame with high, low, close prices
            period: ATR calculation period
            
        Returns:
            Series with ATR values
        """
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)
        
        # Calculate true range
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        # True range is the maximum of these three
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=period).mean()
        
        return atr
