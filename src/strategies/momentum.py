"""
Momentum trading strategy implementation.

This module provides an implementation of a momentum-based trading strategy that
analyzes price movements to identify and capitalize on price trends. The strategy
generates buy signals when momentum is increasing and sell signals when momentum
is decreasing beyond certain thresholds.
"""
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union

import numpy as np
import pandas as pd
from loguru import logger

from src.strategies.base import Strategy, Signal, SignalType


class MomentumStrategy(Strategy):
    """
    Momentum-based trading strategy.
    
    This strategy calculates momentum indicators (rate of change) over different
    time periods and generates trading signals based on momentum thresholds.
    It can use multiple lookback periods and signal thresholds.
    """
    
    # Default parameters if none provided
    DEFAULT_PARAMETERS = {
        "short_period": 5,       # Short-term momentum period
        "medium_period": 10,     # Medium-term momentum period
        "long_period": 20,       # Long-term momentum period
        "buy_threshold": 0.02,   # 2% momentum threshold for buy signal
        "sell_threshold": -0.02, # -2% momentum threshold for sell signal
        "rsi_period": 14,        # Period for RSI calculation
        "rsi_overbought": 70,    # RSI overbought threshold
        "rsi_oversold": 30,      # RSI oversold threshold
        "volume_factor": 1.5,    # Volume increase factor for signal confirmation
        "signal_lookback": 3,    # Number of consecutive bars to confirm signal
    }
    
    def __init__(self, name: str, symbols: List[str], parameters: Dict[str, Any] = None):
        """
        Initialize the momentum strategy.
        
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
        - Thresholds are within reasonable ranges
        - RSI parameters are valid
        """
        # Check that periods are positive integers
        for period_name in ["short_period", "medium_period", "long_period", "rsi_period"]:
            if period_name not in self.parameters:
                self._validation_errors.append(f"Missing required parameter: {period_name}")
                continue
                
            period = self.parameters[period_name]
            if not isinstance(period, int) or period <= 0:
                self._validation_errors.append(
                    f"{period_name} must be a positive integer, got {period}"
                )
        
        # Check that periods are in ascending order
        if (all(p in self.parameters for p in ["short_period", "medium_period", "long_period"]) and
                not (self.parameters["short_period"] < self.parameters["medium_period"] < 
                     self.parameters["long_period"])):
            self._validation_errors.append(
                "Periods must be in ascending order: short_period < medium_period < long_period"
            )
        
        # Check thresholds
        if "buy_threshold" in self.parameters and "sell_threshold" in self.parameters:
            if self.parameters["buy_threshold"] <= self.parameters["sell_threshold"]:
                self._validation_errors.append(
                    f"Buy threshold ({self.parameters['buy_threshold']}) must be greater than "
                    f"sell threshold ({self.parameters['sell_threshold']})"
                )
        
        # Check RSI parameters
        if "rsi_overbought" in self.parameters and "rsi_oversold" in self.parameters:
            if not (0 <= self.parameters["rsi_oversold"] < self.parameters["rsi_overbought"] <= 100):
                self._validation_errors.append(
                    f"RSI parameters must satisfy: 0 <= oversold ({self.parameters['rsi_oversold']}) < "
                    f"overbought ({self.parameters['rsi_overbought']}) <= 100"
                )
        
        # Check volume factor
        if "volume_factor" in self.parameters and self.parameters["volume_factor"] <= 0:
            self._validation_errors.append(
                f"Volume factor must be positive, got {self.parameters['volume_factor']}"
            )
    
    def get_required_data(self) -> Dict[str, Any]:
        """
        Get the data requirements for the momentum strategy.
        
        Returns:
            Dictionary with required data specifications
        """
        # Determine the longest lookback period needed
        max_period = max(
            self.parameters.get("long_period", 20),
            self.parameters.get("rsi_period", 14)
        )
        
        # Add buffer for calculations
        lookback_bars = max_period * 3
        
        return {
            "timeframe": "1d",  # Daily data
            "bar_count": lookback_bars,
            "columns": ["open", "high", "low", "close", "volume"],
            "indicators": ["rsi", "volume_sma"],
            "adjustments": ["splits", "dividends"]
        }
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals based on momentum indicators.
        
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
                required_columns = ["close", "volume"]
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Calculate momentum indicators
                momentum_signals = self._calculate_momentum_signals(df)
                
                # Generate signals based on momentum analysis
                for signal_type, price, timestamp, confidence, metadata in momentum_signals:
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
    
    def _calculate_momentum_signals(self, df: pd.DataFrame) -> List[Tuple[SignalType, float, datetime, float, Dict]]:
        """
        Calculate momentum indicators and generate signals.
        
        Args:
            df: DataFrame with market data for a symbol
            
        Returns:
            List of signal tuples (signal_type, price, timestamp, confidence, metadata)
        """
        signals = []
        
        try:
            # Make a copy to avoid modifying the original
            data = df.copy()
            
            # Calculate momentum over different periods
            short_period = self.parameters["short_period"]
            medium_period = self.parameters["medium_period"]
            long_period = self.parameters["long_period"]
            
            # Rate of change calculation: (current_price - price_n_periods_ago) / price_n_periods_ago
            data[f'mom_short'] = data['close'].pct_change(short_period)
            data[f'mom_medium'] = data['close'].pct_change(medium_period)
            data[f'mom_long'] = data['close'].pct_change(long_period)
            
            # Calculate RSI
            rsi_period = self.parameters["rsi_period"]
            data['rsi'] = self._calculate_rsi(data['close'], rsi_period)
            
            # Calculate volume SMA
            data['volume_sma'] = data['volume'].rolling(window=medium_period).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
            
            # Drop NaN values from calculations
            data.dropna(inplace=True)
            
            # No data to analyze after dropping NaNs
            if len(data) == 0:
                return signals
            
            # Get signal thresholds
            buy_threshold = self.parameters["buy_threshold"]
            sell_threshold = self.parameters["sell_threshold"]
            rsi_overbought = self.parameters["rsi_overbought"]
            rsi_oversold = self.parameters["rsi_oversold"]
            volume_factor = self.parameters["volume_factor"]
            
            # Process each row for signals
            for idx, row in data.iloc[-5:].iterrows():  # Look at most recent 5 bars
                close_price = row['close']
                timestamp = idx if isinstance(idx, datetime) else pd.to_datetime(idx)
                
                # Initialize signal metadata
                metadata = {
                    'mom_short': row['mom_short'],
                    'mom_medium': row['mom_medium'],
                    'mom_long': row['mom_long'],
                    'rsi': row['rsi'],
                    'volume_ratio': row['volume_ratio']
                }
                
                # Buy signal conditions
                buy_condition = (row['mom_short'] > buy_threshold and
                               row['mom_medium'] > 0 and
                               row['rsi'] < 70 and
                               row['volume_ratio'] > volume_factor)
                
                # Sell signal conditions
                sell_condition = (row['mom_short'] < sell_threshold and
                               (row['mom_medium'] < 0 or
                               row['rsi'] > rsi_overbought) and
                               row['volume_ratio'] > volume_factor)
                
                # Calculate confidence scores based on signal strength
                buy_confidence = 0.0
                sell_confidence = 0.0
                
                if buy_condition:
                    # Stronger buy signals get higher confidence
                    mom_strength = min(row['mom_short'] / buy_threshold * 0.5, 0.5)
                    rsi_strength = max(0, (rsi_oversold - row['rsi']) / rsi_oversold) * 0.3
                    volume_strength = min((row['volume_ratio'] - volume_factor) / volume_factor, 1.0) * 0.2
                    
                    buy_confidence = min(mom_strength + rsi_strength + volume_strength, 1.0)
                    
                    signals.append((
                        SignalType.BUY,
                        close_price,
                        timestamp,
                        buy_confidence,
                        metadata
                    ))
                
                elif sell_condition:
                    # Stronger sell signals get higher confidence
                    mom_strength = min(abs(row['mom_short'] / sell_threshold) * 0.5, 0.5)
                    rsi_strength = max(0, (row['rsi'] - rsi_overbought) / (100 - rsi_overbought)) * 0.3
                    volume_strength = min((row['volume_ratio'] - volume_factor) / volume_factor, 1.0) * 0.2
                    
                    sell_confidence = min(mom_strength + rsi_strength + volume_strength, 1.0)
                    
                    signals.append((
                        SignalType.SELL,
                        close_price,
                        timestamp,
                        sell_confidence,
                        metadata
                    ))
            
            return signals
        
        except Exception as e:
            logger.error(f"Error in momentum signal calculation: {str(e)}")
            return []
    
    def _calculate_rsi(self, price_series: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate the Relative Strength Index (RSI) for a price series.
        
        Args:
            price_series: Series of prices
            period: RSI calculation period (default: 14)
            
        Returns:
            Series containing RSI values
        """
        # Calculate price changes
        delta = price_series.diff()
        
        # Split gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss over period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1)  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
