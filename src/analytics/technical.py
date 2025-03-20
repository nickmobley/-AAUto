"""
Technical analysis module for performing market data analysis.

This module provides the TechnicalAnalyzer class which contains various
methods for calculating technical indicators such as RSI, MACD, EMAs,
and determining market trends.
"""

import logging
from typing import Dict, List, Optional, Union, Any, TypedDict, Literal
import numpy as np
from numbers import Number


class MACDResult(TypedDict):
    """Type definition for MACD calculation results."""
    macd_line: float
    signal_line: float
    histogram: float


class TechnicalAnalyzer:
    """
    Performs technical analysis on price data.
    
    This class provides methods to calculate common technical indicators
    such as RSI, MACD, and trend analysis based on moving averages.
    
    Attributes:
        api: An instance of AlphaVantageAPI for fetching market data.
        logger: Logger instance for tracking operations.
    """
    
    def __init__(self, api):
        """
        Initialize the TechnicalAnalyzer with an API client.
        
        Args:
            api: An instance of AlphaVantageAPI for fetching market data.
        """
        self.api = api
        self.logger = logging.getLogger(__name__)
        
    def calculate_rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Calculate Relative Strength Index for a symbol.
        
        The RSI is a momentum oscillator that measures the speed and change of price movements.
        RSI oscillates between 0 and 100. Traditionally, RSI is considered overbought when 
        above 70 and oversold when below 30.
        
        Args:
            symbol: The stock symbol to analyze
            period: The lookback period for RSI calculation (default: 14)
            
        Returns:
            The calculated RSI value or None if data is insufficient or unavailable
            
        Raises:
            ValueError: If period is less than 2
        """
        if period < 2:
            raise ValueError("RSI period must be at least 2")
            
        try:
            data = self.api.get_daily_data(symbol)
            if not data or "Time Series (Daily)" not in data:
                self.logger.warning(f"No data available for {symbol} when calculating RSI")
                return None
                
            time_series = data["Time Series (Daily)"]
            closes = [float(v["4. close"]) for v in time_series.values()]
            
            if len(closes) < period + 1:
                self.logger.warning(f"Insufficient data for {symbol} to calculate RSI with period {period}")
                return None
                
            # Calculate price changes
            deltas = [closes[i] - closes[i+1] for i in range(len(closes)-1)]
            
            # Separate gains and losses
            gains = [d if d > 0 else 0 for d in deltas]
            losses = [-d if d < 0 else 0 for d in deltas]
            
            # Calculate average gains and losses
            avg_gain = sum(gains[:period]) / period
            avg_loss = sum(losses[:period]) / period
            
            # Calculate RSI
            if avg_loss == 0:
                return 100.0  # If no losses, RSI is 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI for {symbol}: {str(e)}")
            return None
        
    def calculate_macd(
        self, 
        symbol: str, 
        fast_period: int = 12, 
        slow_period: int = 26, 
        signal_period: int = 9
    ) -> Optional[MACDResult]:
        """
        Calculate MACD (Moving Average Convergence Divergence) for a symbol.
        
        MACD is a trend-following momentum indicator that shows the relationship
        between two moving averages of a security's price.
        
        Args:
            symbol: The stock symbol to analyze
            fast_period: Period for the fast EMA (default: 12)
            slow_period: Period for the slow EMA (default: 26)
            signal_period: Period for the signal line (default: 9)
            
        Returns:
            A dictionary containing 'macd_line', 'signal_line', and 'histogram' values,
            or None if data is insufficient or unavailable
            
        Raises:
            ValueError: If any period values are invalid
        """
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        if signal_period < 1 or fast_period < 1 or slow_period < 1:
            raise ValueError("All periods must be positive integers")
            
        try:
            data = self.api.get_daily_data(symbol)
            if not data or "Time Series (Daily)" not in data:
                self.logger.warning(f"No data available for {symbol} when calculating MACD")
                return None
                
            time_series = data["Time Series (Daily)"]
            closes = [float(v["4. close"]) for v in time_series.values()]
            
            if len(closes) < slow_period + signal_period:
                self.logger.warning(
                    f"Insufficient data for {symbol} to calculate MACD with parameters "
                    f"({fast_period}, {slow_period}, {signal_period})"
                )
                return None
                
            # Calculate EMAs
            fast_ema = self.calculate_ema(closes, fast_period)
            slow_ema = self.calculate_ema(closes, slow_period)
            
            # Calculate MACD line
            macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
            
            # Calculate signal line
            signal_line = self.calculate_ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
            
            return {
                'macd_line': macd_line[0],
                'signal_line': signal_line[0],
                'histogram': histogram[0]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD for {symbol}: {str(e)}")
            return None
        
    def calculate_ema(self, data: List[float], period: int) -> List[float]:
        """
        Calculate Exponential Moving Average for a data series.
        
        EMA gives more weight to recent prices. The weighting applied to the
        most recent price depends on the specified period.
        
        Args:
            data: List of price data points (oldest to newest)
            period: The lookback period for EMA calculation
            
        Returns:
            List of EMA values corresponding to the input data
            
        Raises:
            ValueError: If period is invalid or data is insufficient
        """
        if period < 1:
            raise ValueError("EMA period must be at least 1")
        if not data or len(data) == 0:
            raise ValueError("Data cannot be empty")
        if len(data) < period:
            raise ValueError(f"Data length ({len(data)}) must be at least equal to period ({period})")
            
        try:
            multiplier = 2 / (period + 1)
            ema = [data[0]]  # First EMA is SMA
            
            for price in data[1:]:
                ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
                
            return ema
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {str(e)}")
            raise
        
    def get_trend(
        self, 
        symbol: str, 
        short_period: int = 10, 
        long_period: int = 50
    ) -> Optional[Literal["UPTREND", "DOWNTREND", "SIDEWAYS"]]:
        """
        Determine market trend based on moving averages.
        
        This method calculates short and long-term moving averages and compares them
        to determine the overall market trend.
        
        Args:
            symbol: The stock symbol to analyze
            short_period: Period for short-term moving average (default: 10)
            long_period: Period for long-term moving average (default: 50)
            
        Returns:
            String indicating trend direction ("UPTREND", "DOWNTREND", or "SIDEWAYS"),
            or None if data is insufficient or unavailable
            
        Raises:
            ValueError: If period values are invalid
        """
        if short_period >= long_period:
            raise ValueError("Short period must be less than long period")
        if short_period < 1 or long_period < 1:
            raise ValueError("All periods must be positive integers")
            
        try:
            data = self.api.get_daily_data(symbol)
            if not data or "Time Series (Daily)" not in data:
                self.logger.warning(f"No data available for {symbol} when determining trend")
                return None
                
            time_series = data["Time Series (Daily)"]
            closes = [float(v["4. close"]) for v in time_series.values()]
            
            if len(closes) < long_period:
                self.logger.warning(
                    f"Insufficient data for {symbol} to determine trend with parameters "
                    f"({short_period}, {long_period})"
                )
                return None
                
            # Calculate moving averages - alternative implementation using numpy for precision
            try:
                short_ma = np.mean(closes[:short_period])
                long_ma = np.mean(closes[:long_period])
            except ImportError:
                # Fallback to standard calculation if numpy not available
                short_ma = sum(closes[:short_period]) / short_period
                long_ma = sum(closes[:long_period]) / long_period
            
            # Determine trend with a threshold to avoid noise
            threshold = 0.001  # 0.1% threshold to determine sideways market
            percent_diff = abs((short_ma - long_ma) / long_ma)
            
            if percent_diff < threshold:
                return "SIDEWAYS"
            elif short_ma > long_ma:
                return "UPTREND"
            else:
                return "DOWNTREND"
                
        except Exception as e:
            self.logger.error(f"Error determining trend for {symbol}: {str(e)}")
            return None

