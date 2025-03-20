import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any

from src.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)

class TrendFollowingStrategy(BaseStrategy):
    """
    A comprehensive trend following strategy that utilizes multiple moving averages,
    ADX for trend strength detection, and volume confirmation.
    
    This strategy aims to capture medium to long-term market trends by:
    1. Identifying trend direction using multiple moving averages
    2. Confirming trend strength using ADX (Average Directional Index)
    3. Validating signals with volume analysis
    4. Providing trade signals with confidence levels
    
    Attributes:
        short_window (int): Short-term moving average period
        medium_window (int): Medium-term moving average period
        long_window (int): Long-term moving average period
        adx_period (int): Period for ADX calculation
        adx_threshold (float): Minimum ADX value to confirm strong trend
        volume_ma_period (int): Period for volume moving average
        volume_factor (float): Required volume factor above average for confirmation
        profit_target_pct (float): Take profit target as percentage
        stop_loss_pct (float): Stop loss as percentage
        max_holding_period (int): Maximum holding period in days
    """
    
    def __init__(
        self,
        short_window: int = 20,
        medium_window: int = 50,
        long_window: int = 200,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        volume_ma_period: int = 20,
        volume_factor: float = 1.5,
        profit_target_pct: float = 15.0,
        stop_loss_pct: float = 5.0,
        max_holding_period: int = 30,
        **kwargs
    ):
        """
        Initialize the Trend Following Strategy.
        
        Args:
            short_window: Period for short-term moving average
            medium_window: Period for medium-term moving average
            long_window: Period for long-term moving average
            adx_period: Period for ADX calculation
            adx_threshold: Minimum ADX value to confirm strong trend
            volume_ma_period: Period for volume moving average
            volume_factor: Required volume factor above average for confirmation
            profit_target_pct: Take profit target as percentage
            stop_loss_pct: Stop loss as percentage
            max_holding_period: Maximum holding period in days
            **kwargs: Additional parameters passed to the base class
        """
        super().__init__(**kwargs)
        
        # Moving averages parameters
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        
        # Trend strength parameters
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        # Volume confirmation parameters
        self.volume_ma_period = volume_ma_period
        self.volume_factor = volume_factor
        
        # Risk management parameters
        self.profit_target_pct = profit_target_pct
        self.stop_loss_pct = stop_loss_pct
        self.max_holding_period = max_holding_period
        
        logger.info(
            f"Initialized Trend Following Strategy with parameters: "
            f"MAs=({short_window},{medium_window},{long_window}), "
            f"ADX=({adx_period},{adx_threshold}), "
            f"Volume=({volume_ma_period},{volume_factor})"
        )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on trend following strategy.
        
        Args:
            data: DataFrame with price and volume data (must include 'close' and 'volume' columns)
                 and datetime index
                 
        Returns:
            DataFrame with original data plus signal columns:
                - 'signal': 1 for buy, -1 for sell, 0 for hold
                - 'confidence': Signal confidence level (0.0-1.0)
                - Additional technical indicator columns used in the strategy
        """
        if data.empty:
            logger.warning("Empty data provided to trend following strategy")
            return data
        
        if 'close' not in data.columns or 'volume' not in data.columns:
            logger.error("Data must contain 'close' and 'volume' columns")
            raise ValueError("Data must contain 'close' and 'volume' columns")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        try:
            # Calculate multiple moving averages
            df = self._calculate_moving_averages(df)
            
            # Calculate ADX for trend strength
            df = self._calculate_adx(df)
            
            # Calculate volume indicators
            df = self._calculate_volume_indicators(df)
            
            # Generate signals and confidence levels
            df = self._calculate_trend_signals(df)
            
            logger.info(f"Generated signals for {len(df)} data points")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating trend following signals: {str(e)}")
            raise
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short, medium, and long-term moving averages.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added moving average columns
        """
        # Calculate simple moving averages
        df['sma_short'] = df['close'].rolling(window=self.short_window).mean()
        df['sma_medium'] = df['close'].rolling(window=self.medium_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_window).mean()
        
        # Calculate exponential moving averages
        df['ema_short'] = df['close'].ewm(span=self.short_window, adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=self.medium_window, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_window, adjust=False).mean()
        
        # Calculate MA slopes (momentum)
        df['sma_short_slope'] = df['sma_short'].diff(5) / 5
        df['ema_short_slope'] = df['ema_short'].diff(5) / 5
        
        # Calculate crossover indicators
        df['sma_short_over_medium'] = (df['sma_short'] > df['sma_medium']).astype(int)
        df['sma_medium_over_long'] = (df['sma_medium'] > df['sma_long']).astype(int)
        df['ema_short_over_medium'] = (df['ema_short'] > df['ema_medium']).astype(int)
        
        # Moving average alignment (all aligned = stronger trend)
        df['ma_alignment'] = ((df['sma_short'] > df['sma_medium']) & 
                              (df['sma_medium'] > df['sma_long']) & 
                              (df['ema_short'] > df['ema_medium']) & 
                              (df['ema_medium'] > df['ema_long'])).astype(int)
        
        # Moving average convergence/divergence crossovers
        df['sma_cross'] = df['sma_short_over_medium'].diff()
        df['ema_cross'] = df['ema_short_over_medium'].diff()
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) for trend strength.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with added ADX columns
        """
        # Calculate True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=self.adx_period).mean()
        
        # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
        df['high_diff'] = df['high'].diff()
        df['low_diff'] = df['low'].diff()
        
        df['+dm'] = np.where(
            (df['high_diff'] > 0) & (df['high_diff'] > df['low_diff'].abs()),
            df['high_diff'],
            0
        )
        
        df['-dm'] = np.where(
            (df['low_diff'] < 0) & (df['low_diff'].abs() > df['high_diff']),
            df['low_diff'].abs(),
            0
        )
        
        # Calculate smoothed +DM and -DM
        df['+di'] = 100 * (df['+dm'].rolling(window=self.adx_period).mean() / df['atr'])
        df['-di'] = 100 * (df['-dm'].rolling(window=self.adx_period).mean() / df['atr'])
        
        # Calculate Directional Index (DX)
        df['di_diff'] = np.abs(df['+di'] - df['-di'])
        df['di_sum'] = df['+di'] + df['-di']
        df['dx'] = 100 * (df['di_diff'] / df['di_sum'])
        
        # Calculate Average Directional Index (ADX)
        df['adx'] = df['dx'].rolling(window=self.adx_period).mean()
        
        # Define trend direction and strength
        df['trend_direction'] = np.where(df['+di'] > df['-di'], 1, -1)
        df['strong_trend'] = (df['adx'] > self.adx_threshold).astype(int)
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators for signal confirmation.
        
        Args:
            df: DataFrame with price and volume data
            
        Returns:
            DataFrame with added volume indicator columns
        """
        # Calculate volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_period).mean()
        
        # Volume ratio to identify increased activity
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # High volume signal (volume significantly above average)
        df['high_volume'] = (df['volume_ratio'] > self.volume_factor).astype(int)
        
        # On-Balance Volume (OBV)
        df['price_change'] = df['close'].diff()
        df['obv'] = np.where(
            df['price_change'] > 0,
            df['volume'],
            np.where(
                df['price_change'] < 0,
                -df['volume'],
                0
            )
        ).cumsum()
        
        # OBV moving average
        df['obv_ma'] = df['obv'].rolling(window=self.volume_ma_period).mean()
        
        # Volume confirming price movement
        df['volume_confirms_price'] = (np.sign(df['price_change']) == np.sign(df['obv'] - df['obv'].shift())).astype(int)
        
        return df
    
    def _calculate_trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate final trading signals based on trend analysis.
        
        Args:
            df: DataFrame with calculated indicators
            
        Returns:
            DataFrame with added signal and confidence columns
        """
        # Initialize signal and confidence columns
        df['signal'] = 0
        df['confidence'] = 0.0
        
        # Calculate buy signals
        buy_conditions = (
            # Trend direction confirmed by moving averages
            (df['sma_short'] > df['sma_medium']) &
            (df['sma_medium'] > df['sma_long']) &
            (df['ema_short'] > df['ema_medium']) &
            
            # Moving average crossover
            (df['sma_cross'] == 1) &
            
            # Trend strength confirmed by ADX
            (df['adx'] > self.adx_threshold) &
            (df['trend_direction'] == 1) &
            
            # Volume confirmation
            (df['volume_ratio'] > 1.0) &
            (df['volume_confirms_price'] == 1)
        )
        
        # Calculate sell signals
        sell_conditions = (
            # Trend direction confirmed by moving averages
            (df['sma_short'] < df['sma_medium']) &
            (df['sma_medium'] < df['sma_long']) &
            (df['ema_short'] < df['ema_medium']) &
            
            # Moving average crossover
            (df['sma_cross'] == -1) &
            
            # Trend strength confirmed by ADX
            (df['adx'] > self.adx_threshold) &
            (df['trend_direction'] == -1) &
            
            # Volume confirmation
            (df['volume_ratio'] > 1.0) &
            (df['volume_confirms_price'] == 1)
        )
        
        # Apply signals
        df.loc[buy_conditions, 'signal'] = 1
        df.loc[sell_conditions, 'signal'] = -1
        
        # Calculate confidence levels (0.0-1.0)
        for idx in df.index:
            if df.loc[idx, 'signal'] != 0:
                confidence = self._calculate_signal_confidence(df, idx)
                df.loc[idx, 'confidence'] = confidence
        
        return df
    
    def _calculate_signal_confidence(self, df: pd.DataFrame, idx) -> float:
        """
        Calculate confidence level for a signal.
        
        Args:
            df: DataFrame with indicators
            idx: Index of the signal to evaluate
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        signal_direction = df.loc[idx, 'signal']
        
        # Base confidence factors
        if signal_direction > 0:  # Buy signal
            # ADX strength (0.0-0.4)
            adx_score = min(df.loc[idx, 'adx'] / 100, 0.4)
            confidence_factors.append(adx_score)
            
            # Moving average alignment (0.0-0.2)
            ma_score = df.loc[idx, 'ma_alignment'] * 0.2
            confidence_factors.append(ma_score)
            
            # Volume confirmation (0.0-0.2)
            volume_score = min((df.loc[idx, 'volume_ratio'] - 1.0) / 2.0, 0.2)
            confidence_factors.append(volume_score)
            
            # OBV confirmation (0.0-0.1)
            obv_score = 0.1 if df.loc[idx, 'volume_confirms_price'] == 1 else 0.0
            confidence_factors.append(obv_score)
            
            # Moving average slope - momentum strength (0.0-0.1)
            slope_score = min(max(df.loc[idx, 'sma_short_slope'] / df.loc[idx, 'close'] * 1000, 0), 0.1)
            confidence_factors.append(slope_score)
            
        elif signal_direction < 0:  # Sell signal
            # ADX strength (0.0-0.4)
            adx_score = min(df.loc[idx, 'adx'] / 100, 0.4)
            confidence_factors.append(adx_score)
            
            # Moving average alignment (0.0-0.2)
            # For sell signals, we want all MAs aligned in descending order
            ma_score = 0.2 if (df.loc[idx, 'sma_short'] < df.loc[idx, 'sma_medium'] and 
                               df.loc[idx, 'sma_medium'] < df.loc[idx, 'sma_long'] and
                               df.loc[idx, 'ema_short'] < df.loc[idx, 'ema_medium'] and
                               df.loc[idx, 'ema_medium'] < df.loc[idx, 'ema_long']) else 0.0
            confidence_factors.append(ma_score)
            
            # Volume confirmation (0.0-0.2)
            volume_score = min((df.loc[idx, 'volume_ratio'] - 1.0) / 2.0, 0.2)
            confidence_factors.append(volume_score)
            
            # OBV confirmation (0.0-0.1)
            obv_score = 0.1 if df.loc[idx, 'volume_confirms_price'] == 1 else 0.0
            confidence_factors.append(obv_score)
            
            # Moving average slope - downward momentum strength (0.0-0.1)
            slope_score = min(max(-df.loc[idx, 'sma_short_slope'] / df.loc[idx, 'close'] * 1000, 0), 0.1)
            confidence_factors.append(slope_score)

        # Calculate final confidence score (sum of all factors)
        return sum(confidence_factors)
    
    def calculate_position_size(self, capital: float, price: float, confidence: float) -> int:
        """
        Calculate the appropriate position size based on risk parameters and signal confidence.
        
        Args:
            capital: Available capital for the trade
            price: Current price of the asset
            confidence: Signal confidence score (0.0-1.0)
            
        Returns:
            Quantity to trade
        """
        # Base risk percentage (1-3% based on confidence)
        base_risk_pct = 0.01 + (confidence * 0.02)
        
        # Calculate dollar risk amount
        risk_amount = capital * base_risk_pct
        
        # Calculate position size based on stop loss
        stop_loss_amount = price * (self.stop_loss_pct / 100)
        
        # Maximum position size based on risk
        max_position = risk_amount / stop_loss_amount
        
        # Adjust position size based on confidence
        adjusted_position = max_position * confidence
        
        # Calculate quantity (whole units only)
        quantity = int(adjusted_position)
        
        logger.info(f"Calculated position size: {quantity} units (confidence: {confidence:.2f})")
        return quantity
    
    def calculate_exit_levels(self, entry_price: float, signal_direction: int) -> Dict[str, float]:
        """
        Calculate exit levels (stop loss and take profit) based on the strategy parameters.
        
        Args:
            entry_price: Entry price of the position
            signal_direction: Signal direction (1 for long, -1 for short)
            
        Returns:
            Dictionary with stop_loss and take_profit prices
        """
        if signal_direction > 0:  # Long position
            stop_loss = entry_price * (1 - self.stop_loss_pct / 100)
            take_profit = entry_price * (1 + self.profit_target_pct / 100)
        else:  # Short position
            stop_loss = entry_price * (1 + self.stop_loss_pct / 100)
            take_profit = entry_price * (1 - self.profit_target_pct / 100)
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def filter_low_confidence_signals(self, signals: pd.DataFrame, min_confidence: float = 0.5) -> pd.DataFrame:
        """
        Filter out low confidence signals from the DataFrame.
        
        Args:
            signals: DataFrame with signal and confidence columns
            min_confidence: Minimum confidence threshold
            
        Returns:
            DataFrame with only high confidence signals
        """
        high_confidence = signals[signals['confidence'] >= min_confidence].copy()
        logger.info(f"Filtered {len(signals) - len(high_confidence)} low confidence signals")
        return high_confidence
    
    def trend_reversal_detected(self, data: pd.DataFrame, lookback: int = 5) -> bool:
        """
        Detect if a trend reversal has occurred within the lookback period.
        
        Args:
            data: DataFrame with calculated indicators
            lookback: Number of periods to look back
            
        Returns:
            True if a trend reversal is detected, False otherwise
        """
        if len(data) < lookback:
            return False
        
        recent_data = data.iloc[-lookback:]
        
        # Detect trend direction change in moving averages
        ma_cross = (recent_data['sma_cross'] != 0).any() or (recent_data['ema_cross'] != 0).any()
        
        # Detect ADX falling below threshold
        adx_weakening = (recent_data['adx'].iloc[-1] < self.adx_threshold and 
                         recent_data['adx'].iloc[0] >= self.adx_threshold)
        
        # Detect trend direction change
        direction_change = (recent_data['trend_direction'].diff() != 0).any()
        
        return ma_cross or adx_weakening or direction_change
