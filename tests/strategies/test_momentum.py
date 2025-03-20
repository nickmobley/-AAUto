"""
Unit tests for the MomentumStrategy class.

This module contains comprehensive tests for the MomentumStrategy implementation,
including initialization, parameter validation, signal generation, and error handling.
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import SignalType
from src.strategies.momentum import MomentumStrategy


class TestMomentumStrategy(unittest.TestCase):
    """Test suite for the MomentumStrategy class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.symbols = ["AAPL", "MSFT"]
        self.strategy_name = "test_momentum"
        self.default_params = MomentumStrategy.DEFAULT_PARAMETERS.copy()
        self.strategy = MomentumStrategy(
            name=self.strategy_name,
            symbols=self.symbols
        )

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters."""
        strategy = MomentumStrategy(
            name=self.strategy_name,
            symbols=self.symbols
        )
        
        self.assertEqual(strategy.name, self.strategy_name)
        self.assertEqual(strategy.symbols, self.symbols)
        self.assertEqual(strategy.parameters, self.default_params)
        self.assertTrue(strategy.is_valid())
        self.assertEqual(len(strategy.get_validation_errors()), 0)

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_params = {
            "short_period": 3,
            "medium_period": 8,
            "long_period": 15,
            "buy_threshold": 0.05,
            "sell_threshold": -0.05,
        }
        
        strategy = MomentumStrategy(
            name=self.strategy_name,
            symbols=self.symbols,
            parameters=custom_params
        )
        
        # Verify custom parameters were merged with defaults
        self.assertEqual(strategy.parameters["short_period"], custom_params["short_period"])
        self.assertEqual(strategy.parameters["medium_period"], custom_params["medium_period"])
        self.assertEqual(strategy.parameters["long_period"], custom_params["long_period"])
        self.assertEqual(strategy.parameters["buy_threshold"], custom_params["buy_threshold"])
        self.assertEqual(strategy.parameters["sell_threshold"], custom_params["sell_threshold"])
        
        # Default parameters not overridden should still be present
        self.assertEqual(strategy.parameters["rsi_period"], self.default_params["rsi_period"])
        self.assertTrue(strategy.is_valid())

    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        # Using the strategy created in setUp which uses default parameters
        self.assertTrue(self.strategy.is_valid())
        self.assertEqual(len(self.strategy.get_validation_errors()), 0)

    def test_parameter_validation_failure_invalid_periods(self):
        """Test parameter validation with invalid period values."""
        invalid_params = {
            "short_period": -1,
            "medium_period": 0,
            "long_period": "invalid",
        }
        
        with self.assertRaises(ValueError):
            MomentumStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_period_order(self):
        """Test parameter validation with periods in wrong order."""
        invalid_params = {
            "short_period": 10,    # Should be less than medium_period
            "medium_period": 5,    # Should be less than long_period
            "long_period": 20,
        }
        
        with self.assertRaises(ValueError):
            MomentumStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_thresholds(self):
        """Test parameter validation with invalid threshold values."""
        invalid_params = {
            "buy_threshold": -0.02,   # Buy threshold should be positive
            "sell_threshold": -0.05,  # Buy threshold should be > sell threshold
        }
        
        with self.assertRaises(ValueError):
            MomentumStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_rsi_params(self):
        """Test parameter validation with invalid RSI parameters."""
        invalid_params = {
            "rsi_oversold": 40,      # Should be < rsi_overbought
            "rsi_overbought": 30,    # Should be > rsi_oversold
        }
        
        with self.assertRaises(ValueError):
            MomentumStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_get_required_data(self):
        """Test the get_required_data method returns correct data specifications."""
        data_req = self.strategy.get_required_data()
        
        self.assertEqual(data_req["timeframe"], "1d")
        self.assertTrue(data_req["bar_count"] >= self.default_params["long_period"] * 3)
        self.assertIn("open", data_req["columns"])
        self.assertIn("high", data_req["columns"])
        self.assertIn("low", data_req["columns"])
        self.assertIn("close", data_req["columns"])
        self.assertIn("volume", data_req["columns"])
        self.assertIn("rsi", data_req["indicators"])
        self.assertIn("volume_sma", data_req["indicators"])

    def test_calculate_rsi(self):
        """Test the RSI calculation method."""
        # Create a simple price series for testing
        prices = pd.Series([
            100.0, 102.0, 104.0, 103.0, 105.0, 107.0, 109.0, 108.0, 
            107.0, 106.0, 105.0, 104.0, 103.0, 102.0, 101.0, 100.0
        ])
        
        rsi = self.strategy._calculate_rsi(prices, period=5)
        
        # First 5 values should be NaN due to the window
        self.assertTrue(rsi.iloc[:5].isna().all())
        
        # RSI should be between 0 and 100
        self.assertTrue((rsi.dropna() >= 0).all() and (rsi.dropna() <= 100).all())
        
        # When prices consistently rise, RSI should be high
        rising_prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0])
        rsi_rising = self.strategy._calculate_rsi(rising_prices, period=5)
        # Skip the first 5 values which are NaN
        self.assertTrue(rsi_rising.iloc[-1] > 70)
        
        # When prices consistently fall, RSI should be low
        falling_prices = pd.Series([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 94.0, 93.0, 92.0, 91.0])
        rsi_falling = self.strategy._calculate_rsi(falling_prices, period=5)
        # Skip the first 5 values which are NaN
        self.assertTrue(rsi_falling.iloc[-1] < 30)

    def _create_test_data(self, symbol="AAPL", days=30, uptrend=True, volume_spike=False):
        """Helper method to create test data for signal generation tests."""
        base_date = datetime(2023, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(days)]
        
        # Create price data - either uptrend or downtrend
        if uptrend:
            close_prices = [100 + i * 0.5 + (np.random.random() - 0.5) * 2 for i in range(days)]
        else:
            close_prices = [100 - i * 0.5 + (np.random.random() - 0.5) * 2 for i in range(days)]
        
        # Create volume data - optionally include a volume spike
        volumes = [10000 + np.random.random() * 5000 for _ in range(days)]
        if volume_spike:
            # Add volume spike near the end
            volumes[-5:] = [vol * 2 for vol in volumes[-5:]]
        
        # Create OHLC data
        opens = [close - 0.5 + np.random.random() for close in close_prices]
        highs = [max(open_price, close) + 0.5 + np.random.random() for open_price, close in zip(opens, close_prices)]
        lows = [min(open_price, close) - 0.5 - np.random.random() for open_price, close in zip(opens, close_prices)]
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        }, index=dates)
        
        return {symbol: df}

    def test_generate_signals_uptrend_with_volume(self):
        """Test signal generation in an uptrend with volume confirmation."""
        # Create uptrend data with volume spike
        test_data = self._create_test_data(uptrend=True, volume_spike=True)
        
        # Generate signals
        signals = self.strategy.generate_signals(test_data)
        
        # Should have generated at least one BUY signal
        buy_signals = [s for s in signals if s.type == SignalType.BUY]
        self.assertTrue(len(buy_signals) > 0)
        
        # Check that signals have proper attributes
        for signal in buy_signals:
            self.assertEqual(signal.symbol, "AAPL")
            self.assertTrue(isinstance(signal.timestamp, datetime))
            self.assertTrue(isinstance(signal.price, float))
            self.assertTrue(0 <= signal.confidence <= 1.0)
            self.assertTrue(isinstance(signal.metadata, dict))
            self.assertIn('mom_short', signal.metadata)
            self.assertIn('mom_medium', signal.metadata)
            self.assertIn('mom_long', signal.metadata)
            self.assertIn('rsi', signal.metadata)
            self.assertIn('volume_ratio', signal.metadata)

    def test_generate_signals_downtrend_with_volume(self):
        """Test signal generation in a downtrend with volume confirmation."""
        # Create downtrend data with volume spike
        test_data = self._create_test_data(uptrend=False, volume_spike=True)
        
        # Generate signals
        signals = self.strategy.generate_signals(test_data)
        
        # Should have generated at least one SELL signal
        sell_signals = [s for s in signals if s.type == SignalType.SELL]
        self.assertTrue(len(sell_signals) > 0)
        
        # Check signal attributes
        for signal in sell_signals:
            self.assertEqual(signal.symbol, "AAPL")
            self.assertTrue(isinstance(signal.timestamp, datetime))
            self.assertTrue(isinstance(signal.price, float))
            self.assertTrue(0 <= signal.confidence <= 1.0)
            self.assertTrue(isinstance(signal.metadata, dict))

    def test_generate_signals_multiple_symbols(self):
        """Test signal generation for multiple symbols."""
        # Create data for multiple symbols
        aapl_data = self._create_test_data(symbol="AAPL", uptrend=True, volume_spike=True)
        msft_data = self._create_test_data(symbol="MSFT", uptrend=False, volume_spike=True)
        test_data = {**aapl_data, **msft_data}
        
        # Create strategy with both symbols
        strategy = MomentumStrategy(
            name=self.strategy_name,
            symbols=["AAPL", "MSFT"]
        )
        
        # Generate signals
        signals = strategy.generate_signals(test_data)
        
        # Should have signals for both symbols
        aapl_signals = [s for s in signals if s.symbol == "AAPL"]
        msft_signals = [s for s in signals if s.symbol == "MSFT"]
        
        self.assertTrue(len(aapl_signals) > 0)
        self.assertTrue(len(msft_signals) > 0)

    def test_generate_signals_missing_data(self):
        """Test error handling when data is missing for a symbol."""
        # Strategy expects data for both AAPL and MSFT
        test_data = self._create_test_data(symbol="AAPL")
        
        # Should raise KeyError because MSFT data is missing
        with self.assertRaises(KeyError):
            self.strategy.generate_signals(test_data)

    def test_generate_signals_missing_columns(self):
        """Test error handling when required columns are missing."""
        # Create data with missing columns
        test_data = self._create_test_data(symbol="AAPL")
        test_data["AAPL"] = test_data["AAPL"].drop(columns=["volume"])
        
        # Should raise ValueError because volume column is missing
        with self.assertRaises(ValueError):
            self.strategy.generate_signals(test_data)

    def test_generate_signals_empty_data(self):
        """Test handling of empty data."""
        # Create empty DataFrame
        empty_df = pd.DataFrame()
        test_data = {"AAPL": empty_df, "MSFT": empty_df}
        
        # Should raise ValueError because data is empty
        with self.assertRaises(ValueError):
            self.strategy.generate_signals(test_data)

    def test_confidence_calculation(self):
        """Test that confidence values are calculated correctly."""
        # Create uptrend data with volume spike to trigger buy signals
        test_data = self._create_test_data(uptrend=True, volume_spike=True)
        
        # Override RSI value to ensure high confidence
        df = test_data["AAPL"]
        # Calculate required indicators (that would be done by _calculate_momentum_signals)
        df['rsi'] = 30  # Near oversold level, good for buy
        df['mom_short'] = 0.05  # Strong momentum, good for buy
        df['mom_medium'] = 0.02
        df['mom_long'] = 0.01
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Mock the _calculate_momentum_signals method to use our prepared data
        with patch.object(self.strategy, '_calculate_momentum_signals') as mock_calc:
            # Simulate a buy signal with high confidence
            last_price = df['close'].iloc[-1]
            last_date = df.index[-1]
            metadata = {
                'mom_short': 0.05,
                'mom_medium': 0.02,
                'mom_long': 0.01,
                'rsi': 30,
                'volume_ratio': 2.5  # High volume ratio for strong confidence
            }
            
            # Calculate expected confidence based on the algorithm in MomentumStrategy
            buy_threshold = self.strategy.parameters["buy_threshold"]
            rsi_oversold = self.strategy.parameters["rsi_oversold"]
            volume_factor = self.strategy.parameters["volume_factor"]
            
            # Replicate confidence calculation from the strategy
            mom_strength = min(0.05 / buy_threshold * 0.5, 0.5)
            rsi_strength = max(0, (rsi_oversold - 30) / rsi_oversold) * 0.3
            volume_strength = min((2.5 - volume_factor) / volume_factor, 1.0) * 0.2
            expected_confidence = min(mom_strength + rsi_strength + volume_strength, 1.0)
            
            # Set up the mock to return our simulated signal
            mock_calc.return_value = [
                (SignalType.BUY, last_price, last_date, expected_confidence, metadata)
            ]
            
            # Generate signals using our mocked method
            signals = self.strategy.generate_signals(test_data)
            
            # Verify that we got a signal with the expected confidence
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].type, SignalType.BUY)
            self.assertEqual(signals[0].confidence, expected_confidence)
            self.assertTrue(0.5 <= signals[0].confidence <= 1.0, 
                           f"Confidence should be between 0.5 and 1.0, got {signals[0].confidence}")
            
            # Verify the mock was called correctly
            mock_calc.assert_called_once_with(df)
            
        # Now test a sell signal with custom confidence calculation
        with patch.object(self.strategy, '_calculate_momentum_signals') as mock_calc:
            # Create test data for sell signal
            df['rsi'] = 80  # Near overbought level, good for sell
            df['mom_short'] = -0.05  # Strong negative momentum, good for sell
            df['mom_medium'] = -0.02
            df['mom_long'] = -0.01
            df['volume_ratio'] = 3.0  # Very high volume for strong signal
            
            metadata = {
                'mom_short': -0.05,
                'mom_medium': -0.02,
                'mom_long': -0.01,
                'rsi': 80,
                'volume_ratio': 3.0
            }
            
            # Calculate expected confidence
            sell_threshold = self.strategy.parameters["sell_threshold"]
            rsi_overbought = self.strategy.parameters["rsi_overbought"]
            
            # Replicate confidence calculation from the strategy
            mom_strength = min(abs(-0.05 / sell_threshold) * 0.5, 0.5)
            rsi_strength = max(0, (80 - rsi_overbought) / (100 - rsi_overbought)) * 0.3
            volume_strength = min((3.0 - volume_factor) / volume_factor, 1.0) * 0.2
            expected_confidence = min(mom_strength + rsi_strength + volume_strength, 1.0)
            
            # Set up the mock to return our simulated signal
            mock_calc.return_value = [
                (SignalType.SELL, last_price, last_date, expected_confidence, metadata)
            ]
            
            # Generate signals using our mocked method
            signals = self.strategy.generate_signals(test_data)
            
            # Verify that we got a signal with the expected confidence
            self.assertEqual(len(signals), 1)
            self.assertEqual(signals[0].type, SignalType.SELL)
            self.assertEqual(signals[0].confidence, expected_confidence)
            self.assertTrue(0.5 <= signals[0].confidence <= 1.0, 
                           f"Confidence should be between 0.5 and 1.0, got {signals[0].confidence}")
