"""
Unit tests for the MeanReversionStrategy class.

This module contains comprehensive tests for the MeanReversionStrategy implementation,
including initialization, parameter validation, signal generation, and error handling.
"""
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.strategies.base import SignalType
from src.strategies.mean_reversion import MeanReversionStrategy


class TestMeanReversionStrategy(unittest.TestCase):
    """Test suite for the MeanReversionStrategy class."""

    def setUp(self):
        """Set up test fixtures before each test."""
        self.symbols = ["AAPL", "MSFT"]
        self.strategy_name = "test_mean_reversion"
        self.default_params = MeanReversionStrategy.DEFAULT_PARAMETERS.copy()
        self.strategy = MeanReversionStrategy(
            name=self.strategy_name,
            symbols=self.symbols
        )

    def test_initialization_with_default_parameters(self):
        """Test initialization with default parameters."""
        strategy = MeanReversionStrategy(
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
            "lookback_period": 15,
            "entry_std_dev": 2.5,
            "exit_std_dev": 0.3,
            "mean_type": "ema",
            "min_history": 40,
        }
        
        strategy = MeanReversionStrategy(
            name=self.strategy_name,
            symbols=self.symbols,
            parameters=custom_params
        )
        
        # Verify custom parameters were merged with defaults
        self.assertEqual(strategy.parameters["lookback_period"], custom_params["lookback_period"])
        self.assertEqual(strategy.parameters["entry_std_dev"], custom_params["entry_std_dev"])
        self.assertEqual(strategy.parameters["exit_std_dev"], custom_params["exit_std_dev"])
        self.assertEqual(strategy.parameters["mean_type"], custom_params["mean_type"])
        self.assertEqual(strategy.parameters["min_history"], custom_params["min_history"])
        
        # Default parameters not overridden should still be present
        self.assertEqual(strategy.parameters["use_atr"], self.default_params["use_atr"])
        self.assertTrue(strategy.is_valid())

    def test_parameter_validation_success(self):
        """Test parameter validation with valid parameters."""
        # Using the strategy created in setUp which uses default parameters
        self.assertTrue(self.strategy.is_valid())
        self.assertEqual(len(self.strategy.get_validation_errors()), 0)

    def test_parameter_validation_failure_invalid_periods(self):
        """Test parameter validation with invalid period values."""
        invalid_params = {
            "lookback_period": -1,
            "min_history": 0,
            "bollinger_period": "invalid",
        }
        
        with self.assertRaises(ValueError):
            MeanReversionStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_std_dev_values(self):
        """Test parameter validation with invalid standard deviation values."""
        invalid_params = {
            "entry_std_dev": -0.5,  # Should be positive
            "exit_std_dev": 0.0,    # Should be positive
        }
        
        with self.assertRaises(ValueError):
            MeanReversionStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_std_dev_order(self):
        """Test parameter validation with exit_std_dev > entry_std_dev."""
        invalid_params = {
            "entry_std_dev": 1.0,  # Should be greater than exit_std_dev
            "exit_std_dev": 2.0,   # Should be less than entry_std_dev
        }
        
        with self.assertRaises(ValueError):
            MeanReversionStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_mean_type(self):
        """Test parameter validation with invalid mean type."""
        invalid_params = {
            "mean_type": "invalid_type",  # Should be one of sma, ema, bollinger
        }
        
        with self.assertRaises(ValueError):
            MeanReversionStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_parameter_validation_failure_atr_multiplier(self):
        """Test parameter validation with invalid ATR multiplier."""
        invalid_params = {
            "use_atr": True,
            "atr_multiplier": -1.0,  # Should be positive
        }
        
        with self.assertRaises(ValueError):
            MeanReversionStrategy(
                name=self.strategy_name,
                symbols=self.symbols,
                parameters=invalid_params
            )

    def test_get_required_data(self):
        """Test the get_required_data method returns correct data specifications."""
        data_req = self.strategy.get_required_data()
        
        self.assertEqual(data_req["timeframe"], "1d")
        max_period = max(
            self.default_params["min_history"],
            self.default_params["lookback_period"],
            self.default_params["bollinger_period"],
            self.default_params["atr_period"]
        )
        self.assertTrue(data_req["bar_count"] >= max_period * 3)
        self.assertIn("open", data_req["columns"])
        self.assertIn("high", data_req["columns"])
        self.assertIn("low", data_req["columns"])
        self.assertIn("close", data_req["columns"])
        self.assertIn("volume", data_req["columns"])
        self.assertIn("atr", data_req["indicators"])
        self.assertIn("bollinger", data_req["indicators"])
        self.assertIn("volume_sma", data_req["indicators"])

    def test_calculate_atr(self):
        """Test the ATR calculation method."""
        # Create a simple price series for testing
        data = pd.DataFrame({
            "high": [110, 112, 108, 115, 113, 117, 114, 119, 116, 121],
            "low":  [100, 102, 98, 105, 103, 107, 104, 109, 106, 111],
            "close": [105, 106, 103, 110, 108, 112, 110, 115, 112, 118]
        })
        
        atr = self.strategy._calculate_atr(data, period=5)
        
        # First value should be NaN due to the shift in close price
        self.assertTrue(pd.isna(atr.iloc[0]))

