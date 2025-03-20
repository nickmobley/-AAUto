"""
Unit tests for the base Strategy class and associated components.

This module tests the functionality of the base Strategy class, Signal class,
and SignalType enum to ensure they work as expected.
"""
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from src.strategies.base import Strategy, Signal, SignalType


class TestSignal(unittest.TestCase):
    """Tests for the Signal dataclass."""

    def test_valid_signal_creation(self):
        """Test that a valid Signal object can be created."""
        signal = Signal(
            type=SignalType.BUY,
            symbol="AAPL",
            timestamp=datetime.now(),
            price=150.0,
            confidence=0.85,
            metadata={"reason": "price_momentum"}
        )
        
        self.assertEqual(signal.type, SignalType.BUY)
        self.assertEqual(signal.symbol, "AAPL")
        self.assertEqual(signal.price, 150.0)
        self.assertEqual(signal.confidence, 0.85)
        self.assertEqual(signal.metadata, {"reason": "price_momentum"})

    def test_signal_default_metadata(self):
        """Test that metadata defaults to an empty dict if None."""
        signal = Signal(
            type=SignalType.SELL,
            symbol="MSFT",
            timestamp=datetime.now(),
            price=300.0,
            confidence=0.7,
        )
        
        self.assertDictEqual(signal.metadata, {})

    def test_invalid_confidence_high(self):
        """Test that an invalid confidence value (>1.0) raises a ValueError."""
        with self.assertRaises(ValueError):
            Signal(
                type=SignalType.BUY,
                symbol="TSLA",
                timestamp=datetime.now(),
                price=900.0,
                confidence=1.5,
            )

    def test_invalid_confidence_low(self):
        """Test that an invalid confidence value (<0.0) raises a ValueError."""
        with self.assertRaises(ValueError):
            Signal(
                type=SignalType.SELL,
                symbol="AMZN",
                timestamp=datetime.now(),
                price=3000.0,
                confidence=-0.2,
            )


class ConcreteStrategy(Strategy):
    """Concrete implementation of Strategy for testing purposes."""
    
    def generate_signals(self, data):
        """Generate test signals based on simple price comparison."""
        signals = []
        for symbol, df in data.items():
            if symbol not in self.symbols:
                continue
                
            if len(df) < 2:
                continue
                
            current_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2]
            
            if current_price > previous_price:
                signals.append(Signal(
                    type=SignalType.BUY,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=0.8,
                    metadata={"price_change": current_price - previous_price}
                ))
            elif current_price < previous_price:
                signals.append(Signal(
                    type=SignalType.SELL,
                    symbol=symbol,
                    timestamp=datetime.now(),
                    price=current_price,
                    confidence=0.7,
                    metadata={"price_change": current_price - previous_price}
                ))
        
        return signals
    
    def get_required_data(self):
        """Return test data requirements."""
        return {
            "timeframe": "1d",
            "history_bars": 10,
            "indicators": ["sma_20", "rsi_14"]
        }
    
    def _validate_parameters(self):
        """Validate test parameters."""
        if "threshold" in self.parameters:
            threshold = self.parameters["threshold"]
            if not isinstance(threshold, (int, float)):
                self._validation_errors.append("threshold must be a number")
            elif threshold <= 0:
                self._validation_errors.append("threshold must be positive")
        
        if "max_positions" in self.parameters:
            max_positions = self.parameters["max_positions"]
            if not isinstance(max_positions, int):
                self._validation_errors.append("max_positions must be an integer")
            elif max_positions <= 0:
                self._validation_errors.append("max_positions must be positive")


class TestStrategy(unittest.TestCase):
    """Tests for the base Strategy class."""

    def setUp(self):
        """Set up test fixtures."""
        self.symbols = ["AAPL", "MSFT", "GOOGL"]
        self.valid_params = {
            "threshold": 1.5,
            "max_positions": 5
        }
        self.strategy = ConcreteStrategy(
            name="test_strategy",
            symbols=self.symbols,
            parameters=self.valid_params
        )

    def test_initialization(self):
        """Test strategy initialization with valid parameters."""
        self.assertEqual(self.strategy.name, "test_strategy")
        self.assertEqual(self.strategy.symbols, self.symbols)
        self.assertEqual(self.strategy.parameters, self.valid_params)
        self.assertTrue(self.strategy.is_active)
        self.assertEqual(self.strategy._validation_errors, [])

    def test_initialization_without_parameters(self):
        """Test strategy initialization without parameters."""
        strategy = ConcreteStrategy(name="test_strategy", symbols=self.symbols)
        self.assertEqual(strategy.parameters, {})
        self.assertTrue(strategy.is_valid())

    def test_update_parameters_valid(self):
        """Test updating parameters with valid values."""
        new_params = {
            "threshold": 2.0,
            "new_param": "value"
        }
        
        self.strategy.update_parameters(new_params)
        
        # Original params should be preserved if not overwritten
        self.assertEqual(self.strategy.parameters["max_positions"], 5)
        # New params should be added
        self.assertEqual(self.strategy.parameters["new_param"], "value")
        # Updated params should have new value
        self.assertEqual(self.strategy.parameters["threshold"], 2.0)

    def test_update_parameters_invalid(self):
        """Test updating parameters with invalid values."""
        invalid_params = {
            "threshold": -1.0  # Invalid as per validation logic
        }
        
        with self.assertRaises(ValueError):
            self.strategy.update_parameters(invalid_params)
            
        # Original parameters should be preserved
        self.assertEqual(self.strategy.parameters["threshold"], 1.5)

    def test_validation_methods(self):
        """Test is_valid and get_validation_errors methods."""
        # Strategy should be valid after initialization with valid params
        self.assertTrue(self.strategy.is_valid())
        self.assertEqual(self.strategy.get_validation_errors(), [])
        
        # Manually set validation errors to test error reporting
        self.strategy._validation_errors = ["Error 1", "Error 2"]
        
        self.assertFalse(self.strategy.is_valid())
        self.assertEqual(self.strategy.get_validation_errors(), ["Error 1", "Error 2"])

    def test_generate_signals(self):
        """Test signal generation with sample data."""
        # Create sample data for testing
        data = {
            "AAPL": pd.DataFrame({
                'open': [145.0, 147.0, 148.0],
                'high': [148.0, 150.0, 152.0],
                'low': [144.0, 146.0, 147.0],
                'close': [146.0, 148.0, 150.0],
                'volume': [1000000, 1200000, 1100000]
            }),
            "MSFT": pd.DataFrame({
                'open': [290.0, 288.0, 285.0],
                'high': [295.0, 292.0, 288.0],
                'low': [288.0, 285.0, 280.0],
                'close': [292.0, 290.0, 282.0],
                'volume': [800000, 750000, 900000]
            })
        }
        
        signals = self.strategy.generate_signals(data)
        
        # Should have 2 signals: BUY for AAPL (price increased) and SELL for MSFT (price decreased)
        self.assertEqual(len(signals), 2)
        
        # Verify signals are correct
        aapl_signal = next((s for s in signals if s.symbol == "AAPL"), None)
        msft_signal = next((s for s in signals if s.symbol == "MSFT"), None)
        
        self.assertIsNotNone(aapl_signal)
        self.assertIsNotNone(msft_signal)
        
        self.assertEqual(aapl_signal.type, SignalType.BUY)
        self.assertEqual(msft_signal.type, SignalType.SELL)
        
        self.assertEqual(aapl_signal.price, 150.0)
        self.assertEqual(msft_signal.price, 282.0)

    def test_get_required_data(self):
        """Test that get_required_data returns expected values."""
        requirements = self.strategy.get_required_data()
        
        self.assertEqual(requirements["timeframe"], "1d")
        self.assertEqual(requirements["history_bars"], 10)
        self.assertEqual(requirements["indicators"], ["sma_20", "rsi_14"])

    def test_missing_symbol_data(self):
        """Test signal generation when data for some symbols is missing."""
        # Data only for one symbol
        data = {
            "AAPL": pd.DataFrame({
                'open': [145.0, 147.0],
                'high': [148.0, 150.0],
                'low': [144.0, 146.0],
                'close': [146.0, 148.0],
                'volume': [1000000, 1200000]
            })
        }
        
        signals = self.strategy.generate_signals(data)
        
        # Should have 1 signal for AAPL
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].symbol, "AAPL")

    def test_string_representations(self):
        """Test __str__ and __repr__ methods."""
        expected_str = "ConcreteStrategy(name=test_strategy, symbols=['AAPL', 'MSFT', 'GOOGL'])"
        expected_repr = "ConcreteStrategy(name=test_strategy, symbols=['AAPL', 'MSFT', 'GOOGL'], parameters={'threshold': 1.5, 'max_positions': 5})"
        
        self.assertEqual(str(self.strategy), expected_str)
        self.assertEqual(repr(self.strategy), expected_repr)


if __name__ == "__main__":
    unittest.main()

