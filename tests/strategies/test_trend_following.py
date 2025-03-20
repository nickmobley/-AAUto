import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.strategies.trend_following import TrendFollowingStrategy


class TestTrendFollowingStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test data for the trend following strategy tests."""
        self.api_client = MagicMock()
        
        # Create sample price data
        self.test_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=100),
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.randint(1000, 10000, 100),
        })
        
        # Set specific trend patterns in the data
        # Uptrend section (indices 20-40)
        for i in range(20, 41):
            self.test_data.loc[i, 'close'] = 90 + (i - 20) * 0.5
            
        # Downtrend section (indices 60-80)
        for i in range(60, 81):
            self.test_data.loc[i, 'close'] = 110 - (i - 60) * 0.5
            
        # Sideways section (indices 40-60)
        for i in range(40, 61):
            self.test_data.loc[i, 'close'] = 100 + np.sin(i-40) * 2
        
        # Configure API client mock to return test data
        self.api_client.get_historical_data.return_value = self.test_data
        
        # Create default strategy instance
        self.strategy = TrendFollowingStrategy(
            symbol="AAPL",
            api_client=self.api_client,
            short_period=10,
            long_period=30,
            adx_period=14,
            adx_threshold=25,
            volume_factor=1.5
        )

    def test_initialization(self):
        """Test strategy initialization with valid parameters."""
        strategy = TrendFollowingStrategy(
            symbol="AAPL",
            api_client=self.api_client,
            short_period=10,
            long_period=30,
            adx_period=14,
            adx_threshold=25,
            volume_factor=1.5
        )
        
        self.assertEqual(strategy.symbol, "AAPL")
        self.assertEqual(strategy.short_period, 10)
        self.assertEqual(strategy.long_period, 30)
        self.assertEqual(strategy.adx_period, 14)
        self.assertEqual(strategy.adx_threshold, 25)
        self.assertEqual(strategy.volume_factor, 1.5)

    def test_initialization_with_invalid_parameters(self):
        """Test strategy initialization with invalid parameters."""
        # Test invalid short period
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(
                symbol="AAPL",
                api_client=self.api_client,
                short_period=0,  # Invalid value
                long_period=30,
                adx_period=14,
                adx_threshold=25,
                volume_factor=1.5
            )
        
        # Test invalid long period
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(
                symbol="AAPL",
                api_client=self.api_client,
                short_period=10,
                long_period=5,  # Should be greater than short_period
                adx_period=14,
                adx_threshold=25,
                volume_factor=1.5
            )
        
        # Test invalid ADX threshold
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(
                symbol="AAPL",
                api_client=self.api_client,
                short_period=10,
                long_period=30,
                adx_period=14,
                adx_threshold=-5,  # Invalid negative value
                volume_factor=1.5
            )
        
        # Test invalid volume factor
        with self.assertRaises(ValueError):
            TrendFollowingStrategy(
                symbol="AAPL",
                api_client=self.api_client,
                short_period=10,
                long_period=30,
                adx_period=14,
                adx_threshold=25,
                volume_factor=0  # Invalid value
            )

    def test_calculate_moving_averages(self):
        """Test calculation of short and long moving averages."""
        self.strategy._load_data()
        short_ma, long_ma = self.strategy._calculate_moving_averages(self.test_data)
        
        # Check if moving averages were calculated
        self.assertIsNotNone(short_ma)
        self.assertIsNotNone(long_ma)
        
        # Check if lengths are correct
        self.assertEqual(len(short_ma), len(self.test_data))
        self.assertEqual(len(long_ma), len(self.test_data))
        
        # Check if NaN values are present in the beginning (as expected)
        self.assertTrue(np.isnan(short_ma.iloc[0]))
        self.assertTrue(np.isnan(long_ma.iloc[0]))
        
        # Check if valid values start after the respective periods
        self.assertFalse(np.isnan(short_ma.iloc[self.strategy.short_period]))
        self.assertFalse(np.isnan(long_ma.iloc[self.strategy.long_period]))
        
        # Test the actual calculation of moving averages
        sample_index = 50  # Choose a point in the middle of the data
        expected_short_ma = self.test_data['close'].iloc[sample_index-self.strategy.short_period+1:sample_index+1].mean()
        self.assertAlmostEqual(short_ma.iloc[sample_index], expected_short_ma, places=5)

    def test_calculate_adx(self):
        """Test ADX (Average Directional Index) calculation."""
        self.strategy._load_data()
        adx = self.strategy._calculate_adx(self.test_data)
        
        # Check if ADX was calculated
        self.assertIsNotNone(adx)
        
        # Check if length is correct
        self.assertEqual(len(adx), len(self.test_data))
        
        # ADX should be NaN for the first few periods
        self.assertTrue(np.isnan(adx.iloc[0]))
        
        # Should have valid values after the ADX period
        valid_index = self.strategy.adx_period * 2  # ADX typically needs 2*period points to be fully valid
        self.assertFalse(np.isnan(adx.iloc[valid_index]))
        
        # ADX should be between 0 and 100
        valid_adx = adx.dropna()
        self.assertTrue((valid_adx >= 0).all() and (valid_adx <= 100).all())

    def test_detect_crossovers(self):
        """Test detection of moving average crossovers."""
        self.strategy._load_data()
        short_ma, long_ma = self.strategy._calculate_moving_averages(self.test_data)
        
        # Create a specific crossover pattern
        test_short_ma = pd.Series([95, 96, 98, 100, 102, 103])
        test_long_ma = pd.Series([100, 100, 100, 100, 100, 100])
        
        # Test bullish crossover (short crosses above long)
        crossovers = self.strategy._detect_crossovers(
            test_short_ma, 
            test_long_ma
        )
        
        # Check if crossover is detected at the right index
        self.assertEqual(len(crossovers), 6)
        self.assertEqual(crossovers.iloc[2], 0)  # No crossover
        self.assertEqual(crossovers.iloc[3], 1)  # Bullish crossover
        
        # Test bearish crossover
        test_short_ma = pd.Series([105, 103, 101, 99, 97, 95])
        crossovers = self.strategy._detect_crossovers(
            test_short_ma, 
            test_long_ma
        )
        
        self.assertEqual(crossovers.iloc[2], 0)  # No crossover
        self.assertEqual(crossovers.iloc[3], -1)  # Bearish crossover

    def test_calculate_volume_signals(self):
        """Test volume confirmation signal calculation."""
        self.strategy._load_data()
        
        # Create test data with volume spikes
        test_data = self.test_data.copy()
        
        # Create volume spike at index 50
        avg_volume = test_data['volume'].iloc[45:50].mean()
        test_data.loc[50, 'volume'] = avg_volume * self.strategy.volume_factor * 1.2
        
        volume_signals = self.strategy._calculate_volume_signals(test_data)
        
        # Check if volume signals were calculated
        self.assertIsNotNone(volume_signals)
        
        # Check if length is correct
        self.assertEqual(len(volume_signals), len(test_data))
        
        # Should detect the volume spike we created
        self.assertTrue(volume_signals.iloc[50])
        
        # Random index without volume spike should be False
        self.assertFalse(volume_signals.iloc[30])

    def test_generate_signals_in_uptrend(self):
        """Test signal generation in an uptrend market."""
        # Create a clear uptrend in the test data
        test_data = self.test_data.copy()
        for i in range(50):
            test_data.loc[i, 'close'] = 100 + i
            
        # Mock the API client to return our uptrend data
        self.api_client.get_historical_data.return_value = test_data
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # In a strong uptrend, we expect buy signals
        self.assertEqual(signals[-1]['signal'], 'buy')
        self.assertTrue(signals[-1]['confidence'] > 0.5)

    def test_generate_signals_in_downtrend(self):
        """Test signal generation in a downtrend market."""
        # Create a clear downtrend in the test data
        test_data = self.test_data.copy()
        for i in range(50):
            test_data.loc[i, 'close'] = 150 - i
            
        # Mock the API client to return our downtrend data
        self.api_client.get_historical_data.return_value = test_data
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # In a strong downtrend, we expect sell signals
        self.assertEqual(signals[-1]['signal'], 'sell')
        self.assertTrue(signals[-1]['confidence'] > 0.5)

    def test_generate_signals_in_sideways_market(self):
        """Test signal generation in a sideways (low trend) market."""
        # Create a sideways pattern in the test data
        test_data = self.test_data.copy()
        for i in range(50):
            test_data.loc[i, 'close'] = 100 + np.sin(i/5) * 3
            
        # Mock the API client to return our sideways data
        self.api_client.get_historical_data.return_value = test_data
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        # In a sideways market with low trend strength, we expect no signal or low confidence
        self.assertTrue(
            signals[-1]['signal'] == 'hold' or 
            signals[-1]['confidence'] < 0.3
        )

    def test_signal_confidence_calculation(self):
        """Test the confidence calculation for trading signals."""
        # Mock internal methods to return controlled values
        with patch.object(self.strategy, '_calculate_moving_averages') as mock_ma:
            with patch.object(self.strategy, '_calculate_adx') as mock_adx:
                with patch.object(self.strategy, '_detect_crossovers') as mock_crossovers:
                    with patch.object(self.strategy, '_calculate_volume_signals') as mock_volume:
                        # Set up mock return values
                        mock_ma.return_value = (
                            pd.Series([90, 95, 100, 105, 110]),  # short MA
                            pd.Series([80, 85, 90, 95, 100])     # long MA
                        )
                        mock_adx.return_value = pd.Series([10, 20, 30, 40, 50])
                        mock_crossovers.return_value = pd.Series([0, 0, 1, 0, 0])  # bullish crossover at index 2
                        mock_volume.return_value = pd.Series([False, False, True, False, False])  # volume spike at crossover
                        
                        # Generate signals
                        self.strategy._load_data()
                        signals = self.strategy.generate_signals()
                        
                        # Check the signal at the crossover point
                        self.assertEqual(signals[2]['signal'], 'buy')
                        
                        # Check confidence calculation
                        # With ADX of 30 (medium trend strength) and volume confirmation
                        # the confidence should be in the medium-high range
                        self.assertTrue(0.6 <= signals[2]['confidence'] <= 0.9)

    def test_parameter_sensitivity(self):
        """Test how different parameters affect signal generation."""
        # Test with longer periods (less sensitive)
        less_sensitive = TrendFollowingStrategy(
            symbol="AAPL",
            api_client=self.api_client,
            short_period=20,    # Longer short period
            long_period=50,     # Longer long period
            adx_period=28,      # Longer ADX period
            adx_threshold=30,   # Higher ADX threshold
            volume_factor=2.0   # Higher volume threshold
        )
        
        # Test with shorter periods (more sensitive)
        more_sensitive = TrendFollowingStrategy(
            symbol="AAPL",
            api_client=self.api_client,
            short_period=5,     # Shorter short period
            long_period=15,     # Shorter long period
            adx_period=7,       # Shorter ADX period
            adx_threshold=20,   # Lower ADX threshold
            volume_factor=1.2   # Lower volume threshold
        )
        
        # Create test data with a moderate trend
        test_data = self.test_data.copy()
        self.api_client.get_historical_data.return_value = test_data
        
        # Generate signals with both strategies
        less_sensitive_signals = less_sensitive.generate_signals()
        more_sensitive_signals = more_sensitive.generate_signals()
        
        # The more sensitive strategy should generate more signals
        less_sensitive_count = sum(1 for s in less_sensitive_signals if s['signal'] != 'hold')
        more_sensitive_count = sum(1 for s in more_sensitive_signals if s['signal'] != 'hold')
        
        # The more sensitive strategy should generate more trade signals
        self.assertGreater(more_sensitive_count, less_sensitive_count)
        
        # The more sensitive strategy should also have different confidence levels
        # Get average confidence for non-hold signals
        less_sensitive_confidence = np.mean([s['confidence'] for s in less_sensitive_signals if s['signal'] != 'hold']) if less_sensitive_count > 0 else 0
        more_sensitive_confidence = np.mean([s['confidence'] for s in more_sensitive_signals if s['signal'] != 'hold']) if more_sensitive_count > 0 else 0
        
        # The confidence levels should be different
        self.assertNotEqual(round(less_sensitive_confidence, 4), round(more_sensitive_confidence, 4))

    def test_api_failure_handling(self):
        """Test the strategy's response to API failures."""
        # Mock API to raise an exception
        self.api_client.get_historical_data.side_effect = Exception("API Connection Error")
        
        # The strategy should handle API errors gracefully
        with self.assertRaises(Exception) as context:
            self.strategy.generate_signals()
            
        # Check that the error is properly propagated with additional context
        self.assertIn("Failed to generate signals", str(context.exception))
        
    def test_empty_data_handling(self):
        """Test strategy behavior with empty data."""
        # Mock API to return empty DataFrame
        self.api_client.get_historical_data.return_value = pd.DataFrame()
        
        # Should raise a ValueError with appropriate message
        with self.assertRaises(ValueError) as context:
            self.strategy.generate_signals()
            
        # Check error message
        self.assertIn("Insufficient data", str(context.exception))
        
    def test_insufficient_data_handling(self):
        """Test strategy behavior with insufficient historical data."""
        # Create data that's shorter than required periods
        short_data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        # Mock API to return this short data
        self.api_client.get_historical_data.return_value = short_data
        
        # Should raise a ValueError with appropriate message
        with self.assertRaises(ValueError) as context:
            self.strategy.generate_signals()
            
        # Check error message
        self.assertIn("requires at least", str(context.exception))
        
    def test_nan_values_handling(self):
        """Test how the strategy handles NaN values in the data."""
        # Create data with NaN values
        data_with_nans = self.test_data.copy()
        data_with_nans.loc[30:35, 'close'] = np.nan
        
        # Mock API to return data with NaNs
        self.api_client.get_historical_data.return_value = data_with_nans
        
        # Generate signals - should handle NaNs gracefully
        signals = self.strategy.generate_signals()
        
        # Check that signals were generated despite NaNs
        self.assertIsNotNone(signals)
        self.assertTrue(len(signals) > 0)
        
        # Indices with NaNs should have 'hold' signals or appropriate confidence
        for i in range(30, 36):
            if i < len(signals):
                self.assertTrue(signals[i]['signal'] == 'hold' or signals[i]['confidence'] < 0.2)

    def test_extreme_market_conditions(self):
        """Test strategy behavior under extreme market conditions like crashes or rallies."""
        # Create market crash scenario
        crash_data = self.test_data.copy()
        for i in range(50, 60):
            crash_data.loc[i, 'close'] = crash_data.loc[i-1, 'close'] * 0.9  # 10% daily drop
            
        self.api_client.get_historical_data.return_value = crash_data
        crash_signals = self.strategy.generate_signals()
        
        # Create market rally scenario
        rally_data = self.test_data.copy()
        for i in range(50, 60):
            rally_data.loc[i, 'close'] = rally_data.loc[i-1, 'close'] * 1.1  # 10% daily gain
            
        self.api_client.get_historical_data.return_value = rally_data
        rally_signals = self.strategy.generate_signals()
        
        # Check that strategy generates strong signals in extreme conditions
        self.assertEqual(crash_signals[59]['signal'], 'sell')
        self.assertTrue(crash_signals[59]['confidence'] > 0.7)
        
        self.assertEqual(rally_signals[59]['signal'], 'buy')
        self.assertTrue(rally_signals[59]['confidence'] > 0.7)
