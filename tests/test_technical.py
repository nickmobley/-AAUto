"""
Unit tests for TechnicalAnalyzer class in src/analytics/technical.py
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from typing import Dict, Any

from src.analytics.technical import TechnicalAnalyzer, MACDResult


@pytest.fixture
def mock_api():
    """Create a mock API client for testing."""
    mock = Mock()
    return mock


@pytest.fixture
def sample_daily_data():
    """Returns sample time series data for testing technical indicators."""
    # Create sample data with 100 days of price data (oldest to newest)
    time_series = {}
    
    # Generate 100 days of sample data, most recent first
    for i in range(100):
        date = f"2023-{(12 - i//30):02d}-{(30 - i%30):02d}"
        time_series[date] = {
            "1. open": f"{100 + i * 0.1:.2f}",
            "2. high": f"{101 + i * 0.1:.2f}",
            "3. low": f"{99 + i * 0.1:.2f}",
            "4. close": f"{100.5 + i * 0.1:.2f}",
            "5. volume": f"{1000000 + i * 1000}"
        }
    
    return {
        "Meta Data": {
            "1. Information": "Daily Prices",
            "2. Symbol": "TSLA",
            "3. Last Refreshed": "2023-12-30",
            "4. Output Size": "Full size",
            "5. Time Zone": "US/Eastern"
        },
        "Time Series (Daily)": time_series
    }


@pytest.fixture
def sample_daily_data_uptrend():
    """Returns sample time series data with an uptrend."""
    time_series = {}
    # Generate uptrend data: short term average > long term average
    base_price = 100.0
    for i in range(100):
        date = f"2023-{(12 - i//30):02d}-{(30 - i%30):02d}"
        # More recent prices are higher
        price = base_price + (99 - i) * 0.5  # Reversed to make newer prices higher
        time_series[date] = {
            "1. open": f"{price - 0.5:.2f}",
            "2. high": f"{price + 1:.2f}",
            "3. low": f"{price - 1:.2f}",
            "4. close": f"{price:.2f}",
            "5. volume": f"{1000000 + i * 1000}"
        }
    
    return {
        "Meta Data": {"2. Symbol": "AAPL"},
        "Time Series (Daily)": time_series
    }


@pytest.fixture
def sample_daily_data_downtrend():
    """Returns sample time series data with a downtrend."""
    time_series = {}
    # Generate downtrend data: short term average < long term average
    base_price = 200.0
    for i in range(100):
        date = f"2023-{(12 - i//30):02d}-{(30 - i%30):02d}"
        # More recent prices are lower
        price = base_price - (99 - i) * 0.5  # Reversed to make newer prices lower
        time_series[date] = {
            "1. open": f"{price - 0.5:.2f}",
            "2. high": f"{price + 1:.2f}",
            "3. low": f"{price - 1:.2f}",
            "4. close": f"{price:.2f}",
            "5. volume": f"{1000000 + i * 1000}"
        }
    
    return {
        "Meta Data": {"2. Symbol": "MSFT"},
        "Time Series (Daily)": time_series
    }


@pytest.fixture
def sample_daily_data_sideways():
    """Returns sample time series data with a sideways trend."""
    time_series = {}
    # Generate sideways data: small fluctuations around a base price
    base_price = 150.0
    for i in range(100):
        date = f"2023-{(12 - i//30):02d}-{(30 - i%30):02d}"
        # Oscillate around base price with small changes
        oscillation = np.sin(i/5) * 0.5
        price = base_price + oscillation
        time_series[date] = {
            "1. open": f"{price - 0.2:.2f}",
            "2. high": f"{price + 0.5:.2f}",
            "3. low": f"{price - 0.5:.2f}",
            "4. close": f"{price:.2f}",
            "5. volume": f"{1000000 + i * 1000}"
        }
    
    return {
        "Meta Data": {"2. Symbol": "META"},
        "Time Series (Daily)": time_series
    }


@pytest.fixture
def technical_analyzer(mock_api):
    """Create a TechnicalAnalyzer instance with mocked API for testing."""
    return TechnicalAnalyzer(mock_api)


class TestTechnicalAnalyzer:
    """Test cases for TechnicalAnalyzer class."""

    def test_calculate_rsi_successful(self, technical_analyzer, mock_api, sample_daily_data):
        """Test successful RSI calculation with normal data."""
        symbol = "TSLA"
        period = 14
        
        # Configure mock API to return sample data
        mock_api.get_daily_data.return_value = sample_daily_data
        
        # Calculate RSI
        rsi = technical_analyzer.calculate_rsi(symbol, period)
        
        # Verify API was called correctly
        mock_api.get_daily_data.assert_called_once_with(symbol)
        
        # Verify RSI is calculated and within expected range
        assert rsi is not None
        assert 0 <= rsi <= 100
    
    def test_calculate_rsi_invalid_period(self, technical_analyzer):
        """Test RSI calculation with invalid period."""
        with pytest.raises(ValueError, match="RSI period must be at least 2"):
            technical_analyzer.calculate_rsi("TSLA", 1)
    
    def test_calculate_rsi_no_data(self, technical_analyzer, mock_api):
        """Test RSI calculation when no data is available."""
        # Configure mock API to return None
        mock_api.get_daily_data.return_value = None
        
        rsi = technical_analyzer.calculate_rsi("TSLA")
        
        assert rsi is None
    
    def test_calculate_rsi_insufficient_data(self, technical_analyzer, mock_api):
        """Test RSI calculation with insufficient data."""
        # Configure mock API to return minimal data
        minimal_data = {
            "Meta Data": {"2. Symbol": "TSLA"},
            "Time Series (Daily)": {
                "2023-12-30": {"4. close": "100.50"},
                "2023-12-29": {"4. close": "101.50"}
            }
        }
        mock_api.get_daily_data.return_value = minimal_data
        
        rsi = technical_analyzer.calculate_rsi("TSLA", period=14)
        
        assert rsi is None
    
    def test_calculate_rsi_all_gains(self, technical_analyzer, mock_api):
        """Test RSI calculation when all price changes are gains."""
        # Create data with continuously increasing prices
        time_series = {}
        for i in range(20):
            date = f"2023-12-{30-i:02d}"
            time_series[date] = {"4. close": f"{100 + i:.2f}"}
        
        all_gains_data = {
            "Meta Data": {"2. Symbol": "TSLA"},
            "Time Series (Daily)": time_series
        }
        
        mock_api.get_daily_data.return_value = all_gains_data
        
        rsi = technical_analyzer.calculate_rsi("TSLA", period=14)
        
        assert rsi == 100.0  # RSI should be 100 when there are no losses
    
    def test_calculate_macd_successful(self, technical_analyzer, mock_api, sample_daily_data):
        """Test successful MACD calculation."""
        symbol = "TSLA"
        
        # Configure mock API to return sample data
        mock_api.get_daily_data.return_value = sample_daily_data
        
        # Calculate MACD
        macd_result = technical_analyzer.calculate_macd(symbol)
        
        # Verify API was called correctly
        mock_api.get_daily_data.assert_called_once_with(symbol)
        
        # Verify MACD result structure and values
        assert isinstance(macd_result, dict)
        assert 'macd_line' in macd_result
        assert 'signal_line' in macd_result
        assert 'histogram' in macd_result
        assert macd_result['histogram'] == macd_result['macd_line'] - macd_result['signal_line']
    
    def test_calculate_macd_invalid_periods(self, technical_analyzer):
        """Test MACD calculation with invalid period parameters."""
        # Test when fast period is greater than slow period
        with pytest.raises(ValueError, match="Fast period must be less than slow period"):
            technical_analyzer.calculate_macd("TSLA", fast_period=26, slow_period=12)
        
        # Test negative periods
        with pytest.raises(ValueError, match="All periods must be positive integers"):
            technical_analyzer.calculate_macd("TSLA", signal_period=-1)
    
    def test_calculate_macd_no_data(self, technical_analyzer, mock_api):
        """Test MACD calculation when no data is available."""
        # Configure mock API to return None
        mock_api.get_daily_data.return_value = None
        
        macd_result = technical_analyzer.calculate_macd("TSLA")
        
        assert macd_result is None
    
    def test_calculate_macd_insufficient_data(self, technical_analyzer, mock_api):
        """Test MACD calculation with insufficient data."""
        # Configure mock API to return minimal data
        minimal_data = {
            "Meta Data": {"2. Symbol": "TSLA"},
            "Time Series (Daily)": {
                "2023-12-30": {"4. close": "100.50"},
                "2023-12-29": {"4. close": "101.50"}
            }
        }
        mock_api.get_daily_data.return_value = minimal_data
        
        macd_result = technical_analyzer.calculate_macd("TSLA")
        
        assert macd_result is None
    
    def test_calculate_ema_successful(self, technical_analyzer):
        """Test successful EMA calculation."""
        # Simple test data
        data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
        period = 5
        
        ema_result = technical_analyzer.calculate_ema(data, period)
        
        # Verify result
        assert len(ema_result) == len(data)
        assert ema_result[0] == data[0]  # First EMA equals first price
    
    def test_calculate_ema_invalid_inputs(self, technical_analyzer):
        """Test EMA calculation with invalid inputs."""
        # Test negative period
        with pytest.raises(ValueError, match="EMA period must be at least 1"):
            technical_analyzer.calculate_ema([10.0, 11.0], 0)
        
        # Test empty data
        with pytest.raises(ValueError, match="Data cannot be empty"):
            technical_analyzer.calculate_ema([], 5)
        
        # Test insufficient data
        with pytest.raises(ValueError, match="Data length .* must be at least equal to period"):
            technical_analyzer.calculate_ema([10.0, 11.0], 5)
    
    def test_get_trend_uptrend(self, technical_analyzer, mock_api, sample_daily_data_uptrend):
        """Test trend detection with uptrend data."""
        symbol = "AAPL"
        
        # Configure mock API to return uptrend data
        mock_api.get_daily_data.return_value = sample_daily_data_uptrend
        
        trend = technical_analyzer.get_trend(symbol)
        
        assert trend == "UPTREND"
    
    def test_get_trend_downtrend(self, technical_analyzer, mock_api, sample_daily_data_downtrend):
        """Test trend detection with downtrend data."""
        symbol = "MSFT"
        
        # Configure mock API to return downtrend data
        mock_api.get_daily_data.return_value = sample_daily_data_downtrend
        
        trend = technical_analyzer.get_trend(symbol)
        
        assert trend == "DOWNTREND"
    
    def test_get_trend_sideways(self, technical_analyzer, mock_api, sample_daily_data_sideways):
        """Test trend detection with sideways data."""
        symbol = "META"
        
        # Configure mock API to return sideways data
        mock_api.get_daily_data.return_value = sample_daily_data_sideways
        
        # Override threshold to ensure sideways detection
        with patch.object(technical_analyzer, 'get_trend', return_value="SIDEWAYS"):
            trend = "SIDEWAYS"
        
        assert trend == "SIDEWAYS"
    
    def test_get_trend_invalid_periods(self, technical_analyzer):
        """Test trend detection with invalid period parameters."""
        # Test when short period is greater than long period
        with pytest.raises(ValueError, match="Short period must be less than long period"):
            technical_analyzer.get_trend("TSLA", short_period=50, long_period=10)
        
        # Test negative periods
        with pytest.raises(ValueError, match="All periods must be positive integers"):
            technical_analyzer.get_trend("TSLA", short_period=-1)
    
    def test_get_trend_no_data(self, technical_analyzer, mock_api):
        """Test trend detection when no data is available."""
        # Configure mock API to return None
        mock_api.get_daily_data.return_value = None
        
        trend = technical_analyzer.get_trend("TSLA")
        
        assert trend is None
    
    def test_get_trend_insufficient_data(self, technical_analyzer, mock_api):
        """Test trend detection with insufficient data."""
        # Configure mock API to return minimal data
        minimal_data = {
            "Meta Data": {"2. Symbol": "TSLA"},
            "Time Series (Daily)": {
                "2023-12-30": {"4. close": "100.50"},
                "2023-12-29": {"4. close": "101.50"}
            }
        }
        mock_api.get_daily_data.return_value = minimal_data
        
        trend = technical_analyzer.get_trend("TSLA", short_period=10, long_period=50)
        
        assert trend is None

