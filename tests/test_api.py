"""
Unit tests for the AlphaVantageAPI class.

These tests cover rate limiting, caching, error handling, and batch processing.
All API calls are mocked to avoid making actual network requests.
"""

import json
import os
import time
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, call

import pytest
import requests

from src.api.alpha_vantage import AlphaVantageAPI, RateLimiter, APICache


@pytest.fixture
def mock_response():
    """Create a mock response object for API call mocking."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "Meta Data": {
            "1. Information": "Daily Prices",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2023-01-10",
        },
        "Time Series (Daily)": {
            "2023-01-10": {
                "1. open": "130.465",
                "2. high": "131.7499",
                "3. low": "128.12",
                "4. close": "130.73",
                "5. volume": "69023854",
            }
        }
    }
    return mock_resp


@pytest.fixture
def mock_intraday_response():
    """Create a mock response object for intraday API call mocking."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "Meta Data": {
            "1. Information": "Intraday (1min) Prices",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2023-01-10 16:00:00",
        },
        "Time Series (1min)": {
            "2023-01-10 16:00:00": {
                "1. open": "130.465",
                "2. high": "131.7499",
                "3. low": "128.12",
                "4. close": "130.73",
                "5. volume": "69023854",
            }
        }
    }
    return mock_resp


@pytest.fixture
def mock_error_response():
    """Create a mock error response object."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "Error Message": "Invalid API call"
    }
    return mock_resp


@pytest.fixture
def mock_rate_limit_response():
    """Create a mock response for rate limit exceeded."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {
        "Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 5 requests per minute and 500 requests per day."
    }
    return mock_resp


@pytest.fixture
def api_instance():
    """Create an AlphaVantageAPI instance for testing."""
    metrics = MagicMock()
    return AlphaVantageAPI(api_key="test_key", metrics_collector=metrics)


@pytest.fixture
def temp_cache_dir(tmpdir):
    """Create a temporary directory for cache testing."""
    cache_dir = Path(tmpdir) / "test_cache"
    cache_dir.mkdir()
    return cache_dir


class TestRateLimiter:
    """Tests for the RateLimiter class."""
    
    def test_rate_limiting(self):
        """Test that rate limiter properly limits API calls."""
        limiter = RateLimiter(max_calls=2, period=1)
        
        @limiter
        def test_func():
            return time.time()
        
        # First two calls should proceed immediately
        start = time.time()
        time1 = test_func()
        time2 = test_func()
        
        assert time.time() - start < 0.5, "First two calls should be immediate"
        
        # Third call should be delayed by rate limiter
        with patch('time.sleep') as mock_sleep:
            time3 = test_func()
            mock_sleep.assert_called_at_least_once()


class TestAPICache:
    """Tests for the APICache class."""
    
    def test_cache_set_get(self, temp_cache_dir):
        """Test that cache correctly stores and retrieves data."""
        cache = APICache(cache_dir=temp_cache_dir, expiry=60)
        
        test_data = {"test": "data", "value": 123}
        key = "test_key"
        
        # Set data in cache
        assert cache.set(key, test_data) is True
        
        # Verify cache file was created
        cache_path = temp_cache_dir / f"{key}.json"
        assert cache_path.exists()
        
        # Get data from cache
        cached_data = cache.get(key)
        assert cached_data is not None
        assert cached_data == test_data
    
    def test_cache_expiry(self, temp_cache_dir):
        """Test that cache correctly handles expiry."""
        # Cache with very short expiry
        cache = APICache(cache_dir=temp_cache_dir, expiry=0.1)
        
        test_data = {"test": "data"}
        key = "expire_key"
        
        # Set data in cache
        cache.set(key, test_data)
        
        # Verify we can retrieve immediately
        assert cache.get(key) == test_data
        
        # Wait for cache to expire
        time.sleep(0.2)
        
        # Verify expired data is not returned
        assert cache.get(key) is None


class TestAlphaVantageAPI:
    """Tests for the AlphaVantageAPI class."""
    
    def test_get_daily_data_success(self, api_instance, mock_response):
        """Test successful daily data retrieval."""
        with patch('requests.get', return_value=mock_response) as mock_get:
            result = api_instance.get_daily_data("AAPL")
            
            # Verify the API was called with correct parameters
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert "function=TIME_SERIES_DAILY" in url
            assert "symbol=AAPL" in url
            assert "apikey=test_key" in url
            
            # Verify result contains expected data
            assert result is not None
            assert "Time Series (Daily)" in result
            assert "Meta Data" in result
    
    def test_get_intraday_data_success(self, api_instance, mock_intraday_response):
        """Test successful intraday data retrieval."""
        with patch('requests.get', return_value=mock_intraday_response) as mock_get:
            result = api_instance.get_intraday_data("AAPL", interval="1min")
            
            # Verify the API was called with correct parameters
            mock_get.assert_called_once()
            url = mock_get.call_args[0][0]
            assert "function=TIME_SERIES_INTRADAY" in url
            assert "symbol=AAPL" in url
            assert "interval=1min" in url
            assert "apikey=test_key" in url
            
            # Verify result contains expected data
            assert result is not None
            assert "Time Series (1min)" in result
            assert "Meta Data" in result
    
    def test_error_handling(self, api_instance, mock_error_response):
        """Test handling of API error responses."""
        with patch('requests.get', return_value=mock_error_response):
            result = api_instance.get_daily_data("INVALID")
            assert result is None
    
    def test_request_exception_handling(self, api_instance):
        """Test handling of request exceptions."""
        with patch('requests.get', side_effect=requests.exceptions.RequestException("Connection error")):
            result = api_instance.get_daily_data("AAPL")
            assert result is None
    
    def test_json_decode_error_handling(self, api_instance):
        """Test handling of JSON decode errors."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        
        with patch('requests.get', return_value=mock_resp):
            result = api_instance.get_daily_data("AAPL")
            assert result is None
    
    def test_rate_limit_handling(self, api_instance, mock_rate_limit_response, mock_response):
        """Test handling of rate limit responses with retry."""
        # Create a sequence of responses: first rate limited, then success
        with patch('requests.get', side_effect=[mock_rate_limit_response, mock_response]) as mock_get:
            with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
                result = api_instance.get_daily_data("AAPL")
                
                # Verify API was called twice (initial + retry)
                assert mock_get.call_count == 2
                
                # Verify we slept before retrying
                mock_sleep.assert_called_once()
                
                # Verify final result is successful
                assert result is not None
                assert "Time Series (Daily)" in result
    
    def test_caching(self, api_instance, mock_response, temp_cache_dir):
        """Test that responses are properly cached."""
        # Replace the cache in the API instance
        api_instance.cache = APICache(cache_dir=temp_cache_dir)
        
        with patch('requests.get', return_value=mock_response) as mock_get:
            # First call should use the API
            result1 = api_instance.get_daily_data("AAPL")
            assert mock_get.call_count == 1
            
            # Second call should use the cache
            result2 = api_instance.get_daily_data("AAPL")
            assert mock_get.call_count == 1  # Still only one call
            
            # Results should be identical
            assert result1 == result2
    
    def test_get_batch_data(self, api_instance):
        """Test batch data retrieval for multiple symbols."""
        with patch.object(api_instance, 'get_daily_data') as mock_get_daily:
            # Mock the get_daily_data method to return different results for different symbols
            def side_effect(symbol):
                return {"symbol": symbol, "data": "test"}
            
            mock_get_daily.side_effect = side_effect
            
            # Get batch data for multiple symbols
            symbols = ["AAPL", "MSFT", "GOOGL"]
            results = api_instance.get_batch_data(symbols)
            
            # Verify get_daily_data was called for each symbol
            assert mock_get_daily.call_count == 3
            mock_get_daily.assert_has_calls([call("AAPL"), call("MSFT"), call("GOOGL")], any_order=True)
            
            # Verify results contain data for all symbols
            assert len(results) == 3
            for symbol in symbols:
                assert symbol in results
                assert results[symbol] is not None
                assert results[symbol]["symbol"] == symbol
    
    def test_metrics_recording(self, api_instance, mock_response):
        """Test that API calls record metrics when metrics collector is provided."""
        with patch('requests.get', return_value=mock_response):
            result = api_instance.get_daily_data("AAPL")
            
            # Verify metrics were recorded
            assert api_instance.metrics.record_api_call.called
            
            # First arg should be "daily"
            call_args = api_instance.metrics.record_api_call.call_args[0]
            assert call_args[0] == "daily"
            
            # Second arg should be a float (elapsed time)
            assert isinstance(call_args[1], float)


if __name__ == "__main__":
    pytest.main()

