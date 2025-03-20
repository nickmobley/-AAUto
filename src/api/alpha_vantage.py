"""
Alpha Vantage API wrapper module with rate limiting and caching capabilities.
This module provides a clean interface to interact with the Alpha Vantage API,
including rate limiting, response caching, and error handling.
"""

import logging
import os
import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from threading import Lock, Timer
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, cast
from concurrent.futures import ThreadPoolExecutor

import requests

# Get API key from environment variable or use default
API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "Y1W59PRPH3SUH44Z")
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Default stocks to track
CACHE_DIR = Path.home() / ".aauto_cache"  # Cache directory
CACHE_EXPIRY = 60 * 60  # Cache expiry in seconds (1 hour)
MAX_REQUESTS_PER_MINUTE = 5  # Alpha Vantage free tier limit
BATCH_SIZE = 3  # Number of symbols to request in a batch

# Type definitions
T = TypeVar('T')
JSONDict = Dict[str, Any]
APIResponse = Optional[JSONDict]


class RateLimiter:
    """Rate limiter to prevent exceeding API limits"""
    
    def __init__(self, max_calls: int, period: int = 60):
        """
        Initialize the rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the period
            period: Time period in seconds for which the rate limit applies
        """
        self.max_calls = max_calls
        self.period = period  # in seconds
        self.calls: List[float] = []
        self.lock = Lock()
        self.backoff_factor = 1.5  # Exponential backoff factor
        self.max_backoff = 60  # Maximum backoff time in seconds
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with self.lock:
                now = time.time()
                # Remove calls older than the period
                self.calls = [call_time for call_time in self.calls if call_time > now - self.period]
                
                # If we've reached the max calls, wait using exponential backoff
                attempts = 0
                while len(self.calls) >= self.max_calls:
                    attempts += 1
                    sleep_time = min(
                        self.max_backoff,
                        (self.calls[0] + self.period - now) * (self.backoff_factor ** attempts)
                    )
                    if sleep_time > 0:
                        logging.warning(f"Rate limit reached. Waiting for {sleep_time:.2f} seconds")
                        time.sleep(sleep_time)
                    # Update the time and calls list after waiting
                    now = time.time()
                    self.calls = [call_time for call_time in self.calls if call_time > now - self.period]
                
                # Make the API call
                result = func(*args, **kwargs)
                
                # Record this call
                self.calls.append(time.time())
                return result
        return wrapper


class APICache:
    """Cache for API responses to reduce API calls"""
    
    def __init__(self, cache_dir: Union[str, Path] = CACHE_DIR, expiry: int = CACHE_EXPIRY):
        """
        Initialize the API cache.
        
        Args:
            cache_dir: Directory to store cache files
            expiry: Cache expiry time in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.expiry = expiry
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, key: str) -> Path:
        """
        Generate a file path for a cache key.
        
        Args:
            key: Cache key to generate path for
            
        Returns:
            Path object for the cache file
        """
        return self.cache_dir / f"{key}.json"
    
    def get(self, key: str) -> Optional[JSONDict]:
        """
        Get data from cache if it exists and is not expired.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached data if available and not expired, None otherwise
        """
        cache_path = self.get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        # Check if cache is expired
        modified_time = cache_path.stat().st_mtime
        if time.time() - modified_time > self.expiry:
            logging.debug(f"Cache expired for {key}")
            return None
            
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
                logging.debug(f"Cache hit for {key}")
                return data
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to read cache for {key}: {e}")
            return None
    
    def set(self, key: str, data: JSONDict) -> bool:
        """
        Store data in cache.
        
        Args:
            key: Cache key to store data under
            data: Data to cache
            
        Returns:
            True if caching was successful, False otherwise
        """
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                logging.debug(f"Cached data for {key}")
            return True
        except IOError as e:
            logging.warning(f"Failed to write cache for {key}: {e}")
            return False


class AlphaVantageAPI:
    """Wrapper for Alpha Vantage API with rate limiting and caching"""
    
    def __init__(self, api_key: str = API_KEY, metrics_collector: Any = None):
        """
        Initialize the Alpha Vantage API wrapper.
        
        Args:
            api_key: Alpha Vantage API key
            metrics_collector: Optional metrics collector for performance tracking
        """
        self.api_key = api_key
        self.cache = APICache()
        self.metrics = metrics_collector
        self.executor = ThreadPoolExecutor(max_workers=BATCH_SIZE)
        
    @RateLimiter(MAX_REQUESTS_PER_MINUTE)
    def get_intraday_data(self, symbol: str, interval: str = "1min", output_size: str = "compact") -> APIResponse:
        """
        Get intraday time series data for a symbol.
        
        Args:
            symbol: Stock symbol to fetch data for
            interval: Time interval between data points (1min, 5min, 15min, 30min, 60min)
            output_size: Amount of data to return (compact or full)
            
        Returns:
            JSON response from API or None if request failed
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            json.JSONDecodeError: If the response contains invalid JSON
        """
        start_time = time.time()
        cache_key = f"intraday_{symbol}_{interval}_{output_size}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            if self.metrics:
                self.metrics.record_api_call(f"intraday_{interval}", time.time() - start_time)
            return cached_data
            
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}"
            f"&outputsize={output_size}&apikey={self.api_key}"
        )
        
        try:
            logging.info(f"Fetching intraday data for {symbol}")
            response = requests.get(url, timeout=10)
            
            # Record API call timing if metrics collector is available
            if self.metrics:
                self.metrics.record_api_call(f"intraday_{interval}", time.time() - start_time)
            
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logging.error(f"API error for {symbol}: {data['Error Message']}")
                return None
                
            # Check for valid time series data
            time_series_key = f"Time Series ({interval})"
            if "Time Series" not in data and time_series_key not in data:
                # Check for API request limit message
                if "Note" in data:
                    logging.warning(f"API limit reached: {data['Note']}")
                    # Wait before retrying to avoid rate limit issues
                    time.sleep(60)  
                    return self.get_intraday_data(symbol, interval, output_size)
                logging.error(f"Unexpected API response format for {symbol}: {data}")
                return None
                
            # Cache the successful response
            self.cache.set(cache_key, data)
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response for {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error for {symbol}: {e}")
            return None
            
    @RateLimiter(MAX_REQUESTS_PER_MINUTE)
    def get_daily_data(self, symbol: str, output_size: str = "compact") -> APIResponse:
        """
        Get daily time series data for a symbol.
        
        Args:
            symbol: Stock symbol to fetch data for
            output_size: Amount of data to return (compact or full)
            
        Returns:
            JSON response from API or None if request failed
            
        Raises:
            requests.exceptions.RequestException: If the request fails
            json.JSONDecodeError: If the response contains invalid JSON
        """
        start_time = time.time()
        cache_key = f"daily_{symbol}_{output_size}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            if self.metrics:
                self.metrics.record_api_call("daily", time.time() - start_time)
            return cached_data
            
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize={output_size}&apikey={self.api_key}"
        )
        
        try:
            logging.info(f"Fetching daily data for {symbol}")
            response = requests.get(url, timeout=10)
            
            # Record API call timing if metrics collector is available
            if self.metrics:
                self.metrics.record_api_call("daily", time.time() - start_time)
            
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logging.error(f"API error for {symbol}: {data['Error Message']}")
                return None
                
            if "Time Series (Daily)" not in data:
                # Check for API request limit message
                if "Note" in data:
                    logging.warning(f"API limit reached: {data['Note']}")
                    # Wait before retrying to avoid rate limit issues
                    time.sleep(60)
                    return self.get_daily_data(symbol, output_size)
                logging.error(f"Unexpected API response format for {symbol}: {data}")
                return None
                
            # Cache the successful response
            self.cache.set(cache_key, data)
            return data
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed for {symbol}: {e}")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse API response for {symbol}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error for {symbol}: {e}")
            return None
    
    def get_batch_data(self, symbols: List[str], data_type: str = "daily") -> Dict[str, APIResponse]:
        """
        Get data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols to fetch data for
            data_type: Type of data to fetch (daily or intraday)
            
        Returns:
            Dictionary mapping symbols to their data responses
        """
        if data_type == "intraday":
            futures = {symbol: self.executor.submit(self.get_intraday_data, symbol) 
                      for symbol in symbols}
        else:
            futures = {symbol: self.executor.submit(self.get_daily_data, symbol) 
                      for symbol in symbols}
        
        results = {}
        for symbol, future in futures.items():
            try:
                results[symbol] = future.result()
            except Exception as e:
                logging.error(f"Error fetching batch data for {symbol}: {e}")
                results[symbol] = None
                
        return results

