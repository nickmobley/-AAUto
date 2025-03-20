import requests
import logging
import time
import os
import json
import argparse
import signal
import sys
from datetime import datetime, timedelta
from threading import Timer, Lock
from functools import wraps
from pathlib import Path
import statistics
import collections
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import math
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aauto.log")
    ]
)

# Get API key from environment variable or use default
API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "Y1W59PRPH3SUH44Z")
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]  # Default stocks to track
CACHE_DIR = Path.home() / ".aauto_cache"  # Cache directory
CACHE_EXPIRY = 60 * 60  # Cache expiry in seconds (1 hour)
MAX_REQUESTS_PER_MINUTE = 5  # Alpha Vantage free tier limit
BATCH_SIZE = 3  # Number of symbols to request in a batch
PERFORMANCE_LOG_FILE = "performance_metrics.json"
class RateLimiter:
    """Rate limiter to prevent exceeding API limits"""
    
    def __init__(self, max_calls, period=60):
        self.max_calls = max_calls
        self.period = period  # in seconds
        self.calls = []
        self.lock = Lock()
        self.backoff_factor = 1.5  # Exponential backoff factor
        self.max_backoff = 60  # Maximum backoff time in seconds
        
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
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
    
    def __init__(self, cache_dir=CACHE_DIR, expiry=CACHE_EXPIRY):
        self.cache_dir = Path(cache_dir)
        self.expiry = expiry
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, key):
        """Generate a file path for a cache key"""
        return self.cache_dir / f"{key}.json"
    
    def get(self, key):
        """Get data from cache if it exists and is not expired"""
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
    
    def set(self, key, data):
        """Store data in cache"""
        cache_path = self.get_cache_path(key)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
                logging.debug(f"Cached data for {key}")
        except IOError as e:
            logging.warning(f"Failed to write cache for {key}: {e}")


class MetricsCollector:
    """Collect and analyze performance metrics"""
    
    def __init__(self, log_file=PERFORMANCE_LOG_FILE):
        self.log_file = log_file
        self.metrics = {
            "api_calls": collections.Counter(),
            "api_response_times": collections.defaultdict(list),
            "strategy_execution_times": collections.defaultdict(list),
            "strategy_profits": collections.defaultdict(list),
            "resource_usage": collections.defaultdict(list),
            "total_profit_history": [],
            "cycle_times": []
        }
        self.load_metrics()
        
    def load_metrics(self):
        """Load existing metrics from file if available"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    loaded_metrics = json.load(f)
                    for key, value in loaded_metrics.items():
                        if key == "api_calls":
                            self.metrics[key] = collections.Counter(value)
                        elif key in ["api_response_times", "strategy_execution_times", 
                                  "strategy_profits", "resource_usage"]:
                            self.metrics[key] = collections.defaultdict(list)
                            for k, v in value.items():
                                self.metrics[key][k] = v
                        else:
                            self.metrics[key] = value
                logging.info(f"Loaded metrics from {self.log_file}")
        except (json.JSONDecodeError, IOError) as e:
            logging.warning(f"Failed to load metrics: {e}")
            
    def save_metrics(self):
        """Save metrics to file"""
        try:
            serializable_metrics = {}
            for key, value in self.metrics.items():
                if isinstance(value, collections.Counter):
                    serializable_metrics[key] = dict(value)
                elif isinstance(value, collections.defaultdict):
                    serializable_metrics[key] = dict(value)
                else:
                    serializable_metrics[key] = value
                    
            with open(self.log_file, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
            logging.info(f"Saved metrics to {self.log_file}")
        except IOError as e:
            logging.error(f"Failed to save metrics: {e}")

    def record_api_call(self, endpoint, response_time):
        """Record API call and its response time"""
        self.metrics["api_calls"][endpoint] += 1
        self.metrics["api_response_times"][endpoint].append(response_time)
        
    def record_strategy_execution(self, strategy_name, execution_time, profit):
        """Record strategy execution metrics"""
        self.metrics["strategy_execution_times"][strategy_name].append(execution_time)
        self.metrics["strategy_profits"][strategy_name].append(profit)
        
    def record_cycle_time(self, cycle_time):
        """Record time taken for a complete trading cycle"""
        self.metrics["cycle_times"].append(cycle_time)
        
    def record_total_profit(self, profit):
        """Record total profit at a point in time"""
        self.metrics["total_profit_history"].append(profit)

class AlphaVantageAPI:
    """Wrapper for Alpha Vantage API with rate limiting and caching"""
    
    def __init__(self, api_key=API_KEY, metrics_collector=None):
        self.api_key = api_key
        self.cache = APICache()
        self.metrics = metrics_collector
        self.executor = ThreadPoolExecutor(max_workers=BATCH_SIZE)
        
    @RateLimiter(MAX_REQUESTS_PER_MINUTE)
    def get_intraday_data(self, symbol, interval="1min", output_size="compact"):
        """Get intraday time series data for a symbol"""
        cache_key = f"intraday_{symbol}_{interval}_{output_size}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
            
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}"
            f"&outputsize={output_size}&apikey={self.api_key}"
        )
        
        try:
            logging.info(f"Fetching intraday data for {symbol}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Check for API errors
            if "Error Message" in data:
                logging.error(f"API error: {data['Error Message']}")
                return None
                
            if "Time Series" not in data and "Time Series (1min)" not in data:
                # Check for API request limit message
                if "Note" in data:
                    logging.warning(f"API limit reached: {data['Note']}")
                    time.sleep(60)  # Wait a minute before retrying
                    return self.get_intraday_data(symbol, interval, output_size)
                logging.error(f"Unexpected API response format: {data}")
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
            
    @RateLimiter(MAX_REQUESTS_PER_MINUTE)
    def get_daily_data(self, symbol, output_size="compact"):
        """Get daily time series data for a symbol"""
        cache_key = f"daily_{symbol}_{output_size}"
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            return cached_data
            
        url = (
            f"https://www.alphavantage.co/query?"
            f"function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize={output_size}&apikey={self.api_key}"
        )
        
        try:
            logging.info(f"Fetching daily data for {symbol}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if "Time Series (Daily)" not in data:
                logging.error(f"Unexpected API response format: {data}")
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


class RiskManager:
    """Manages trading risk and position sizing"""
    
    def __init__(self, initial_capital=10000.0, max_risk_per_trade=0.02):
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.positions = {}
        self.trade_history = []
        
    def calculate_position_size(self, symbol, entry_price, stop_loss):
        """Calculate safe position size based on risk parameters"""
        if stop_loss >= entry_price:
            return 0  # Invalid stop loss for long position
            
        risk_amount = self.capital * self.max_risk_per_trade
        risk_per_share = entry_price - stop_loss
        position_size = risk_amount / risk_per_share
        return position_size
        
    def set_stop_loss(self, symbol, entry_price, risk_percentage=0.02):
        """Calculate stop loss price based on risk percentage"""
        stop_loss = entry_price * (1 - risk_percentage)
        return stop_loss
        
    def set_take_profit(self, symbol, entry_price, risk_reward_ratio=2):
        """Calculate take profit based on risk-reward ratio"""
        stop_loss = self.set_stop_loss(symbol, entry_price)
        risk = entry_price - stop_loss
        take_profit = entry_price + (risk * risk_reward_ratio)
        return take_profit
        
    def record_trade(self, symbol, entry_price, exit_price, position_size):
        """Record trade details for analysis"""
        profit_loss = (exit_price - entry_price) * position_size
        self.capital += profit_loss
        
        trade = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'profit_loss': profit_loss,
            'timestamp': datetime.now()
        }
        self.trade_history.append(trade)
        return profit_loss
        
    def get_max_drawdown(self):
        """Calculate maximum drawdown from trade history"""
        if not self.trade_history:
            return 0
            
        peak = self.trade_history[0]['profit_loss']
        max_drawdown = 0
        
        for trade in self.trade_history:
            profit_loss = trade['profit_loss']
            if profit_loss > peak:
                peak = profit_loss
            drawdown = (peak - profit_loss) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        return max_drawdown

class TechnicalAnalyzer:
    """Performs technical analysis on price data"""
    
    def __init__(self, api):
        self.api = api
        
    def calculate_rsi(self, symbol, period=14):
        """Calculate Relative Strength Index"""
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return None
            
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in time_series.values()]
        
        if len(closes) < period + 1:
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
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_macd(self, symbol, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return None
            
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in time_series.values()]
        
        if len(closes) < slow_period + signal_period:
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
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        multiplier = 2 / (period + 1)
        ema = [data[0]]  # First EMA is SMA
        
        for price in data[1:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
            
        return ema
        
    def get_trend(self, symbol, short_period=10, long_period=50):
        """Determine trend based on moving averages"""
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return None
            
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in time_series.values()]
        
        if len(closes) < long_period:
            return None
            
        short_ma = sum(closes[:short_period]) / short_period
        long_ma = sum(closes[:long_period]) / long_period
        
        if short_ma > long_ma:
            return "UPTREND"
        elif short_ma < long_ma:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"

class MarketSentiment:
    """Analyzes market sentiment using various indicators and news sources"""
    
    def __init__(self, api):
        self.api = api
        self.sentiment_cache = {}
        self.market_states = collections.defaultdict(str)
        self.correlation_matrix = {}
        
    def analyze_market_regime(self, symbol, period=30):
        """Determine the current market regime (trending, volatile, ranging)"""
        volatility = self.calculate_volatility(symbol, period)
        trend = self.detect_trend(symbol, period)
        
        if volatility > 0.02:  # High volatility threshold
            if abs(trend) > 0.1:  # Strong trend
                return "VOLATILE_TRENDING"
            return "VOLATILE_RANGING"
        else:
            if abs(trend) > 0.1:
                return "STABLE_TRENDING"
            return "STABLE_RANGING"
    
    def calculate_volatility(self, symbol, period):
        """Calculate historical volatility"""
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return 0
            
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in list(time_series.values())[:period]]
        
        if len(closes) < 2:
            return 0
            
        returns = [math.log(closes[i] / closes[i+1]) for i in range(len(closes)-1)]
        return statistics.stdev(returns) * math.sqrt(252)  # Annualized volatility
    
    def detect_trend(self, symbol, period):
        """Detect price trend using linear regression"""
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return 0
            
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in list(time_series.values())[:period]]
        
        if len(closes) < period:
            return 0
            
        # Simple linear regression
        x = list(range(len(closes)))
        y = closes
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
        
        slope = numerator / denominator if denominator != 0 else 0
        return slope / mean_y  # Normalized trend
    
    def calculate_correlation(self, symbols, period=30):
        """Calculate correlation matrix between symbols"""
        price_data = {}
        
        # Gather price data for all symbols
        for symbol in symbols:
            data = self.api.get_daily_data(symbol)
            if data and "Time Series (Daily)" in data:
                time_series = data["Time Series (Daily)"]
                closes = [float(v["4. close"]) for v in list(time_series.values())[:period]]
                if len(closes) >= period:
                    price_data[symbol] = closes
        
        # Calculate correlation matrix
        correlation_matrix = {}
        for symbol1 in price_data:
            correlation_matrix[symbol1] = {}
            for symbol2 in price_data:
                if len(price_data[symbol1]) == len(price_data[symbol2]):
                    correlation = self.calculate_correlation_coefficient(
                        price_data[symbol1], price_data[symbol2]
                    )
                    correlation_matrix[symbol1][symbol2] = correlation
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def calculate_correlation_coefficient(self, x, y):
        """Calculate Pearson correlation coefficient between two price series"""
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        covariance = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        std_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
        std_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))
        
        if std_x == 0 or std_y == 0:
            return 0
            
        return covariance / (std_x * std_y)
    
    def get_portfolio_risk(self, symbols, weights):
        """Calculate portfolio risk based on correlations and volatilities"""
        if not self.correlation_matrix:
            self.calculate_correlation(symbols)
            
        portfolio_risk = 0
        for i, symbol1 in enumerate(symbols):
            volatility1 = self.calculate_volatility(symbol1, 30)
            for j, symbol2 in enumerate(symbols):
                volatility2 = self.calculate_volatility(symbol2, 30)
                correlation = self.correlation_matrix.get(symbol1, {}).get(symbol2, 0)
                portfolio_risk += (
                    weights[i] * weights[j] * 
                    volatility1 * volatility2 * 
                    correlation
                )
        
        return math.sqrt(portfolio_risk)
    
    def optimize_portfolio_weights(self, symbols, target_risk=0.15):
        """Optimize portfolio weights to achieve target risk level"""
        n = len(symbols)
        initial_weights = [1/n] * n  # Start with equal weights
        
        # Simple optimization using gradient descent
        learning_rate = 0.01
        max_iterations = 100
        
        weights = initial_weights.copy()
        for _ in range(max_iterations):
            current_risk = self.get_portfolio_risk(symbols, weights)
            if abs(current_risk - target_risk) < 0.01:
                break
                
            # Adjust weights based on risk difference
            adjustment = learning_rate * (target_risk - current_risk)
            weights = [w * (1 + adjustment) for w in weights]
            
            # Normalize weights to sum to 1
            total = sum(weights)
            weights = [w / total for w in weights]
        
        return dict(zip(symbols, weights))

class MachineLearning:
    """Handles machine learning models for price prediction and pattern recognition"""
    
    def __init__(self, api):
        self.api = api
        self.models = {}
        self.scalers = {}
        self.prediction_cache = {}
        self.pattern_cache = {}
        
    def prepare_data(self, symbol, lookback=30, prediction_days=5):
        """Prepare data for machine learning models"""
        data = self.api.get_daily_data(symbol, output_size="full")
        if not data or "Time Series (Daily)" not in data:
            return None, None
            
        # Extract features
        time_series = data["Time Series (Daily)"]
        dates = list(time_series.keys())
        features = []
        targets = []
        
        for i in range(len(dates) - lookback - prediction_days):
            feature_window = []
            for j in range(lookback):
                current_date = dates[i + j]
                daily_data = time_series[current_date]
                
                # Create feature vector
                feature_vector = [
                    float(daily_data['1. open']),
                    float(daily_data['2. high']),
                    float(daily_data['3. low']),
                    float(daily_data['4. close']),
                    float(daily_data['5. volume'])
                ]
                feature_window.extend(feature_vector)
            
            # Target is the closing price 'prediction_days' days ahead
            target_date = dates[i + lookback + prediction_days - 1]
            target = float(time_series[target_date]['4. close'])
            
            features.append(feature_window)
            targets.append(target)
        
        return np.array(features), np.array(targets)
    
    def train_model(self, symbol, lookback=30, prediction_days=5):
        """Train a machine learning model for price prediction"""
        X, y = self.prepare_data(symbol, lookback, prediction_days)
        if X is None or len(X) < 100:  # Ensure enough data
            return False
        
        # Scale the data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = scaler.fit_transform(y.reshape(-1, 1))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_scaled, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train.ravel())
        
        # Store model and scaler
        self.models[symbol] = model
        self.scalers[symbol] = scaler
        
        return True
    
    def predict_price(self, symbol, current_data):
        """Predict future price movement"""
        if symbol not in self.models:
            if not self.train_model(symbol):
                return None
        
        # Prepare current data
        if len(current_data) != self.models[symbol].n_features_in_:
            return None
        
        # Scale and predict
        X_scaled = self.scalers[symbol].transform([current_data])
        prediction_scaled = self.models[symbol].predict(X_scaled)
        prediction = self.scalers[symbol].inverse_transform(
            prediction_scaled.reshape(-1, 1)
        )
        
        return prediction[0][0]
    
    def detect_patterns(self, symbol, window_size=20):
        """Detect common chart patterns"""
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return None
            
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in list(time_series.values())[:window_size]]
        
        if len(closes) < window_size:
            return None
        
        patterns = {
            "trend": self.detect_trend_pattern(closes),
            "support_resistance": self.find_support_resistance(closes),
            "volatility": self.calculate_volatility_pattern(closes)
        }
        
        return patterns
    
    def detect_trend_pattern(self, prices):
        """Detect trend patterns using linear regression"""
        x = np.array(range(len(prices)))
        y = np.array(prices)
        
        # Calculate trend line
        slope, intercept = np.polyfit(x, y, 1)
        r_squared = np.corrcoef(x, y)[0, 1]**2
        
        if abs(r_squared) > 0.7:  # Strong trend
            if slope > 0:
                return "STRONG_UPTREND"
            else:
                return "STRONG_DOWNTREND"
        elif abs(r_squared) > 0.3:  # Weak trend
            if slope > 0:
                return "WEAK_UPTREND"
            else:
                return "WEAK_DOWNTREND"
        else:
            return "NO_TREND"
    
    def find_support_resistance(self, prices):
        """Find potential support and resistance levels"""
        # Calculate local minima and maxima
        peaks = []
        troughs = []
        
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        return {
            "resistance": [p[1] for p in peaks],
            "support": [t[1] for t in troughs]
        }
    
    def calculate_volatility_pattern(self, prices):
        """Analyze volatility patterns"""
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        if volatility > 0.02:
            return "HIGH_VOLATILITY"
        elif volatility > 0.01:
            return "MEDIUM_VOLATILITY"
        else:
            return "LOW_VOLATILITY"

class NewsAnalyzer:
    """Analyzes news sentiment and market impact"""
    
    def __init__(self, api):
        self.api = api
        self.sentiment_scores = collections.defaultdict(float)
        self.news_impact = collections.defaultdict(list)
        self.keywords = {
            'positive': ['growth', 'profit', 'success', 'innovation', 'breakthrough',
                       'launch', 'partnership', 'expansion', 'beat', 'upgrade'],
            'negative': ['loss', 'decline', 'risk', 'lawsuit', 'investigation',
                       'scandal', 'downgrade', 'bearish', 'crash', 'bankruptcy'],
            'neutral': ['announce', 'report', 'update', 'change', 'plan',
                      'schedule', 'maintain', 'hold', 'continue', 'steady']
        }
    
    def analyze_news_sentiment(self, symbol, news_items):
        """Analyze sentiment from news articles"""
        if not news_items:
            return 0.0
            
        total_score = 0
        weight_sum = 0
        
        for item in news_items:
            # Calculate base sentiment
            positive_count = sum(1 for word in self.keywords['positive'] 
                               if word in item['title'].lower())
            negative_count = sum(1 for word in self.keywords['negative'] 
                               if word in item['title'].lower())
            
            # Calculate sentiment score (-1 to 1)
            if positive_count + negative_count == 0:
                sentiment = 0
            else:
                sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            
            # Apply time decay weight (more recent news has higher weight)
            time_weight = self.calculate_time_weight(item['timestamp'])
            
            # Apply source reliability weight
            source_weight = self.get_source_reliability(item['source'])
            
            # Calculate final weighted score
            weight = time_weight * source_weight
            total_score += sentiment * weight
            weight_sum += weight
        
        # Calculate weighted average sentiment
        final_sentiment = total_score / weight_sum if weight_sum > 0 else 0
        
        # Store sentiment score
        self.sentiment_scores[symbol] = final_sentiment
        return final_sentiment
    
    def calculate_time_weight(self, timestamp, decay_factor=0.1):
        """Calculate time-based weight for news items"""
        time_diff = datetime.now() - datetime.fromtimestamp(timestamp)
        hours_old = time_diff.total_seconds() / 3600
        return math.exp(-decay_factor * hours_old)
    
    def get_source_reliability(self, source):
        """Get reliability weight for news source"""
        reliability_scores = {
            'reuters': 0.9,
            'bloomberg': 0.9,
            'wsj': 0.85,
            'cnbc': 0.8,
            'seeking_alpha': 0.75,
            'yahoo_finance': 0.7,
            'default': 0.5
        }
        return reliability_scores.get(source.lower(), reliability_scores['default'])
    
    def analyze_market_impact(self, symbol, sentiment_score):
        """Analyze potential market impact based on sentiment"""
        # Get recent price data
        data = self.api.get_daily_data(symbol)
        if not data or "Time Series (Daily)" not in data:
            return None
        
        # Calculate price volatility
        volatility = self.calculate_volatility(data)
        
        # Calculate impact score
        impact = {
            'magnitude': abs(sentiment_score) * (1 + volatility),
            'direction': 1 if sentiment_score > 0 else -1,
            'confidence': min(abs(sentiment_score) * 2, 1.0),
            'volatility_factor': volatility
        }
        
        self.news_impact[symbol].append(impact)
        return impact
    
    def calculate_volatility(self, data, window=10):
        """Calculate recent price volatility"""
        time_series = data["Time Series (Daily)"]
        closes = [float(v["4. close"]) for v in list(time_series.values())[:window]]
        
        if len(closes) < 2:
            return 0
        
        returns = [math.log(closes[i] / closes[i+1]) for i in range(len(closes)-1)]
        return statistics.stdev(returns) if returns else 0
    
    def get_combined_score(self, symbol):
        """Get combined score considering sentiment and market impact"""
        sentiment = self.sentiment_scores.get(symbol, 0)
        impacts = self.news_impact.get(symbol, [])
        
        if not impacts:
            return sentiment
        
        # Calculate weighted impact score
        total_impact = sum(impact['magnitude'] * impact['direction'] * impact['confidence']
                         for impact in impacts)
        avg_impact = total_impact / len(impacts)
        
        # Combine sentiment and impact scores
        combined_score = (sentiment + avg_impact) / 2
        
        return combined_score

class TradeOptimizer:
    """Optimizes trade execution and strategy parameters"""
    
    def __init__(self, api):
        self.api = api
        self.execution_history = collections.defaultdict(list)
        self.slippage_stats = collections.defaultdict(list)
        self.optimal_params = {}
        
    def optimize_entry_timing(self, symbol, target_price, window=5):
        """Optimize trade entry timing based on intraday patterns"""
        data = self.api.get_intraday_data(symbol, interval="1min")
        if not data or "Time Series (1min)" not in data:
            return None
            
        time_series = data["Time Series (1min)"]
        prices = [(k, float(v["4. close"])) for k, v in time_series.items()]
        
        # Analyze price momentum
        momentum = self.calculate_momentum(prices, window)
        
        # Check volume profile
        volume_profile = self.analyze_volume_profile(time_series)
        
        # Determine optimal entry based on price momentum and volume
        optimal_entry = {
            'price': target_price,
            'timing': self.get_optimal_timing(momentum, volume_profile),
            'confidence': self.calculate_entry_confidence(momentum, volume_profile)
        }
        
        return optimal_entry
    
    def calculate_momentum(self, prices, window):
        """Calculate price momentum"""
        if len(prices) < window:
            return 0
            
        recent_prices = prices[:window]
        price_changes = [p[1] - prices[i+1][1] for i, p in enumerate(recent_prices[:-1])]
        return sum(price_changes) / len(price_changes)
    
    def analyze_volume_profile(self, time_series):
        """Analyze trading volume profile"""
        volumes = [(k, float(v["5. volume"])) for k, v in time_series.items()]
        avg_volume = sum(v[1] for v in volumes) / len(volumes)
        
        profile = {
            'average': avg_volume,
            'trend': self.calculate_volume_trend(volumes),
            'spikes': self.detect_volume_spikes(volumes, avg_volume)
        }
        
        return profile
    
    def calculate_volume_trend(self, volumes, window=10):
        """Calculate volume trend"""
        if len(volumes) < window:
            return 0
            
        recent_volumes = [v[1] for v in volumes[:window]]
        volume_changes = [recent_volumes[i] - recent_volumes[i+1] 
                       for i in range(len(recent_volumes)-1)]
        return sum(volume_changes) / len(volume_changes)
    
    def detect_volume_spikes(self, volumes, avg_volume, threshold=2.0):
        """Detect significant volume spikes"""
        spikes = []
        for timestamp, volume in volumes:
            if volume > avg_volume * threshold:
                spikes.append({
                    'timestamp': timestamp,
                    'volume': volume,
                    'magnitude': volume / avg_volume
                })
        return spikes
    
    def get_optimal_timing(self, momentum, volume_profile):
        """Determine optimal trade timing"""
        if momentum > 0 and volume_profile['trend'] > 0:
            return 'IMMEDIATE'
        elif len(volume_profile['spikes']) > 0:
            return 'AFTER_SPIKE'
        else:
            return 'WAIT_FOR_CONFIRMATION'
    
    def calculate_entry_confidence(self, momentum, volume_profile):
        """Calculate confidence score for trade entry"""
        momentum_score = min(abs(momentum) / 0.01, 1.0)  # Normalize momentum
        volume_score = min(volume_profile['trend'] / volume_profile['average'], 1.0)
        
        # Weight the scores (can be adjusted based on backtesting)
        confidence = (momentum_score * 0.6) + (volume_score * 0.4)
        return min(max(confidence, 0), 1)  # Ensure between 0 and 1
    
    def record_execution(self, symbol, order_type, target_price, executed_price):
        """Record trade execution details"""
        slippage = abs(executed_price - target_price) / target_price
        
        execution = {
            'timestamp': datetime.now(),
            'target_price': target_price,
            'executed_price': executed_price,
            'slippage': slippage,
            'order_type': order_type
        }
        
        self.execution_history[symbol].append(execution)
        self.slippage_stats[symbol].append(slippage)
    
    def get_execution_analytics(self, symbol):
        """Get execution quality analytics"""
        if symbol not in self.execution_history:
            return None
            
        executions = self.execution_history[symbol]
        slippages = self.slippage_stats[symbol]
        
        analytics = {
            'avg_slippage': sum(slippages) / len(slippages),
            'max_slippage': max(slippages),
            'min_slippage': min(slippages),
            'total_executions': len(executions),
            'recent_quality': self.calculate_recent_quality(executions)
        }
        
        return analytics
    
    def calculate_recent_quality(self, executions, window=10):
        """Calculate recent execution quality"""
        if not executions:
            return 0
            
        recent = executions[-window:]
        avg_slippage = sum(e['slippage'] for e in recent) / len(recent)
        quality_score = 1 - min(avg_slippage * 10, 1)  # Convert slippage to quality score
        return quality_score

class ProfitCalculator:
    """Calculate profits based on market data and strategies"""
    
    def __init__(self, api, symbols=DEFAULT_SYMBOLS):
        self.api = api
        self.symbols = symbols
        self.strategy_history = {}  # Track historical performance
        self.risk_manager = RiskManager()
        self.technical_analyzer = TechnicalAnalyzer(api)
        self.machine_learning = MachineLearning(api)
        self.news_analyzer = NewsAnalyzer(api)
        self.trade_optimizer = TradeOptimizer(api)
        
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        data = self.api.get_intraday_data(symbol)
        if not data or "Time Series (1min)" not in data:
            logging.warning(f"Failed to get latest price for {symbol}, trying backup method")
            # Try daily data as a backup
            daily_data = self.api.get_daily_data(symbol)
            if not daily_data or "Time Series (Daily)" not in daily_data:
                logging.error(f"Unable to get price data for {symbol}")
                return None
            
            # Get the latest daily price
            latest_date = list(daily_data["Time Series (Daily)"].keys())[0]
            return float(daily_data["Time Series (Daily)"][latest_date]["4. close"])
        
        # Get the latest intraday price
        latest_timestamp = list(data["Time Series (1min)"].keys())[0]
        return float(data["Time Series (1min)"][latest_timestamp]["4. close"])
        
    def get_price_change_percent(self, symbol, days=7):
        """Calculate price change percentage over the specified days"""
        data = self.api.get_daily_data(symbol, output_size="full")
        if not data or "Time Series (Daily)" not in data:
            logging.error(f"Unable to get historical data for {symbol}")
            return 0
            
        time_series = data["Time Series (Daily)"]
        dates = list(time_series.keys())
        
        if len(dates) < days + 1:
            logging.warning(f"Not enough historical data for {symbol}, using available data")
            days = len(dates) - 1
            if days <= 0:
                return 0
                
        current_price = float(time_series[dates[0]]["4. close"])
        past_price = float(time_series[dates[days]]["4. close"])
        
        if past_price == 0:
            return 0
            
        return ((current_price - past_price) / past_price) * 100
        
    def get_volatility(self, symbol, days=30):
        """Calculate price volatility (standard deviation) over the specified days"""
        data = self.api.get_daily_data(symbol, output_size="full")
        if not data or "Time Series (Daily)" not in data:
            logging.error(f"Unable to get historical data for {symbol}")
            return 0
            
        time_series = data["Time Series (Daily)"]
        dates = list(time_series.keys())
        
        if len(dates) < days:
            logging.warning(f"Not enough historical data for {symbol}, using available data")
            days = len(dates)
            if days <= 1:
                return 0
                
        # Calculate standard deviation of closing prices
        prices = [float(time_series[dates[i]]["4. close"]) for i in range(min(days, len(dates)))]
        return statistics.stdev(prices) if len(prices) > 1 else 0
        
    def calculate_profit(self, strategy, allocated_time, allocated_resources=10):
        """Calculate profit based on market data and strategy"""
        # Select a symbol based on strategy
        if strategy == "investments":
            # Use a less volatile stock for investments
            volatilities = [(s, self.get_volatility(s)) for s in self.symbols]
            symbol = min(volatilities, key=lambda x: x[1])[0]
        elif strategy == "trading":
            # Use a more volatile stock for trading
            volatilities = [(s, self.get_volatility(s)) for s in self.symbols]
            symbol = max(volatilities, key=lambda x: x[1])[0]
        else:
            # Default to the first symbol for other strategies
            symbol = self.symbols[0]
            
        latest_price = self.get_latest_price(symbol)
        if latest_price is None:
            # Fallback to a random value if API fails
            import random
            latest_price = random.uniform(100, 200)
            logging.warning(f"Using fallback price for {strategy}: {latest_price}")
            
        # Get recent price change percentage
        price_change = self.get_price_change_percent(symbol)
        
        # Use technical analysis for better decision making
        rsi = self.technical_analyzer.calculate_rsi(symbol)
        macd = self.technical_analyzer.calculate_macd(symbol)
        trend = self.technical_analyzer.get_trend(symbol)
        
        # Use risk management for position sizing
        stop_loss = self.risk_manager.set_stop_loss(symbol, latest_price)
        take_profit = self.risk_manager.set_take_profit(symbol, latest_price)
        position_size = self.risk_manager.calculate_position_size(symbol, latest_price, stop_loss)
        
        # Calculate base profit
        base_profit = 0
        if strategy == "freelancing":
            # Freelancing: relatively stable income
            base_profit = allocated_time * 3.5 + (allocated_resources * 0.2)
        elif strategy == "investments":
            # Investments: affected by market conditions and trend
            market_factor = 1 + (price_change / 100)
            trend_factor = 1.2 if trend == "UPTREND" else (0.8 if trend == "DOWNTREND" else 1.0)
            base_profit = allocated_time * 1.2 * market_factor * trend_factor + (allocated_resources * 0.5 * market_factor)
        elif strategy == "trading":
            # Trading: higher risk/reward, more affected by technical indicators
            volatility = self.get_volatility(symbol)
            volatility_factor = max(0.5, min(2.0, volatility / 10))  # Normalize volatility factor
            # Incorporate RSI for trading decisions
            rsi_factor = 1.0
            if rsi is not None:
                if rsi < 30:  # Oversold condition
                    rsi_factor = 1.5  # Potential buying opportunity
                elif rsi > 70:  # Overbought condition
                    rsi_factor = 0.7  # Reduce exposure

            # Use MACD for trend confirmation
            macd_factor = 1.0
            if macd is not None:
                if macd['histogram'] > 0 and macd['macd_line'] > macd['signal_line']:
                    macd_factor = 1.3  # Strong upward momentum
                elif macd['histogram'] < 0 and macd['macd_line'] < macd['signal_line']:
                    macd_factor = 0.8  # Strong downward momentum

            # Combine technical indicators with market factors
            market_factor = 1 + (abs(price_change) / 50)  # Absolute value as trading profits from movement in either direction
            technical_factor = (rsi_factor + macd_factor) / 2

            # Calculate final trading profit
            base_profit = (
                allocated_time * 2.5 * 
                volatility_factor * 
                market_factor * 
                technical_factor * 
                (allocated_resources * 0.3)
            )

            # Apply position sizing and risk management
            position_adjustment = min(1.0, position_size / (allocated_resources * 100))
            base_profit *= position_adjustment

            # Record trade in risk manager
            self.risk_manager.record_trade(
                symbol=symbol,
                entry_price=latest_price,
                exit_price=latest_price * (1 + (base_profit / (position_size * latest_price))),
                position_size=position_size
            )
            
        # Record strategy performance
        # Record strategy performance
        if strategy not in self.strategy_history:
            self.strategy_history[strategy] = []
        
        # Add some randomness to simulate real-world variability (±15%)
        import random
        randomness = random.uniform(0.85, 1.15)
        final_profit = base_profit * randomness
        
        # Record strategy performance
        self.strategy_history[strategy].append(final_profit)
        
        return final_profit
        
    def get_best_strategy(self):
        """Return the best performing strategy based on average profit"""
        if not self.strategy_history:
            return None
            
        avg_profits = {}
        for strategy, profits in self.strategy_history.items():
            if profits:  # Only consider strategies with at least one profit record
                avg_profits[strategy] = sum(profits) / len(profits)
                
        if not avg_profits:
            return None
            
        best_strategy = max(avg_profits.items(), key=lambda x: x[1])
        return best_strategy


# Signal handler for graceful shutdown
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    logging.info("Shutting down gracefully...")
    shutdown_requested = True


# Main execution
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Autonomous Money-Making Agent")
    parser.add_argument("--start", action="store_true", help="Start the autonomous agent")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, 
                    help=f"Stock symbols to use (default: {', '.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--interval", type=int, default=10, 
                    help="Interval between cycles in seconds (default: 10)")
    parser.add_argument("--api-key", default=API_KEY,
                    help="Alpha Vantage API key (default: from env or built-in key)")
    parser.add_argument("--risk-level", type=float, default=0.02,
                    help="Maximum risk per trade (default: 0.02 or 2%)")
    parser.add_argument("--target-return", type=float, default=0.15,
                    help="Target annual return (default: 0.15 or 15%)")
    args = parser.parse_args()
    
    if args.start:
        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, signal_handler)
        
        # Initialize components
        api = AlphaVantageAPI(api_key=args.api_key)
        calculator = ProfitCalculator(api, symbols=args.symbols)
        
        # Initialize performance monitoring
        metrics_collector = MetricsCollector()
        
        # Initial resources
        resources = {
            "time": 10.0,          # Time units available
            "skills_exp": 5.0,     # Skill experience level
            "threshold": 20,       # Resource warning threshold
            "replenish_rate": 1.7  # Rate at which resources replenish each cycle
        }
        
        # Strategy configuration
        strategies = {
            "investments": {
                "min_requirements": {"time": 10.0, "skills_exp": 3.0},
                "risk_profile": "low",
                "time_horizon": "long"
            },
            "trading": {
                "min_requirements": {"time": 5.0, "skills_exp": 2.0},
                "risk_profile": "high",
                "time_horizon": "short"
            },
            "freelancing": {
                "min_requirements": {"time": 15.0, "skills_exp": 7.0},
                "risk_profile": "medium",
                "time_horizon": "medium"
            },
            "affiliate_marketing": {
                "min_requirements": {"time": 12.0, "skills_exp": 5.0},
                "risk_profile": "medium",
                "time_horizon": "medium"
            }
        }
        
        # Portfolio optimization
        portfolio = {
            "cash": 10000.0,  # Initial capital
            "positions": {},
            "target_risk": args.risk_level,
            "target_return": args.target_return
        }
        
        current_strategy_index = 0
        total_profit = 0
        cycle_count = 0
        
        logging.info("Autonomous Money-Making Agent started.")
        
        # Main loop
        while not shutdown_requested:
            cycle_start_time = time.time()
            cycle_count += 1
            logging.info(f"Starting cycle {cycle_count}")
            
            try:
                # Update market analysis
                for symbol in args.symbols:
                    # Technical analysis
                    rsi = calculator.technical_analyzer.calculate_rsi(symbol)
                    macd = calculator.technical_analyzer.calculate_macd(symbol)
                    trend = calculator.technical_analyzer.get_trend(symbol)
                    
                    # Machine learning prediction
                    prediction = calculator.machine_learning.predict_price(symbol, [])
                    patterns = calculator.machine_learning.detect_patterns(symbol)
                    
                    # News analysis
                    news_impact = calculator.news_analyzer.get_combined_score(symbol)
                    
                    logging.info(f"Analysis for {symbol}:")
                    logging.info(f"RSI: {rsi}, Trend: {trend}")
                    logging.info(f"Predicted price movement: {prediction}")
                    logging.info(f"News sentiment: {news_impact}")
                
                # Execute strategies
                for strategy_name, strategy_config in strategies.items():
                    requirements = strategy_config["min_requirements"]
                    
                    if (resources["time"] >= requirements["time"] and 
                        resources["skills_exp"] >= requirements["skills_exp"]):
                        
                        # Calculate optimal position size
                        symbol = calculator.symbols[0]  # Default symbol
                        entry_price = calculator.get_latest_price(symbol)
                        
                        if entry_price:
                            # Optimize trade entry
                            entry_timing = calculator.trade_optimizer.optimize_entry_timing(
                                symbol, entry_price
                            )
                            
                            if entry_timing and entry_timing['confidence'] > 0.7:
                                # Execute trade
                                profit = calculator.calculate_profit(
                                    strategy_name,
                                    resources["time"],
                                    resources["skills_exp"]
                                )
                                
                                if profit:
                                    total_profit += profit
                                    portfolio["cash"] += profit
                                    
                                    # Record metrics
                                    metrics_collector.record_strategy_execution(
                                        strategy_name,
                                        time.time() - cycle_start_time,
                                        profit
                                    )
                                    
                                    logging.info(
                                        f"Executed {strategy_name} strategy. "
                                        f"Profit: {profit:.2f}, Total: {total_profit:.2f}"
                                    )
                
                # Replenish resources
                resources["time"] += resources["replenish_rate"]
                resources["skills_exp"] += resources["replenish_rate"] / 2
                
                # Check resource levels
                if resources["time"] < resources["threshold"]:
                    logging.warning(
                        f"Resource alarm: time below threshold "
                        f"({resources['time']:.2f} < {resources['threshold']})"
                    )
                
                # Generate and save performance report
                if cycle_count % 10 == 0:  # Every 10 cycles
                    report = metrics_collector.generate_summary_report()
                    metrics_collector.save_metrics()
                    
                    # Generate performance visualizations
                    metrics_collector.visualize_performance()
                    
                    logging.info(f"Performance Report: {json.dumps(report, indent=2)}")
                
            except Exception as e:
                logging.error(f"Error in cycle {cycle_count}: {str(e)}")
                logging.exception("Detailed error information:")
            
            # Wait for next cycle
            cycle_time = time.time() - cycle_start_time
            metrics_collector.record_cycle_time(cycle_time)
            
            sleep_time = max(0, args.interval - cycle_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Graceful shutdown
        logging.info("Shutting down...")
        metrics_collector.save_metrics()
        logging.info(f"Final total profit: {total_profit:.2f}")
        logging.info("Shutdown complete.")
