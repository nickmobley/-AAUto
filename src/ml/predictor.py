import os
import logging
import pickle
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, TypedDict, cast
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aauto.log")
    ]
)

class ModelMetadata(TypedDict):
    """Type definition for model metadata."""
    symbol: str
    version: str
    created_at: str
    features: List[str]
    lookback: int
    prediction_days: int
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]


@dataclass
class PredictionResult:
    """Data class for prediction results with confidence."""
    value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    

class MachineLearning:
    """
    Handles machine learning models for price prediction and pattern recognition.
    
    This class provides functionality for:
    - Training and persisting machine learning models
    - Price prediction with confidence intervals
    - Market pattern detection and analysis
    - Feature importance analysis
    - Model versioning and metadata tracking
    """
    
    def __init__(self, api, model_dir: str = "models", default_lookback: int = 30, default_prediction_days: int = 5):
        """
        Initialize the ML predictor with API connection and configuration.
        
        Args:
            api: API client instance for data retrieval
            model_dir: Directory to store trained models
            default_lookback: Default number of days to use for training features
            default_prediction_days: Default number of days to predict ahead
        """
        self.api = api
        self.models: Dict[str, RandomForestRegressor] = {}
        self.scalers: Dict[str, Dict[str, MinMaxScaler]] = {}
        self.prediction_cache: Dict[str, PredictionResult] = {}
        self.pattern_cache: Dict[str, Dict[str, Any]] = {}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.default_lookback = default_lookback
        self.default_prediction_days = default_prediction_days
        self.model_metadata: Dict[str, ModelMetadata] = {}
        
    def prepare_data(
        self, 
        symbol: str, 
        lookback: int = None, 
        prediction_days: int = None
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """
        Prepare data for machine learning models with proper feature engineering.
        
        Args:
            symbol: Stock symbol to retrieve data for
            lookback: Number of days to look back for features
            prediction_days: Number of days ahead to predict
            
        Returns:
            Tuple containing: 
            - Feature matrix X
            - Target vector y
            - List of feature names
        """
        # Use defaults if not specified
        lookback = lookback or self.default_lookback
        prediction_days = prediction_days or self.default_prediction_days
        
        try:
            data = self.api.get_daily_data(symbol, output_size="full")
            if not data or "Time Series (Daily)" not in data:
                logging.error(f"Failed to retrieve data for {symbol}")
                return None, None, None
                
            # Extract features
            time_series = data["Time Series (Daily)"]
            dates = list(time_series.keys())
            features = []
            targets = []
            
            if len(dates) < lookback + prediction_days:
                logging.warning(f"Insufficient historical data for {symbol}. Need at least {lookback + prediction_days} days.")
                return None, None, None
            
            feature_names = []
            
            for i in range(len(dates) - lookback - prediction_days):
                feature_window = []
                for j in range(lookback):
                    current_date = dates[i + j]
                    daily_data = time_series[current_date]
                    
                    # Extract base price data
                    open_price = float(daily_data['1. open'])
                    high_price = float(daily_data['2. high'])
                    low_price = float(daily_data['3. low'])
                    close_price = float(daily_data['4. close'])
                    volume = float(daily_data['5. volume'])
                    
                    # Create feature vector with additional derived features
                    feature_vector = [
                        open_price,
                        high_price,
                        low_price,
                        close_price,
                        volume,
                        high_price - low_price,  # Daily range
                        (close_price - open_price) / open_price,  # Daily return
                        (volume - float(time_series[dates[i + j - 1]]['5. volume'])) / float(time_series[dates[i + j - 1]]['5. volume']) if j > 0 else 0  # Volume change
                    ]
                    
                    # Only create feature names once
                    if i == 0 and j == 0:
                        feature_names = [
                            'open', 'high', 'low', 'close', 'volume', 
                            'range', 'return', 'volume_change'
                        ]
                        
                    feature_window.extend(feature_vector)
                
                # Target is the closing price 'prediction_days' days ahead
                target_date = dates[i + lookback + prediction_days - 1]
                target = float(time_series[target_date]['4. close'])
                
                features.append(feature_window)
                targets.append(target)
            
            return np.array(features), np.array(targets), feature_names
            
        except Exception as e:
            logging.error(f"Error preparing data for {symbol}: {str(e)}")
            return None, None, None
    
    def train_model(
        self, 
        symbol: str, 
        lookback: int = None, 
        prediction_days: int = None,
        hyperparameters: Dict[str, Any] = None
    ) -> bool:
        """
        Train a machine learning model for price prediction.
        
        Args:
            symbol: Stock symbol to train for
            lookback: Number of days to look back for features
            prediction_days: Number of days ahead to predict
            hyperparameters: Custom hyperparameters for the model
            
        Returns:
            Boolean indicating training success
        """
        lookback = lookback or self.default_lookback
        prediction_days = prediction_days or self.default_prediction_days
        
        # Default hyperparameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Override defaults with custom hyperparameters
        model_params = {**default_params, **(hyperparameters or {})}
        
        try:
            X, y, feature_names = self.prepare_data(symbol, lookback, prediction_days)
            if X is None or y is None or len(X) < 100:  # Ensure enough data
                logging.warning(f"Insufficient data for training model for {symbol}")
                return False
            
            # Initialize scalers for features and target
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            # Scale the data
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_scaled, test_size=0.2, random_state=42
            )
            
            # Train model
            model = RandomForestRegressor(**model_params)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y_scaled, cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            # Store model, scalers, and metadata
            self.models[symbol] = model
            self.scalers[symbol] = {'X': scaler_X, 'y': scaler_y}
            
            # Create and store model metadata
            version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            self.model_metadata[symbol] = {
                'symbol': symbol,
                'version': version,
                'created_at': datetime.datetime.now().isoformat(),
                'features': feature_names or [],
                'lookback': lookback,
                'prediction_days': prediction_days,
                'metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'cv_rmse': float(cv_rmse)
                },
                'hyperparameters': model_params
            }
            
            # Save the model
            self.save_model(symbol)
            
            logging.info(f"Successfully trained model for {symbol} (R²: {r2:.3f}, RMSE: {np.sqrt(mse):.3f})")
            return True
            
        except Exception as e:
            logging.error(f"Error training model for {symbol}: {str(e)}")
            return False
    
    def save_model(self, symbol: str) -> bool:
        """
        Save the trained model, scalers, and metadata to disk.
        
        Args:
            symbol: Symbol of the model to save
            
        Returns:
            Boolean indicating save success
        """
        if symbol not in self.models or symbol not in self.scalers:
            logging.error(f"No model found for {symbol}")
            return False
            
        try:
            # Create symbol directory
            symbol_dir = self.model_dir / symbol
            symbol_dir.mkdir(exist_ok=True)
            
            # Get version from metadata
            version = self.model_metadata[symbol]['version']
            
            # Save model
            joblib.dump(
                self.models[symbol],
                symbol_dir / f"model_{version}.joblib"
            )
            
            # Save scalers
            joblib.dump(
                self.scalers[symbol],
                symbol_dir / f"scalers_{version}.joblib"
            )
            
            # Save metadata
            with open(symbol_dir / f"metadata_{version}.json", 'w') as f:
                import json
                json.dump(self.model_metadata[symbol], f, indent=2)
                
            # Create a symlink to the latest version
            latest_model = symbol_dir / "model_latest.joblib"
            latest_scalers = symbol_dir / "scalers_latest.joblib"
            latest_metadata = symbol_dir / "metadata_latest.json"
            
            # Remove existing symlinks if they exist
            for link in [latest_model, latest_scalers, latest_metadata]:
                if link.exists():
                    link.unlink()
                
            # Create new symlinks
            latest_model.symlink_to(f"model_{version}.joblib")
            latest_scalers.symlink_to(f"scalers_{version}.joblib")
            latest_metadata.symlink_to(f"metadata_{version}.json")
            
            logging.info(f"Model for {symbol} (v{version}) saved successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error saving model for {symbol}: {str(e)}")
            return False
    
    def load_model(self, symbol: str, version: str = "latest") -> bool:
        """
        Load a trained model, scalers, and metadata from disk.
        
        Args:
            symbol: Symbol of the model to load
            version: Model version to load ("latest" or specific version ID)
            
        Returns:
            Boolean indicating load success
        """
        try:
            symbol_dir = self.model_dir / symbol
            
            if not symbol_dir.exists():
                logging.warning(f"No models found for {symbol}")
                return False
                
            # Determine which files to load
            model_file = symbol_dir / f"model_{version}.joblib"
            scalers_file = symbol_dir / f"scalers_{version}.joblib"
            metadata_file = symbol_dir / f"metadata_{version}.json"
            
            # Check if files exist
            if not model_file.exists() or not scalers_file.exists() or not metadata_file.exists():
                logging.warning(f"Could not find {version} model files for {symbol}")
                return False
                
            # Load model and scalers
            self.models[symbol] = joblib.load(model_file)
            self.scalers[symbol] = joblib.load(scalers_file)
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                import json
                self.model_metadata[symbol] = json.load(f)
                
            logging.info(f"Model for {symbol} (v{version}) loaded successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error loading model for {symbol}: {str(e)}")
            return False
    
    def predict_price(
        self, 
        symbol: str, 
        current_data: List[float] = None,
        confidence_level: float = 0.95
    ) -> Optional[PredictionResult]:
        """
        Predict future price movement with confidence intervals.
        
        Args:
            symbol: Stock symbol to predict for
            current_data: Current market data (if None, will fetch latest data)
            confidence_level: Confidence level for prediction intervals (0-1)
            
        Returns:
            PredictionResult object with value, confidence and bounds, or None if prediction fails
        """
        # Check if we need to load or train the model
        if symbol not in self.models:
            model_loaded = self.load_model(symbol)
            if not model_loaded:
                logging.info(f"No existing model found for {symbol}, training new model")
                if not self.train_model(symbol):
                    logging.error(f"Failed to train model for {symbol}")
                    return None
        
        try:
            # If no data provided, fetch latest
            if current_data is None:
                lookback = self.model_metadata[symbol]['lookback']
                data = self.api.get_daily_data(symbol, output_size="compact")
                
                if not data or "Time Series (Daily)" not in data:
                    logging.error(f"Failed to retrieve current data for {symbol}")
                    return None
                    
                # Process the latest data
                time_series = data["Time Series (Daily)"]
                dates = list(time_series.keys())
                
                if len(dates) < lookback:
                    logging.warning(f"Insufficient recent data for {symbol}. Need at least {lookback} days.")
                    return None
                
                # Prepare feature vector from recent data
                feature_window = []
                for j in range(lookback):
                    idx = min(j, len(dates) - 1)  # Avoid index errors
                    current_date = dates[idx]
                    daily_data = time_series[current_date]
                    
                    # Extract base price data
                    open_price = float(daily_data['1. open'])
                    high_price = float(daily_data['2. high'])
                    low_price = float(daily_data['3. low'])
                    close_price = float(daily_data['4. close'])
                    volume = float(daily_data['5. volume'])
                    
                    # Calculate derived features
                    daily_range = high_price - low_price
                    daily_return = (close_price - open_price) / open_price if open_price > 0 else 0
                    
                    # Volume change requires previous day's data
                    volume_change = 0
                    if j > 0 and idx + 1 < len(dates):
                        prev_date = dates[idx + 1]
                        prev_volume = float(time_series[prev_date]['5. volume'])
                        volume_change = (volume - prev_volume) / prev_volume if prev_volume > 0 else 0
                    
                    # Create feature vector
                    feature_vector = [
                        open_price, high_price, low_price, close_price, volume,
                        daily_range, daily_return, volume_change
                    ]
                    
                    feature_window.extend(feature_vector)
                
                current_data = feature_window
            
            # Scale the input data
            if symbol not in self.scalers:
                logging.error(f"No scalers found for {symbol}")
                return None
                
            scaler_X = self.scalers[symbol]['X']
            scaler_y = self.scalers[symbol]['y']
            
            # Reshape and scale input data
            X = np.array([current_data]).reshape(1, -1)
            X_scaled = scaler_X.transform(X)
            
            # Get base prediction
            model = self.models[symbol]
            
            # Make predictions with the model and all trees
            predictions = []
            for tree in model.estimators_:
                predictions.append(tree.predict(X_scaled)[0])
            
            # Calculate statistics
            mean_pred = np.mean(predictions)
            std_dev = np.std(predictions)
            
            # Calculate confidence interval
            from scipy import stats
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin = z_score * std_dev
            
            # Transform predictions back to original scale
            mean_pred_orig = scaler_y.inverse_transform([[mean_pred]])[0][0]
            lower_bound = scaler_y.inverse_transform([[mean_pred - margin]])[0][0]
            upper_bound = scaler_y.inverse_transform([[mean_pred + margin]])[0][0]
            
            # Calculate confidence score (normalized inverse of standard deviation)
            confidence_score = 1.0 - min(std_dev / (max(predictions) - min(predictions) + 1e-10), 1.0)
            
            # Create and cache the prediction result
            result = PredictionResult(
                value=float(mean_pred_orig),
                confidence=float(confidence_score),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound)
            )
            
            # Cache the result
            cache_key = f"{symbol}_{datetime.datetime.now().strftime('%Y%m%d')}"
            self.prediction_cache[cache_key] = result
            
            logging.info(f"Price prediction for {symbol}: {result.value:.2f} [{result.lower_bound:.2f} - {result.upper_bound:.2f}], confidence: {result.confidence:.2f}")
            return result
            
        except Exception as e:
            logging.error(f"Error predicting price for {symbol}: {str(e)}")
            return None
    
    def detect_patterns(
        self,
        symbol: str,
        lookback_days: int = 60
    ) -> Dict[str, Any]:
        """
        Detect common market patterns in the price data.
        
        Args:
            symbol: Stock symbol to analyze
            lookback_days: Number of days to analyze for patterns
            
        Returns:
            Dictionary containing detected patterns and their probabilities
        """
        try:
            # Check if we have cached results
            cache_key = f"{symbol}_{lookback_days}_{datetime.datetime.now().strftime('%Y%m%d')}"
            if cache_key in self.pattern_cache:
                return self.pattern_cache[cache_key]
            
            # Get historical data
            data = self.api.get_daily_data(symbol, output_size="full")
            if not data or "Time Series (Daily)" not in data:
                logging.error(f"Failed to retrieve data for pattern detection for {symbol}")
                return {"error": "Failed to retrieve data"}
                
            # Extract price data
            time_series = data["Time Series (Daily)"]
            dates = list(time_series.keys())[:lookback_days]  # Get most recent days
            
            if len(dates) < lookback_days:
                logging.warning(f"Insufficient data for pattern detection. Got {len(dates)} days, need {lookback_days}.")
                return {"error": "Insufficient data"}
            
            # Extract price series
            closes = []
            opens = []
            highs = []
            lows = []
            volumes = []
            
            for date in dates:
                daily_data = time_series[date]
                opens.append(float(daily_data["1. open"]))
                highs.append(float(daily_data["2. high"]))
                lows.append(float(daily_data["3. low"]))
                closes.append(float(daily_data["4. close"]))
                volumes.append(float(daily_data["5. volume"]))
            
            # Reverse lists to have chronological order
            opens.reverse()
            highs.reverse()
            lows.reverse()
            closes.reverse()
            volumes.reverse()
            
            # Convert to numpy arrays
            opens_arr = np.array(opens)
            highs_arr = np.array(highs)
            lows_arr = np.array(lows)
            closes_arr = np.array(closes)
            volumes_arr = np.array(volumes)
            
            # Calculate returns
            returns = np.diff(closes_arr) / closes_arr[:-1]
            
            # Initialize pattern detection results
            patterns = {
                "trend": self._detect_trend(closes_arr),
                "support_resistance": self._detect_support_resistance(closes_arr, highs_arr, lows_arr),
                "volatility": self._calculate_volatility(returns),
                "momentum": self._detect_momentum(closes_arr, volumes_arr),
                "reversal_patterns": self._detect_reversal_patterns(opens_arr, highs_arr, lows_arr, closes_arr),
                "volume_patterns": self._detect_volume_patterns(closes_arr, volumes_arr)
            }
            
            # Cache the results
            self.pattern_cache[cache_key] = patterns
            
            return patterns
            
        except Exception as e:
            logging.error(f"Error detecting patterns for {symbol}: {str(e)}")
            return {"error": str(e)}
    
    def _detect_trend(self, closes: np.ndarray) -> Dict[str, Any]:
        """
        Detect price trends using linear regression and moving averages.
        
        Args:
            closes: Array of closing prices
            
        Returns:
            Dictionary with trend information
        """
        # Linear regression trend
        from sklearn.linear_model import LinearRegression
        
        x = np.arange(len(closes)).reshape(-1, 1)
        model = LinearRegression().fit(x, closes)
        slope = model.coef_[0]
        
        # Calculate moving averages
        ma_20 = np.mean(closes[-20:]) if len(closes) >= 20 else None
        ma_50 = np.mean(closes[-50:]) if len(closes) >= 50 else None
        ma_200 = np.mean(closes[-200:]) if len(closes) >= 200 else None
        
        # Determine trend strength and direction
        trend_strength = abs(slope) / np.mean(closes) * 100  # As percentage of price
        
        latest_close = closes[-1]
        
        trend_signals = []
        
        # Check for golden/death crosses (MA crossovers)
        if ma_20 is not None and ma_50 is not None:
            if ma_20 > ma_50 and closes[-10] < ma_50:  # Golden cross (20 crossed above 50)
                trend_signals.append("golden_cross")
            elif ma_20 < ma_50 and closes[-10] > ma_50:  # Death cross (20 crossed below 50)
                trend_signals.append("death_cross")
        
        # Determine trend direction based on multiple factors
        if slope > 0 and (ma_20 is None or latest_close > ma_20):
            direction = "strong_uptrend" if trend_strength > 1.0 else "uptrend"
        elif slope > 0:
            direction = "weak_uptrend"
        elif slope < 0 and (ma_20 is None or latest_close < ma_20):
            direction = "strong_downtrend" if trend_strength > 1.0 else "downtrend"
        elif slope < 0:
            direction = "weak_downtrend"
        else:
            direction = "sideways"
        
        return {
            "direction": direction,
            "strength": float(trend_strength),
            "slope": float(slope),
            "signals": trend_signals,
            "moving_averages": {
                "ma_20": float(ma_20) if ma_20 is not None else None,
                "ma_50": float(ma_50) if ma_50 is not None else None,
                "ma_200": float(ma_200) if ma_200 is not None else None
            }
        }
    
    def _detect_support_resistance(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray) -> Dict[str, Any]:
        """
        Detect support and resistance levels using local extrema.
        
        Args:
            closes: Array of closing prices
            highs: Array of high prices
            lows: Array of low prices
            
        Returns:
            Dictionary with support and resistance levels
        """
        # Find local minima and maxima
        from scipy.signal import argrelextrema
        import numpy as np
        
        # Use a window of 5 days
        order = 5
        
        # Find local maxima (resistance candidates)
        maximums = argrelextrema(highs, np.greater, order=order)[0]
        resistance_levels = [float(highs[i]) for i in maximums]
        
        # Find local minima (support candidates)
        minimums = argrelextrema(lows, np.less, order=order)[0]
        support_levels = [float(lows[i]) for i in minimums]
        
        # Cluster nearby levels (within 2% of current price)
        current_price = closes[-1]
        tolerance = current_price * 0.02
        
        # Cluster support levels
        clustered_support = []
        for level in support_levels:
            # Skip levels that are too far from current price (> 20%)
            if abs(level - current_price) / current_price > 0.2:
                continue
                
            # Check if this level is close to an existing cluster
            added_to_cluster = False
            for i, cluster in enumerate(clustered_support):
                if abs(level - cluster["level"]) < tolerance:
                    # Add to existing cluster
                    clustered_support[i]["count"] += 1
                    clustered_support[i]["strength"] += 1
                    added_to_cluster = True
                    break
                    
            # If not added to a cluster, create a new one
            if not added_to_cluster:
                clustered_support.append({
                    "level": level,
                    "count": 1,
                    "strength": 1
                })
                
        # Cluster resistance levels
        clustered_resistance = []
        for level in resistance_levels:
            # Skip levels that are too far from current price (> 20%)
            if abs(level - current_price) / current_price > 0.2:
                continue
                
            # Check if this level is close to an existing cluster
            added_to_cluster = False
            for i, cluster in enumerate(clustered_resistance):
                if abs(level - cluster["level"]) < tolerance:
                    # Add to existing cluster
                    clustered_resistance[i]["count"] += 1
                    clustered_resistance[i]["strength"] += 1
                    added_to_cluster = True
                    break
                    
            # If not added to a cluster, create a new one
            if not added_to_cluster:
                clustered_resistance.append({
                    "level": level,
                    "count": 1,
                    "strength": 1
                })
                
        # Sort by strength
        clustered_support.sort(key=lambda x: x["strength"], reverse=True)
        clustered_resistance.sort(key=lambda x: x["strength"], reverse=True)
        
        # Determine if price is near a support or resistance
        price_at_support = False
        price_at_resistance = False
        
        for support in clustered_support:
            if current_price <= support["level"] * 1.02:  # Within 2% above support
                support["active"] = True
                price_at_support = True
            else:
                support["active"] = False
                
        for resistance in clustered_resistance:
            if current_price >= resistance["level"] * 0.98:  # Within 2% below resistance
                resistance["active"] = True
                price_at_resistance = True
            else:
                resistance["active"] = False
                
        return {
            "support_levels": clustered_support[:3],  # Return top 3 support levels
            "resistance_levels": clustered_resistance[:3],  # Return top 3 resistance levels
            "price_at_support": price_at_support,
            "price_at_resistance": price_at_resistance,
            "current_price": float(current_price)
        }
    
    def _calculate_volatility(self, returns: np.ndarray) -> Dict[str, Any]:
        """
        Calculate volatility metrics for price returns.
        
        Args:
            returns: Array of price returns
            
        Returns:
            Dictionary with volatility metrics
        """
        try:
            # Standard deviation of returns (annualized)
            std_dev = np.std(returns)
            annualized_volatility = std_dev * np.sqrt(252)  # Assuming 252 trading days in a year
            
            # Calculate historical volatility percentile
            rolling_std = [np.std(returns[max(0, i-20):i+1]) for i in range(len(returns)) if i >= 5]
            current_vol = rolling_std[-1] if rolling_std else std_dev
            
            # Determine if volatility is increasing or decreasing
            vol_trend = "increasing" if len(rolling_std) > 10 and current_vol > np.mean(rolling_std[-10:]) else "decreasing"
            
            # Volatility classification
            vol_level = "high" if annualized_volatility > 0.3 else "medium" if annualized_volatility > 0.15 else "low"
            
            # Calculate Average True Range (ATR) for the last 14 periods
            if len(returns) >= 14:
                atr_value = np.mean(np.abs(returns[-14:]))
            else:
                atr_value = np.mean(np.abs(returns))
                
            return {
                "daily_volatility": float(std_dev),
                "annualized_volatility": float(annualized_volatility),
                "volatility_level": vol_level,
                "volatility_trend": vol_trend,
                "atr_14": float(atr_value)
            }
            
        except Exception as e:
            logging.error(f"Error calculating volatility: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _detect_momentum(self, closes: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Detect momentum patterns in price and volume data.
        
        This method identifies various momentum indicators, including:
        - Relative Strength Index (RSI)
        - Rate of Change (ROC)
        - Moving Average Convergence Divergence (MACD)
        - On-Balance Volume (OBV)
        - Volume Price Trend (VPT)
        
        Args:
            closes: Array of closing prices
            volumes: Array of trading volumes
            
        Returns:
            Dictionary with momentum indicators and signals
        """
        try:
            # Calculate RSI (Relative Strength Index)
            def calculate_rsi(prices, period=14):
                # Calculate price changes
                deltas = np.diff(prices)
                seed = deltas[:period+1]
                up = seed[seed >= 0].sum() / period
                down = -seed[seed < 0].sum() / period
                
                if down == 0:  # Avoid division by zero
                    return 100
                    
                rs = up / down
                rsi = np.zeros_like(prices)
                rsi[:period] = 100. - 100. / (1. + rs)
                
                # Calculate RSI for remaining data
                for i in range(period, len(prices)):
                    delta = deltas[i-1]
                    if delta > 0:
                        upval = delta
                        downval = 0
                    else:
                        upval = 0
                        downval = -delta
                        
                    up = (up * (period - 1) + upval) / period
                    down = (down * (period - 1) + downval) / period
                    
                    if down == 0:  # Avoid division by zero
                        rsi[i] = 100
                    else:
                        rs = up / down
                        rsi[i] = 100. - 100. / (1. + rs)
                        
                return rsi
                
            # Calculate Rate of Change (ROC)
            def calculate_roc(prices, period=14):
                roc = np.zeros_like(prices)
                for i in range(period, len(prices)):
                    roc[i] = ((prices[i] - prices[i-period]) / prices[i-period]) * 100
                return roc
                
            # Calculate MACD (Moving Average Convergence Divergence)
            def calculate_macd(prices, fast=12, slow=26, signal=9):
                # Calculate EMAs
                ema_fast = np.zeros_like(prices)
                ema_slow = np.zeros_like(prices)
                
                # Initialize EMAs with SMA
                ema_fast[:fast] = np.mean(prices[:fast])
                ema_slow[:slow] = np.mean(prices[:slow])
                
                # Calculate EMAs
                alpha_fast = 2 / (fast + 1)
                alpha_slow = 2 / (slow + 1)
                
                for i in range(fast, len(prices)):
                    ema_fast[i] = prices[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
                    
                for i in range(slow, len(prices)):
                    ema_slow[i] = prices[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
                    
                # Calculate MACD line
                macd_line = ema_fast - ema_slow
                
                # Calculate signal line (EMA of MACD line)
                signal_line = np.zeros_like(macd_line)
                signal_line[:slow+signal] = np.mean(macd_line[slow:slow+signal])
                
                alpha_signal = 2 / (signal + 1)
                for i in range(slow+signal, len(macd_line)):
                    signal_line[i] = macd_line[i] * alpha_signal + signal_line[i-1] * (1 - alpha_signal)
                    
                # Calculate histogram
                histogram = macd_line - signal_line
                
                return macd_line, signal_line, histogram
                
            # Calculate On-Balance Volume (OBV)
            def calculate_obv(closes, volumes):
                obv = np.zeros_like(closes)
                obv[0] = volumes[0]
                
                for i in range(1, len(closes)):
                    if closes[i] > closes[i-1]:
                        obv[i] = obv[i-1] + volumes[i]
                    elif closes[i] < closes[i-1]:
                        obv[i] = obv[i-1] - volumes[i]
                    else:
                        obv[i] = obv[i-1]
                        
                return obv

            # Get momentum indicators
            rsi = calculate_rsi(closes)
            roc = calculate_roc(closes)
            macd_line, macd_signal, macd_histogram = calculate_macd(closes)
            obv = calculate_obv(closes, volumes)
            
            # Get latest values
            current_rsi = rsi[-1]
            current_roc = roc[-1]
            current_macd = macd_line[-1]
            current_macd_signal = macd_signal[-1]
            current_macd_histogram = macd_histogram[-1]
            
            # Determine momentum signals
            momentum_signals = []
            
            # RSI signals
            if current_rsi > 70:
                momentum_signals.append("overbought_rsi")
            elif current_rsi < 30:
                momentum_signals.append("oversold_rsi")
                
            # MACD signals
            if current_macd > current_macd_signal and macd_line[-2] <= macd_signal[-2]:
                momentum_signals.append("macd_bullish_crossover")
            elif current_macd < current_macd_signal and macd_line[-2] >= macd_signal[-2]:
                momentum_signals.append("macd_bearish_crossover")
                
            # Rate of change signals
            if current_roc > 5:
                momentum_signals.append("strong_momentum")
            elif current_roc < -5:
                momentum_signals.append("strong_negative_momentum")
                
            # OBV trend
            obv_trend = "rising" if np.mean(obv[-5:]) > np.mean(obv[-10:-5]) else "falling"
            
            # Determine overall momentum
            if current_rsi > 60 and current_roc > 0 and current_macd > 0:
                momentum_strength = "strong_bullish"
            elif current_rsi > 50 and current_roc > 0:
                momentum_strength = "bullish"
            elif current_rsi < 40 and current_roc < 0 and current_macd < 0:
                momentum_strength = "strong_bearish"
            elif current_rsi < 50 and current_roc < 0:
                momentum_strength = "bearish"
            else:
                momentum_strength = "neutral"
                
            return {
                "indicators": {
                    "rsi": float(current_rsi),
                    "roc": float(current_roc),
                    "macd": float(current_macd),
                    "macd_signal": float(current_macd_signal),
                    "macd_histogram": float(current_macd_histogram),
                    "obv_trend": obv_trend
                },
                "signals": momentum_signals,
                "strength": momentum_strength
            }
            
        except Exception as e:
            logging.error(f"Error detecting momentum: {str(e)}")
            return {
                "error": str(e)
            }
    
    def _detect_volume_patterns(self, closes: np.ndarray, volumes: np.ndarray) -> Dict[str, Any]:
        """
        Detect volume-based patterns and anomalies.
        
        This method identifies various volume patterns that may indicate trend strength,
        potential reversals, or breakouts, including:
        - Volume spikes
        - Volume trend divergence
        - Decreasing volume in trends
        - Volume breakouts
        - Accumulation/distribution patterns
        
        Args:
            closes: Array of closing prices
            volumes: Array of trading volumes
            
        Returns:
            Dictionary with detected volume patterns and their significance
        """
        try:
            # Check if we have enough data
            if len(volumes) < 10:
                return {"error": "Insufficient data for volume pattern detection"}
            
            # Initialize results
            volume_patterns = []
            pattern_details = {}
            
            # Calculate basic volume metrics
            avg_volume = np.mean(volumes[-10:])
            max_volume = np.max(volumes[-10:])
            min_volume = np.min(volumes[-10:])
            latest_volume = volumes[-1]
            
            # Calculate volume moving averages
            vol_ma5 = np.mean(volumes[-5:])
            vol_ma20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else None
            
            # Calculate price changes
            price_changes = np.diff(closes)
            
            # Calculate volume trend
            volume_trend = "increasing" if vol_ma5 > avg_volume else "decreasing"
            
            # Check for volume spike (significantly higher than average)
            if latest_volume > 2 * avg_volume:
                volume_patterns.append("volume_spike")
                pattern_details["volume_spike"] = {
                    "significance": "high",
                    "value": float(latest_volume / avg_volume)
                }
            
            # Check for unusually low volume
            if latest_volume < 0.5 * avg_volume:
                volume_patterns.append("low_volume")
                pattern_details["low_volume"] = {
                    "significance": "medium",
                    "value": float(latest_volume / avg_volume)
                }
            
            # Check for volume and price divergence
            # (Price making new highs with decreasing volume or new lows with decreasing volume)
            if len(closes) >= 10 and len(volumes) >= 10:
                price_increasing = closes[-1] > np.max(closes[-10:-1])
                price_decreasing = closes[-1] < np.min(closes[-10:-1])
                volume_decreasing = volumes[-1] < np.mean(volumes[-5:])
                
                if price_increasing and volume_decreasing:
                    volume_patterns.append("bearish_volume_divergence")
                    pattern_details["bearish_volume_divergence"] = {
                        "significance": "high"
                    }
                elif price_decreasing and volume_decreasing:
                    volume_patterns.append("bullish_volume_divergence")
                    pattern_details["bullish_volume_divergence"] = {
                        "significance": "high"
                    }
            
            # Check for Chaikin Money Flow pattern
            def calculate_cmf(closes, highs, lows, volumes, period=20):
                """Calculate Chaikin Money Flow"""
                if len(closes) < period:
                    return None
                
                money_flow_volume = []
                for i in range(len(closes)):
                    if highs[i] == lows[i]:  # Avoid division by zero
                        money_flow_multiplier = 0
                    else:
                        money_flow_multiplier = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
                    money_flow_volume.append(money_flow_multiplier * volumes[i])
                
                cmf = []
                for i in range(period-1, len(closes)):
                    period_volume_sum = np.sum(volumes[i-(period-1):i+1])
                    period_mfv_sum = np.sum(money_
        """
        Detect common price reversal patterns in candlestick data.
        
        This method identifies various candlestick patterns that may indicate potential market reversals, including:
        - Doji patterns
        - Hammers and hanging men
        - Engulfing patterns
        - Morning/Evening stars
        - Head and shoulders patterns
        - Double tops/bottoms
        
        Args:
            opens: Array of opening prices
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of closing prices
            
        Returns:
            Dictionary with detected reversal patterns and their probabilities
        """
        try:
            # Check for minimum data requirements
            if len(closes) < 5:
                return {"error": "Insufficient data for pattern detection"}
            
            patterns = []
            pattern_details = {}
            
            # Helper functions for candlestick patterns
            def is_doji(idx):
                """Check if candle at idx is a doji (open and close are very close)"""
                body_size = abs(closes[idx] - opens[idx])
                high_low_range = highs[idx] - lows[idx]
                return body_size <= 0.1 * high_low_range
            
            def is_hammer(idx):
                """Check for hammer (bullish) or hanging man (bearish) pattern"""
                body_size = abs(closes[idx] - opens[idx])
                high_low_range = highs[idx] - lows[idx]
                upper_shadow = highs[idx] - max(opens[idx], closes[idx])
                lower_shadow = min(opens[idx], closes[idx]) - lows[idx]
                
                # Hammer conditions: small body, long lower shadow, small upper shadow
                return (body_size <= 0.3 * high_low_range and 
                        lower_shadow >= 2 * body_size and 
                        upper_shadow <= 0.1 * high_low_range)
            
            def is_engulfing(idx):
                """Check for bullish or bearish engulfing pattern"""
                # Need at least one previous candle
                if idx < 1:
                    return None
                    
                # Current and previous candle data
                curr_open, curr_close = opens[idx], closes[idx]
                prev_open, prev_close = opens[idx-1], closes[idx-1]
                
                # Bullish engulfing (current candle is bullish and engulfs previous bearish candle)
                if (curr_close > curr_open and  # Current is bullish
                    prev_close < prev_open and  # Previous is bearish
                    curr_open <= prev_close and  # Current open below previous close
                    curr_close >= prev_open):   # Current close above previous open
                    return "bullish_engulfing"
                    
                # Bearish engulfing (current candle is bearish and engulfs previous bullish candle)
                elif (curr_close < curr_open and  # Current is bearish
                      prev_close > prev_open and  # Previous is bullish
                      curr_open >= prev_close and  # Current open above previous close
                      curr_close <= prev_open):   # Current close below previous open
                    return "bearish_engulfing"
                    
                return None
            
            def is_morning_star(idx):
                """Check for morning star pattern (bullish reversal)"""
                # Need at least two previous candles
                if idx < 2:
                    return False
                    
                # First candle is bearish with large body
                first_body = abs(opens[idx-2] - closes[idx-2])
                first_bearish = closes[idx-2] < opens[idx-2]
                
                # Second candle is small (doji-like)
                second_body = abs(opens[idx-1] - closes[idx-1])
                second_small = second_body < 0.3 * first_body
                
                # Third candle is bullish with large body, closing into first candle
                third_body = abs(opens[idx] - closes[idx])
                third_bullish = closes[idx] > opens[idx]
                penetration = (closes[idx] > (opens[idx-2] + closes[idx-2]) / 2)
                
                return (first_bearish and second_small and third_bullish and 
                        third_body > 0.6 * first_body and penetration)
            
            def is_evening_star(idx):
                """Check for evening star pattern (bearish reversal)"""
                # Need at least two previous candles
                if idx < 2:
                    return False
                    
                # First candle is bullish with large body
                first_body = abs(opens[idx-2] - closes[idx-2])
                first_bullish = closes[idx-2] > opens[idx-2]
                
                # Second candle is small (doji-like)
                second_body = abs(opens[idx-1] - closes[idx-1])
                second_small = second_body < 0.3 * first_body
                
                # Third candle is bearish with large body, closing into first candle
                third_body = abs(opens[idx] - closes[idx])
                third_bearish = closes[idx] < opens[idx]
                penetration = (closes[idx] < (opens[idx-2] + closes[idx-2]) / 2)
                
                return (first_bullish and second_small and third_bearish and 
                        third_body > 0.6 * first_body and penetration)
            
            def is_shooting_star(idx):
                """Check for shooting star pattern (bearish reversal)"""
                if idx < 1:
                    return False
                    
                body_size = abs(opens[idx] - closes[idx])
                high_low_range = highs[idx] - lows[idx]
                upper_shadow = highs[idx] - max(opens[idx], closes[idx])
                lower_shadow = min(opens[idx], closes[idx]) - lows[idx]
                
                # Shooting star conditions: small body at bottom, long upper shadow, minimal lower shadow
                return (body_size <= 0.3 * high_low_range and 
                        upper_shadow >= 2 * body_size and 
                        lower_shadow <= 0.1 * high_low_range and
                        closes[idx-1] > opens[idx-1])  # Previous candle was bullish
            
            def is_harami(idx):
                """Check for harami pattern (potential reversal)"""
                if idx < 1:
                    return None
                    
                curr_body = abs(opens[idx] - closes[idx])
                prev_body = abs(opens[idx-1] - closes[idx-1])
                
                # Current candle body must be inside previous candle body
                curr_inside_prev = (max(opens[idx], closes[idx]) < max(opens[idx-1], closes[idx-1]) and
                                   min(opens[idx], closes[idx]) > min(opens[idx-1], closes[idx-1]))
                
                # Current candle body must be smaller than previous
                curr_smaller = curr_body < 0.7 * prev_body
                
                if curr_inside_prev and curr_smaller:
                    # Bullish harami (previous bearish, current bullish)
                    if closes[idx-1] < opens[idx-1] and closes[idx] > opens[idx]:
                        return "bullish_harami"
                    # Bearish harami (previous bullish, current bearish)
                    elif closes[idx-1] > opens[idx-1] and closes[idx] < opens[idx]:
                        return "bearish_harami"
                
                return None
            
            def is_double_top_bottom(data, window=20):
                """Check for double top or double bottom patterns"""
                # Need enough data for pattern
                if len(data) < window:
                    return None
                    
                # Get local maxima and minima
                from scipy.signal import find_peaks
                
                # Find potential tops (peaks)
                peaks, _ = find_peaks(data, distance=5)
                if len(peaks) >= 2:
                    # Check the last two peaks for similar heights (within 2%)
                    peak1, peak2 = peaks[-2], peaks[-1]
                    if abs(data[peak1] - data[peak2]) / data[peak1] < 0.02:
                        return "double_top"
                
                # Find potential bottoms (valleys)
                valleys, _ = find_peaks(-data, distance=5)
                if len(valleys) >= 2:
                    # Check the last two valleys for similar depths (within 2%)
                    valley1, valley2 = valleys[-2], valleys[-1]
                    if abs(data[valley1] - data[valley2]) / data[valley1] < 0.02:
                        return "double_bottom"
                
                return None
            
            # Check for patterns in the recent data
            for i in range(min(20, len(closes) - 1), len(closes)):
                # Check individual candle patterns
                if is_doji(i):
                    patterns.append("doji")
                    pattern_details["doji"] = {"day": i, "significance": "medium"}
                
                if is_hammer(i):
                    # Determine if it's a hammer (bullish) or hanging man (bearish)
                    # Hammer occurs in downtrend, hanging man in uptrend
                    downtrend = np.mean(closes[i-5:i]) > closes[i]
                    pattern_type = "hammer" if downtrend else "hanging_man"
                    patterns.append(pattern_type)
                    pattern_details[pattern_type] = {"day": i, "significance": "high"}
                
                # Check multi-candle patterns
                engulfing = is_engulfing(i)
                if engulfing:
                    patterns.append(engulfing)
                    pattern_details[engulfing] = {"day": i, "significance": "high"}
                
                if is_morning_star(i):
                    patterns.append("morning_star")
                    pattern_details["morning_star"] = {"day": i, "significance": "very_high"}
                
                if is_evening_star(i):
                    patterns.append("evening_star")
                    pattern_details["evening_star"] = {"day": i, "significance": "very_high"}
                
                if is_shooting_star(i):
                    patterns.append("shooting_star")
                    pattern_details["shooting_star"] = {"day": i, "significance": "high"}
                
                harami = is_harami(i)
                if harami:
                    patterns.append(harami)
                    pattern_details[harami] = {"day": i, "significance": "medium"}
            
            # Check for double tops/bottoms using the entire closing price array
            double_pattern = is_double_top_bottom(closes)
            if double_pattern:
                patterns.append(double_pattern)
                pattern_details[double_pattern] = {"significance": "very_high"}
            
            # Determine overall reversal signals
            has_bullish_reversal = any(p in patterns for p in 
                                      ["hammer", "morning_star", "bullish_engulfing", 
                                       "bullish_harami", "double_bottom"])
            
            has_bearish_reversal = any(p in patterns for p in 
                                      ["hanging_man", "evening_star", "bearish_engulfing", 
                                       "bearish_harami", "double_top", "shooting_star"])
            
            # Return the results
            return {
                "patterns": patterns,
                "pattern_details": pattern_details,
                "bullish_reversal_signal": has_bullish_reversal,
                "bearish_reversal_signal": has_bearish_reversal
            }
