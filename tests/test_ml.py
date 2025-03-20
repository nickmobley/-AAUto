import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import datetime
import tempfile
import shutil
from pathlib import Path

from src.ml.predictor import MachineLearning, PredictionResult, ModelMetadata


class TestMachineLearning:
    """Test suite for the MachineLearning class."""

    @pytest.fixture
    def mock_api(self):
        """Create a mock API client for testing."""
        mock_api = Mock()
        
        # Mock daily data response
        mock_api.get_daily_data.return_value = {
            "Time Series (Daily)": {
                "2023-01-30": {
                    "1. open": "150.10", 
                    "2. high": "155.20", 
                    "3. low": "149.50", 
                    "4. close": "154.80", 
                    "5. volume": "10000000"
                },
                "2023-01-29": {
                    "1. open": "148.50", 
                    "2. high": "151.30", 
                    "3. low": "147.90", 
                    "4. close": "150.20", 
                    "5. volume": "9500000"
                },
                "2023-01-28": {
                    "1. open": "145.00", 
                    "2. high": "149.40", 
                    "3. low": "144.80", 
                    "4. close": "148.30", 
                    "5. volume": "8900000"
                },
                "2023-01-27": {
                    "1. open": "144.20", 
                    "2. high": "146.70", 
                    "3. low": "143.50", 
                    "4. close": "145.10", 
                    "5. volume": "8500000"
                },
                "2023-01-26": {
                    "1. open": "142.00", 
                    "2. high": "144.50", 
                    "3. low": "141.80", 
                    "4. close": "144.00", 
                    "5. volume": "8000000"
                }
            }
        }
        
        # Create a more extensive mock dataset for training
        extended_data = {"Time Series (Daily)": {}}
        base_date = datetime.datetime(2023, 1, 1)
        base_price = 100.0
        
        # Generate 200 days of sample data for training
        for i in range(200):
            date_str = (base_date + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            # Create some price momentum and randomness
            momentum = i * 0.1
            noise = np.random.normal(0, 2)
            price = base_price + momentum + noise
            
            extended_data["Time Series (Daily)"][date_str] = {
                "1. open": str(price - 1),
                "2. high": str(price + 1),
                "3. low": str(price - 2),
                "4. close": str(price),
                "5. volume": str(int(8000000 + i * 10000 + np.random.normal(0, 500000)))
            }
        
        # Create a method to return the extended dataset when needed
        def get_extended_data(symbol, output_size="full"):
            if output_size == "full":
                return extended_data
            else:
                # Return just the last 5 days for compact size
                compact_data = {"Time Series (Daily)": {}}
                dates = list(extended_data["Time Series (Daily)"].keys())[-5:]
                for date in dates:
                    compact_data["Time Series (Daily)"][date] = extended_data["Time Series (Daily)"][date]
                return compact_data
        
        # Replace the standard mock with our extended data function when needed
        mock_api.get_extended_data = get_extended_data
        
        return mock_api
    
    @pytest.fixture
    def ml_model(self, mock_api, tmp_path):
        """Create a MachineLearning instance with a temporary model directory."""
        model_dir = tmp_path / "models"
        return MachineLearning(mock_api, model_dir=str(model_dir))
    
    def test_initialization(self, ml_model, mock_api):
        """Test that the ML model initializes properly."""
        assert ml_model.api == mock_api
        assert ml_model.models == {}
        assert ml_model.scalers == {}
        assert ml_model.prediction_cache == {}
        assert ml_model.pattern_cache == {}
        assert ml_model.default_lookback == 30
        assert ml_model.default_prediction_days == 5
        assert Path(ml_model.model_dir).exists()
    
    def test_prepare_data_valid_input(self, ml_model, mock_api):
        """Test data preparation with valid input."""
        # Setup extended data for this test
        mock_api.get_daily_data = mock_api.get_extended_data
        
        # Test with minimal lookback to work with our test data
        X, y, feature_names = ml_model.prepare_data("AAPL", lookback=3, prediction_days=1)
        
        # Verify the results
        assert X is not None
        assert y is not None
        assert feature_names is not None
        assert len(feature_names) > 0
        assert X.shape[1] == len(feature_names) * 3  # 3 days of features
        assert len(y) == len(X)
    
    def test_prepare_data_insufficient_data(self, ml_model, mock_api):
        """Test data preparation with insufficient historical data."""
        # Set lookback beyond available data
        X, y, feature_names = ml_model.prepare_data("AAPL", lookback=10, prediction_days=5)
        
        # Should return None for all values due to insufficient data
        assert X is None
        assert y is None
        assert feature_names is None
    
    def test_prepare_data_api_error(self, ml_model, mock_api):
        """Test data preparation with API error."""
        # Simulate API error
        mock_api.get_daily_data.side_effect = Exception("API connection error")
        
        X, y, feature_names = ml_model.prepare_data("AAPL")
        
        # Should return None for all values due to API error
        assert X is None
        assert y is None
        assert feature_names is None
    
    @patch('joblib.dump')
    def test_train_model_success(self, mock_dump, ml_model, mock_api):
        """Test successful model training."""
        # Setup extended data for this test
        mock_api.get_daily_data = mock_api.get_extended_data
        
        # Mock the prepare_data method to return controlled test data
        with patch.object(ml_model, 'prepare_data') as mock_prepare:
            # Create synthetic data for testing
            n_samples = 100
            n_features = 24  # 3 days * 8 features per day
            X = np.random.rand(n_samples, n_features)
            y = np.random.rand(n_samples)
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
            mock_prepare.return_value = (X, y, feature_names)
            
            # Train the model
            result = ml_model.train_model("AAPL", lookback=3, prediction_days=1)
            
            # Verify results
            assert result is True
            assert "AAPL" in ml_model.models
            assert "AAPL" in ml_model.scalers
            assert "AAPL" in ml_model.model_metadata
            assert mock_dump.call_count >= 2  # Model and scaler should be saved
    
    def test_train_model_insufficient_data(self, ml_model, mock_api):
        """Test model training with insufficient data."""
        # Make prepare_data return None to simulate insufficient data
        with patch.object(ml_model, 'prepare_data') as mock_prepare:
            mock_prepare.return_value = (None, None, None)
            
            result = ml_model.train_model("AAPL")
            
            # Should return False due to insufficient data
            assert result is False
            assert "AAPL" not in ml_model.models
    
    def test_train_model_exception(self, ml_model, mock_api):
        """Test model training with exception."""
        # Make prepare_data raise an exception
        with patch.object(ml_model, 'prepare_data') as mock_prepare:
            mock_prepare.side_effect = Exception("Error preparing data")
            
            result = ml_model.train_model("AAPL")
            
            # Should return False due to exception
            assert result is False
            assert "AAPL" not in ml_model.models
    
    @patch('joblib.dump')
    def test_save_model_success(self, mock_dump, ml_model):
        """Test successful model saving."""
        # Create a model and metadata to save
        ml_model.models["AAPL"] = Mock()
        ml_model.scalers["AAPL"] = {"X": Mock(), "y": Mock()}
        ml_model.model_metadata["AAPL"] = {
            "symbol": "AAPL",
            "version": "20230101120000",
            "created_at": "2023-01-01T12:00:00",
            "features": ["feature1", "feature2"],
            "lookback": 30,
            "prediction_days": 5,
            "metrics": {"mse": 0.01, "r2": 0.95},
            "hyperparameters": {"n_estimators": 100}
        }
        
        # Test saving
        with patch('json.dump') as mock_json_dump:
            with patch('pathlib.Path.symlink_to') as mock_symlink:
                result = ml_model.save_model("AAPL")
                
                # Verify results
                assert result is True
                assert mock_dump.call_count == 2  # Model and scaler should be saved
                assert mock_json_dump.call_count == 1  # Metadata should be saved
                assert mock_symlink.call_count == 3  # Three symlinks should be created
    
    def test_save_model_no_model(self, ml_model):
        """Test saving when no model exists."""
        # Attempt to save non-existent model
        result = ml_model.save_model("AAPL")
        
        # Should return False
        assert result is False
    
    @patch('joblib.load')
    def test_load_model_success(self, mock_load, ml_model):
        """Test successful model loading."""
        # Setup mock files
        model_file = Path(ml_model.model_dir) / "AAPL" / "model_latest.joblib"
        scalers_file = Path(ml_model.model_dir) / "AAPL" / "scalers_latest.joblib"
        metadata_file = Path(ml_model.model_dir) / "AAPL" / "metadata_latest.json"
        
        # Create necessary directories
        (Path(ml_model.model_dir) / "AAPL").mkdir(parents=True, exist_ok=True)
        
        # Mock file checks
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = True
            
            # Mock loading files
            mock_load.return_value = Mock()
            
            with patch('json.load') as mock_json_load:
                mock_json_load.return_value = {
                    "symbol": "AAPL",
                    "version": "20230101120000",
                    "features": ["feature1", "feature2"],
                    "lookback": 30,
                    "prediction_days": 5
                }
                
                # Test loading
                with patch('builtins.open', MagicMock()):
                    result = ml_model.load_model("AAPL")
                    
                    # Verify results
                    assert result is True
                    assert "AAPL" in ml_model.models
                    assert "AAPL" in ml_model.scalers
                    assert "AAPL" in ml_model.model_metadata
    
    def test_load_model_missing_files(self, ml_model):
        """Test loading when files don't exist."""
        # Mock file checks to return False
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False
            
            result = ml_model.load_model("AAPL")
            
            # Should return False
            assert result is False
            assert "AAPL" not in ml_model.models
    
    def test_predict_price_with_model(self, ml_model, mock_api):
        """Test price prediction with an existing model."""
        # Setup a mock model and scalers
        ml_model.models["AAPL"] = Mock()
        ml_model.models["AAPL"].estimators_ = [Mock(), Mock(), Mock()]
        for estimator in ml_model.models["AAPL"].estimators_:
            estimator.predict.return_value = np.array([0.5])
        
        ml_model.scalers["AAPL"] = {
            "X": Mock(),
            "y": Mock()
        }
        ml_model.scalers["AAPL"]["X"].transform.return_value = np.array([[0.1, 0.2, 0.3]])
        ml_model.scalers["AAPL"]["y"].inverse_transform.return_value = np.array([[150.0]])
        
        ml_model.model_metadata["AAPL"] = {
            "lookback": 3
        }
        
        # Test prediction with provided data
        current_data = [0.1, 0.2, 0.3]
        result = ml_model.predict_price("AAPL", current_data)
        
        # Verify the result
        assert result is not None
        assert isinstance(result, PredictionResult)
        assert result.value == 150.0
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'lower_bound')
        assert hasattr(result, 'upper_bound')
    
    def test_predict_price_load_model(self, ml_model):
        """Test prediction triggering model loading."""
        # Mock load_model to return True
        with patch.object(ml_model, 'load_model') as mock_load:
            mock_load.return_value = True
            
            # Setup minimal mocks needed for predict

