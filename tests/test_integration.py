import os
import json
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from src.core.trader import Trader
from src.api.alpha_vantage import AlphaVantageAPI
from src.analytics.technical import TechnicalAnalyzer
from src.risk.manager import RiskManager
from src.ml.predictor import MachineLearning
from src.strategies.manager import StrategyManager
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.trader_main import TraderMain


class TestIntegration:
    @pytest.fixture
    def test_config(self):
        """Load test configuration"""
        config_path = os.path.join(os.path.dirname(__file__), 'test_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing"""
        # Generate 100 days of mock data for 4 symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
        data = {}
        
        for symbol in symbols:
            # Create price series with some trend, mean-reversion, and momentum patterns
            base_price = np.random.uniform(100, 500)
            trend = np.linspace(0, np.random.uniform(20, 50), 100)
            mean_rev = np.sin(np.linspace(0, 4*np.pi, 100)) * np.random.uniform(5, 15)
            momentum = np.cumsum(np.random.normal(0, 1, 100)) * np.random.uniform(2, 5)
            noise = np.random.normal(0, 2, 100)
            
            prices = base_price + trend + mean_rev + momentum + noise
            prices = np.maximum(prices, 1)  # Ensure no negative prices
            
            # Create volume with some correlation to price changes
            base_volume = np.random.uniform(1000000, 5000000)
            volume_noise = np.random.normal(0, 0.3, 100)
            price_diff = np.diff(prices, prepend=prices[0])
            volume_price_correlation = np.abs(price_diff) * np.random.uniform(50000, 200000)
            volume = base_volume + volume_price_correlation + volume_noise * base_volume
            volume = np.maximum(volume, 10000)  # Ensure reasonable volume
            
            # Calculate moving averages for convenience in tests
            sma_10 = np.concatenate([np.full(9, np.nan), np.convolve(prices, np.ones(10)/10, mode='valid')])
            sma_20 = np.concatenate([np.full(19, np.nan), np.convolve(prices, np.ones(20)/20, mode='valid')])
            sma_50 = np.concatenate([np.full(49, np.nan), np.convolve(prices, np.ones(50)/50, mode='valid')])
            
            # Create date index
            dates = pd.date_range(end=pd.Timestamp.today(), periods=100).strftime('%Y-%m-%d')
            
            # Create DataFrame with all required data
            df = pd.DataFrame({
                'date': dates,
                'open': prices * np.random.uniform(0.99, 1.0, 100),
                'high': prices * np.random.uniform(1.01, 1.03, 100),
                'low': prices * np.random.uniform(0.97, 0.99, 100),
                'close': prices,
                'volume': volume.astype(int),
                'sma_10': sma_10,
                'sma_20': sma_20,
                'sma_50': sma_50
            })
            
            data[symbol] = df
            
        return data
    
    @pytest.fixture
    def mock_api(self, mock_market_data):
        """Create a mock API that returns the test market data"""
        mock_api = MagicMock(spec=AlphaVantageAPI)
        
        def mock_get_daily_adjusted(symbol, **kwargs):
            if symbol in mock_market_data:
                return mock_market_data[symbol]
            else:
                raise ValueError(f"Symbol {symbol} not found in mock data")
        
        mock_api.get_daily_adjusted.side_effect = mock_get_daily_adjusted
        return mock_api
    
    @pytest.fixture
    def strategy_manager(self, mock_api, test_config):
        """Create a strategy manager with the test configuration"""
        # Create necessary dependencies
        technical_analyzer = TechnicalAnalyzer(mock_api)
        risk_manager = RiskManager(test_config['risk_management'])
        ml_predictor = MachineLearning(mock_api, test_config['machine_learning'])
        
        # Create strategies
        momentum = MomentumStrategy(
            mock_api, 
            technical_analyzer,
            test_config['strategies']['momentum']
        )
        
        mean_reversion = MeanReversionStrategy(
            mock_api,
            technical_analyzer,
            test_config['strategies']['mean_reversion']
        )
        
        trend_following = TrendFollowingStrategy(
            mock_api,
            technical_analyzer,
            test_config['strategies']['trend_following']
        )
        
        # Create strategy manager
        manager = StrategyManager(test_config['strategy_weights'])
        manager.add_strategy("momentum", momentum)
        manager.add_strategy("mean_reversion", mean_reversion)
        manager.add_strategy("trend_following", trend_following)
        
        return manager
    
    @pytest.fixture
    def trader(self, mock_api, strategy_manager, test_config):
        """Create a trader instance with mock components"""
        risk_manager = RiskManager(test_config['risk_management'])
        
        trader = Trader(
            api_client=mock_api,
            strategy_manager=strategy_manager,
            risk_manager=risk_manager,
            initial_capital=test_config['initial_capital'],
            symbols=test_config['symbols']
        )
        
        return trader
    
    @pytest.fixture
    def trader_main(self, test_config):
        """Create a TraderMain instance with the test configuration"""
        with patch('src.trader_main.open', mock_open(read_data=json.dumps(test_config))):
            with patch('src.trader_main.AlphaVantageAPI'):
                with patch('src.trader_main.TechnicalAnalyzer'):
                    with patch('src.trader_main.RiskManager'):
                        with patch('src.trader_main.MachineLearning'):
                            with patch('src.trader_main.StrategyManager'):
                                trader_main = TraderMain('dummy_path.json')
                                # Mock internal trader
                                trader_main._trader = MagicMock()
                                return trader_main
    
    def test_strategy_combination_weighted_average(self, strategy_manager, mock_market_data):
        """Test weighted average strategy combination method"""
        symbol = 'AAPL'
        mock_signals = {
            "momentum": {"signal": 1.0, "confidence": 0.8},
            "mean_reversion": {"signal": -1.0, "confidence": 0.7},
            "trend_following": {"signal": 0.5, "confidence": 0.9}
        }
        
        # Patch the strategies to return the mock signals
        with patch.object(strategy_manager._strategies["momentum"], "generate_signal", return_value=mock_signals["momentum"]):
            with patch.object(strategy_manager._strategies["mean_reversion"], "generate_signal", return_value=mock_signals["mean_reversion"]):
                with patch.object(strategy_manager._strategies["trend_following"], "generate_signal", return_value=mock_signals["trend_following"]):
                    # Get combined signal using weighted average
                    combined_signal = strategy_manager.get_combined_signal(
                        symbol, 
                        mock_market_data[symbol], 
                        "weighted_average"
                    )
                    
                    # Calculate expected result manually
                    weights = strategy_manager._strategy_weights
                    expected_signal = (
                        weights["momentum"] * mock_signals["momentum"]["signal"] * mock_signals["momentum"]["confidence"] +
                        weights["mean_reversion"] * mock_signals["mean_reversion"]["signal"] * mock_signals["mean_reversion"]["confidence"] +
                        weights["trend_following"] * mock_signals["trend_following"]["signal"] * mock_signals["trend_following"]["confidence"]
                    ) / sum(weights.values())
                    
                    assert combined_signal["signal"] == pytest.approx(expected_signal)
                    assert "confidence" in combined_signal
                    assert combined_signal["confidence"] > 0 and combined_signal["confidence"] <= 1
    
    def test_strategy_combination_majority_vote(self, strategy_manager, mock_market_data):
        """Test majority vote strategy combination method"""
        symbol = 'MSFT'
        mock_signals = {
            "momentum": {"signal": 1.0, "confidence": 0.8},
            "mean_reversion": {"signal": 1.0, "confidence": 0.7},
            "trend_following": {"signal": -1.0, "confidence": 0.9}
        }
        
        # Patch the strategies to return the mock signals
        with patch.object(strategy_manager._strategies["momentum"], "generate_signal", return_value=mock_signals["momentum"]):
            with patch.object(strategy_manager._strategies["mean_reversion"], "generate_signal", return_value=mock_signals["mean_reversion"]):
                with patch.object(strategy_manager._strategies["trend_following"], "generate_signal", return_value=mock_signals["trend_following"]):
                    # Get combined signal using majority vote
                    combined_signal = strategy_manager.get_combined_signal(
                        symbol, 
                        mock_market_data[symbol], 
                        "majority_vote"
                    )
                    
                    # Majority is positive (2 positive, 1 negative)
                    assert combined_signal["signal"] > 0
                    assert "confidence" in combined_signal
                    assert combined_signal["confidence"] > 0 and combined_signal["confidence"] <= 1
    
    def test_strategy_combination_highest_confidence(self, strategy_manager, mock_market_data):
        """Test highest confidence strategy combination method"""
        symbol = 'GOOGL'
        mock_signals = {
            "momentum": {"signal": 1.0, "confidence": 0.8},
            "mean_reversion": {"signal": -1.0, "confidence": 0.7},
            "trend_following": {"signal": -0.5, "confidence": 0.9}  # Highest confidence
        }
        
        # Patch the strategies to return the mock signals
        with patch.object(strategy_manager._strategies["momentum"], "generate_signal", return_value=mock_signals["momentum"]):
            with patch.object(strategy_manager._strategies["mean_reversion"], "generate_signal", return_value=mock_signals["mean_reversion"]):
                with patch.object(strategy_manager._strategies["trend_following"], "generate_signal", return_value=mock_signals["trend_following"]):
                    # Get combined signal using highest confidence
                    combined_signal = strategy_manager.get_combined_signal(
                        symbol, 
                        mock_market_data[symbol], 
                        "highest_confidence"
                    )
                    
                    # Should choose trend_following signal as it has highest confidence
                    assert combined_signal["signal"] == mock_signals["trend_following"]["signal"]
                    assert combined_signal["confidence"] == mock_signals["trend_following"]["confidence"]
    
    def test_portfolio_management(self, trader, mock_market_data):
        """Test portfolio management functionality"""
        # Patch internal methods
        with patch.object(trader, '_get_market_data', return_value=mock_market_data['AAPL']):
            with patch.object(trader._strategy_manager, 'get_combined_signal', return_value={"signal": 1.0, "confidence": 0.9}):
                with patch.object(trader, '_execute_buy', return_value=True):
                    # Initialize portfolio
                    initial_capital = trader._cash
                    
                    # Execute a buy
                    trader.process_symbol('AAPL')
                    
                    # Check portfolio state after buy
                    assert trader._positions.get('AAPL') is not None
                    assert trader._cash < initial_capital
                    
                    # Execute a sell (change signal to sell)
                    with patch.object(trader._strategy_manager, 'get_combined_signal', return_value={"signal": -1.0, "confidence": 0.9}):
                        with patch.object(trader, '_execute_sell', return_value=True):
                            trader.process_symbol('AAPL')
                            
                            # Check portfolio state after sell
                            assert 'AAPL' not in trader._positions or trader._positions['AAPL']['shares'] == 0
                            assert trader._cash > 0
                            
                            # Check trade history
                            assert len(trader._trade_history) > 0
                            assert any(trade['symbol'] == 'AAPL' for trade in trader._trade_history)
    
    def test_risk_management(self, trader, mock_market_data):
        """Test risk management functionality"""
        # Patch to simulate a risky scenario (high volatility)
        with patch.object(trader._risk_manager, 'calculate_position_size', return_value=5):
            with patch.object(trader._strategy_manager, 'get_combined_signal', return_value={"signal": 1.0, "confidence": 0.9}):
                with patch.object(trader, '_get_market_data', return_value=mock_market_data['AMZN']):
                    with patch.object(trader, '_execute_buy', return_value=True):
                        # Initial state
                        initial_capital = trader._cash
                        
                        # Process a symbol with risk management in effect
                        trader.process_symbol('AMZN')
                        
                        # Check if position size was limited by risk management
                        assert trader._positions.get('AMZN', {}).get('shares', 0) == 5
                        
                        # Calculate expected cost
                        last_price = mock_market_data['AMZN']['close'].iloc[-1]
                        expected_cost = 5 * last_price
                        
                        # Check if cash was deducted correctly
                        assert pytest.approx(initial_capital - trader._cash) == expected_cost
    
    def test_complete_trading_loop(self, trader_main):
        """Test the complete trading loop execution"""
        # Setup mocks for the trading loop
        trader_main._trader.get_portfolio_summary.return_value = {
            'total_value': 11000.0,
            'cash': 5000.0,
            'positions': {'AAPL': {'shares': 10, 'avg_price': 150.0, 'current_price': 160.0}},
            'profit_loss': 100.0,
            'profit_loss_pct': 0.01
        }
        
        # Setup trade execution mocks
        trader_main._trader.process_symbol.return_value = True
        trader_main._trader.update_portfolio_value.return_value = 11100.0
        
        # Mock the market status
        with patch('src.trader_main.is_market_open', return_value=True):
            # Execute trading loop for a few iterations
            for _ in range(3):
                trader_main._execute_trading_loop()
                
            # Verify interactions
            assert trader_main._trader.process_symbol.call_count > 0
            assert trader_main._trader.update_portfolio_value.call_count > 0
            assert trader_main._trader.get_portfolio_summary.call_count > 0
            
            # Check that the loop properly tracked portfolio changes
            assert trader_main._latest_portfolio_value != trader_main._initial_portfolio_value
            
            # Verify proper shutdown doesn't happen during normal operation
            assert not trader_main._shutdown_flag
            
    def test_error_handling_api_failure(self, trader_main):
        """Test error handling when API fails"""
        # Mock API failure in process_symbol
        trader_main._trader.process_symbol.side_effect = Exception("API Connection Error")
        trader_main._trader.get_portfolio_summary.return_value = {
            'total_value': 10000.0,
            'cash': 10000.0,
            'positions': {},
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0
        }
        
        # Execute trading loop
        with patch('src.trader_main.is_market_open', return_value=True):
            with patch('src.trader_main.logging') as mock_logging:
                trader_main._execute_trading_loop()
                
                # Verify error was logged but system continued
                mock_logging.error.assert_called()
                # System should remain operational
                assert not trader_main._shutdown_flag
                
    def test_system_shutdown(self, trader_main):
        """Test system shutdown behavior"""
        # Setup normal behavior
        trader_main._trader.get_portfolio_summary.return_value = {
            'total_value': 10500.0,
            'cash': 5000.0,
            'positions': {'AAPL': {'shares': 10, 'avg_price': 150.0, 'current_price': 155.0}},
            'profit_loss': 50.0,
            'profit_loss_pct': 0.005
        }
        
        # Set up clean termination
        with patch('src.trader_main.is_market_open', return_value=True):
            with patch('src.trader_main.logging') as mock_logging:
                # Trigger shutdown during trading loop
                def set_shutdown(*args, **kwargs):
                    trader_main._shutdown_flag = True
                    return True
                
                trader_main._trader.process_symbol.side_effect = set_shutdown
                
                # Run the trading loop
                trader_main._execute_trading_loop()
                
                # Verify shutdown procedure was initiated
                assert trader_main._shutdown_flag
                
                # Now actually test the shutdown process
                trader_main.shutdown()
                
                # Verify all positions are closed
                trader_main._trader.close_all_positions.assert_called_once()
                
                # Verify final portfolio summary is logged
                mock_logging.info.assert_any_call(
                    "Final portfolio summary: %s", 
                    trader_main._trader.get_portfolio_summary.return_value
                )
                
    def test_recovery_scenario(self, trader_main):
        """Test system recovery after errors"""
        # Setup for simulating a sequence of errors and recovery
        error_counter = [0]
        
        def side_effect_with_recovery(*args, **kwargs):
            if error_counter[0] < 2:
                error_counter[0] += 1
                raise Exception(f"Simulated error #{error_counter[0]}")
            return True
        
        # Set up trader to fail twice and then recover
        trader_main._trader.process_symbol.side_effect = side_effect_with_recovery
        trader_main._trader.get_portfolio_summary.return_value = {
            'total_value': 10000.0,
            'cash': 10000.0,
            'positions': {},
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0
        }
        
        # Run multiple trading loops to test recovery
        with patch('src.trader_main.is_market_open', return_value=True):
            with patch('src.trader_main.logging') as mock_logging:
                # First loop - should have an error
                trader_main._execute_trading_loop()
                mock_logging.error.assert_called()
                
                # Second loop - should have another error
                mock_logging.reset_mock()
                trader_main._execute_trading_loop()
                mock_logging.error.assert_called()
                
                # Third loop - should recover
                mock_logging.reset_mock()
                trader_main._execute_trading_loop()
                # This time should not log an error
                assert not mock_logging.error.called
                
                # Verify system remained operational throughout
                assert not trader_main._shutdown_flag
                
    def test_market_closed_behavior(self, trader_main):
        """Test behavior when market is closed"""
        # Setup mocks
        trader_main._trader.get_portfolio_summary.return_value = {
            'total_value': 10800.0,
            'cash': 4800.0,
            'positions': {'MSFT': {'shares': 20, 'avg_price': 300.0, 'current_price': 300.0}},
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0
        }
        
        # Simulate market closed
        with patch('src.trader_main.is_market_open', return_value=False):
            with patch('src.trader_main.logging') as mock_logging:
                # Execute trading loop
                trader_main._execute_trading_loop()
                
                # Verify market closed message was logged
                mock_logging.info.assert_any_call("Market is closed. Skipping trading cycle.")
                
                # Verify no trading occurred
                trader_main._trader.process_symbol.assert_not_called()
                
                # System should remain operational for next market open
                assert not trader_main._shutdown_flag
