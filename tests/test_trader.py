"""
Unit tests for the Trader class.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.trader import Trader


class TestTrader(unittest.TestCase):
    """Test cases for the Trader class functionality."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the dependencies
        self.mock_api = Mock()
        self.mock_technical_analyzer = Mock()
        self.mock_risk_manager = Mock()
        self.mock_ml = Mock()

        # Create mock market data
        self.mock_market_data = pd.DataFrame({
            'open': [100.0, 101.0, 102.0, 103.0, 104.0],
            'high': [105.0, 106.0, 107.0, 108.0, 109.0],
            'low': [95.0, 96.0, 97.0, 98.0, 99.0],
            'close': [103.0, 104.0, 105.0, 106.0, 107.0],
            'volume': [1000, 1100, 1200, 1300, 1400],
        }, index=pd.date_range(start='2023-01-01', periods=5))

        # Patch the dependencies
        with patch('src.api.alpha_vantage.AlphaVantageAPI') as mock_api_class, \
             patch('src.analytics.technical.TechnicalAnalyzer') as mock_ta_class, \
             patch('src.risk.manager.RiskManager') as mock_rm_class, \
             patch('src.ml.predictor.MachineLearning') as mock_ml_class:
            
            # Configure the mocks
            mock_api_class.return_value = self.mock_api
            mock_ta_class.return_value = self.mock_technical_analyzer
            mock_rm_class.return_value = self.mock_risk_manager
            mock_ml_class.return_value = self.mock_ml

            # Set up responses for the mock API
            self.mock_api.get_daily.return_value = self.mock_market_data
            self.mock_api.get_intraday.return_value = self.mock_market_data
            
            # Set up responses for the technical analyzer
            self.mock_technical_analyzer.calculate_rsi.return_value = [30, 40, 50, 60, 70]
            self.mock_technical_analyzer.calculate_macd.return_value = ([0.5, 1.0, 1.5, 2.0, 2.5], [0.3, 0.8, 1.3, 1.8, 2.3])
            self.mock_technical_analyzer.calculate_ema.return_value = [100, 101, 102, 103, 104]
            self.mock_technical_analyzer.determine_trend.return_value = 'uptrend'
            
            # Set up responses for the risk manager
            self.mock_risk_manager.calculate_position_size.return_value = 10
            self.mock_risk_manager.max_positions = 5
            
            # Set up responses for the machine learning module
            self.mock_ml.predict_price.return_value = 110.0
            self.mock_ml.detect_patterns.return_value = ['bullish_engulfing']
            
            # Create the trader instance
            self.trader = Trader('dummy_api_key')
            
            # Replace the dependencies with mocks
            self.trader.api = self.mock_api
            self.trader.technical_analyzer = self.mock_technical_analyzer
            self.trader.risk_manager = self.mock_risk_manager
            self.trader.ml = self.mock_ml

    def test_initialization(self):
        """Test that the Trader class initializes correctly."""
        trader = Trader('test_api_key', risk_percentage=3.0, max_positions=10, portfolio_size=20000.0)
        
        self.assertEqual(trader.start_balance, 20000.0)
        self.assertEqual(trader.current_balance, 20000.0)
        self.assertEqual(trader.active_positions, {})
        self.assertEqual(trader.trade_history, [])
        self.assertEqual(trader.watchlist, [])

    def test_add_to_watchlist(self):
        """Test adding symbols to the watchlist."""
        self.trader.add_to_watchlist(['AAPL', 'MSFT', 'GOOGL'])
        self.assertEqual(self.trader.watchlist, ['AAPL', 'MSFT', 'GOOGL'])
        
        # Test adding a duplicate
        self.trader.add_to_watchlist(['AAPL', 'AMZN'])
        self.assertEqual(self.trader.watchlist, ['AAPL', 'MSFT', 'GOOGL', 'AMZN'])

    def test_remove_from_watchlist(self):
        """Test removing symbols from the watchlist."""
        self.trader.add_to_watchlist(['AAPL', 'MSFT', 'GOOGL'])
        self.trader.remove_from_watchlist('MSFT')
        self.assertEqual(self.trader.watchlist, ['AAPL', 'GOOGL'])
        
        # Test removing a symbol not in the watchlist
        self.trader.remove_from_watchlist('AMZN')
        self.assertEqual(self.trader.watchlist, ['AAPL', 'GOOGL'])

    def test_analyze_symbol(self):
        """Test the comprehensive analysis of a symbol."""
        analysis = self.trader.analyze_symbol('AAPL')
        
        self.assertEqual(analysis['symbol'], 'AAPL')
        self.assertEqual(analysis['current_price'], 107.0)
        self.assertEqual(analysis['technical']['rsi'], 70)
        self.assertEqual(analysis['technical']['macd'], 2.5)
        self.assertEqual(analysis['technical']['macd_signal'], 2.3)
        self.assertEqual(analysis['technical']['trend'], 'uptrend')
        self.assertEqual(analysis['ml_prediction']['price_prediction'], 110.0)
        self.assertEqual(analysis['ml_prediction']['pattern_detected'], ['bullish_engulfing'])
        
        # Verify API and analyzer methods were called
        self.mock_api.get_daily.assert_called_once_with('AAPL')
        self.mock_technical_analyzer.calculate_rsi.assert_called_once()
        self.mock_technical_analyzer.calculate_macd.assert_called_once()
        self.mock_technical_analyzer.determine_trend.assert_called_once()
        self.mock_ml.predict_price.assert_called_once_with('AAPL', days=5)
        self.mock_ml.detect_patterns.assert_called_once()

    def test_generate_trade_signal_buy(self):
        """Test generating a buy trade signal."""
        # Configure conditions for a buy signal
        analysis = {
            'symbol': 'AAPL',
            'current_price': 100.0,
            'technical': {
                'rsi': 25,  # Oversold
                'macd': 1.0,
                'macd_signal': 0.5,  # MACD above signal
                'ema50': 110,
                'ema200': 100,  # Golden cross
                'trend': 'uptrend'
            },
            'ml_prediction': {
                'price_prediction': 105.0,  # >2% increase
                'pattern_detected': ['bullish_engulfing']
            }
        }
        
        signal, stop_loss, take_profit = self.trader.generate_trade_signal(analysis)
        
        self.assertEqual(signal, 'buy')
        self.assertEqual(stop_loss, 98.0)  # 2% below entry
        self.assertEqual(take_profit, 106.0)  # 6% above entry

    def test_generate_trade_signal_sell(self):
        """Test generating a sell trade signal."""
        # Configure conditions for a sell signal
        analysis = {
            'symbol': 'AAPL',
            'current_price': 100.0,
            'technical': {
                'rsi': 75,  # Overbought
                'macd': 0.5,
                'macd_signal': 1.0,  # MACD below signal
                'ema50': 90,
                'ema200': 100,  # Death cross
                'trend': 'downtrend'
            },
            'ml_prediction': {
                'price_prediction': 95.0,  # >2% decrease
                'pattern_detected': ['bearish_engulfing']
            }
        }
        
        signal, stop_loss, take_profit = self.trader.generate_trade_signal(analysis)
        
        self.assertEqual(signal, 'sell')
        self.assertEqual(stop_loss, 102.0)  # 2% above entry
        self.assertEqual(take_profit, 94.0)  # 6% below entry

    def test_generate_trade_signal_hold(self):
        """Test generating a hold trade signal."""
        # Configure conditions for a hold signal (mixed or neutral indicators)
        analysis = {
            'symbol': 'AAPL',
            'current_price': 100.0,
            'technical': {
                'rsi': 50,  # Neutral
                'macd': 1.0,
                'macd_signal': 1.0,  # MACD equal to signal
                'ema50': 100,
                'ema200': 100,  # No cross
                'trend': 'sideways'
            },
            'ml_prediction': {
                'price_prediction': 101.0,  # <2% change
                'pattern_detected': []
            }
        }
        
        signal, stop_loss, take_profit = self.trader.generate_trade_signal(analysis)
        
        self.assertEqual(signal, 'hold')
        self.assertEqual(stop_loss, 0.0)
        self.assertEqual(take_profit, 0.0)

    def test_execute_trade_buy(self):
        """Test executing a buy trade."""
        result = self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        
        self.assertEqual(result['status'], 'executed')
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['signal'], 'buy')
        self.assertEqual(result['entry_price'], 100.0)
        self.assertEqual(result['position_size'], 10)
        self.assertEqual(result['stop_loss'], 98.0)
        self.assertEqual(result['take_profit'], 106.0)
        
        # Verify risk manager was called to calculate position size
        self.mock_risk_manager.calculate_position_size.assert_called_once_with('AAPL', 100.0, 98.0)
        
        # Verify portfolio state updated
        self.assertEqual(self.trader.current_balance, 9000.0)  # 10000 - (10 * 100.0)
        self.assertEqual(len(self.trader.active_positions), 1)
        self.assertEqual(len(self.trader.trade_history), 1)
        self.assertIn('AAPL', self.trader.active_positions)

    def test_execute_trade_sell(self):
        """Test executing a sell trade."""
        result = self.trader.execute_trade('MSFT', 'sell', 200.0, 204.0, 188.0)
        
        self.assertEqual(result['status'], 'executed')
        self.assertEqual(result['symbol'], 'MSFT')
        self.assertEqual(result['signal'], 'sell')
        self.assertEqual(result['entry_price'], 200.0)
        self.assertEqual(result['position_size'], 10)
        self.assertEqual(result['stop_loss'], 204.0)
        self.assertEqual(result['take_profit'], 188.0)
        
        # Verify risk manager was called
        self.mock_risk_manager.calculate_position_size.assert_called_once_with('MSFT', 200.0, 204.0)
        
        # For short positions, we don't decrease balance immediately
        self.assertEqual(self.trader.current_balance, 10000.0)
        self.assertEqual(len(self.trader.active_positions), 1)
        self.assertEqual(len(self.trader.trade_history), 1)
        self.assertIn('MSFT', self.trader.active_positions)

    def test_execute_trade_hold(self):
        """Test that a hold signal doesn't execute a trade."""
        result = self.trader.execute_trade('AAPL', 'hold', 100.0, 0.0, 0.0)
        
        self.assertEqual(result['status'], 'skipped')
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['signal'], 'hold')
        
        # Verify no trades or positions were created
        self.assertEqual(len(self.trader.active_positions), 0)
        self.assertEqual(len(self.trader.trade_history), 0)
        self.assertEqual(self.trader.current_balance, 10000.0)

    def test_execute_trade_risk_management_rejection(self):
        """Test that risk management can reject a trade."""
        # Make risk manager return 0 for position size
        self.mock_risk_manager.calculate_position_size.return_value = 0
        
        result = self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['signal'], 'buy')
        
        # Verify no trades or positions were created
        self.assertEqual(len(self.trader.active_positions), 0)
        self.assertEqual(len(self.trader.trade_history), 0)

    def test_check_position_status_no_position(self):
        """Test checking status for a non-existent position."""
        result = self.trader.check_position_status('AAPL', 100.0)
        
        self.assertEqual(result['status'], 'not_found')
        self.assertEqual(result['symbol'], 'AAPL')

    def test_check_position_status_active(self):
        """Test checking status for an active position."""
        # Create a position first
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        
        # Check status at a price between stop loss and take profit
        result = self.trader.check_position_status('AAPL', 102.0)
        
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['status'], 'open')
        self.assertEqual(result['unrealized_pnl'], 20.0)  # (102 - 100) * 10 shares
        self.assertEqual(result['pnl_percentage'], 2.0)  # 2% increase

    def test_check_position_status_take_profit(self):
        """Test checking status when take profit is triggered."""
        # Create a position first
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        
        # Check status at take profit level
        result = self.trader.check_position_status('AAPL', 106.0)
        
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['status'], 'closed')
        self.assertEqual(result['close_reason'], 'take_profit')
        self.assertEqual(result['realized_pnl'], 60.0)  # (106 - 100) * 10 shares
        self.assertEqual(result['pnl_percentage'], 6.0)  # 6% increase
        
        # Verify the position was removed from active positions
        self.assertNotIn('AAPL', self.trader.active_positions)
        # Verify the trade history was updated
        self.assertEqual(len(self.trader.trade_history), 2)  # Open + close
        
    def test_check_position_status_stop_loss(self):
        """Test checking status when stop loss is triggered."""
        # Create a position first
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        
        # Check status at stop loss level
        result = self.trader.check_position_status('AAPL', 98.0)
        
        self.assertEqual(result['symbol'], 'AAPL')
        self.assertEqual(result['status'], 'closed')
        self.assertEqual(result['close_reason'], 'stop_loss')
        self.assertEqual(result['realized_pnl'], -20.0)  # (98 - 100) * 10 shares
        self.assertEqual(result['pnl_percentage'], -2.0)  # -2% decrease
        
        # Verify the position was removed from active positions
        self.assertNotIn('AAPL', self.trader.active_positions)
        # Verify the trade history was updated
        self.assertEqual(len(self.trader.trade_history), 2)  # Open + close

    def test_calculate_portfolio_summary(self):
        """Test calculating portfolio summary."""
        # Create some positions first
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        self.trader.execute_trade('MSFT', 'buy', 200.0, 196.0, 212.0)
        
        # Mock current prices
        self.mock_api.get_current_price = Mock()
        self.mock_api.get_current_price.side_effect = lambda symbol: {
            'AAPL': 103.0,
            'MSFT': 204.0
        }[symbol]
        
        # Calculate portfolio summary
        summary = self.trader.calculate_portfolio_summary()
        
        self.assertEqual(summary['total_positions'], 2)
        self.assertEqual(summary['total_investment'], 3000.0)  # (10*100) + (10*200)
        self.assertEqual(summary['current_value'], 3070.0)  # (10*103) + (10*204)
        self.assertEqual(summary['unrealized_pnl'], 70.0)  # (3*10) + (4*10)
        self.assertEqual(summary['total_pnl_percentage'], 2.33)  # 70/3000 ≈ 2.33%
        self.assertEqual(summary['winning_positions'], 2)
        self.assertEqual(summary['losing_positions'], 0)

    def test_calculate_risk_metrics(self):
        """Test calculating risk metrics for the portfolio."""
        # Create some positions first
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        self.trader.execute_trade('MSFT', 'sell', 200.0, 204.0, 188.0)
        
        # Mock market data for volatility calculations
        extended_mock_data = pd.DataFrame({
            'close': [95.0, 98.0, 100.0, 102.0, 103.0, 101.0, 
                      99.0, 100.0, 102.0, 104.0, 103.0, 105.0,
                      106.0, 104.0, 102.0, 103.0, 105.0, 106.0,
                      107.0, 108.0]
        }, index=pd.date_range(start='2023-01-01', periods=20))
        
        self.mock_api.get_historical_data = Mock(return_value=extended_mock_data)
        
        # Calculate risk metrics
        risk_metrics = self.trader.calculate_risk_metrics()
        
        # Verify risk metrics
        self.assertIn('portfolio_var', risk_metrics)
        self.assertIn('portfolio_volatility', risk_metrics)
        self.assertIn('max_drawdown', risk_metrics)
        self.assertIn('sharpe_ratio', risk_metrics)
        self.assertIn('sortino_ratio', risk_metrics)
        self.assertIn('risk_reward_ratio', risk_metrics)
        self.assertTrue(0 <= risk_metrics['portfolio_var'] <= 100)
        self.assertTrue(0 <= risk_metrics['portfolio_volatility'] <= 10)
        self.assertTrue(0 <= risk_metrics['max_drawdown'] <= 100)
        
        # Verify API was called
        self.mock_api.get_historical_data.assert_called()

    def test_manage_positions(self):
        """Test managing multiple positions with different actions."""
        # Setup positions
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        self.trader.execute_trade('MSFT', 'buy', 200.0, 196.0, 212.0)
        self.trader.execute_trade('GOOGL', 'sell', 1000.0, 1020.0, 980.0)
        
        # Set up different price movements for different symbols
        current_prices = {
            'AAPL': 105.0,  # Near take profit
            'MSFT': 196.0,  # At stop loss
            'GOOGL': 990.0,  # Between entry and take profit (for short)
            'AMZN': 150.0    # Not in portfolio
        }
        
        self.mock_api.get_current_price = Mock(side_effect=lambda symbol: current_prices[symbol])
        
        # Test managing all positions
        managed_positions = self.trader.manage_positions(['AAPL', 'MSFT', 'GOOGL', 'AMZN'])
        
        # Check results of management
        self.assertEqual(managed_positions['active_positions'], 2)  # AAPL and GOOGL remain open
        self.assertEqual(managed_positions['closed_positions'], 1)  # MSFT hit stop loss
        self.assertEqual(managed_positions['not_found_positions'], 1)  # AMZN not in portfolio
        
        # MSFT should be removed from active positions
        self.assertNotIn('MSFT', self.trader.active_positions)
        # AAPL and GOOGL should still be in active positions
        self.assertIn('AAPL', self.trader.active_positions)
        self.assertIn('GOOGL', self.trader.active_positions)
        
        # Trade history should now have 4 entries (3 opens + 1 close)
        self.assertEqual(len(self.trader.trade_history), 4)

    def test_generate_trade_history_report(self):
        """Test generating a trade history report."""
        # Create and close some trades
        self.trader.execute_trade('AAPL', 'buy', 100.0, 98.0, 106.0)
        self.trader.check_position_status('AAPL', 106.0)  # Close at take profit
        
        self.trader.execute_trade('MSFT', 'buy', 200.0, 196.0, 212.0)
        self.trader.check_position_status('MSFT', 195.0)  # Close below stop loss
        
        self.trader.execute_trade('GOOGL', 'sell', 1000.0, 1020.0, 980.0)
        self.trader.check_position_status('GOOGL', 982.0)  # Close near take profit
        
        # Generate report
        report = self.trader.generate_trade_history_report()
        
        # Check report contents
        self.assertEqual(report['total_trades'], 6)  # 3 opens + 3 closes
        self.assertEqual(report['winning_trades'], 2)  # AAPL and GOOGL
        self.assertEqual(report['losing_trades'], 1)  # MSFT
        self.assertGreaterEqual(report['win_rate'], 66.6)  # 2/3 = 66.6%
        self.assertIn('average_win', report)
        self.assertIn('average_loss', report)
        self.assertIn('profit_factor', report)
        self.assertIn('max_consecutive_wins', report)
        self.assertIn('max_consecutive_losses', report)
        self.assertIn('largest_win', report)
        self.assertIn('largest_loss', report)
        self.assertIn('average_holding_time', report)
        
        # Verify metrics
        self.assertEqual(report['largest_win']['symbol'], 'AAPL')
        self.assertEqual(report['largest_loss']['symbol'], 'MSFT')
        
        # Verify data frame
        self.assertIsInstance(report['trade_history_df'], pd.DataFrame)
        self.assertTrue(len(report['trade_history_df']) > 0)
        
    def test_calculate_drawdown(self):
        """Test calculating drawdown from equity curve."""
        # Create mock equity curve
        equity_curve = pd.Series([
            10000, 10200, 10400, 10300, 10100, 9900, 9800, 9900, 10000, 10300, 10500
        ], index=pd.date_range(start='2023-01-01', periods=11))
        
        # Calculate drawdown
        drawdown, max_drawdown = self.trader.calculate_drawdown(equity_curve)
        
        # Verify drawdown calculation
        self.assertEqual(max_drawdown, 6.67)  # (10500 - 9800) / 10500 = 6.67%
        self.assertEqual(len(drawdown), len(equity_curve))
        self.assertEqual(drawdown.min(), -6.67)
        
        # The drawdown should be zero at equity high points
        self.assertEqual(drawdown.iloc[0], 0)
        self.assertEqual(drawdown.iloc[-1], 0)

