"""
Trader module that integrates all trading components including API client,
technical analysis, risk management, and machine learning for automated trading.
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from src.api.alpha_vantage import AlphaVantageAPI
from src.analytics.technical import TechnicalAnalyzer
from src.risk.manager import RiskManager
from src.ml.predictor import MachineLearning

# Configure logging
logging.basicConfig(
    filename='aauto.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Trader:
    """
    Main trader class that integrates all components of the automated trading system.
    
    This class coordinates the interaction between market data retrieval, technical analysis,
    risk management, and machine learning to make trading decisions and execute trades.
    """
    
    def __init__(
        self, 
        api_key: str,
        risk_percentage: float = 2.0,
        max_positions: int = 5,
        portfolio_size: float = 10000.0,
        use_ml: bool = True,
        cache_data: bool = True
    ):
        """
        Initialize the Trader with all necessary components.
        
        Args:
            api_key: Alpha Vantage API key
            risk_percentage: Percentage of portfolio to risk per trade (default: 2%)
            max_positions: Maximum number of concurrent positions (default: 5)
            portfolio_size: Total portfolio value in USD (default: $10,000)
            use_ml: Whether to use machine learning predictions (default: True)
            cache_data: Whether to cache API responses (default: True)
        """
        logger.info("Initializing Trader system")
        
        # Initialize components
        self.api = AlphaVantageAPI(api_key, use_cache=cache_data)
        self.technical_analyzer = TechnicalAnalyzer(self.api)
        self.risk_manager = RiskManager(
            portfolio_size=portfolio_size,
            risk_percentage=risk_percentage,
            max_positions=max_positions
        )
        self.ml = MachineLearning(self.api) if use_ml else None
        
        # Trading state
        self.active_positions: Dict[str, Dict] = {}
        self.orders_pending: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.watchlist: List[str] = []
        
        # Performance tracking
        self.start_balance = portfolio_size
        self.current_balance = portfolio_size
        self.start_date = datetime.now()
        
        logger.info(f"Trader initialized with portfolio size: ${portfolio_size}")
    
    def add_to_watchlist(self, symbols: List[str]) -> None:
        """
        Add symbols to the watchlist for monitoring.
        
        Args:
            symbols: List of stock symbols to add to watchlist
        """
        for symbol in symbols:
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
                logger.info(f"Added {symbol} to watchlist")
    
    def remove_from_watchlist(self, symbol: str) -> None:
        """
        Remove a symbol from the watchlist.
        
        Args:
            symbol: Stock symbol to remove
        """
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            logger.info(f"Removed {symbol} from watchlist")
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """
        Perform comprehensive analysis of a symbol.
        
        Args:
            symbol: Stock symbol to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info(f"Analyzing {symbol}")
        
        # Get market data
        market_data = self.api.get_daily(symbol)
        
        # Perform technical analysis
        rsi = self.technical_analyzer.calculate_rsi(market_data)
        macd, signal = self.technical_analyzer.calculate_macd(market_data)
        ema50 = self.technical_analyzer.calculate_ema(market_data, period=50)
        ema200 = self.technical_analyzer.calculate_ema(market_data, period=200)
        trend = self.technical_analyzer.determine_trend(market_data)
        
        # Get current price from the most recent data
        current_price = market_data.iloc[-1]['close'] if not market_data.empty else None
        
        # ML prediction if enabled
        price_prediction = None
        pattern_detected = None
        if self.ml:
            price_prediction = self.ml.predict_price(symbol, days=5)
            pattern_detected = self.ml.detect_patterns(market_data)
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'technical': {
                'rsi': rsi[-1] if isinstance(rsi, list) and rsi else None,
                'macd': macd[-1] if isinstance(macd, list) and macd else None,
                'macd_signal': signal[-1] if isinstance(signal, list) and signal else None,
                'ema50': ema50[-1] if isinstance(ema50, list) and ema50 else None,
                'ema200': ema200[-1] if isinstance(ema200, list) and ema200 else None,
                'trend': trend
            },
            'ml_prediction': {
                'price_prediction': price_prediction,
                'pattern_detected': pattern_detected
            } if self.ml else None
        }
        
        logger.info(f"Analysis completed for {symbol}")
        return analysis
    
    def generate_trade_signal(self, analysis: Dict) -> Tuple[str, float, float]:
        """
        Generate a trade signal based on technical and ML analysis.
        
        Args:
            analysis: Dictionary containing analysis results
            
        Returns:
            Tuple containing (signal, stop_loss, take_profit)
            where signal is one of: 'buy', 'sell', 'hold'
        """
        signal = 'hold'
        stop_loss = 0.0
        take_profit = 0.0
        
        # Skip if no price data
        if not analysis['current_price']:
            return signal, stop_loss, take_profit
        
        current_price = analysis['current_price']
        tech = analysis['technical']
        
        # Basic technical analysis rules
        rsi_overbought = tech['rsi'] > 70 if tech['rsi'] else False
        rsi_oversold = tech['rsi'] < 30 if tech['rsi'] else False
        
        macd_bullish = (tech['macd'] > tech['macd_signal']) if (tech['macd'] and tech['macd_signal']) else False
        macd_bearish = (tech['macd'] < tech['macd_signal']) if (tech['macd'] and tech['macd_signal']) else False
        
        # Trend analysis
        uptrend = tech['trend'] == 'uptrend'
        downtrend = tech['trend'] == 'downtrend'
        
        # Golden cross / death cross
        golden_cross = (tech['ema50'] > tech['ema200']) if (tech['ema50'] and tech['ema200']) else False
        death_cross = (tech['ema50'] < tech['ema200']) if (tech['ema50'] and tech['ema200']) else False
        
        # Integrate ML predictions if available
        ml_bullish = False
        ml_bearish = False
        if analysis['ml_prediction'] and analysis['ml_prediction']['price_prediction']:
            predicted_price = analysis['ml_prediction']['price_prediction']
            ml_bullish = predicted_price > current_price * 1.02  # 2% increase predicted
            ml_bearish = predicted_price < current_price * 0.98  # 2% decrease predicted
        
        # Generate buy signal
        if ((rsi_oversold and macd_bullish) or 
            (uptrend and golden_cross) or 
            (ml_bullish and (uptrend or macd_bullish))):
            signal = 'buy'
            # Set stop loss at 2% below entry
            stop_loss = current_price * 0.98
            # Set take profit at 6% above entry (3:1 reward-to-risk ratio)
            take_profit = current_price * 1.06
        
        # Generate sell signal
        elif ((rsi_overbought and macd_bearish) or 
              (downtrend and death_cross) or 
              (ml_bearish and (downtrend or macd_bearish))):
            signal = 'sell'
            # For short positions (if supported)
            stop_loss = current_price * 1.02
            take_profit = current_price * 0.94
        
        logger.info(f"Generated {signal} signal for {analysis['symbol']} at {current_price}")
        return signal, stop_loss, take_profit
    
    def execute_trade(self, symbol: str, signal: str, price: float, 
                     stop_loss: float, take_profit: float) -> Dict:
        """
        Execute a trade based on the generated signal.
        
        Args:
            symbol: Stock symbol to trade
            signal: Trade signal ('buy' or 'sell')
            price: Current price for the trade
            stop_loss: Stop loss price level
            take_profit: Take profit price level
            
        Returns:
            Dictionary with trade details
        """
        if signal not in ['buy', 'sell']:
            logger.info(f"No trade executed for {symbol} (signal: {signal})")
            return {'status': 'skipped', 'symbol': symbol, 'signal': signal}
        
        # Check if we can take this position based on risk management
        position_size = self.risk_manager.calculate_position_size(
            symbol, price, stop_loss
        )
        
        # Skip if position size is too small or we have max positions
        if position_size <= 0 or (
            len(self.active_positions) >= self.risk_manager.max_positions and
            symbol not in self.active_positions
        ):
            logger.info(f"Trade rejected for {symbol} due to risk management constraints")
            return {'status': 'rejected', 'symbol': symbol, 'signal': signal}
        
        # Create trade record
        trade = {
            'symbol': symbol,
            'signal': signal,
            'entry_price': price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_time': datetime.now(),
            'status': 'open',
            'exit_price': None,
            'exit_time': None,
            'profit_loss': None,
            'profit_loss_pct': None
        }
        
        # In a real implementation, this would connect to a broker API
        # For simulation purposes, we'll just log the trade
        logger.info(f"Executed {signal} trade for {symbol} - {position_size} shares at ${price}")
        
        # Update portfolio state
        if signal == 'buy':
            invested_amount = position_size * price
            self.current_balance -= invested_amount
        
        # Store the active position
        self.active_positions[symbol] = trade
        self.trade_history.append(trade)
        
        return {'status': 'executed', **trade}
    
    def check_position_status(self, symbol: str, current_price: float) -> Dict:
        """
        Check and update status of an open position based on current price.
        
        Args:
            symbol: Symbol to check
            current_price: Current market price
            
        Returns:
            Updated position information
        """
        if symbol not in self.active_positions:
            return {'status': 'not_found', 'symbol': symbol}
        
        position = self.active_positions[symbol]
        
        # Check if stop loss or take profit has been hit
        if position['signal'] == 'buy':
            if current_price <= position['stop_loss']:
                return self._close_position(symbol, current_price, 'stop_loss')
            elif current_price >= position['take_profit']:
                return self._close_position(symbol, current_price, 'take_profit')
        elif position['signal'] == 'sell':  # Short position
            if current_price >= position['stop_loss']:
                return self._close_position(symbol, current_price, 'stop_loss')
            elif current_price <= position['take_profit']:
                return self._close_position(symbol, current_price, 'take_profit')
        
        # Update unrealized P&L
        if position['signal'] == 'buy':
            unrealized_pl = (current_price - position['entry_price']) * position['position_size']
            unrealized_pl_pct = (current_price / position['entry_price'] - 1) * 100
        else:  # Short position
            unrealized_pl = (position['entry_price'] - current_price) * position['position_size']
            unrealized_pl_pct = (position['entry_price'] / current_price - 1) * 100
        
        position['unrealized_pl'] = unrealized_pl
        position['unrealized_pl_pct'] = unrealized_pl_pct
        
        return position
    
    def _close_position(self, symbol: str, exit_price: float, reason: str) -> Dict:
        """
        Close an open position and calculate profit/loss.
        
        Args:
            symbol: Symbol to close position for
            exit_price: Exit price
            reason: Reason for closing ('stop_loss', 'take_profit', 'manual')
            
        Returns:
            Closed position information
        """
        if symbol not in self.active_positions:
            return {'status': 'not_found', 'symbol': symbol}
        
        position = self.active_positions[symbol]
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['status'] = 'closed'
        position['close_reason'] = reason
        
        # Calculate P&L
        if position['signal'] == 'buy':
            profit_loss = (exit_price - position['entry_price']) * position['position_size']
            profit_loss_pct = (exit_price / position['entry_price'] - 1) * 100
        else:  # Short position
            profit_loss = (position['entry_price'] - exit_price) * position['position_size']
            profit_loss_pct = (position['entry_price'] / exit_price - 1) * 100
        
        position['profit_loss'] = profit_loss
        position['profit_loss_pct'] = profit_loss_pct
        
        # Update portfolio balance
        self.current_balance += (position['position_size'] * exit_price)
        
        # Log the trade result
        logger.info(
            f"Closed {position['signal']} position for {symbol} at ${exit_price} - "
            f"P&L: ${profit_loss:.2f} ({profit_loss_pct:.2f}%) - Reason: {reason}"
        )
        
        # Remove from active positions and update trade history
        closed_position = self.active_positions.pop(symbol)
        for i, trade in enumerate(self.trade_history):
            if trade['symbol'] == symbol and trade['entry_time'] == position['entry_time']:
                self.trade_history[i] = position
                break
        
        return position
    
    def get_portfolio_summary(self) -> Dict:
        """
        Get a comprehensive summary of the current portfolio state.
        
        Returns:
            Dictionary containing portfolio metrics
        """
        # Calculate total invested amount
        invested_amount = sum(
            pos['position_size'] * pos['entry_price'] 
            for pos in self.active_positions.values()
        )
        
        # Calculate unrealized P&L
        unrealized_pl = sum(
            pos.get('unrealized_pl', 0) 
            for pos in self.active_positions.values()
        )
        
        # Calculate realized P&L from closed trades
        realized_pl = sum(
            trade.get('profit_loss', 0) 
            for trade in self.trade_history 
            if trade['status'] == 'closed'
        )
        
        # Calculate total portfolio value
        portfolio_value = self.current_balance + invested_amount
        
        # Calculate performance metrics
        portfolio_return = (portfolio_value / self.start_balance - 1) * 100
        win_trades = [t for t in self.trade_history if t['status'] == 'closed' and t.get('profit_loss', 0) > 0]
        loss_trades = [t for t in self.trade_history if t['status'] == 'closed' and t.get('profit_loss', 0) <= 0]
        
        win_rate = (len(win_trades) / len(self.trade_history)) * 100 if self.trade_history else 0
        
        # Return comprehensive summary
        return {
            'timestamp': datetime.now(),
            'start_date': self.start_date,
            'days_active': (datetime.now() - self.start_date).days,
            'start_balance': self.start_balance,
            'current_balance': self.current_balance,
            'cash_available': self.current_balance,
            'invested_amount': invested_amount,
            'portfolio_value': portfolio_value,
            'realized_pl': realized_pl,
            'unrealized_pl': unrealized_pl,
            'total_pl': realized_pl + unrealized_pl,
            'portfolio_return_pct': portfolio_return,
            'open_positions': len(self.active_positions),
            'closed_positions': len([t for t in self.trade_history if t['status'] == 'closed']),
            'win_rate': win_rate,
            'average_win': sum(t.get('profit_loss', 0) for t in win_trades) / len(win_trades) if win_trades else 0,
            'average_loss': sum(t.get('profit_loss', 0) for t in loss_trades) / len(loss_trades) if loss_trades else 0,
            'largest_win': max([t.get('profit_loss', 0) for t in win_trades]) if win_trades else 0,
            'largest_loss': min([t.get('profit_loss', 0) for t in loss_trades]) if loss_trades else 0,
        }
    
    def calculate_risk_metrics(self) -> Dict:
        """
        Calculate risk metrics for the current portfolio.
        
        Returns:
            Dictionary containing risk metrics
        """
        # Extract profit/loss percentages from closed trades
        returns = [
            trade.get('profit_loss_pct', 0) / 100  # Convert percentage to decimal
            for trade in self.trade_history
            if trade['status'] == 'closed'
        ]
        
        # Default values if not enough data
        if len(returns) < 2:
            return {
                'sharpe_ratio': None,
                'sortino_ratio': None,
                'max_drawdown': None,
                'volatility': None,
                'profit_factor': None,
                'risk_reward_ratio': None,
                'expectancy': None,
            }
        
        # Calculate volatility (standard deviation of returns)
        volatility = pd.Series(returns).std() if returns else 0
        
        # Calculate downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        downside_deviation = pd.Series(negative_returns).std() if negative_returns else 0
        
        # Calculate maximum drawdown
        equity_curve = []
        current_equity = 1.0  # Starting with 1 unit
        max_equity = 1.0
        max_drawdown = 0.0
        
        for ret in returns:
            current_equity *= (1 + ret)
            max_equity = max(max_equity, current_equity)
            drawdown = (max_equity - current_equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
            equity_curve.append(current_equity)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        avg_return = pd.Series(returns).mean() if returns else 0
        sharpe_ratio = avg_return / volatility if volatility > 0 else 0
        
        # Calculate Sortino Ratio
        sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
        
        # Calculate profit factor
        gross_profit = sum(r for r in returns if r > 0)
        gross_loss = abs(sum(r for r in returns if r < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate risk-reward ratio
        avg_win = sum(r for r in returns if r > 0) / len([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = sum(r for r in returns if r < 0) / len([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calculate expectancy
        win_rate = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'volatility': volatility * 100,  # Convert to percentage
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward_ratio,
            'expectancy': expectancy * 100,  # Convert to percentage
            'win_rate': win_rate * 100,  # Convert to percentage
        }
    
    def get_trade_statistics(self) -> Dict:
        """
        Get detailed trade statistics for analysis.
        
        Returns:
            Dictionary containing detailed trade statistics
        """
        closed_trades = [t for t in self.trade_history if t['status'] == 'closed']
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'metrics': None
            }
        
        # Calculate trade metrics
        profitable_trades = [t for t in closed_trades if t.get('profit_loss', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit_loss', 0) <= 0]
        
        # Calculate durations
        trade_durations = []
        for trade in closed_trades:
            if trade['exit_time'] and trade['entry_time']:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600  # in hours
                trade_durations.append(duration)
        
        # Group trades by symbols
        trades_by_symbol = {}
        for trade in closed_trades:
            symbol = trade['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(trade)
        
        # Calculate per-symbol statistics
        symbols_stats = {}
        for symbol, trades in trades_by_symbol.items():
            symbol_pl = sum(t.get('profit_loss', 0) for t in trades)
            profitable = [t for t in trades if t.get('profit_loss', 0) > 0]
            
            symbols_stats[symbol] = {
                'trades': len(trades),
                'win_rate': (len(profitable) / len(trades)) * 100 if trades else 0,
                'net_pl': symbol_pl,
                'avg_pl_per_trade': symbol_pl / len(trades) if trades else 0
            }
        
        return {
            'total_trades': len(closed_trades),
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'win_rate': (len(profitable_trades) / len(closed_trades)) * 100 if closed_trades else 0,
            'avg_profit': sum(t.get('profit_loss', 0) for t in profitable_trades) / len(profitable_trades) if profitable_trades else 0,
            'avg_loss': sum(t.get('profit_loss', 0) for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            'largest_profit': max([t.get('profit_loss', 0) for t in profitable_trades]) if profitable_trades else 0,
            'largest_loss': min([t.get('profit_loss', 0) for t in losing_trades]) if losing_trades else 0,
            'avg_trade_duration_hours': sum(trade_durations) / len(trade_durations) if trade_durations else 0,
            'trades_per_day': len(closed_trades) / max(1, (datetime.now() - self.start_date).days),
            'symbols_performance': symbols_stats,
            'stop_loss_hits': len([t for t in closed_trades if t.get('close_reason') == 'stop_loss']),
            'take_profit_hits': len([t for t in closed_trades if t.get('close_reason') == 'take_profit']),
            'manual_closes': len([t for t in closed_trades if t.get('close_reason') == 'manual']),
        }
    
    def adjust_position(self, symbol: str, adjustment_type: str, adjustment_value: float) -> Dict:
        """
        Adjust parameters of an existing position (stop loss, take profit).
        
        Args:
            symbol: Symbol of the position to adjust
            adjustment_type: Type of adjustment ('stop_loss', 'take_profit')
            adjustment_value: New value for the parameter
            
        Returns:
            Updated position information
        """
        if symbol not in self.active_positions:
            return {'status': 'not_found', 'symbol': symbol}
        
        position = self.active_positions[symbol]
        
        if adjustment_type == 'stop_loss':
            old_value = position['stop_loss']
            position['stop_loss'] = adjustment_value
            logger.info(f"Adjusted stop loss for {symbol} from {old_value} to {adjustment_value}")
        
        elif adjustment_type == 'take_profit':
            old_value = position['take_profit']
            position['take_profit'] = adjustment_value
            logger.info(f"Adjusted take profit for {symbol} from {old_value} to {adjustment_value}")
        
        # Update the position in trade history as well
        for i, trade in enumerate(self.trade_history):
            if trade['symbol'] == symbol and trade['status'] == 'open':
                self.trade_history[i] = position
                break
        
        return position
    
    def get_position_details(self, symbol: str = None) -> Dict:
        """
        Get detailed information about positions.
        
        Args:
            symbol: Optional symbol to get position details for. If None, returns all positions.
            
        Returns:
            Dictionary with position details
        """
        if symbol:
            if symbol in self.active_positions:
                return self.active_positions[symbol]
            return {'status': 'not_found', 'symbol': symbol}
        
        # Return all positions with calculated metrics
        positions = []
        for symbol, position in self.active_positions.items():
            # Get the latest price to calculate current values
            try:
                latest_data = self.api.get_intraday(symbol, interval='5min', output_size='compact')
                latest_price = latest_data.iloc[-1]['close'] if not latest_data.empty else position['entry_price']
                
                # Calculate current P&L
                if position['signal'] == 'buy':
                    current_pl = (latest_price - position['entry_price']) * position['position_size']
                    current_pl_pct = (latest_price / position['entry_price'] - 1) * 100
                else:  # Short position
                    current_pl = (position['entry_price'] - latest_price) * position['position_size']
                    current_pl_pct = (position['entry_price'] / latest_price - 1) * 100
                
                position_copy = position.copy()
                position_copy.update({
                    'current_price': latest_price,
                    'current_pl': current_pl,
                    'current_pl_pct': current_pl_pct,
                    'position_value': latest_price * position['position_size'],
                    'days_open': (datetime.now() - position['entry_time']).days,
                    'hours_open': (datetime.now() - position['entry_time']).total_seconds() / 3600,
                })
                
                positions.append(position_copy)
                
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {str(e)}")
                positions.append(position)
        
        return {
            'total_positions': len(positions),
            'total_value': sum(p.get('position_value', p['position_size'] * p['entry_price']) for p in positions),
            'positions': positions
        }
    
    def export_trade_history(self, format: str = 'dict') -> Union[Dict, pd.DataFrame]:
        """
        Export the trading history in the specified format.
        
        Args:
            format: Output format ('dict' or 'dataframe')
            
        Returns:
            Trade history in the requested format
        """
        if format == 'dataframe':
            return pd.DataFrame(self.trade_history)
        
        return {
            'total_trades': len(self.trade_history),
            'trades': self.trade_history
        }
