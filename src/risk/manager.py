"""
Risk Management Module

This module provides risk management functionality for trading operations,
including position sizing, stop loss calculation, and portfolio risk metrics.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, TypedDict
from datetime import datetime
import logging
from decimal import Decimal

# Configure module logger
logger = logging.getLogger(__name__)


class TradeRecord(TypedDict):
    """Type definition for a trade record"""
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    profit_loss: float
    timestamp: datetime


class RiskManager:
    """
    Manages trading risk and position sizing with advanced risk control features.
    
    This class provides methods to calculate appropriate position sizes based on
    risk tolerance, set stop losses and take profit targets, and track trade history
    for risk analysis. It includes validation for all risk parameters to ensure
    they are within acceptable ranges.
    
    Attributes:
        capital (float): Current trading capital amount
        max_risk_per_trade (float): Maximum percentage of capital to risk per trade (0.01 = 1%)
        max_position_size_percent (float): Maximum percentage of capital for any single position
        positions (Dict[str, Dict[str, Any]]): Currently open positions
        trade_history (List[TradeRecord]): Historical trade records
    """
    
    def __init__(
        self, 
        initial_capital: float = 10000.0, 
        max_risk_per_trade: float = 0.02,
        max_position_size_percent: float = 0.20
    ) -> None:
        """
        Initialize the RiskManager with capital and risk parameters.
        
        Args:
            initial_capital: Starting capital amount (default: 10000.0)
            max_risk_per_trade: Maximum risk per trade as decimal (default: 0.02 or 2%)
            max_position_size_percent: Maximum position size as decimal (default: 0.20 or 20%)
            
        Raises:
            ValueError: If risk parameters are outside acceptable ranges
        """
        # Validate input parameters
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not 0 < max_risk_per_trade <= 0.1:
            raise ValueError("Max risk per trade must be between 0 and 0.1 (0-10%)")
        if not 0 < max_position_size_percent <= 1.0:
            raise ValueError("Max position size percent must be between 0 and 1.0 (0-100%)")
            
        self.capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size_percent = max_position_size_percent
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.trade_history: List[TradeRecord] = []
        
        logger.info(f"RiskManager initialized with capital: ${initial_capital:.2f}, "
                   f"max risk per trade: {max_risk_per_trade:.2%}, "
                   f"max position size: {max_position_size_percent:.2%}")
        
    def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: float, 
        stop_loss: float
    ) -> float:
        """
        Calculate safe position size based on risk parameters.
        
        Args:
            symbol: Trading symbol (e.g., stock ticker)
            entry_price: Planned entry price per share/contract
            stop_loss: Stop loss price level
            
        Returns:
            float: Recommended position size (quantity of shares/contracts)
            
        Raises:
            ValueError: If entry_price or stop_loss is invalid
        """
        # Input validation
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {entry_price}")
        if stop_loss <= 0:
            raise ValueError(f"Stop loss must be positive: {stop_loss}")
            
        # For short positions
        if entry_price < stop_loss:
            logger.warning(f"Entry price {entry_price} below stop loss {stop_loss} - short position detected")
            risk_per_share = stop_loss - entry_price
        # For long positions
        elif stop_loss < entry_price:
            risk_per_share = entry_price - stop_loss
        else:
            logger.error(f"Invalid stop loss equal to entry price for {symbol}")
            return 0.0
            
        # Calculate position size based on risk
        risk_amount = self.capital * self.max_risk_per_trade
        
        # Avoid division by zero
        if risk_per_share <= 0:
            logger.error(f"Risk per share must be positive for {symbol}")
            return 0.0
            
        # Calculate raw position size
        position_size = risk_amount / risk_per_share
        
        # Apply maximum position size constraint
        max_position_size = self.capital * self.max_position_size_percent / entry_price
        final_position_size = min(position_size, max_position_size)
        
        logger.info(f"Position size for {symbol}: {final_position_size:.2f} shares "
                   f"(risk: ${risk_amount:.2f}, risk per share: ${risk_per_share:.2f})")
        
        return final_position_size
        
    def set_stop_loss(
        self, 
        symbol: str, 
        entry_price: float, 
        risk_percentage: float = 0.02
    ) -> float:
        """
        Calculate stop loss price based on risk percentage.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price per share/contract
            risk_percentage: Percentage of entry price to risk (default: 0.02 or 2%)
            
        Returns:
            float: Recommended stop loss price
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {entry_price}")
        if not 0 < risk_percentage < 0.5:
            raise ValueError(f"Risk percentage must be between 0 and 0.5 (0-50%): {risk_percentage}")
            
        stop_loss = entry_price * (1 - risk_percentage)
        
        logger.info(f"Set stop loss for {symbol} at ${stop_loss:.2f} "
                   f"(entry: ${entry_price:.2f}, risk: {risk_percentage:.2%})")
        
        return stop_loss
        
    def set_take_profit(
        self, 
        symbol: str, 
        entry_price: float, 
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take profit target based on risk-reward ratio.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price per share/contract
            risk_reward_ratio: Desired risk-reward ratio (default: 2.0)
            
        Returns:
            float: Recommended take profit price
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {entry_price}")
        if risk_reward_ratio <= 0:
            raise ValueError(f"Risk-reward ratio must be positive: {risk_reward_ratio}")
            
        try:
            # Use default risk percentage to calculate stop loss
            stop_loss = self.set_stop_loss(symbol, entry_price)
            
            # Calculate risk amount
            risk = entry_price - stop_loss
            
            # Calculate take profit based on risk-reward ratio
            take_profit = entry_price + (risk * risk_reward_ratio)
            
            logger.info(f"Set take profit for {symbol} at ${take_profit:.2f} "
                       f"(entry: ${entry_price:.2f}, R:R ratio: {risk_reward_ratio})")
            
            return take_profit
            
        except Exception as e:
            logger.error(f"Error setting take profit for {symbol}: {str(e)}")
            # Return a default take profit 2% above entry
            return entry_price * 1.02
        
    def record_trade(
        self, 
        symbol: str, 
        entry_price: float, 
        exit_price: float, 
        position_size: float
    ) -> float:
        """
        Record trade details for analysis and update capital.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price per share/contract
            exit_price: Exit price per share/contract
            position_size: Size of position (quantity of shares/contracts)
            
        Returns:
            float: Profit or loss from the trade
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if entry_price <= 0:
            raise ValueError(f"Entry price must be positive: {entry_price}")
        if exit_price <= 0:
            raise ValueError(f"Exit price must be positive: {exit_price}")
        if position_size <= 0:
            raise ValueError(f"Position size must be positive: {position_size}")
            
        # Calculate profit/loss
        profit_loss = (exit_price - entry_price) * position_size
        
        # Update capital
        self.capital += profit_loss
        
        # Create trade record
        trade: TradeRecord = {
            'symbol': symbol,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_size': position_size,
            'profit_loss': profit_loss,
            'timestamp': datetime.now()
        }
        
        # Add to trade history
        self.trade_history.append(trade)
        
        logger.info(f"Recorded trade for {symbol}: P&L ${profit_loss:.2f}, "
                   f"updated capital: ${self.capital:.2f}")
        
        return profit_loss
        
    def get_max_drawdown(self) -> float:
        """
        Calculate maximum drawdown from trade history.
        
        Maximum drawdown measures the largest peak-to-trough decline
        in portfolio value, expressed as a percentage of the peak value.
        
        Returns:
            float: Maximum drawdown as a decimal (0.1 = 10%)
        """
        if not self.trade_history:
            logger.warning("No trade history available for drawdown calculation")
            return 0.0
            
        # Track cumulative capital after each trade
        capital_history = [self.trade_history[0]['profit_loss']]
        for trade in self.trade_history[1:]:
            capital_history.append(capital_history[-1] + trade['profit_loss'])
            
        # Calculate maximum drawdown
        peak = capital_history[0]
        max_drawdown = 0.0
        
        for capital in capital_history:
            if capital > peak:
                peak = capital
            drawdown = (peak - capital) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        logger.info(f"Maximum drawdown: {max_drawdown:.2%}")
        return max_drawdown
        
    def get_position_exposure(self) -> float:
        """
        Calculate current position exposure as percentage of capital.
        
        Returns:
            float: Current exposure as a decimal (0.5 = 50%)
        """
        total_exposure = sum(
            pos.get('value', 0.0) for pos in self.positions.values()
        )
        exposure_ratio = total_exposure / self.capital if self.capital > 0 else 0
        
        logger.info(f"Current position exposure: {exposure_ratio:.2%} of capital")
        return exposure_ratio
        
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive risk metrics for portfolio analysis.
        
        Returns:
            Dict[str, float]: Dictionary of risk metrics including:
                - max_drawdown: Maximum historical drawdown
                - current_exposure: Current position exposure
                - win_rate: Percentage of winning trades
                - profit_factor: Gross profits divided by gross losses
                - risk_per_trade: Current risk per trade setting
        """
        # Calculate win rate
        if self.trade_history:
            winning_trades = sum(1 for trade in self.trade_history if trade['profit_loss'] > 0)
            win_rate = winning_trades / len(self.trade_history)
            
            # Calculate profit factor
            gross_profit = sum(trade['profit_loss'] for trade in self.trade_history 
                             if trade['profit_loss'] > 0)
            gross_loss = abs(sum(trade['profit_loss'] for trade in self.trade_history 
                               if trade['profit_loss'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
            
        return {
            'max_drawdown': self.get_max_drawdown(),
            'current_exposure': self.get_position_exposure(),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'risk_per_trade': self.max_risk_per_trade
        }

