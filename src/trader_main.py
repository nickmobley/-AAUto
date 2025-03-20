import os
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

from src.api.alpha_vantage import AlphaVantageAPI
from src.analytics.technical import TechnicalAnalyzer
from src.risk.manager import RiskManager
from src.ml.predictor import MachineLearning
from src.strategies.manager import StrategyManager
from src.strategies.base import BaseStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.trend_following import TrendFollowingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("aauto.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TraderMain:
    """
    Main trading system class that orchestrates the entire trading process.
    
    This class integrates strategy management, risk analysis, trade execution,
    and performance tracking into a unified trading system.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the trading system with configuration settings.
        
        Args:
            config_path: Path to the configuration file (default: "config.json")
        """
        self.config = self._load_config(config_path)
        self.api_client = self._init_api_client()
        self.tech_analyzer = self._init_technical_analyzer()
        self.risk_manager = self._init_risk_manager()
        self.ml_engine = self._init_ml_engine()
        self.strategy_manager = self._init_strategy_manager()
        
        # State tracking
        self.active_positions = {}
        self.trade_history = []
        self.portfolio_value = self.config.get("initial_capital", 10000.0)
        self.is_running = False
        self.last_update_time = None
        
        logger.info("TraderMain initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from the specified file path.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate essential configuration parameters
            required_fields = ['api_key', 'risk_parameters', 'strategy_weights']
            missing_fields = [field for field in required_fields if field not in config]
            
            if missing_fields:
                raise ValueError(f"Missing required configuration fields: {missing_fields}")
                
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {config_path}")
            raise ValueError("Invalid configuration format")
    
    def _init_api_client(self) -> AlphaVantageAPI:
        """Initialize and return the API client for market data."""
        api_key = self.config.get("api_key", os.getenv("ALPHA_VANTAGE_API_KEY"))
        if not api_key:
            raise ValueError("API key is required either in config or as environment variable")
        
        cache_enabled = self.config.get("enable_cache", True)
        rate_limit = self.config.get("api_rate_limit", 5)
        
        return AlphaVantageAPI(api_key, cache_enabled=cache_enabled, rate_limit=rate_limit)
    
    def _init_technical_analyzer(self) -> TechnicalAnalyzer:
        """Initialize and return the technical analysis engine."""
        return TechnicalAnalyzer(self.api_client)
    
    def _init_risk_manager(self) -> RiskManager:
        """Initialize and return the risk management system."""
        risk_params = self.config.get("risk_parameters", {})
        max_position_size = risk_params.get("max_position_size", 0.1)
        max_risk_per_trade = risk_params.get("max_risk_per_trade", 0.02)
        stop_loss_pct = risk_params.get("stop_loss_pct", 0.05)
        
        return RiskManager(
            max_position_size=max_position_size,
            max_risk_per_trade=max_risk_per_trade,
            stop_loss_pct=stop_loss_pct,
            portfolio_value=self.portfolio_value
        )
    
    def _init_ml_engine(self) -> MachineLearning:
        """Initialize and return the machine learning prediction engine."""
        ml_params = self.config.get("ml_parameters", {})
        model_path = ml_params.get("model_path", "models/price_predictor.pkl")
        
        return MachineLearning(self.api_client, model_path=model_path)
    
    def _init_strategy_manager(self) -> StrategyManager:
        """Initialize and return the strategy manager with configured strategies."""
        # Create strategy instances
        momentum_params = self.config.get("momentum_parameters", {})
        mean_reversion_params = self.config.get("mean_reversion_parameters", {})
        trend_following_params = self.config.get("trend_following_parameters", {})
        
        strategies = {
            "momentum": MomentumStrategy(
                self.api_client, 
                self.tech_analyzer,
                **momentum_params
            ),
            "mean_reversion": MeanReversionStrategy(
                self.api_client, 
                self.tech_analyzer,
                **mean_reversion_params
            ),
            "trend_following": TrendFollowingStrategy(
                self.api_client, 
                self.tech_analyzer,
                **trend_following_params
            )
        }
        
        # Get strategy weights from config
        strategy_weights = self.config.get("strategy_weights", {
            "momentum": 0.33,
            "mean_reversion": 0.33,
            "trend_following": 0.34
        })
        
        # Determine combination method
        combination_method = self.config.get("combination_method", "weighted_average")
        
        return StrategyManager(
            strategies=strategies,
            strategy_weights=strategy_weights,
            combination_method=combination_method
        )
    
    def start(self):
        """Start the trading system main loop."""
        if self.is_running:
            logger.warning("Trading system is already running")
            return
        
        self.is_running = True
        logger.info("Trading system started")
        
        try:
            self._run_trading_loop()
        except Exception as e:
            logger.error(f"Error in trading loop: {str(e)}", exc_info=True)
            self.is_running = False
        
    def stop(self):
        """Stop the trading system gracefully."""
        logger.info("Stopping trading system...")
        self.is_running = False
        
        # Close any open positions if configured to do so
        if self.config.get("close_positions_on_stop", True):
            self._close_all_positions()
        
        logger.info("Trading system stopped")
    
    def _run_trading_loop(self):
        """Main trading loop that processes market data and executes trades."""
        update_interval = self.config.get("update_interval_seconds", 60)
        symbols = self.config.get("symbols", ["AAPL", "MSFT", "GOOGL", "AMZN"])
        
        while self.is_running:
            current_time = datetime.now()
            self.last_update_time = current_time
            
            logger.info(f"Running trading iteration at {current_time}")
            
            # Update portfolio value first
            self._update_portfolio_value()
            
            # Process each symbol
            for symbol in symbols:
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {str(e)}", exc_info=True)
            
            # Wait for next iteration
            if self.is_running:
                time.sleep(update_interval)
    
    def _process_symbol(self, symbol: str):
        """
        Process a single symbol - analyze, get signals, and execute trades.
        
        Args:
            symbol: The trading symbol to process
        """
        logger.info(f"Processing symbol: {symbol}")
        
        # Get current market data
        current_price = self._get_current_price(symbol)
        if not current_price:
            logger.warning(f"Could not get current price for {symbol}, skipping")
            return
        
        # Get trading signals from the strategy manager
        signals = self._get_trading_signals(symbol)
        if not signals:
            logger.info(f"No actionable signals for {symbol}")
            return
        
        # Check if we have an existing position
        has_position = symbol in self.active_positions
        
        # Process buy signals
        if signals.get("action") == "buy" and not has_position:
            confidence = signals.get("confidence", 0.0)
            if confidence >= self.config.get("min_confidence_threshold", 0.7):
                self._execute_buy(symbol, current_price, confidence, signals)
        
        # Process sell signals
        elif signals.get("action") == "sell" and has_position:
            confidence = signals.get("confidence", 0.0)
            if confidence >= self.config.get("min_confidence_threshold", 0.6):
                self._execute_sell(symbol, current_price, confidence, signals)
        
        # Process hold signals - update stop losses if needed
        elif has_position:
            self._update_position(symbol, current_price)
    
    def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Current price or None if unavailable
        """
        try:
            data = self.api_client.get_quote(symbol)
            if "price" in data:
                return float(data["price"])
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None
    
    def _get_trading_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading signals from the strategy manager.
        
        Args:
            symbol: The trading symbol
            
        Returns:
            Dictionary containing action, confidence, and metadata
        """
        # Get market data needed for analysis
        try:
            price_data = self.api_client.get_daily(symbol)
            
            # Get ML predictions if enabled
            ml_predictions = None
            if self.config.get("use_ml_predictions", True):
                ml_predictions = self.ml_engine.predict_price(symbol, days=5)
            
            # Run strategy analysis
            signals = self.strategy_manager.get_combined_signal(
                symbol=symbol,
                price_data=price_data,
                ml_predictions=ml_predictions
            )
            
            return signals
        except Exception as e:
            logger.error(f"Error getting trading signals for {symbol}: {str(e)}")
            return {}
    
    def _execute_buy(self, symbol: str, price: float, confidence: float, signals: Dict[str, Any]):
        """
        Execute a buy order for the specified symbol.
        
        Args:
            symbol: The trading symbol
            price: Current market price
            confidence: Signal confidence (0.0-1.0)
            signals: Full signals dictionary with metadata
        """
        logger.info(f"Executing BUY for {symbol} at ${price:.2f} (confidence: {confidence:.2f})")
        
        # Calculate position size based on risk
        position_info = self.risk_manager.calculate_position_size(
            symbol=symbol,
            entry_price=price,
            confidence=confidence
        )
        
        quantity = position_info["quantity"]
        stop_loss = position_info["stop_loss"]
        take_profit = position_info["take_profit"]
        
        if quantity <= 0:
            logger.warning(f"Calculated quantity for {symbol} is {quantity}, skipping buy")
            return
        
        # Execute order (simulated)
        order_cost = price * quantity
        if order_cost > self.portfolio_value:
            logger.warning(f"Insufficient funds to buy {quantity} shares of {symbol}")
            return
        
        # Record the position
        self.active_positions[symbol] = {
            "entry_price": price,
            "quantity": quantity,
            "entry_time": datetime.now(),
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "signals": signals
        }
        
        # Update portfolio value
        self.portfolio_value -= order_cost
        
        # Record trade in history
        self.trade_history.append({
            "symbol": symbol,
            "action": "buy",
            "price": price,
            "quantity": quantity,
            "time": datetime.now(),
            "confidence": confidence,
            "cost": order_cost
        })
        
        logger.info(f"BUY executed: {quantity} shares of {symbol} at ${price:.2f}")
    
    def _execute_sell(self, symbol: str, price: float, confidence: float, signals: Dict[str, Any]):
        """
        Execute a sell order for the specified symbol.
        
        Args:
            symbol: The trading symbol
            price: Current market price
            confidence: Signal confidence (0.0-1.0)
            signals: Full signals dictionary with metadata
        """
        if symbol not in self.active_positions:
            logger.warning(f"Attempting to sell {symbol} but no position exists")
            return
        
        position = self.active_positions[symbol]
        quantity = position["quantity"]
        entry_price = position["entry_price"]
        
        logger.info(f"Executing SELL for {symbol} at ${price:.2f} (confidence: {confidence:.2f})")
        
        # Calculate profit/loss
        proceeds = price * quantity
        cost_basis = entry_price * quantity
        profit_loss = proceeds - cost_basis
        profit_loss_pct = (price - entry_price) / entry_price * 100
        
        # Update portfolio value
        self.portfolio_value += proceeds
        
        # Record trade in history
        self.trade_history.append({
            "symbol": symbol,
            "action": "sell",
            "price": price,
            "quantity": quantity,
            "time": datetime.now(),
            "confidence": confidence,
            "proceeds": proceeds,
            "profit_loss": profit_loss,
            "profit_loss_pct": profit_loss_pct
        })
        
        # Remove the position
        del self.active_positions[symbol]
        
        logger.info(f"SELL executed: {quantity} shares of {symbol} at ${price:.2f} (P/L: ${profit_loss:.2f}, {profit_loss_pct:.2f}%)")
    
    def _update_position(self, symbol: str, current_price: float):
        """
        Update an existing position with current market data.
        
        This includes updating trailing stop loss if applicable, checking
        if stop loss or take profit levels have been hit, and updating
        position metrics.
        
        Args:
            symbol: The trading symbol
            current_price: Current market price
        """
        if symbol not in self.active_positions:
            return
            
        position = self.active_positions[symbol]
        entry_price = position["entry_price"]
        stop_loss = position["stop_loss"]
        take_profit = position["take_profit"]
        
        # Check if stop loss has been hit
        if current_price <= stop_loss:
            logger.info(f"Stop loss triggered for {symbol} at ${current_price:.2f}")
            self._execute_sell(
                symbol, 
                current_price, 
                1.0,  # Max confidence for stop loss
                {"action": "sell", "reason": "stop_loss"}
            )
            return
            
        # Check if take profit has been hit
        if take_profit and current_price >= take_profit:
            logger.info(f"Take profit triggered for {symbol} at ${current_price:.2f}")
            self._execute_sell(
                symbol,
                current_price,
                1.0,  # Max confidence for take profit
                {"action": "sell", "reason": "take_profit"}
            )
            return
            
        # Update trailing stop loss if enabled and price has moved favorably
        if self.config.get("use_trailing_stop", True):
            price_change_pct = (current_price - entry_price) / entry_price
            trailing_threshold = self.config.get("trailing_stop_threshold", 0.02)
            
            if price_change_pct > trailing_threshold:
                # Calculate new stop loss based on trailing percentage
                trailing_pct = self.config.get("trailing_stop_pct", 0.75)
                max_price_move = current_price - entry_price
                new_stop_cushion = max_price_move * trailing_pct
                new_stop_loss = entry_price + new_stop_cushion
                
                # Only move stop loss up, never down
                if new_stop_loss > position["stop_loss"]:
                    position["stop_loss"] = new_stop_loss
                    logger.info(f"Updated trailing stop for {symbol} to ${new_stop_loss:.2f}")
        
        # Update position information with current metrics
        unrealized_pl = (current_price - entry_price) * position["quantity"]
        unrealized_pl_pct = (current_price - entry_price) / entry_price * 100
        position["current_price"] = current_price
        position["unrealized_pl"] = unrealized_pl
        position["unrealized_pl_pct"] = unrealized_pl_pct
        position["last_update_time"] = datetime.now()
    
    def _update_portfolio_value(self):
        """
        Update the total portfolio value based on cash and current positions.
        
        This recalculates the total portfolio value by adding:
        1. Available cash
        2. Market value of all active positions
        """
        # Start with available cash
        total_value = self.portfolio_value
        
        # Add market value of all positions
        for symbol, position in self.active_positions.items():
            current_price = self._get_current_price(symbol)
            if current_price:
                position_value = current_price * position["quantity"]
                total_value += position_value
                
                # Update position with current market value
                position["current_price"] = current_price
                position["current_value"] = position_value
        
        # Calculate change in portfolio value
        initial_capital = self.config.get("initial_capital", 10000.0)
        performance_pct = (total_value - initial_capital) / initial_capital * 100
        
        logger.info(f"Portfolio value: ${total_value:.2f} (Change: {performance_pct:.2f}%)")
        
        # Record portfolio snapshot for historical tracking
        portfolio_snapshot = {
            "timestamp": datetime.now(),
            "total_value": total_value,
            "cash": self.portfolio_value,
            "positions_value": total_value - self.portfolio_value,
            "performance_pct": performance_pct
        }
        
        # Store snapshot in history if configured
        if hasattr(self, "portfolio_history"):
            self.portfolio_history.append(portfolio_snapshot)
    
    def _close_all_positions(self):
        """
        Close all active positions at current market prices.
        
        This method is typically called during system shutdown to liquidate
        all holdings and convert to cash.
        """
        if not self.active_positions:
            logger.info("No active positions to close")
            return
            
        logger.info(f"Closing all active positions ({len(self.active_positions)} total)")
        
        # Create a copy of the symbols list since we'll be modifying the dictionary
        symbols_to_close = list(self.active_positions.keys())
        
        total_proceeds = 0.0
        total_profit_loss = 0.0
        
        for symbol in symbols_to_close:
            current_price = self._get_current_price(symbol)
            if not current_price:
                logger.warning(f"Could not get current price for {symbol}, skipping position close")
                continue
                
            position = self.active_positions[symbol]
            quantity = position["quantity"]
            entry_price = position["entry_price"]
            
            # Calculate profit/loss
            proceeds = current_price * quantity
            cost_basis = entry_price * quantity
            profit_loss = proceeds - cost_basis
            
            # Add to totals
            total_proceeds += proceeds
            total_profit_loss += profit_loss
            
            # Execute the sell
            self._execute_sell(
                symbol,
                current_price,
                1.0,  # Max confidence for system close
                {"action": "sell", "reason": "system_close"}
            )
            
            logger.info(f"Closed position for {symbol}: P/L ${profit_loss:.2f}")
        
        logger.info(f"All positions closed. Total proceeds: ${total_proceeds:.2f}, Total P/L: ${total_profit_loss:.2f}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio status.
        
        Returns:
            Dictionary containing portfolio metrics and position information
        """
        # Calculate total position value
        position_value = 0.0
        for symbol, position in self.active_positions.items():
            if "current_value" in position:
                position_value += position["current_value"]
        
        # Calculate total portfolio value
        total_value = self.portfolio_value + position_value
        
        # Calculate performance metrics
        initial_capital = self.config.get("initial_capital", 10000.0)
        performance_pct = (total_value - initial_capital) / initial_capital * 100
        
        # Count winning and losing positions
        winning_positions = 0
        losing_positions = 0
        
        for position in self.active_positions.values():
            if position.get("unrealized_pl", 0) > 0:
                winning_positions += 1
            else:
                losing_positions += 1
        
        # Create summary
        return {
            "timestamp": datetime.now(),
            "total_value": total_value,
            "cash": self.portfolio_value,
            "position_value": position_value,
            "performance_pct": performance_pct,
            "active_positions": len(self.active_positions),
            "winning_positions": winning_positions,
            "losing_positions": losing_positions,
            "last_update": self.last_update_time
        }
