"""
Testing Framework for Adaptive Trading System

This module provides a comprehensive testing infrastructure including:
1. Market simulation framework
2. Component testing framework 
3. Integration testing system
4. Performance testing suite

The framework is designed to ensure thorough testing of all system components
with proper integration and validation capabilities.
"""

import asyncio
import logging
import time
import unittest
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


#-------------------------------------------------------
# Market Simulation Framework
#-------------------------------------------------------

class MarketType(Enum):
    """Types of markets that can be simulated."""
    EQUITIES = "equities"
    FOREX = "forex"
    CRYPTO = "crypto"
    FUTURES = "futures"
    OPTIONS = "options"


class MarketRegime(Enum):
    """Market regimes that can be simulated."""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    HIGH_CORRELATION = "high_correlation"
    SECTOR_ROTATION = "sector_rotation"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    FLASH_CRASH = "flash_crash"
    BLACK_SWAN = "black_swan"


@dataclass
class MarketEvent:
    """Represents a market event in the simulation."""
    timestamp: float
    event_type: str
    symbol: str
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketData:
    """Container for market data used in simulations."""
    
    def __init__(self, 
                 data: Optional[pd.DataFrame] = None,
                 symbols: Optional[List[str]] = None,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None):
        """Initialize with optional data."""
        self.data = data if data is not None else pd.DataFrame()
        self.symbols = symbols if symbols is not None else []
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time = end_time
        self._event_queue = asyncio.Queue()
        
    async def add_event(self, event: MarketEvent) -> None:
        """Add event to the queue."""
        await self._event_queue.put(event)
        
    async def get_event(self) -> Optional[MarketEvent]:
        """Get next event from the queue."""
        if self._event_queue.empty():
            return None
        return await self._event_queue.get()
    
    def get_ohlcv(self, 
                  symbol: str, 
                  start_time: Optional[float] = None, 
                  end_time: Optional[float] = None) -> pd.DataFrame:
        """Get OHLCV data for a symbol within a time range."""
        if symbol not in self.symbols:
            return pd.DataFrame()
            
        filtered = self.data[self.data['symbol'] == symbol]
        
        if start_time is not None:
            filtered = filtered[filtered['timestamp'] >= start_time]
        if end_time is not None:
            filtered = filtered[filtered['timestamp'] <= end_time]
            
        return filtered


class MarketSimulator:
    """Simulates market conditions for testing."""
    
    def __init__(self, 
                 market_type: MarketType = MarketType.EQUITIES,
                 market_regime: MarketRegime = MarketRegime.RANGING,
                 symbols: Optional[List[str]] = None,
                 start_time: Optional[float] = None,
                 end_time: Optional[float] = None,
                 volatility: float = 0.15,
                 tick_interval: float = 1.0):
        """Initialize the market simulator with configuration."""
        self.market_type = market_type
        self.market_regime = market_regime
        self.symbols = symbols if symbols is not None else ["AAPL", "MSFT", "GOOGL"]
        self.start_time = start_time if start_time is not None else time.time()
        self.end_time = end_time
        self.volatility = volatility
        self.tick_interval = tick_interval
        self.market_data = MarketData(symbols=self.symbols, start_time=self.start_time, end_time=self.end_time)
        self._running = False
        self._task = None
    
    async def start(self) -> None:
        """Start the market simulation."""
        self._running = True
        self._task = asyncio.create_task(self._run_simulation())
        logger.info(f"Started market simulation for {self.market_type.value} in {self.market_regime.value} regime")
    
    async def stop(self) -> None:
        """Stop the market simulation."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped market simulation")
    
    async def _run_simulation(self) -> None:
        """Run the market simulation loop."""
        current_time = self.start_time
        
        while self._running:
            if self.end_time and current_time > self.end_time:
                break
                
            for symbol in self.symbols:
                # Generate simulated price and volume based on market regime
                price, volume = self._generate_market_data(symbol, current_time)
                
                # Create and publish market event
                event = MarketEvent(
                    timestamp=current_time,
                    event_type="tick",
                    symbol=symbol,
                    data={
                        "price": price,
                        "volume": volume,
                        "bid": price * 0.999,
                        "ask": price * 1.001,
                    }
                )
                await self.market_data.add_event(event)
            
            current_time += self.tick_interval
            await asyncio.sleep(self.tick_interval / 100)  # Scaled for faster simulation
    
    def _generate_market_data(self, symbol: str, timestamp: float) -> Tuple[float, float]:
        """Generate simulated market data based on regime."""
        # Base price and volume (would be more sophisticated in a real implementation)
        base_price = 100.0 + hash(symbol) % 900  # Different base price per symbol
        base_volume = 10000 + hash(symbol[::-1]) % 90000  # Different base volume per symbol
        
        # Adjust based on market regime
        if self.market_regime == MarketRegime.TRENDING_BULL:
            price_factor = 1.0 + (0.0001 * (timestamp - self.start_time)) + (np.random.randn() * self.volatility * 0.1)
            volume_factor = 1.0 + np.random.randn() * 0.2
        elif self.market_regime == MarketRegime.TRENDING_BEAR:
            price_factor = 1.0 - (0.0001 * (timestamp - self.start_time)) + (np.random.randn() * self.volatility * 0.1)
            volume_factor = 1.0 + np.random.randn() * 0.3
        elif self.market_regime == MarketRegime.VOLATILE:
            price_factor = 1.0 + (np.random.randn() * self.volatility * 0.5)
            volume_factor = 1.0 + abs(np.random.randn() * 0.5)
        elif self.market_regime == MarketRegime.FLASH_CRASH:
            # Sudden price drop if within "crash window"
            crash_window = (self.end_time - self.start_time) * 0.4 if self.end_time else 3600
            crash_time = self.start_time + crash_window
            if timestamp >= crash_time and timestamp < crash_time + 300:  # 5-minute crash
                crash_progress = (timestamp - crash_time) / 300  # 0 to 1 during crash
                price_factor = 1.0 - (0.2 * crash_progress) + (np.random.randn() * self.volatility)
                volume_factor = 3.0 + abs(np.random.randn() * 2.0)  # High volume during crash
            else:
                price_factor = 1.0 + (np.random.randn() * self.volatility * 0.2)
                volume_factor = 1.0 + np.random.randn() * 0.2
        else:  # Default ranging behavior
            price_factor = 1.0 + (np.random.randn() * self.volatility * 0.2)
            volume_factor = 1.0 + np.random.randn() * 0.2
        
        return base_price * price_factor, base_volume * volume_factor


class OrderBook:
    """Simulated order book for testing execution systems."""
    
    def __init__(self, symbol: str):
        """Initialize an empty order book for a symbol."""
        self.symbol = symbol
        self.bids = []  # List of (price, quantity, order_id) tuples
        self.asks = []  # List of (price, quantity, order_id) tuples
        self.last_order_id = 0
    
    def add_bid(self, price: float, quantity: float) -> int:
        """Add a bid to the order book."""
        self.last_order_id += 1
        self.bids.append((price, quantity, self.last_order_id))
        self.bids.sort(key=lambda x: -x[0])  # Sort by price descending
        return self.last_order_id
    
    def add_ask(self, price: float, quantity: float) -> int:
        """Add an ask to the order book."""
        self.last_order_id += 1
        self.asks.append((price, quantity, self.last_order_id))
        self.asks.sort(key=lambda x: x[0])  # Sort by price ascending
        return self.last_order_id
    
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order by ID."""
        for i, (_, _, oid) in enumerate(self.bids):
            if oid == order_id:
                self.bids.pop(i)
                return True
                
        for i, (_, _, oid) in enumerate(self.asks):
            if oid == order_id:
                self.asks.pop(i)
                return True
                
        return False
    
    def match_orders(self) -> List[Dict[str, Any]]:
        """Match orders and return executed trades."""
        trades = []
        
        while self.bids and self.asks:
            best_bid = self.bids[0]
            best_ask = self.asks[0]
            
            if best_bid[0] >= best_ask[0]:  # Bid price >= Ask price -> Trade
                execute_price = best_ask[0]  # Execute at ask price (taker pays)
                execute_quantity = min(best_bid[1], best_ask[1])
                
                # Record the trade
                trades.append({
                    "symbol": self.symbol,
                    "price": execute_price,
                    "quantity": execute_quantity,
                    "timestamp": time.time(),
                    "bid_order_id": best_bid[2],
                    "ask_order_id": best_ask[2],
                })
                
                # Update quantities or remove orders
                remaining_bid_qty = best_bid[1] - execute_quantity
                remaining_ask_qty = best_ask[1] - execute_quantity
                
                if remaining_bid_qty > 0:
                    self.bids[0] = (best_bid[0], remaining_bid_qty, best_bid[2])
                else:
                    self.bids.pop(0)
                    
                if remaining_ask_qty > 0:
                    self.asks[0] = (best_ask[0], remaining_ask_qty, best_ask[2])
                else:
                    self.asks.pop(0)
            else:
                break  # No more matches possible
                
        return trades
    
    def get_bid_ask_spread(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get current bid-ask spread."""
        best_bid = self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        spread = best_ask - best_bid if (best_bid is not None and best_ask is not None) else None
        return best_bid, best_ask, spread


class ExchangeSimulator:
    """Simulates a trading exchange with order books."""
    
    def __init__(self, symbols: Optional[List[str]] = None):
        """Initialize the exchange simulator."""
        self.symbols = symbols if symbols is not None else ["AAPL", "MSFT", "GOOGL"]
        self.order_books = {symbol: OrderBook(symbol) for symbol in self.symbols}
        self.trades = []
        self.fees = 0.001  # 0.1% fee
    
    def place_limit_order(self, 
                         symbol: str, 
                         side: str, 
                         price: float, 
                         quantity: float) -> Optional[int]:
        """Place a limit order on the exchange."""
        if symbol not in self.order_books:
            return None
            
        order_book = self.order_books[symbol]
        
        if side.upper() == "BUY":
            return order_book.add_bid(price, quantity)
        elif side.upper() == "SELL":
            return order_book.add_ask(price, quantity)
        else:
            return None
    
    def place_market_order(self,
                          symbol: str,
                          side: str,
                          quantity: float) -> List[Dict[str, Any]]:
        """Place a market order and return executed trades."""
        if symbol not in self.order_books:
            return []
            
        order_book = self.order_books[symbol]
        remaining_quantity = quantity
        executed_trades = []
        
        if side.upper() == "BUY":
            while remaining_quantity > 0 and order_book.asks:
                best_ask = order_book.asks[0]
                execute_quantity = min(remaining_quantity, best_ask[1])
                
                # Record the trade
                trade = {
                    "symbol": symbol,
                    "price": best_ask[0],
                    "quantity": execute_quantity,
                    "timestamp": time.time(),
                    "side": "BUY",
                    "fee": best_ask[0] * execute_quantity * self.fees
                }
                executed_trades.append(trade)
                self.trades.append(trade)
                
                # Update the order book
                if execute_quantity == best_ask[1]:
                    order_book.asks.pop(0)
                else:
                    order_book.asks[0] = (best_ask[0], best_ask[1] - execute_quantity, best_ask[2])
                
                remaining_quantity -= execute_quantity
                
        elif side.upper() == "SELL":
            while remaining_quantity > 0 and order_book.bids:
                best_

