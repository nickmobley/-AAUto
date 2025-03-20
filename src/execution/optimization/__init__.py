"""
Execution Optimization Module

This module provides sophisticated execution optimization capabilities:
1. Smart order routing to find optimal execution venues
2. Market impact analysis and minimization
3. Adaptive execution strategies based on market conditions
4. Real-time execution quality analysis
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Protocol, Set, Tuple, TypeVar, Union, cast
import uuid

# Setup module logger
logger = logging.getLogger(__name__)

# Type definitions
OrderId = str
VenueId = str
Price = float
Quantity = float
Symbol = str

T = TypeVar('T')


class OrderType(Enum):
    """Order types supported by the execution system."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    ICEBERG = "iceberg"
    TWAP = "twap"
    VWAP = "vwap"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class MarketCondition(Enum):
    """Market condition classifications."""
    NORMAL = "normal"
    VOLATILE = "volatile"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    LOW_LIQUIDITY = "low_liquidity"
    HIGH_LIQUIDITY = "high_liquidity"


class ExecutionAlgorithm(Enum):
    """Available execution algorithms."""
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    POV = "pov"    # Percentage of Volume
    IS = "is"      # Implementation Shortfall
    ADAPTIVE = "adaptive"  # Dynamic algorithm selection


@dataclass
class OrderRequest:
    """Represents an order request before execution."""
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    order_type: OrderType
    limit_price: Optional[Price] = None
    stop_price: Optional[Price] = None
    algo_type: Optional[ExecutionAlgorithm] = None
    algo_params: Optional[Dict[str, Union[str, float, int, bool]]] = None
    time_in_force: Optional[str] = "DAY"
    request_id: str = ""
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class ExecutionVenue:
    """Information about an execution venue."""
    id: VenueId
    name: str
    fee_structure: Dict[str, float]
    supported_order_types: Set[OrderType]
    liquidity_score: float
    latency_ms: float
    
    @property
    def effective_cost(self) -> float:
        """Calculate the effective cost score (lower is better)."""
        return (self.fee_structure.get("taker", 0.0) * 100) + (self.latency_ms / 10)


@dataclass
class MarketImpactEstimate:
    """Estimated market impact for an order."""
    price_impact_bps: float
    expected_slippage_bps: float
    confidence_score: float
    impact_factors: Dict[str, float]


@dataclass
class ExecutionResult:
    """Results from an executed order."""
    order_id: OrderId
    request_id: str
    symbol: Symbol
    side: OrderSide
    executed_quantity: Quantity
    average_price: Price
    venue_id: VenueId
    timestamp: datetime
    execution_time_ms: float
    market_impact_bps: float
    slippage_bps: float
    fees: float
    is_complete: bool


class MarketDataProvider(Protocol):
    """Protocol for market data providers."""
    
    async def get_orderbook(self, symbol: Symbol) -> Dict[str, any]:
        """Get the current orderbook for a symbol."""
        ...
    
    async def get_recent_trades(self, symbol: Symbol, limit: int = 100) -> List[Dict[str, any]]:
        """Get recent trades for a symbol."""
        ...
    
    async def get_market_condition(self, symbol: Symbol) -> MarketCondition:
        """Get the current market condition for a symbol."""
        ...


class ExecutionVenueProvider(Protocol):
    """Protocol for execution venue providers."""
    
    async def get_available_venues(self, symbol: Symbol) -> List[ExecutionVenue]:
        """Get available venues for trading a symbol."""
        ...
    
    async def execute_order(self, venue_id: VenueId, order: OrderRequest) -> ExecutionResult:
        """Execute an order on a specific venue."""
        ...


class MarketImpactAnalyzer:
    """Analyzes and predicts market impact of orders."""
    
    def __init__(self, market_data_provider: MarketDataProvider):
        self.market_data_provider = market_data_provider
        self._impact_model_cache: Dict[Symbol, Dict[str, float]] = {}
    
    async def estimate_market_impact(self, order: OrderRequest) -> MarketImpactEstimate:
        """
        Estimate the market impact for an order.
        
        Args:
            order: The order to estimate impact for
            
        Returns:
            A market impact estimate
        """
        symbol = order.symbol
        
        # Get current market conditions
        orderbook = await self.market_data_provider.get_orderbook(symbol)
        recent_trades = await self.market_data_provider.get_recent_trades(symbol)
        market_condition = await self.market_data_provider.get_market_condition(symbol)
        
        # Calculate order size relative to recent volume
        recent_volume = sum(trade.get("quantity", 0) for trade in recent_trades)
        order_size_ratio = order.quantity / max(recent_volume, 1)
        
        # Calculate available liquidity on the relevant side
        if order.side == OrderSide.BUY:
            side_liquidity = sum(level.get("quantity", 0) for level in orderbook.get("asks", []))
        else:
            side_liquidity = sum(level.get("quantity", 0) for level in orderbook.get("bids", []))
        
        liquidity_ratio = order.quantity / max(side_liquidity, 1)
        
        # Adjust for market conditions
        condition_factor = 1.0
        if market_condition == MarketCondition.VOLATILE:
            condition_factor = 1.5
        elif market_condition == MarketCondition.LOW_LIQUIDITY:
            condition_factor = 1.7
        
        # Calculate price impact
        base_impact = 10.0  # Base impact in bps
        price_impact_bps = base_impact * order_size_ratio * liquidity_ratio * condition_factor
        
        # Calculate expected slippage
        expected_slippage_bps = price_impact_bps * 0.8
        
        # Populate impact factors
        impact_factors = {
            "order_size_ratio": order_size_ratio,
            "liquidity_ratio": liquidity_ratio,
            "condition_factor": condition_factor,
            "market_volatility": market_condition.value
        }
        
        return MarketImpactEstimate(
            price_impact_bps=price_impact_bps,
            expected_slippage_bps=expected_slippage_bps,
            confidence_score=0.8,  # Placeholder, could be calculated from historical accuracy
            impact_factors=impact_factors
        )
    
    async def update_impact_model(self, symbol: Symbol, actual_results: List[ExecutionResult]) -> None:
        """
        Update the impact model based on actual execution results.
        
        Args:
            symbol: The market symbol
            actual_results: Recent execution results to learn from
        """
        if not actual_results:
            return
        
        # Calculate prediction errors
        error_sum = 0.0
        for result in actual_results:
            # Get the cached prediction if available
            cached_model = self._impact_model_cache.get(symbol, {})
            predicted_impact = cached_model.get("predicted_impact", 10.0)
            actual_impact = result.market_impact_bps
            
            error_sum += (actual_impact - predicted_impact) ** 2
        
        rmse = (error_sum / len(actual_results)) ** 0.5
        logger.info(f"Market impact model RMSE for {symbol}: {rmse:.2f} bps")
        
        # Update model parameters based on recent results
        # (In a real implementation, this would use more sophisticated machine learning)
        self._impact_model_cache.setdefault(symbol, {})
        self._impact_model_cache[symbol]["rmse"] = rmse
        
        # Log the update
        logger.info(f"Updated market impact model for {symbol}")


class ExecutionQualityAnalyzer:
    """Analyzes execution quality in real-time and historically."""
    
    def __init__(self):
        self._execution_history: Dict[Symbol, List[ExecutionResult]] = {}
        self._benchmark_prices: Dict[Symbol, Dict[str, float]] = {}
    
    async def analyze_execution(self, result: ExecutionResult) -> Dict[str, float]:
        """
        Analyze the quality of a completed execution.
        
        Args:
            result: The execution result to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        symbol = result.symbol
        
        # Store in history for future analysis
        self._execution_history.setdefault(symbol, []).append(result)
        
        # Calculate basic quality metrics
        metrics = {
            "market_impact_bps": result.market_impact_bps,
            "slippage_bps": result.slippage_bps,
            "execution_speed_ms": result.execution_time_ms,
            "fees_bps": (result.fees / (result.average_price * result.executed_quantity)) * 10000,
        }
        
        # Calculate implementation shortfall if benchmark price available
        benchmarks = self._benchmark_prices.get(symbol, {})
        if "arrival" in benchmarks:
            arrival_price = benchmarks["arrival"]
            if result.side == OrderSide.BUY:
                shortfall_bps = (result.average_price - arrival_price) / arrival_price * 10000
            else:
                shortfall_bps = (arrival_price - result.average_price) / arrival_price * 10000
            metrics["implementation_shortfall_bps"] = shortfall_bps
        
        return metrics
    
    async def set_benchmark_price(self, symbol: Symbol, benchmark_type: str, price: float) -> None:
        """
        Set a benchmark price for execution quality analysis.
        
        Args:
            symbol: The market symbol
            benchmark_type: Type of benchmark (e.g., 'arrival', 'vwap', 'close')
            price: The benchmark price
        """
        self._benchmark_prices.setdefault(symbol, {})[benchmark_type] = price
    
    async def get_historical_performance(self, symbol: Symbol, metrics: List[str], 
                                        lookback_count: int = 100) -> Dict[str, float]:
        """
        Get historical execution performance statistics.
        
        Args:
            symbol: The market symbol
            metrics: List of metrics to calculate
            lookback_count: Number of recent executions to include
            
        Returns:
            Dictionary of average metric values
        """
        history = self._execution_history.get(symbol, [])
        
        if not history:
            return {metric: 0.0 for metric in metrics}
        
        recent_executions = history[-lookback_count:] if lookback_count > 0 else history
        
        # Calculate averages for requested metrics
        results = {}
        for metric in metrics:
            if metric == "market_impact_bps":
                results[metric] = sum(e.market_impact_bps for e in recent_executions) / len(recent_executions)
            elif metric == "slippage_bps":
                results[metric] = sum(e.slippage_bps for e in recent_executions) / len(recent_executions)
            elif metric == "execution_time_ms":
                results[metric] = sum(e.execution_time_ms for e in recent_executions) / len(recent_executions)
            elif metric == "fill_rate":
                results[metric] = sum(1.0 if e.is_complete else e.executed_quantity / e.executed_quantity 
                                    for e in recent_executions) / len(recent_executions)
            
        return results


class BaseExecutionStrategy(ABC):
    """Base class for execution strategies."""
    
    @abstractmethod
    async def execute(self, order: OrderRequest, venues: List[ExecutionVenue], 
                     impact_analyzer: MarketImpactAnalyzer,
                     venue_provider: ExecutionVenueProvider) -> List[ExecutionResult]:
        """
        Execute an order using the strategy.
        
        Args:
            order: The order to execute
            venues: Available venues for execution
            impact_analyzer: Market impact analyzer
            venue_provider: Execution venue provider
            
        Returns:
            List of execution results
        """
        pass


class TWAPStrategy(BaseExecutionStrategy):
    """Time-Weighted Average Price execution strategy."""
    
    async def execute(self, order: OrderRequest, venues: List[ExecutionVenue],
                     impact_analyzer: MarketImpactAnalyzer,
                     venue_provider: ExecutionVenueProvider) -> List[ExecutionResult]:
        """
        Execute using TWAP strategy.
        
        Args:
            order: The order to execute
            venues: Available venues for execution
            impact_analyzer: Market impact analyzer
            venue_provider: Execution venue provider
            
        Returns:
            List of execution results
        """
        # Get strategy parameters
        params = order.algo_params or {}
        num_slices = int(params.get("num_slices", 10))
        interval_seconds = float(params.get("interval_seconds", 30.0))
        
        # Calculate slice size
        slice_qty = order.quantity / num_slices
        
        # Select best venue
        venue = max(venues, key=lambda v: -v.effective_cost)
        
        results = []
        remaining_qty = order.quantity
        
        # Execute slices
        for i in range(num_slices):
            if remaining_qty <= 0:
                break
                
            # Adjust final slice for rounding
            if i == num_slices - 1:
                slice_qty = remaining_qty
            
            # Create slice order
            slice_order = OrderRequest(
                symbol=order.symbol,
                side=order.side,
                quantity=slice_qty,
                order_type=OrderType.LIMIT if order.limit_price else OrderType.MARKET,
                limit_price=order.limit_price,
                request_id=f"{order.request_id}_slice_{i}",
            )
            
            # Execute slice
            result = await venue_provider.execute_order(venue.id, slice_order)
            results.append(result)
            
            # Update remaining quantity
            remaining_qty -= result.executed_quantity
            
            # Wait for interval unless last slice
            if i < num_slices - 1:
                await asyncio.sleep(interval_seconds)
                
        return results


class POVStrategy(Base

