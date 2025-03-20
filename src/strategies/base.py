"""
Base strategy interface defining the contract for all trading strategies.

This module provides the abstract base class that all trading strategies must implement.
It defines the common interface and functionality that strategies share.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd


class SignalType(Enum):
    """Enum representing different types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """Data class representing a trading signal with associated metadata."""
    type: SignalType
    symbol: str
    timestamp: datetime
    price: float
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate signal data after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Signal confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    It provides common functionality and ensures that all strategies have a
    consistent interface for generating signals and managing their state.
    """
    
    def __init__(self, name: str, symbols: List[str], parameters: Dict[str, Any] = None):
        """
        Initialize the strategy with name, symbols, and optional parameters.
        
        Args:
            name: A unique name for the strategy instance
            symbols: List of trading symbols this strategy will analyze
            parameters: Optional dictionary of strategy-specific parameters
        """
        self.name = name
        self.symbols = symbols
        self.parameters = parameters or {}
        self.is_active = True
        self._last_update_time = None
        self._validation_errors = []
        
        # Validate the strategy parameters
        self._validate_parameters()
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """
        Generate trading signals based on the provided market data.
        
        Args:
            data: Dictionary mapping symbols to their respective DataFrames of market data
                 Expected DataFrame columns include: 'open', 'high', 'low', 'close', 'volume'
        
        Returns:
            List of Signal objects representing trading recommendations
        
        Raises:
            ValueError: If the data format is invalid
            KeyError: If required symbols are missing from the data
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> Dict[str, Any]:
        """
        Get the data requirements for this strategy.
        
        Returns:
            Dictionary describing the data requirements (timeframe, indicators, history length)
        """
        pass
    
    @abstractmethod
    def _validate_parameters(self) -> None:
        """
        Validate the strategy parameters to ensure they are valid.
        
        This method should check that all required parameters are present and valid.
        It should populate self._validation_errors with any issues found.
        
        Returns:
            None
        """
        pass
    
    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update the strategy parameters.
        
        Args:
            parameters: Dictionary of strategy parameters to update
            
        Raises:
            ValueError: If the parameters are invalid
        """
        original_parameters = self.parameters.copy()
        self.parameters.update(parameters)
        self._validation_errors = []
        
        try:
            self._validate_parameters()
            if self._validation_errors:
                error_msg = "; ".join(self._validation_errors)
                raise ValueError(f"Invalid parameters: {error_msg}")
        except Exception as e:
            # Restore original parameters on validation failure
            self.parameters = original_parameters
            raise e
    
    def is_valid(self) -> bool:
        """
        Check if the strategy is valid and ready to use.
        
        Returns:
            True if the strategy is valid, False otherwise
        """
        return len(self._validation_errors) == 0
    
    def get_validation_errors(self) -> List[str]:
        """
        Get the list of validation errors, if any.
        
        Returns:
            List of validation error messages
        """
        return self._validation_errors
    
    def __str__(self) -> str:
        """Return string representation of the strategy."""
        return f"{self.__class__.__name__}(name={self.name}, symbols={self.symbols})"
    
    def __repr__(self) -> str:
        """Return detailed string representation of the strategy."""
        return f"{self.__class__.__name__}(name={self.name}, symbols={self.symbols}, parameters={self.parameters})"

