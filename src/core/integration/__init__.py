"""
Core Integration System

This module provides standardized integration capabilities for the entire application,
handling communication patterns, external system integration, data transformation,
error handling, and retry mechanisms.
"""

import asyncio
import functools
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Union, cast
from urllib.parse import urljoin

import aiohttp
import backoff
import pydantic
from pydantic import BaseModel, Field, validator

# Import core components for integration
try:
    from src.core.monitoring import MetricsCollector, MonitoringService
    from src.core.recovery import RecoveryManager, RecoveryStrategy
    from src.core.logging import LoggerFactory
    HAS_CORE_DEPENDENCIES = True
except ImportError:
    HAS_CORE_DEPENDENCIES = False
    # Create fallback implementations if core modules aren't available yet
    class MetricsCollector:
        async def record_latency(self, system: str, operation: str, latency: float) -> None: pass
        async def increment_counter(self, name: str, tags: Dict[str, str]) -> None: pass
            
    class MonitoringService:
        def get_metrics_collector(self) -> MetricsCollector: 
            return MetricsCollector()
            
    class RecoveryManager:
        def register_recovery_target(self, target: Any) -> None: pass
        
    class RecoveryStrategy(Enum):
        RETRY = "retry"
        CIRCUIT_BREAKER = "circuit_breaker"
        FALLBACK = "fallback"
        
    class LoggerFactory:
        @staticmethod
        def get_logger(name: str) -> logging.Logger:
            return logging.getLogger(name)

# Type variables for generics
T = TypeVar('T')
R = TypeVar('R')

# ==================================================================================
# Communication Patterns
# ==================================================================================

class CommunicationPattern(Enum):
    """Supported communication patterns for integration."""
    REQUEST_RESPONSE = "request_response"
    PUBLISH_SUBSCRIBE = "publish_subscribe"
    FIRE_AND_FORGET = "fire_and_forget"
    STREAM = "stream"
    SAGA = "saga"
    CIRCUIT_BREAKER = "circuit_breaker"


class MessageFormat(Enum):
    """Supported message formats for integration."""
    JSON = "json"
    XML = "xml"
    PROTOBUF = "protobuf"
    AVRO = "avro"
    PLAIN_TEXT = "plain_text"
    BINARY = "binary"


@dataclass
class IntegrationMessage(Generic[T]):
    """Base message structure for integration."""
    payload: T
    metadata: Dict[str, Any] = Field(default_factory=dict)
    id: str = ""
    correlation_id: str = ""
    timestamp: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class MessageTransformer(Generic[T, R], ABC):
    """Base class for message transformers."""
    
    @abstractmethod
    async def transform(self, message: IntegrationMessage[T]) -> IntegrationMessage[R]:
        """Transform a message from type T to type R."""
        pass


class DefaultJsonTransformer(MessageTransformer[Dict[str, Any], str]):
    """Transforms dictionary to JSON string."""
    
    async def transform(self, message: IntegrationMessage[Dict[str, Any]]) -> IntegrationMessage[str]:
        payload_json = json.dumps(message.payload)
        return IntegrationMessage(
            payload=payload_json,
            metadata=message.metadata,
            id=message.id,
            correlation_id=message.correlation_id,
            timestamp=message.timestamp
        )


class DefaultJsonParser(MessageTransformer[str, Dict[str, Any]]):
    """Parses JSON string to dictionary."""
    
    async def transform(self, message: IntegrationMessage[str]) -> IntegrationMessage[Dict[str, Any]]:
        payload_dict = json.loads(message.payload)
        return IntegrationMessage(
            payload=payload_dict,
            metadata=message.metadata,
            id=message.id,
            correlation_id=message.correlation_id,
            timestamp=message.timestamp
        )


# ==================================================================================
# External System Integration
# ==================================================================================

class IntegrationStatus(Enum):
    """Status of an integration operation."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    RETRY = "retry"
    CIRCUIT_OPEN = "circuit_open"


class IntegrationType(Enum):
    """Types of external system integrations."""
    REST_API = "rest_api"
    DATABASE = "database"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    SOAP = "soap"
    GRAPHQL = "graphql"
    GRPC = "grpc"
    WEBHOOK = "webhook"
    

class IntegrationConfig(BaseModel):
    """Configuration for integration with external systems."""
    name: str
    type: IntegrationType
    base_url: Optional[str] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    headers: Dict[str, str] = Field(default_factory=dict)
    auth_config: Optional[Dict[str, Any]] = None
    
    @validator('retry_attempts')
    def validate_retry_attempts(cls, v):
        if v < 0:
            raise ValueError("retry_attempts must be non-negative")
        return v
    
    @validator('timeout', 'retry_delay', 'circuit_breaker_timeout')
    def validate_positive_floats(cls, v):
        if v <= 0:
            raise ValueError("value must be positive")
        return v


class IntegrationResult(Generic[T]):
    """Result of an integration operation."""
    status: IntegrationStatus
    data: Optional[T] = None
    error: Optional[Exception] = None
    metadata: Dict[str, Any]
    
    def __init__(
        self, 
        status: IntegrationStatus, 
        data: Optional[T] = None, 
        error: Optional[Exception] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.status = status
        self.data = data
        self.error = error
        self.metadata = metadata or {}
        
    @property
    def is_success(self) -> bool:
        return self.status == IntegrationStatus.SUCCESS
    
    @property
    def is_failure(self) -> bool:
        return not self.is_success


class ExternalSystemClient(Generic[T, R], ABC):
    """Base class for external system clients."""
    
    def __init__(self, config: IntegrationConfig, recovery_manager: Optional[RecoveryManager] = None):
        self.config = config
        self.recovery_manager = recovery_manager
        if recovery_manager:
            recovery_manager.register_recovery_target(self)
        self.logger = LoggerFactory.get_logger(f"integration.client.{config.name}")
        
    @abstractmethod
    async def send(self, message: IntegrationMessage[T]) -> IntegrationResult[R]:
        """Send a message to the external system."""
        pass


class RestApiClient(ExternalSystemClient[Dict[str, Any], Dict[str, Any]]):
    """Client for REST API integration."""
    
    def __init__(
        self, 
        config: IntegrationConfig, 
        recovery_manager: Optional[RecoveryManager] = None,
        monitoring_service: Optional[MonitoringService] = None
    ):
        super().__init__(config, recovery_manager)
        self.monitoring_service = monitoring_service or MonitoringService()
        self.metrics = self.monitoring_service.get_metrics_collector()
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers=self.config.headers, 
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None
    
    @backoff.on_exception(
        backoff.expo, 
        (aiohttp.ClientError, asyncio.TimeoutError), 
        max_tries=lambda self: self.config.retry_attempts + 1,
        giveup=lambda e: isinstance(e, aiohttp.ClientResponseError) and e.status not in [429, 500, 502, 503, 504]
    )
    async def send(self, message: IntegrationMessage[Dict[str, Any]]) -> IntegrationResult[Dict[str, Any]]:
        """Send a request to the REST API."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers=self.config.headers, 
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            
        start_time = time.time()
        metadata = {"request_time": start_time}
        
        method = message.metadata.get("http_method", "GET")
        endpoint = message.metadata.get("endpoint", "")
        params = message.metadata.get("params", {})
        
        if not self.config.base_url:
            return IntegrationResult(
                status=IntegrationStatus.FAILURE,
                error=ValueError("base_url is required for REST API integration"),
                metadata=metadata
            )
            
        url = urljoin(self.config.base_url, endpoint)
        
        try:
            async with getattr(self.session, method.lower())(
                url,
                params=params,
                json=message.payload if method.upper() in ["POST", "PUT", "PATCH"] else None,
                ssl=message.metadata.get("ssl", None)
            ) as response:
                response.raise_for_status()
                data = await response.json()
                
                latency = time.time() - start_time
                await self.metrics.record_latency(
                    system=self.config.name,
                    operation=f"{method}_{endpoint}",
                    latency=latency
                )
                
                metadata.update({
                    "response_time": time.time(),
                    "status_code": response.status,
                    "latency": latency
                })
                
                return IntegrationResult(
                    status=IntegrationStatus.SUCCESS,
                    data=data,
                    metadata=metadata
                )
                
        except aiohttp.ClientResponseError as e:
            await self.metrics.increment_counter(
                name="api_error",
                tags={"system": self.config.name, "endpoint": endpoint, "status_code": str(e.status)}
            )
            
            self.logger.error(f"API error: {e.status} - {e.message}")
            metadata.update({
                "error_time": time.time(),
                "status_code": e.status,
                "error_type": "response_error"
            })
            
            return IntegrationResult(
                status=IntegrationStatus.FAILURE,
                error=e,
                metadata=metadata
            )
            
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            await self.metrics.increment_counter(
                name="api_error",
                tags={"system": self.config.name, "endpoint": endpoint, "error_type": type(e).__name__}
            )
            
            self.logger.error(f"Connection error: {str(e)}")
            metadata.update({
                "error_time": time.time(),
                "error_type": type(e).__name__
            })
            
            return IntegrationResult(
                status=IntegrationStatus.FAILURE,
                error=e,
                metadata=metadata
            )
            
        except Exception as e:
            await self.metrics.increment_counter(
                name="api_error",
                tags={"system": self.config.name, "endpoint": endpoint, "error_type": "unexpected"}
            )
            
            self.logger.exception(f"Unexpected error: {str(e)}")
            metadata.update({
                "error_time": time.time(),
                "error_type": "unexpected"
            })
            
            return IntegrationResult(
                status=IntegrationStatus.FAILURE,
                error=e,
                metadata=metadata
            )


# ==================================================================================
# Integration Service Registry
# ==================================================================================

class IntegrationRegistry:
    """Registry for integration services."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IntegrationRegistry, cls).__new__(cls)
            cls._instance._clients = {}
            cls._instance._transformers = {}
            cls._instance.logger = LoggerFactory.get_logger("integration.registry")
        return cls._instance
    
    def register_client(self, name: str, client: ExternalSystemClient):
        """Register an external system client."""
        self._clients[name] = client
        self.logger.info(f"Registered integration client: {name}")
        
    def get_client(self, name: str) -> Optional[ExternalSystemClient]:
        """Get an external system client by name."""
        return self._clients.get(name)
    
    def register_transformer(self, source_type: Type, target_type: Type, transformer: MessageTransformer):
        """Register a message transformer."""
        key = (source_type, target_type)
        self._transformers[key] = transformer
        self.logger.info(f"Registered transformer: {source_type.__name__} -> {target_type.__name__}")
        
    def get_transformer(self, source_type: Type, target_type: Type) -> Optional[MessageTransformer]:
        """Get a message transformer by source and target types."""
        return self._transformers.get((source_type, target_type))


# ==================================================================================
# Data Transformation
# ==================================================================================

class TransformationPipeline(Generic[T, R]):
    """Pipeline for transforming data through multiple steps."""
    
    def __init__(self):
        self.steps: List[MessageTransformer] = []
        self.logger = LoggerFactory.get_logger("integration.transformation")
        
    def add_step(self, transformer: MessageTransformer) -> 'TransformationPipeline':
        """Add a transformation step to the pipeline."""
        self.steps.append(transformer)
        return self
        
    async def execute(self, message: IntegrationMessage[T]) -> IntegrationResult[R]:
        """Execute the transformation pipeline."""
        try:
            current_message = message
            for step in self.steps:
                current_message = await step.transform(current_message)
                
            return IntegrationResult(
                status=IntegrationStatus.SUCCESS,
                data=cast(R, current_message.payload),
                metadata={"original_metadata": message.metadata}
            )
        except Exception as e:
            self.logger.exception(f"Transformation error: {str(e)}")
            return IntegrationResult(
                status=IntegrationStatus.FAILURE,
                error=e,
                metadata={"error_type": "transformation_error"}
            )


class ModelTransformer(MessageTransformer[

