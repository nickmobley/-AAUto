"""
Analytics Module - Core analytics engine for real-time data processing and insights generation.

This module provides comprehensive analytics capabilities including:
- Real-time analytics processing
- Pattern recognition
- Predictive analytics
- Performance attribution
- Risk analytics

It integrates with monitoring, validation, and uncertainty systems to ensure
reliable and actionable insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Union, TypeVar, Generic, Callable, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from pathlib import Path
import json
import uuid

# Type definitions
T = TypeVar('T')
DataPoint = TypeVar('DataPoint')
MetricValue = Union[float, int, bool, str, List[float], np.ndarray]

# Configure logger
logger = logging.getLogger(__name__)


class AnalyticsEventType(Enum):
    """Types of analytics events that can be processed."""
    MARKET_DATA = "market_data"
    TRADE_EXECUTION = "trade_execution"
    STRATEGY_SIGNAL = "strategy_signal"
    RISK_ALERT = "risk_alert"
    PERFORMANCE_UPDATE = "performance_update"
    ANOMALY_DETECTED = "anomaly_detected"
    PATTERN_RECOGNIZED = "pattern_recognized"
    PREDICTION_GENERATED = "prediction_generated"


@dataclass
class AnalyticsEvent:
    """Base class for all analytics events."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: AnalyticsEventType = field(default=None)
    source: str = field(default="")
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value if self.event_type else None,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalyticsEvent':
        """Create event from dictionary."""
        event = cls()
        event.id = data.get("id", str(uuid.uuid4()))
        event.timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now()
        event.event_type = AnalyticsEventType(data["event_type"]) if "event_type" in data else None
        event.source = data.get("source", "")
        event.data = data.get("data", {})
        event.metadata = data.get("metadata", {})
        return event


class AnalyticsEngine:
    """
    Core analytics engine that processes events, detects patterns,
    generates predictions, and analyzes performance and risk.
    
    This class integrates with monitoring, validation, and uncertainty systems
    to ensure reliable and actionable insights.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.event_queue = asyncio.Queue()
        self.processors = {}
        self.pattern_detectors = {}
        self.predictive_models = {}
        self.performance_analyzers = {}
        self.risk_analyzers = {}
        self.is_running = False
        self.event_history: List[AnalyticsEvent] = []
        self.max_history_size = self.config.get("max_history_size", 10000)
        
        # Integration points
        self.uncertainty_provider = None
        self.validation_service = None
        self.monitoring_service = None
        
        logger.info("Analytics Engine initialized")
    
    def register_processor(self, event_type: AnalyticsEventType, processor: Callable[[AnalyticsEvent], Any]) -> None:
        """Register a processor for a specific event type."""
        if event_type not in self.processors:
            self.processors[event_type] = []
        self.processors[event_type].append(processor)
        logger.debug(f"Registered processor for {event_type.value}")
    
    def register_pattern_detector(self, name: str, detector: Callable[[List[AnalyticsEvent]], Optional[Dict[str, Any]]]) -> None:
        """Register a pattern detection algorithm."""
        self.pattern_detectors[name] = detector
        logger.debug(f"Registered pattern detector: {name}")
    
    def register_predictive_model(self, name: str, model: Callable[[List[AnalyticsEvent]], Dict[str, Any]]) -> None:
        """Register a predictive model."""
        self.predictive_models[name] = model
        logger.debug(f"Registered predictive model: {name}")
    
    def register_performance_analyzer(self, name: str, analyzer: Callable[[List[AnalyticsEvent]], Dict[str, MetricValue]]) -> None:
        """Register a performance analysis function."""
        self.performance_analyzers[name] = analyzer
        logger.debug(f"Registered performance analyzer: {name}")
    
    def register_risk_analyzer(self, name: str, analyzer: Callable[[List[AnalyticsEvent]], Dict[str, MetricValue]]) -> None:
        """Register a risk analysis function."""
        self.risk_analyzers[name] = analyzer
        logger.debug(f"Registered risk analyzer: {name}")
    
    def set_uncertainty_provider(self, provider: Any) -> None:
        """Set the uncertainty provider for analytics calculations."""
        self.uncertainty_provider = provider
        logger.debug("Uncertainty provider set")
    
    def set_validation_service(self, service: Any) -> None:
        """Set the validation service for analytics output verification."""
        self.validation_service = service
        logger.debug("Validation service set")
    
    def set_monitoring_service(self, service: Any) -> None:
        """Set the monitoring service for performance tracking."""
        self.monitoring_service = service
        logger.debug("Monitoring service set")
    
    async def publish_event(self, event: AnalyticsEvent) -> None:
        """Publish an event to the analytics engine."""
        await self.event_queue.put(event)
        logger.debug(f"Event published: {event.event_type.value if event.event_type else 'unknown'}")
    
    async def _store_event(self, event: AnalyticsEvent) -> None:
        """Store event in history with size limit enforcement."""
        self.event_history.append(event)
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
    
    async def _process_event(self, event: AnalyticsEvent) -> None:
        """Process a single event through registered processors."""
        if event.event_type in self.processors:
            for processor in self.processors[event.event_type]:
                try:
                    await asyncio.to_thread(processor, event)
                except Exception as e:
                    logger.error(f"Error processing event {event.id}: {str(e)}")
        
        # Store event after processing
        await self._store_event(event)
    
    async def _detect_patterns(self) -> List[Dict[str, Any]]:
        """Run all registered pattern detectors on event history."""
        patterns = []
        for name, detector in self.pattern_detectors.items():
            try:
                pattern = await asyncio.to_thread(detector, self.event_history)
                if pattern:
                    pattern["detector"] = name
                    pattern["timestamp"] = datetime.now().isoformat()
                    patterns.append(pattern)
                    logger.info(f"Pattern detected by {name}: {pattern.get('pattern_type', 'unknown')}")
            except Exception as e:
                logger.error(f"Error in pattern detector {name}: {str(e)}")
        return patterns
    
    async def _generate_predictions(self) -> List[Dict[str, Any]]:
        """Generate predictions using all registered predictive models."""
        predictions = []
        for name, model in self.predictive_models.items():
            try:
                prediction = await asyncio.to_thread(model, self.event_history)
                if prediction:
                    prediction["model"] = name
                    prediction["timestamp"] = datetime.now().isoformat()
                    
                    # Add uncertainty estimates if available
                    if self.uncertainty_provider:
                        try:
                            uncertainty = await asyncio.to_thread(
                                self.uncertainty_provider.estimate_uncertainty,
                                prediction
                            )
                            prediction["uncertainty"] = uncertainty
                        except Exception as e:
                            logger.error(f"Error estimating uncertainty for {name}: {str(e)}")
                    
                    predictions.append(prediction)
                    logger.info(f"Prediction generated by {name}")
            except Exception as e:
                logger.error(f"Error in predictive model {name}: {str(e)}")
        return predictions
    
    async def _analyze_performance(self) -> Dict[str, Dict[str, MetricValue]]:
        """Analyze performance using all registered performance analyzers."""
        performance_metrics = {}
        for name, analyzer in self.performance_analyzers.items():
            try:
                metrics = await asyncio.to_thread(analyzer, self.event_history)
                if metrics:
                    performance_metrics[name] = metrics
                    logger.debug(f"Performance metrics generated by {name}")
            except Exception as e:
                logger.error(f"Error in performance analyzer {name}: {str(e)}")
        
        # Validate metrics if validation service is available
        if self.validation_service and performance_metrics:
            try:
                await asyncio.to_thread(
                    self.validation_service.validate_metrics,
                    "performance",
                    performance_metrics
                )
            except Exception as e:
                logger.error(f"Error validating performance metrics: {str(e)}")
        
        return performance_metrics
    
    async def _analyze_risk(self) -> Dict[str, Dict[str, MetricValue]]:
        """Analyze risk using all registered risk analyzers."""
        risk_metrics = {}
        for name, analyzer in self.risk_analyzers.items():
            try:
                metrics = await asyncio.to_thread(analyzer, self.event_history)
                if metrics:
                    risk_metrics[name] = metrics
                    logger.debug(f"Risk metrics generated by {name}")
            except Exception as e:
                logger.error(f"Error in risk analyzer {name}: {str(e)}")
        
        # Validate metrics if validation service is available
        if self.validation_service and risk_metrics:
            try:
                await asyncio.to_thread(
                    self.validation_service.validate_metrics,
                    "risk",
                    risk_metrics
                )
            except Exception as e:
                logger.error(f"Error validating risk metrics: {str(e)}")
        
        return risk_metrics
    
    async def _monitor_analytics_performance(self) -> None:
        """Monitor the performance of the analytics engine itself."""
        if not self.monitoring_service:
            return
        
        try:
            metrics = {
                "event_queue_size": self.event_queue.qsize(),
                "event_history_size": len(self.event_history),
                "pattern_detectors_count": len(self.pattern_detectors),
                "predictive_models_count": len(self.predictive_models),
                "performance_analyzers_count": len(self.performance_analyzers),
                "risk_analyzers_count": len(self.risk_analyzers),
                "timestamp": datetime.now().isoformat()
            }
            
            await asyncio.to_thread(
                self.monitoring_service.report_metrics,
                "analytics_engine",
                metrics
            )
        except Exception as e:
            logger.error(f"Error monitoring analytics performance: {str(e)}")
    
    async def _processing_loop(self) -> None:
        """Main processing loop for the analytics engine."""
        pattern_interval = self.config.get("pattern_detection_interval_seconds", 60)
        prediction_interval = self.config.get("prediction_interval_seconds", 300)
        performance_interval = self.config.get("performance_analysis_interval_seconds", 600)
        risk_interval = self.config.get("risk_analysis_interval_seconds", 300)
        monitoring_interval = self.config.get("monitoring_interval_seconds", 30)
        
        last_pattern_time = datetime.now()
        last_prediction_time = datetime.now()
        last_performance_time = datetime.now()
        last_risk_time = datetime.now()
        last_monitoring_time = datetime.now()
        
        while self.is_running:
            # Process events
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                pass  # No events to process
            
            now = datetime.now()
            
            # Run periodic tasks
            if (now - last_pattern_time).total_seconds() >= pattern_interval:
                patterns = await self._detect_patterns()
                for pattern in patterns:
                    # Create and publish pattern event
                    pattern_event = AnalyticsEvent(
                        event_type=AnalyticsEventType.PATTERN_RECOGNIZED,
                        source="analytics_engine",
                        data=pattern
                    )
                    await self.publish_event(pattern_event)
                last_pattern_time = now
            
            if (now - last_prediction_time).total_seconds() >= prediction_interval:
                predictions = await self._generate_predictions()
                for prediction in predictions:
                    # Create and publish prediction event
                    prediction_event = AnalyticsEvent(
                        event_type=AnalyticsEventType.PREDICTION_GENERATED,
                        source="analytics_engine",
                        data=prediction
                    )
                    await self.publish_event(prediction_event)
                last_prediction_time = now
            
            if (now - last_performance_time).total_seconds() >= performance_interval:
                performance_metrics = await self._analyze_performance()
                if performance_metrics:
                    # Create and publish performance update event
                    performance_event = AnalyticsEvent(
                        event_type=AnalyticsEventType.PERFORMANCE_UPDATE,
                        source="analytics_engine",
                        data={"metrics": performance_metrics}
                    )
                    await self.publish_event(performance_event)
                last_performance_time = now
            
            if (now - last_risk_time).total_seconds() >= risk_interval:
                risk_metrics = await self._analyze_risk()
                if risk_metrics:
                    # Create and publish risk alert event
                    risk_event = AnalyticsEvent(
                        event_type=AnalyticsEventType.RISK_ALERT,
                        source="analytics_engine",
                        data={"metrics": risk_metrics}
                    )
                    await self.publish_event(risk_event)
                last_risk_time = now
            
            if (now - last_monitoring_time).total_seconds() >= monitoring

