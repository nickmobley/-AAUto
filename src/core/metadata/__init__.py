"""
Metadata Management System for Tracking Data, Models, Decisions, and System Evolution.

This module provides a comprehensive framework for tracking metadata across the entire system:
- Data Lineage: How data flows from sources through transformations
- Model Adaptation: History of model changes, parameters, and performance
- Decision Provenance: Tracking of decisions, factors, and outcomes
- Regulatory Compliance: Monitoring and documenting regulatory requirements
- System Evolution: Changes to system components, configurations, and behaviors
"""

import asyncio
import datetime
import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


class MetadataType(Enum):
    """Types of metadata tracked by the system."""
    DATA_LINEAGE = "data_lineage"
    MODEL_ADAPTATION = "model_adaptation"
    DECISION = "decision"
    REGULATORY = "regulatory"
    SYSTEM_EVOLUTION = "system_evolution"


@dataclass
class MetadataEvent:
    """Base class for all metadata events."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    type: MetadataType = field(default=None)
    component: str = field(default="")
    version: str = field(default="1.0.0")
    user: str = field(default="system")
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary."""
        data = asdict(self)
        data['type'] = self.type.value if self.type else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataEvent':
        """Create an event from a dictionary."""
        if 'type' in data and data['type']:
            data['type'] = MetadataType(data['type'])
        return cls(**data)


@dataclass
class DataLineageEvent(MetadataEvent):
    """Tracks data flow from sources through transformations."""
    source_id: str = field(default="")
    source_type: str = field(default="")
    operation: str = field(default="")
    input_data_ids: List[str] = field(default_factory=list)
    output_data_id: str = field(default="")
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        self.type = MetadataType.DATA_LINEAGE


@dataclass
class ModelAdaptationEvent(MetadataEvent):
    """Tracks history of model changes, parameters, and performance."""
    model_id: str = field(default="")
    model_type: str = field(default="")
    adaptation_type: str = field(default="")
    previous_version: str = field(default="")
    new_version: str = field(default="")
    parameters_before: Dict[str, Any] = field(default_factory=dict)
    parameters_after: Dict[str, Any] = field(default_factory=dict)
    performance_metrics_before: Dict[str, float] = field(default_factory=dict)
    performance_metrics_after: Dict[str, float] = field(default_factory=dict)
    training_data_id: str = field(default="")
    validation_data_id: str = field(default="")
    reason_for_adaptation: str = field(default="")
    
    def __post_init__(self):
        self.type = MetadataType.MODEL_ADAPTATION


@dataclass
class DecisionEvent(MetadataEvent):
    """Tracks decisions, including factors, alternatives, and outcomes."""
    decision_id: str = field(default="")
    decision_type: str = field(default="")
    description: str = field(default="")
    factors: List[Dict[str, Any]] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    selected_alternative: Dict[str, Any] = field(default_factory=dict)
    confidence: float = field(default=0.0)
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    actual_outcome: Dict[str, Any] = field(default_factory=dict)
    related_decisions: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.type = MetadataType.DECISION


@dataclass
class RegulatoryEvent(MetadataEvent):
    """Tracks regulatory compliance information."""
    regulation_id: str = field(default="")
    regulation_name: str = field(default="")
    jurisdiction: str = field(default="")
    compliance_status: str = field(default="unknown")
    requirements: List[Dict[str, Any]] = field(default_factory=list)
    validation_methods: List[str] = field(default_factory=list)
    documentation_links: List[str] = field(default_factory=list)
    expiration_date: Optional[str] = field(default=None)
    notes: str = field(default="")
    
    def __post_init__(self):
        self.type = MetadataType.REGULATORY


@dataclass
class SystemEvolutionEvent(MetadataEvent):
    """Tracks changes to system components, configurations, and behaviors."""
    change_id: str = field(default="")
    change_type: str = field(default="")
    previous_state: Dict[str, Any] = field(default_factory=dict)
    new_state: Dict[str, Any] = field(default_factory=dict)
    change_reason: str = field(default="")
    approved_by: str = field(default="")
    related_components: List[str] = field(default_factory=list)
    rollback_plan: Dict[str, Any] = field(default_factory=dict)
    performance_impact: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.type = MetadataType.SYSTEM_EVOLUTION


class MetadataStore:
    """Storage and retrieval system for metadata events."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the metadata store.
        
        Args:
            storage_path: Path to store metadata. If None, use in-memory storage.
        """
        self.storage_path = storage_path
        self.in_memory_store: Dict[str, List[MetadataEvent]] = {
            t.value: [] for t in MetadataType
        }
        self._ensure_storage_path()
    
    def _ensure_storage_path(self):
        """Create storage directory if it doesn't exist."""
        if self.storage_path:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            for t in MetadataType:
                type_path = self.storage_path / t.value
                type_path.mkdir(exist_ok=True)
    
    async def store(self, event: MetadataEvent) -> str:
        """Store a metadata event.
        
        Args:
            event: The metadata event to store.
            
        Returns:
            The ID of the stored event.
        """
        if event.type is None:
            raise ValueError("Event type must be specified")
        
        # Store in memory
        self.in_memory_store[event.type.value].append(event)
        
        # Store to disk if path is specified
        if self.storage_path:
            file_path = self.storage_path / event.type.value / f"{event.id}.json"
            event_dict = event.to_dict()
            
            try:
                await asyncio.to_thread(
                    lambda: file_path.write_text(json.dumps(event_dict, indent=2))
                )
            except Exception as e:
                logger.error(f"Failed to write metadata to disk: {e}")
        
        return event.id
    
    async def retrieve(self, event_id: str, event_type: Optional[MetadataType] = None) -> Optional[MetadataEvent]:
        """Retrieve a metadata event by ID.
        
        Args:
            event_id: The ID of the event to retrieve.
            event_type: The type of the event, if known.
            
        Returns:
            The metadata event, or None if not found.
        """
        # Try to find in memory first
        if event_type:
            for event in self.in_memory_store[event_type.value]:
                if event.id == event_id:
                    return event
        else:
            for events in self.in_memory_store.values():
                for event in events:
                    if event.id == event_id:
                        return event
        
        # Look on disk if not found in memory
        if self.storage_path:
            if event_type:
                types_to_check = [event_type.value]
            else:
                types_to_check = [t.value for t in MetadataType]
                
            for type_name in types_to_check:
                file_path = self.storage_path / type_name / f"{event_id}.json"
                if file_path.exists():
                    try:
                        data = json.loads(await asyncio.to_thread(file_path.read_text))
                        return self._create_event_from_dict(data)
                    except Exception as e:
                        logger.error(f"Failed to read metadata from disk: {e}")
        
        return None
    
    async def query(self, 
                   event_type: Optional[MetadataType] = None,
                   component: Optional[str] = None,
                   time_range: Optional[Tuple[str, str]] = None,
                   tags: Optional[List[str]] = None,
                   limit: int = 100) -> List[MetadataEvent]:
        """Query metadata events with filters.
        
        Args:
            event_type: Filter by event type.
            component: Filter by component name.
            time_range: Filter by time range (start, end) in ISO format.
            tags: Filter by tags (must contain all listed tags).
            limit: Maximum number of events to return.
            
        Returns:
            List of matching metadata events.
        """
        result = []
        
        # Determine which types to query
        if event_type:
            types_to_query = [event_type.value]
        else:
            types_to_query = [t.value for t in MetadataType]
        
        # In-memory query
        for type_name in types_to_query:
            for event in self.in_memory_store[type_name]:
                if self._matches_filters(event, component, time_range, tags):
                    result.append(event)
                    if len(result) >= limit:
                        return result[:limit]
        
        # Disk query if needed
        if self.storage_path and len(result) < limit:
            for type_name in types_to_query:
                type_dir = self.storage_path / type_name
                if not type_dir.exists():
                    continue
                
                for file_path in sorted(type_dir.glob("*.json"), 
                                       key=lambda p: p.stat().st_mtime, 
                                       reverse=True):
                    if len(result) >= limit:
                        break
                    
                    try:
                        data = json.loads(await asyncio.to_thread(file_path.read_text))
                        event = self._create_event_from_dict(data)
                        if self._matches_filters(event, component, time_range, tags):
                            result.append(event)
                    except Exception as e:
                        logger.error(f"Failed to read metadata from disk: {e}")
        
        return result[:limit]
    
    def _matches_filters(self, 
                        event: MetadataEvent,
                        component: Optional[str],
                        time_range: Optional[Tuple[str, str]],
                        tags: Optional[List[str]]) -> bool:
        """Check if an event matches the filter criteria."""
        if component and event.component != component:
            return False
        
        if time_range:
            start, end = time_range
            if event.timestamp < start or event.timestamp > end:
                return False
        
        if tags:
            event_tags_set = set(event.tags)
            if not all(tag in event_tags_set for tag in tags):
                return False
        
        return True
    
    def _create_event_from_dict(self, data: Dict[str, Any]) -> MetadataEvent:
        """Create the appropriate event subclass from a dictionary."""
        if 'type' not in data:
            return MetadataEvent.from_dict(data)
        
        event_type = data['type']
        if event_type == MetadataType.DATA_LINEAGE.value:
            return DataLineageEvent.from_dict(data)
        elif event_type == MetadataType.MODEL_ADAPTATION.value:
            return ModelAdaptationEvent.from_dict(data)
        elif event_type == MetadataType.DECISION.value:
            return DecisionEvent.from_dict(data)
        elif event_type == MetadataType.REGULATORY.value:
            return RegulatoryEvent.from_dict(data)
        elif event_type == MetadataType.SYSTEM_EVOLUTION.value:
            return SystemEvolutionEvent.from_dict(data)
        else:
            return MetadataEvent.from_dict(data)


class MetadataTracker:
    """Main interface for tracking and querying metadata."""
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure only one metadata tracker exists."""
        if cls._instance is None:
            cls._instance = super(MetadataTracker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, storage_path: Optional[Union[str, Path]] = None):
        """Initialize the metadata tracker.
        
        Args:
            storage_path: Path to store metadata. If None, use in-memory storage.
        """
        if self._initialized:
            return
        
        if isinstance(storage_path, str):
            storage_path = Path(storage_path)
        
        self.store = MetadataStore(storage_path)
        self._initialized = True
    
    async def track_data_lineage(self, 
                               source_id: str,
                               source_type: str,
                               operation: str,
                               input_data_ids: List[str],
                               output_data_id: str,
                               component: str,
                               parameters: Dict[str, Any] = None,
                               data_quality_metrics: Dict[str, float] = None,
                               tags: List[str] = None) -> str:
        """Track a data lineage event.
        
        Args:
            source_id: ID of the data source.
            source_type: Type of data source.

