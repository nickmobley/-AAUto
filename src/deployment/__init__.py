"""
Deployment management system for AAUto.

This module provides comprehensive deployment capabilities including:
- Component deployment management
- Version control and tracking
- Rollback mechanisms
- System state verification
- Integration with orchestration, monitoring, and configuration
"""

import asyncio
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import uuid

# Try to import core system components, with graceful fallback if not available
try:
    from src.core.orchestration import Orchestrator
    from src.core.monitoring import MonitoringSystem
    from src.core.config import ConfigurationManager
    from src.core.integration import IntegrationSystem
    from src.core.recovery import RecoverySystem
    CORE_IMPORTS_AVAILABLE = True
except ImportError:
    CORE_IMPORTS_AVAILABLE = False
    logging.warning("Core system imports not available. Running in standalone mode.")


class DeploymentState(Enum):
    """Enum representing the possible states of a deployment."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PARTIALLY_DEPLOYED = "partially_deployed"
    VERIFIED = "verified"


class ComponentType(Enum):
    """Enum representing the types of components that can be deployed."""
    CORE = "core"
    STRATEGY = "strategy"
    DATA = "data"
    ANALYTICS = "analytics"
    ML = "ml"
    RISK = "risk"
    EXECUTION = "execution"
    API = "api"
    CLI = "cli"
    CUSTOM = "custom"


@dataclass
class VersionInfo:
    """Class for tracking version information of components."""
    major: int
    minor: int
    patch: int
    build: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __str__(self) -> str:
        """String representation of version."""
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.build:
            return f"{base}-{self.build}"
        return base
    
    @classmethod
    def from_string(cls, version_str: str) -> "VersionInfo":
        """Create VersionInfo from string representation."""
        if "-" in version_str:
            base, build = version_str.split("-", 1)
        else:
            base, build = version_str, ""
            
        parts = base.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version string: {version_str}")
            
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            build=build
        )
    
    def increment_patch(self) -> "VersionInfo":
        """Create a new version with incremented patch number."""
        return VersionInfo(self.major, self.minor, self.patch + 1)
    
    def increment_minor(self) -> "VersionInfo":
        """Create a new version with incremented minor number."""
        return VersionInfo(self.major, self.minor + 1, 0)
    
    def increment_major(self) -> "VersionInfo":
        """Create a new version with incremented major number."""
        return VersionInfo(self.major + 1, 0, 0)


@dataclass
class ComponentMetadata:
    """Metadata for a deployable component."""
    name: str
    type: ComponentType
    version: VersionInfo
    dependencies: List[str] = field(default_factory=list)
    config_requirements: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    author: str = ""
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


@dataclass
class DeploymentRecord:
    """Record of a deployment operation."""
    id: str
    component_name: str
    version_from: Optional[VersionInfo]
    version_to: VersionInfo
    state: DeploymentState
    timestamp_start: datetime
    timestamp_end: Optional[datetime] = None
    executed_by: str = "system"
    logs: List[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    verification_results: Dict[str, Any] = field(default_factory=dict)
    is_rollback: bool = False
    rollback_from: Optional[str] = None
    
    def add_log(self, message: str) -> None:
        """Add a log message to the deployment record."""
        timestamp = datetime.now().isoformat()
        self.logs.append(f"[{timestamp}] {message}")
    
    def complete(self, state: DeploymentState) -> None:
        """Mark the deployment as complete with the given state."""
        self.state = state
        self.timestamp_end = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary for serialization."""
        return {
            "id": self.id,
            "component_name": self.component_name,
            "version_from": str(self.version_from) if self.version_from else None,
            "version_to": str(self.version_to),
            "state": self.state.value,
            "timestamp_start": self.timestamp_start.isoformat(),
            "timestamp_end": self.timestamp_end.isoformat() if self.timestamp_end else None,
            "executed_by": self.executed_by,
            "logs": self.logs,
            "artifacts": self.artifacts,
            "verification_results": self.verification_results,
            "is_rollback": self.is_rollback,
            "rollback_from": self.rollback_from
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeploymentRecord":
        """Create a DeploymentRecord from a dictionary."""
        return cls(
            id=data["id"],
            component_name=data["component_name"],
            version_from=VersionInfo.from_string(data["version_from"]) if data.get("version_from") else None,
            version_to=VersionInfo.from_string(data["version_to"]),
            state=DeploymentState(data["state"]),
            timestamp_start=datetime.fromisoformat(data["timestamp_start"]),
            timestamp_end=datetime.fromisoformat(data["timestamp_end"]) if data.get("timestamp_end") else None,
            executed_by=data.get("executed_by", "system"),
            logs=data.get("logs", []),
            artifacts=data.get("artifacts", {}),
            verification_results=data.get("verification_results", {}),
            is_rollback=data.get("is_rollback", False),
            rollback_from=data.get("rollback_from")
        )


class VerificationError(Exception):
    """Exception raised when a deployment verification fails."""
    pass


class DeploymentError(Exception):
    """Exception raised when a deployment operation fails."""
    pass


class RollbackError(Exception):
    """Exception raised when a rollback operation fails."""
    pass


class DeploymentSystem:
    """
    Main deployment system that manages component deployments,
    version control, rollbacks, and verification.
    """
    
    def __init__(
        self,
        deployment_dir: Union[str, Path] = "./deployments",
        backup_dir: Union[str, Path] = "./backups",
        orchestrator: Optional[Any] = None,
        monitoring_system: Optional[Any] = None,
        config_manager: Optional[Any] = None,
        integration_system: Optional[Any] = None
    ):
        """
        Initialize the deployment system.
        
        Args:
            deployment_dir: Directory to store deployment records
            backup_dir: Directory to store backups for rollbacks
            orchestrator: Optional orchestrator integration
            monitoring_system: Optional monitoring system integration
            config_manager: Optional configuration manager integration
            integration_system: Optional integration system
        """
        self.deployment_dir = Path(deployment_dir)
        self.backup_dir = Path(backup_dir)
        self.orchestrator = orchestrator
        self.monitoring_system = monitoring_system
        self.config_manager = config_manager
        self.integration_system = integration_system
        
        # Create necessary directories
        self.deployment_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize records storage
        self.records_dir = self.deployment_dir / "records"
        self.records_dir.mkdir(exist_ok=True)
        
        # Initialize components metadata storage
        self.components_dir = self.deployment_dir / "components"
        self.components_dir.mkdir(exist_ok=True)
        
        # Load deployment history
        self._deployment_history: Dict[str, List[DeploymentRecord]] = self._load_deployment_history()
        
        # Cache of component metadata
        self._component_metadata: Dict[str, ComponentMetadata] = self._load_component_metadata()
        
        # Register with orchestrator if available
        if CORE_IMPORTS_AVAILABLE and self.orchestrator is not None:
            self.orchestrator.register_system("deployment", self)
    
    def _load_deployment_history(self) -> Dict[str, List[DeploymentRecord]]:
        """Load deployment history from disk."""
        history: Dict[str, List[DeploymentRecord]] = {}
        
        for record_file in self.records_dir.glob("*.json"):
            try:
                with open(record_file, "r") as f:
                    record_data = json.load(f)
                    record = DeploymentRecord.from_dict(record_data)
                    
                    if record.component_name not in history:
                        history[record.component_name] = []
                    
                    history[record.component_name].append(record)
            except Exception as e:
                logging.error(f"Error loading deployment record {record_file}: {e}")
        
        # Sort records by timestamp
        for component, records in history.items():
            history[component] = sorted(records, key=lambda r: r.timestamp_start)
        
        return history
    
    def _load_component_metadata(self) -> Dict[str, ComponentMetadata]:
        """Load component metadata from disk."""
        metadata: Dict[str, ComponentMetadata] = {}
        
        for metadata_file in self.components_dir.glob("*.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                    component_name = metadata_file.stem
                    metadata[component_name] = ComponentMetadata(
                        name=component_name,
                        type=ComponentType(data["type"]),
                        version=VersionInfo.from_string(data["version"]),
                        dependencies=data.get("dependencies", []),
                        config_requirements=data.get("config_requirements", {}),
                        resource_requirements=data.get("resource_requirements", {}),
                        author=data.get("author", ""),
                        description=data.get("description", ""),
                        created_at=datetime.fromisoformat(data["created_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                        tags=data.get("tags", [])
                    )
            except Exception as e:
                logging.error(f"Error loading component metadata {metadata_file}: {e}")
        
        return metadata
    
    def _save_component_metadata(self, metadata: ComponentMetadata) -> None:
        """Save component metadata to disk."""
        metadata_file = self.components_dir / f"{metadata.name}.json"
        data = {
            "name": metadata.name,
            "type": metadata.type.value,
            "version": str(metadata.version),
            "dependencies": metadata.dependencies,
            "config_requirements": metadata.config_requirements,
            "resource_requirements": metadata.resource_requirements,
            "author": metadata.author,
            "description": metadata.description,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "tags": metadata.tags
        }
        
        with open(metadata_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Update cache
        self._component_metadata[metadata.name] = metadata
    
    def _save_deployment_record(self, record: DeploymentRecord) -> None:
        """Save deployment record to disk."""
        record_file = self.records_dir / f"{record.id}.json"
        with open(record_file, "w") as f:
            json.dump(record.to_dict(), f, indent=2)
        
        # Update history cache
        if record.component_name not in self._deployment_history:
            self._deployment_history[record.component_name] = []
        
        self._deployment_history[record.component_name].append(record)
    
    def _create_backup(self, component_name: str, version: VersionInfo) -> str:
        """
        Create a backup of a component for potential rollback.
        
        Args:
            component_name: Name of the component to backup
            version: Version of the component to backup
            
        Returns:
            Backup ID
        """
        backup_id = str(uuid.uuid4())
        component_path = Path(f"src/{component_name.replace('.', '/')}")
        backup_path = self.backup_dir / backup_id
        
        if not component_path.exists():
            raise DeploymentError(f"Component path does not exist: {component_path}")
        
        # Create backup directory
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy component files to backup
        backup_component_path = backup_path / component_name.replace(".", "/")
        backup_component_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copytree(component_path, backup_component_path)
            
            # Save version info
            with open(backup_path / "version.json", "w") as f:
                json.dump({
                    "component": component_name,
                    "version": str(version),
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
                
            return backup_id
        except Exception as e:
            # Clean up partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise DeploymentError(f"Failed to create backup: {str(e)}")
    
    def _restore_from_backup(self, backup_id: str) -> Tuple[str, VersionInfo]:
        """
        Restore a component from a backup.
        
        Args:
            backup_id: ID of the backup to restore from
            
        Returns:
            Tuple of (component_name, version)
        """
        backup_path = self.backup_dir / backup_id
        
        if not backup_path.exists():
            raise RollbackError(f"Backup not found: {backup_id}")
        
        # Load version info
        try:
            with open(backup_path

