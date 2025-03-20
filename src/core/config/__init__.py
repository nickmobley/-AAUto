"""
Configuration Management System

This module provides a comprehensive configuration management system for the trading platform,
featuring dynamic configurations, versioning, validation, change tracking, and component-specific
configurations.

Features:
- Dynamic configuration updates at runtime
- Configuration versioning and history
- Schema-based configuration validation
- Change tracking and audit log
- Component-specific configuration inheritance
- Integration with the coordination system
"""

import asyncio
import copy
import datetime
import json
import logging
import os
import pathlib
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast, Generic

# Type definitions
ConfigValue = Union[str, int, float, bool, Dict[str, Any], List[Any], None]
ConfigDict = Dict[str, ConfigValue]
ValidationResult = Tuple[bool, List[str]]
ConfigChangeHandler = Callable[[str, ConfigValue, ConfigValue], None]
T = TypeVar('T')

class ConfigChangeType(Enum):
    """Enumeration of configuration change types."""
    CREATED = "created"
    UPDATED = "updated"
    DELETED = "deleted"
    RESET = "reset"


@dataclass
class ConfigChange:
    """Represents a single configuration change."""
    timestamp: datetime.datetime
    path: str
    change_type: ConfigChangeType
    old_value: Optional[ConfigValue] = None
    new_value: Optional[ConfigValue] = None
    user: Optional[str] = None
    component: Optional[str] = None
    description: Optional[str] = None
    change_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConfigVersion:
    """Represents a specific version of the configuration."""
    version_id: str
    timestamp: datetime.datetime
    config_data: ConfigDict
    description: Optional[str] = None
    user: Optional[str] = None
    changes: List[ConfigChange] = field(default_factory=list)


class ConfigSchema:
    """Schema definition and validation for configuration objects."""
    
    def __init__(self, schema_definition: Dict[str, Any]):
        """Initialize with schema definition."""
        self.schema = schema_definition
        
    def validate(self, config: ConfigDict) -> ValidationResult:
        """Validate a configuration against the schema.
        
        Returns:
            Tuple containing (is_valid, error_messages)
        """
        is_valid = True
        errors: List[str] = []
        
        for key, schema_def in self.schema.items():
            # Check required fields
            if schema_def.get("required", False) and key not in config:
                is_valid = False
                errors.append(f"Missing required field: {key}")
                continue
                
            if key not in config:
                continue
                
            value = config[key]
            
            # Type validation
            if "type" in schema_def:
                expected_type = schema_def["type"]
                
                if expected_type == "string" and not isinstance(value, str):
                    is_valid = False
                    errors.append(f"Field {key} must be a string")
                    
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    is_valid = False
                    errors.append(f"Field {key} must be a number")
                    
                elif expected_type == "boolean" and not isinstance(value, bool):
                    is_valid = False
                    errors.append(f"Field {key} must be a boolean")
                    
                elif expected_type == "array" and not isinstance(value, list):
                    is_valid = False
                    errors.append(f"Field {key} must be an array")
                    
                elif expected_type == "object" and not isinstance(value, dict):
                    is_valid = False
                    errors.append(f"Field {key} must be an object")
            
            # Range validation
            if isinstance(value, (int, float)):
                if "minimum" in schema_def and value < schema_def["minimum"]:
                    is_valid = False
                    errors.append(f"Field {key} must be ≥ {schema_def['minimum']}")
                    
                if "maximum" in schema_def and value > schema_def["maximum"]:
                    is_valid = False
                    errors.append(f"Field {key} must be ≤ {schema_def['maximum']}")
            
            # String pattern validation
            if isinstance(value, str) and "pattern" in schema_def:
                pattern = re.compile(schema_def["pattern"])
                if not pattern.match(value):
                    is_valid = False
                    errors.append(f"Field {key} must match pattern {schema_def['pattern']}")
            
            # Enum validation
            if "enum" in schema_def and value not in schema_def["enum"]:
                is_valid = False
                errors.append(f"Field {key} must be one of {schema_def['enum']}")
                
            # Custom validation
            if "validate" in schema_def and callable(schema_def["validate"]):
                custom_valid, custom_errors = schema_def["validate"](value)
                if not custom_valid:
                    is_valid = False
                    errors.extend(custom_errors)
        
        return is_valid, errors


class ComponentConfig:
    """Configuration specific to a component with inheritance capabilities."""
    
    def __init__(
        self, 
        component_name: str, 
        config_manager: 'ConfigurationManager',
        schema: Optional[ConfigSchema] = None
    ):
        """Initialize component configuration.
        
        Args:
            component_name: Unique name of the component
            config_manager: Reference to the parent configuration manager
            schema: Optional schema for validating component configuration
        """
        self.component_name = component_name
        self.config_manager = config_manager
        self.schema = schema
        self._local_config: ConfigDict = {}
        self._change_handlers: List[ConfigChangeHandler] = []
        
    def get(self, key: str, default: Optional[T] = None) -> Union[ConfigValue, T]:
        """Get configuration value with inheritance from global config."""
        # First check local config
        if key in self._local_config:
            return self._local_config[key]
            
        # Then check global config with component prefix
        prefixed_key = f"{self.component_name}.{key}"
        if self.config_manager.has(prefixed_key):
            return self.config_manager.get(prefixed_key)
            
        # Finally check global config for common settings
        if self.config_manager.has(key):
            return self.config_manager.get(key)
            
        return default
        
    def set(self, key: str, value: ConfigValue, description: Optional[str] = None) -> None:
        """Set component-specific configuration value."""
        old_value = self.get(key)
        
        # Perform validation if schema is available
        if self.schema:
            test_config = copy.deepcopy(self._local_config)
            test_config[key] = value
            is_valid, errors = self.schema.validate(test_config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {errors}")
        
        self._local_config[key] = value
        
        # Create change record
        change = ConfigChange(
            timestamp=datetime.datetime.now(),
            path=f"{self.component_name}.{key}",
            change_type=ConfigChangeType.UPDATED if old_value is not None else ConfigChangeType.CREATED,
            old_value=old_value,
            new_value=value,
            component=self.component_name,
            description=description
        )
        
        # Notify config manager of the change
        self.config_manager._record_change(change)
        
        # Notify local change handlers
        for handler in self._change_handlers:
            handler(key, old_value, value)
            
    def update(self, config_dict: ConfigDict, description: Optional[str] = None) -> None:
        """Update multiple configuration values at once."""
        # Validate entire configuration if schema exists
        if self.schema:
            test_config = copy.deepcopy(self._local_config)
            test_config.update(config_dict)
            is_valid, errors = self.schema.validate(test_config)
            if not is_valid:
                raise ValueError(f"Invalid configuration: {errors}")
        
        # Apply updates
        for key, value in config_dict.items():
            self.set(key, value, description)
    
    def register_change_handler(self, handler: ConfigChangeHandler) -> None:
        """Register a handler to be called when configuration changes."""
        self._change_handlers.append(handler)
        
    def unregister_change_handler(self, handler: ConfigChangeHandler) -> None:
        """Unregister a previously registered change handler."""
        if handler in self._change_handlers:
            self._change_handlers.remove(handler)


class ConfigurationManager:
    """Central configuration management system with versioning and change tracking."""
    
    def __init__(self, config_dir: Optional[Union[str, pathlib.Path]] = None):
        """Initialize the configuration manager.
        
        Args:
            config_dir: Optional directory to store configuration files
        """
        self._config: ConfigDict = {}
        self._components: Dict[str, ComponentConfig] = {}
        self._version_history: List[ConfigVersion] = []
        self._change_history: List[ConfigChange] = []
        self._global_schema: Optional[ConfigSchema] = None
        self._change_handlers: Dict[str, List[ConfigChangeHandler]] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Storage location
        if config_dir:
            self.config_dir = pathlib.Path(config_dir)
            self.config_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.config_dir = None
    
    def set_schema(self, schema: ConfigSchema) -> None:
        """Set the global schema for configuration validation."""
        self._global_schema = schema
    
    async def load(self, config_file: Optional[Union[str, pathlib.Path]] = None) -> None:
        """Load configuration from file."""
        if not config_file and not self.config_dir:
            return
        
        if not config_file and self.config_dir:
            config_file = self.config_dir / "config.json"
            
        config_path = pathlib.Path(config_file)
        if not config_path.exists():
            return
            
        try:
            async with self._lock:
                config_data = json.loads(config_path.read_text())
                
                # Validate if schema exists
                if self._global_schema:
                    is_valid, errors = self._global_schema.validate(config_data)
                    if not is_valid:
                        logging.error(f"Invalid configuration file: {errors}")
                        return
                
                self._config = config_data
                
                # Create initial version
                version = ConfigVersion(
                    version_id=str(uuid.uuid4()),
                    timestamp=datetime.datetime.now(),
                    config_data=copy.deepcopy(self._config),
                    description="Initial load from file"
                )
                self._version_history.append(version)
                
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
    
    async def save(self, config_file: Optional[Union[str, pathlib.Path]] = None) -> None:
        """Save configuration to file."""
        if not config_file and not self.config_dir:
            return
        
        if not config_file and self.config_dir:
            config_file = self.config_dir / "config.json"
            
        config_path = pathlib.Path(config_file)
        
        try:
            async with self._lock:
                config_path.write_text(json.dumps(self._config, indent=2))
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
    
    def get(self, key: str, default: Optional[T] = None) -> Union[ConfigValue, T]:
        """Get configuration value by key."""
        keys = key.split(".")
        value: Any = self._config
        
        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def has(self, key: str) -> bool:
        """Check if a configuration key exists."""
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return False
        return True
    
    async def set(
        self, 
        key: str, 
        value: ConfigValue, 
        description: Optional[str] = None,
        user: Optional[str] = None
    ) -> None:
        """Set configuration value."""
        keys = key.split(".")
        
        async with self._lock:
            # Validate with schema if available
            if self._global_schema:
                # Create a copy of the config to test the change
                test_config = copy.deepcopy(self._config)
                
                # Apply the change to the test config
                current = test_config
                for i, k in enumerate(keys[:-1]):
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                
                # Validate the test config
                is_valid, errors = self._global_schema.validate(test_config)
                if not is_valid:
                    raise ValueError(f"Invalid configuration: {errors}")
            
            # Get the old value for change tracking
            old_value = self.get(key)
            
            # Apply the change
            current = self._config
            for i, k in enumerate(keys[:-1]):
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # Determine change type
            change_type = ConfigChangeType.CREATED
            if keys[-1] in current:
                change_type = ConfigChangeType.UPDATED if current[keys[-1]] != value else None
            
            # Only proceed if there's an actual change
            if change_type:
                current[keys[-1]] = value
                
                # Record the change
                change = ConfigChange(
                    timestamp=datetime.datetime.now(),
                    path=key,
                    change_type=change_type,
                    old_value=old_value,
                    new_value=value,
                    user=user,
                    description=description
                )
                
                self._record_change(change)
                
                # Notify handlers
                await self._notify_handlers(key, old_value, value)
    
    async def update(
        self, 
        config_dict: ConfigDict, 
        description: Optional[str] = None,
        user: Optional[str] = None
    ) -> None:
        """Update multiple configuration values at once."""
        # First validate the entire update if schema exists
        if self._global_schema:
            # Create a copy of the config to test the changes
            test_config = copy.deepcopy(self._config)
            
            # Apply all changes to the test config
            for key, value in _flatten_dict(config_dict).items():
                keys = key.split(".")
                current = test_config
                for i, k in enumerate(keys[:-1]):
                    if k not in current

"""
Adaptive Configuration Manager for the trading system.

This module provides a sophisticated configuration management system with:
- Version control for configurations
- Effectiveness tracking
- Automatic parameter optimization
- Integration with performance analytics
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union, cast

import numpy as np
from pydantic import BaseModel, Field, validator

# Type definitions
ConfigKey = str
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]
ConfigDict = Dict[ConfigKey, ConfigValue]
T = TypeVar('T')

logger = logging.getLogger(__name__)


class ConfigVersion(BaseModel):
    """Version information for a configuration."""
    
    version_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    description: str = ""
    parent_version: Optional[str] = None
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComponentConfig(BaseModel):
    """Configuration for a specific component."""
    
    component_id: str
    current_version: str
    config_data: ConfigDict = Field(default_factory=dict)
    versions: Dict[str, ConfigVersion] = Field(default_factory=dict)
    
    def create_version(self, description: str = "", config_data: Optional[ConfigDict] = None) -> str:
        """Create a new version of this configuration."""
        version = ConfigVersion(
            description=description,
            parent_version=self.current_version
        )
        
        self.versions[version.version_id] = version
        
        if config_data:
            self.config_data = config_data
            
        self.current_version = version.version_id
        return version.version_id


class OptimizationResult(BaseModel):
    """Result of a parameter optimization."""
    
    parameter: str
    original_value: ConfigValue
    optimized_value: ConfigValue
    improvement: float
    confidence: float


class ParameterConstraint(BaseModel):
    """Constraints for parameter optimization."""
    
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


class AdaptiveConfigManager:
    """
    Adaptive Configuration Manager that handles component configurations
    with version control, effectiveness tracking, and automatic optimization.
    """
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True, parents=True)
        
        self.components: Dict[str, ComponentConfig] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load all existing component configurations."""
        if not self.config_dir.exists():
            return
            
        for file_path in self.config_dir.glob("*.json"):
            try:
                component_id = file_path.stem
                with open(file_path, "r") as f:
                    config_data = json.load(f)
                
                component_config = ComponentConfig(**config_data)
                self.components[component_id] = component_config
                self._locks[component_id] = asyncio.Lock()
                
                logger.info(f"Loaded configuration for component {component_id}")
            except Exception as e:
                logger.error(f"Error loading configuration {file_path}: {e}")
    
    async def _save_config(self, component_id: str) -> None:
        """Save a component configuration to disk."""
        if component_id not in self.components:
            return
            
        config_path = self.config_dir / f"{component_id}.json"
        
        async with self._get_lock(component_id):
            component_config = self.components[component_id]
            
            with open(config_path, "w") as f:
                json.dump(component_config.dict(), f, indent=2)
    
    def _get_lock(self, component_id: str) -> asyncio.Lock:
        """Get or create a lock for a component."""
        if component_id not in self._locks:
            self._locks[component_id] = asyncio.Lock()
        return self._locks[component_id]
    
    async def register_component(
        self, 
        component_id: str, 
        initial_config: ConfigDict = None
    ) -> ComponentConfig:
        """
        Register a new component or retrieve an existing one.
        """
        async with self._get_lock(component_id):
            if component_id in self.components:
                return self.components[component_id]
            
            # Create new component configuration
            initial_config = initial_config or {}
            
            version = ConfigVersion()
            component_config = ComponentConfig(
                component_id=component_id,
                current_version=version.version_id,
                config_data=initial_config,
                versions={version.version_id: version}
            )
            
            self.components[component_id] = component_config
            await self._save_config(component_id)
            
            logger.info(f"Registered new component: {component_id}")
            return component_config
    
    async def get_config(self, component_id: str) -> ConfigDict:
        """
        Get the current configuration for a component.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            return self.components[component_id].config_data.copy()
    
    async def get_parameter(
        self, 
        component_id: str, 
        param_key: str, 
        default: T = None
    ) -> Union[ConfigValue, T]:
        """
        Get a specific parameter from a component's configuration.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                if default is not None:
                    return default
                raise ValueError(f"Component {component_id} not registered")
            
            config = self.components[component_id].config_data
            return config.get(param_key, default)
    
    async def update_config(
        self, 
        component_id: str, 
        config_data: ConfigDict, 
        description: str = ""
    ) -> str:
        """
        Update a component's configuration and create a new version.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            component = self.components[component_id]
            version_id = component.create_version(description, config_data)
            
            await self._save_config(component_id)
            logger.info(f"Updated configuration for {component_id}, version: {version_id}")
            
            return version_id
    
    async def update_parameter(
        self, 
        component_id: str, 
        param_key: str, 
        value: ConfigValue, 
        description: str = ""
    ) -> str:
        """
        Update a specific parameter in a component's configuration.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            component = self.components[component_id]
            new_config = component.config_data.copy()
            new_config[param_key] = value
            
            version_id = component.create_version(
                description=f"Updated {param_key}: {description}",
                config_data=new_config
            )
            
            await self._save_config(component_id)
            logger.info(f"Updated parameter {param_key} for {component_id}, version: {version_id}")
            
            return version_id
    
    async def record_performance(
        self, 
        component_id: str, 
        metrics: Dict[str, float], 
        version_id: Optional[str] = None
    ) -> None:
        """
        Record performance metrics for a specific configuration version.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            component = self.components[component_id]
            version_id = version_id or component.current_version
            
            if version_id not in component.versions:
                raise ValueError(f"Version {version_id} not found for component {component_id}")
            
            # Update performance metrics
            version = component.versions[version_id]
            for metric, value in metrics.items():
                version.performance_metrics[metric] = value
            
            await self._save_config(component_id)
            logger.info(f"Recorded performance for {component_id}, version: {version_id}")
    
    async def get_version_history(
        self, 
        component_id: str, 
        with_performance: bool = False
    ) -> List[ConfigVersion]:
        """
        Get the version history for a component.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            component = self.components[component_id]
            versions = list(component.versions.values())
            
            # Sort versions by timestamp
            versions.sort(key=lambda v: v.timestamp)
            
            if not with_performance:
                # Filter out versions with no performance data
                versions = [v for v in versions if v.performance_metrics]
            
            return versions
    
    async def rollback(self, component_id: str, version_id: str) -> None:
        """
        Rollback configuration to a previous version.
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            component = self.components[component_id]
            
            if version_id not in component.versions:
                raise ValueError(f"Version {version_id} not found for component {component_id}")
            
            # Use the data from the older version but create a new version for tracking
            old_version = component.versions[version_id]
            
            new_version = ConfigVersion(
                description=f"Rollback to version {version_id}",
                parent_version=component.current_version
            )
            
            component.versions[new_version.version_id] = new_version
            component.current_version = new_version.version_id
            
            await self._save_config(component_id)
            logger.info(f"Rolled back {component_id} to version {version_id}")
    
    async def optimize_parameter(
        self,
        component_id: str,
        param_key: str,
        metric_key: str,
        constraints: Optional[ParameterConstraint] = None,
        test_values: Optional[List[Any]] = None,
        minimize: bool = False,
        iterations: int = 10,
        test_function: Optional[callable] = None
    ) -> OptimizationResult:
        """
        Optimize a parameter based on performance metrics.
        
        Args:
            component_id: The component to optimize
            param_key: The parameter to optimize
            metric_key: The performance metric to optimize
            constraints: Optional constraints for the parameter
            test_values: Optional list of values to test
            minimize: Whether to minimize (True) or maximize (False) the metric
            iterations: Number of iterations for optimization
            test_function: Optional function to test parameter values
                           signature: async def test(value) -> float
        
        Returns:
            OptimizationResult with the best parameter value and improvement
        """
        async with self._get_lock(component_id):
            if component_id not in self.components:
                raise ValueError(f"Component {component_id} not registered")
            
            component = self.components[component_id]
            current_value = component.config_data.get(param_key)
            
            if current_value is None:
                raise ValueError(f"Parameter {param_key} not found in component {component_id}")
            
            # Get the current performance as baseline
            versions = await self.get_version_history(component_id, with_performance=True)
            current_version = next((v for v in versions if v.version_id == component.current_version), None)
            
            if not current_version or metric_key not in current_version.performance_metrics:
                # No baseline performance, create one if we have a test function
                if not test_function:
                    raise ValueError(f"No baseline performance for {metric_key} and no test function provided")
                
                baseline_performance = await test_function(current_value)
            else:
                baseline_performance = current_version.performance_metrics[metric_key]
            
            # Generate test values based on constraints or defaults
            if test_values:
                parameter_values = test_values
            else:
                parameter_values = self._generate_test_values(
                    current_value, constraints, iterations
                )
            
            # Test each value and record performance
            results = []
            
            for value in parameter_values:
                if value == current_value:
                    # Skip testing the current value again
                    performance = baseline_performance
                elif test_function:
                    # Use the provided test function
                    performance = await test_function(value)
                else:
                    # Use historical data if available
                    version_with_value = self._find_version_with_param_value(
                        component, param_key, value
                    )
                    
                    if not version_with_value or metric_key not in version_with_value.performance_metrics:
                        logger.warning(f"No performance data for {param_key}={value}, skipping")
                        continue
                    
                    performance = version_with_value.performance_metrics[metric_key]
                
                # Record result (higher is better unless minimize=True)
                score = -performance if minimize else performance
                results.append((value, score, performance))
            
            if not results:
                logger.warning("No valid test results obtained for optimization")
                return OptimizationResult(
                    parameter=param_key,
                    original_value=current_value,
                    optimized_value=current_value,
                    improvement=0.0,
                    confidence=0.0
                )
            
            # Find the best value
            best_value, best_score, best_performance = max(results, key=lambda x: x[1])
            
            # Calculate improvement
            baseline_score = -baseline_performance if minimize else baseline_performance
            improvement = best_score - baseline_score
            
            # Calculate confidence (more results = higher confidence)
            confidence = min(0.95, 0.5 + (0.05 * len(results)))
            
            # If there's meaningful improvement, update the parameter
            if improvement > 0:
                await self.update_parameter(
                    component_id,
                    param_key,
                    best_value,
                    f"Optimized for {metric_key}: {best_performance:.4f} vs {baseline_performance:.4f}"
                )
            
            return OptimizationResult(
                parameter=param_key,
                original_value=current_value

