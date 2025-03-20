import json
import os
import logging
from collections import Counter, defaultdict
from typing import Dict, Any, Optional, Union

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    A class for collecting, storing, and retrieving metrics.
    This class handles proper serialization of complex data types like Counter and defaultdict.
    """
    
    def __init__(self, metrics_dir: str = "metrics"):
        """
        Initialize a new MetricsCollector.
        
        Args:
            metrics_dir: The directory where metrics will be stored.
        """
        self.metrics_dir = metrics_dir
        self.metrics: Dict[str, Any] = {}
        os.makedirs(metrics_dir, exist_ok=True)
    
    def add_metric(self, name: str, value: Any) -> None:
        """
        Add a metric to the collector.
        
        Args:
            name: The name of the metric.
            value: The value of the metric.
        """
        self.metrics[name] = value
        
    def get_metric(self, name: str) -> Optional[Any]:
        """
        Get a metric by name.
        
        Args:
            name: The name of the metric to retrieve.
            
        Returns:
            The value of the metric, or None if the metric does not exist.
        """
        return self.metrics.get(name)
    
    def update_counter(self, name: str, key: Any, increment: int = 1) -> None:
        """
        Update a counter metric.
        
        Args:
            name: The name of the counter metric.
            key: The key within the counter to update.
            increment: The amount to increment the counter by.
        """
        if name not in self.metrics:
            self.metrics[name] = Counter()
        
        self.metrics[name][key] += increment
        
    def update_defaultdict(self, name: str, key: Any, value: Any) -> None:
        """
        Update a defaultdict metric.
        
        Args:
            name: The name of the defaultdict metric.
            key: The key within the defaultdict to update.
            value: The value to add to the list at the specified key.
        """
        if name not in self.metrics:
            self.metrics[name] = defaultdict(list)
            
        self.metrics[name][key].append(value)
    
    def _serialize_value(self, value: Any) -> Dict[str, Any]:
        """
        Serialize a value for storage.
        
        Args:
            value: The value to serialize.
            
        Returns:
            A serialized representation of the value.
        """
        if isinstance(value, Counter):
            return {
                "type": "Counter",
                "data": dict(value)
            }
        elif isinstance(value, defaultdict):
            return {
                "type": "defaultdict",
                "data": dict(value),
                "default_factory": type(value.default_factory()).__name__
            }
        elif isinstance(value, (dict, list, int, float, str, bool, type(None))):
            return {
                "type": type(value).__name__,
                "data": value
            }
        else:
            logger.warning(f"Cannot serialize {type(value).__name__}. Converting to string.")
            return {
                "type": "str",
                "data": str(value)
            }
    
    def _deserialize_value(self, serialized: Dict[str, Any]) -> Any:
        """
        Deserialize a value from storage.
        
        Args:
            serialized: The serialized representation of the value.
            
        Returns:
            The deserialized value.
        """
        value_type = serialized.get("type")
        data = serialized.get("data")
        
        if value_type == "Counter":
            return Counter(data)
        elif value_type == "defaultdict":
            if serialized.get("default_factory") == "list":
                result = defaultdict(list)
                for k, v in data.items():
                    result[k] = v
                return result
            else:
                # Default to list if default_factory is unknown
                logger.warning(f"Unknown default_factory: {serialized.get('default_factory')}. Using list.")
                result = defaultdict(list)
                for k, v in data.items():
                    result[k] = v
                return result
        elif value_type == "dict":
            return data
        elif value_type == "list":
            return data
        elif value_type in ("int", "float", "str", "bool", "NoneType"):
            return data
        else:
            logger.warning(f"Unknown type: {value_type}. Returning raw data.")
            return data
    
    def save_metrics(self, filename: str) -> None:
        """
        Save metrics to a file.
        
        Args:
            filename: The name of the file to save the metrics to.
        """
        filepath = os.path.join(self.metrics_dir, filename)
        
        # Serialize all metrics
        serialized_metrics = {}
        for key, value in self.metrics.items():
            serialized_metrics[key] = self._serialize_value(value)
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(serialized_metrics, f, indent=2)
            logger.info(f"Metrics saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving metrics to {filepath}: {e}")
    
    def load_metrics(self, filename: str) -> None:
        """
        Load metrics from a file.
        
        Args:
            filename: The name of the file to load the metrics from.
        """
        filepath = os.path.join(self.metrics_dir, filename)
        
        try:
            with open(filepath, 'r') as f:
                serialized_metrics = json.load(f)
            
            # Deserialize all metrics
            self.metrics = {}
            for key, value in serialized_metrics.items():
                self.metrics[key] = self._deserialize_value(value)
                
            logger.info(f"Metrics loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Metrics file {filepath} not found.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {filepath}.")
        except Exception as e:
            logger.error(f"Error loading metrics from {filepath}: {e}")

