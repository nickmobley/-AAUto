"""
Performance Analysis Module

This module provides comprehensive performance monitoring, analysis, and optimization
capabilities for the trading system. It integrates with other system components to
track performance metrics, identify bottlenecks, profile execution, and provide
optimization recommendations.
"""

import asyncio
import time
import cProfile
import pstats
import io
import logging
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Tuple, Set, Union, TypeVar, Generic
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from collections import defaultdict, deque
import inspect
import sys
import os
import json
import threading
from contextlib import contextmanager

# Import integration modules
try:
    from ..core.monitoring import MonitoringSystem
    from ..analytics import AnalyticsEngine
    from ..verification import VerificationSystem
    from ..core.logging import get_logger
    HAS_INTEGRATIONS = True
except ImportError:
    HAS_INTEGRATIONS = False
    get_logger = lambda name: logging.getLogger(name)

logger = get_logger("performance")

class PerformanceMetric:
    """Base class for performance metrics"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.last_updated = self.created_at
        self._metadata: Dict[str, Any] = {}
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata for the metric"""
        self._metadata[key] = value
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get all metadata"""
        return self._metadata.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "metadata": self._metadata
        }


class TimingMetric(PerformanceMetric):
    """Metric for timing operations"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.total_time: float = 0.0
        self.call_count: int = 0
        self.min_time: float = float('inf')
        self.max_time: float = 0.0
        self.times: deque = deque(maxlen=1000)  # Store last 1000 timings
    
    def record(self, execution_time: float) -> None:
        """Record a new execution time"""
        self.total_time += execution_time
        self.call_count += 1
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.times.append(execution_time)
        self.last_updated = datetime.now()
    
    def avg_time(self) -> float:
        """Calculate average execution time"""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0
    
    def recent_avg_time(self, n: int = 100) -> float:
        """Calculate average execution time for the most recent n calls"""
        recent = list(self.times)[-n:] if n < len(self.times) else list(self.times)
        return sum(recent) / len(recent) if recent else 0.0
    
    def percentile(self, p: float) -> float:
        """Calculate the p-th percentile of execution times"""
        if not self.times:
            return 0.0
        sorted_times = sorted(self.times)
        k = (len(sorted_times) - 1) * p
        f = int(k)
        c = math.ceil(k)
        if f == c:
            return sorted_times[f]
        return sorted_times[f] * (c - k) + sorted_times[c] * (k - f)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "total_time": self.total_time,
            "call_count": self.call_count,
            "min_time": self.min_time,
            "max_time": self.max_time,
            "avg_time": self.avg_time(),
            "recent_avg_time": self.recent_avg_time(),
            "p50": self.percentile(0.5),
            "p95": self.percentile(0.95),
            "p99": self.percentile(0.99)
        })
        return base_dict


class MemoryMetric(PerformanceMetric):
    """Metric for memory usage"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.allocations: Dict[str, int] = defaultdict(int)
        self.peak_memory: int = 0
        self.current_memory: int = 0
        self.snapshots: List[Dict[str, Any]] = []
    
    def record_allocation(self, size: int, location: str) -> None:
        """Record a memory allocation"""
        self.allocations[location] += size
        self.current_memory += size
        self.peak_memory = max(self.peak_memory, self.current_memory)
        self.last_updated = datetime.now()
    
    def record_deallocation(self, size: int, location: str) -> None:
        """Record a memory deallocation"""
        self.allocations[location] -= size
        self.current_memory -= size
        self.last_updated = datetime.now()
    
    def take_snapshot(self) -> None:
        """Take a memory snapshot"""
        self.snapshots.append({
            "timestamp": datetime.now().isoformat(),
            "allocations": dict(self.allocations),
            "current_memory": self.current_memory,
            "peak_memory": self.peak_memory
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "current_memory": self.current_memory,
            "peak_memory": self.peak_memory,
            "allocation_count": len(self.allocations),
            "top_allocations": dict(sorted(self.allocations.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:10])
        })
        return base_dict


class CPUMetric(PerformanceMetric):
    """Metric for CPU usage"""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.cpu_times: List[Dict[str, float]] = []
        self.total_user_time: float = 0.0
        self.total_system_time: float = 0.0
    
    def record(self, user_time: float, system_time: float) -> None:
        """Record CPU times"""
        self.cpu_times.append({
            "timestamp": datetime.now().isoformat(),
            "user_time": user_time,
            "system_time": system_time
        })
        self.total_user_time += user_time
        self.total_system_time += system_time
        self.last_updated = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary"""
        base_dict = super().to_dict()
        base_dict.update({
            "total_user_time": self.total_user_time,
            "total_system_time": self.total_system_time,
            "total_samples": len(self.cpu_times),
            "recent_samples": self.cpu_times[-10:] if self.cpu_times else []
        })
        return base_dict


class PerformanceMetricsCollector:
    """Collects and manages performance metrics"""
    
    def __init__(self):
        self._metrics: Dict[str, PerformanceMetric] = {}
        self._tag_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        
    def get_metric(self, name: str) -> Optional[PerformanceMetric]:
        """Get a metric by name"""
        with self._lock:
            return self._metrics.get(name)
    
    def add_metric(self, metric: PerformanceMetric, tags: List[str] = None) -> None:
        """Add a new metric"""
        with self._lock:
            self._metrics[metric.name] = metric
            if tags:
                for tag in tags:
                    self._tag_index[tag].add(metric.name)
    
    def get_metrics_by_tag(self, tag: str) -> List[PerformanceMetric]:
        """Get all metrics with a specific tag"""
        with self._lock:
            return [self._metrics[name] for name in self._tag_index.get(tag, set()) 
                   if name in self._metrics]
    
    def get_all_metrics(self) -> Dict[str, PerformanceMetric]:
        """Get all metrics"""
        with self._lock:
            return self._metrics.copy()
    
    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert all metrics to dictionary"""
        with self._lock:
            return {name: metric.to_dict() for name, metric in self._metrics.items()}


class BottleneckAnalyzer:
    """Analyzes system bottlenecks based on performance metrics"""
    
    def __init__(self, metrics_collector: PerformanceMetricsCollector):
        self.metrics_collector = metrics_collector
        self.bottleneck_thresholds = {
            "timing": {
                "critical": 1.0,  # 1 second
                "warning": 0.1    # 100 ms
            },
            "memory": {
                "critical": 1024 * 1024 * 100,  # 100 MB
                "warning": 1024 * 1024 * 10     # 10 MB
            },
            "cpu": {
                "critical": 0.9,  # 90% utilization
                "warning": 0.7    # 70% utilization
            }
        }
        self.bottlenecks: List[Dict[str, Any]] = []
    
    def analyze_timing_metrics(self) -> List[Dict[str, Any]]:
        """Analyze timing metrics for bottlenecks"""
        bottlenecks = []
        timing_metrics = [m for m in self.metrics_collector.get_all_metrics().values() 
                         if isinstance(m, TimingMetric)]
        
        for metric in timing_metrics:
            avg_time = metric.avg_time()
            recent_avg = metric.recent_avg_time()
            p95 = metric.percentile(0.95)
            
            if p95 > self.bottleneck_thresholds["timing"]["critical"]:
                severity = "critical"
            elif p95 > self.bottleneck_thresholds["timing"]["warning"]:
                severity = "warning"
            else:
                continue  # No bottleneck
                
            bottlenecks.append({
                "type": "timing",
                "metric": metric.name,
                "severity": severity,
                "value": p95,
                "avg_value": avg_time,
                "recent_avg": recent_avg,
                "threshold": self.bottleneck_thresholds["timing"][severity],
                "timestamp": datetime.now().isoformat(),
                "recommendation": f"Optimize the {metric.name} operation which is taking {p95:.2f}s at p95"
            })
            
        return bottlenecks
    
    def analyze_memory_metrics(self) -> List[Dict[str, Any]]:
        """Analyze memory metrics for bottlenecks"""
        bottlenecks = []
        memory_metrics = [m for m in self.metrics_collector.get_all_metrics().values() 
                         if isinstance(m, MemoryMetric)]
        
        for metric in memory_metrics:
            if metric.peak_memory > self.bottleneck_thresholds["memory"]["critical"]:
                severity = "critical"
            elif metric.peak_memory > self.bottleneck_thresholds["memory"]["warning"]:
                severity = "warning"
            else:
                continue  # No bottleneck
                
            top_allocations = dict(sorted(metric.allocations.items(), 
                                         key=lambda x: x[1], 
                                         reverse=True)[:3])
            
            bottlenecks.append({
                "type": "memory",
                "metric": metric.name,
                "severity": severity,
                "value": metric.peak_memory,
                "current_value": metric.current_memory,
                "threshold": self.bottleneck_thresholds["memory"][severity],
                "timestamp": datetime.now().isoformat(),
                "top_allocations": top_allocations,
                "recommendation": f"Reduce memory usage in {metric.name}, particularly in " +
                                 f"{', '.join(top_allocations.keys())}"
            })
            
        return bottlenecks
    
    def analyze_cpu_metrics(self) -> List[Dict[str, Any]]:
        """Analyze CPU metrics for bottlenecks"""
        bottlenecks = []
        cpu_metrics = [m for m in self.metrics_collector.get_all_metrics().values() 
                      if isinstance(m, CPUMetric)]
        
        for metric in cpu_metrics:
            if not metric.cpu_times:
                continue
                
            # Calculate CPU utilization based on last 10 samples
            recent_samples = metric.cpu_times[-10:]
            if not recent_samples:
                continue
                
            total_time = sum(s["user_time"] + s["system_time"] for s in recent_samples)
            elapsed_time = (datetime.fromisoformat(recent_samples[-1]["timestamp"]) - 
                          datetime.fromisoformat(recent_samples[0]["timestamp"])).total_seconds()
            
            if elapsed_time <= 0:
                continue
                
            utilization = total_time / (elapsed_time * os.cpu_count() if os.cpu_count() else 1)
            
            if utilization > self.bottleneck_thresholds["cpu"]["critical"]:
                severity = "critical"
            elif utilization > self.bottleneck_thresholds["cpu"]["warning"]:
                severity = "warning"
            else:
                continue  # No bottleneck
                
            bottlenecks.append({
                "type": "cpu",
                "metric": metric.name,
                "severity": severity,
                "value": utilization,
                "threshold": self.bottleneck_thresholds["cpu"][severity],
                "timestamp": datetime.now().isoformat(),
                "recommendation": f"Optimize CPU usage in {metric.name}, consider parallel processing or algorithm optimization"
            })
            
        return bottlenecks
    
    def analyze(self) -> List[Dict[str, Any]]:
        """Analyze all metrics for bottlenecks"""
        bottlenecks = []
        bottlenecks.extend(self.analyze_timing_metrics())
        bottlenecks.extend(self.analyze

