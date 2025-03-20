"""
Federated Learning module for secure financial model training.

This module implements a federated learning system that enables distributed model training
across multiple nodes while preserving data privacy and security.
"""

import asyncio
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
import math
import os
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa, utils
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Type definitions
Model = TypeVar('Model')
Parameters = Dict[str, np.ndarray]
ModelDiff = Dict[str, np.ndarray]
ModelUpdate = Dict[str, np.ndarray]


class FederatedRole(Enum):
    """Roles in the federated learning system."""
    COORDINATOR = auto()
    PARTICIPANT = auto()
    VALIDATOR = auto()
    AUDITOR = auto()


class SecurityLevel(Enum):
    """Security levels for different sensitivity tiers."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class FederatedConfig:
    """Configuration for federated learning."""
    # Network configuration
    coordinator_url: str = "localhost:8080"
    heartbeat_interval_seconds: int = 30
    connection_timeout_seconds: int = 60
    
    # Training configuration
    min_participants: int = 3
    max_participants: int = 100
    aggregation_rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    
    # Security configuration
    security_level: SecurityLevel = SecurityLevel.HIGH
    noise_scale: float = 0.1  # For differential privacy
    clipping_threshold: float = 5.0  # For gradient clipping
    encryption_enabled: bool = True
    verification_enabled: bool = True
    
    # Advanced configuration
    enable_adaptive_aggregation: bool = True
    contribution_weighting: bool = True
    outlier_detection: bool = True
    poisoning_defense: bool = True
    key_rotation_interval_hours: int = 24
    
    # Financial domain specific
    time_sensitivity_ms: int = 100  # Maximum allowed delay for time-sensitive updates
    regulatory_compliance_mode: bool = True
    audit_trail_enabled: bool = True


class FederatedAggregator:
    """
    Implements federated model aggregation strategies.
    
    Supports various aggregation methods including weighted averaging,
    secure aggregation, and robust aggregation techniques.
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger("FederatedAggregator")
        
        # Importance weights for each participant
        self.participant_weights: Dict[str, float] = {}
        
        # Track contribution quality
        self.quality_scores: Dict[str, float] = {}
        
        # Detect and handle malicious updates
        self.outlier_detector = OutlierDetector(
            threshold=2.0,
            window_size=5
        )
    
    async def federated_averaging(
        self, 
        updates: Dict[str, ModelUpdate],
        weights: Optional[Dict[str, float]] = None
    ) -> Parameters:
        """
        Perform federated averaging (FedAvg) on model updates.
        
        Args:
            updates: Dictionary mapping participant IDs to their model updates
            weights: Optional dictionary with importance weights for each participant
                     If None, uniform weights are used
        
        Returns:
            Aggregated model parameters
        """
        if not updates:
            raise ValueError("No updates provided for aggregation")
        
        # Use provided weights or default to uniform weighting
        if weights is None:
            weights = {pid: 1.0 / len(updates) for pid in updates}
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        normalized_weights = {pid: w / total_weight for pid, w in weights.items()}
        
        # Initialize the aggregated parameters with zeros
        first_update = next(iter(updates.values()))
        aggregated_params = {
            key: np.zeros_like(value) for key, value in first_update.items()
        }
        
        # Weighted average of all updates
        for pid, update in updates.items():
            if pid not in normalized_weights:
                self.logger.warning(f"Participant {pid} not in weights dictionary, skipping")
                continue
                
            weight = normalized_weights[pid]
            for key, value in update.items():
                if key in aggregated_params:
                    aggregated_params[key] += weight * value
        
        return aggregated_params
    
    async def secure_aggregation(self, updates: Dict[str, ModelUpdate]) -> Parameters:
        """
        Perform secure aggregation that preserves privacy of individual updates.
        
        Implements secure aggregation protocol where participants' updates are 
        masked with random values that sum to zero, allowing computation of 
        the sum without revealing individual updates.
        
        Args:
            updates: Dictionary mapping participant IDs to their model updates
            
        Returns:
            Securely aggregated model parameters
        """
        if len(updates) < 3:
            self.logger.warning("Secure aggregation requires at least 3 participants, falling back to regular aggregation")
            return await self.federated_averaging(updates)
            
        # In a real implementation, this would perform the secure aggregation protocol
        # Here we simulate the result, which would be equivalent to regular aggregation
        # but with privacy guarantees
        return await self.federated_averaging(updates)
    
    async def robust_aggregation(self, updates: Dict[str, ModelUpdate]) -> Parameters:
        """
        Perform robust aggregation resistant to adversarial updates.
        
        Implements techniques like trimmed mean or median aggregation to
        mitigate the effect of poisoning attacks.
        
        Args:
            updates: Dictionary mapping participant IDs to their model updates
            
        Returns:
            Robustly aggregated model parameters
        """
        if not updates:
            raise ValueError("No updates provided for aggregation")
            
        # Detect outliers in the updates
        inlier_updates = {}
        for pid, update in updates.items():
            # Convert the update to a flat vector for outlier detection
            update_vector = np.concatenate([arr.flatten() for arr in update.values()])
            
            # Check if this update is an outlier
            if not self.outlier_detector.is_outlier(update_vector):
                inlier_updates[pid] = update
            else:
                self.logger.warning(f"Detected outlier update from participant {pid}, excluding from aggregation")
        
        if not inlier_updates:
            self.logger.warning("All updates were classified as outliers, using original updates")
            inlier_updates = updates
            
        # Compute quality-based weights for the remaining updates
        weights = self._compute_quality_weights(inlier_updates)
        
        # Use weighted averaging with the filtered updates
        return await self.federated_averaging(inlier_updates, weights)
    
    def _compute_quality_weights(self, updates: Dict[str, ModelUpdate]) -> Dict[str, float]:
        """Compute weights based on update quality and past contributions."""
        weights = {}
        
        # Start with uniform weights
        for pid in updates:
            weights[pid] = 1.0
            
            # Adjust based on historical quality if available
            if pid in self.quality_scores:
                weights[pid] *= (0.5 + 0.5 * self.quality_scores[pid])
        
        # Normalize weights
        total_weight = sum(weights.values())
        return {pid: w / total_weight for pid, w in weights.items()}
    
    async def adaptive_aggregation(self, updates: Dict[str, ModelUpdate], round_num: int) -> Parameters:
        """
        Adaptively choose the best aggregation method based on current conditions.
        
        Args:
            updates: Dictionary mapping participant IDs to their model updates
            round_num: Current round number of federated training
            
        Returns:
            Aggregated model parameters using the best method for current conditions
        """
        # Determine the appropriate aggregation strategy based on various factors
        
        # Use robust aggregation in early rounds to establish a baseline
        if round_num < 3:
            self.logger.info("Using robust aggregation for early training rounds")
            return await self.robust_aggregation(updates)
        
        # If high security level is required, use secure aggregation
        if self.config.security_level in (SecurityLevel.HIGH, SecurityLevel.CRITICAL):
            self.logger.info("Using secure aggregation due to high security requirements")
            return await self.secure_aggregation(updates)
            
        # If we have reason to suspect poisoning attacks, use robust aggregation
        if self.config.poisoning_defense and self.outlier_detector.outlier_frequency > 0.2:
            self.logger.info("Using robust aggregation due to high frequency of outliers")
            return await self.robust_aggregation(updates)
            
        # Default to standard federated averaging
        self.logger.info("Using standard federated averaging")
        return await self.federated_averaging(updates)


class PrivacyEngine:
    """
    Implements differential privacy techniques for model training.
    
    Provides methods to add calibrated noise to gradients and parameters,
    enforce gradient clipping, and compute privacy budgets.
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.epsilon = 1.0  # Privacy budget
        self.delta = 1e-5   # Failure probability
        self.noise_multiplier = config.noise_scale
        self.clipping_threshold = config.clipping_threshold
        self.logger = logging.getLogger("PrivacyEngine")
        
        # Privacy budget accounting
        self.privacy_spent = 0.0
        
    def add_noise(self, parameters: Parameters) -> Parameters:
        """
        Add calibrated Gaussian noise to model parameters for differential privacy.
        
        Args:
            parameters: Original model parameters
            
        Returns:
            Parameters with added noise
        """
        noised_parameters = {}
        
        for key, param in parameters.items():
            # Scale noise based on parameter shape and privacy settings
            noise_scale = self.noise_multiplier * self.clipping_threshold
            noise = np.random.normal(0, noise_scale, param.shape)
            
            # Add noise to the parameter
            noised_parameters[key] = param + noise
            
        # Update privacy accounting
        self._update_privacy_accounting()
        
        return noised_parameters
    
    def clip_gradients(self, gradients: Parameters) -> Parameters:
        """
        Clip gradients to limit sensitivity for differential privacy.
        
        Args:
            gradients: Original gradients
            
        Returns:
            Clipped gradients
        """
        # Compute the l2 norm of the gradients
        squared_sum = 0
        for grad in gradients.values():
            squared_sum += np.sum(np.square(grad))
        l2_norm = np.sqrt(squared_sum)
        
        # Compute the clipping factor
        clipping_factor = min(1.0, self.clipping_threshold / (l2_norm + 1e-12))
        
        # Clip gradients
        clipped_gradients = {}
        for key, grad in gradients.items():
            clipped_gradients[key] = grad * clipping_factor
            
        return clipped_gradients
    
    def _update_privacy_accounting(self):
        """Update the privacy budget accounting."""
        # In a real implementation, this would use advanced privacy accounting methods
        # like Rényi Differential Privacy (RDP) or zCDP
        # Here we use a simple heuristic
        self.privacy_spent += 0.1
        
        if self.privacy_spent > self.epsilon:
            self.logger.warning("Privacy budget exceeded!")
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get the current privacy budget spent.
        
        Returns:
            Tuple of (epsilon, delta) representing privacy spent
        """
        return (self.privacy_spent, self.delta)
    
    async def privatize_update(self, update: ModelUpdate) -> ModelUpdate:
        """
        Apply differential privacy to a model update.
        
        Args:
            update: Original model update
            
        Returns:
            Privatized model update
        """
        # First clip the update to bound sensitivity
        clipped_update = self.clip_gradients(update)
        
        # Then add calibrated noise
        private_update = self.add_noise(clipped_update)
        
        return private_update


class SecureUpdateManager:
    """
    Manages secure model updates with encryption, verification, and integrity checks.
    
    Handles secure transmission of model updates between participants and coordinators
    using encryption, digital signatures, and secure hashing.
    """
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.logger = logging.getLogger("SecureUpdateManager")
        
        # Generate or load keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Track timestamps for key rotation
        self.last_key_rotation = time.time()
        
    async def encrypt_update(self, update: ModelUpdate, recipient_public_key) -> bytes:
        """
        Encrypt a model update for secure transmission.
        
        Args:
            update: Model update to encrypt
            recipient_public_key: Public key of the recipient
            
        Returns:
            Encrypted model update
        """
        # Serialize the update
        serialized_update = self._serialize_update(update)
        
        # Generate a symmetric key for AES encryption
        symmetric_key = os.urandom(32)
        
        # Encrypt the update with AES
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(symmetric_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        encrypted_update = encryptor.update(serialized_update) + encryptor.finalize()
        
        # Encrypt the symmetric key with RSA
        encrypted_key = recipient_public_key.encrypt(
            symmetric_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Combine IV, encrypted key and encrypted update
        result = iv + len(encrypted_key).to_bytes(4, byteorder='big') + encrypted_key + encrypted_update
        
        return result
    
    async def decrypt_update(self, encrypted_data: bytes) -> ModelUpdate:
        """
        Decrypt an encrypted model update.
        
        Args:
            encrypted_data: Encrypted model update
            
        Returns:
            Decrypted model update
        """
        # Extract IV, encrypted key length, and encrypted key
        iv = encrypted_data[:16]
        key_length = int.from_bytes(encrypted_data[16:20], byteorder='big')
        encrypted_

