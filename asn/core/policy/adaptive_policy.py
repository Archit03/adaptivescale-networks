"""
Adaptive Policy implementations for AdaptiveScale Networks.

This module provides various adaptive policies for neural network scaling,
including hierarchical policies, task-aware policies, and dynamic scaling.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class PolicyConfig:
    """Configuration for adaptive policies."""
    
    # Network architecture
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    num_layers: int = 3
    dropout_rate: float = 0.1
    
    # Activation and normalization
    activation: str = 'relu'
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    
    # Attention mechanism
    use_attention: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Policy-specific parameters
    temperature: float = 1.0
    entropy_weight: float = 0.01
    exploration_noise: float = 0.1
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    
    # Task adaptation
    num_tasks: int = 4
    task_embedding_dim: int = 64
    use_task_embedding: bool = True
    
    # Hierarchical settings
    num_hierarchy_levels: int = 3
    level_dims: List[int] = field(default_factory=lambda: [64, 128, 256])


class BasePolicy(ABC, nn.Module):
    """Base class for all adaptive policies."""
    
    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the policy."""
        pass
    
    def get_action_probabilities(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Get action probabilities from the policy."""
        output = self.forward(x, context)
        return F.softmax(output['logits'] / self.config.temperature, dim=-1)
    
    def sample_action(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Sample action from the policy."""
        probs = self.get_action_probabilities(x, context)
        return torch.multinomial(probs, 1).squeeze(-1)
    
    def compute_entropy(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute policy entropy."""
        probs = self.get_action_probabilities(x, context)
        log_probs = torch.log(probs + 1e-8)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy


class AdaptivePolicy(BasePolicy):
    """
    Basic adaptive policy for neural network scaling decisions.
    
    Uses a multi-layer perceptron with optional attention mechanism
    to make scaling decisions based on layer information and context.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        
        # Input processing
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Main network
        layers = []
        for i in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            elif config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            
            layers.append(self._get_activation())
            layers.append(nn.Dropout(config.dropout_rate))
        
        self.main_network = nn.Sequential(*layers)
        
        # Attention mechanism
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                batch_first=True
            )
        
        # Output head
        self.output_head = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Value head for policy gradient methods
        self.value_head = nn.Linear(config.hidden_dim, 1)
        
        logger.info(f"Initialized AdaptivePolicy with {self._count_parameters()} parameters")
    
    def _get_activation(self):
        """Get activation function based on config."""
        if self.config.activation.lower() == 'relu':
            return nn.ReLU()
        elif self.config.activation.lower() == 'gelu':
            return nn.GELU()
        elif self.config.activation.lower() == 'swish':
            return nn.SiLU()
        elif self.config.activation.lower() == 'tanh':
            return nn.Tanh()
        else:
            return nn.ReLU()
    
    def _count_parameters(self):
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the adaptive policy.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            context: Optional context information
            
        Returns:
            Dictionary containing logits, values, and other outputs
        """
        batch_size = x.size(0)
        
        # Input projection
        h = self.input_projection(x)
        
        # Main network processing
        h = self.main_network(h)
        
        # Apply attention if enabled
        if self.config.use_attention:
            # Self-attention
            h = h.unsqueeze(1)  # Add sequence dimension
            attended_h, attention_weights = self.attention(h, h, h)
            h = attended_h.squeeze(1)  # Remove sequence dimension
        else:
            attention_weights = None
        
        # Output heads
        logits = self.output_head(h)
        values = self.value_head(h)
        
        # Add exploration noise during training
        if self.training and self.config.exploration_noise > 0:
            noise = torch.randn_like(logits) * self.config.exploration_noise
            logits = logits + noise
        
        return {
            'logits': logits,
            'values': values.squeeze(-1),
            'hidden_states': h,
            'attention_weights': attention_weights
        }


class HierarchicalPolicy(BasePolicy):
    """
    Hierarchical policy that makes decisions at multiple levels.
    
    Useful for making coarse-to-fine scaling decisions across
    different granularities (e.g., model-wide, layer-wise, parameter-wise).
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        
        # Hierarchical levels
        self.num_levels = config.num_hierarchy_levels
        self.level_dims = config.level_dims or [config.hidden_dim] * self.num_levels
        
        # Input processing
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Level-specific networks
        self.level_networks = nn.ModuleList()
        self.level_outputs = nn.ModuleList()
        
        for i, level_dim in enumerate(self.level_dims):
            # Level network
            level_net = nn.Sequential(
                nn.Linear(config.hidden_dim if i == 0 else self.level_dims[i-1], level_dim),
                nn.LayerNorm(level_dim) if config.use_layer_norm else nn.Identity(),
                self._get_activation(),
                nn.Dropout(config.dropout_rate)
            )
            self.level_networks.append(level_net)
            
            # Level output
            level_output = nn.Linear(level_dim, config.output_dim)
            self.level_outputs.append(level_output)
        
        # Fusion mechanism
        fusion_input_dim = sum(self.level_dims)
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Value network
        self.value_network = nn.Linear(fusion_input_dim, 1)
        
        logger.info(f"Initialized HierarchicalPolicy with {self.num_levels} levels")
    
    def _get_activation(self):
        """Get activation function based on config."""
        return AdaptivePolicy._get_activation(self)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the hierarchical policy.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            context: Optional context information
            
        Returns:
            Dictionary containing hierarchical outputs
        """
        # Input projection
        h = self.input_projection(x)
        
        # Process through hierarchical levels
        level_features = []
        level_logits = []
        
        current_input = h
        for i, (level_net, level_output) in enumerate(zip(self.level_networks, self.level_outputs)):
            # Process at current level
            level_feat = level_net(current_input)
            level_features.append(level_feat)
            
            # Generate level-specific logits
            level_logit = level_output(level_feat)
            level_logits.append(level_logit)
            
            # Use level features as input to next level
            current_input = level_feat
        
        # Fuse all level features
        fused_features = torch.cat(level_features, dim=-1)
        fused_logits = self.fusion_network(fused_features)
        
        # Compute values
        values = self.value_network(fused_features)
        
        return {
            'logits': fused_logits,
            'values': values.squeeze(-1),
            'level_logits': level_logits,
            'level_features': level_features,
            'fused_features': fused_features
        }


class TaskAwarePolicy(BasePolicy):
    """
    Task-aware policy that adapts its behavior based on task context.
    
    Uses task embeddings to condition the policy decisions on the current task.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        
        # Task embedding
        if config.use_task_embedding:
            self.task_embedding = nn.Embedding(config.num_tasks, config.task_embedding_dim)
            self.task_projection = nn.Linear(config.task_embedding_dim, config.hidden_dim)
        
        # Input processing
        input_dim = config.input_dim
        if config.use_task_embedding:
            input_dim += config.hidden_dim  # Add task projection dimension
        
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        
        # Task-conditioned network
        self.task_conditioned_layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim) if config.use_layer_norm else nn.Identity(),
                self._get_activation(),
                nn.Dropout(config.dropout_rate)
            )
            self.task_conditioned_layers.append(layer)
        
        # Task-specific output heads
        self.task_output_heads = nn.ModuleList()
        self.task_value_heads = nn.ModuleList()
        
        for _ in range(config.num_tasks):
            output_head = nn.Linear(config.hidden_dim, config.output_dim)
            value_head = nn.Linear(config.hidden_dim, 1)
            self.task_output_heads.append(output_head)
            self.task_value_heads.append(value_head)
        
        # Shared output head (fallback)
        self.shared_output_head = nn.Linear(config.hidden_dim, config.output_dim)
        self.shared_value_head = nn.Linear(config.hidden_dim, 1)
        
        logger.info(f"Initialized TaskAwarePolicy for {config.num_tasks} tasks")
    
    def _get_activation(self):
        """Get activation function based on config."""
        return AdaptivePolicy._get_activation(self)
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the task-aware policy.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            context: Context containing 'task_id' tensor [batch_size]
            
        Returns:
            Dictionary containing task-conditioned outputs
        """
        batch_size = x.size(0)
        
        # Get task information
        if context is not None and 'task_id' in context:
            task_ids = context['task_id']
        else:
            # Default to task 0 if no task information provided
            task_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Process task embeddings
        if self.config.use_task_embedding:
            task_embeds = self.task_embedding(task_ids)
            task_features = self.task_projection(task_embeds)
            
            # Concatenate with input
            x = torch.cat([x, task_features], dim=-1)
        
        # Input projection
        h = self.input_projection(x)
        
        # Process through task-conditioned layers
        for layer in self.task_conditioned_layers:
            h = layer(h)
        
        # Generate task-specific outputs
        batch_logits = []
        batch_values = []
        
        for i in range(batch_size):
            task_id = task_ids[i].item()
            
            if task_id < len(self.task_output_heads):
                # Use task-specific head
                logits = self.task_output_heads[task_id](h[i:i+1])
                values = self.task_value_heads[task_id](h[i:i+1])
            else:
                # Use shared head for unknown tasks
                logits = self.shared_output_head(h[i:i+1])
                values = self.shared_value_head(h[i:i+1])
            
            batch_logits.append(logits)
            batch_values.append(values)
        
        # Concatenate batch results
        final_logits = torch.cat(batch_logits, dim=0)
        final_values = torch.cat(batch_values, dim=0)
        
        return {
            'logits': final_logits,
            'values': final_values.squeeze(-1),
            'hidden_states': h,
            'task_ids': task_ids,
            'task_embeddings': task_embeds if self.config.use_task_embedding else None
        }


class DynamicScalingPolicy(BasePolicy):
    """
    Dynamic scaling policy that adapts based on model performance and resource constraints.
    
    Incorporates performance metrics, resource usage, and uncertainty estimates
    to make intelligent scaling decisions.
    """
    
    def __init__(self, config: PolicyConfig):
        super().__init__(config)
        
        # Performance tracking
        self.performance_window = 100
        self.performance_history = []
        
        # Resource tracking
        self.resource_weight = 0.1
        self.performance_weight = 0.7
        self.uncertainty_weight = 0.2
        
        # Input processing with additional context
        context_dim = 64  # For performance metrics, resource usage, etc.
        self.context_processor = nn.Sequential(
            nn.Linear(context_dim, config.hidden_dim // 2),
            nn.LayerNorm(config.hidden_dim // 2) if config.use_layer_norm else nn.Identity(),
            self._get_activation(),
            nn.Dropout(config.dropout_rate)
        )
        
        total_input_dim = config.input_dim + config.hidden_dim // 2
        self.input_projection = nn.Linear(total_input_dim, config.hidden_dim)
        
        # Main processing network
        self.main_network = nn.Sequential()
        for i in range(config.num_layers):
            self.main_network.add_module(f'layer_{i}', nn.Linear(config.hidden_dim, config.hidden_dim))
            if config.use_layer_norm:
                self.main_network.add_module(f'norm_{i}', nn.LayerNorm(config.hidden_dim))
            self.main_network.add_module(f'activation_{i}', self._get_activation())
            self.main_network.add_module(f'dropout_{i}', nn.Dropout(config.dropout_rate))
        
        # Multiple output heads for different aspects
        self.scaling_head = nn.Linear(config.hidden_dim, config.output_dim)  # Scaling decisions
        self.confidence_head = nn.Linear(config.hidden_dim, 1)  # Confidence in decision
        self.value_head = nn.Linear(config.hidden_dim, 1)  # Value estimation
        
        # Adaptive temperature
        self.temperature_net = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Softplus()
        )
        
        logger.info("Initialized DynamicScalingPolicy")
    
    def _get_activation(self):
        """Get activation function based on config."""
        return AdaptivePolicy._get_activation(self)
    
    def update_performance(self, performance_metric: float):
        """Update performance history."""
        self.performance_history.append(performance_metric)
        if len(self.performance_history) > self.performance_window:
            self.performance_history.pop(0)
    
    def get_performance_trend(self) -> float:
        """Compute performance trend from recent history."""
        if len(self.performance_history) < 2:
            return 0.0
        
        recent = self.performance_history[-10:]  # Last 10 measurements
        if len(recent) < 2:
            return 0.0
        
        # Simple linear trend
        x = torch.arange(len(recent), dtype=torch.float32)
        y = torch.tensor(recent, dtype=torch.float32)
        
        # Linear regression
        n = len(recent)
        sum_x = x.sum()
        sum_y = y.sum()
        sum_xy = (x * y).sum()
        sum_x2 = (x * x).sum()
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x + 1e-8)
        return slope.item()
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the dynamic scaling policy.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            context: Context containing performance metrics, resource usage, etc.
            
        Returns:
            Dictionary containing scaling decisions and metadata
        """
        batch_size = x.size(0)
        
        # Process context information
        if context is not None:
            # Extract context features
            context_features = []
            
            # Performance metrics
            if 'performance_metrics' in context:
                context_features.append(context['performance_metrics'])
            else:
                # Use default performance features
                perf_trend = self.get_performance_trend()
                perf_features = torch.tensor([
                    perf_trend,
                    len(self.performance_history) / self.performance_window,
                    np.mean(self.performance_history) if self.performance_history else 0.0
                ], device=x.device).unsqueeze(0).repeat(batch_size, 1)
                context_features.append(perf_features)
            
            # Resource usage
            if 'resource_usage' in context:
                context_features.append(context['resource_usage'])
            else:
                # Default resource features
                resource_features = torch.zeros(batch_size, 16, device=x.device)
                context_features.append(resource_features)
            
            # Uncertainty estimates
            if 'uncertainty' in context:
                context_features.append(context['uncertainty'].unsqueeze(-1))
            else:
                uncertainty_features = torch.zeros(batch_size, 1, device=x.device)
                context_features.append(uncertainty_features)
            
            # Combine context features
            context_tensor = torch.cat(context_features, dim=-1)
            
            # Pad or truncate to expected size
            if context_tensor.size(-1) < 64:
                padding = torch.zeros(batch_size, 64 - context_tensor.size(-1), device=x.device)
                context_tensor = torch.cat([context_tensor, padding], dim=-1)
            elif context_tensor.size(-1) > 64:
                context_tensor = context_tensor[:, :64]
        else:
            # Default context
            context_tensor = torch.zeros(batch_size, 64, device=x.device)
        
        # Process context
        context_features = self.context_processor(context_tensor)
        
        # Combine input and context
        combined_input = torch.cat([x, context_features], dim=-1)
        
        # Input projection
        h = self.input_projection(combined_input)
        
        # Main processing
        h = self.main_network(h)
        
        # Generate outputs
        scaling_logits = self.scaling_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        values = self.value_head(h)
        
        # Adaptive temperature
        temperature = self.temperature_net(h) + 0.1  # Minimum temperature
        
        # Apply temperature scaling
        scaled_logits = scaling_logits / temperature
        
        return {
            'logits': scaled_logits,
            'raw_logits': scaling_logits,
            'values': values.squeeze(-1),
            'confidence': confidence.squeeze(-1),
            'temperature': temperature.squeeze(-1),
            'hidden_states': h,
            'context_features': context_features
        }