"""
Rank Predictor implementations for AdaptiveScale Networks.

This module provides various rank prediction methods for determining
optimal SVD ranks based on layer characteristics and task requirements.
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
class RankPredictorConfig:
    """Configuration for rank predictors."""
    
    # Network architecture
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 10
    num_layers: int = 3
    dropout_rate: float = 0.1
    
    # Layer normalization
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    
    # Attention mechanism
    use_attention: bool = True
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    
    # Rank prediction specifics
    max_rank: int = 512
    min_rank: int = 1
    rank_discretization: str = 'linear'  # 'linear', 'log', 'custom'
    custom_ranks: List[int] = field(default_factory=list)
    
    # Performance prediction
    predict_performance: bool = True
    performance_weight: float = 0.5
    compression_weight: float = 0.3
    efficiency_weight: float = 0.2
    
    # Ensemble settings
    ensemble_size: int = 5
    ensemble_method: str = 'voting'  # 'voting', 'averaging', 'weighted'
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    temperature: float = 1.0


class BaseRankPredictor(ABC, nn.Module):
    """Base class for all rank predictors."""
    
    def __init__(self, config: RankPredictorConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize rank discretization
        self.rank_options = self._initialize_rank_options()
        self.config.output_dim = len(self.rank_options)
        
    def _initialize_rank_options(self) -> List[int]:
        """Initialize available rank options based on configuration."""
        if self.config.custom_ranks:
            return sorted(self.config.custom_ranks)
        
        if self.config.rank_discretization == 'linear':
            return list(range(self.config.min_rank, self.config.max_rank + 1, 
                            max(1, (self.config.max_rank - self.config.min_rank) // self.config.output_dim)))
        elif self.config.rank_discretization == 'log':
            log_min = math.log(max(1, self.config.min_rank))
            log_max = math.log(self.config.max_rank)
            log_steps = np.linspace(log_min, log_max, self.config.output_dim)
            return [int(math.exp(step)) for step in log_steps]
        else:
            # Default linear
            return list(range(self.config.min_rank, self.config.max_rank + 1, 
                            max(1, (self.config.max_rank - self.config.min_rank) // self.config.output_dim)))
    
    @abstractmethod
    def forward(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass of the rank predictor."""
        pass
    
    def predict_rank(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Predict the optimal rank for given layer information."""
        output = self.forward(layer_info, task_context)
        rank_probs = F.softmax(output['logits'] / self.config.temperature, dim=-1)
        predicted_indices = torch.argmax(rank_probs, dim=-1)
        
        # Convert indices to actual ranks
        ranks = torch.tensor([self.rank_options[idx] for idx in predicted_indices], 
                           device=self.device, dtype=torch.float32)
        return ranks
    
    def get_rank_distribution(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Get probability distribution over ranks."""
        output = self.forward(layer_info, task_context)
        rank_probs = F.softmax(output['logits'] / self.config.temperature, dim=-1)
        
        return {
            'probabilities': rank_probs,
            'ranks': torch.tensor(self.rank_options, device=self.device, dtype=torch.float32),
            'expected_rank': torch.sum(rank_probs * torch.tensor(self.rank_options, device=self.device, dtype=torch.float32), dim=-1)
        }


class MLPRankPredictor(BaseRankPredictor):
    """
    Multi-Layer Perceptron based rank predictor.
    
    Uses a simple MLP to predict ranks based on layer characteristics.
    """
    
    def __init__(self, config: RankPredictorConfig):
        super().__init__(config)
        
        # Input projection
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        
        # MLP layers
        layers = []
        for i in range(config.num_layers):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            elif config.use_batch_norm:
                layers.append(nn.BatchNorm1d(config.hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
        
        self.mlp = nn.Sequential(*layers)
        
        # Output heads
        self.rank_head = nn.Linear(config.hidden_dim, self.config.output_dim)
        
        if config.predict_performance:
            self.performance_head = nn.Linear(config.hidden_dim, 3)  # accuracy, compression, efficiency
        
        logger.info(f"Initialized MLPRankPredictor with {self._count_parameters()} parameters")
    
    def _count_parameters(self):
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the MLP rank predictor.
        
        Args:
            layer_info: Layer information tensor [batch_size, input_dim]
            task_context: Optional task context tensor
            
        Returns:
            Dictionary containing rank predictions and performance estimates
        """
        # Incorporate task context if provided
        if task_context is not None:
            # Simple concatenation for now
            x = torch.cat([layer_info, task_context], dim=-1)
            # Adjust input projection if needed
            if not hasattr(self, '_context_adjusted'):
                new_input_dim = x.size(-1)
                self.input_projection = nn.Linear(new_input_dim, self.config.hidden_dim).to(x.device)
                self._context_adjusted = True
        else:
            x = layer_info
        
        # Input projection
        h = self.input_projection(x)
        
        # MLP processing
        h = self.mlp(h)
        
        # Generate outputs
        rank_logits = self.rank_head(h)
        
        outputs = {
            'logits': rank_logits,
            'hidden_states': h
        }
        
        # Performance prediction
        if self.config.predict_performance:
            performance_pred = self.performance_head(h)
            outputs.update({
                'accuracy_pred': performance_pred[:, 0],
                'compression_pred': performance_pred[:, 1],
                'efficiency_pred': performance_pred[:, 2]
            })
        
        return outputs


class AttentionRankPredictor(BaseRankPredictor):
    """
    Attention-based rank predictor.
    
    Uses attention mechanisms to focus on relevant layer characteristics
    for rank prediction.
    """
    
    def __init__(self, config: RankPredictorConfig):
        super().__init__(config)
        
        # Input processing
        self.input_projection = nn.Linear(config.input_dim, config.hidden_dim)
        self.positional_encoding = PositionalEncoding(config.hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.attention_dropout,
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Layer normalization for each attention layer
        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.hidden_dim * 4, config.hidden_dim),
                nn.Dropout(config.dropout_rate)
            ) for _ in range(config.num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim) for _ in range(config.num_layers)
        ])
        
        # Output heads
        self.rank_head = nn.Linear(config.hidden_dim, self.config.output_dim)
        self.confidence_head = nn.Linear(config.hidden_dim, 1)
        
        if config.predict_performance:
            self.performance_head = nn.Linear(config.hidden_dim, 3)
        
        logger.info(f"Initialized AttentionRankPredictor with {self._count_parameters()} parameters")
    
    def _count_parameters(self):
        """Count the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the attention-based rank predictor.
        
        Args:
            layer_info: Layer information tensor [batch_size, seq_len, input_dim]
            task_context: Optional task context tensor
            
        Returns:
            Dictionary containing rank predictions and attention weights
        """
        batch_size = layer_info.size(0)
        
        # Handle 2D input by adding sequence dimension
        if layer_info.dim() == 2:
            layer_info = layer_info.unsqueeze(1)  # [batch_size, 1, input_dim]
        
        # Input projection
        h = self.input_projection(layer_info)
        
        # Add positional encoding
        h = self.positional_encoding(h)
        
        # Store attention weights
        attention_weights = []
        
        # Process through attention layers
        for i, (attn_layer, attn_norm, ffn_layer, ffn_norm) in enumerate(
            zip(self.attention_layers, self.attention_norms, self.ffn_layers, self.ffn_norms)
        ):
            # Self-attention
            attn_output, attn_weight = attn_layer(h, h, h)
            attention_weights.append(attn_weight)
            
            # Residual connection and normalization
            h = attn_norm(h + attn_output)
            
            # Feed-forward network
            ffn_output = ffn_layer(h)
            h = ffn_norm(h + ffn_output)
        
        # Global pooling (mean over sequence dimension)
        h = h.mean(dim=1)
        
        # Generate outputs
        rank_logits = self.rank_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        
        outputs = {
            'logits': rank_logits,
            'confidence': confidence.squeeze(-1),
            'hidden_states': h,
            'attention_weights': attention_weights
        }
        
        # Performance prediction
        if self.config.predict_performance:
            performance_pred = self.performance_head(h)
            outputs.update({
                'accuracy_pred': performance_pred[:, 0],
                'compression_pred': performance_pred[:, 1],
                'efficiency_pred': performance_pred[:, 2]
            })
        
        return outputs


class EnsembleRankPredictor(BaseRankPredictor):
    """
    Ensemble of rank predictors for improved robustness and accuracy.
    
    Combines multiple rank predictors using voting or averaging strategies.
    """
    
    def __init__(self, config: RankPredictorConfig):
        super().__init__(config)
        
        # Create ensemble of predictors
        self.predictors = nn.ModuleList()
        for i in range(config.ensemble_size):
            # Alternate between MLP and Attention predictors
            if i % 2 == 0:
                predictor = MLPRankPredictor(config)
            else:
                predictor = AttentionRankPredictor(config)
            self.predictors.append(predictor)
        
        # Ensemble weights (learnable)
        if config.ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(config.ensemble_size))
        
        logger.info(f"Initialized EnsembleRankPredictor with {config.ensemble_size} predictors")
    
    def forward(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the ensemble rank predictor.
        
        Args:
            layer_info: Layer information tensor
            task_context: Optional task context tensor
            
        Returns:
            Dictionary containing ensemble predictions
        """
        predictor_outputs = []
        all_logits = []
        
        # Get predictions from all ensemble members
        for predictor in self.predictors:
            output = predictor(layer_info, task_context)
            predictor_outputs.append(output)
            all_logits.append(output['logits'])
        
        # Combine predictions based on ensemble method
        if self.config.ensemble_method == 'voting':
            # Hard voting: majority wins
            predictions = [torch.argmax(logits, dim=-1) for logits in all_logits]
            stacked_preds = torch.stack(predictions, dim=-1)
            ensemble_pred = torch.mode(stacked_preds, dim=-1)[0]
            
            # Convert back to logits (one-hot)
            ensemble_logits = torch.zeros_like(all_logits[0])
            ensemble_logits.scatter_(-1, ensemble_pred.unsqueeze(-1), 1.0)
            
        elif self.config.ensemble_method == 'averaging':
            # Simple averaging
            ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
            
        elif self.config.ensemble_method == 'weighted':
            # Weighted averaging
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_logits = torch.stack([w * logits for w, logits in zip(weights, all_logits)], dim=0)
            ensemble_logits = weighted_logits.sum(dim=0)
            
        else:
            # Default to averaging
            ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
        
        # Compute ensemble uncertainty (disagreement between predictors)
        prediction_variance = torch.var(torch.stack(all_logits, dim=0), dim=0).mean(dim=-1)
        
        outputs = {
            'logits': ensemble_logits,
            'prediction_variance': prediction_variance,
            'individual_logits': all_logits,
            'individual_outputs': predictor_outputs
        }
        
        # Aggregate performance predictions if available
        if self.config.predict_performance and 'accuracy_pred' in predictor_outputs[0]:
            accuracy_preds = [out['accuracy_pred'] for out in predictor_outputs]
            compression_preds = [out['compression_pred'] for out in predictor_outputs]
            efficiency_preds = [out['efficiency_pred'] for out in predictor_outputs]
            
            outputs.update({
                'accuracy_pred': torch.stack(accuracy_preds, dim=0).mean(dim=0),
                'compression_pred': torch.stack(compression_preds, dim=0).mean(dim=0),
                'efficiency_pred': torch.stack(efficiency_preds, dim=0).mean(dim=0),
                'accuracy_std': torch.stack(accuracy_preds, dim=0).std(dim=0),
                'compression_std': torch.stack(compression_preds, dim=0).std(dim=0),
                'efficiency_std': torch.stack(efficiency_preds, dim=0).std(dim=0)
            })
        
        return outputs


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-like architectures."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return x


class RankPredictor:
    """
    Main interface for rank prediction.
    
    Provides a unified interface for different rank prediction methods
    and includes training, evaluation, and prediction utilities.
    """
    
    def __init__(self, predictor_type: str = 'attention', config: RankPredictorConfig = None):
        if config is None:
            config = RankPredictorConfig()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize predictor based on type
        if predictor_type.lower() == 'mlp':
            self.predictor = MLPRankPredictor(config)
        elif predictor_type.lower() == 'attention':
            self.predictor = AttentionRankPredictor(config)
        elif predictor_type.lower() == 'ensemble':
            self.predictor = EnsembleRankPredictor(config)
        else:
            raise ValueError(f"Unknown predictor type: {predictor_type}")
        
        self.predictor.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Training history
        self.training_history = []
        
        logger.info(f"Initialized RankPredictor with {predictor_type} predictor")
    
    def predict(self, layer_info: torch.Tensor, task_context: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Predict optimal ranks for given layer information.
        
        Args:
            layer_info: Layer characteristics tensor
            task_context: Optional task context
            
        Returns:
            Dictionary containing predictions and metadata
        """
        self.predictor.eval()
        
        with torch.no_grad():
            layer_info = layer_info.to(self.device)
            if task_context is not None:
                task_context = task_context.to(self.device)
            
            # Get predictions
            output = self.predictor(layer_info, task_context)
            rank_dist = self.predictor.get_rank_distribution(layer_info, task_context)
            predicted_ranks = self.predictor.predict_rank(layer_info, task_context)
            
            results = {
                'predicted_ranks': predicted_ranks.cpu(),
                'rank_probabilities': rank_dist['probabilities'].cpu(),
                'expected_ranks': rank_dist['expected_rank'].cpu(),
                'available_ranks': rank_dist['ranks'].cpu()
            }
            
            # Add confidence if available
            if 'confidence' in output:
                results['confidence'] = output['confidence'].cpu()
            
            # Add performance predictions if available
            if 'accuracy_pred' in output:
                results.update({
                    'predicted_accuracy': output['accuracy_pred'].cpu(),
                    'predicted_compression': output['compression_pred'].cpu(),
                    'predicted_efficiency': output['efficiency_pred'].cpu()
                })
            
            return results
    
    def train_step(self, layer_info: torch.Tensor, target_ranks: torch.Tensor, 
                   task_context: Optional[torch.Tensor] = None,
                   performance_targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            layer_info: Layer characteristics tensor
            target_ranks: Target rank values
            task_context: Optional task context
            performance_targets: Optional performance targets
            
        Returns:
            Dictionary of training metrics
        """
        self.predictor.train()
        self.optimizer.zero_grad()
        
        # Move to device
        layer_info = layer_info.to(self.device)
        target_ranks = target_ranks.to(self.device)
        if task_context is not None:
            task_context = task_context.to(self.device)
        
        # Forward pass
        output = self.predictor(layer_info, task_context)
        
        # Convert target ranks to indices
        target_indices = []
        for rank in target_ranks:
            closest_idx = torch.argmin(torch.abs(torch.tensor(self.predictor.rank_options, device=self.device) - rank))
            target_indices.append(closest_idx)
        target_indices = torch.stack(target_indices)
        
        # Rank prediction loss
        rank_loss = F.cross_entropy(output['logits'], target_indices)
        total_loss = rank_loss
        
        # Performance prediction loss
        perf_loss = 0.0
        if self.config.predict_performance and performance_targets is not None:
            if 'accuracy' in performance_targets:
                acc_loss = F.mse_loss(output['accuracy_pred'], performance_targets['accuracy'].to(self.device))
                total_loss += self.config.performance_weight * acc_loss
                perf_loss += acc_loss.item()
            
            if 'compression' in performance_targets:
                comp_loss = F.mse_loss(output['compression_pred'], performance_targets['compression'].to(self.device))
                total_loss += self.config.compression_weight * comp_loss
                perf_loss += comp_loss.item()
            
            if 'efficiency' in performance_targets:
                eff_loss = F.mse_loss(output['efficiency_pred'], performance_targets['efficiency'].to(self.device))
                total_loss += self.config.efficiency_weight * eff_loss
                perf_loss += eff_loss.item()
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            predicted_indices = torch.argmax(output['logits'], dim=-1)
            accuracy = (predicted_indices == target_indices).float().mean().item()
        
        metrics = {
            'total_loss': total_loss.item(),
            'rank_loss': rank_loss.item(),
            'performance_loss': perf_loss,
            'accuracy': accuracy
        }
        
        return metrics
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'training_history': self.training_history
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint