"""
Model-Agnostic Meta-Learning (MAML) implementation for AdaptiveScale Networks.

This module implements MAML and its variants for fast adaptation to new tasks.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """Configuration for MAML training."""
    
    # Core MAML parameters
    inner_lr: float = 0.01
    meta_lr: float = 0.001
    num_inner_steps: int = 5
    
    # MAML variants
    first_order: bool = False  # First-order MAML (FOMAML)
    allow_unused: bool = True
    allow_nograd: bool = False
    
    # Task sampling
    tasks_per_batch: int = 4
    shots_per_task: int = 5
    query_shots_per_task: int = 10
    
    # Training parameters
    num_epochs: int = 100
    meta_batch_size: int = 32
    
    # Regularization
    l2_reg: float = 0.0
    gradient_clip: float = 1.0
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50


class MAMLTrainer:
    """
    MAML trainer for fast adaptation to new tasks.
    
    Implements both first-order and second-order MAML variants.
    """
    
    def __init__(self, model: nn.Module, config: MAMLConfig):
        self.model = model
        self.config = config
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.meta_lr
        )
        
        # Statistics tracking
        self.meta_losses = []
        self.adaptation_losses = []
        self.query_losses = []
        
        # Device
        self.device = next(model.parameters()).device
        
        logger.info(f"Initialized MAML trainer with config: {config}")
    
    def inner_update(self, support_data: Tuple[torch.Tensor, torch.Tensor], 
                     fast_weights: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform inner loop update on support data.
        
        Args:
            support_data: Tuple of (support_x, support_y)
            fast_weights: Current fast weights (if None, use model weights)
            
        Returns:
            Updated fast weights
        """
        support_x, support_y = support_data
        support_x, support_y = support_x.to(self.device), support_y.to(self.device)
        
        # Initialize fast weights if not provided
        if fast_weights is None:
            fast_weights = {name: param.clone() for name, param in self.model.named_parameters()
                          if param.requires_grad}
        
        # Forward pass with fast weights
        logits = self._forward_with_weights(support_x, fast_weights)
        loss = F.cross_entropy(logits, support_y)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            loss, 
            fast_weights.values(),
            create_graph=not self.config.first_order,
            retain_graph=True,
            allow_unused=self.config.allow_unused
        )
        
        # Update fast weights
        updated_weights = {}
        for (name, param), grad in zip(fast_weights.items(), gradients):
            if grad is not None:
                updated_weights[name] = param - self.config.inner_lr * grad
            else:
                updated_weights[name] = param
        
        return updated_weights
    
    def _forward_with_weights(self, x: torch.Tensor, 
                             weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using custom weights.
        
        Args:
            x: Input tensor
            weights: Dictionary of weights to use
            
        Returns:
            Model output
        """
        # This is a simplified implementation
        # In practice, you'd need to implement functional forward pass
        # for your specific model architecture
        
        # Save original weights
        original_weights = {}
        for name, param in self.model.named_parameters():
            if name in weights:
                original_weights[name] = param.data.clone()
                param.data = weights[name]
        
        # Forward pass
        output = self.model(x)
        
        # Restore original weights
        for name, param in self.model.named_parameters():
            if name in original_weights:
                param.data = original_weights[name]
        
        return output
    
    def meta_update(self, batch_tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                                torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        Perform meta-update across a batch of tasks.
        
        Args:
            batch_tasks: List of (support_x, support_y, query_x, query_y) tuples
            
        Returns:
            Dictionary of losses and metrics
        """
        meta_loss = 0.0
        task_losses = []
        adaptation_accuracies = []
        query_accuracies = []
        
        for task_data in batch_tasks:
            support_x, support_y, query_x, query_y = task_data
            
            # Initialize fast weights
            fast_weights = {name: param.clone() for name, param in self.model.named_parameters()
                          if param.requires_grad}
            
            # Inner loop updates
            adaptation_loss = 0.0
            for step in range(self.config.num_inner_steps):
                fast_weights = self.inner_update((support_x, support_y), fast_weights)
                
                # Track adaptation loss
                with torch.no_grad():
                    logits = self._forward_with_weights(support_x, fast_weights)
                    adaptation_loss += F.cross_entropy(logits, support_y.to(self.device)).item()
            
            # Query loss for meta-update
            query_x, query_y = query_x.to(self.device), query_y.to(self.device)
            query_logits = self._forward_with_weights(query_x, fast_weights)
            query_loss = F.cross_entropy(query_logits, query_y)
            
            meta_loss += query_loss
            task_losses.append(query_loss.item())
            
            # Track accuracies
            with torch.no_grad():
                # Adaptation accuracy
                support_logits = self._forward_with_weights(support_x, fast_weights)
                adapt_acc = (support_logits.argmax(dim=1) == support_y.to(self.device)).float().mean().item()
                adaptation_accuracies.append(adapt_acc)
                
                # Query accuracy
                query_acc = (query_logits.argmax(dim=1) == query_y).float().mean().item()
                query_accuracies.append(query_acc)
        
        # Average meta loss
        meta_loss = meta_loss / len(batch_tasks)
        
        # L2 regularization
        if self.config.l2_reg > 0:
            l2_loss = sum(p.pow(2).sum() for p in self.model.parameters())
            meta_loss += self.config.l2_reg * l2_loss
        
        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip
            )
        
        self.meta_optimizer.step()
        
        # Collect metrics
        metrics = {
            'meta_loss': meta_loss.item(),
            'avg_task_loss': np.mean(task_losses),
            'avg_adaptation_accuracy': np.mean(adaptation_accuracies),
            'avg_query_accuracy': np.mean(query_accuracies),
            'std_task_loss': np.std(task_losses),
            'std_adaptation_accuracy': np.std(adaptation_accuracies),
            'std_query_accuracy': np.std(query_accuracies)
        }
        
        return metrics
    
    def train_epoch(self, task_loader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            task_loader: DataLoader yielding batches of tasks
            
        Returns:
            Dictionary of epoch metrics
        """
        self.model.train()
        epoch_metrics = []
        
        for batch_idx, batch_tasks in enumerate(task_loader):
            metrics = self.meta_update(batch_tasks)
            epoch_metrics.append(metrics)
            
            if batch_idx % self.config.log_interval == 0:
                logger.info(f"Batch {batch_idx}: {metrics}")
        
        # Average metrics across batches
        avg_metrics = {}
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in epoch_metrics])
        
        return avg_metrics
    
    def evaluate(self, eval_tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                           torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        Evaluate on a set of tasks.
        
        Args:
            eval_tasks: List of evaluation tasks
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_accuracies = []
        all_losses = []
        
        with torch.no_grad():
            for support_x, support_y, query_x, query_y in eval_tasks:
                # Fast adaptation
                fast_weights = {name: param.clone() for name, param in self.model.named_parameters()
                              if param.requires_grad}
                
                for _ in range(self.config.num_inner_steps):
                    fast_weights = self.inner_update((support_x, support_y), fast_weights)
                
                # Evaluate on query set
                query_x, query_y = query_x.to(self.device), query_y.to(self.device)
                query_logits = self._forward_with_weights(query_x, fast_weights)
                query_loss = F.cross_entropy(query_logits, query_y)
                query_accuracy = (query_logits.argmax(dim=1) == query_y).float().mean()
                
                all_accuracies.append(query_accuracy.item())
                all_losses.append(query_loss.item())
        
        return {
            'eval_accuracy': np.mean(all_accuracies),
            'eval_loss': np.mean(all_losses),
            'eval_accuracy_std': np.std(all_accuracies),
            'eval_loss_std': np.std(all_losses)
        }
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint


class FirstOrderMAML(MAMLTrainer):
    """First-order MAML (FOMAML) implementation."""
    
    def __init__(self, model: nn.Module, config: MAMLConfig):
        config.first_order = True
        super().__init__(model, config)
        logger.info("Initialized First-Order MAML")


class SecondOrderMAML(MAMLTrainer):
    """Second-order MAML implementation."""
    
    def __init__(self, model: nn.Module, config: MAMLConfig):
        config.first_order = False
        super().__init__(model, config)
        logger.info("Initialized Second-Order MAML")


def create_maml_trainer(model: nn.Module, config: MAMLConfig, 
                       variant: str = "second_order") -> MAMLTrainer:
    """
    Factory function to create MAML trainer variants.
    
    Args:
        model: Neural network model
        config: MAML configuration
        variant: MAML variant ("first_order" or "second_order")
        
    Returns:
        MAML trainer instance
    """
    if variant.lower() == "first_order":
        return FirstOrderMAML(model, config)
    elif variant.lower() == "second_order":
        return SecondOrderMAML(model, config)
    else:
        raise ValueError(f"Unknown MAML variant: {variant}")


# Utility functions for MAML
def compute_maml_loss(model: nn.Module, support_data: Tuple[torch.Tensor, torch.Tensor],
                     query_data: Tuple[torch.Tensor, torch.Tensor], 
                     inner_lr: float = 0.01, num_steps: int = 5) -> torch.Tensor:
    """
    Compute MAML loss for a single task.
    
    Args:
        model: Neural network model
        support_data: Support set (x, y)
        query_data: Query set (x, y)
        inner_lr: Inner loop learning rate
        num_steps: Number of inner loop steps
        
    Returns:
        MAML loss tensor
    """
    support_x, support_y = support_data
    query_x, query_y = query_data
    
    # Get initial parameters
    fast_weights = {name: param for name, param in model.named_parameters() if param.requires_grad}
    
    # Inner loop updates
    for _ in range(num_steps):
        # Support loss
        support_pred = model(support_x)  # This would need functional forward pass
        support_loss = F.cross_entropy(support_pred, support_y)
        
        # Compute gradients
        grads = torch.autograd.grad(support_loss, fast_weights.values(), create_graph=True)
        
        # Update fast weights
        fast_weights = {name: param - inner_lr * grad 
                       for (name, param), grad in zip(fast_weights.items(), grads)}
    
    # Query loss with updated weights
    query_pred = model(query_x)  # This would need functional forward pass with fast_weights
    query_loss = F.cross_entropy(query_pred, query_y)
    
    return query_loss