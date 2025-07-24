"""
Meta-Learning module for AdaptiveScale Networks.

This module provides implementations of various meta-learning algorithms
including MAML, continual learning, few-shot learning, and task adaptation.
"""

from .maml import (
    MAMLTrainer,
    MAMLConfig,
    FirstOrderMAML,
    SecondOrderMAML
)
from .continual import (
    ContinualLearner,
    EWCLoss,
    ElasticWeightConsolidation
)

__all__ = [
    # MAML implementations
    'MAMLTrainer',
    'MAMLConfig', 
    'FirstOrderMAML',
    'SecondOrderMAML',
    
    # Continual learning
    'ContinualLearner',
    'EWCLoss',
    'ElasticWeightConsolidation',
]

# Version info
__version__ = "1.0.0"

# Default configurations
DEFAULT_MAML_CONFIG = {
    'inner_lr': 0.01,
    'meta_lr': 0.001,
    'num_inner_steps': 5,
    'first_order': False,
    'allow_unused': True,
    'allow_nograd': False
}

DEFAULT_CONTINUAL_CONFIG = {
    'ewc_lambda': 1000.0,
    'memory_strength': 0.5,
    'online_ewc': True,
    'gamma': 1.0
}

def get_default_config(algorithm: str = 'maml'):
    """
    Get default configuration for meta-learning algorithms.
    
    Args:
        algorithm: Algorithm name ('maml', 'continual')
        
    Returns:
        dict: Default configuration
    """
    configs = {
        'maml': DEFAULT_MAML_CONFIG,
        'continual': DEFAULT_CONTINUAL_CONFIG
    }
    
    return configs.get(algorithm, DEFAULT_MAML_CONFIG)


def create_meta_learner(algorithm: str, config: dict = None, **kwargs):
    """
    Factory function to create meta-learning algorithms.
    
    Args:
        algorithm: Algorithm name
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Meta-learning algorithm instance
    """
    if config is None:
        config = get_default_config(algorithm)
    config.update(kwargs)
    
    if algorithm.lower() == 'maml':
        return MAMLTrainer(**config)
    elif algorithm.lower() == 'continual':
        return ContinualLearner(**config)
    else:
        raise ValueError(f"Unknown meta-learning algorithm: {algorithm}")


# Utility functions
def compute_meta_gradient(loss, parameters, create_graph=True, retain_graph=True):
    """Compute meta-gradients for MAML-style algorithms."""
    import torch
    
    gradients = torch.autograd.grad(
        loss,
        parameters,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True
    )
    
    return gradients


def update_parameters(parameters, gradients, lr):
    """Update parameters using gradients."""
    updated_params = []
    for param, grad in zip(parameters, gradients):
        if grad is not None:
            updated_params.append(param - lr * grad)
        else:
            updated_params.append(param)
    return updated_params