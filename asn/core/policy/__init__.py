"""
Policy module for AdaptiveScale Networks.

This module provides adaptive policies for neural network scaling,
including rank prediction, uncertainty estimation, and hierarchical adaptation.
"""

from .adaptive_policy import (
    AdaptivePolicy,
    HierarchicalPolicy,
    TaskAwarePolicy,
    DynamicScalingPolicy,
    PolicyConfig
)
from .rank_predictor import (
    RankPredictor,
    AttentionRankPredictor,
    MLPRankPredictor,
    EnsembleRankPredictor,
    RankPredictorConfig
)
from .uncertainty import (
    UncertaintyEstimator,
    BayesianUncertaintyEstimator,
    EnsembleUncertaintyEstimator,
    MCDropoutUncertainty,
    UncertaintyConfig
)

__all__ = [
    # Adaptive policies
    'AdaptivePolicy',
    'HierarchicalPolicy',
    'TaskAwarePolicy',
    'DynamicScalingPolicy',
    'PolicyConfig',
    
    # Rank predictors
    'RankPredictor',
    'AttentionRankPredictor',
    'MLPRankPredictor',
    'EnsembleRankPredictor',
    'RankPredictorConfig',
    
    # Uncertainty estimators
    'UncertaintyEstimator',
    'BayesianUncertaintyEstimator',
    'EnsembleUncertaintyEstimator',
    'MCDropoutUncertainty',
    'UncertaintyConfig',
]

# Version info
__version__ = "1.0.0"

# Default configurations
DEFAULT_POLICY_CONFIG = {
    'hidden_dim': 256,
    'num_layers': 3,
    'dropout_rate': 0.1,
    'activation': 'relu',
    'use_attention': True,
    'temperature': 1.0
}

DEFAULT_RANK_PREDICTOR_CONFIG = {
    'input_dim': 128,
    'hidden_dim': 256,
    'output_dim': 10,
    'num_layers': 3,
    'dropout_rate': 0.1,
    'use_layer_norm': True
}

DEFAULT_UNCERTAINTY_CONFIG = {
    'num_samples': 10,
    'dropout_rate': 0.1,
    'temperature': 1.0,
    'ensemble_size': 5,
    'uncertainty_threshold': 0.5
}

def get_default_config(component: str = 'policy'):
    """
    Get default configuration for policy components.
    
    Args:
        component: Component name ('policy', 'rank_predictor', 'uncertainty')
        
    Returns:
        dict: Default configuration
    """
    configs = {
        'policy': DEFAULT_POLICY_CONFIG,
        'rank_predictor': DEFAULT_RANK_PREDICTOR_CONFIG,
        'uncertainty': DEFAULT_UNCERTAINTY_CONFIG
    }
    
    return configs.get(component, DEFAULT_POLICY_CONFIG)


def create_policy(policy_type: str, config: dict = None, **kwargs):
    """
    Factory function to create policy components.
    
    Args:
        policy_type: Policy type name
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Policy component instance
    """
    if config is None:
        config = get_default_config('policy')
    config.update(kwargs)
    
    if policy_type.lower() == 'adaptive':
        return AdaptivePolicy(**config)
    elif policy_type.lower() == 'hierarchical':
        return HierarchicalPolicy(**config)
    elif policy_type.lower() == 'task_aware':
        return TaskAwarePolicy(**config)
    elif policy_type.lower() == 'dynamic':
        return DynamicScalingPolicy(**config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def create_rank_predictor(predictor_type: str, config: dict = None, **kwargs):
    """
    Factory function to create rank predictors.
    
    Args:
        predictor_type: Predictor type name
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Rank predictor instance
    """
    if config is None:
        config = get_default_config('rank_predictor')
    config.update(kwargs)
    
    if predictor_type.lower() == 'attention':
        return AttentionRankPredictor(**config)
    elif predictor_type.lower() == 'mlp':
        return MLPRankPredictor(**config)
    elif predictor_type.lower() == 'ensemble':
        return EnsembleRankPredictor(**config)
    else:
        raise ValueError(f"Unknown rank predictor type: {predictor_type}")


def create_uncertainty_estimator(estimator_type: str, config: dict = None, **kwargs):
    """
    Factory function to create uncertainty estimators.
    
    Args:
        estimator_type: Estimator type name
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Uncertainty estimator instance
    """
    if config is None:
        config = get_default_config('uncertainty')
    config.update(kwargs)
    
    if estimator_type.lower() == 'bayesian':
        return BayesianUncertaintyEstimator(**config)
    elif estimator_type.lower() == 'ensemble':
        return EnsembleUncertaintyEstimator(**config)
    elif estimator_type.lower() == 'mc_dropout':
        return MCDropoutUncertainty(**config)
    else:
        raise ValueError(f"Unknown uncertainty estimator type: {estimator_type}")


# Utility functions
def compute_policy_entropy(probabilities):
    """Compute entropy of policy probabilities."""
    import torch
    import torch.nn.functional as F
    
    log_probs = torch.log(probabilities + 1e-8)
    entropy = -torch.sum(probabilities * log_probs, dim=-1)
    return entropy


def apply_temperature_scaling(logits, temperature=1.0):
    """Apply temperature scaling to logits."""
    import torch
    
    return logits / temperature


def compute_confidence_interval(values, confidence=0.95):
    """Compute confidence interval for a set of values."""
    import numpy as np
    from scipy import stats
    
    alpha = 1 - confidence
    n = len(values)
    mean = np.mean(values)
    std_err = stats.sem(values)
    
    # t-distribution for small samples
    t_val = stats.t.ppf(1 - alpha/2, n-1)
    margin_error = t_val * std_err
    
    return {
        'mean': mean,
        'lower': mean - margin_error,
        'upper': mean + margin_error,
        'std_err': std_err
    }