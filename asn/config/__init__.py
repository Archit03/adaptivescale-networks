"""
Configuration module for AdaptiveScale Networks.

This module provides configuration classes and utilities for setting up
ASN experiments, including base configurations, model-specific configs,
and experiment configurations.
"""

from .base_config import ASNConfig
from .model_configs import (
    GPT2Config,
    LlamaConfig,
    DialoGPTConfig,
    get_model_config
)
from .experiment_configs import (
    ExperimentConfig,
    BaselineConfig,
    AblationConfig,
    ScalingConfig,
    create_experiment_config
)

__all__ = [
    # Base configuration
    'ASNConfig',
    
    # Model configurations
    'GPT2Config',
    'LlamaConfig', 
    'DialoGPTConfig',
    'get_model_config',
    
    # Experiment configurations
    'ExperimentConfig',
    'BaselineConfig',
    'AblationConfig',
    'ScalingConfig',
    'create_experiment_config',
]

# Version information
__version__ = "1.0.0"

# Default configuration
DEFAULT_CONFIG = ASNConfig()

def get_default_config():
    """Get default ASN configuration"""
    return ASNConfig()

def load_config(config_path: str):
    """Load configuration from file"""
    return ASNConfig.from_yaml(config_path)

def create_config(model_name: str = "gpt2", **kwargs):
    """Create configuration with model-specific defaults"""
    if model_name.startswith("gpt2"):
        base_config = GPT2Config()
    elif "llama" in model_name.lower():
        base_config = LlamaConfig()
    elif "dialogpt" in model_name.lower():
        base_config = DialoGPTConfig()
    else:
        base_config = ASNConfig()
    
    # Update with provided kwargs
    base_config.update(**kwargs)
    
    return base_config