"""
Model-specific configurations for AdaptiveScale Networks.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import torch
from .base_config import ASNConfig


@dataclass
class GPT2Config(ASNConfig):
    """Configuration optimized for GPT-2 models"""
    
    # Model settings
    model_name: str = "gpt2"
    max_length: int = 512
    max_new_tokens: int = 64
    
    # SVD settings optimized for GPT-2
    svd_scales: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    target_layers: List[str] = field(default_factory=lambda: ["c_fc", "c_proj", "c_attn"])
    max_rank_ratio: float = 0.7
    use_randomized_svd: bool = True
    
    # Training settings
    batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 3e-5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 2
    
    # Memory settings
    gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    max_memory_gb: float = 8.0
    
    # Progressive training stages
    progressive_stages: List[Dict] = field(default_factory=lambda: [
        {'tasks': ['math'], 'iterations': 100, 'lr_scale': 1.0},
        {'tasks': ['math', 'qa'], 'iterations': 100, 'lr_scale': 0.8},
        {'tasks': ['math', 'qa', 'reasoning'], 'iterations': 150, 'lr_scale': 0.6},
        {'tasks': ['math', 'qa', 'reasoning', 'code'], 'iterations': 150, 'lr_scale': 0.4}
    ])


@dataclass
class GPT2MediumConfig(GPT2Config):
    """Configuration for GPT-2 Medium model"""
    
    model_name: str = "gpt2-medium"
    batch_size: int = 2
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_memory_gb: float = 12.0
    
    # More conservative SVD settings for larger model
    svd_scales: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    max_rank_ratio: float = 0.7


@dataclass
class GPT2LargeConfig(GPT2Config):
    """Configuration for GPT-2 Large model"""
    
    model_name: str = "gpt2-large"
    batch_size: int = 1
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_memory_gb: float = 16.0
    cpu_offload: bool = True
    
    # Conservative SVD settings for large model
    svd_scales: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    max_rank_ratio: float = 0.6


@dataclass
class DialoGPTConfig(ASNConfig):
    """Configuration optimized for DialoGPT models"""
    
    # Model settings
    model_name: str = "microsoft/DialoGPT-medium"
    max_length: int = 512
    max_new_tokens: int = 128
    
    # DialoGPT specific - disable mixed precision due to known issues
    use_mixed_precision: bool = False
    dtype: torch.dtype = torch.float32
    
    # SVD settings
    svd_scales: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9])
    target_layers: List[str] = field(default_factory=lambda: ["c_fc", "c_proj", "c_attn"])
    max_rank_ratio: float = 0.8
    
    # Training settings
    batch_size: int = 2
    eval_batch_size: int = 4
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.15
    
    # Task focus - DialoGPT is better for conversational tasks
    task_types: List[str] = field(default_factory=lambda: ['qa', 'reasoning', 'code'])
    
    # Progressive training adapted for dialogue model
    progressive_stages: List[Dict] = field(default_factory=lambda: [
        {'tasks': ['qa'], 'iterations': 150, 'lr_scale': 1.0},
        {'tasks': ['qa', 'reasoning'], 'iterations': 200, 'lr_scale': 0.8},
        {'tasks': ['qa', 'reasoning', 'code'], 'iterations': 150, 'lr_scale': 0.6}
    ])


@dataclass
class LlamaConfig(ASNConfig):
    """Configuration optimized for LLaMA models"""
    
    # Model settings
    model_name: str = "meta-llama/Llama-2-7b-hf"
    max_length: int = 2048
    max_new_tokens: int = 256
    
    # SVD settings for transformer architecture
    svd_scales: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    target_layers: List[str] = field(default_factory=lambda: ["gate_proj", "up_proj", "down_proj", "q_proj", "k_proj", "v_proj", "o_proj"])
    max_rank_ratio: float = 0.7
    use_randomized_svd: bool = True
    
    # Training settings for larger model
    batch_size: int = 1
    eval_batch_size: int = 2
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 8
    
    # Memory management for large model
    gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    max_memory_gb: float = 24.0
    cpu_offload: bool = True
    
    # Extended training for capability
    num_iterations: int = 800
    progressive_stages: List[Dict] = field(default_factory=lambda: [
        {'tasks': ['qa'], 'iterations': 200, 'lr_scale': 1.0},
        {'tasks': ['qa', 'reasoning'], 'iterations': 200, 'lr_scale': 0.8},
        {'tasks': ['qa', 'reasoning', 'math'], 'iterations': 200, 'lr_scale': 0.6},
        {'tasks': ['qa', 'reasoning', 'math', 'code'], 'iterations': 200, 'lr_scale': 0.4}
    ])


@dataclass
class CodeLlamaConfig(LlamaConfig):
    """Configuration optimized for Code Llama models"""
    
    model_name: str = "codellama/CodeLlama-7b-Python-hf"
    
    # Code-specific settings
    max_length: int = 4096
    max_new_tokens: int = 512
    
    # Focus on code and reasoning tasks
    task_types: List[str] = field(default_factory=lambda: ['code', 'reasoning', 'math'])
    
    # Progressive training focused on code
    progressive_stages: List[Dict] = field(default_factory=lambda: [
        {'tasks': ['code'], 'iterations': 200, 'lr_scale': 1.0},
        {'tasks': ['code', 'math'], 'iterations': 200, 'lr_scale': 0.8},
        {'tasks': ['code', 'math', 'reasoning'], 'iterations': 200, 'lr_scale': 0.6}
    ])


@dataclass
class T5Config(ASNConfig):
    """Configuration optimized for T5 models"""
    
    # Model settings
    model_name: str = "t5-base"
    max_length: int = 512
    max_new_tokens: int = 128
    
    # T5-specific layer names
    target_layers: List[str] = field(default_factory=lambda: ["wi", "wo", "q", "k", "v", "o"])
    
    # SVD settings
    svd_scales: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    max_rank_ratio: float = 0.8
    
    # Training settings
    batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.1


# Model configuration registry
MODEL_CONFIGS = {
    "gpt2": GPT2Config,
    "gpt2-medium": GPT2MediumConfig,
    "gpt2-large": GPT2LargeConfig,
    "gpt2-xl": GPT2LargeConfig,  # Use large config for XL
    "microsoft/DialoGPT-small": DialoGPTConfig,
    "microsoft/DialoGPT-medium": DialoGPTConfig,
    "microsoft/DialoGPT-large": DialoGPTConfig,
    "meta-llama/Llama-2-7b-hf": LlamaConfig,
    "meta-llama/Llama-2-13b-hf": LlamaConfig,
    "codellama/CodeLlama-7b-Python-hf": CodeLlamaConfig,
    "codellama/CodeLlama-13b-Python-hf": CodeLlamaConfig,
    "t5-small": T5Config,
    "t5-base": T5Config,
    "t5-large": T5Config,
}


def get_model_config(model_name: str, **kwargs) -> ASNConfig:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        **kwargs: Additional configuration overrides
        
    Returns:
        ASNConfig: Model-specific configuration
    """
    # Find exact match first
    if model_name in MODEL_CONFIGS:
        config_class = MODEL_CONFIGS[model_name]
    else:
        # Try partial matching
        config_class = None
        for registered_name, registered_config in MODEL_CONFIGS.items():
            if registered_name in model_name or model_name in registered_name:
                config_class = registered_config
                break
        
        # Default to base config if no match found
        if config_class is None:
            config_class = ASNConfig
            
    # Create config instance
    config = config_class()
    
    # Override model name
    config.model_name = model_name
    
    # Apply additional overrides
    config.update(**kwargs)
    
    return config


def list_available_models() -> List[str]:
    """List all available model configurations"""
    return list(MODEL_CONFIGS.keys())


def register_model_config(model_name: str, config_class: type):
    """Register a new model configuration"""
    MODEL_CONFIGS[model_name] = config_class


# Utility functions for common model families
def is_gpt2_model(model_name: str) -> bool:
    """Check if model is from GPT-2 family"""
    return "gpt2" in model_name.lower()


def is_llama_model(model_name: str) -> bool:
    """Check if model is from LLaMA family"""
    return "llama" in model_name.lower()


def is_dialogpt_model(model_name: str) -> bool:
    """Check if model is from DialoGPT family"""
    return "dialogpt" in model_name.lower()


def is_t5_model(model_name: str) -> bool:
    """Check if model is from T5 family"""
    return "t5" in model_name.lower()


def get_model_family(model_name: str) -> str:
    """Get the model family name"""
    if is_gpt2_model(model_name):
        return "gpt2"
    elif is_llama_model(model_name):
        return "llama"
    elif is_dialogpt_model(model_name):
        return "dialogpt"
    elif is_t5_model(model_name):
        return "t5"
    else:
        return "unknown"