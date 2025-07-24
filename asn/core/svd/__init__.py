"""
SVD module for AdaptiveScale Networks.

This module provides comprehensive singular value decomposition capabilities
for neural network compression and adaptation, including:

- Multi-scale SVD decomposition
- Hierarchical parameter compression
- Adaptive rank selection
- Efficient matrix operations
- Compression analysis and utilities

Main Components:
- MultiScaleSVDDecomposer: Main decomposition engine
- SVDLayer: SVD-decomposed neural network layer
- SVDConfig: Configuration for SVD operations
- Utility functions for analysis and optimization
"""

from .decomposer import (
    MultiScaleSVDDecomposer,
    SVDLayer,
    SVDConfig,
    randomized_svd,
    compute_svd_rank_for_compression,
    energy_based_rank_selection,
    batch_svd_decomposition
)

from .utils import (
    analyze_singular_values,
    compute_compression_metrics,
    find_optimal_rank,
    benchmark_svd_methods,
    randomized_svd as utils_randomized_svd,
    adaptive_randomized_svd,
    hierarchical_svd,
    progressive_svd_update,
    svd_based_initialization,
    compute_svd_gradient,
    validate_svd_decomposition,
    create_svd_summary_report
)

__all__ = [
    # Main decomposer classes
    'MultiScaleSVDDecomposer',
    'SVDLayer',
    'SVDConfig',
    
    # Core decomposition functions
    'randomized_svd',
    'compute_svd_rank_for_compression', 
    'energy_based_rank_selection',
    'batch_svd_decomposition',
    
    # Analysis utilities
    'analyze_singular_values',
    'compute_compression_metrics',
    'find_optimal_rank',
    'benchmark_svd_methods',
    
    # Advanced SVD methods
    'adaptive_randomized_svd',
    'hierarchical_svd',
    'progressive_svd_update',
    'svd_based_initialization',
    
    # Gradient and validation utilities
    'compute_svd_gradient',
    'validate_svd_decomposition',
    'create_svd_summary_report',
]

# Version information
__version__ = "1.0.0"

# Default configuration
DEFAULT_SVD_CONFIG = SVDConfig()

def get_default_config():
    """Get default SVD configuration."""
    return SVDConfig()

def create_decomposer(config=None, **kwargs):
    """
    Factory function to create SVD decomposer.
    
    Args:
        config: SVDConfig instance (optional)
        **kwargs: Configuration parameters
        
    Returns:
        MultiScaleSVDDecomposer instance
    """
    if config is None:
        config = SVDConfig(**kwargs)
    elif kwargs:
        # Update config with provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return MultiScaleSVDDecomposer(config)

def quick_decompose(model, scales=None, target_layers=None):
    """
    Quick SVD decomposition with sensible defaults.
    
    Args:
        model: PyTorch model to decompose
        scales: Compression scales (default: [0.3, 0.5, 0.7])
        target_layers: Target layer patterns (default: common patterns)
        
    Returns:
        Decomposition results
    """
    config = SVDConfig()
    
    if scales is not None:
        config.svd_scales = scales
    else:
        config.svd_scales = [0.3, 0.5, 0.7]  # Conservative defaults
    
    if target_layers is not None:
        config.target_layers = target_layers
    
    decomposer = MultiScaleSVDDecomposer(config)
    return decomposer.decompose_model(model)

def compress_model(model, compression_ratio=0.5, preserve_energy=0.95):
    """
    Compress model with target compression ratio.
    
    Args:
        model: PyTorch model
        compression_ratio: Target compression ratio
        preserve_energy: Energy preservation ratio
        
    Returns:
        Compressed model and statistics
    """
    config = SVDConfig(
        target_compression_ratio=compression_ratio,
        preserve_energy_ratio=preserve_energy,
        adaptive_rank_selection=True,
        rank_selection_metric="reconstruction_error"
    )
    
    decomposer = MultiScaleSVDDecomposer(config)
    results = decomposer.decompose_model(model)
    
    # Apply best compression scale
    best_scale = min(config.svd_scales, key=lambda s: abs(
        results['compression_stats'][s].get('avg_compression_ratio', 1.0) - compression_ratio
    ))
    
    compressed_model = decomposer.apply_decomposition(
        model, results['decomposed_weights'], best_scale
    )
    
    return compressed_model, results

def analyze_model_compressibility(model):
    """
    Analyze how compressible a model is using SVD.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Analysis results
    """
    import torch
    from collections import defaultdict
    
    analysis = defaultdict(list)
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.data
            
            # Analyze singular values
            sv_analysis = analyze_singular_values(weight)
            
            analysis['layer_names'].append(name)
            analysis['shapes'].append(weight.shape)
            analysis['param_counts'].append(weight.numel())
            analysis['condition_numbers'].append(sv_analysis['condition_number'])
            analysis['effective_ranks_95'].append(sv_analysis['effective_rank_95'])
            analysis['effective_ranks_99'].append(sv_analysis['effective_rank_99'])
            analysis['decay_rates'].append(sv_analysis['decay_rate'])
    
    # Compute summary statistics
    if analysis['param_counts']:
        total_params = sum(analysis['param_counts'])
        avg_condition = sum(analysis['condition_numbers']) / len(analysis['condition_numbers'])
        avg_decay_rate = sum(analysis['decay_rates']) / len(analysis['decay_rates'])
        
        # Estimate potential compression
        avg_effective_rank_95 = sum(analysis['effective_ranks_95']) / len(analysis['effective_ranks_95'])
        total_effective_params_95 = sum(
            min(shape) * avg_effective_rank_95 / min(shape) * (shape[0] + shape[1])
            for shape in analysis['shapes']
        )
        potential_compression_95 = total_effective_params_95 / total_params
        
        summary = {
            'total_parameters': total_params,
            'num_linear_layers': len(analysis['layer_names']),
            'average_condition_number': avg_condition,
            'average_decay_rate': avg_decay_rate,
            'average_effective_rank_95': avg_effective_rank_95,
            'potential_compression_ratio_95': potential_compression_95,
            'compressibility_score': min(1.0, avg_decay_rate * (1.0 - potential_compression_95))
        }
    else:
        summary = {
            'total_parameters': 0,
            'num_linear_layers': 0,
            'compressibility_score': 0.0
        }
    
    return {
        'layer_analysis': dict(analysis),
        'summary': summary
    }

def benchmark_compression_methods(model, test_scales=None):
    """
    Benchmark different compression methods and scales.
    
    Args:
        model: PyTorch model to benchmark
        test_scales: Scales to test (default: [0.1, 0.3, 0.5, 0.7, 0.9])
        
    Returns:
        Benchmark results
    """
    if test_scales is None:
        test_scales = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    results = {}
    
    # Test each scale
    for scale in test_scales:
        config = SVDConfig(
            svd_scales=[scale],
            adaptive_rank_selection=True,
            rank_selection_metric="reconstruction_error"
        )
        
        decomposer = MultiScaleSVDDecomposer(config)
        decomp_results = decomposer.decompose_model(model)
        
        # Extract key metrics
        if scale in decomp_results['compression_stats']:
            scale_stats = decomp_results['compression_stats'][scale]
            
            if scale_stats:
                avg_compression = sum(
                    stats['compression_ratio'] for stats in scale_stats.values()
                ) / len(scale_stats)
                
                total_original = sum(
                    stats['original_params'] for stats in scale_stats.values()
                )
                total_compressed = sum(
                    stats['compressed_params'] for stats in scale_stats.values()
                )
                
                results[scale] = {
                    'average_compression_ratio': avg_compression,
                    'total_compression_ratio': total_compressed / total_original,
                    'memory_savings_mb': (total_original - total_compressed) * 4 / (1024 * 1024),
                    'num_layers_compressed': len(scale_stats)
                }
    
    return results

# Convenience functions for common use cases
def compress_transformer(model, compression_ratio=0.5):
    """Compress transformer model with appropriate settings."""
    config = SVDConfig(
        target_layers=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        svd_scales=[compression_ratio],
        adaptive_rank_selection=True,
        rank_selection_metric="energy",
        preserve_energy_ratio=0.95
    )
    
    decomposer = MultiScaleSVDDecomposer(config)
    results = decomposer.decompose_model(model)
    compressed_model = decomposer.apply_decomposition(
        model, results['decomposed_weights'], compression_ratio
    )
    
    return compressed_model, results

def compress_gpt2(model, compression_ratio=0.5):
    """Compress GPT-2 model with GPT-2 specific layer patterns."""
    config = SVDConfig(
        target_layers=["c_attn", "c_proj", "c_fc"],
        svd_scales=[compression_ratio],
        adaptive_rank_selection=True,
        rank_selection_metric="reconstruction_error",
        preserve_energy_ratio=0.95,
        max_rank_ratio=0.8
    )
    
    decomposer = MultiScaleSVDDecomposer(config)
    results = decomposer.decompose_model(model)
    compressed_model = decomposer.apply_decomposition(
        model, results['decomposed_weights'], compression_ratio
    )
    
    return compressed_model, results

def compress_llama(model, compression_ratio=0.5):
    """Compress LLaMA model with LLaMA specific layer patterns."""
    config = SVDConfig(
        target_layers=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        svd_scales=[compression_ratio],
        adaptive_rank_selection=True,
        rank_selection_metric="energy",
        preserve_energy_ratio=0.96,  # Higher preservation for LLaMA
        max_rank_ratio=0.7
    )
    
    decomposer = MultiScaleSVDDecomposer(config)
    results = decomposer.decompose_model(model)
    compressed_model = decomposer.apply_decomposition(
        model, results['decomposed_weights'], compression_ratio
    )
    
    return compressed_model, results

# Model-specific configurations
MODEL_CONFIGS = {
    'gpt2': {
        'target_layers': ["c_attn", "c_proj", "c_fc"],
        'max_rank_ratio': 0.8,
        'preserve_energy_ratio': 0.95
    },
    'llama': {
        'target_layers': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        'max_rank_ratio': 0.7,
        'preserve_energy_ratio': 0.96
    },
    'bert': {
        'target_layers': ["query", "key", "value", "dense"],
        'max_rank_ratio': 0.8,
        'preserve_energy_ratio': 0.95
    },
    'transformer': {
        'target_layers': ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
        'max_rank_ratio': 0.8,
        'preserve_energy_ratio': 0.95
    }
}

def get_model_config(model_type):
    """Get SVD configuration for specific model type."""
    if model_type.lower() in MODEL_CONFIGS:
        base_config = SVDConfig()
        model_specific = MODEL_CONFIGS[model_type.lower()]
        
        for key, value in model_specific.items():
            setattr(base_config, key, value)
        
        return base_config
    else:
        return SVDConfig()  # Default configuration

def create_compression_report(model, results, output_path=None):
    """
    Create a detailed compression report.
    
    Args:
        model: Original model
        results: Decomposition results
        output_path: Path to save report
        
    Returns:
        Report string
    """
    return create_svd_summary_report(results, output_path)

# Validation utilities
def verify_compressed_model(original_model, compressed_model, test_input):
    """
    Verify that compressed model produces similar outputs to original.
    
    Args:
        original_model: Original model
        compressed_model: Compressed model
        test_input: Test input tensor
        
    Returns:
        Verification results
    """
    import torch
    
    original_model.eval()
    compressed_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        compressed_output = compressed_model(test_input)
        
        # Compute differences
        if isinstance(original_output, torch.Tensor):
            mse = torch.nn.functional.mse_loss(compressed_output, original_output)
            cosine_sim = torch.nn.functional.cosine_similarity(
                compressed_output.flatten(), original_output.flatten(), dim=0
            )
            max_diff = torch.max(torch.abs(compressed_output - original_output))
            
            return {
                'mse_loss': mse.item(),
                'cosine_similarity': cosine_sim.item(),
                'max_absolute_difference': max_diff.item(),
                'relative_error': (mse / torch.var(original_output)).item()
            }
        else:
            # Handle complex outputs (like transformer outputs with multiple tensors)
            return {'message': 'Complex output structure - manual verification needed'}

# Export key functions for easy access
__all__.extend([
    'get_default_config',
    'create_decomposer', 
    'quick_decompose',
    'compress_model',
    'analyze_model_compressibility',
    'benchmark_compression_methods',
    'compress_transformer',
    'compress_gpt2', 
    'compress_llama',
    'get_model_config',
    'create_compression_report',
    'verify_compressed_model'
])

# Logging setup
import logging
logger = logging.getLogger(__name__)
logger.info("AdaptiveScale Networks SVD module initialized")

# Check for optional dependencies
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available - plotting functions will be disabled")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn not available - advanced plotting will be disabled")

# Module metadata
__author__ = "EllanorAI DeepLearning Research Team"
__email__ = "architsood@ellanorai.org"
__status__ = "Research"