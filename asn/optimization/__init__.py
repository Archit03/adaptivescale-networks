"""
Optimization module for AdaptiveScale Networks.

This module provides advanced optimization techniques including specialized
optimizers, learning rate schedulers, and the Cross-Entropy Method (CEM)
for few-shot inference optimization. It focuses on optimization strategies
tailored for adaptive neural network scaling scenarios.

Components:
- Advanced optimizers (AdamW variants, LAMB, AdaBelief, etc.)
- Adaptive learning rate schedulers with warmup and cosine annealing
- Cross-Entropy Method (CEM) for parameter optimization
- Meta-learning optimizers for MAML and related algorithms
- Gradient clipping and normalization utilities
- Optimization monitoring and analysis tools
"""

from .optimizers import (
    # Core optimizer classes
    AdaptiveOptimizer,
    MetaOptimizer,
    CompressedOptimizer,
    
    # Specific optimizer implementations
    AdamWScheduleFree,
    LAMBOptimizer,
    AdaBeliefOptimizer,
    RAdamOptimizer,
    LookaheadOptimizer,
    SAMOptimizer,
    
    # Factory functions
    create_optimizer,
    get_optimizer_config,
    
    # Utilities
    compute_gradient_norm,
    apply_gradient_clipping,
    get_parameter_stats
)

from .schedulers import (
    # Scheduler classes
    AdaptiveScheduler,
    WarmupScheduler,
    CosineAnnealingScheduler,
    PolynomialDecayScheduler,
    ExponentialDecayScheduler,
    CyclicalScheduler,
    OneCycleScheduler,
    
    # Meta-learning schedulers
    MetaLearningScheduler,
    InnerLoopScheduler,
    
    # Factory functions
    create_scheduler,
    get_scheduler_config,
    
    # Utilities
    plot_schedule,
    analyze_schedule_convergence
)

from .cem import (
    # CEM implementation
    CrossEntropyMethod,
    CEMConfig,
    CEMResults,
    
    # CEM variants
    AdaptiveCEM,
    NoisyCEM,
    EliteCEM,
    
    # Parameter distributions
    ParameterDistribution,
    GaussianDistribution,
    CategoricalDistribution,
    MixtureDistribution,
    
    # Utilities
    run_cem_optimization,
    create_parameter_distribution,
    analyze_cem_convergence
)

__all__ = [
    # Optimizers
    'AdaptiveOptimizer',
    'MetaOptimizer', 
    'CompressedOptimizer',
    'AdamWScheduleFree',
    'LAMBOptimizer',
    'AdaBeliefOptimizer',
    'RAdamOptimizer',
    'LookaheadOptimizer',
    'SAMOptimizer',
    'create_optimizer',
    'get_optimizer_config',
    'compute_gradient_norm',
    'apply_gradient_clipping',
    'get_parameter_stats',
    
    # Schedulers
    'AdaptiveScheduler',
    'WarmupScheduler',
    'CosineAnnealingScheduler',
    'PolynomialDecayScheduler',
    'ExponentialDecayScheduler',
    'CyclicalScheduler',
    'OneCycleScheduler',
    'MetaLearningScheduler',
    'InnerLoopScheduler',
    'create_scheduler',
    'get_scheduler_config',
    'plot_schedule',
    'analyze_schedule_convergence',
    
    # CEM
    'CrossEntropyMethod',
    'CEMConfig',
    'CEMResults',
    'AdaptiveCEM',
    'NoisyCEM',
    'EliteCEM',
    'ParameterDistribution',
    'GaussianDistribution',
    'CategoricalDistribution',
    'MixtureDistribution',
    'run_cem_optimization',
    'create_parameter_distribution',
    'analyze_cem_convergence',
]

# Version information
__version__ = "1.0.0"

# Default configurations
DEFAULT_OPTIMIZER_CONFIG = {
    'type': 'adamw',
    'lr': 1e-3,
    'weight_decay': 0.01,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'amsgrad': False,
    'gradient_clipping': 1.0
}

DEFAULT_SCHEDULER_CONFIG = {
    'type': 'cosine_annealing',
    'warmup_steps': 1000,
    'max_steps': 10000,
    'min_lr': 1e-6,
    'num_cycles': 1
}

DEFAULT_CEM_CONFIG = {
    'population_size': 100,
    'elite_ratio': 0.2,
    'num_iterations': 50,
    'noise_std': 0.1,
    'convergence_threshold': 1e-6
}

def create_optimization_suite(model, config: dict = None):
    """
    Create a complete optimization suite with optimizer, scheduler, and CEM.
    
    Args:
        model: Model to optimize
        config: Configuration dictionary
        
    Returns:
        Dictionary containing optimizer, scheduler, and CEM instances
    """
    if config is None:
        config = {}
    
    # Create optimizer
    optimizer_config = config.get('optimizer', DEFAULT_OPTIMIZER_CONFIG)
    optimizer = create_optimizer(model.parameters(), optimizer_config)
    
    # Create scheduler
    scheduler_config = config.get('scheduler', DEFAULT_SCHEDULER_CONFIG)
    scheduler = create_scheduler(optimizer, scheduler_config)
    
    # Create CEM for few-shot optimization
    cem_config = config.get('cem', DEFAULT_CEM_CONFIG)
    cem = CrossEntropyMethod(cem_config)
    
    return {
        'optimizer': optimizer,
        'scheduler': scheduler,
        'cem': cem,
        'config': config
    }

def get_default_config(component: str = 'optimizer'):
    """
    Get default configuration for optimization components.
    
    Args:
        component: Component type ('optimizer', 'scheduler', 'cem')
        
    Returns:
        Default configuration dictionary
    """
    configs = {
        'optimizer': DEFAULT_OPTIMIZER_CONFIG,
        'scheduler': DEFAULT_SCHEDULER_CONFIG,
        'cem': DEFAULT_CEM_CONFIG
    }
    
    return configs.get(component, DEFAULT_OPTIMIZER_CONFIG)

# Module logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"AdaptiveScale Networks Optimization module v{__version__} loaded")
logger.info("Available components: Optimizers, Schedulers, Cross-Entropy Method")

# Performance recommendations
try:
    import torch
    if torch.cuda.is_available():
        logger.info("CUDA available - consider using GPU-optimized optimization strategies")
    
    # Check for mixed precision support
    if hasattr(torch.cuda, 'amp'):
        logger.info("Mixed precision available - enable for faster training")
        
except ImportError:
    logger.warning("PyTorch not available - optimization module may have limited functionality")

# Module health check
def health_check():
    """Perform health check of optimization module."""
    try:
        # Test basic functionality
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # Test factory functions
        test_params = [torch.tensor([1.0], requires_grad=True)]
        optimizer = create_optimizer(test_params, DEFAULT_OPTIMIZER_CONFIG)
        scheduler = create_scheduler(optimizer, DEFAULT_SCHEDULER_CONFIG)
        
        logger.info("Optimization module health check: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Optimization module health check: FAILED - {e}")
        return False

if health_check():
    logger.info("Optimization module ready for use")
else:
    logger.warning("Optimization module initialized with warnings")