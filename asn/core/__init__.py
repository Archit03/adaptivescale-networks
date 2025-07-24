"""
Core module for AdaptiveScale Networks.

This module provides the fundamental components for adaptive neural network scaling,
including SVD decomposition, adaptive policies, meta-learning, and uncertainty estimation.

Main Components:
- SVD: Multi-scale singular value decomposition for parameter compression
- Policy: Adaptive policies for intelligent scaling decisions  
- Meta-Learning: MAML and continual learning for fast adaptation
- Uncertainty: Bayesian and ensemble methods for uncertainty quantification

The core module serves as the foundation for the entire ASN framework,
providing the essential building blocks for adaptive neural network scaling.
"""

import logging
from typing import Dict, List, Optional, Any, Union

# Import core submodules
from . import svd
from . import policy  
from . import meta_learning

# Import key classes and functions from submodules
from .svd import (
    MultiScaleSVDDecomposer,
    SVDLayer,
    SVDConfig,
    compress_model,
    analyze_model_compressibility,
    quick_decompose
)

from .policy import (
    AdaptivePolicy,
    HierarchicalPolicy,
    TaskAwarePolicy,
    DynamicScalingPolicy,
    PolicyConfig,
    RankPredictor,
    AttentionRankPredictor,
    RankPredictorConfig,
    UncertaintyEstimator,
    BayesianUncertaintyEstimator,
    UncertaintyConfig
)

from .meta_learning import (
    MAMLTrainer,
    MAMLConfig,
    ContinualLearner,
    EWCLoss,
    ElasticWeightConsolidation,
    ContinualConfig
)

__all__ = [
    # Submodules
    'svd',
    'policy',
    'meta_learning',
    
    # SVD Components
    'MultiScaleSVDDecomposer',
    'SVDLayer', 
    'SVDConfig',
    'compress_model',
    'analyze_model_compressibility',
    'quick_decompose',
    
    # Policy Components
    'AdaptivePolicy',
    'HierarchicalPolicy',
    'TaskAwarePolicy',
    'DynamicScalingPolicy',
    'PolicyConfig',
    'RankPredictor',
    'AttentionRankPredictor', 
    'RankPredictorConfig',
    'UncertaintyEstimator',
    'BayesianUncertaintyEstimator',
    'UncertaintyConfig',
    
    # Meta-Learning Components
    'MAMLTrainer',
    'MAMLConfig',
    'ContinualLearner',
    'EWCLoss',
    'ElasticWeightConsolidation',
    'ContinualConfig',
    
    # Factory functions
    'create_adaptive_system',
    'create_svd_decomposer',
    'create_policy_network',
    'create_meta_learner',
    'create_uncertainty_estimator',
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "ASN Research Team"
__email__ = "research@asn.ai"

# Set up logging
logger = logging.getLogger(__name__)

# Default configurations for integrated systems
DEFAULT_ADAPTIVE_CONFIG = {
    'svd_config': {
        'svd_scales': [0.1, 0.3, 0.5, 0.7, 0.9],
        'target_layers': ["c_fc", "c_proj", "c_attn"],
        'adaptive_rank_selection': True,
        'rank_selection_metric': 'reconstruction_error',
        'preserve_energy_ratio': 0.95
    },
    'policy_config': {
        'input_dim': 128,
        'hidden_dim': 256,
        'num_layers': 3,
        'use_attention': True,
        'temperature': 1.0
    },
    'meta_learning_config': {
        'inner_lr': 0.01,
        'meta_lr': 0.001,
        'num_inner_steps': 5,
        'first_order': False
    },
    'uncertainty_config': {
        'mc_dropout_samples': 10,
        'ensemble_size': 5,
        'temperature': 1.0
    }
}


class AdaptiveScaleCore:
    """
    Core adaptive scaling system that integrates all components.
    
    This class serves as the main interface for the AdaptiveScale Networks
    framework, coordinating between SVD decomposition, adaptive policies,
    meta-learning, and uncertainty estimation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive scaling core.
        
        Args:
            config: Configuration dictionary for all components
        """
        if config is None:
            config = DEFAULT_ADAPTIVE_CONFIG
        
        self.config = config
        self.device = self._get_device()
        
        # Initialize components
        self.svd_decomposer = None
        self.policy_network = None
        self.meta_learner = None
        self.uncertainty_estimator = None
        
        # System state
        self.is_initialized = False
        self.current_model = None
        self.decomposition_cache = {}
        
        logger.info("AdaptiveScaleCore initialized")
    
    def _get_device(self):
        """Get the appropriate device for computation."""
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def initialize_components(self, model=None):
        """
        Initialize all core components.
        
        Args:
            model: Optional model to analyze for component initialization
        """
        logger.info("Initializing core components...")
        
        # Initialize SVD decomposer
        svd_config = SVDConfig(**self.config.get('svd_config', {}))
        self.svd_decomposer = MultiScaleSVDDecomposer(svd_config)
        
        # Initialize policy network
        policy_config = PolicyConfig(**self.config.get('policy_config', {}))
        self.policy_network = AdaptivePolicy(policy_config)
        
        # Initialize meta-learner
        if model is not None:
            maml_config = MAMLConfig(**self.config.get('meta_learning_config', {}))
            self.meta_learner = MAMLTrainer(model, maml_config)
        
        # Initialize uncertainty estimator
        uncertainty_config = UncertaintyConfig(**self.config.get('uncertainty_config', {}))
        self.uncertainty_estimator = UncertaintyEstimator('bayesian', uncertainty_config)
        
        self.is_initialized = True
        logger.info("All core components initialized successfully")
    
    def decompose_model(self, model, target_scale: float = None):
        """
        Decompose model using SVD.
        
        Args:
            model: PyTorch model to decompose
            target_scale: Target compression scale
            
        Returns:
            Decomposition results
        """
        if not self.svd_decomposer:
            self.initialize_components(model)
        
        self.current_model = model
        results = self.svd_decomposer.decompose_model(model, target_scale)
        
        # Cache results
        cache_key = id(model)
        self.decomposition_cache[cache_key] = results
        
        return results
    
    def make_scaling_decision(self, layer_info, context=None):
        """
        Make intelligent scaling decisions using policy network.
        
        Args:
            layer_info: Information about the layer
            context: Additional context for decision making
            
        Returns:
            Scaling decision
        """
        if not self.policy_network:
            self.initialize_components()
        
        import torch
        
        # Ensure layer_info is a tensor
        if not isinstance(layer_info, torch.Tensor):
            if isinstance(layer_info, (list, tuple)):
                layer_info = torch.tensor(layer_info, dtype=torch.float32)
            else:
                raise ValueError("layer_info must be a tensor or convertible to tensor")
        
        layer_info = layer_info.to(self.device)
        
        # Get policy decision
        with torch.no_grad():
            policy_output = self.policy_network(layer_info.unsqueeze(0), context)
            probabilities = self.policy_network.get_action_probabilities(layer_info.unsqueeze(0), context)
        
        return {
            'action_probabilities': probabilities.cpu(),
            'recommended_action': torch.argmax(probabilities, dim=-1).cpu().item(),
            'confidence': torch.max(probabilities).cpu().item(),
            'policy_values': policy_output['values'].cpu()
        }
    
    def adapt_to_task(self, support_data, query_data):
        """
        Adapt model to new task using meta-learning.
        
        Args:
            support_data: Support set for adaptation
            query_data: Query set for evaluation
            
        Returns:
            Adaptation results
        """
        if not self.meta_learner:
            raise RuntimeError("Meta-learner not initialized. Call initialize_components() first.")
        
        # Perform meta-learning adaptation
        metrics = self.meta_learner.meta_update([(support_data[0], support_data[1], 
                                                  query_data[0], query_data[1])])
        
        return metrics
    
    def estimate_uncertainty(self, inputs):
        """
        Estimate uncertainty for given inputs.
        
        Args:
            inputs: Input tensors for uncertainty estimation
            
        Returns:
            Uncertainty estimates
        """
        if not self.uncertainty_estimator:
            self.initialize_components()
        
        return self.uncertainty_estimator.predict_with_uncertainty(inputs, return_all=True)
    
    def get_system_status(self):
        """Get status of all core components."""
        return {
            'is_initialized': self.is_initialized,
            'has_svd_decomposer': self.svd_decomposer is not None,
            'has_policy_network': self.policy_network is not None,
            'has_meta_learner': self.meta_learner is not None,
            'has_uncertainty_estimator': self.uncertainty_estimator is not None,
            'current_model_loaded': self.current_model is not None,
            'decomposition_cache_size': len(self.decomposition_cache),
            'device': str(self.device)
        }
    
    def save_state(self, filepath: str):
        """Save the current state of all components."""
        import torch
        
        state = {
            'config': self.config,
            'is_initialized': self.is_initialized,
            'decomposition_cache': self.decomposition_cache
        }
        
        # Save component states if available
        if self.policy_network:
            state['policy_state'] = self.policy_network.state_dict()
        
        if self.uncertainty_estimator:
            state['uncertainty_state'] = self.uncertainty_estimator.estimator.state_dict()
        
        if self.meta_learner and hasattr(self.meta_learner, 'model'):
            state['meta_learner_state'] = self.meta_learner.model.state_dict()
        
        torch.save(state, filepath)
        logger.info(f"Core state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load previously saved state."""
        import torch
        
        state = torch.load(filepath, map_location=self.device)
        
        self.config = state['config']
        self.is_initialized = state['is_initialized']
        self.decomposition_cache = state.get('decomposition_cache', {})
        
        # Restore component states
        if self.is_initialized:
            self.initialize_components()
            
            if 'policy_state' in state and self.policy_network:
                self.policy_network.load_state_dict(state['policy_state'])
            
            if 'uncertainty_state' in state and self.uncertainty_estimator:
                self.uncertainty_estimator.estimator.load_state_dict(state['uncertainty_state'])
        
        logger.info(f"Core state loaded from {filepath}")


# Factory functions for easy component creation
def create_adaptive_system(config: Dict[str, Any] = None) -> AdaptiveScaleCore:
    """
    Factory function to create a complete adaptive scaling system.
    
    Args:
        config: Configuration for all components
        
    Returns:
        AdaptiveScaleCore instance
    """
    return AdaptiveScaleCore(config)


def create_svd_decomposer(scales: List[float] = None, 
                         target_layers: List[str] = None,
                         **kwargs) -> MultiScaleSVDDecomposer:
    """
    Factory function to create SVD decomposer.
    
    Args:
        scales: SVD compression scales
        target_layers: Target layer patterns
        **kwargs: Additional configuration
        
    Returns:
        MultiScaleSVDDecomposer instance
    """
    config = SVDConfig(**kwargs)
    
    if scales is not None:
        config.svd_scales = scales
    if target_layers is not None:
        config.target_layers = target_layers
    
    return MultiScaleSVDDecomposer(config)


def create_policy_network(policy_type: str = 'adaptive',
                         input_dim: int = 128,
                         hidden_dim: int = 256,
                         **kwargs) -> Union[AdaptivePolicy, HierarchicalPolicy, TaskAwarePolicy, DynamicScalingPolicy]:
    """
    Factory function to create policy networks.
    
    Args:
        policy_type: Type of policy ('adaptive', 'hierarchical', 'task_aware', 'dynamic')
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional configuration
        
    Returns:
        Policy network instance
    """
    config = PolicyConfig(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs)
    
    if policy_type.lower() == 'adaptive':
        return AdaptivePolicy(config)
    elif policy_type.lower() == 'hierarchical':
        return HierarchicalPolicy(config)
    elif policy_type.lower() == 'task_aware':
        return TaskAwarePolicy(config)
    elif policy_type.lower() == 'dynamic':
        return DynamicScalingPolicy(config)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def create_meta_learner(model, algorithm: str = 'maml', **kwargs) -> Union[MAMLTrainer, ContinualLearner]:
    """
    Factory function to create meta-learning algorithms.
    
    Args:
        model: Base model for meta-learning
        algorithm: Algorithm type ('maml', 'continual')
        **kwargs: Additional configuration
        
    Returns:
        Meta-learning algorithm instance
    """
    if algorithm.lower() == 'maml':
        config = MAMLConfig(**kwargs)
        return MAMLTrainer(model, config)
    elif algorithm.lower() == 'continual':
        config = ContinualConfig(**kwargs)
        return ContinualLearner(model, config)
    else:
        raise ValueError(f"Unknown meta-learning algorithm: {algorithm}")


def create_uncertainty_estimator(estimator_type: str = 'bayesian', **kwargs) -> UncertaintyEstimator:
    """
    Factory function to create uncertainty estimators.
    
    Args:
        estimator_type: Type of estimator ('bayesian', 'ensemble', 'mc_dropout')
        **kwargs: Additional configuration
        
    Returns:
        UncertaintyEstimator instance
    """
    config = UncertaintyConfig(**kwargs)
    return UncertaintyEstimator(estimator_type, config)


# Integrated workflows
def compress_and_adapt(model, compression_ratio: float = 0.5, 
                      task_data=None, **kwargs):
    """
    Integrated workflow: compress model and adapt to task.
    
    Args:
        model: Model to compress and adapt
        compression_ratio: Target compression ratio
        task_data: Optional task data for adaptation
        **kwargs: Additional configuration
        
    Returns:
        Compressed and adapted model with statistics
    """
    # Create core system
    core = create_adaptive_system()
    core.initialize_components(model)
    
    # Compress model
    decomp_results = core.decompose_model(model)
    
    # Find best scale for target compression
    best_scale = min(core.svd_decomposer.config.svd_scales, 
                    key=lambda s: abs(
                        decomp_results['compression_stats'][s].get('avg_compression_ratio', 1.0) - compression_ratio
                    ))
    
    compressed_model = core.svd_decomposer.apply_decomposition(
        model, decomp_results['decomposed_weights'], best_scale
    )
    
    # Adapt to task if provided
    adaptation_results = None
    if task_data is not None:
        support_data, query_data = task_data
        adaptation_results = core.adapt_to_task(support_data, query_data)
    
    return {
        'compressed_model': compressed_model,
        'compression_results': decomp_results,
        'adaptation_results': adaptation_results,
        'best_scale': best_scale,
        'core_system': core
    }


def analyze_and_optimize(model, analysis_config: Dict[str, Any] = None):
    """
    Comprehensive analysis and optimization workflow.
    
    Args:
        model: Model to analyze
        analysis_config: Configuration for analysis
        
    Returns:
        Analysis results and optimization recommendations
    """
    if analysis_config is None:
        analysis_config = {}
    
    # Model compressibility analysis
    compressibility = analyze_model_compressibility(model)
    
    # SVD analysis
    core = create_adaptive_system()
    decomp_results = core.decompose_model(model)
    
    # Policy analysis (simulate different scenarios)
    import torch
    sample_layer_info = torch.randn(10, 128)  # Sample layer characteristics
    policy_decisions = []
    
    for layer_info in sample_layer_info:
        decision = core.make_scaling_decision(layer_info)
        policy_decisions.append(decision)
    
    # Generate recommendations
    avg_compression = compressibility['summary']['potential_compression_ratio_95']
    compressibility_score = compressibility['summary']['compressibility_score']
    
    if compressibility_score > 0.7:
        recommendation = "Highly compressible - aggressive compression recommended"
        suggested_ratio = 0.3
    elif compressibility_score > 0.4:
        recommendation = "Moderately compressible - balanced compression recommended"
        suggested_ratio = 0.5
    else:
        recommendation = "Low compressibility - conservative compression recommended"
        suggested_ratio = 0.7
    
    return {
        'compressibility_analysis': compressibility,
        'decomposition_results': decomp_results,
        'policy_analysis': policy_decisions,
        'recommendations': {
            'overall_recommendation': recommendation,
            'suggested_compression_ratio': suggested_ratio,
            'compressibility_score': compressibility_score,
            'expected_compression': avg_compression
        },
        'core_system': core
    }


# Module initialization
logger.info(f"AdaptiveScale Networks Core module v{__version__} loaded")
logger.info("Available components: SVD, Policy, Meta-Learning, Uncertainty Estimation")

# Check for optional dependencies
try:
    import torch
    logger.info(f"PyTorch {torch.__version__} detected")
except ImportError:
    logger.error("PyTorch is required but not installed")

try:
    import transformers
    logger.info(f"Transformers {transformers.__version__} detected")
except ImportError:
    logger.warning("Transformers not available - some model-specific features may be limited")

# Performance recommendations
import torch
if torch.cuda.is_available():
    logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
    logger.info("Recommendation: Use GPU acceleration for large models")
else:
    logger.info("CUDA not available - using CPU")
    logger.info("Recommendation: Consider smaller models or enable CPU optimizations")

# Module health check
def health_check():
    """Perform a health check of the core module."""
    try:
        # Test basic functionality
        config = SVDConfig()
        policy_config = PolicyConfig()
        
        # Test factory functions
        decomposer = create_svd_decomposer()
        policy = create_policy_network()
        
        logger.info("Core module health check: PASSED")
        return True
    except Exception as e:
        logger.error(f"Core module health check: FAILED - {e}")
        return False

# Run health check on import
if health_check():
    logger.info("AdaptiveScale Networks Core module ready for use")
else:
    logger.warning("Core module initialization completed with warnings")