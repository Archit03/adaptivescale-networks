"""
Experiment-specific configurations for AdaptiveScale Networks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import logging
from .base_config import ASNConfig
from .model_configs import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Base class for experiment configurations"""
    
    name: str = "default_experiment"
    description: str = "Default ASN experiment"
    model_config: ASNConfig = field(default_factory=ASNConfig)
    
    # Experiment metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Logging and monitoring
    log_level: str = "INFO"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100
    
    def __post_init__(self):
        """Post-initialization setup"""
        # Set up logging level
        logging.getLogger().setLevel(getattr(logging, self.log_level.upper()))
        
        # Add experiment name to model config paths
        if hasattr(self.model_config, 'output_dir'):
            self.model_config.output_dir = self.model_config.output_dir / self.name
            self.model_config.checkpoint_dir = self.model_config.checkpoint_dir / self.name
            self.model_config.log_dir = self.model_config.log_dir / self.name
    
    def get_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        import hashlib
        import json
        
        # Create hash from configuration
        config_str = json.dumps(self.model_config.to_dict(), sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        return f"{self.name}_{config_hash}"


@dataclass
class BaselineConfig(ExperimentConfig):
    """Configuration for baseline experiments"""
    
    name: str = "baseline"
    description: str = "Baseline ASN experiment with standard settings"
    
    def __init__(self, model_name: str = "gpt2", **kwargs):
        # Get model-specific configuration
        self.model_config = get_model_config(model_name)
        
        # Apply any overrides
        if kwargs:
            self.model_config.update(**kwargs)
        
        # Set baseline-specific settings
        self.model_config.use_continual_learning = False
        self.model_config.cem_iterations = 5  # Reduced for baseline
        self.model_config.num_iterations = 300  # Shorter training
        
        super().__init__()


@dataclass
class AblationConfig(ExperimentConfig):
    """Configuration for ablation studies"""
    
    name: str = "ablation"
    description: str = "Ablation study configuration"
    
    # Ablation settings
    ablation_type: str = "component"  # component, hyperparameter, architecture
    disabled_components: List[str] = field(default_factory=list)
    
    def __init__(self, model_name: str = "gpt2", ablation_type: str = "component", 
                 disabled_components: List[str] = None, **kwargs):
        self.ablation_type = ablation_type
        self.disabled_components = disabled_components or []
        
        # Get base configuration
        self.model_config = get_model_config(model_name)
        
        # Apply ablation settings
        self._apply_ablation_settings()
        
        # Apply any additional overrides
        if kwargs:
            self.model_config.update(**kwargs)
        
        super().__init__()
    
    def _apply_ablation_settings(self):
        """Apply ablation-specific settings"""
        for component in self.disabled_components:
            if component == "meta_learning":
                self.model_config.meta_learning_rate = 0.0
                self.model_config.num_inner_steps = 0
            elif component == "uncertainty":
                self.model_config.mc_dropout_samples = 1
                self.model_config.ensemble_size = 1
            elif component == "continual_learning":
                self.model_config.use_continual_learning = False
                self.model_config.ewc_lambda = 0.0
            elif component == "cem":
                self.model_config.cem_iterations = 1
                self.model_config.cem_population_size = 4
            elif component == "progressive_training":
                # Flatten to single stage
                total_iterations = sum(stage['iterations'] for stage in self.model_config.progressive_stages)
                self.model_config.progressive_stages = [
                    {'tasks': self.model_config.task_types, 'iterations': total_iterations, 'lr_scale': 1.0}
                ]
            elif component == "svd_multi_scale":
                # Use only single scale
                self.model_config.svd_scales = [0.5]
            
        logger.info(f"Applied ablation settings for components: {self.disabled_components}")


@dataclass
class ScalingConfig(ExperimentConfig):
    """Configuration for scaling studies"""
    
    name: str = "scaling"
    description: str = "Scaling study configuration"
    
    # Scaling dimensions
    scale_dimension: str = "model_size"  # model_size, data_size, computation
    scale_factors: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0])
    
    def __init__(self, model_name: str = "gpt2", scale_dimension: str = "model_size",
                 scale_factors: List[float] = None, **kwargs):
        self.scale_dimension = scale_dimension
        self.scale_factors = scale_factors or [0.5, 1.0, 2.0, 4.0]
        
        # Get base configuration
        self.model_config = get_model_config(model_name)
        
        # Apply scaling settings
        self._apply_scaling_settings()
        
        # Apply any additional overrides
        if kwargs:
            self.model_config.update(**kwargs)
        
        super().__init__()
    
    def _apply_scaling_settings(self):
        """Apply scaling-specific settings"""
        base_factor = 1.0  # Use the middle factor as base
        
        if self.scale_dimension == "model_size":
            # Scale SVD parameters
            base_scales = self.model_config.svd_scales
            self.model_config.svd_scales = [s * base_factor for s in base_scales]
            
        elif self.scale_dimension == "data_size":
            # Scale training iterations and batch size
            self.model_config.num_iterations = int(self.model_config.num_iterations * base_factor)
            for stage in self.model_config.progressive_stages:
                stage['iterations'] = int(stage['iterations'] * base_factor)
        
        elif self.scale_dimension == "computation":
            # Scale CEM and meta-learning parameters
            self.model_config.cem_iterations = max(1, int(self.model_config.cem_iterations * base_factor))
            self.model_config.cem_population_size = max(4, int(self.model_config.cem_population_size * base_factor))
            self.model_config.num_inner_steps = max(1, int(self.model_config.num_inner_steps * base_factor))


@dataclass
class FewShotConfig(ExperimentConfig):
    """Configuration for few-shot learning experiments"""
    
    name: str = "few_shot"
    description: str = "Few-shot learning configuration"
    
    # Few-shot specific settings
    shot_sizes: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    num_episodes: int = 100
    
    def __init__(self, model_name: str = "gpt2", shot_sizes: List[int] = None, **kwargs):
        self.shot_sizes = shot_sizes or [1, 3, 5, 10]
        
        # Get base configuration
        self.model_config = get_model_config(model_name)
        
        # Optimize for few-shot learning
        self.model_config.cem_iterations = 15  # More CEM iterations
        self.model_config.cem_population_size = 30
        self.model_config.meta_learning_rate = 1e-2  # Higher meta-learning rate
        self.model_config.num_inner_steps = 10
        
        # Apply any additional overrides
        if kwargs:
            self.model_config.update(**kwargs)
        
        super().__init__()


@dataclass
class BenchmarkConfig(ExperimentConfig):
    """Configuration for benchmark evaluation"""
    
    name: str = "benchmark"
    description: str = "Benchmark evaluation configuration"
    
    # Benchmark settings
    benchmark_tasks: List[str] = field(default_factory=lambda: ['qa', 'math', 'reasoning', 'code'])
    cross_validation_folds: int = 5
    statistical_tests: bool = True
    
    def __init__(self, model_name: str = "gpt2", benchmark_tasks: List[str] = None, **kwargs):
        self.benchmark_tasks = benchmark_tasks or ['qa', 'math', 'reasoning', 'code']
        
        # Get base configuration
        self.model_config = get_model_config(model_name)
        
        # Focus on evaluation
        self.model_config.task_types = self.benchmark_tasks
        self.model_config.max_eval_samples = 1000  # More evaluation samples
        self.model_config.eval_frequency = 25  # More frequent evaluation
        
        # Apply any additional overrides
        if kwargs:
            self.model_config.update(**kwargs)
        
        super().__init__()


@dataclass
class HyperparameterConfig(ExperimentConfig):
    """Configuration for hyperparameter optimization"""
    
    name: str = "hyperopt"
    description: str = "Hyperparameter optimization configuration"
    
    # Hyperparameter search settings
    search_space: Dict[str, Any] = field(default_factory=dict)
    optimization_metric: str = "overall_accuracy"
    num_trials: int = 50
    
    def __init__(self, model_name: str = "gpt2", search_space: Dict[str, Any] = None, **kwargs):
        self.search_space = search_space or self._get_default_search_space()
        
        # Get base configuration
        self.model_config = get_model_config(model_name)
        
        # Shorter training for hyperparameter search
        self.model_config.num_iterations = 200
        for stage in self.model_config.progressive_stages:
            stage['iterations'] = max(25, stage['iterations'] // 4)
        
        # Apply any additional overrides
        if kwargs:
            self.model_config.update(**kwargs)
        
        super().__init__()
    
    def _get_default_search_space(self) -> Dict[str, Any]:
        """Get default hyperparameter search space"""
        return {
            'learning_rate': {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
            'batch_size': {'type': 'choice', 'choices': [1, 2, 4, 8]},
            'kl_coeff': {'type': 'log_uniform', 'low': 0.001, 'high': 1.0},
            'entropy_coeff': {'type': 'log_uniform', 'low': 0.001, 'high': 0.1},
            'cem_population_size': {'type': 'choice', 'choices': [10, 20, 30, 40]},
            'cem_elite_ratio': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
            'meta_learning_rate': {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-1},
            'num_inner_steps': {'type': 'choice', 'choices': [3, 5, 10, 15]},
        }


# Experiment configuration registry
EXPERIMENT_CONFIGS = {
    "baseline": BaselineConfig,
    "ablation": AblationConfig,
    "scaling": ScalingConfig,
    "few_shot": FewShotConfig,
    "benchmark": BenchmarkConfig,
    "hyperopt": HyperparameterConfig,
}


def create_experiment_config(experiment_type: str, model_name: str = "gpt2", **kwargs) -> ExperimentConfig:
    """
    Create an experiment configuration.
    
    Args:
        experiment_type: Type of experiment (baseline, ablation, scaling, etc.)
        model_name: Name of the model to use
        **kwargs: Additional configuration parameters
        
    Returns:
        ExperimentConfig: Configured experiment
    """
    if experiment_type not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment type: {experiment_type}. "
                        f"Available types: {list(EXPERIMENT_CONFIGS.keys())}")
    
    config_class = EXPERIMENT_CONFIGS[experiment_type]
    return config_class(model_name=model_name, **kwargs)


def list_experiment_types() -> List[str]:
    """List all available experiment types"""
    return list(EXPERIMENT_CONFIGS.keys())


def register_experiment_config(experiment_type: str, config_class: type):
    """Register a new experiment configuration"""
    EXPERIMENT_CONFIGS[experiment_type] = config_class


# Preset experiment configurations
def get_quick_test_config(model_name: str = "gpt2") -> ExperimentConfig:
    """Get configuration for quick testing"""
    config = BaselineConfig(model_name=model_name)
    
    # Quick test settings
    config.model_config.num_iterations = 50
    config.model_config.progressive_stages = [
        {'tasks': ['qa'], 'iterations': 50, 'lr_scale': 1.0}
    ]
    config.model_config.max_eval_samples = 100
    config.model_config.cem_iterations = 3
    config.model_config.cem_population_size = 8
    config.name = "quick_test"
    config.description = "Quick test configuration for development"
    
    return config


def get_full_evaluation_config(model_name: str = "gpt2") -> ExperimentConfig:
    """Get configuration for comprehensive evaluation"""
    config = BenchmarkConfig(model_name=model_name)
    
    # Comprehensive settings
    config.model_config.num_iterations = 1000
    config.model_config.max_eval_samples = 2000
    config.model_config.eval_frequency = 50
    config.cross_validation_folds = 10
    config.name = "full_evaluation"
    config.description = "Comprehensive evaluation configuration"
    
    return config


def get_research_config(model_name: str = "gpt2") -> ExperimentConfig:
    """Get configuration for research experiments"""
    config = ExperimentConfig()
    config.model_config = get_model_config(model_name)
    
    # Research settings
    config.model_config.use_continual_learning = True
    config.model_config.use_interpretability = True
    config.model_config.profile_memory = True
    config.model_config.cem_iterations = 20
    config.model_config.cem_population_size = 40
    config.name = "research"
    config.description = "Research configuration with all features enabled"
    
    return config