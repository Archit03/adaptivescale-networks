"""
Base configuration for AdaptiveScale Networks (ASN).

This module provides the core ASNConfig class that serves as the foundation
for all model-specific and experiment-specific configurations.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import yaml
import json
import torch
import logging

logger = logging.getLogger(__name__)


@dataclass
class ASNConfig:
    """
    Base configuration class for AdaptiveScale Networks.
    
    This class contains all the core configuration parameters for ASN,
    including model settings, training parameters, SVD decomposition settings,
    meta-learning parameters, and evaluation configurations.
    """
    
    # ============================================================================
    # Model Configuration
    # ============================================================================
    
    # Basic model settings
    model_name: str = "gpt2"
    device: str = "auto"  # auto, cuda, cpu
    dtype: torch.dtype = torch.float16
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    
    # Model generation settings
    max_length: int = 512
    max_new_tokens: int = 64
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Memory management
    max_memory_gb: float = 16.0
    cpu_offload: bool = False
    gradient_accumulation_steps: int = 1
    
    # ============================================================================
    # SVD Decomposition Configuration
    # ============================================================================
    
    # SVD scales for multi-scale decomposition
    svd_scales: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    
    # Target layers for SVD decomposition
    target_layers: List[str] = field(default_factory=lambda: ["c_fc", "c_proj", "c_attn"])
    
    # SVD parameters
    max_rank_ratio: float = 0.8  # Maximum rank ratio for SVD decomposition
    min_rank: int = 1
    use_randomized_svd: bool = False
    svd_driver: str = "gesvd"  # gesvd, gesvda, gesdd
    numerical_stability_eps: float = 1e-8
    
    # Compression settings
    target_compression_ratio: float = 0.5
    adaptive_rank_selection: bool = True
    rank_selection_metric: str = "reconstruction_error"  # reconstruction_error, frobenius_norm
    
    # ============================================================================
    # Training Configuration
    # ============================================================================
    
    # Basic training settings
    num_iterations: int = 500
    batch_size: int = 2
    eval_batch_size: int = 4
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Learning rate scheduling
    lr_scheduler_type: str = "cosine"  # linear, cosine, constant, polynomial
    cosine_restart_cycles: int = 1
    polynomial_power: float = 1.0
    
    # Optimization settings
    optimizer_type: str = "adamw"  # adamw, adam, sgd
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # ============================================================================
    # Task Configuration
    # ============================================================================
    
    # Supported tasks
    task_types: List[str] = field(default_factory=lambda: ['qa', 'math', 'reasoning', 'code'])
    
    # Progressive training stages
    progressive_stages: List[Dict] = field(default_factory=lambda: [
        {'tasks': ['math'], 'iterations': 100, 'lr_scale': 1.0},
        {'tasks': ['math', 'qa'], 'iterations': 100, 'lr_scale': 0.8},
        {'tasks': ['math', 'qa', 'reasoning'], 'iterations': 150, 'lr_scale': 0.6},
        {'tasks': ['math', 'qa', 'reasoning', 'code'], 'iterations': 150, 'lr_scale': 0.4}
    ])
    
    # ============================================================================
    # Meta-Learning Configuration (MAML)
    # ============================================================================
    
    # MAML settings
    meta_learning_rate: float = 1e-3
    num_inner_steps: int = 5
    inner_learning_rate: float = 1e-2
    
    # Task sampling
    tasks_per_batch: int = 4
    shots_per_task: int = 5
    query_shots_per_task: int = 10
    
    # Meta-learning optimization
    first_order_maml: bool = False
    allow_unused: bool = True
    allow_nograd: bool = False
    
    # ============================================================================
    # Reinforcement Learning Configuration (GRPO)
    # ============================================================================
    
    # GRPO (Generalized Reward Policy Optimization) settings
    num_samples_per_question: int = 4
    kl_coeff: float = 0.1
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    
    # PPO-style settings
    ppo_epsilon: float = 0.2
    ppo_epochs: int = 1
    advantage_normalization: bool = True
    
    # Reward settings
    reward_model_name: Optional[str] = None
    use_length_normalization: bool = True
    length_penalty: float = 0.0
    
    # ============================================================================
    # CEM (Cross-Entropy Method) Configuration
    # ============================================================================
    
    # CEM optimization parameters
    cem_population_size: int = 20
    cem_elite_ratio: float = 0.3
    cem_iterations: int = 10
    cem_alpha: float = 0.9
    cem_convergence_threshold: float = 1e-4
    
    # CEM noise settings
    cem_noise_std: float = 0.1
    cem_noise_decay: float = 0.95
    cem_temperature: float = 1.0
    
    # ============================================================================
    # Uncertainty Estimation Configuration
    # ============================================================================
    
    # Bayesian uncertainty settings
    mc_dropout_samples: int = 10
    dropout_rate: float = 0.1
    
    # Ensemble settings
    ensemble_size: int = 5
    ensemble_temperature: float = 1.0
    
    # Uncertainty quantification
    uncertainty_threshold: float = 0.5
    epistemic_weight: float = 0.5
    aleatoric_weight: float = 0.5
    
    # ============================================================================
    # Continual Learning Configuration
    # ============================================================================
    
    # Elastic Weight Consolidation (EWC)
    use_continual_learning: bool = True
    ewc_lambda: float = 1000.0
    ewc_gamma: float = 1.0
    online_ewc: bool = True
    
    # Memory replay
    memory_replay_size: int = 1000
    memory_replay_batch_size: int = 32
    
    # ============================================================================
    # Evaluation Configuration
    # ============================================================================
    
    # Evaluation settings
    max_eval_samples: int = 500
    eval_frequency: int = 50
    save_predictions: bool = True
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Statistical testing
    use_statistical_testing: bool = True
    significance_level: float = 0.05
    
    # Few-shot evaluation
    few_shot_k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    few_shot_episodes: int = 100
    
    # ============================================================================
    # Monitoring and Logging Configuration
    # ============================================================================
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = "asn-experiments"
    wandb_entity: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: str = ""
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    
    # Logging settings
    log_level: str = "INFO"
    log_frequency: int = 10
    save_logs: bool = True
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_frequency: int = 100
    max_checkpoints: int = 5
    
    # ============================================================================
    # Path Configuration
    # ============================================================================
    
    # Output directories
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    cache_dir: Path = field(default_factory=lambda: Path("cache"))
    
    # Data directories
    data_dir: Path = field(default_factory=lambda: Path("data"))
    results_dir: Path = field(default_factory=lambda: Path("results"))
    
    # ============================================================================
    # Advanced Features
    # ============================================================================
    
    # Interpretability
    use_interpretability: bool = False
    attention_visualization: bool = False
    gradient_attribution: bool = False
    
    # Profiling
    profile_memory: bool = False
    profile_compute: bool = False
    
    # Debugging
    debug_mode: bool = False
    verbose: bool = False
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set up device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Ensure paths are Path objects
        for attr_name in ['output_dir', 'checkpoint_dir', 'log_dir', 'cache_dir', 'data_dir', 'results_dir']:
            path_value = getattr(self, attr_name)
            if isinstance(path_value, str):
                setattr(self, attr_name, Path(path_value))
        
        # Create directories if they don't exist
        for path in [self.output_dir, self.checkpoint_dir, self.log_dir, self.cache_dir, self.results_dir]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Validate configurations
        self._validate_config()
        
        # Set up logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        if self.deterministic:
            self._set_random_seeds()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate SVD scales
        if not all(0.0 < scale <= 1.0 for scale in self.svd_scales):
            raise ValueError("SVD scales must be between 0 and 1")
        
        # Validate batch sizes
        if self.batch_size <= 0 or self.eval_batch_size <= 0:
            raise ValueError("Batch sizes must be positive")
        
        # Validate learning rates
        if self.learning_rate <= 0 or self.meta_learning_rate <= 0:
            raise ValueError("Learning rates must be positive")
        
        # Validate CEM parameters
        if not 0.0 < self.cem_elite_ratio < 1.0:
            raise ValueError("CEM elite ratio must be between 0 and 1")
        
        # Validate progressive stages
        if not self.progressive_stages:
            self.progressive_stages = [
                {'tasks': self.task_types, 'iterations': self.num_iterations, 'lr_scale': 1.0}
            ]
        
        # Ensure total iterations match
        total_stage_iterations = sum(stage['iterations'] for stage in self.progressive_stages)
        if total_stage_iterations != self.num_iterations:
            logger.warning(f"Total stage iterations ({total_stage_iterations}) != num_iterations ({self.num_iterations})")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        log_level = getattr(logging, self.log_level.upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.log_dir / "asn.log") if self.save_logs else logging.NullHandler()
            ]
        )
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
        
        # Re-run post-init validation
        self._validate_config()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = asdict(self)
        
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, torch.dtype):
                config_dict[key] = str(value)
        
        return config_dict
    
    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file"""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def to_json(self, file_path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'ASNConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> 'ASNConfig':
        """Load configuration from JSON file"""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ASNConfig':
        """Create configuration from dictionary"""
        # Handle special types
        if 'dtype' in config_dict and isinstance(config_dict['dtype'], str):
            dtype_mapping = {
                'torch.float16': torch.float16,
                'torch.float32': torch.float32,
                'torch.bfloat16': torch.bfloat16,
                'float16': torch.float16,
                'float32': torch.float32,
                'bfloat16': torch.bfloat16,
            }
            config_dict['dtype'] = dtype_mapping.get(config_dict['dtype'], torch.float16)
        
        # Handle Path objects
        path_keys = ['output_dir', 'checkpoint_dir', 'log_dir', 'cache_dir', 'data_dir', 'results_dir']
        for key in path_keys:
            if key in config_dict and isinstance(config_dict[key], str):
                config_dict[key] = Path(config_dict[key])
        
        return cls(**config_dict)
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'dtype': self.dtype,
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'do_sample': self.do_sample,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training-specific configuration"""
        return {
            'num_iterations': self.num_iterations,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'optimizer_type': self.optimizer_type,
            'lr_scheduler_type': self.lr_scheduler_type,
            'max_grad_norm': self.max_grad_norm,
        }
    
    def get_svd_config(self) -> Dict[str, Any]:
        """Get SVD-specific configuration"""
        return {
            'svd_scales': self.svd_scales,
            'target_layers': self.target_layers,
            'max_rank_ratio': self.max_rank_ratio,
            'min_rank': self.min_rank,
            'use_randomized_svd': self.use_randomized_svd,
            'svd_driver': self.svd_driver,
            'numerical_stability_eps': self.numerical_stability_eps,
        }
    
    def summary(self) -> str:
        """Get configuration summary"""
        summary_lines = [
            "=== ASN Configuration Summary ===",
            f"Model: {self.model_name}",
            f"Device: {self.device}",
            f"Tasks: {', '.join(self.task_types)}",
            f"SVD Scales: {len(self.svd_scales)} scales",
            f"Training Iterations: {self.num_iterations}",
            f"Batch Size: {self.batch_size}",
            f"Learning Rate: {self.learning_rate}",
            f"Output Directory: {self.output_dir}",
            "=================================="
        ]
        return "\n".join(summary_lines)
    
    def __str__(self) -> str:
        """String representation"""
        return self.summary()
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"ASNConfig(model_name='{self.model_name}', num_iterations={self.num_iterations}, tasks={self.task_types})"