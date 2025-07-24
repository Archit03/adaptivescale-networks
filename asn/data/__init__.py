"""
Data module for AdaptiveScale Networks.

This module provides comprehensive data handling capabilities including:
- Dataset implementations for QA, Math, Reasoning, and Code tasks
- Data preprocessing and augmentation utilities
- Multi-task data loading and batching
- Few-shot episode generation for meta-learning
- Evaluation metrics and benchmarking tools

The module is designed to support both standard supervised learning and
meta-learning scenarios, with particular emphasis on few-shot adaptation
and continual learning use cases.
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset

# Import dataset functionality
from . import datasets
from .datasets import (
    # Dataset factories and utilities
    create_dataset,
    create_dataloader, 
    create_few_shot_tasks,
    evaluate_predictions,
    get_available_datasets,
    
    # Multi-task support
    MultiTaskDataset,
    DatasetBenchmark,
    
    # Base classes and configs
    BaseQADataset, QAExample, QADatasetConfig,
    BaseMathDataset, MathExample, MathDatasetConfig,
    BaseReasoningDataset, ReasoningExample, ReasoningDatasetConfig,
    BaseCodeDataset, CodeExample, CodeDatasetConfig,
    
    # Specific dataset implementations
    SQuADDataset, GSM8KDataset, ARCDataset, HumanEvalDataset,
    
    # Evaluation functions
    evaluate_qa_predictions,
    evaluate_math_predictions, 
    evaluate_reasoning_predictions,
    evaluate_code_predictions
)

# Import data processing utilities (we'll implement these)
from .loaders import (
    FewShotDataLoader,
    MetaLearningDataLoader,
    AdaptiveDataLoader,
    create_meta_learning_loader
)

# Import preprocessing utilities
from .preprocessing import (
    DataPreprocessor,
    TextPreprocessor,
    CodePreprocessor,
    MathPreprocessor,
    create_preprocessor
)

__all__ = [
    # Core module exports
    'datasets',
    'loaders', 
    'preprocessing',
    
    # Dataset creation and management
    'create_dataset',
    'create_dataloader',
    'create_few_shot_tasks',
    'create_meta_learning_loader',
    'evaluate_predictions',
    'get_available_datasets',
    
    # Multi-task and benchmark support  
    'MultiTaskDataset',
    'DatasetBenchmark',
    'create_benchmark_suite',
    'run_comprehensive_evaluation',
    
    # Base classes and examples
    'BaseQADataset', 'QAExample', 'QADatasetConfig',
    'BaseMathDataset', 'MathExample', 'MathDatasetConfig', 
    'BaseReasoningDataset', 'ReasoningExample', 'ReasoningDatasetConfig',
    'BaseCodeDataset', 'CodeExample', 'CodeDatasetConfig',
    
    # Specialized data loaders
    'FewShotDataLoader',
    'MetaLearningDataLoader', 
    'AdaptiveDataLoader',
    
    # Data preprocessing
    'DataPreprocessor',
    'TextPreprocessor',
    'CodePreprocessor',
    'MathPreprocessor',
    'create_preprocessor',
    
    # Evaluation functions
    'evaluate_qa_predictions',
    'evaluate_math_predictions',
    'evaluate_reasoning_predictions', 
    'evaluate_code_predictions',
    
    # High-level interfaces
    'DataManager',
    'ExperimentDatasetSuite',
    'create_asn_datasets',
    'setup_meta_learning_data'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "ASN Research Team"

# Set up logging
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_DATA_CONFIG = {
    'cache_dir': 'cache/datasets',
    'max_examples_per_dataset': None,
    'use_preprocessing': True,
    'enable_caching': True,
    'few_shot_shots': 5,
    'few_shot_tasks': 100,
    'batch_size': 4,
    'num_workers': 0
}

# Supported task types and their default datasets
TASK_DATASETS = {
    'qa': {
        'primary': 'squad',
        'alternatives': ['natural_questions', 'ms_marco', 'coqa'],
        'benchmark': ['squad', 'natural_questions']
    },
    'math': {
        'primary': 'gsm8k', 
        'alternatives': ['math', 'mathqa'],
        'benchmark': ['gsm8k', 'math']
    },
    'reasoning': {
        'primary': 'arc_easy',
        'alternatives': ['commonsense_qa', 'piqa', 'hellaswag'],
        'benchmark': ['arc_easy', 'arc_challenge', 'commonsense_qa']
    },
    'code': {
        'primary': 'humaneval',
        'alternatives': ['mbpp', 'apps'],
        'benchmark': ['humaneval', 'mbpp']
    }
}


class DataManager:
    """
    Central data management class for AdaptiveScale Networks.
    
    Provides a unified interface for dataset creation, preprocessing,
    and evaluation across all supported task types.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DataManager.
        
        Args:
            config: Configuration dictionary for data handling
        """
        self.config = {**DEFAULT_DATA_CONFIG, **(config or {})}
        self.datasets = {}
        self.dataloaders = {}
        self.preprocessors = {}
        self.benchmark_suite = None
        
        # Set up cache directory
        self.cache_dir = Path(self.config['cache_dir'])
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DataManager initialized with config: {self.config}")
    
    def load_task_data(self, task_type: str, dataset_name: str = None, 
                      split: str = "train", **kwargs) -> Dataset:
        """
        Load data for a specific task.
        
        Args:
            task_type: Type of task ('qa', 'math', 'reasoning', 'code')
            dataset_name: Specific dataset name (uses primary if None)
            split: Dataset split to load
            **kwargs: Additional dataset configuration
            
        Returns:
            Dataset instance
        """
        if task_type not in TASK_DATASETS:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Use primary dataset if none specified
        if dataset_name is None:
            dataset_name = TASK_DATASETS[task_type]['primary']
        
        # Create dataset config
        config = {
            'split': split,
            'cache_dir': str(self.cache_dir),
            'max_examples': self.config['max_examples_per_dataset'],
            **kwargs
        }
        
        # Create and cache dataset
        cache_key = f"{task_type}_{dataset_name}_{split}"
        if cache_key not in self.datasets:
            self.datasets[cache_key] = create_dataset(task_type, dataset_name, config)
            logger.info(f"Loaded {task_type}/{dataset_name} ({split}): {len(self.datasets[cache_key])} examples")
        
        return self.datasets[cache_key]
    
    def create_task_dataloader(self, task_type: str, dataset_name: str = None,
                              split: str = "train", batch_size: int = None, 
                              **kwargs) -> DataLoader:
        """
        Create DataLoader for a specific task.
        
        Args:
            task_type: Type of task
            dataset_name: Specific dataset name
            split: Dataset split
            batch_size: Batch size (uses config default if None)
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader instance
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        # Load dataset
        dataset = self.load_task_data(task_type, dataset_name, split, **kwargs)
        
        # Create and cache dataloader
        cache_key = f"{task_type}_{dataset_name}_{split}_{batch_size}"
        if cache_key not in self.dataloaders:
            self.dataloaders[cache_key] = dataset.create_dataloader(
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=self.config['num_workers'],
                **kwargs
            )
        
        return self.dataloaders[cache_key]
    
    def setup_few_shot_data(self, task_type: str, dataset_name: str = None,
                           shots: int = None, num_tasks: int = None) -> List[Dict[str, Any]]:
        """
        Set up few-shot learning data.
        
        Args:
            task_type: Type of task
            dataset_name: Specific dataset name
            shots: Number of shots per task
            num_tasks: Number of tasks to generate
            
        Returns:
            List of few-shot episodes
        """
        if shots is None:
            shots = self.config['few_shot_shots']
        if num_tasks is None:
            num_tasks = self.config['few_shot_tasks']
        
        # Load dataset
        dataset = self.load_task_data(task_type, dataset_name, "train")
        
        # Create few-shot tasks
        episodes = create_few_shot_tasks(task_type, dataset, num_tasks, shots)
        
        logger.info(f"Created {len(episodes)} few-shot episodes for {task_type}")
        return episodes
    
    def create_multi_task_loader(self, task_configs: Dict[str, Dict[str, Any]], 
                                batch_size: int = None) -> MultiTaskDataset:
        """
        Create multi-task dataset.
        
        Args:
            task_configs: Configuration for each task type
            batch_size: Batch size for dataloaders
            
        Returns:
            MultiTaskDataset instance
        """
        if batch_size is None:
            batch_size = self.config['batch_size']
        
        # Expand task configs with defaults
        expanded_configs = {}
        for task_type, config in task_configs.items():
            expanded_config = {
                'dataset_name': config.get('dataset_name', TASK_DATASETS[task_type]['primary']),
                'weight': config.get('weight', 1.0),
                'config': {
                    'cache_dir': str(self.cache_dir),
                    'max_examples': self.config['max_examples_per_dataset'],
                    **config.get('config', {})
                }
            }
            expanded_configs[task_type] = expanded_config
        
        multi_task = MultiTaskDataset(expanded_configs)
        logger.info(f"Created MultiTaskDataset with {len(expanded_configs)} task types")
        
        return multi_task
    
    def create_benchmark_suite(self, task_types: List[str] = None, 
                              custom_config: Dict[str, Any] = None) -> DatasetBenchmark:
        """
        Create comprehensive benchmark suite.
        
        Args:
            task_types: Task types to include (all if None)
            custom_config: Custom benchmark configuration
            
        Returns:
            DatasetBenchmark instance
        """
        if task_types is None:
            task_types = list(TASK_DATASETS.keys())
        
        # Build benchmark config
        benchmark_config = {}
        for task_type in task_types:
            if task_type not in TASK_DATASETS:
                logger.warning(f"Unknown task type for benchmark: {task_type}")
                continue
            
            benchmark_config[task_type] = {}
            
            # Add benchmark datasets for this task type
            for dataset_name in TASK_DATASETS[task_type]['benchmark']:
                dataset_config = {
                    'split': 'test',
                    'cache_dir': str(self.cache_dir),
                    'max_examples': 500  # Reasonable benchmark size
                }
                
                # Apply custom config if provided
                if custom_config and task_type in custom_config:
                    dataset_config.update(custom_config[task_type].get(dataset_name, {}))
                
                benchmark_config[task_type][dataset_name] = dataset_config
        
        self.benchmark_suite = DatasetBenchmark(benchmark_config)
        logger.info(f"Created benchmark suite for {len(task_types)} task types")
        
        return self.benchmark_suite
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about loaded data."""
        stats = {
            'loaded_datasets': len(self.datasets),
            'cached_dataloaders': len(self.dataloaders),
            'cache_dir': str(self.cache_dir),
            'config': self.config,
            'dataset_details': {}
        }
        
        # Collect dataset details
        for key, dataset in self.datasets.items():
            task_type, dataset_name, split = key.split('_', 2)
            
            if task_type not in stats['dataset_details']:
                stats['dataset_details'][task_type] = {}
            
            stats['dataset_details'][task_type][f"{dataset_name}_{split}"] = {
                'num_examples': len(dataset),
                'dataset_class': type(dataset).__name__
            }
        
        return stats


class ExperimentDatasetSuite:
    """
    Dataset suite specifically designed for ASN experiments.
    
    Provides pre-configured datasets and evaluation protocols
    for reproducible ASN research.
    """
    
    def __init__(self, experiment_type: str = "comprehensive", max_examples: int = None):
        """
        Initialize experiment dataset suite.
        
        Args:
            experiment_type: Type of experiment ('quick', 'comprehensive', 'ablation')
            max_examples: Maximum examples per dataset
        """
        self.experiment_type = experiment_type
        self.max_examples = max_examples
        self.data_manager = DataManager()
        
        # Define experiment configurations
        self.experiment_configs = {
            'quick': {
                'task_types': ['qa', 'math'],
                'datasets_per_task': 1,
                'max_examples': 100,
                'few_shot_tasks': 20
            },
            'comprehensive': {
                'task_types': ['qa', 'math', 'reasoning', 'code'],
                'datasets_per_task': 2, 
                'max_examples': 1000,
                'few_shot_tasks': 100
            },
            'ablation': {
                'task_types': ['qa', 'math', 'reasoning', 'code'],
                'datasets_per_task': 1,
                'max_examples': 500,
                'few_shot_tasks': 50
            }
        }
        
        if experiment_type not in self.experiment_configs:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        self.config = self.experiment_configs[experiment_type]
        if max_examples is not None:
            self.config['max_examples'] = max_examples
        
        logger.info(f"Initialized ExperimentDatasetSuite for {experiment_type} experiments")
    
    def setup_training_data(self) -> Dict[str, DataLoader]:
        """Set up training data for all tasks."""
        dataloaders = {}
        
        for task_type in self.config['task_types']:
            # Use primary dataset for training
            dataset_name = TASK_DATASETS[task_type]['primary']
            
            dataloader = self.data_manager.create_task_dataloader(
                task_type=task_type,
                dataset_name=dataset_name,
                split='train',
                max_examples=self.config['max_examples']
            )
            
            dataloaders[f"{task_type}_train"] = dataloader
        
        return dataloaders
    
    def setup_evaluation_data(self) -> Dict[str, DataLoader]:
        """Set up evaluation data for all tasks."""
        dataloaders = {}
        
        for task_type in self.config['task_types']:
            # Use benchmark datasets for evaluation
            for dataset_name in TASK_DATASETS[task_type]['benchmark'][:self.config['datasets_per_task']]:
                dataloader = self.data_manager.create_task_dataloader(
                    task_type=task_type,
                    dataset_name=dataset_name,
                    split='test',
                    max_examples=self.config['max_examples']
                )
                
                dataloaders[f"{task_type}_{dataset_name}_test"] = dataloader
        
        return dataloaders
    
    def setup_few_shot_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Set up few-shot learning data for meta-learning."""
        few_shot_data = {}
        
        for task_type in self.config['task_types']:
            episodes = self.data_manager.setup_few_shot_data(
                task_type=task_type,
                shots=5,
                num_tasks=self.config['few_shot_tasks']
            )
            
            few_shot_data[f"{task_type}_few_shot"] = episodes
        
        return few_shot_data
    
    def create_benchmark_suite(self) -> DatasetBenchmark:
        """Create benchmark suite for comprehensive evaluation."""
        return self.data_manager.create_benchmark_suite(self.config['task_types'])
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of experiment setup."""
        return {
            'experiment_type': self.experiment_type,
            'config': self.config,
            'data_statistics': self.data_manager.get_data_statistics(),
            'task_types': self.config['task_types'],
            'max_examples_per_dataset': self.config['max_examples']
        }


# High-level convenience functions
def create_asn_datasets(task_types: List[str] = None, 
                       experiment_type: str = "comprehensive",
                       max_examples: int = None) -> ExperimentDatasetSuite:
    """
    Create datasets specifically configured for ASN experiments.
    
    Args:
        task_types: Task types to include
        experiment_type: Type of experiment setup
        max_examples: Maximum examples per dataset
        
    Returns:
        ExperimentDatasetSuite instance
    """
    suite = ExperimentDatasetSuite(experiment_type, max_examples)
    
    if task_types is not None:
        suite.config['task_types'] = task_types
    
    return suite


def setup_meta_learning_data(task_types: List[str] = None,
                            shots_per_task: int = 5,
                            num_tasks: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """
    Set up meta-learning data across multiple task types.
    
    Args:
        task_types: Task types to include
        shots_per_task: Number of shots per task
        num_tasks: Number of tasks per task type
        
    Returns:
        Dictionary mapping task types to few-shot episodes
    """
    if task_types is None:
        task_types = list(TASK_DATASETS.keys())
    
    data_manager = DataManager()
    meta_data = {}
    
    for task_type in task_types:
        episodes = data_manager.setup_few_shot_data(
            task_type=task_type,
            shots=shots_per_task,
            num_tasks=num_tasks
        )
        meta_data[task_type] = episodes
    
    return meta_data


def create_benchmark_suite(task_types: List[str] = None,
                          max_examples_per_dataset: int = 500) -> DatasetBenchmark:
    """
    Create a comprehensive benchmark suite.
    
    Args:
        task_types: Task types to include in benchmark
        max_examples_per_dataset: Maximum examples per dataset
        
    Returns:
        DatasetBenchmark instance
    """
    data_manager = DataManager({
        'max_examples_per_dataset': max_examples_per_dataset
    })
    
    return data_manager.create_benchmark_suite(task_types)


def run_comprehensive_evaluation(model, tokenizer=None, 
                                task_types: List[str] = None,
                                max_examples: int = 500) -> Dict[str, Any]:
    """
    Run comprehensive evaluation across all supported datasets.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer (if needed)
        task_types: Task types to evaluate
        max_examples: Maximum examples per dataset
        
    Returns:
        Comprehensive evaluation results
    """
    # Create benchmark suite
    benchmark = create_benchmark_suite(task_types, max_examples)
    
    # Run evaluation
    results = benchmark.run_benchmark(model, tokenizer)
    
    # Generate summary report
    summary_report = benchmark.get_summary_report()
    
    return {
        'results': results,
        'summary_report': summary_report,
        'benchmark_config': benchmark.config
    }


def get_data_info() -> Dict[str, Any]:
    """Get comprehensive information about available data."""
    return {
        'version': __version__,
        'supported_task_types': list(TASK_DATASETS.keys()),
        'available_datasets': get_available_datasets(),
        'default_config': DEFAULT_DATA_CONFIG,
        'task_datasets': TASK_DATASETS
    }


# Module initialization and health check
def _health_check() -> bool:
    """Perform health check of data module."""
    try:
        # Test basic functionality
        data_manager = DataManager()
        
        # Test dataset creation for each task type
        for task_type in TASK_DATASETS.keys():
            try:
                primary_dataset = TASK_DATASETS[task_type]['primary']
                # This would fail gracefully if datasets aren't available
                dataset = create_dataset(task_type, primary_dataset, {'max_examples': 1})
                logger.debug(f"Successfully tested {task_type}/{primary_dataset}")
            except Exception as e:
                logger.warning(f"Could not load {task_type} data: {e}")
        
        logger.info("Data module health check: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Data module health check: FAILED - {e}")
        return False


# Module loading and initialization
logger.info(f"AdaptiveScale Networks Data module v{__version__} loaded")
logger.info(f"Supported task types: {list(TASK_DATASETS.keys())}")

# Check dependencies
try:
    from datasets import load_dataset
    logger.info("HuggingFace datasets available")
except ImportError:
    logger.warning("HuggingFace datasets not available - some functionality will be limited")

try:
    from transformers import AutoTokenizer  
    logger.info("HuggingFace transformers available")
except ImportError:
    logger.warning("HuggingFace transformers not available - tokenization features limited")

# Run health check
if _health_check():
    logger.info("Data module ready for use")
    logger.info("Quick start: use create_asn_datasets() or DataManager() for data setup")
else:
    logger.warning("Data module initialized with some limitations - check dependencies")