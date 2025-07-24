"""
Datasets module for AdaptiveScale Networks.

This module provides comprehensive dataset implementations for various tasks
including question answering, mathematical reasoning, logical reasoning, and
code generation. It supports both standard benchmarks and few-shot learning
scenarios for meta-learning applications.

Available Dataset Types:
- QA: Question Answering (SQuAD, Natural Questions, MS MARCO, CoQA)
- Math: Mathematical Reasoning (GSM8K, MATH, MathQA, AQuA)  
- Reasoning: Logical Reasoning (ARC, CommonsenseQA, PIQA, HellaSwag)
- Code: Code Generation (HumanEval, MBPP, CodeContests, APPS)
"""

import logging
from typing import Dict, List, Optional, Any, Union
from torch.utils.data import DataLoader

# Import dataset implementations
from .qa_datasets import (
    BaseQADataset,
    QAExample,
    QADatasetConfig,
    SQuADDataset,
    NaturalQuestionsDataset,
    MSMARCODataset,
    CoQADataset,
    CustomQADataset,
    QADatasetFactory,
    create_qa_dataloader,
    evaluate_qa_predictions,
    create_few_shot_qa_task
)

from .math_datasets import (
    BaseMathDataset,
    MathExample,
    MathDatasetConfig,
    GSM8KDataset,
    MATHDataset,
    MathQADataset,
    AQuADataset,
    CustomMathDataset,
    MathDatasetFactory,
    create_math_dataloader,
    evaluate_math_predictions,
    create_few_shot_math_tasks
)

from .reasoning_datasets import (
    BaseReasoningDataset,
    ReasoningExample,
    ReasoningDatasetConfig,
    ARCDataset,
    CommonsenseQADataset,
    PIQADataset,
    HellaSwagDataset,
    WinograndeDataset,
    OpenBookQADataset,
    CustomReasoningDataset,
    ReasoningDatasetFactory,
    create_reasoning_dataloader,
    evaluate_reasoning_predictions,
    create_few_shot_reasoning_tasks
)

from .code_datasets import (
    BaseCodeDataset,
    CodeExample,
    CodeDatasetConfig,
    HumanEvalDataset,
    MBPPDataset,
    CodeContestsDataset,
    APPSDataset,
    CustomCodeDataset,
    CodeDatasetFactory,
    create_code_dataloader,
    evaluate_code_predictions,
    create_few_shot_code_tasks,
    extract_code_metrics
)

__all__ = [
    # Base classes and configs
    'BaseQADataset', 'QAExample', 'QADatasetConfig',
    'BaseMathDataset', 'MathExample', 'MathDatasetConfig', 
    'BaseReasoningDataset', 'ReasoningExample', 'ReasoningDatasetConfig',
    'BaseCodeDataset', 'CodeExample', 'CodeDatasetConfig',
    
    # QA datasets
    'SQuADDataset', 'NaturalQuestionsDataset', 'MSMARCODataset', 
    'CoQADataset', 'CustomQADataset', 'QADatasetFactory',
    
    # Math datasets
    'GSM8KDataset', 'MATHDataset', 'MathQADataset', 
    'AQuADataset', 'CustomMathDataset', 'MathDatasetFactory',
    
    # Reasoning datasets
    'ARCDataset', 'CommonsenseQADataset', 'PIQADataset', 
    'HellaSwagDataset', 'WinograndeDataset', 'OpenBookQADataset',
    'CustomReasoningDataset', 'ReasoningDatasetFactory',
    
    # Code datasets
    'HumanEvalDataset', 'MBPPDataset', 'CodeContestsDataset',
    'APPSDataset', 'CustomCodeDataset', 'CodeDatasetFactory',
    
    # Utility functions
    'create_qa_dataloader', 'create_math_dataloader', 
    'create_reasoning_dataloader', 'create_code_dataloader',
    'evaluate_qa_predictions', 'evaluate_math_predictions',
    'evaluate_reasoning_predictions', 'evaluate_code_predictions',
    'create_few_shot_qa_task', 'create_few_shot_math_tasks',
    'create_few_shot_reasoning_tasks', 'create_few_shot_code_tasks',
    'extract_code_metrics',
    
    # Factory functions
    'create_dataset', 'create_dataloader', 'create_few_shot_tasks',
    'evaluate_predictions', 'get_available_datasets'
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "ASN Research Team"

# Set up logging
logger = logging.getLogger(__name__)

# Dataset type mapping
DATASET_TYPES = {
    'qa': {
        'factory': QADatasetFactory,
        'config': QADatasetConfig,
        'datasets': ['squad', 'squad_v2', 'natural_questions', 'ms_marco', 'coqa', 'custom']
    },
    'math': {
        'factory': MathDatasetFactory,
        'config': MathDatasetConfig,
        'datasets': ['gsm8k', 'math', 'mathqa', 'aqua', 'custom']
    },
    'reasoning': {
        'factory': ReasoningDatasetFactory,
        'config': ReasoningDatasetConfig,
        'datasets': ['arc_easy', 'arc_challenge', 'commonsense_qa', 'piqa', 'hellaswag', 'winogrande', 'openbookqa', 'custom']
    },
    'code': {
        'factory': CodeDatasetFactory,
        'config': CodeDatasetConfig,
        'datasets': ['humaneval', 'mbpp', 'codecontests', 'apps', 'custom']
    }
}

# Evaluation function mapping
EVALUATION_FUNCTIONS = {
    'qa': evaluate_qa_predictions,
    'math': evaluate_math_predictions,
    'reasoning': evaluate_reasoning_predictions,
    'code': evaluate_code_predictions
}

# DataLoader creation functions
DATALOADER_FUNCTIONS = {
    'qa': create_qa_dataloader,
    'math': create_math_dataloader,
    'reasoning': create_reasoning_dataloader,
    'code': create_code_dataloader
}

# Few-shot task creation functions
FEW_SHOT_FUNCTIONS = {
    'qa': create_few_shot_qa_task,
    'math': create_few_shot_math_tasks,
    'reasoning': create_few_shot_reasoning_tasks,
    'code': create_few_shot_code_tasks
}


def create_dataset(task_type: str, dataset_name: str, config: Dict[str, Any] = None, **kwargs):
    """
    Universal dataset creation function.
    
    Args:
        task_type: Type of task ('qa', 'math', 'reasoning', 'code')
        dataset_name: Name of specific dataset
        config: Configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        Dataset instance
    """
    if task_type not in DATASET_TYPES:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(DATASET_TYPES.keys())}")
    
    type_info = DATASET_TYPES[task_type]
    factory = type_info['factory']
    config_class = type_info['config']
    
    # Create config if not provided
    if config is None:
        config = config_class(dataset_name=dataset_name, **kwargs)
    elif isinstance(config, dict):
        config = config_class(dataset_name=dataset_name, **config, **kwargs)
    
    return factory.create_dataset(dataset_name, config, **kwargs)


def create_dataloader(task_type: str, dataset_name: str, batch_size: int = 4, 
                     split: str = "train", max_examples: int = None, **kwargs) -> DataLoader:
    """
    Universal DataLoader creation function.
    
    Args:
        task_type: Type of task ('qa', 'math', 'reasoning', 'code')
        dataset_name: Name of specific dataset
        batch_size: Batch size
        split: Dataset split
        max_examples: Maximum number of examples
        **kwargs: Additional configuration
        
    Returns:
        DataLoader instance
    """
    if task_type not in DATALOADER_FUNCTIONS:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(DATALOADER_FUNCTIONS.keys())}")
    
    dataloader_fn = DATALOADER_FUNCTIONS[task_type]
    return dataloader_fn(
        dataset_name=dataset_name,
        batch_size=batch_size,
        split=split,
        max_examples=max_examples,
        **kwargs
    )


def create_few_shot_tasks(task_type: str, dataset, num_tasks: int = 100, 
                         shots_per_task: int = 5, **kwargs) -> List[Dict[str, Any]]:
    """
    Universal few-shot task creation function.
    
    Args:
        task_type: Type of task ('qa', 'math', 'reasoning', 'code')
        dataset: Dataset instance
        num_tasks: Number of tasks to create
        shots_per_task: Number of examples per task
        **kwargs: Additional parameters
        
    Returns:
        List of few-shot tasks
    """
    if task_type not in FEW_SHOT_FUNCTIONS:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(FEW_SHOT_FUNCTIONS.keys())}")
    
    few_shot_fn = FEW_SHOT_FUNCTIONS[task_type]
    return few_shot_fn(dataset, num_tasks, shots_per_task, **kwargs)


def evaluate_predictions(task_type: str, predictions: List[str], 
                        ground_truths: List[str], **kwargs) -> Dict[str, float]:
    """
    Universal prediction evaluation function.
    
    Args:
        task_type: Type of task ('qa', 'math', 'reasoning', 'code')
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        **kwargs: Additional evaluation parameters
        
    Returns:
        Dictionary of evaluation metrics
    """
    if task_type not in EVALUATION_FUNCTIONS:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(EVALUATION_FUNCTIONS.keys())}")
    
    eval_fn = EVALUATION_FUNCTIONS[task_type]
    return eval_fn(predictions, ground_truths, **kwargs)


def get_available_datasets(task_type: str = None) -> Union[List[str], Dict[str, List[str]]]:
    """
    Get list of available datasets.
    
    Args:
        task_type: Specific task type (if None, return all)
        
    Returns:
        List of dataset names or dictionary mapping task types to datasets
    """
    if task_type is None:
        return {t: info['datasets'] for t, info in DATASET_TYPES.items()}
    
    if task_type not in DATASET_TYPES:
        raise ValueError(f"Unknown task type: {task_type}. Available: {list(DATASET_TYPES.keys())}")
    
    return DATASET_TYPES[task_type]['datasets']


class MultiTaskDataset:
    """
    Multi-task dataset that combines multiple task types.
    
    Useful for training models on diverse tasks simultaneously.
    """
    
    def __init__(self, task_configs: Dict[str, Dict[str, Any]]):
        """
        Initialize multi-task dataset.
        
        Args:
            task_configs: Dictionary mapping task types to their configurations
                         Format: {task_type: {dataset_name: name, config: config_dict}}
        """
        self.task_configs = task_configs
        self.datasets = {}
        self.task_weights = {}
        
        # Create individual datasets
        for task_type, config in task_configs.items():
            dataset_name = config['dataset_name']
            dataset_config = config.get('config', {})
            
            self.datasets[task_type] = create_dataset(task_type, dataset_name, dataset_config)
            self.task_weights[task_type] = config.get('weight', 1.0)
        
        logger.info(f"Initialized MultiTaskDataset with {len(self.datasets)} task types")
    
    def get_task_dataloaders(self, batch_size: int = 4, **kwargs) -> Dict[str, DataLoader]:
        """Get DataLoaders for all tasks."""
        dataloaders = {}
        
        for task_type, dataset in self.datasets.items():
            dataloaders[task_type] = dataset.create_dataloader(batch_size=batch_size, **kwargs)
        
        return dataloaders
    
    def get_balanced_batch(self, dataloaders: Dict[str, DataLoader]) -> Dict[str, Any]:
        """Get a balanced batch across all tasks."""
        batch = {}
        
        for task_type, dataloader in dataloaders.items():
            try:
                task_batch = next(iter(dataloader))
                batch[task_type] = task_batch
            except StopIteration:
                # Dataloader exhausted, skip this task
                continue
        
        return batch
    
    def get_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all datasets."""
        stats = {}
        
        for task_type, dataset in self.datasets.items():
            stats[task_type] = {
                'num_examples': len(dataset),
                'task_weight': self.task_weights[task_type],
                'dataset_type': type(dataset).__name__
            }
        
        return stats


class DatasetBenchmark:
    """
    Benchmark suite for evaluating models across multiple datasets.
    """
    
    def __init__(self, benchmark_config: Dict[str, Any]):
        """
        Initialize benchmark suite.
        
        Args:
            benchmark_config: Configuration for benchmark datasets
        """
        self.config = benchmark_config
        self.datasets = {}
        self.results = {}
        
        # Load benchmark datasets
        for task_type, task_datasets in benchmark_config.items():
            self.datasets[task_type] = {}
            
            for dataset_name, config in task_datasets.items():
                self.datasets[task_type][dataset_name] = create_dataset(
                    task_type, dataset_name, config
                )
        
        logger.info(f"Initialized DatasetBenchmark with {sum(len(d) for d in self.datasets.values())} datasets")
    
    def run_benchmark(self, model, tokenizer=None, **kwargs) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run benchmark evaluation on all datasets.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer (if needed)
            **kwargs: Additional evaluation parameters
            
        Returns:
            Nested dictionary of results: {task_type: {dataset_name: {metric: value}}}
        """
        results = {}
        
        for task_type, task_datasets in self.datasets.items():
            results[task_type] = {}
            
            for dataset_name, dataset in task_datasets.items():
                logger.info(f"Evaluating on {task_type}/{dataset_name}")
                
                # Create dataloader
                dataloader = dataset.create_dataloader(batch_size=1, shuffle=False)
                
                # Run inference (placeholder - would need actual model evaluation)
                predictions = self._run_inference(model, dataloader, tokenizer)
                ground_truths = self._extract_ground_truths(dataset)
                
                # Evaluate predictions
                metrics = evaluate_predictions(task_type, predictions, ground_truths)
                results[task_type][dataset_name] = metrics
                
                logger.info(f"Results for {dataset_name}: {metrics}")
        
        self.results = results
        return results
    
    def _run_inference(self, model, dataloader, tokenizer):
        """Run model inference (placeholder implementation)."""
        # This would be implemented based on the specific model interface
        predictions = []
        
        for batch in dataloader:
            # Placeholder: generate random predictions
            if isinstance(batch, list):
                predictions.extend(["placeholder"] * len(batch))
            else:
                predictions.append("placeholder")
        
        return predictions
    
    def _extract_ground_truths(self, dataset):
        """Extract ground truth answers from dataset."""
        ground_truths = []
        
        for example in dataset.examples:
            if hasattr(example, 'answers') and example.answers:
                ground_truths.append(example.answers[0])
            elif hasattr(example, 'answer'):
                ground_truths.append(str(example.answer))
            else:
                ground_truths.append("unknown")
        
        return ground_truths
    
    def get_summary_report(self) -> str:
        """Generate summary report of benchmark results."""
        if not self.results:
            return "No benchmark results available. Run benchmark first."
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("DATASET BENCHMARK SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for task_type, task_results in self.results.items():
            report_lines.append(f"TASK TYPE: {task_type.upper()}")
            report_lines.append("-" * 40)
            
            for dataset_name, metrics in task_results.items():
                report_lines.append(f"  Dataset: {dataset_name}")
                
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"    {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"    {metric}: {value}")
                
                report_lines.append("")
            
            report_lines.append("")
        
        return "\n".join(report_lines)


# Utility functions for dataset management
def validate_dataset_config(task_type: str, config: Dict[str, Any]) -> bool:
    """
    Validate dataset configuration.
    
    Args:
        task_type: Type of task
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if task_type not in DATASET_TYPES:
        raise ValueError(f"Unknown task type: {task_type}")
    
    config_class = DATASET_TYPES[task_type]['config']
    
    # Check required fields (simplified validation)
    required_fields = ['dataset_name']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    return True


def get_dataset_info(task_type: str, dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a specific dataset.
    
    Args:
        task_type: Type of task
        dataset_name: Name of dataset
        
    Returns:
        Dictionary with dataset information
    """
    if task_type not in DATASET_TYPES:
        raise ValueError(f"Unknown task type: {task_type}")
    
    available_datasets = DATASET_TYPES[task_type]['datasets']
    
    if dataset_name not in available_datasets:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available_datasets}")
    
    # Return basic info (could be expanded with more details)
    return {
        'task_type': task_type,
        'dataset_name': dataset_name,
        'description': f"{task_type} dataset: {dataset_name}",
        'supported_splits': ['train', 'validation', 'test'],
        'metrics': list(EVALUATION_FUNCTIONS[task_type].__annotations__.get('return', {}).keys()) if hasattr(EVALUATION_FUNCTIONS[task_type], '__annotations__') else ['accuracy']
    }


# Module initialization
logger.info(f"AdaptiveScale Networks Datasets module v{__version__} loaded")
logger.info(f"Available task types: {list(DATASET_TYPES.keys())}")
logger.info(f"Total available datasets: {sum(len(info['datasets']) for info in DATASET_TYPES.values())}")

# Check for optional dependencies
try:
    from datasets import load_dataset
    logger.info("HuggingFace datasets library available")
except ImportError:
    logger.warning("HuggingFace datasets library not available - some datasets will be unavailable")

try:
    from transformers import AutoTokenizer
    logger.info("HuggingFace transformers library available")
except ImportError:
    logger.warning("HuggingFace transformers library not available - tokenization features limited")

# Performance recommendations
logger.info("Dataset module ready - use create_dataset() or create_dataloader() for quick setup")
logger.info("For multi-task training, use MultiTaskDataset class")
logger.info("For comprehensive evaluation, use DatasetBenchmark class")