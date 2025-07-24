"""
Evaluation module for AdaptiveScale Networks.

This module provides comprehensive evaluation capabilities including
benchmarks, metrics, few-shot evaluation, and statistical analysis.
"""

from .evaluator import (
    ASNEvaluator,
    EvaluationConfig,
    EvaluationResults,
    ModelEvaluator
)
from .metrics import (
    MetricsCalculator,
    QAMetrics,
    MathMetrics,
    ReasoningMetrics,
    CodeMetrics,
    CompressionMetrics,
    compute_exact_match,
    compute_f1_score,
    compute_rouge_scores,
    compute_bleu_score,
    compute_code_similarity
)
from .benchmarks import (
    BenchmarkRunner,
    BenchmarkConfig,
    SQuADBenchmark,
    GSM8KBenchmark,
    ARCBenchmark,
    MBPPBenchmark,
    HumanEvalBenchmark,
    create_benchmark,
    run_benchmark_suite
)
from .few_shot import (
    FewShotEvaluator,
    FewShotConfig,
    FewShotResults,
    create_few_shot_episodes,
    evaluate_few_shot_performance
)

__all__ = [
    # Main evaluator
    'ASNEvaluator',
    'EvaluationConfig',
    'EvaluationResults',
    'ModelEvaluator',
    
    # Metrics
    'MetricsCalculator',
    'QAMetrics',
    'MathMetrics',
    'ReasoningMetrics',
    'CodeMetrics',
    'CompressionMetrics',
    'compute_exact_match',
    'compute_f1_score',
    'compute_rouge_scores',
    'compute_bleu_score',
    'compute_code_similarity',
    
    # Benchmarks
    'BenchmarkRunner',
    'BenchmarkConfig',
    'SQuADBenchmark',
    'GSM8KBenchmark',
    'ARCBenchmark',
    'MBPPBenchmark',
    'HumanEvalBenchmark',
    'create_benchmark',
    'run_benchmark_suite',
    
    # Few-shot evaluation
    'FewShotEvaluator',
    'FewShotConfig',
    'FewShotResults',
    'create_few_shot_episodes',
    'evaluate_few_shot_performance',
]

# Version info
__version__ = "1.0.0"

# Default configurations
DEFAULT_EVALUATION_CONFIG = {
    'batch_size': 8,
    'max_samples': 1000,
    'num_workers': 4,
    'device': 'auto',
    'metrics': ['accuracy', 'f1', 'rouge'],
    'save_predictions': True,
    'statistical_tests': True
}

DEFAULT_BENCHMARK_CONFIG = {
    'datasets': ['squad', 'gsm8k', 'arc_easy', 'mbpp'],
    'max_samples_per_dataset': 500,
    'cross_validation': True,
    'cv_folds': 5,
    'confidence_level': 0.95
}

DEFAULT_FEW_SHOT_CONFIG = {
    'k_values': [1, 3, 5, 10],
    'num_episodes': 100,
    'support_query_split': 0.5,
    'random_seed': 42
}

def get_default_config(component: str = 'evaluation'):
    """
    Get default configuration for evaluation components.
    
    Args:
        component: Component name ('evaluation', 'benchmark', 'few_shot')
        
    Returns:
        dict: Default configuration
    """
    configs = {
        'evaluation': DEFAULT_EVALUATION_CONFIG,
        'benchmark': DEFAULT_BENCHMARK_CONFIG,
        'few_shot': DEFAULT_FEW_SHOT_CONFIG
    }
    
    return configs.get(component, DEFAULT_EVALUATION_CONFIG)


def create_evaluator(evaluator_type: str = 'asn', config: dict = None, **kwargs):
    """
    Factory function to create evaluators.
    
    Args:
        evaluator_type: Type of evaluator
        config: Configuration dictionary
        **kwargs: Additional arguments
        
    Returns:
        Evaluator instance
    """
    if config is None:
        config = get_default_config('evaluation')
    config.update(kwargs)
    
    if evaluator_type.lower() == 'asn':
        return ASNEvaluator(**config)
    elif evaluator_type.lower() == 'model':
        return ModelEvaluator(**config)
    elif evaluator_type.lower() == 'few_shot':
        return FewShotEvaluator(**config)
    else:
        raise ValueError(f"Unknown evaluator type: {evaluator_type}")


def quick_evaluate(model, dataset, task_type: str, **kwargs):
    """
    Quick evaluation function for common use cases.
    
    Args:
        model: Model to evaluate
        dataset: Dataset for evaluation
        task_type: Type of task
        **kwargs: Additional evaluation parameters
        
    Returns:
        Evaluation results
    """
    config = get_default_config('evaluation')
    config.update(kwargs)
    
    evaluator = ASNEvaluator(config)
    return evaluator.evaluate(model, dataset, task_type)


# Utility functions
def compare_models(models: dict, dataset, task_type: str, **kwargs):
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: Dictionary of {model_name: model} pairs
        dataset: Evaluation dataset
        task_type: Type of task
        **kwargs: Additional parameters
        
    Returns:
        Comparison results
    """
    results = {}
    evaluator = ASNEvaluator(**kwargs)
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        results[name] = evaluator.evaluate(model, dataset, task_type)
    
    return results


def statistical_significance_test(results1, results2, metric: str = 'accuracy', 
                                alpha: float = 0.05):
    """
    Test statistical significance between two evaluation results.
    
    Args:
        results1: First evaluation results
        results2: Second evaluation results
        metric: Metric to compare
        alpha: Significance level
        
    Returns:
        Statistical test results
    """
    from scipy import stats
    import numpy as np
    
    # Extract metric values
    if isinstance(results1, dict) and metric in results1:
        values1 = results1[metric] if isinstance(results1[metric], list) else [results1[metric]]
    else:
        values1 = [results1]
    
    if isinstance(results2, dict) and metric in results2:
        values2 = results2[metric] if isinstance(results2[metric], list) else [results2[metric]]
    else:
        values2 = [results2]
    
    # Perform t-test
    statistic, p_value = stats.ttest_ind(values1, values2)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'mean_diff': np.mean(values1) - np.mean(values2),
        'effect_size': (np.mean(values1) - np.mean(values2)) / np.sqrt((np.var(values1) + np.var(values2)) / 2)
    }