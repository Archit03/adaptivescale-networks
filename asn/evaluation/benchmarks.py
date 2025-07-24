"""
Benchmarks module for AdaptiveScale Networks.

This module provides comprehensive benchmark implementations for evaluating
ASN models across various tasks including QA, Math, Reasoning, and Code generation.
It includes dataset-specific benchmarks and utilities for running benchmark suites.
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    from transformers import AutoTokenizer, PreTrainedModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from ..data.datasets import create_dataset, QADatasetConfig, MathDatasetConfig, ReasoningDatasetConfig, CodeDatasetConfig
from .metrics import MetricsCalculator, compute_exact_match, compute_f1_score

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    
    # Dataset settings
    max_samples: int = 1000
    batch_size: int = 8
    num_workers: int = 4
    seed: int = 42
    
    # Evaluation settings
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Statistical analysis
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    statistical_tests: bool = True
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    cv_strategy: str = "stratified"  # stratified, random, time_series
    
    # Output settings
    save_predictions: bool = True
    save_detailed_results: bool = True
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    
    # Performance settings
    timeout_per_sample: float = 30.0
    max_memory_gb: float = 16.0
    
    # Debugging
    debug_mode: bool = False
    verbose: bool = False


@dataclass
class BenchmarkResults:
    """Results from benchmark evaluation."""
    
    benchmark_name: str
    dataset_name: str
    model_name: str
    task_type: str
    
    # Core metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Statistical analysis
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, Any] = field(default_factory=dict)
    
    # Performance data
    num_samples: int = 0
    total_time: float = 0.0
    avg_time_per_sample: float = 0.0
    throughput: float = 0.0
    
    # Detailed results
    predictions: List[str] = field(default_factory=list)
    ground_truths: List[str] = field(default_factory=list)
    sample_scores: List[float] = field(default_factory=list)
    
    # Error analysis
    error_analysis: Dict[str, Any] = field(default_factory=dict)
    failure_cases: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    timestamp: str = ""
    config: Optional[BenchmarkConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'benchmark_name': self.benchmark_name,
            'dataset_name': self.dataset_name,
            'model_name': self.model_name,
            'task_type': self.task_type,
            'metrics': self.metrics,
            'confidence_intervals': self.confidence_intervals,
            'statistical_significance': self.statistical_significance,
            'num_samples': self.num_samples,
            'total_time': self.total_time,
            'avg_time_per_sample': self.avg_time_per_sample,
            'throughput': self.throughput,
            'error_analysis': self.error_analysis,
            'timestamp': self.timestamp
        }
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"=== {self.benchmark_name} Results ===",
            f"Dataset: {self.dataset_name}",
            f"Model: {self.model_name}",
            f"Task: {self.task_type}",
            f"Samples: {self.num_samples}",
            f"Time: {self.total_time:.2f}s ({self.throughput:.2f} samples/s)",
            "",
            "Metrics:"
        ]
        
        for metric, value in self.metrics.items():
            ci = self.confidence_intervals.get(metric, (None, None))
            if ci[0] is not None:
                lines.append(f"  {metric}: {value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                lines.append(f"  {metric}: {value:.4f}")
        
        return "\n".join(lines)


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._get_device()
        self.metrics_calculator = MetricsCalculator()
        
        # Set up output directory
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} benchmark")
    
    def _get_device(self) -> torch.device:
        """Get computation device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)
    
    @abstractmethod
    def load_dataset(self) -> Any:
        """Load the benchmark dataset."""
        pass
    
    @abstractmethod
    def evaluate_model(self, model, dataset) -> BenchmarkResults:
        """Evaluate model on the benchmark."""
        pass
    
    def run_benchmark(self, model, tokenizer=None, **kwargs) -> BenchmarkResults:
        """
        Run complete benchmark evaluation.
        
        Args:
            model: Model to evaluate
            tokenizer: Optional tokenizer
            **kwargs: Additional arguments
            
        Returns:
            Benchmark results
        """
        # Set random seeds for reproducibility
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        
        # Load dataset
        dataset = self.load_dataset()
        
        # Prepare model
        model = self._prepare_model(model)
        
        # Run evaluation
        start_time = time.time()
        results = self.evaluate_model(model, dataset)
        total_time = time.time() - start_time
        
        # Update timing information
        results.total_time = total_time
        results.avg_time_per_sample = total_time / max(results.num_samples, 1)
        results.throughput = results.num_samples / total_time if total_time > 0 else 0
        results.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        results.config = self.config
        
        # Perform statistical analysis
        if self.config.statistical_tests:
            results = self._compute_statistical_analysis(results)
        
        # Save results if requested
        if self.config.save_detailed_results:
            self._save_results(results)
        
        logger.info(f"Benchmark completed: {results.summary()}")
        return results
    
    def _prepare_model(self, model):
        """Prepare model for evaluation."""
        model.eval()
        model = model.to(self.device)
        
        # Enable mixed precision if requested
        if self.config.mixed_precision and hasattr(torch.cuda, 'amp'):
            model = torch.cuda.amp.autocast()(model)
        
        # Compile model if requested (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            model = torch.compile(model)
        
        return model
    
    def _compute_statistical_analysis(self, results: BenchmarkResults) -> BenchmarkResults:
        """Compute statistical analysis including confidence intervals."""
        if not results.sample_scores:
            return results
        
        from scipy import stats
        
        # Compute confidence intervals using bootstrap
        for metric_name, metric_value in results.metrics.items():
            if metric_name in ['accuracy', 'f1_score', 'exact_match']:
                # Bootstrap confidence interval
                scores = np.array(results.sample_scores)
                bootstrap_means = []
                
                for _ in range(self.config.bootstrap_samples):
                    bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                alpha = 1 - self.config.confidence_level
                lower = np.percentile(bootstrap_means, 100 * alpha / 2)
                upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
                
                results.confidence_intervals[metric_name] = (lower, upper)
        
        return results
    
    def _save_results(self, results: BenchmarkResults):
        """Save benchmark results to file."""
        # Save main results
        results_path = self.output_dir / f"{results.benchmark_name}_{results.model_name}_{results.timestamp.replace(':', '-')}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        # Save predictions if requested
        if self.config.save_predictions and results.predictions:
            pred_path = self.output_dir / f"{results.benchmark_name}_{results.model_name}_predictions.json"
            
            predictions_data = {
                'predictions': results.predictions,
                'ground_truths': results.ground_truths,
                'sample_scores': results.sample_scores
            }
            
            with open(pred_path, 'w') as f:
                json.dump(predictions_data, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def _generate_predictions(self, model, dataset, tokenizer=None) -> Tuple[List[str], List[str], List[float]]:
        """Generate predictions from model."""
        predictions = []
        ground_truths = []
        sample_scores = []
        
        # Create data loader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if self.config.debug_mode and batch_idx >= 5:
                    break
                
                try:
                    # Move batch to device
                    if isinstance(batch, dict):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                               for k, v in batch.items()}
                    
                    # Generate predictions
                    batch_predictions = self._model_generate(model, batch, tokenizer)
                    batch_ground_truths = self._extract_ground_truths(batch)
                    
                    # Compute sample scores
                    batch_scores = self._compute_sample_scores(batch_predictions, batch_ground_truths)
                    
                    predictions.extend(batch_predictions)
                    ground_truths.extend(batch_ground_truths)
                    sample_scores.extend(batch_scores)
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    if self.config.debug_mode:
                        raise
                    continue
        
        return predictions, ground_truths, sample_scores
    
    @abstractmethod
    def _model_generate(self, model, batch, tokenizer=None) -> List[str]:
        """Generate predictions from model for a batch."""
        pass
    
    @abstractmethod
    def _extract_ground_truths(self, batch) -> List[str]:
        """Extract ground truth answers from batch."""
        pass
    
    @abstractmethod
    def _compute_sample_scores(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """Compute per-sample scores."""
        pass


class SQuADBenchmark(BaseBenchmark):
    """SQuAD (Stanford Question Answering Dataset) benchmark."""
    
    def __init__(self, config: BenchmarkConfig, version: str = "1.1"):
        super().__init__(config)
        self.version = version
        self.benchmark_name = f"SQuAD_{version}"
        self.task_type = "qa"
    
    def load_dataset(self):
        """Load SQuAD dataset."""
        dataset_config = QADatasetConfig(
            dataset_name="squad" if self.version == "1.1" else "squad_v2",
            split="validation",
            max_examples=self.config.max_samples
        )
        
        return create_dataset("qa", "squad", dataset_config)
    
    def evaluate_model(self, model, dataset) -> BenchmarkResults:
        """Evaluate model on SQuAD."""
        # Generate predictions
        predictions, ground_truths, sample_scores = self._generate_predictions(model, dataset)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_qa_metrics(predictions, ground_truths)
        
        # Create results
        results = BenchmarkResults(
            benchmark_name=self.benchmark_name,
            dataset_name="SQuAD",
            model_name=getattr(model, 'name', 'Unknown'),
            task_type=self.task_type,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions if self.config.save_predictions else [],
            ground_truths=ground_truths if self.config.save_predictions else [],
            sample_scores=sample_scores
        )
        
        return results
    
    def _model_generate(self, model, batch, tokenizer=None) -> List[str]:
        """Generate QA predictions."""
        # This is a simplified implementation
        # In practice, you'd need to handle the specific model's generation method
        
        if hasattr(model, 'generate') and tokenizer is not None:
            # For generative models
            input_ids = batch.get('input_ids', batch.get('problem_input_ids'))
            attention_mask = batch.get('attention_mask', batch.get('problem_attention_mask'))
            
            with torch.cuda.amp.autocast() if self.config.mixed_precision else torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1
                )
            
            # Decode predictions
            predictions = []
            for output in outputs:
                # Skip input tokens
                generated_tokens = output[input_ids.shape[1]:]
                pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predictions.append(pred_text.strip())
            
            return predictions
        else:
            # For classification/span-based models
            # This would need to be implemented based on your specific model architecture
            return ["placeholder_answer"] * batch['input_ids'].size(0)
    
    def _extract_ground_truths(self, batch) -> List[str]:
        """Extract ground truth answers."""
        if 'answer_text' in batch:
            return batch['answer_text'] if isinstance(batch['answer_text'], list) else [batch['answer_text']]
        elif 'answers' in batch:
            # Handle multiple answers format
            answers = batch['answers']
            if isinstance(answers[0], list):
                return [ans[0] if ans else "" for ans in answers]
            return answers
        else:
            return [""] * len(batch.get('input_ids', []))
    
    def _compute_sample_scores(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """Compute F1 scores for each sample."""
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            f1 = compute_f1_score(pred, truth)
            scores.append(f1)
        return scores


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K (Grade School Math 8K) benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.benchmark_name = "GSM8K"
        self.task_type = "math"
    
    def load_dataset(self):
        """Load GSM8K dataset."""
        dataset_config = MathDatasetConfig(
            dataset_name="gsm8k",
            split="test",
            max_examples=self.config.max_samples
        )
        
        return create_dataset("math", "gsm8k", dataset_config)
    
    def evaluate_model(self, model, dataset) -> BenchmarkResults:
        """Evaluate model on GSM8K."""
        predictions, ground_truths, sample_scores = self._generate_predictions(model, dataset)
        
        # Compute math-specific metrics
        metrics = self.metrics_calculator.compute_math_metrics(predictions, ground_truths)
        
        results = BenchmarkResults(
            benchmark_name=self.benchmark_name,
            dataset_name="GSM8K",
            model_name=getattr(model, 'name', 'Unknown'),
            task_type=self.task_type,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions if self.config.save_predictions else [],
            ground_truths=ground_truths if self.config.save_predictions else [],
            sample_scores=sample_scores
        )
        
        return results
    
    def _model_generate(self, model, batch, tokenizer=None) -> List[str]:
        """Generate math problem solutions."""
        if hasattr(model, 'generate') and tokenizer is not None:
            input_ids = batch.get('problem_input_ids', batch.get('input_ids'))
            attention_mask = batch.get('problem_attention_mask', batch.get('attention_mask'))
            
            with torch.cuda.amp.autocast() if self.config.mixed_precision else torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,  # Longer for math solutions
                    do_sample=True,
                    temperature=0.1,
                    top_p=0.9
                )
            
            predictions = []
            for output in outputs:
                generated_tokens = output[input_ids.shape[1]:]
                pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predictions.append(pred_text.strip())
            
            return predictions
        else:
            return ["placeholder_solution"] * batch['input_ids'].size(0)
    
    def _extract_ground_truths(self, batch) -> List[str]:
        """Extract ground truth answers."""
        if 'answer_text' in batch:
            return batch['answer_text'] if isinstance(batch['answer_text'], list) else [batch['answer_text']]
        return [""] * len(batch.get('input_ids', []))
    
    def _compute_sample_scores(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """Compute exact match scores for math problems."""
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            score = self._extract_and_compare_numbers(pred, truth)
            scores.append(score)
        return scores
    
    def _extract_and_compare_numbers(self, prediction: str, ground_truth: str) -> float:
        """Extract and compare numerical answers."""
        import re
        
        def extract_number(text):
            # Extract final numerical answer
            numbers = re.findall(r'[\d,]+\.?\d*', text.replace(',', ''))
            return float(numbers[-1]) if numbers else None
        
        pred_num = extract_number(prediction)
        truth_num = extract_number(ground_truth)
        
        if pred_num is not None and truth_num is not None:
            return 1.0 if abs(pred_num - truth_num) < 1e-6 else 0.0
        
        return 0.0


class ARCBenchmark(BaseBenchmark):
    """AI2 Reasoning Challenge (ARC) benchmark."""
    
    def __init__(self, config: BenchmarkConfig, subset: str = "ARC-Easy"):
        super().__init__(config)
        self.subset = subset
        self.benchmark_name = f"ARC_{subset.split('-')[1]}"
        self.task_type = "reasoning"
    
    def load_dataset(self):
        """Load ARC dataset."""
        dataset_config = ReasoningDatasetConfig(
            dataset_name="arc_easy" if "Easy" in self.subset else "arc_challenge",
            split="test",
            max_examples=self.config.max_samples
        )
        
        return create_dataset("reasoning", dataset_config.dataset_name, dataset_config)
    
    def evaluate_model(self, model, dataset) -> BenchmarkResults:
        """Evaluate model on ARC."""
        predictions, ground_truths, sample_scores = self._generate_predictions(model, dataset)
        
        # Compute reasoning metrics
        metrics = self.metrics_calculator.compute_reasoning_metrics(predictions, ground_truths)
        
        results = BenchmarkResults(
            benchmark_name=self.benchmark_name,
            dataset_name=f"ARC-{self.subset}",
            model_name=getattr(model, 'name', 'Unknown'),
            task_type=self.task_type,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions if self.config.save_predictions else [],
            ground_truths=ground_truths if self.config.save_predictions else [],
            sample_scores=sample_scores
        )
        
        return results
    
    def _model_generate(self, model, batch, tokenizer=None) -> List[str]:
        """Generate reasoning predictions."""
        if hasattr(model, 'generate') and tokenizer is not None:
            input_ids = batch.get('question_input_ids', batch.get('input_ids'))
            attention_mask = batch.get('question_attention_mask', batch.get('attention_mask'))
            
            with torch.cuda.amp.autocast() if self.config.mixed_precision else torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.1
                )
            
            predictions = []
            for output in outputs:
                generated_tokens = output[input_ids.shape[1]:]
                pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                # Extract choice letter (A, B, C, D)
                choice_match = re.search(r'\b[A-D]\b', pred_text)
                pred_choice = choice_match.group(0) if choice_match else pred_text.strip()[:1]
                predictions.append(pred_choice)
            
            return predictions
        else:
            return ["A"] * batch['input_ids'].size(0)
    
    def _extract_ground_truths(self, batch) -> List[str]:
        """Extract ground truth choices."""
        if 'answer_text' in batch:
            answers = batch['answer_text'] if isinstance(batch['answer_text'], list) else [batch['answer_text']]
            return [ans.strip()[:1] if ans else "A" for ans in answers]
        return ["A"] * len(batch.get('input_ids', []))
    
    def _compute_sample_scores(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """Compute exact match scores for multiple choice."""
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            score = 1.0 if pred.upper() == truth.upper() else 0.0
            scores.append(score)
        return scores


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval code generation benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.benchmark_name = "HumanEval"
        self.task_type = "code"
    
    def load_dataset(self):
        """Load HumanEval dataset."""
        dataset_config = CodeDatasetConfig(
            dataset_name="humaneval",
            split="test",
            max_examples=self.config.max_samples
        )
        
        return create_dataset("code", "humaneval", dataset_config)
    
    def evaluate_model(self, model, dataset) -> BenchmarkResults:
        """Evaluate model on HumanEval."""
        predictions, ground_truths, sample_scores = self._generate_predictions(model, dataset)
        
        # Compute code-specific metrics
        metrics = self.metrics_calculator.compute_code_metrics(predictions, ground_truths)
        
        results = BenchmarkResults(
            benchmark_name=self.benchmark_name,
            dataset_name="HumanEval",
            model_name=getattr(model, 'name', 'Unknown'),
            task_type=self.task_type,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions if self.config.save_predictions else [],
            ground_truths=ground_truths if self.config.save_predictions else [],
            sample_scores=sample_scores
        )
        
        return results
    
    def _model_generate(self, model, batch, tokenizer=None) -> List[str]:
        """Generate code completions."""
        if hasattr(model, 'generate') and tokenizer is not None:
            input_ids = batch.get('prompt_input_ids', batch.get('input_ids'))
            attention_mask = batch.get('prompt_attention_mask', batch.get('attention_mask'))
            
            with torch.cuda.amp.autocast() if self.config.mixed_precision else torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=300,  # Longer for code
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.95
                )
            
            predictions = []
            for output in outputs:
                generated_tokens = output[input_ids.shape[1]:]
                pred_code = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                predictions.append(pred_code.strip())
            
            return predictions
        else:
            return ["def solution():\n    pass"] * batch['input_ids'].size(0)
    
    def _extract_ground_truths(self, batch) -> List[str]:
        """Extract ground truth code."""
        if 'solution_text' in batch:
            return batch['solution_text'] if isinstance(batch['solution_text'], list) else [batch['solution_text']]
        return [""] * len(batch.get('input_ids', []))
    
    def _compute_sample_scores(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """Compute code execution scores."""
        scores = []
        for pred, truth in zip(predictions, ground_truths):
            # This would need actual code execution testing
            # For now, use a simple syntax check
            try:
                compile(pred, '<string>', 'exec')
                scores.append(1.0)  # Syntax is valid
            except SyntaxError:
                scores.append(0.0)  # Syntax error
        return scores


class MBPPBenchmark(BaseBenchmark):
    """Mostly Basic Python Problems (MBPP) benchmark."""
    
    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.benchmark_name = "MBPP"
        self.task_type = "code"
    
    def load_dataset(self):
        """Load MBPP dataset."""
        dataset_config = CodeDatasetConfig(
            dataset_name="mbpp",
            split="test",
            max_examples=self.config.max_samples
        )
        
        return create_dataset("code", "mbpp", dataset_config)
    
    def evaluate_model(self, model, dataset) -> BenchmarkResults:
        """Evaluate model on MBPP."""
        predictions, ground_truths, sample_scores = self._generate_predictions(model, dataset)
        
        metrics = self.metrics_calculator.compute_code_metrics(predictions, ground_truths)
        
        results = BenchmarkResults(
            benchmark_name=self.benchmark_name,
            dataset_name="MBPP",
            model_name=getattr(model, 'name', 'Unknown'),
            task_type=self.task_type,
            metrics=metrics,
            num_samples=len(predictions),
            predictions=predictions if self.config.save_predictions else [],
            ground_truths=ground_truths if self.config.save_predictions else [],
            sample_scores=sample_scores
        )
        
        return results
    
    def _model_generate(self, model, batch, tokenizer=None) -> List[str]:
        """Generate code for MBPP problems."""
        # Similar to HumanEval but potentially different formatting
        return HumanEvalBenchmark._model_generate(self, model, batch, tokenizer)
    
    def _extract_ground_truths(self, batch) -> List[str]:
        """Extract ground truth code."""
        return HumanEvalBenchmark._extract_ground_truths(self, batch)
    
    def _compute_sample_scores(self, predictions: List[str], ground_truths: List[str]) -> List[float]:
        """Compute code execution scores."""
        return HumanEvalBenchmark._compute_sample_scores(self, predictions, ground_truths)


class BenchmarkRunner:
    """Runs multiple benchmarks and compiles results."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = {}
        
        # Available benchmarks
        self.available_benchmarks = {
            'squad': SQuADBenchmark,
            'squad_v2': lambda config: SQuADBenchmark(config, version="2.0"),
            'gsm8k': GSM8KBenchmark,
            'arc_easy': lambda config: ARCBenchmark(config, subset="ARC-Easy"),
            'arc_challenge': lambda config: ARCBenchmark(config, subset="ARC-Challenge"),
            'humaneval': HumanEvalBenchmark,
            'mbpp': MBPPBenchmark
        }
        
        logger.info(f"BenchmarkRunner initialized with {len(self.available_benchmarks)} benchmarks")
    
    def run_benchmark(self, benchmark_name: str, model, tokenizer=None, **kwargs) -> BenchmarkResults:
        """
        Run a single benchmark.
        
        Args:
            benchmark_name: Name of benchmark to run
            model: Model to evaluate
            tokenizer: Optional tokenizer
            **kwargs: Additional arguments
            
        Returns:
            Benchmark results
        """
        if benchmark_name not in self.available_benchmarks:
            raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(self.available_benchmarks.keys())}")
        
        # Create benchmark instance
        benchmark_class = self.available_benchmarks[benchmark_name]
        benchmark = benchmark_class(self.config)
        
        # Run benchmark
        logger.info(f"Running benchmark: {benchmark_name}")
        results = benchmark.run_benchmark(model, tokenizer, **kwargs)
        
        # Store results
        self.results[benchmark_name] = results
        
        return results
    
    def run_benchmark_suite(self, benchmark_names: List[str], model, tokenizer=None, 
                           **kwargs) -> Dict[str, BenchmarkResults]:
        """
        Run multiple benchmarks.
        
        Args:
            benchmark_names: List of benchmark names
            model: Model to evaluate
            tokenizer: Optional tokenizer
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of benchmark results
        """
        suite_results = {}
        
        for benchmark_name in benchmark_names:
            try:
                results = self.run_benchmark(benchmark_name, model, tokenizer, **kwargs)
                suite_results[benchmark_name] = results
                
                # Log progress
                logger.info(f"Completed {benchmark_name}: {results.metrics}")
                
            except Exception as e:
                logger.error(f"Failed to run benchmark {benchmark_name}: {e}")
                if self.config.debug_mode:
                    raise
                continue
        
        # Generate suite summary
        self._generate_suite_summary(suite_results)
        
        return suite_results
    
    def compare_models(self, models: Dict[str, Any], benchmark_names: List[str], 
                      tokenizers: Dict[str, Any] = None, **kwargs) -> Dict[str, Dict[str, BenchmarkResults]]:
        """
        Compare multiple models across benchmarks.
        
        Args:
            models: Dictionary of {model_name: model} pairs
            benchmark_names: List of benchmarks to run
            tokenizers: Optional dictionary of {model_name: tokenizer} pairs
            **kwargs: Additional arguments
            
        Returns:
            Nested dictionary of results: {model_name: {benchmark_name: results}}
        """
        comparison_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")
            
            tokenizer = tokenizers.get(model_name) if tokenizers else None
            model_results = self.run_benchmark_suite(benchmark_names, model, tokenizer, **kwargs)
            comparison_results[model_name] = model_results
        
        # Generate comparison report
        self._generate_comparison_report(comparison_results)
        
        return comparison_results
    
    def _generate_suite_summary(self, results: Dict[str, BenchmarkResults]):
        """Generate summary of benchmark suite results."""
        logger.info("=== Benchmark Suite Summary ===")
        
        for benchmark_name, result in results.items():
            logger.info(f"{benchmark_name}:")
            for metric, value in result.metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
            logger.info(f"  Time: {result.total_time:.2f}s")
            logger.info("")
    
    def _generate_comparison_report(self, results: Dict[str, Dict[str, BenchmarkResults]]):
        """Generate comparison report across models."""
        logger.info("=== Model Comparison Report ===")
        
        # Get all benchmarks and metrics
        all_benchmarks = set()
        all_metrics = set()
        
        for model_results in results.values():
            all_benchmarks.update(model_results.keys())
            for benchmark_results in model_results.values():
                all_metrics.update(benchmark_results.metrics.keys())
        
        # Create comparison table for each metric
        for metric in all_metrics:
            logger.info(f"\n{metric.upper()} Comparison:")
            logger.info("-" * 50)
            
            for benchmark in all_benchmarks:
                logger.info(f"\n{benchmark}:")
                
                benchmark_scores = {}
                for model_name, model_results in results.items():
                    if benchmark in model_results and metric in model_results[benchmark].metrics:
                        score = model_results[benchmark].metrics[metric]
                        benchmark_scores[model_name] = score
                
                # Sort by score
                sorted_scores = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
                
                for rank, (model_name, score) in enumerate(sorted_scores, 1):
                    logger.info(f"  {rank}. {model_name}: {score:.4f}")
    
    def save_all_results(self, output_path: Union[str, Path]):
        """Save all benchmark results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert results to serializable format
        serializable_results = {}
        for benchmark_name, results in self.results.items():
            serializable_results[benchmark_name] = results.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"All results saved to {output_path}")
    
    def generate_report(self, output_path: Union[str, Path] = None) -> str:
        """Generate comprehensive benchmark report."""
        report_lines = [
            "=" * 80,
            "ADAPTIVE SCALE NETWORKS - BENCHMARK REPORT",
            "=" * 80,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Benchmarks: {len(self.results)}",
            ""
        ]
        
        # Summary table
        if self.results:
            report_lines.extend([
                "SUMMARY",
                "-" * 40,
                f"{'Benchmark':<20} {'Samples':<10} {'Time(s)':<10} {'Accuracy':<10}",
                "-" * 50
            ])
            
            for benchmark_name, results in self.results.items():
                accuracy = results.metrics.get('accuracy', results.metrics.get('exact_match', 0.0))
                report_lines.append(
                    f"{benchmark_name:<20} {results.num_samples:<10} "
                    f"{results.total_time:<10.2f} {accuracy:<10.4f}"
                )
            
            report_lines.append("")
        
        # Detailed results
        for benchmark_name, results in self.results.items():
            report_lines.extend([
                f"BENCHMARK: {benchmark_name.upper()}",
                "-" * 40,
                f"Dataset: {results.dataset_name}",
                f"Task Type: {results.task_type}",
                f"Samples: {results.num_samples}",
                f"Total Time: {results.total_time:.2f}s",
                f"Throughput: {results.throughput:.2f} samples/s",
                "",
                "Metrics:"
            ])
            
            for metric, value in results.metrics.items():
                ci = results.confidence_intervals.get(metric, (None, None))
                if ci[0] is not None:
                    report_lines.append(f"  {metric}: {value:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
                else:
                    report_lines.append(f"  {metric}: {value:.4f}")
            
            report_lines.extend(["", ""])
        
        report = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report)
            
            logger.info(f"Report saved to {output_path}")
        
        return report
    
    def create_visualization(self, output_dir: Union[str, Path] = None):
        """Create visualization plots for benchmark results."""
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Accuracy comparison across benchmarks
            fig, ax = plt.subplots(figsize=(12, 6))
            
            benchmarks = []
            accuracies = []
            
            for benchmark_name, results in self.results.items():
                accuracy = results.metrics.get('accuracy', results.metrics.get('exact_match', 0.0))
                benchmarks.append(benchmark_name)
                accuracies.append(accuracy)
            
            bars = ax.bar(benchmarks, accuracies)
            ax.set_ylabel('Accuracy')
            ax.set_title('Benchmark Performance Comparison')
            ax.set_ylim(0, 1.0)
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'benchmark_accuracy.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 2. Performance vs Accuracy scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            throughputs = []
            accuracies = []
            labels = []
            
            for benchmark_name, results in self.results.items():
                accuracy = results.metrics.get('accuracy', results.metrics.get('exact_match', 0.0))
                throughput = results.throughput
                
                throughputs.append(throughput)
                accuracies.append(accuracy)
                labels.append(benchmark_name)
            
            scatter = ax.scatter(throughputs, accuracies, s=100, alpha=0.7)
            
            # Add labels
            for i, label in enumerate(labels):
                ax.annotate(label, (throughputs[i], accuracies[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            ax.set_xlabel('Throughput (samples/s)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Performance vs Accuracy Trade-off')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'performance_accuracy.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 3. Metrics heatmap
            if len(self.results) > 1:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # Collect all metrics
                all_metrics = set()
                for results in self.results.values():
                    all_metrics.update(results.metrics.keys())
                
                # Create data matrix
                data = []
                benchmark_names = []
                
                for benchmark_name, results in self.results.items():
                    row = []
                    for metric in sorted(all_metrics):
                        value = results.metrics.get(metric, 0.0)
                        row.append(value)
                    data.append(row)
                    benchmark_names.append(benchmark_name)
                
                # Create heatmap
                sns.heatmap(data, 
                           xticklabels=sorted(all_metrics),
                           yticklabels=benchmark_names,
                           annot=True, 
                           fmt='.3f',
                           cmap='YlOrRd',
                           ax=ax)
                
                ax.set_title('Benchmark Metrics Heatmap')
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(output_dir / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
                plt.show()
            
            logger.info("Visualizations created successfully")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available for visualization")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")


# Factory functions and utilities
def create_benchmark(benchmark_name: str, config: BenchmarkConfig = None, **kwargs) -> BaseBenchmark:
    """
    Create a benchmark instance.
    
    Args:
        benchmark_name: Name of benchmark
        config: Benchmark configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Benchmark instance
    """
    if config is None:
        config = BenchmarkConfig(**kwargs)
    
    benchmark_map = {
        'squad': SQuADBenchmark,
        'squad_v2': lambda c: SQuADBenchmark(c, version="2.0"),
        'gsm8k': GSM8KBenchmark,
        'arc_easy': lambda c: ARCBenchmark(c, subset="ARC-Easy"),
        'arc_challenge': lambda c: ARCBenchmark(c, subset="ARC-Challenge"),
        'humaneval': HumanEvalBenchmark,
        'mbpp': MBPPBenchmark
    }
    
    if benchmark_name not in benchmark_map:
        raise ValueError(f"Unknown benchmark: {benchmark_name}. Available: {list(benchmark_map.keys())}")
    
    benchmark_class = benchmark_map[benchmark_name]
    return benchmark_class(config)


def run_benchmark_suite(model, benchmark_names: List[str] = None, 
                       tokenizer=None, config: BenchmarkConfig = None,
                       **kwargs) -> Dict[str, BenchmarkResults]:
    """
    Convenience function to run a benchmark suite.
    
    Args:
        model: Model to evaluate
        benchmark_names: List of benchmarks to run (default: all)
        tokenizer: Optional tokenizer
        config: Benchmark configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary of benchmark results
    """
    if config is None:
        config = BenchmarkConfig(**kwargs)
    
    if benchmark_names is None:
        benchmark_names = ['squad', 'gsm8k', 'arc_easy', 'humaneval']
    
    runner = BenchmarkRunner(config)
    return runner.run_benchmark_suite(benchmark_names, model, tokenizer)


def compare_benchmark_results(results1: BenchmarkResults, results2: BenchmarkResults, 
                             significance_level: float = 0.05) -> Dict[str, Any]:
    """
    Compare two benchmark results for statistical significance.
    
    Args:
        results1: First benchmark results
        results2: Second benchmark results
        significance_level: Statistical significance level
        
    Returns:
        Comparison results with statistical tests
    """
    from scipy import stats
    import numpy as np
    
    comparison = {
        'benchmark': results1.benchmark_name,
        'model1': results1.model_name,
        'model2': results2.model_name,
        'metrics_comparison': {},
        'overall_winner': None,
        'significant_differences': []
    }
    
    # Compare each metric
    better_count = {'model1': 0, 'model2': 0}
    
    for metric in results1.metrics.keys():
        if metric in results2.metrics:
            value1 = results1.metrics[metric]
            value2 = results2.metrics[metric]
            
            # Determine winner for this metric
            winner = 'model1' if value1 > value2 else 'model2'
            better_count[winner] += 1
            
            metric_comparison = {
                'metric': metric,
                'value1': value1,
                'value2': value2,
                'difference': value1 - value2,
                'relative_difference': (value1 - value2) / max(value2, 1e-10),
                'winner': winner
            }
            
            # Statistical test if sample scores available
            if (results1.sample_scores and results2.sample_scores and 
                len(results1.sample_scores) > 1 and len(results2.sample_scores) > 1):
                
                # Perform t-test
                statistic, p_value = stats.ttest_ind(results1.sample_scores, results2.sample_scores)
                
                metric_comparison.update({
                    'statistical_test': {
                        'test': 't-test',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < significance_level,
                        'significance_level': significance_level
                    }
                })
                
                if p_value < significance_level:
                    comparison['significant_differences'].append(metric)
            
            comparison['metrics_comparison'][metric] = metric_comparison
    
    # Determine overall winner
    if better_count['model1'] > better_count['model2']:
        comparison['overall_winner'] = 'model1'
    elif better_count['model2'] > better_count['model1']:
        comparison['overall_winner'] = 'model2'
    else:
        comparison['overall_winner'] = 'tie'
    
    return comparison


def load_benchmark_results(file_path: Union[str, Path]) -> Dict[str, BenchmarkResults]:
    """
    Load benchmark results from file.
    
    Args:
        file_path: Path to results file
        
    Returns:
        Dictionary of benchmark results
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for benchmark_name, result_dict in data.items():
        # Reconstruct BenchmarkResults object
        result = BenchmarkResults(
            benchmark_name=result_dict['benchmark_name'],
            dataset_name=result_dict['dataset_name'],
            model_name=result_dict['model_name'],
            task_type=result_dict['task_type'],
            metrics=result_dict['metrics'],
            confidence_intervals=result_dict.get('confidence_intervals', {}),
            statistical_significance=result_dict.get('statistical_significance', {}),
            num_samples=result_dict['num_samples'],
            total_time=result_dict['total_time'],
            avg_time_per_sample=result_dict['avg_time_per_sample'],
            throughput=result_dict['throughput'],
            error_analysis=result_dict.get('error_analysis', {}),
            timestamp=result_dict['timestamp']
        )
        results[benchmark_name] = result
    
    return results


def validate_model_interface(model, tokenizer=None) -> Dict[str, bool]:
    """
    Validate that model has required interface for benchmarking.
    
    Args:
        model: Model to validate
        tokenizer: Optional tokenizer to validate
        
    Returns:
        Dictionary of validation results
    """
    validation = {
        'has_eval_method': hasattr(model, 'eval'),
        'has_to_method': hasattr(model, 'to'),
        'has_forward_method': hasattr(model, 'forward') or hasattr(model, '__call__'),
        'has_generate_method': hasattr(model, 'generate'),
        'is_pytorch_module': isinstance(model, nn.Module),
        'tokenizer_available': tokenizer is not None,
        'tokenizer_has_decode': hasattr(tokenizer, 'decode') if tokenizer else False
    }
    
    # Check if model is ready for evaluation
    validation['ready_for_evaluation'] = all([
        validation['has_eval_method'],
        validation['has_to_method'],
        validation['has_forward_method']
    ])
    
    validation['ready_for_generation'] = all([
        validation['ready_for_evaluation'],
        validation['has_generate_method'],
        validation['tokenizer_has_decode']
    ])
    
    return validation


def get_benchmark_requirements(benchmark_name: str) -> Dict[str, Any]:
    """
    Get requirements for a specific benchmark.
    
    Args:
        benchmark_name: Name of benchmark
        
    Returns:
        Dictionary of requirements
    """
    requirements = {
        'squad': {
            'task_type': 'qa',
            'requires_generation': True,
            'requires_tokenizer': True,
            'expected_input_format': 'question_context',
            'expected_output_format': 'text_answer'
        },
        'gsm8k': {
            'task_type': 'math',
            'requires_generation': True,
            'requires_tokenizer': True,
            'expected_input_format': 'math_problem',
            'expected_output_format': 'numerical_answer'
        },
        'arc_easy': {
            'task_type': 'reasoning',
            'requires_generation': True,
            'requires_tokenizer': True,
            'expected_input_format': 'multiple_choice',
            'expected_output_format': 'choice_letter'
        },
        'humaneval': {
            'task_type': 'code',
            'requires_generation': True,
            'requires_tokenizer': True,
            'expected_input_format': 'code_prompt',
            'expected_output_format': 'python_code'
        }
    }
    
    return requirements.get(benchmark_name, {})


def create_benchmark_suite(task_types: List[str] = None, 
                          difficulty_levels: List[str] = None,
                          config: BenchmarkConfig = None) -> List[str]:
    """
    Create a comprehensive benchmark suite based on task types and difficulty.
    
    Args:
        task_types: Task types to include ['qa', 'math', 'reasoning', 'code']
        difficulty_levels: Difficulty levels ['easy', 'medium', 'hard']
        config: Benchmark configuration
        
    Returns:
        List of benchmark names
    """
    if task_types is None:
        task_types = ['qa', 'math', 'reasoning', 'code']
    
    if difficulty_levels is None:
        difficulty_levels = ['easy', 'medium', 'hard']
    
    benchmark_suite = []
    
    # Map task types to benchmarks
    task_benchmark_map = {
        'qa': ['squad'],
        'math': ['gsm8k'],
        'reasoning': ['arc_easy', 'arc_challenge'],
        'code': ['humaneval', 'mbpp']
    }
    
    # Map difficulty to benchmarks
    difficulty_map = {
        'easy': ['squad', 'arc_easy', 'mbpp'],
        'medium': ['gsm8k', 'humaneval'],
        'hard': ['squad_v2', 'arc_challenge']
    }
    
    # Build suite based on criteria
    for task_type in task_types:
        if task_type in task_benchmark_map:
            for benchmark in task_benchmark_map[task_type]:
                if any(benchmark in difficulty_map.get(level, []) for level in difficulty_levels):
                    if benchmark not in benchmark_suite:
                        benchmark_suite.append(benchmark)
    
    logger.info(f"Created benchmark suite with {len(benchmark_suite)} benchmarks: {benchmark_suite}")
    return benchmark_suite


def analyze_benchmark_coverage(results: Dict[str, BenchmarkResults]) -> Dict[str, Any]:
    """
    Analyze coverage and completeness of benchmark results.
    
    Args:
        results: Dictionary of benchmark results
        
    Returns:
        Coverage analysis
    """
    analysis = {
        'total_benchmarks': len(results),
        'task_coverage': defaultdict(int),
        'total_samples': 0,
        'total_time': 0.0,
        'avg_accuracy': 0.0,
        'benchmark_distribution': {},
        'performance_summary': {}
    }
    
    accuracies = []
    
    for benchmark_name, result in results.items():
        # Task coverage
        analysis['task_coverage'][result.task_type] += 1
        
        # Aggregate metrics
        analysis['total_samples'] += result.num_samples
        analysis['total_time'] += result.total_time
        
        # Accuracy tracking
        accuracy = result.metrics.get('accuracy', result.metrics.get('exact_match', 0.0))
        accuracies.append(accuracy)
        
        # Distribution
        analysis['benchmark_distribution'][benchmark_name] = {
            'samples': result.num_samples,
            'time': result.total_time,
            'accuracy': accuracy
        }
    
    if accuracies:
        analysis['avg_accuracy'] = np.mean(accuracies)
        analysis['std_accuracy'] = np.std(accuracies)
        analysis['min_accuracy'] = min(accuracies)
        analysis['max_accuracy'] = max(accuracies)
    
    # Performance summary
    analysis['performance_summary'] = {
        'total_throughput': analysis['total_samples'] / analysis['total_time'] if analysis['total_time'] > 0 else 0,
        'avg_time_per_sample': analysis['total_time'] / analysis['total_samples'] if analysis['total_samples'] > 0 else 0,
        'task_coverage_ratio': len(analysis['task_coverage']) / 4,  # Assuming 4 main task types
    }
    
    return dict(analysis)


def export_benchmark_comparison(results_dict: Dict[str, Dict[str, BenchmarkResults]], 
                               output_path: Union[str, Path], 
                               format: str = 'html') -> None:
    """
    Export benchmark comparison in various formats.
    
    Args:
        results_dict: Nested dictionary of {model_name: {benchmark_name: results}}
        output_path: Output file path
        format: Export format ('html', 'csv', 'json', 'latex')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'html':
        _export_html_comparison(results_dict, output_path)
    elif format == 'csv':
        _export_csv_comparison(results_dict, output_path)
    elif format == 'json':
        _export_json_comparison(results_dict, output_path)
    elif format == 'latex':
        _export_latex_comparison(results_dict, output_path)
    else:
        raise ValueError(f"Unsupported export format: {format}")
    
    logger.info(f"Benchmark comparison exported to {output_path}")


def _export_html_comparison(results_dict: Dict[str, Dict[str, BenchmarkResults]], 
                           output_path: Path) -> None:
    """Export comparison as HTML table."""
    html_content = [
        "<!DOCTYPE html>",
        "<html><head><title>ASN Benchmark Comparison</title>",
        "<style>",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }",
        "th { background-color: #f2f2f2; }",
        ".best { background-color: #d4edda; font-weight: bold; }",
        "</style></head><body>",
        "<h1>ASN Benchmark Comparison Report</h1>",
        f"<p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>"
    ]
    
    # Get all benchmarks and models
    all_benchmarks = set()
    all_models = list(results_dict.keys())
    
    for model_results in results_dict.values():
        all_benchmarks.update(model_results.keys())
    
    all_benchmarks = sorted(all_benchmarks)
    
    # Create table for each metric
    all_metrics = set()
    for model_results in results_dict.values():
        for benchmark_results in model_results.values():
            all_metrics.update(benchmark_results.metrics.keys())
    
    for metric in sorted(all_metrics):
        html_content.extend([
            f"<h2>{metric.replace('_', ' ').title()}</h2>",
            "<table>",
            "<tr><th>Model</th>" + "".join(f"<th>{benchmark}</th>" for benchmark in all_benchmarks) + "</tr>"
        ])
        
        for model_name in all_models:
            row = [f"<td>{model_name}</td>"]
            
            for benchmark in all_benchmarks:
                if (model_name in results_dict and 
                    benchmark in results_dict[model_name] and 
                    metric in results_dict[model_name][benchmark].metrics):
                    
                    value = results_dict[model_name][benchmark].metrics[metric]
                    row.append(f"<td>{value:.4f}</td>")
                else:
                    row.append("<td>-</td>")
            
            html_content.append("<tr>" + "".join(row) + "</tr>")
        
        html_content.append("</table><br>")
    
    html_content.extend(["</body></html>"])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(html_content))


def _export_csv_comparison(results_dict: Dict[str, Dict[str, BenchmarkResults]], 
                          output_path: Path) -> None:
    """Export comparison as CSV."""
    import pandas as pd
    
    # Flatten results into rows
    rows = []
    for model_name, model_results in results_dict.items():
        for benchmark_name, benchmark_results in model_results.items():
            row = {
                'model': model_name,
                'benchmark': benchmark_name,
                'task_type': benchmark_results.task_type,
                'num_samples': benchmark_results.num_samples,
                'total_time': benchmark_results.total_time,
                'throughput': benchmark_results.throughput
            }
            row.update(benchmark_results.metrics)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)


def _export_json_comparison(results_dict: Dict[str, Dict[str, BenchmarkResults]], 
                           output_path: Path) -> None:
    """Export comparison as JSON."""
    # Convert to serializable format
    serializable = {}
    for model_name, model_results in results_dict.items():
        serializable[model_name] = {}
        for benchmark_name, benchmark_results in model_results.items():
            serializable[model_name][benchmark_name] = benchmark_results.to_dict()
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)


def _export_latex_comparison(results_dict: Dict[str, Dict[str, BenchmarkResults]], 
                            output_path: Path) -> None:
    """Export comparison as LaTeX table."""
    # Get all benchmarks and models
    all_benchmarks = set()
    all_models = list(results_dict.keys())
    
    for model_results in results_dict.values():
        all_benchmarks.update(model_results.keys())
    
    all_benchmarks = sorted(all_benchmarks)
    
    latex_content = [
        "\\documentclass{article}",
        "\\usepackage{booktabs}",
        "\\usepackage{array}",
        "\\begin{document}",
        "\\title{ASN Benchmark Comparison}",
        "\\maketitle",
        ""
    ]
    
    # Create table for main accuracy metric
    latex_content.extend([
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * len(all_benchmarks)}}}",
        "\\toprule",
        "Model & " + " & ".join(all_benchmarks) + " \\\\",
        "\\midrule"
    ])
    
    for model_name in all_models:
        row = [model_name.replace('_', '\\_')]
        
        for benchmark in all_benchmarks:
            if (model_name in results_dict and 
                benchmark in results_dict[model_name]):
                
                metrics = results_dict[model_name][benchmark].metrics
                accuracy = metrics.get('accuracy', metrics.get('exact_match', 0.0))
                row.append(f"{accuracy:.3f}")
            else:
                row.append("-")
        
        latex_content.append(" & ".join(row) + " \\\\")
    
    latex_content.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Benchmark Results Comparison}",
        "\\end{table}",
        "\\end{document}"
    ])
    
    with open(output_path, 'w') as f:
        f.write("\n".join(latex_content))