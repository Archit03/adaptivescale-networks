"""
Main evaluator for AdaptiveScale Networks.

This module provides the core evaluation functionality for ASN models,
including comprehensive evaluation across multiple tasks and metrics.
"""

import logging
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

from .metrics import MetricsCalculator, MetricResult
from ..data.datasets import create_dataset
from ..data.loaders import create_meta_learning_loader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    
    # Basic settings
    batch_size: int = 8
    max_samples: Optional[int] = None
    num_workers: int = 0
    device: str = "auto"
    
    # Metrics configuration
    task_types: List[str] = field(default_factory=lambda: ['qa', 'math', 'reasoning', 'code'])
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'exact_match'])
    
    # Output settings
    save_predictions: bool = True
    save_detailed_results: bool = True
    output_dir: Path = field(default_factory=lambda: Path("evaluation_results"))
    
    # Statistical analysis
    statistical_tests: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Generation settings
    max_new_tokens: int = 64
    temperature: float = 0.7
    do_sample: bool = False
    top_p: float = 0.9
    
    # Few-shot settings
    few_shot_k: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    few_shot_episodes: int = 100
    
    # Cross-validation
    use_cross_validation: bool = False
    cv_folds: int = 5
    
    # Timeout settings
    evaluation_timeout: float = 3600.0  # 1 hour
    per_sample_timeout: float = 30.0    # 30 seconds per sample


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    task_type: str
    dataset_name: str
    metrics: Dict[str, MetricResult]
    predictions: List[str] = field(default_factory=list)
    ground_truths: List[Union[str, List[str]]] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    num_samples: int = 0
    evaluation_time: float = 0.0
    model_info: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            'task_type': self.task_type,
            'dataset_name': self.dataset_name,
            'metrics': {k: {'value': v.value, 'details': v.details} for k, v in self.metrics.items()},
            'predictions': self.predictions,
            'ground_truths': self.ground_truths,
            'num_samples': self.num_samples,
            'evaluation_time': self.evaluation_time,
            'model_info': self.model_info,
            'config': self.config
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save results to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved evaluation results to {filepath}")


class ModelEvaluator:
    """Base model evaluator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Create output directory
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized ModelEvaluator with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        return device
    
    def generate_predictions(self, model, dataloader: DataLoader, 
                           tokenizer=None) -> Tuple[List[str], List[Union[str, List[str]]], List[Dict]]:
        """
        Generate predictions from model.
        
        Args:
            model: Model to evaluate
            dataloader: DataLoader for evaluation data
            tokenizer: Tokenizer for decoding
            
        Returns:
            Tuple of (predictions, ground_truths, examples)
        """
        model.eval()
        predictions = []
        ground_truths = []
        examples = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if time.time() - start_time > self.config.evaluation_timeout:
                    logger.warning("Evaluation timeout reached")
                    break
                
                if self.config.max_samples and len(predictions) >= self.config.max_samples:
                    break
                
                try:
                    # Move batch to device
                    if isinstance(batch, dict):
                        batch_predictions = self._generate_batch_predictions(
                            model, batch, tokenizer
                        )
                        batch_ground_truths = self._extract_ground_truths(batch)
                        batch_examples = self._extract_examples(batch)
                    else:
                        # Handle list of examples
                        batch_predictions = []
                        batch_ground_truths = []
                        batch_examples = batch
                        
                        for example in batch:
                            pred = self._generate_single_prediction(model, example, tokenizer)
                            truth = self._extract_single_ground_truth(example)
                            
                            batch_predictions.append(pred)
                            batch_ground_truths.append(truth)
                    
                    predictions.extend(batch_predictions)
                    ground_truths.extend(batch_ground_truths)
                    examples.extend(batch_examples)
                    
                    if (batch_idx + 1) % 10 == 0:
                        logger.info(f"Processed {batch_idx + 1} batches, {len(predictions)} samples")
                
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    continue
        
        logger.info(f"Generated {len(predictions)} predictions in {time.time() - start_time:.2f}s")
        
        return predictions, ground_truths, examples
    
    def _generate_batch_predictions(self, model, batch: Dict[str, Any], 
                                  tokenizer) -> List[str]:
        """Generate predictions for a batch."""
        # Move tensors to device
        input_ids = batch.get('input_ids', batch.get('question_input_ids'))
        attention_mask = batch.get('attention_mask', batch.get('question_attention_mask'))
        
        if input_ids is None:
            # Handle cases where input format is different
            return [""] * len(batch.get('example_id', []))
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                pad_token_id=tokenizer.eos_token_id if tokenizer else None
            )
        
        # Decode predictions
        if tokenizer:
            # Remove input tokens from generated output
            generated_tokens = outputs[:, input_ids.shape[1]:]
            predictions = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        else:
            # Fallback to string conversion
            predictions = [str(output) for output in outputs]
        
        return [pred.strip() for pred in predictions]
    
    def _generate_single_prediction(self, model, example: Dict[str, Any], 
                                   tokenizer) -> str:
        """Generate prediction for a single example."""
        # This is a simplified implementation
        # In practice, you'd need to handle the specific input format
        
        if hasattr(model, 'generate'):
            # Handle generative models
            if 'formatted_input' in example:
                input_text = example['formatted_input']
            elif 'question' in example:
                input_text = example['question']
            else:
                input_text = str(example)
            
            if tokenizer:
                inputs = tokenizer(input_text, return_tensors='pt').to(self.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=self.config.do_sample
                    )
                
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                return prediction.strip()
        
        # Fallback
        return ""
    
    def _extract_ground_truths(self, batch: Dict[str, Any]) -> List[Union[str, List[str]]]:
        """Extract ground truth answers from batch."""
        if 'answer_text' in batch:
            return batch['answer_text']
        elif 'answers' in batch:
            return batch['answers']
        elif 'target_text' in batch:
            return batch['target_text']
        else:
            # Try to find answer-like fields
            for key in batch.keys():
                if 'answer' in key.lower() or 'target' in key.lower():
                    return batch[key]
            
            return [""] * len(next(iter(batch.values())))
    
    def _extract_single_ground_truth(self, example: Dict[str, Any]) -> Union[str, List[str]]:
        """Extract ground truth from single example."""
        if 'answer' in example:
            answer = example['answer']
            if isinstance(answer, dict) and 'text' in answer:
                return answer['text']
            return str(answer)
        elif 'answers' in example:
            return example['answers']
        else:
            return ""
    
    def _extract_examples(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract example metadata from batch."""
        batch_size = len(next(iter(batch.values())))
        examples = []
        
        for i in range(batch_size):
            example = {}
            for key, values in batch.items():
                if isinstance(values, (list, tuple)) and len(values) > i:
                    example[key] = values[i]
                elif not isinstance(values, (list, tuple)):
                    example[key] = values
            examples.append(example)
        
        return examples


class ASNEvaluator(ModelEvaluator):
    """Main evaluator for AdaptiveScale Networks."""
    
    def __init__(self, config: EvaluationConfig = None):
        if config is None:
            config = EvaluationConfig()
        
        super().__init__(config)
        
        # Initialize metrics calculators for different task types
        self.metrics_calculators = {
            task_type: MetricsCalculator(task_type) 
            for task_type in config.task_types
        }
        
        logger.info("Initialized ASNEvaluator")
    
    def evaluate(self, model, dataset, task_type: str, 
                tokenizer=None, dataset_name: str = None) -> EvaluationResults:
        """
        Evaluate model on a specific dataset.
        
        Args:
            model: Model to evaluate
            dataset: Dataset or DataLoader for evaluation
            task_type: Type of task
            tokenizer: Tokenizer for the model
            dataset_name: Name of the dataset
            
        Returns:
            EvaluationResults
        """
        logger.info(f"Starting evaluation on {task_type} task")
        
        start_time = time.time()
        
        # Setup data loader
        if not isinstance(dataset, DataLoader):
            from torch.utils.data import DataLoader
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
        else:
            dataloader = dataset
        
        # Generate predictions
        predictions, ground_truths, examples = self.generate_predictions(
            model, dataloader, tokenizer
        )
        
        if not predictions:
            logger.warning("No predictions generated")
            return EvaluationResults(
                task_type=task_type,
                dataset_name=dataset_name or "unknown",
                metrics={},
                num_samples=0,
                evaluation_time=time.time() - start_time
            )
        
        # Compute metrics
        metrics_calculator = self.metrics_calculators.get(task_type, MetricsCalculator(task_type))
        
        # Prepare additional arguments for metrics computation
        metrics_kwargs = {}
        if task_type == 'reasoning' and examples:
            # Extract choices for reasoning tasks
            choices_list = []
            for example in examples:
                if 'choices_text' in example:
                    # Parse choices from text format
                    choices = example['choices_text'].split('\n')
                    choices = [choice.split(') ', 1)[1] if ') ' in choice else choice 
                              for choice in choices if choice.strip()]
                    choices_list.append(choices)
                elif 'choices' in example:
                    choices_list.append(example['choices'])
                else:
                    choices_list.append([])
            
            if choices_list:
                metrics_kwargs['choices_list'] = choices_list
        
        elif task_type == 'code' and examples:
            # Extract test cases for code tasks
            test_cases_list = []
            for example in examples:
                test_cases = example.get('test_cases', [])
                test_cases_list.append(test_cases)
            
            if test_cases_list:
                metrics_kwargs['test_cases_list'] = test_cases_list
        
        # Compute metrics
        metrics = metrics_calculator.compute_all_metrics(
            predictions, ground_truths, **metrics_kwargs
        )
        
        # Add confidence intervals if statistical tests are enabled
        if self.config.statistical_tests:
            for metric_name, metric_result in metrics.items():
                if hasattr(metric_result, 'details') and 'individual_scores' in metric_result.details:
                    scores = metric_result.details['individual_scores']
                    ci = self._compute_confidence_interval(scores)
                    metric_result.confidence_interval = ci
        
        # Create results
        results = EvaluationResults(
            task_type=task_type,
            dataset_name=dataset_name or "unknown",
            metrics=metrics,
            predictions=predictions if self.config.save_predictions else [],
            ground_truths=ground_truths if self.config.save_predictions else [],
            examples=examples if self.config.save_detailed_results else [],
            num_samples=len(predictions),
            evaluation_time=time.time() - start_time,
            model_info=self._get_model_info(model),
            config=self.config.__dict__
        )
        
        # Save results if requested
        if self.config.save_detailed_results:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{task_type}_{dataset_name or 'unknown'}_{timestamp}.json"
            results.save(self.config.output_dir / filename)
        
        # Log summary
        self._log_evaluation_summary(results)
        
        return results
    
    def evaluate_multiple_tasks(self, model, datasets: Dict[str, Any], 
                               tokenizer=None) -> Dict[str, EvaluationResults]:
        """
        Evaluate model on multiple tasks.
        
        Args:
            model: Model to evaluate
            datasets: Dictionary mapping task names to datasets
            tokenizer: Tokenizer for the model
            
        Returns:
            Dictionary of evaluation results
        """
        results = {}
        
        for task_name, task_data in datasets.items():
            if isinstance(task_data, dict):
                task_type = task_data.get('task_type', task_name)
                dataset = task_data.get('dataset')
                dataset_name = task_data.get('name', task_name)
            else:
                task_type = task_name
                dataset = task_data
                dataset_name = task_name
            
            logger.info(f"Evaluating on {task_name}")
            
            try:
                result = self.evaluate(
                    model, dataset, task_type, tokenizer, dataset_name
                )
                results[task_name] = result
            except Exception as e:
                logger.error(f"Failed to evaluate {task_name}: {e}")
                continue
        
        # Compute overall statistics
        overall_results = self._compute_overall_statistics(results)
        results['overall'] = overall_results
        
        return results
    
    def evaluate_few_shot(self, model, dataset, task_type: str, 
                         tokenizer=None, k: int = 5) -> Dict[str, EvaluationResults]:
        """
        Evaluate model in few-shot setting.
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            task_type: Type of task
            tokenizer: Tokenizer
            k: Number of shots
            
        Returns:
            Dictionary of few-shot results
        """
        from .few_shot import FewShotEvaluator, FewShotConfig
        
        few_shot_config = FewShotConfig(
            k_values=[k],
            num_episodes=self.config.few_shot_episodes
        )
        
        few_shot_evaluator = FewShotEvaluator(few_shot_config)
        
        return few_shot_evaluator.evaluate(model, dataset, task_type, tokenizer)
    
    def cross_validate(self, model, dataset, task_type: str, 
                      tokenizer=None, n_folds: int = None) -> Dict[str, Any]:
        """
        Perform cross-validation evaluation.
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            task_type: Type of task
            tokenizer: Tokenizer
            n_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        if n_folds is None:
            n_folds = self.config.cv_folds
        
        from sklearn.model_selection import KFold
        
        # Convert dataset to list if needed
        if hasattr(dataset, 'examples'):
            examples = dataset.examples
        else:
            examples = list(dataset)
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(examples)):
            logger.info(f"Evaluating fold {fold_idx + 1}/{n_folds}")
            
            # Create test dataset for this fold
            test_examples = [examples[i] for i in test_idx]
            
            # Create a simple dataset wrapper
            class SimpleDataset:
                def __init__(self, examples):
                    self.examples = examples
                
                def __len__(self):
                    return len(self.examples)
                
                def __getitem__(self, idx):
                    return self.examples[idx]
            
            fold_dataset = SimpleDataset(test_examples)
            
            # Evaluate on this fold
            fold_result = self.evaluate(
                model, fold_dataset, task_type, tokenizer, f"fold_{fold_idx}"
            )
            
            fold_results.append(fold_result)
        
        # Aggregate results across folds
        aggregated_results = self._aggregate_cv_results(fold_results)
        
        return {
            'fold_results': fold_results,
            'aggregated_results': aggregated_results,
            'n_folds': n_folds
        }
    
    def _compute_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Compute confidence interval for scores."""
        from .metrics import bootstrap_confidence_interval
        
        return bootstrap_confidence_interval(
            scores, 
            confidence=self.config.confidence_level,
            num_samples=self.config.bootstrap_samples
        )
    
    def _get_model_info(self, model) -> Dict[str, Any]:
        """Extract model information."""
        info = {
            'model_class': type(model).__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Add model-specific info if available
        if hasattr(model, 'config'):
            info['model_config'] = str(model.config)
        
        return info
    
    def _log_evaluation_summary(self, results: EvaluationResults):
        """Log evaluation summary."""
        logger.info(f"Evaluation Summary for {results.task_type}:")
        logger.info(f"  Dataset: {results.dataset_name}")
        logger.info(f"  Samples: {results.num_samples}")
        logger.info(f"  Time: {results.evaluation_time:.2f}s")
        
        for metric_name, metric_result in results.metrics.items():
            logger.info(f"  {metric_name}: {metric_result.value:.4f}")
    
    def _compute_overall_statistics(self, results: Dict[str, EvaluationResults]) -> Dict[str, Any]:
        """Compute overall statistics across multiple tasks."""
        overall_stats = {
            'total_samples': sum(r.num_samples for r in results.values()),
            'total_time': sum(r.evaluation_time for r in results.values()),
            'task_results': {}
        }
        
        # Aggregate metrics across tasks
        all_metrics = defaultdict(list)
        
        for task_name, result in results.items():
            task_metrics = {}
            for metric_name, metric_result in result.metrics.items():
                task_metrics[metric_name] = metric_result.value
                all_metrics[metric_name].append(metric_result.value)
            
            overall_stats['task_results'][task_name] = {
                'metrics': task_metrics,
                'num_samples': result.num_samples,
                'evaluation_time': result.evaluation_time
            }
        
        # Compute average metrics
        overall_stats['average_metrics'] = {
            metric_name: np.mean(values) 
            for metric_name, values in all_metrics.items()
        }
        
        # Compute weighted average (by number of samples)
        weighted_metrics = {}
        total_samples = overall_stats['total_samples']
        
        for metric_name in all_metrics.keys():
            weighted_sum = sum(
                result.metrics[metric_name].value * result.num_samples
                for result in results.values()
                if metric_name in result.metrics
            )
            weighted_metrics[metric_name] = weighted_sum / total_samples if total_samples > 0 else 0
        
        overall_stats['weighted_metrics'] = weighted_metrics
        
        return overall_stats
    
    def _aggregate_cv_results(self, fold_results: List[EvaluationResults]) -> Dict[str, Any]:
        """Aggregate cross-validation results."""
        aggregated = {
            'mean_metrics': {},
            'std_metrics': {},
            'min_metrics': {},
            'max_metrics': {},
            'total_samples': sum(r.num_samples for r in fold_results),
            'average_time': np.mean([r.evaluation_time for r in fold_results])
        }
        
        # Get all metric names
        all_metric_names = set()
        for result in fold_results:
            all_metric_names.update(result.metrics.keys())
        
        # Aggregate each metric
        for metric_name in all_metric_names:
            values = [
                result.metrics[metric_name].value 
                for result in fold_results 
                if metric_name in result.metrics
            ]
            
            if values:
                aggregated['mean_metrics'][metric_name] = np.mean(values)
                aggregated['std_metrics'][metric_name] = np.std(values)
                aggregated['min_metrics'][metric_name] = np.min(values)
                aggregated['max_metrics'][metric_name] = np.max(values)
        
        return aggregated


def main():
    """Main function for command-line evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ASN models")
    parser.add_argument("--model-path", required=True, help="Path to model")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--task-type", required=True, choices=['qa', 'math', 'reasoning', 'code'])
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory")
    parser.add_argument("--max-samples", type=int, help="Maximum samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    
    args = parser.parse_args()
    
    # Setup configuration
    config = EvaluationConfig(
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        output_dir=Path(args.output_dir),
        save_predictions=True,
        save_detailed_results=True
    )
    
    # Initialize evaluator
    evaluator = ASNEvaluator(config)
    
    # Load model (placeholder - would need actual loading logic)
    logger.info(f"Loading model from {args.model_path}")
    # model = load_model(args.model_path)
    
    # Load dataset
    logger.info(f"Loading dataset {args.dataset}")
    dataset = create_dataset(args.task_type, args.dataset)
    
    # Run evaluation
    # results = evaluator.evaluate(model, dataset, args.task_type)
    
    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()