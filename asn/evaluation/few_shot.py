"""
Few-shot evaluation module for AdaptiveScale Networks.

This module provides specialized evaluation capabilities for few-shot learning
scenarios, including episode generation and meta-learning evaluation.
"""

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import numpy as np
from collections import defaultdict

from .metrics import MetricsCalculator, MetricResult
from .evaluator import EvaluationResults

logger = logging.getLogger(__name__)


@dataclass
class FewShotConfig:
    """Configuration for few-shot evaluation."""
    
    # Few-shot settings
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10])
    num_episodes: int = 100
    support_query_split: float = 0.5
    
    # Episode generation
    random_seed: int = 42
    stratified_sampling: bool = True
    balance_classes: bool = True
    
    # Evaluation settings
    max_context_length: int = 2048
    prompt_template: str = "Examples:\n{examples}\n\nQuestion: {question}\nAnswer:"
    example_separator: str = "\n\n"
    
    # Statistical analysis
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    
    # Output settings
    save_episodes: bool = False
    save_detailed_results: bool = True


@dataclass
class FewShotEpisode:
    """Container for a few-shot episode."""
    
    support_set: List[Dict[str, Any]]
    query_set: List[Dict[str, Any]]
    task_info: Dict[str, Any] = field(default_factory=dict)
    episode_id: str = ""
    
    def __len__(self) -> int:
        return len(self.support_set) + len(self.query_set)
    
    def get_support_size(self) -> int:
        return len(self.support_set)
    
    def get_query_size(self) -> int:
        return len(self.query_set)


@dataclass 
class FewShotResults:
    """Container for few-shot evaluation results."""
    
    task_type: str
    k_value: int
    episodes: List[FewShotEpisode] = field(default_factory=list)
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    
    # Episode statistics
    num_episodes: int = 0
    avg_support_size: float = 0.0
    avg_query_size: float = 0.0
    
    # Performance statistics
    episode_performances: List[Dict[str, float]] = field(default_factory=list)
    
    # Metadata
    evaluation_time: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'task_type': self.task_type,
            'k_value': self.k_value,
            'num_episodes': self.num_episodes,
            'avg_support_size': self.avg_support_size,
            'avg_query_size': self.avg_query_size,
            'metrics': {k: v.value for k, v in self.metrics.items()},
            'evaluation_time': self.evaluation_time
        }


class EpisodeGenerator:
    """Generator for few-shot episodes."""
    
    def __init__(self, config: FewShotConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def generate_episodes(self, dataset, task_type: str, k: int, 
                         num_episodes: int = None) -> List[FewShotEpisode]:
        """
        Generate few-shot episodes from dataset.
        
        Args:
            dataset: Dataset to sample from
            task_type: Type of task
            k: Number of support examples per episode
            num_episodes: Number of episodes to generate
            
        Returns:
            List of few-shot episodes
        """
        if num_episodes is None:
            num_episodes = self.config.num_episodes
        
        # Convert dataset to list if needed
        if hasattr(dataset, 'examples'):
            examples = dataset.examples
        elif hasattr(dataset, '__iter__'):
            examples = list(dataset)
        else:
            raise ValueError("Dataset must be iterable or have 'examples' attribute")
        
        logger.info(f"Generating {num_episodes} episodes with k={k} from {len(examples)} examples")
        
        # Group examples by class/type if stratified sampling is enabled
        if self.config.stratified_sampling:
            grouped_examples = self._group_examples_by_class(examples, task_type)
        else:
            grouped_examples = {'all': examples}
        
        episodes = []
        
        for episode_idx in range(num_episodes):
            try:
                episode = self._generate_single_episode(
                    grouped_examples, task_type, k, episode_idx
                )
                if episode:
                    episodes.append(episode)
            except Exception as e:
                logger.warning(f"Failed to generate episode {episode_idx}: {e}")
                continue
        
        logger.info(f"Generated {len(episodes)} valid episodes")
        return episodes
    
    def _group_examples_by_class(self, examples: List[Dict[str, Any]], 
                                task_type: str) -> Dict[str, List[Dict[str, Any]]]:
        """Group examples by class for stratified sampling."""
        groups = defaultdict(list)
        
        for example in examples:
            # Determine class based on task type
            if task_type == 'reasoning':
                # Group by reasoning type or difficulty
                class_key = example.get('reasoning_type', example.get('difficulty', 'default'))
            elif task_type == 'math':
                # Group by problem type or difficulty
                class_key = example.get('problem_type', example.get('difficulty', 'default'))
            elif task_type == 'qa':
                # Group by topic or length
                class_key = example.get('topic', 'default')
            elif task_type == 'code':
                # Group by difficulty or language
                class_key = example.get('difficulty', example.get('language', 'default'))
            else:
                class_key = 'default'
            
            groups[str(class_key)].append(example)
        
        # Filter out groups with insufficient examples
        min_examples = max(10, 2 * self.config.k_values[-1])  # Need enough for support and query
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_examples}
        
        if not filtered_groups:
            logger.warning("No groups have sufficient examples, using all examples")
            return {'all': examples}
        
        logger.debug(f"Grouped examples into {len(filtered_groups)} classes: "
                    f"{[(k, len(v)) for k, v in filtered_groups.items()]}")
        
        return filtered_groups
    
    def _generate_single_episode(self, grouped_examples: Dict[str, List[Dict[str, Any]]], 
                                task_type: str, k: int, episode_idx: int) -> Optional[FewShotEpisode]:
        """Generate a single few-shot episode."""
        # Choose a class to sample from
        if len(grouped_examples) == 1:
            class_key = list(grouped_examples.keys())[0]
        else:
            class_key = random.choice(list(grouped_examples.keys()))
        
        available_examples = grouped_examples[class_key].copy()
        
        # Calculate support and query sizes
        support_size = k
        query_size = max(1, int(k / self.config.support_query_split) - k)
        total_needed = support_size + query_size
        
        if len(available_examples) < total_needed:
            # Fall back to all available examples from any class
            all_examples = []
            for examples in grouped_examples.values():
                all_examples.extend(examples)
            
            if len(all_examples) < total_needed:
                logger.warning(f"Not enough examples for episode {episode_idx}")
                return None
            
            available_examples = all_examples
        
        # Sample examples
        random.shuffle(available_examples)
        support_examples = available_examples[:support_size]
        query_examples = available_examples[support_size:support_size + query_size]
        
        # Create episode
        episode = FewShotEpisode(
            support_set=support_examples,
            query_set=query_examples,
            task_info={
                'class': class_key,
                'task_type': task_type,
                'k': k
            },
            episode_id=f"episode_{episode_idx}"
        )
        
        return episode


class FewShotEvaluator:
    """Evaluator for few-shot learning scenarios."""
    
    def __init__(self, config: FewShotConfig = None):
        if config is None:
            config = FewShotConfig()
        
        self.config = config
        self.episode_generator = EpisodeGenerator(config)
        
        logger.info("Initialized FewShotEvaluator")
    
    def evaluate(self, model, dataset, task_type: str, 
                tokenizer=None) -> Dict[str, FewShotResults]:
        """
        Evaluate model in few-shot setting across different k values.
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            task_type: Type of task
            tokenizer: Tokenizer for the model
            
        Returns:
            Dictionary mapping k values to results
        """
        results = {}
        
        for k in self.config.k_values:
            logger.info(f"Evaluating {k}-shot performance")
            
            try:
                k_results = self.evaluate_k_shot(
                    model, dataset, task_type, k, tokenizer
                )
                results[f"{k}_shot"] = k_results
            except Exception as e:
                logger.error(f"Failed to evaluate {k}-shot: {e}")
                continue
        
        # Compute summary across k values
        results['summary'] = self._compute_k_shot_summary(results)
        
        return results
    
    def evaluate_k_shot(self, model, dataset, task_type: str, k: int, 
                       tokenizer=None) -> FewShotResults:
        """
        Evaluate model for specific k-shot setting.
        
        Args:
            model: Model to evaluate
            dataset: Dataset for evaluation
            task_type: Type of task
            k: Number of shots
            tokenizer: Tokenizer for the model
            
        Returns:
            FewShotResults for this k value
        """
        start_time = time.time()
        
        # Generate episodes
        episodes = self.episode_generator.generate_episodes(
            dataset, task_type, k, self.config.num_episodes
        )
        
        if not episodes:
            logger.warning(f"No episodes generated for {k}-shot evaluation")
            return FewShotResults(task_type=task_type, k_value=k)
        
        # Evaluate each episode
        all_predictions = []
        all_ground_truths = []
        episode_performances = []
        
        model.eval()
        
        for episode_idx, episode in enumerate(episodes):
            try:
                # Generate predictions for this episode
                episode_predictions, episode_ground_truths = self._evaluate_episode(
                    model, episode, task_type, tokenizer
                )
                
                if episode_predictions:
                    all_predictions.extend(episode_predictions)
                    all_ground_truths.extend(episode_ground_truths)
                    
                    # Compute episode-level performance
                    episode_metrics = self._compute_episode_metrics(
                        episode_predictions, episode_ground_truths, task_type
                    )
                    episode_performances.append(episode_metrics)
                
                if (episode_idx + 1) % 20 == 0:
                    logger.info(f"Evaluated {episode_idx + 1}/{len(episodes)} episodes")
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate episode {episode_idx}: {e}")
                continue
        
        # Compute overall metrics
        if all_predictions:
            metrics_calculator = MetricsCalculator(task_type)
            overall_metrics = metrics_calculator.compute_all_metrics(
                all_predictions, all_ground_truths
            )
        else:
            overall_metrics = {}
        
        # Create results
        results = FewShotResults(
            task_type=task_type,
            k_value=k,
            episodes=episodes if self.config.save_episodes else [],
            metrics=overall_metrics,
            num_episodes=len(episodes),
            avg_support_size=np.mean([ep.get_support_size() for ep in episodes]),
            avg_query_size=np.mean([ep.get_query_size() for ep in episodes]),
            episode_performances=episode_performances,
            evaluation_time=time.time() - start_time,
            config=self.config.__dict__
        )
        
        logger.info(f"{k}-shot evaluation completed: {len(all_predictions)} predictions generated")
        
        return results
    
    def _evaluate_episode(self, model, episode: FewShotEpisode, task_type: str, 
                         tokenizer) -> Tuple[List[str], List[str]]:
        """Evaluate a single episode."""
        # Create few-shot prompt from support set
        few_shot_prompt = self._create_few_shot_prompt(episode.support_set, task_type)
        
        predictions = []
        ground_truths = []
        
        # Generate predictions for query set
        for query_example in episode.query_set:
            try:
                # Create full prompt
                query_text = self._extract_query_text(query_example, task_type)
                full_prompt = few_shot_prompt + f"\n\nQuestion: {query_text}\nAnswer:"
                
                # Generate prediction
                if tokenizer:
                    inputs = tokenizer(
                        full_prompt,
                        return_tensors='pt',
                        max_length=self.config.max_context_length,
                        truncation=True
                    )
                    
                    # Move to device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=64,
                            temperature=0.7,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    # Decode prediction
                    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    prediction = prediction.strip()
                else:
                    # Fallback for models without tokenizer
                    prediction = ""
                
                # Extract ground truth
                ground_truth = self._extract_ground_truth(query_example, task_type)
                
                predictions.append(prediction)
                ground_truths.append(ground_truth)
                
            except Exception as e:
                logger.warning(f"Failed to process query example: {e}")
                continue
        
        return predictions, ground_truths
    
    def _create_few_shot_prompt(self, support_set: List[Dict[str, Any]], 
                               task_type: str) -> str:
        """Create few-shot prompt from support examples."""
        examples = []
        
        for example in support_set:
            # Format example based on task type
            if task_type == 'qa':
                question = example.get('question', '')
                answer = example.get('answer', '')
                if isinstance(answer, dict):
                    answer = answer.get('text', str(answer))
                formatted_example = f"Question: {question}\nAnswer: {answer}"
                
            elif task_type == 'math':
                problem = example.get('question', example.get('problem', ''))
                answer = example.get('answer', '')
                formatted_example = f"Problem: {problem}\nAnswer: {answer}"
                
            elif task_type == 'reasoning':
                question = example.get('question', '')
                choices = example.get('choices', [])
                answer = example.get('answer', '')
                
                if choices:
                    choices_text = '\n'.join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                    formatted_example = f"Question: {question}\nChoices:\n{choices_text}\nAnswer: {answer}"
                else:
                    formatted_example = f"Question: {question}\nAnswer: {answer}"
                
            elif task_type == 'code':
                problem = example.get('problem', example.get('prompt', ''))
                code = example.get('code', example.get('canonical_solution', ''))
                formatted_example = f"Problem: {problem}\nCode: {code}"
                
            else:
                # Generic format
                formatted_example = str(example)
            
            examples.append(formatted_example)
        
        # Join examples
        prompt = self.config.example_separator.join(examples)
        
        return f"Examples:\n{prompt}"
    
    def _extract_query_text(self, example: Dict[str, Any], task_type: str) -> str:
        """Extract query text from example."""
        if task_type == 'qa':
            return example.get('question', '')
        elif task_type == 'math':
            return example.get('question', example.get('problem', ''))
        elif task_type == 'reasoning':
            question = example.get('question', '')
            choices = example.get('choices', [])
            
            if choices:
                choices_text = '\n'.join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
                return f"{question}\nChoices:\n{choices_text}"
            else:
                return question
        elif task_type == 'code':
            return example.get('problem', example.get('prompt', ''))
        else:
            return str(example)
    
    def _extract_ground_truth(self, example: Dict[str, Any], task_type: str) -> str:
        """Extract ground truth answer from example."""
        if 'answer' in example:
            answer = example['answer']
            if isinstance(answer, dict):
                return answer.get('text', str(answer))
            return str(answer)
        elif 'answers' in example:
            answers = example['answers']
            if isinstance(answers, list) and answers:
                return str(answers[0])
            return str(answers)
        else:
            return ""
    
    def _compute_episode_metrics(self, predictions: List[str], ground_truths: List[str], 
                                task_type: str) -> Dict[str, float]:
        """Compute metrics for a single episode."""
        if not predictions:
            return {}
        
        metrics_calculator = MetricsCalculator(task_type)
        metrics = metrics_calculator.compute_all_metrics(predictions, ground_truths)
        
        # Convert to simple dict with float values
        episode_metrics = {}
        for metric_name, metric_result in metrics.items():
            episode_metrics[metric_name] = metric_result.value
        
        return episode_metrics
    
    def _compute_k_shot_summary(self, results: Dict[str, FewShotResults]) -> Dict[str, Any]:
        """Compute summary statistics across different k values."""
        summary = {
            'k_values': [],
            'performance_trends': {},
            'best_k': {},
            'worst_k': {}
        }
        
        # Extract k values (excluding 'summary')
        k_shot_results = {k: v for k, v in results.items() if k != 'summary'}
        
        if not k_shot_results:
            return summary
        
        # Extract k values
        k_values = []
        for k_str in k_shot_results.keys():
            try:
                k = int(k_str.split('_')[0])
                k_values.append(k)
            except:
                continue
        
        k_values.sort()
        summary['k_values'] = k_values
        
        # Analyze performance trends
        metrics_names = set()
        for result in k_shot_results.values():
            metrics_names.update(result.metrics.keys())
        
        for metric_name in metrics_names:
            metric_values = []
            valid_k_values = []
            
            for k in k_values:
                k_str = f"{k}_shot"
                if k_str in k_shot_results and metric_name in k_shot_results[k_str].metrics:
                    metric_values.append(k_shot_results[k_str].metrics[metric_name].value)
                    valid_k_values.append(k)
            
            if metric_values:
                summary['performance_trends'][metric_name] = {
                    'k_values': valid_k_values,
                    'values': metric_values,
                    'trend': self._analyze_trend(metric_values)
                }
                
                # Find best and worst k
                best_idx = np.argmax(metric_values)
                worst_idx = np.argmin(metric_values)
                
                summary['best_k'][metric_name] = {
                    'k': valid_k_values[best_idx],
                    'value': metric_values[best_idx]
                }
                
                summary['worst_k'][metric_name] = {
                    'k': valid_k_values[worst_idx],
                    'value': metric_values[worst_idx]
                }
        
        return summary
    
    def _analyze_trend(self, values: List[float]) -> str:
        """Analyze trend in performance values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple trend analysis
        differences = [values[i+1] - values[i] for i in range(len(values)-1)]
        
        if all(d >= 0 for d in differences):
            return "increasing"
        elif all(d <= 0 for d in differences):
            return "decreasing"
        elif sum(differences) > 0:
            return "mostly_increasing"
        elif sum(differences) < 0:
            return "mostly_decreasing"
        else:
            return "mixed"


# Utility functions
def create_few_shot_episodes(dataset, task_type: str, k: int, num_episodes: int = 100, 
                            config: FewShotConfig = None) -> List[FewShotEpisode]:
    """
    Create few-shot episodes from a dataset.
    
    Args:
        dataset: Dataset to sample from
        task_type: Type of task
        k: Number of support examples
        num_episodes: Number of episodes to generate
        config: Configuration for episode generation
        
    Returns:
        List of few-shot episodes
    """
    if config is None:
        config = FewShotConfig()
    
    generator = EpisodeGenerator(config)
    return generator.generate_episodes(dataset, task_type, k, num_episodes)


def evaluate_few_shot_performance(model, episodes: List[FewShotEpisode], 
                                 task_type: str, tokenizer=None,
                                 config: FewShotConfig = None) -> FewShotResults:
    """
    Evaluate model performance on few-shot episodes.
    
    Args:
        model: Model to evaluate
        episodes: List of few-shot episodes
        task_type: Type of task
        tokenizer: Tokenizer for the model
        config: Configuration for evaluation
        
    Returns:
        Few-shot evaluation results
    """
    if config is None:
        config = FewShotConfig()
    
    evaluator = FewShotEvaluator(config)
    
    # Create a temporary dataset from episodes
    class EpisodeDataset:
        def __init__(self, episodes):
            self.episodes = episodes
            self.examples = []
            for episode in episodes:
                self.examples.extend(episode.support_set + episode.query_set)
        
        def __len__(self):
            return len(self.examples)
        
        def __iter__(self):
            return iter(self.examples)
    
    dataset = EpisodeDataset(episodes)
    
    # Determine k value from episodes
    k = episodes[0].get_support_size() if episodes else 1
    
    return evaluator.evaluate_k_shot(model, dataset, task_type, k, tokenizer)


def compare_few_shot_performance(results1: FewShotResults, results2: FewShotResults,
                               metric: str = 'accuracy') -> Dict[str, Any]:
    """
    Compare few-shot performance between two models.
    
    Args:
        results1: Results from first model
        results2: Results from second model
        metric: Metric to compare
        
    Returns:
        Comparison results
    """
    comparison = {
        'model1_performance': 0.0,
        'model2_performance': 0.0,
        'difference': 0.0,
        'relative_improvement': 0.0,
        'better_model': None
    }
    
    if metric in results1.metrics and metric in results2.metrics:
        perf1 = results1.metrics[metric].value
        perf2 = results2.metrics[metric].value
        
        comparison['model1_performance'] = perf1
        comparison['model2_performance'] = perf2
        comparison['difference'] = perf2 - perf1
        comparison['relative_improvement'] = (perf2 - perf1) / perf1 * 100 if perf1 > 0 else 0
        comparison['better_model'] = 'model2' if perf2 > perf1 else 'model1'
        
        # Statistical significance test if episode performances are available
        if (results1.episode_performances and results2.episode_performances and
            len(results1.episode_performances) == len(results2.episode_performances)):
            
            from .metrics import statistical_significance_test
            
            scores1 = [ep.get(metric, 0) for ep in results1.episode_performances]
            scores2 = [ep.get(metric, 0) for ep in results2.episode_performances]
            
            if scores1 and scores2:
                significance_test = statistical_significance_test(scores1, scores2)
                comparison['statistical_test'] = significance_test
    
    return comparison