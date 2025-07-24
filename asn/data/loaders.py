"""
Advanced data loaders for AdaptiveScale Networks.

This module provides specialized data loaders for meta-learning, few-shot learning,
and adaptive training scenarios. It includes support for episode-based sampling,
dynamic batching, and multi-task coordination.
"""

import logging
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Iterator, Callable
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """Configuration for specialized data loaders."""
    
    # Basic settings
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False
    
    # Few-shot settings
    shots_per_task: int = 5
    query_shots_per_task: int = 10
    num_tasks_per_batch: int = 4
    
    # Meta-learning settings
    inner_batch_size: int = 8
    meta_batch_size: int = 4
    support_query_ratio: float = 0.5
    
    # Adaptive settings
    adaptive_batch_size: bool = False
    min_batch_size: int = 1
    max_batch_size: int = 16
    memory_threshold: float = 0.8
    
    # Sampling settings
    sampling_strategy: str = "uniform"  # uniform, balanced, curriculum, adaptive
    task_weighting: Dict[str, float] = field(default_factory=dict)
    difficulty_curriculum: bool = False
    
    # Performance settings
    prefetch_factor: int = 2
    persistent_workers: bool = False
    timeout: float = 0.0


class EpisodeSampler(Sampler):
    """
    Sampler for generating episodes for meta-learning.
    
    Creates episodes by sampling support and query sets from the same task/class.
    """
    
    def __init__(self, dataset, shots_per_task: int = 5, query_shots_per_task: int = 10,
                 num_episodes: int = 100, task_key: str = None):
        """
        Initialize episode sampler.
        
        Args:
            dataset: Dataset to sample from
            shots_per_task: Number of support examples per task
            query_shots_per_task: Number of query examples per task
            num_episodes: Number of episodes to generate
            task_key: Key to identify task/class (auto-detect if None)
        """
        self.dataset = dataset
        self.shots_per_task = shots_per_task
        self.query_shots_per_task = query_shots_per_task
        self.num_episodes = num_episodes
        self.task_key = task_key
        
        # Group examples by task/class
        self.task_groups = self._group_by_task()
        self.task_list = list(self.task_groups.keys())
        
        logger.info(f"EpisodeSampler initialized: {len(self.task_list)} tasks, {num_episodes} episodes")
    
    def _group_by_task(self) -> Dict[str, List[int]]:
        """Group dataset indices by task/class."""
        groups = defaultdict(list)
        
        for idx in range(len(self.dataset)):
            example = self.dataset.examples[idx] if hasattr(self.dataset, 'examples') else self.dataset[idx]
            
            # Determine task identifier
            if self.task_key:
                task_id = getattr(example, self.task_key, 'default')
            else:
                # Auto-detect task identifier
                if hasattr(example, 'reasoning_type'):
                    task_id = example.reasoning_type
                elif hasattr(example, 'problem_type'):
                    task_id = example.problem_type
                elif hasattr(example, 'difficulty'):
                    task_id = example.difficulty
                else:
                    task_id = 'default'
            
            groups[str(task_id)].append(idx)
        
        # Filter out tasks with insufficient examples
        min_examples = self.shots_per_task + self.query_shots_per_task
        filtered_groups = {k: v for k, v in groups.items() if len(v) >= min_examples}
        
        logger.debug(f"Task groups: {[(k, len(v)) for k, v in filtered_groups.items()]}")
        return filtered_groups
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate episodes."""
        for _ in range(self.num_episodes):
            # Sample a task
            if not self.task_list:
                break
                
            task_id = random.choice(self.task_list)
            task_examples = self.task_groups[task_id].copy()
            
            # Ensure we have enough examples
            if len(task_examples) < self.shots_per_task + self.query_shots_per_task:
                continue
            
            # Sample support and query sets
            random.shuffle(task_examples)
            support_indices = task_examples[:self.shots_per_task]
            query_indices = task_examples[self.shots_per_task:self.shots_per_task + self.query_shots_per_task]
            
            # Yield combined episode
            yield support_indices + query_indices
    
    def __len__(self) -> int:
        return self.num_episodes


class AdaptiveBatchSampler(Sampler):
    """
    Adaptive batch sampler that adjusts batch size based on memory usage and example complexity.
    """
    
    def __init__(self, dataset, config: LoaderConfig):
        self.dataset = dataset
        self.config = config
        self.current_batch_size = config.batch_size
        self.memory_usage_history = deque(maxlen=10)
        self.batch_time_history = deque(maxlen=10)
        
        # Complexity scoring
        self.example_complexities = self._compute_example_complexities()
        
    def _compute_example_complexities(self) -> List[float]:
        """Compute complexity scores for all examples."""
        complexities = []
        
        for idx in range(len(self.dataset)):
            try:
                example = self.dataset.examples[idx] if hasattr(self.dataset, 'examples') else self.dataset[idx]
                complexity = self._estimate_complexity(example)
                complexities.append(complexity)
            except:
                complexities.append(1.0)  # Default complexity
        
        return complexities
    
    def _estimate_complexity(self, example) -> float:
        """Estimate complexity of an example."""
        complexity = 1.0
        
        # Text length-based complexity
        if hasattr(example, 'question'):
            complexity += len(example.question) / 1000.0
        if hasattr(example, 'context'):
            complexity += len(example.context) / 2000.0
        if hasattr(example, 'choices'):
            complexity += len(example.choices) * 0.1
        
        # Task-specific complexity
        if hasattr(example, 'difficulty'):
            difficulty_weights = {'easy': 0.8, 'medium': 1.0, 'hard': 1.5}
            complexity *= difficulty_weights.get(example.difficulty, 1.0)
        
        return max(0.1, min(3.0, complexity))  # Clamp between 0.1 and 3.0
    
    def _adjust_batch_size(self) -> int:
        """Adjust batch size based on recent performance."""
        if not self.config.adaptive_batch_size:
            return self.config.batch_size
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.memory_usage_history.append(memory_used)
            
            avg_memory = np.mean(self.memory_usage_history)
            
            if avg_memory > self.config.memory_threshold:
                # Reduce batch size
                self.current_batch_size = max(
                    self.config.min_batch_size,
                    int(self.current_batch_size * 0.8)
                )
            elif avg_memory < self.config.memory_threshold * 0.6:
                # Increase batch size
                self.current_batch_size = min(
                    self.config.max_batch_size,
                    int(self.current_batch_size * 1.2)
                )
        
        return self.current_batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate adaptive batches."""
        indices = list(range(len(self.dataset)))
        
        # Sort by complexity for curriculum learning
        if self.config.difficulty_curriculum:
            indices.sort(key=lambda i: self.example_complexities[i])
        else:
            random.shuffle(indices)
        
        i = 0
        while i < len(indices):
            current_batch_size = self._adjust_batch_size()
            
            # Select batch considering complexity
            batch_indices = []
            batch_complexity = 0.0
            max_complexity = current_batch_size * 1.5  # Complexity budget
            
            while len(batch_indices) < current_batch_size and i < len(indices):
                idx = indices[i]
                example_complexity = self.example_complexities[idx]
                
                if batch_complexity + example_complexity <= max_complexity:
                    batch_indices.append(idx)
                    batch_complexity += example_complexity
                
                i += 1
                
                # If we can't fit more complex examples, fill with simpler ones
                if len(batch_indices) == 0 and i < len(indices):
                    batch_indices.append(indices[i])
                    i += 1
                    break
            
            if batch_indices:
                yield batch_indices
    
    def __len__(self) -> int:
        return (len(self.dataset) + self.current_batch_size - 1) // self.current_batch_size


class FewShotDataLoader:
    """
    Data loader for few-shot learning scenarios.
    
    Generates episodes with support and query sets for meta-learning training.
    """
    
    def __init__(self, dataset, config: LoaderConfig):
        self.dataset = dataset
        self.config = config
        self.episode_sampler = EpisodeSampler(
            dataset,
            shots_per_task=config.shots_per_task,
            query_shots_per_task=config.query_shots_per_task,
            num_episodes=100  # Default number of episodes
        )
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over few-shot episodes."""
        for episode_indices in self.episode_sampler:
            # Split into support and query
            support_size = self.config.shots_per_task
            support_indices = episode_indices[:support_size]
            query_indices = episode_indices[support_size:]
            
            # Get examples
            support_examples = [self.dataset[i] for i in support_indices]
            query_examples = [self.dataset[i] for i in query_indices]
            
            # Create episode dict
            episode = {
                'support': self._collate_examples(support_examples),
                'query': self._collate_examples(query_examples),
                'task_info': {
                    'support_size': len(support_examples),
                    'query_size': len(query_examples),
                    'episode_id': time.time()  # Simple episode ID
                }
            }
            
            yield episode
    
    def _collate_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of examples into a batch."""
        if not examples:
            return {}
        
        # Use dataset's collate function if available
        if hasattr(self.dataset, '_collate_fn'):
            return self.dataset._collate_fn(examples)
        
        # Default collation
        collated = {}
        first_example = examples[0]
        
        for key in first_example.keys():
            if isinstance(first_example[key], torch.Tensor):
                collated[key] = torch.stack([ex[key] for ex in examples])
            elif isinstance(first_example[key], str):
                collated[key] = [ex[key] for ex in examples]
            else:
                collated[key] = [ex[key] for ex in examples]
        
        return collated
    
    def __len__(self) -> int:
        return len(self.episode_sampler)


class MetaLearningDataLoader:
    """
    Data loader optimized for meta-learning algorithms like MAML.
    
    Provides batches of tasks with support and query sets.
    """
    
    def __init__(self, datasets: Dict[str, Dataset], config: LoaderConfig):
        """
        Initialize meta-learning data loader.
        
        Args:
            datasets: Dictionary mapping task types to datasets
            config: Loader configuration
        """
        self.datasets = datasets
        self.config = config
        self.task_types = list(datasets.keys())
        
        # Create few-shot loaders for each task type
        self.few_shot_loaders = {}
        for task_type, dataset in datasets.items():
            self.few_shot_loaders[task_type] = FewShotDataLoader(dataset, config)
        
        logger.info(f"MetaLearningDataLoader initialized with {len(self.task_types)} task types")
    
    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        """Iterate over meta-learning batches."""
        # Create iterators for each task type
        task_iterators = {
            task_type: iter(loader)
            for task_type, loader in self.few_shot_loaders.items()
        }
        
        while task_iterators:
            meta_batch = []
            
            # Sample tasks for this meta-batch
            available_tasks = list(task_iterators.keys())
            if not available_tasks:
                break
            
            # Use task weighting if specified
            if self.config.task_weighting:
                weights = [self.config.task_weighting.get(task, 1.0) for task in available_tasks]
                sampled_tasks = np.random.choice(
                    available_tasks,
                    size=min(self.config.num_tasks_per_batch, len(available_tasks)),
                    replace=False,
                    p=np.array(weights) / np.sum(weights)
                )
            else:
                sampled_tasks = random.sample(
                    available_tasks,
                    min(self.config.num_tasks_per_batch, len(available_tasks))
                )
            
            # Get episodes from sampled tasks
            for task_type in sampled_tasks:
                try:
                    episode = next(task_iterators[task_type])
                    episode['task_type'] = task_type
                    meta_batch.append(episode)
                except StopIteration:
                    # Remove exhausted iterator
                    del task_iterators[task_type]
            
            if meta_batch:
                yield meta_batch
    
    def __len__(self) -> int:
        return min(len(loader) for loader in self.few_shot_loaders.values())


class AdaptiveDataLoader:
    """
    Adaptive data loader that adjusts its behavior based on training dynamics.
    
    Features dynamic batch sizing, curriculum learning, and performance-aware sampling.
    """
    
    def __init__(self, dataset, config: LoaderConfig):
        self.dataset = dataset
        self.config = config
        self.performance_history = deque(maxlen=100)
        self.current_epoch = 0
        
        # Create adaptive sampler
        self.sampler = AdaptiveBatchSampler(dataset, config)
        
        # Performance tracking
        self.batch_times = deque(maxlen=50)
        self.throughput_history = deque(maxlen=20)
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate with adaptive behavior."""
        start_time = time.time()
        total_batches = 0
        
        for batch_indices in self.sampler:
            batch_start = time.time()
            
            # Get batch examples
            batch_examples = [self.dataset[i] for i in batch_indices]
            
            # Collate batch
            batch = self._collate_batch(batch_examples)
            
            # Add adaptive metadata
            batch['_adaptive_info'] = {
                'batch_size': len(batch_indices),
                'epoch': self.current_epoch,
                'complexity_score': np.mean([self.sampler.example_complexities[i] for i in batch_indices]),
                'batch_id': total_batches
            }
            
            # Track timing
            batch_time = time.time() - batch_start
            self.batch_times.append(batch_time)
            
            total_batches += 1
            yield batch
        
        # Update throughput statistics
        total_time = time.time() - start_time
        if total_time > 0:
            throughput = total_batches / total_time
            self.throughput_history.append(throughput)
            
        self.current_epoch += 1
    
    def _collate_batch(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate batch with adaptive considerations."""
        if hasattr(self.dataset, '_collate_fn'):
            return self.dataset._collate_fn(examples)
        
        # Default collation
        collated = {}
        if not examples:
            return collated
            
        first_example = examples[0]
        
        for key in first_example.keys():
            if isinstance(first_example[key], torch.Tensor):
                collated[key] = torch.stack([ex[key] for ex in examples])
            elif isinstance(first_example[key], str):
                collated[key] = [ex[key] for ex in examples]
            else:
                collated[key] = [ex[key] for ex in examples]
        
        return collated
    
    def update_performance(self, loss: float, accuracy: float = None):
        """Update performance history for adaptive behavior."""
        self.performance_history.append({
            'loss': loss,
            'accuracy': accuracy,
            'epoch': self.current_epoch,
            'timestamp': time.time()
        })
        
        # Adjust sampler based on performance
        if len(self.performance_history) >= 5:
            recent_losses = [p['loss'] for p in list(self.performance_history)[-5:]]
            if np.mean(recent_losses) > np.mean([p['loss'] for p in self.performance_history]):
                # Performance degrading, increase curriculum difficulty more slowly
                self.sampler.config.difficulty_curriculum = True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader performance statistics."""
        return {
            'current_epoch': self.current_epoch,
            'avg_batch_time': np.mean(self.batch_times) if self.batch_times else 0,
            'throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'current_batch_size': self.sampler.current_batch_size,
            'performance_trend': self._compute_performance_trend()
        }
    
    def _compute_performance_trend(self) -> str:
        """Compute performance trend."""
        if len(self.performance_history) < 10:
            return "insufficient_data"
        
        recent = [p['loss'] for p in list(self.performance_history)[-5:]]
        older = [p['loss'] for p in list(self.performance_history)[-10:-5]]
        
        if np.mean(recent) < np.mean(older):
            return "improving"
        elif np.mean(recent) > np.mean(older):
            return "degrading"
        else:
            return "stable"
    
    def __len__(self) -> int:
        return len(self.sampler)


class MultiTaskDataLoader:
    """
    Data loader for multi-task learning scenarios.
    
    Coordinates multiple datasets and provides balanced or weighted sampling across tasks.
    """
    
    def __init__(self, datasets: Dict[str, Dataset], config: LoaderConfig):
        self.datasets = datasets
        self.config = config
        self.task_types = list(datasets.keys())
        
        # Create individual loaders
        self.task_loaders = {}
        for task_type, dataset in datasets.items():
            task_config = LoaderConfig(
                batch_size=config.batch_size // len(self.task_types),  # Distribute batch size
                **{k: v for k, v in config.__dict__.items() if k != 'batch_size'}
            )
            
            self.task_loaders[task_type] = DataLoader(
                dataset,
                batch_size=task_config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                collate_fn=getattr(dataset, '_collate_fn', None)
            )
        
        logger.info(f"MultiTaskDataLoader initialized with {len(self.task_types)} tasks")
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over multi-task batches."""
        # Create iterators
        task_iterators = {
            task_type: iter(loader)
            for task_type, loader in self.task_loaders.items()
        }
        
        while task_iterators:
            multi_task_batch = {}
            
            # Get batch from each available task
            tasks_to_remove = []
            for task_type, iterator in task_iterators.items():
                try:
                    task_batch = next(iterator)
                    multi_task_batch[task_type] = task_batch
                except StopIteration:
                    tasks_to_remove.append(task_type)
            
            # Remove exhausted iterators
            for task_type in tasks_to_remove:
                del task_iterators[task_type]
            
            if multi_task_batch:
                # Add multi-task metadata
                multi_task_batch['_multi_task_info'] = {
                    'active_tasks': list(multi_task_batch.keys()),
                    'num_tasks': len(multi_task_batch),
                    'task_weights': {
                        task: self.config.task_weighting.get(task, 1.0)
                        for task in multi_task_batch.keys()
                    }
                }
                
                yield multi_task_batch
    
    def __len__(self) -> int:
        return min(len(loader) for loader in self.task_loaders.values())


# Factory functions for creating specialized loaders
def create_meta_learning_loader(datasets: Dict[str, Dataset], 
                               shots_per_task: int = 5,
                               query_shots_per_task: int = 10,
                               num_tasks_per_batch: int = 4,
                               **kwargs) -> MetaLearningDataLoader:
    """
    Create a meta-learning data loader.
    
    Args:
        datasets: Dictionary mapping task types to datasets
        shots_per_task: Number of support examples per task
        query_shots_per_task: Number of query examples per task
        num_tasks_per_batch: Number of tasks per meta-batch
        **kwargs: Additional configuration
        
    Returns:
        MetaLearningDataLoader instance
    """
    config = LoaderConfig(
        shots_per_task=shots_per_task,
        query_shots_per_task=query_shots_per_task,
        num_tasks_per_batch=num_tasks_per_batch,
        **kwargs
    )
    
    return MetaLearningDataLoader(datasets, config)


def create_few_shot_loader(dataset: Dataset,
                          shots_per_task: int = 5,
                          query_shots_per_task: int = 10,
                          num_episodes: int = 100,
                          **kwargs) -> FewShotDataLoader:
    """
    Create a few-shot learning data loader.
    
    Args:
        dataset: Dataset to create episodes from
        shots_per_task: Number of support examples
        query_shots_per_task: Number of query examples
        num_episodes: Number of episodes to generate
        **kwargs: Additional configuration
        
    Returns:
        FewShotDataLoader instance
    """
    config = LoaderConfig(
        shots_per_task=shots_per_task,
        query_shots_per_task=query_shots_per_task,
        **kwargs
    )
    
    loader = FewShotDataLoader(dataset, config)
    loader.episode_sampler.num_episodes = num_episodes
    
    return loader


def create_adaptive_loader(dataset: Dataset,
                          initial_batch_size: int = 4,
                          adaptive_batch_size: bool = True,
                          difficulty_curriculum: bool = False,
                          **kwargs) -> AdaptiveDataLoader:
    """
    Create an adaptive data loader.
    
    Args:
        dataset: Dataset to load from
        initial_batch_size: Initial batch size
        adaptive_batch_size: Whether to adapt batch size
        difficulty_curriculum: Whether to use curriculum learning
        **kwargs: Additional configuration
        
    Returns:
        AdaptiveDataLoader instance
    """
    config = LoaderConfig(
        batch_size=initial_batch_size,
        adaptive_batch_size=adaptive_batch_size,
        difficulty_curriculum=difficulty_curriculum,
        **kwargs
    )
    
    return AdaptiveDataLoader(dataset, config)


def create_multi_task_loader(datasets: Dict[str, Dataset],
                            batch_size: int = 8,
                            task_weighting: Dict[str, float] = None,
                            **kwargs) -> MultiTaskDataLoader:
    """
    Create a multi-task data loader.
    
    Args:
        datasets: Dictionary mapping task names to datasets
        batch_size: Total batch size across all tasks
        task_weighting: Weights for task sampling
        **kwargs: Additional configuration
        
    Returns:
        MultiTaskDataLoader instance
    """
    config = LoaderConfig(
        batch_size=batch_size,
        task_weighting=task_weighting or {},
        **kwargs
    )
    
    return MultiTaskDataLoader(datasets, config)


# Utility functions
def compute_loader_efficiency(loader, num_batches: int = 10) -> Dict[str, float]:
    """
    Compute efficiency metrics for a data loader.
    
    Args:
        loader: Data loader to analyze
        num_batches: Number of batches to sample for timing
        
    Returns:
        Dictionary of efficiency metrics
    """
    times = []
    batch_sizes = []
    
    start_time = time.time()
    
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
            
        batch_start = time.time()
        
        # Simulate processing time
        if isinstance(batch, dict):
            batch_size = len(next(iter(batch.values())))
        else:
            batch_size = len(batch)
        
        batch_sizes.append(batch_size)
        times.append(time.time() - batch_start)
    
    total_time = time.time() - start_time
    
    return {
        'avg_batch_time': np.mean(times),
        'std_batch_time': np.std(times),
        'avg_batch_size': np.mean(batch_sizes),
        'throughput': sum(batch_sizes) / total_time,
        'total_time': total_time,
        'num_batches': len(times)
    }