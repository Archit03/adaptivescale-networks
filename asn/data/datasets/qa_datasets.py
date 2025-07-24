"""
Question Answering (QA) datasets for AdaptiveScale Networks.

This module provides implementations for various QA datasets including SQuAD,
Natural Questions, MS MARCO, and others. It handles data loading, preprocessing,
and few-shot sampling for meta-learning scenarios.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict

try:
    from datasets import load_dataset, Dataset as HFDataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    """Single QA example with metadata."""
    
    id: str
    question: str
    context: str
    answers: List[str]
    answer_start: List[int] = field(default_factory=list)
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'question': self.question,
            'context': self.context,
            'answers': self.answers,
            'answer_start': self.answer_start,
            'title': self.title,
            'metadata': self.metadata
        }


@dataclass
class QADatasetConfig:
    """Configuration for QA datasets."""
    
    # Dataset settings
    dataset_name: str = "squad"
    split: str = "train"
    max_examples: Optional[int] = None
    
    # Tokenization settings
    tokenizer_name: str = "gpt2"
    max_seq_length: int = 512
    max_question_length: int = 128
    max_answer_length: int = 64
    
    # Processing settings
    stride: int = 128
    pad_to_max_length: bool = True
    return_overflowing_tokens: bool = True
    
    # Few-shot settings
    shots_per_task: int = 5
    query_shots_per_task: int = 10
    num_tasks: int = 100
    
    # Data augmentation
    use_augmentation: bool = False
    augmentation_probability: float = 0.1
    
    # Filtering
    min_answer_length: int = 1
    max_answer_length_filter: int = 200
    min_context_length: int = 50
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


class BaseQADataset(Dataset):
    """Base class for QA datasets."""
    
    def __init__(self, config: QADatasetConfig):
        self.config = config
        self.examples = []
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Initialized {self.__class__.__name__} with config: {config.dataset_name}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single example."""
        example = self.examples[idx]
        
        if self.tokenizer:
            return self._tokenize_example(example)
        else:
            return example.to_dict()
    
    def _tokenize_example(self, example: QAExample) -> Dict[str, torch.Tensor]:
        """Tokenize a single example."""
        # Combine question and context
        question_context = f"{example.question} {self.tokenizer.sep_token} {example.context}"
        
        # Tokenize input
        inputs = self.tokenizer(
            question_context,
            max_length=self.config.max_seq_length,
            padding='max_length' if self.config.pad_to_max_length else False,
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize answer (for generation tasks)
        if example.answers:
            answer_text = example.answers[0]  # Use first answer
            answer_tokens = self.tokenizer(
                answer_text,
                max_length=self.config.max_answer_length,
                padding='max_length' if self.config.pad_to_max_length else False,
                truncation=True,
                return_tensors='pt'
            )
            
            inputs['answer_input_ids'] = answer_tokens['input_ids']
            inputs['answer_attention_mask'] = answer_tokens['attention_mask']
        
        # Add metadata
        inputs['example_id'] = example.id
        inputs['question_text'] = example.question
        inputs['context_text'] = example.context
        inputs['answer_text'] = example.answers[0] if example.answers else ""
        
        # Squeeze batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                inputs[key] = inputs[key].squeeze(0)
        
        return inputs
    
    def _filter_example(self, example: QAExample) -> bool:
        """Filter examples based on quality criteria."""
        # Check answer length
        if example.answers:
            answer_len = len(example.answers[0])
            if answer_len < self.config.min_answer_length or answer_len > self.config.max_answer_length_filter:
                return False
        
        # Check context length
        if len(example.context) < self.config.min_context_length:
            return False
        
        # Check if answer is in context (for extractive QA)
        if example.answers and example.answers[0].lower() not in example.context.lower():
            return False
        
        return True
    
    def get_few_shot_episodes(self, num_episodes: int = None) -> List[Dict[str, List[QAExample]]]:
        """
        Generate few-shot learning episodes.
        
        Args:
            num_episodes: Number of episodes to generate
            
        Returns:
            List of episodes, each containing support and query sets
        """
        if num_episodes is None:
            num_episodes = self.config.num_tasks
        
        episodes = []
        
        for _ in range(num_episodes):
            # Randomly sample examples
            available_examples = list(self.examples)
            random.shuffle(available_examples)
            
            # Split into support and query
            support_size = self.config.shots_per_task
            query_size = self.config.query_shots_per_task
            total_needed = support_size + query_size
            
            if len(available_examples) < total_needed:
                logger.warning(f"Not enough examples for episode. Need {total_needed}, have {len(available_examples)}")
                continue
            
            support_examples = available_examples[:support_size]
            query_examples = available_examples[support_size:support_size + query_size]
            
            episodes.append({
                'support': support_examples,
                'query': query_examples
            })
        
        return episodes
    
    def create_dataloader(self, batch_size: int = 8, shuffle: bool = True, **kwargs) -> DataLoader:
        """Create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        if not batch:
            return {}
        
        # Handle both tokenized and raw formats
        first_item = batch[0]
        
        if isinstance(first_item, dict) and 'input_ids' in first_item:
            # Tokenized format
            collated = {}
            
            for key in first_item.keys():
                if isinstance(first_item[key], torch.Tensor):
                    collated[key] = torch.stack([item[key] for item in batch])
                elif isinstance(first_item[key], str):
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = [item[key] for item in batch]
            
            return collated
        else:
            # Raw format - convert to list of dictionaries
            return batch


class SQuADDataset(BaseQADataset):
    """Stanford Question Answering Dataset (SQuAD) implementation."""
    
    def __init__(self, config: QADatasetConfig, version: str = "1.1"):
        super().__init__(config)
        self.version = version
        self.load_data()
    
    def load_data(self):
        """Load SQuAD data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for SQuAD loading")
        
        # Load dataset
        if self.version == "2.0":
            dataset_name = "squad_v2"
        else:
            dataset_name = "squad"
        
        logger.info(f"Loading SQuAD {self.version} dataset...")
        
        try:
            dataset = load_dataset(dataset_name, split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to QAExample format
            for item in dataset:
                example = QAExample(
                    id=item['id'],
                    question=item['question'],
                    context=item['context'],
                    answers=item['answers']['text'] if item['answers']['text'] else [""],
                    answer_start=item['answers']['answer_start'] if item['answers']['answer_start'] else [0],
                    title=item.get('title', ''),
                    metadata={'is_impossible': item.get('is_impossible', False)}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} SQuAD examples")
            
        except Exception as e:
            logger.error(f"Failed to load SQuAD dataset: {e}")
            self.examples = []


class NaturalQuestionsDataset(BaseQADataset):
    """Natural Questions dataset implementation."""
    
    def __init__(self, config: QADatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load Natural Questions data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for Natural Questions loading")
        
        logger.info("Loading Natural Questions dataset...")
        
        try:
            dataset = load_dataset("natural_questions", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to QAExample format
            for item in dataset:
                # Extract short answers
                short_answers = []
                if item['annotations']['short_answers']:
                    for short_answer in item['annotations']['short_answers']:
                        if short_answer['start_token'] != -1:
                            start_token = short_answer['start_token']
                            end_token = short_answer['end_token']
                            # Extract text from tokens (simplified)
                            answer_text = ' '.join(item['document']['tokens']['token'][start_token:end_token])
                            short_answers.append(answer_text)
                
                if not short_answers:
                    continue  # Skip examples without answers
                
                # Use document text as context (simplified)
                context = ' '.join(item['document']['tokens']['token'][:1000])  # Limit context size
                
                example = QAExample(
                    id=item['id'],
                    question=item['question']['text'],
                    context=context,
                    answers=short_answers,
                    answer_start=[0] * len(short_answers),  # Simplified
                    metadata={'example_id': item['id']}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} Natural Questions examples")
            
        except Exception as e:
            logger.error(f"Failed to load Natural Questions dataset: {e}")
            self.examples = []


class MSMARCODataset(BaseQADataset):
    """MS MARCO dataset implementation."""
    
    def __init__(self, config: QADatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load MS MARCO data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for MS MARCO loading")
        
        logger.info("Loading MS MARCO dataset...")
        
        try:
            dataset = load_dataset("ms_marco", "v1.1", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to QAExample format
            for item in dataset:
                if not item['answers'] or not item['passages']:
                    continue
                
                # Use first passage as context
                context = item['passages']['passage_text'][0] if item['passages']['passage_text'] else ""
                
                example = QAExample(
                    id=item['query_id'],
                    question=item['query'],
                    context=context,
                    answers=item['answers'],
                    answer_start=[0] * len(item['answers']),  # MS MARCO doesn't provide start positions
                    metadata={'query_type': item.get('query_type', 'unknown')}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} MS MARCO examples")
            
        except Exception as e:
            logger.error(f"Failed to load MS MARCO dataset: {e}")
            self.examples = []


class CoQADataset(BaseQADataset):
    """Conversational Question Answering (CoQA) dataset implementation."""
    
    def __init__(self, config: QADatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load CoQA data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for CoQA loading")
        
        logger.info("Loading CoQA dataset...")
        
        try:
            dataset = load_dataset("coqa", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to QAExample format
            for item in dataset:
                story = item['story']
                questions = item['questions']
                answers = item['answers']
                
                # Create examples for each question-answer pair
                for i, (question, answer) in enumerate(zip(questions, answers)):
                    # Build conversational context
                    conv_context = story
                    if i > 0:
                        # Add previous Q&A pairs for context
                        for j in range(i):
                            conv_context += f"\nQ: {questions[j]['input_text']}\nA: {answers[j]['input_text']}"
                    
                    example = QAExample(
                        id=f"{item['id']}_{i}",
                        question=question['input_text'],
                        context=conv_context,
                        answers=[answer['input_text']],
                        answer_start=[answer.get('span_start', 0)],
                        metadata={
                            'story_id': item['id'],
                            'turn_id': i,
                            'source': item.get('source', 'unknown')
                        }
                    )
                    
                    if self._filter_example(example):
                        self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} CoQA examples")
            
        except Exception as e:
            logger.error(f"Failed to load CoQA dataset: {e}")
            self.examples = []


class CustomQADataset(BaseQADataset):
    """Custom QA dataset for loading from local files."""
    
    def __init__(self, config: QADatasetConfig, data_path: str):
        super().__init__(config)
        self.data_path = Path(data_path)
        self.load_data()
    
    def load_data(self):
        """Load custom QA data from file."""
        logger.info(f"Loading custom QA data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            # Expect data in format: List[Dict] with keys: id, question, context, answers
            for item in data:
                example = QAExample(
                    id=item.get('id', f"custom_{len(self.examples)}"),
                    question=item['question'],
                    context=item['context'],
                    answers=item.get('answers', []),
                    answer_start=item.get('answer_start', [0] * len(item.get('answers', []))),
                    title=item.get('title', ''),
                    metadata=item.get('metadata', {})
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} custom QA examples")
            
        except Exception as e:
            logger.error(f"Failed to load custom QA data: {e}")
            self.examples = []


class QADatasetFactory:
    """Factory for creating QA datasets."""
    
    DATASETS = {
        'squad': SQuADDataset,
        'squad_v2': lambda config: SQuADDataset(config, version="2.0"),
        'natural_questions': NaturalQuestionsDataset,
        'ms_marco': MSMARCODataset,
        'coqa': CoQADataset,
        'custom': CustomQADataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_name: str, config: QADatasetConfig = None, **kwargs) -> BaseQADataset:
        """
        Create a QA dataset.
        
        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration
            **kwargs: Additional arguments for dataset creation
            
        Returns:
            QA dataset instance
        """
        if config is None:
            config = QADatasetConfig(dataset_name=dataset_name)
        
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")
        
        dataset_class = cls.DATASETS[dataset_name]
        
        if dataset_name == 'custom':
            if 'data_path' not in kwargs:
                raise ValueError("data_path required for custom dataset")
            return dataset_class(config, kwargs['data_path'])
        else:
            return dataset_class(config)
    
    @classmethod
    def list_available_datasets(cls) -> List[str]:
        """List all available QA datasets."""
        return list(cls.DATASETS.keys())


# Utility functions
def create_qa_dataloader(dataset_name: str, 
                        split: str = "train",
                        batch_size: int = 8,
                        max_examples: int = None,
                        tokenizer_name: str = "gpt2",
                        **kwargs) -> DataLoader:
    """
    Convenience function to create a QA DataLoader.
    
    Args:
        dataset_name: Name of the QA dataset
        split: Dataset split
        batch_size: Batch size
        max_examples: Maximum number of examples
        tokenizer_name: Tokenizer to use
        **kwargs: Additional configuration
        
    Returns:
        DataLoader for the QA dataset
    """
    config = QADatasetConfig(
        dataset_name=dataset_name,
        split=split,
        max_examples=max_examples,
        tokenizer_name=tokenizer_name,
        **kwargs
    )
    
    dataset = QADatasetFactory.create_dataset(dataset_name, config)
    return dataset.create_dataloader(batch_size=batch_size)


def evaluate_qa_predictions(predictions: List[str], 
                          ground_truths: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate QA predictions using standard metrics.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answer lists
        
    Returns:
        Dictionary of evaluation metrics
    """
    from collections import Counter
    import string
    import re
    
    def normalize_answer(s):
        """Normalize answer for evaluation."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    def exact_match_score(prediction, ground_truth):
        return normalize_answer(prediction) == normalize_answer(ground_truth)
    
    def f1_score(prediction, ground_truth):
        pred_tokens = normalize_answer(prediction).split()
        truth_tokens = normalize_answer(ground_truth).split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common_tokens.values())
        
        if num_same == 0:
            return 0
        
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    # Compute metrics
    exact_matches = []
    f1_scores = []
    
    for pred, truth_list in zip(predictions, ground_truths):
        # Find best match among ground truths
        best_exact = max([exact_match_score(pred, truth) for truth in truth_list])
        best_f1 = max([f1_score(pred, truth) for truth in truth_list])
        
        exact_matches.append(best_exact)
        f1_scores.append(best_f1)
    
    return {
        'exact_match': sum(exact_matches) / len(exact_matches) * 100,
        'f1_score': sum(f1_scores) / len(f1_scores) * 100,
        'num_examples': len(predictions)
    }


def create_few_shot_qa_task(dataset: BaseQADataset, 
                           num_tasks: int = 100,
                           shots_per_task: int = 5) -> List[Dict[str, Any]]:
    """
    Create few-shot QA tasks for meta-learning.
    
    Args:
        dataset: QA dataset
        num_tasks: Number of tasks to create
        shots_per_task: Number of examples per task
        
    Returns:
        List of few-shot tasks
    """
    # Override dataset config for few-shot
    dataset.config.shots_per_task = shots_per_task
    dataset.config.num_tasks = num_tasks
    
    return dataset.get_few_shot_episodes(num_tasks)