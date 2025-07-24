"""
Logical reasoning datasets for AdaptiveScale Networks.

This module provides implementations for various logical reasoning datasets
including ARC, CommonsenseQA, PIQA, HellaSwag, and others. It handles 
multiple choice questions, logical inference, and commonsense reasoning.
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
class ReasoningExample:
    """Single logical reasoning example."""
    
    id: str
    question: str
    choices: List[str]
    answer: Union[str, int]
    context: Optional[str] = None
    explanation: Optional[str] = None
    reasoning_type: str = "multiple_choice"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'question': self.question,
            'choices': self.choices,
            'answer': self.answer,
            'context': self.context,
            'explanation': self.explanation,
            'reasoning_type': self.reasoning_type,
            'difficulty': self.difficulty,
            'metadata': self.metadata
        }
    
    def get_correct_choice_text(self) -> Optional[str]:
        """Get the text of the correct choice."""
        if isinstance(self.answer, int) and 0 <= self.answer < len(self.choices):
            return self.choices[self.answer]
        elif isinstance(self.answer, str):
            # Try to find matching choice
            answer_lower = self.answer.lower().strip()
            for i, choice in enumerate(self.choices):
                if choice.lower().strip() == answer_lower:
                    return choice
                # Also check for letter matching (A, B, C, D)
                if len(answer_lower) == 1 and ord(answer_lower) - ord('a') == i:
                    return choice
        return None
    
    def get_correct_choice_index(self) -> Optional[int]:
        """Get the index of the correct choice."""
        if isinstance(self.answer, int):
            return self.answer if 0 <= self.answer < len(self.choices) else None
        elif isinstance(self.answer, str):
            answer_lower = self.answer.lower().strip()
            # Check for direct text match
            for i, choice in enumerate(self.choices):
                if choice.lower().strip() == answer_lower:
                    return i
            # Check for letter match (A, B, C, D)
            if len(answer_lower) == 1:
                idx = ord(answer_lower) - ord('a')
                if 0 <= idx < len(self.choices):
                    return idx
        return None


@dataclass
class ReasoningDatasetConfig:
    """Configuration for reasoning datasets."""
    
    # Dataset settings
    dataset_name: str = "arc"
    split: str = "train"
    subset: Optional[str] = "ARC-Easy"  # For datasets with subsets
    max_examples: Optional[int] = None
    
    # Tokenization settings
    tokenizer_name: str = "gpt2"
    max_seq_length: int = 512
    max_choice_length: int = 128
    
    # Processing settings
    shuffle_choices: bool = False
    include_choice_letters: bool = True
    format_template: str = "Question: {question}\nChoices: {choices}\nAnswer: {answer}"
    
    # Few-shot settings
    shots_per_task: int = 5
    query_shots_per_task: int = 10
    num_tasks: int = 100
    
    # Filtering
    min_choices: int = 2
    max_choices: int = 10
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


class BaseReasoningDataset(Dataset):
    """Base class for reasoning datasets."""
    
    def __init__(self, config: ReasoningDatasetConfig):
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
    
    def _tokenize_example(self, example: ReasoningExample) -> Dict[str, torch.Tensor]:
        """Tokenize a reasoning example."""
        # Format choices
        if self.config.include_choice_letters:
            choices_text = "\n".join([f"({chr(65+i)}) {choice}" for i, choice in enumerate(example.choices)])
        else:
            choices_text = "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(example.choices)])
        
        # Create full question text
        question_with_choices = f"Question: {example.question}\n\nChoices:\n{choices_text}"
        
        if example.context:
            question_with_choices = f"Context: {example.context}\n\n{question_with_choices}"
        
        # Get correct answer text
        correct_choice_text = example.get_correct_choice_text()
        correct_choice_idx = example.get_correct_choice_index()
        
        if correct_choice_text:
            if self.config.include_choice_letters:
                answer_text = f"({chr(65+correct_choice_idx)}) {correct_choice_text}"
            else:
                answer_text = correct_choice_text
        else:
            answer_text = str(example.answer)
        
        # Format using template
        full_text = self.config.format_template.format(
            question=example.question,
            choices=choices_text,
            answer=answer_text,
            context=example.context or ""
        )
        
        # Tokenize full text
        inputs = self.tokenizer(
            full_text,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize just the question for generation
        question_inputs = self.tokenizer(
            question_with_choices + "\n\nAnswer:",
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize each choice separately
        choice_inputs = []
        for choice in example.choices:
            choice_tokens = self.tokenizer(
                choice,
                max_length=self.config.max_choice_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            choice_inputs.append(choice_tokens)
        
        # Stack choice inputs
        choice_input_ids = torch.stack([c['input_ids'] for c in choice_inputs])
        choice_attention_masks = torch.stack([c['attention_mask'] for c in choice_inputs])
        
        # Add all components
        inputs['question_input_ids'] = question_inputs['input_ids']
        inputs['question_attention_mask'] = question_inputs['attention_mask']
        inputs['choice_input_ids'] = choice_input_ids
        inputs['choice_attention_mask'] = choice_attention_masks
        
        # Add metadata
        inputs['example_id'] = example.id
        inputs['question_text'] = example.question
        inputs['choices_text'] = example.choices
        inputs['answer_text'] = answer_text
        inputs['correct_choice_idx'] = torch.tensor(correct_choice_idx if correct_choice_idx is not None else -1)
        inputs['reasoning_type'] = example.reasoning_type
        inputs['difficulty'] = example.difficulty
        
        # Squeeze batch dimensions
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1 and key not in ['choice_input_ids', 'choice_attention_mask']:
                inputs[key] = inputs[key].squeeze(0)
        
        # Handle choice tensors specially (keep first dim for multiple choices)
        inputs['choice_input_ids'] = inputs['choice_input_ids'].squeeze(1)
        inputs['choice_attention_mask'] = inputs['choice_attention_mask'].squeeze(1)
        
        return inputs
    
    def _filter_example(self, example: ReasoningExample) -> bool:
        """Filter examples based on quality criteria."""
        # Check number of choices
        num_choices = len(example.choices)
        if num_choices < self.config.min_choices or num_choices > self.config.max_choices:
            return False
        
        # Check if answer is valid
        if example.get_correct_choice_index() is None:
            return False
        
        return True
    
    def get_few_shot_episodes(self, num_episodes: int = None) -> List[Dict[str, List[ReasoningExample]]]:
        """Generate few-shot learning episodes for reasoning tasks."""
        if num_episodes is None:
            num_episodes = self.config.num_tasks
        
        episodes = []
        
        # Group examples by reasoning type
        type_groups = defaultdict(list)
        for example in self.examples:
            type_groups[example.reasoning_type].append(example)
        
        for _ in range(num_episodes):
            # Choose a reasoning type for this episode
            if type_groups:
                reasoning_type = random.choice(list(type_groups.keys()))
                available_examples = list(type_groups[reasoning_type])
            else:
                available_examples = list(self.examples)
            
            random.shuffle(available_examples)
            
            # Split into support and query
            support_size = self.config.shots_per_task
            query_size = self.config.query_shots_per_task
            total_needed = support_size + query_size
            
            if len(available_examples) < total_needed:
                # Fall back to all examples if not enough in this type
                available_examples = list(self.examples)
                random.shuffle(available_examples)
                
                if len(available_examples) < total_needed:
                    continue
            
            support_examples = available_examples[:support_size]
            query_examples = available_examples[support_size:support_size + query_size]
            
            episodes.append({
                'support': support_examples,
                'query': query_examples,
                'reasoning_type': reasoning_type if type_groups else 'mixed'
            })
        
        return episodes
    
    def create_dataloader(self, batch_size: int = 4, shuffle: bool = True, **kwargs) -> DataLoader:
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
        
        first_item = batch[0]
        
        if isinstance(first_item, dict) and 'input_ids' in first_item:
            # Tokenized format
            collated = {}
            
            for key in first_item.keys():
                if isinstance(first_item[key], torch.Tensor):
                    if key in ['choice_input_ids', 'choice_attention_mask']:
                        # Handle variable number of choices
                        max_choices = max(item[key].size(0) for item in batch)
                        padded_tensors = []
                        for item in batch:
                            tensor = item[key]
                            if tensor.size(0) < max_choices:
                                # Pad with zeros
                                pad_size = (max_choices - tensor.size(0),) + tensor.shape[1:]
                                padding = torch.zeros(pad_size, dtype=tensor.dtype)
                                tensor = torch.cat([tensor, padding], dim=0)
                            padded_tensors.append(tensor)
                        collated[key] = torch.stack(padded_tensors)
                    else:
                        collated[key] = torch.stack([item[key] for item in batch])
                elif isinstance(first_item[key], str):
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = [item[key] for item in batch]
            
            return collated
        else:
            return batch


class ARCDataset(BaseReasoningDataset):
    """AI2 Reasoning Challenge (ARC) dataset implementation."""
    
    def __init__(self, config: ReasoningDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load ARC data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for ARC loading")
        
        logger.info(f"Loading ARC dataset (subset: {self.config.subset})...")
        
        try:
            # ARC has ARC-Easy and ARC-Challenge subsets
            if self.config.subset:
                dataset = load_dataset("ai2_arc", self.config.subset, split=self.config.split, cache_dir=self.config.cache_dir)
            else:
                dataset = load_dataset("ai2_arc", "ARC-Easy", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to ReasoningExample format
            for item in dataset:
                choices = item['choices']['text']
                labels = item['choices']['label']
                
                # Find correct answer index
                answer_label = item['answerKey']
                try:
                    answer_idx = labels.index(answer_label)
                except ValueError:
                    # Try alternative matching
                    answer_idx = ord(answer_label.upper()) - ord('A') if len(answer_label) == 1 else 0
                    if answer_idx >= len(choices):
                        continue
                
                example = ReasoningExample(
                    id=item['id'],
                    question=item['question'],
                    choices=choices,
                    answer=answer_idx,
                    reasoning_type="science_qa",
                    difficulty="easy" if self.config.subset == "ARC-Easy" else "hard",
                    metadata={
                        'source': 'arc',
                        'subset': self.config.subset,
                        'answer_key': answer_label
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} ARC examples")
            
        except Exception as e:
            logger.error(f"Failed to load ARC dataset: {e}")
            self.examples = []


class CommonsenseQADataset(BaseReasoningDataset):
    """CommonsenseQA dataset implementation."""
    
    def __init__(self, config: ReasoningDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load CommonsenseQA data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for CommonsenseQA loading")
        
        logger.info("Loading CommonsenseQA dataset...")
        
        try:
            dataset = load_dataset("commonsense_qa", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to ReasoningExample format
            for item in dataset:
                choices = item['choices']['text']
                labels = item['choices']['label']
                
                # Find correct answer index
                answer_label = item['answerKey']
                try:
                    answer_idx = labels.index(answer_label)
                except ValueError:
                    answer_idx = ord(answer_label.upper()) - ord('A') if len(answer_label) == 1 else 0
                    if answer_idx >= len(choices):
                        continue
                
                example = ReasoningExample(
                    id=item['id'],
                    question=item['question'],
                    choices=choices,
                    answer=answer_idx,
                    reasoning_type="commonsense",
                    difficulty="medium",
                    metadata={
                        'source': 'commonsense_qa',
                        'answer_key': answer_label,
                        'question_concept': item.get('question_concept', '')
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} CommonsenseQA examples")
            
        except Exception as e:
            logger.error(f"Failed to load CommonsenseQA dataset: {e}")
            self.examples = []


class PIQADataset(BaseReasoningDataset):
    """Physical Interaction QA (PIQA) dataset implementation."""
    
    def __init__(self, config: ReasoningDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load PIQA data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for PIQA loading")
        
        logger.info("Loading PIQA dataset...")
        
        try:
            dataset = load_dataset("piqa", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to ReasoningExample format
            for i, item in enumerate(dataset):
                choices = [item['sol1'], item['sol2']]
                answer_idx = item['label']  # 0 or 1
                
                example = ReasoningExample(
                    id=f"piqa_{i}",
                    question=item['goal'],
                    choices=choices,
                    answer=answer_idx,
                    reasoning_type="physical_reasoning",
                    difficulty="medium",
                    metadata={'source': 'piqa'}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} PIQA examples")
            
        except Exception as e:
            logger.error(f"Failed to load PIQA dataset: {e}")
            self.examples = []


class HellaSwagDataset(BaseReasoningDataset):
    """HellaSwag dataset implementation."""
    
    def __init__(self, config: ReasoningDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load HellaSwag data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for HellaSwag loading")
        
        logger.info("Loading HellaSwag dataset...")
        
        try:
            dataset = load_dataset("hellaswag", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to ReasoningExample format
            for item in dataset:
                context = item['ctx']
                choices = item['endings']
                answer_idx = int(item['label']) if item['label'] else 0
                
                example = ReasoningExample(
                    id=item['ind'],
                    question="What happens next?",
                    choices=choices,
                    answer=answer_idx,
                    context=context,
                    reasoning_type="commonsense_inference",
                    difficulty="hard",
                    metadata={
                        'source': 'hellaswag',
                        'activity_label': item.get('activity_label', ''),
                        'source_id': item.get('source_id', '')
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} HellaSwag examples")
            
        except Exception as e:
            logger.error(f"Failed to load HellaSwag dataset: {e}")
            self.examples = []


class WinograndeDataset(BaseReasoningDataset):
    """Winogrande dataset implementation."""
    
    def __init__(self, config: ReasoningDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load Winogrande data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for Winogrande loading")
        
        logger.info("Loading Winogrande dataset...")
        
        try:
            # Winogrande has different sizes: xs, s, m, l, xl
            size = self.config.subset or "winogrande_l"
            dataset = load_dataset("winogrande", size, split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to ReasoningExample format
            for item in dataset:
                sentence = item['sentence']
                option1 = item['option1']
                option2 = item['option2']
                choices = [option1, option2]
                
                # Answer is 1 or 2, convert to 0-indexed
                answer_idx = int(item['answer']) - 1 if item['answer'] else 0
                
                example = ReasoningExample(
                    id=f"winogrande_{item.get('qID', len(self.examples))}",
                    question=sentence,
                    choices=choices,
                    answer=answer_idx,
                    reasoning_type="coreference_resolution",
                    difficulty="hard",
                    metadata={
                        'source': 'winogrande',
                        'size': size
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} Winogrande examples")
            
        except Exception as e:
            logger.error(f"Failed to load Winogrande dataset: {e}")
            self.examples = []


class OpenBookQADataset(BaseReasoningDataset):
    """OpenBookQA dataset implementation."""
    
    def __init__(self, config: ReasoningDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load OpenBookQA data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for OpenBookQA loading")
        
        logger.info("Loading OpenBookQA dataset...")
        
        try:
            dataset = load_dataset("openbookqa", "main", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to ReasoningExample format
            for item in dataset:
                choices = item['choices']['text']
                labels = item['choices']['label']
                
                # Find correct answer index
                answer_label = item['answerKey']
                try:
                    answer_idx = labels.index(answer_label)
                except ValueError:
                    answer_idx = ord(answer_label.upper()) - ord('A') if len(answer_label) == 1 else 0
                    if answer_idx >= len(choices):
                        continue
                
                example = ReasoningExample(
                    id=item['id'],
                    question=item['question_stem'],
                    choices=choices,
                    answer=answer_idx,
                    reasoning_type="science_qa",
                    difficulty="medium",
                    metadata={
                        'source': 'openbookqa',
                        'answer_key': answer_label
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} OpenBookQA examples")
            
        except Exception as e:
            logger.error(f"Failed to load OpenBookQA dataset: {e}")
            self.examples = []


class CustomReasoningDataset(BaseReasoningDataset):
    """Custom reasoning dataset for loading from local files."""
    
    def __init__(self, config: ReasoningDatasetConfig, data_path: str):
        super().__init__(config)
        self.data_path = Path(data_path)
        self.load_data()
    
    def load_data(self):
        """Load custom reasoning data from file."""
        logger.info(f"Loading custom reasoning data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            # Expect data in format: List[Dict] with keys: id, question, choices, answer
            for item in data:
                example = ReasoningExample(
                    id=item.get('id', f"custom_{len(self.examples)}"),
                    question=item['question'],
                    choices=item['choices'],
                    answer=item['answer'],
                    context=item.get('context'),
                    explanation=item.get('explanation'),
                    reasoning_type=item.get('reasoning_type', 'general'),
                    difficulty=item.get('difficulty', 'medium'),
                    metadata=item.get('metadata', {})
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} custom reasoning examples")
            
        except Exception as e:
            logger.error(f"Failed to load custom reasoning data: {e}")
            self.examples = []


class ReasoningDatasetFactory:
    """Factory for creating reasoning datasets."""
    
    DATASETS = {
        'arc_easy': lambda config: ARCDataset({**config.__dict__, 'subset': 'ARC-Easy'}),
        'arc_challenge': lambda config: ARCDataset({**config.__dict__, 'subset': 'ARC-Challenge'}),
        'commonsense_qa': CommonsenseQADataset,
        'piqa': PIQADataset,
        'hellaswag': HellaSwagDataset,
        'winogrande': WinograndeDataset,
        'openbookqa': OpenBookQADataset,
        'custom': CustomReasoningDataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_name: str, config: ReasoningDatasetConfig = None, **kwargs) -> BaseReasoningDataset:
        """Create a reasoning dataset."""
        if config is None:
            config = ReasoningDatasetConfig(dataset_name=dataset_name)
        
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")
        
        dataset_class = cls.DATASETS[dataset_name]
        
        if dataset_name == 'custom':
            if 'data_path' not in kwargs:
                raise ValueError("data_path required for custom dataset")
            return dataset_class(config, kwargs['data_path'])
        elif callable(dataset_class):
            return dataset_class(config)
        else:
            return dataset_class(config)
    
    @classmethod
    def list_available_datasets(cls) -> List[str]:
        """List all available reasoning datasets."""
        return list(cls.DATASETS.keys())


# Utility functions
def evaluate_reasoning_predictions(predictions: List[Union[str, int]], 
                                 ground_truths: List[Union[str, int]],
                                 choices_list: List[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate reasoning predictions with choice accuracy.
    
    Args:
        predictions: List of predicted answers (can be indices or text)
        ground_truths: List of ground truth answers
        choices_list: List of choice lists for each example
        
    Returns:
        Dictionary of evaluation metrics
    """
    def normalize_answer(answer, choices=None):
        """Normalize answer for comparison."""
        if isinstance(answer, int):
            return answer
        
        if isinstance(answer, str):
            answer = answer.strip().lower()
            
            # Try to extract choice letter (A, B, C, D)
            if len(answer) == 1 and 'a' <= answer <= 'z':
                return ord(answer) - ord('a')
            
            # Try to match against choices if provided
            if choices:
                for i, choice in enumerate(choices):
                    if choice.strip().lower() == answer:
                        return i
        
        return answer
    
    correct = 0
    total = len(predictions)
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
        choices = choices_list[i] if choices_list and i < len(choices_list) else None
        
        norm_pred = normalize_answer(pred, choices)
        norm_truth = normalize_answer(truth, choices)
        
        if norm_pred == norm_truth:
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    
    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def create_reasoning_dataloader(dataset_name: str,
                              split: str = "train",
                              batch_size: int = 4,
                              max_examples: int = None,
                              **kwargs) -> DataLoader:
    """Convenience function to create reasoning DataLoader."""
    config = ReasoningDatasetConfig(
        dataset_name=dataset_name,
        split=split,
        max_examples=max_examples,
        **kwargs
    )
    
    dataset = ReasoningDatasetFactory.create_dataset(dataset_name, config)
    return dataset.create_dataloader(batch_size=batch_size)


def create_few_shot_reasoning_tasks(dataset: BaseReasoningDataset,
                                  num_tasks: int = 100,
                                  shots_per_task: int = 5) -> List[Dict[str, Any]]:
    """Create few-shot reasoning tasks for meta-learning."""
    dataset.config.shots_per_task = shots_per_task
    dataset.config.num_tasks = num_tasks
    
    return dataset.get_few_shot_episodes(num_tasks)