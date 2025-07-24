"""
Mathematical reasoning datasets for AdaptiveScale Networks.

This module provides implementations for various mathematical reasoning datasets
including GSM8K, MATH, MathQA, and others. It handles complex mathematical
expressions, step-by-step solutions, and numerical evaluation.
"""

import json
import logging
import random
import re
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

try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    HAS_SYMPY = False

logger = logging.getLogger(__name__)


@dataclass
class MathExample:
    """Single mathematical reasoning example."""
    
    id: str
    problem: str
    solution: str
    answer: Union[str, float, int]
    steps: List[str] = field(default_factory=list)
    problem_type: str = "arithmetic"
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'problem': self.problem,
            'solution': self.solution,
            'answer': self.answer,
            'steps': self.steps,
            'problem_type': self.problem_type,
            'difficulty': self.difficulty,
            'metadata': self.metadata
        }
    
    def extract_numerical_answer(self) -> Optional[float]:
        """Extract numerical answer from string."""
        if isinstance(self.answer, (int, float)):
            return float(self.answer)
        
        # Try to extract number from string
        if isinstance(self.answer, str):
            # Look for patterns like $123.45 or 123.45
            patterns = [
                r'\$?([\d,]+\.?\d*)',
                r'([\d,]+\.?\d*)',
                r'(-?[\d,]+\.?\d*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, self.answer.replace(',', ''))
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
        
        return None


@dataclass
class MathDatasetConfig:
    """Configuration for mathematical reasoning datasets."""
    
    # Dataset settings
    dataset_name: str = "gsm8k"
    split: str = "train"
    max_examples: Optional[int] = None
    
    # Tokenization settings
    tokenizer_name: str = "gpt2"
    max_seq_length: int = 1024
    max_solution_length: int = 512
    
    # Processing settings
    include_steps: bool = True
    parse_latex: bool = True
    normalize_numbers: bool = True
    
    # Few-shot settings
    shots_per_task: int = 5
    query_shots_per_task: int = 10
    num_tasks: int = 100
    
    # Problem type filtering
    problem_types: List[str] = field(default_factory=lambda: ["arithmetic", "algebra", "geometry", "probability"])
    difficulty_levels: List[str] = field(default_factory=lambda: ["easy", "medium", "hard"])
    
    # Answer validation
    validate_answers: bool = True
    numerical_tolerance: float = 1e-6
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


class BaseMathDataset(Dataset):
    """Base class for mathematical reasoning datasets."""
    
    def __init__(self, config: MathDatasetConfig):
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
    
    def _tokenize_example(self, example: MathExample) -> Dict[str, torch.Tensor]:
        """Tokenize a mathematical example."""
        # Format problem with solution
        if self.config.include_steps and example.steps:
            # Include step-by-step solution
            steps_text = "\nStep-by-step solution:\n" + "\n".join(f"{i+1}. {step}" for i, step in enumerate(example.steps))
            input_text = f"Problem: {example.problem}{steps_text}\nFinal answer: {example.answer}"
        else:
            # Simple problem-solution format
            input_text = f"Problem: {example.problem}\nSolution: {example.solution}\nAnswer: {example.answer}"
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize just the problem for generation tasks
        problem_inputs = self.tokenizer(
            f"Problem: {example.problem}\nSolution:",
            max_length=self.config.max_seq_length // 2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize solution for training
        solution_inputs = self.tokenizer(
            example.solution,
            max_length=self.config.max_solution_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Add all tokenized components
        inputs['problem_input_ids'] = problem_inputs['input_ids']
        inputs['problem_attention_mask'] = problem_inputs['attention_mask']
        inputs['solution_input_ids'] = solution_inputs['input_ids']
        inputs['solution_attention_mask'] = solution_inputs['attention_mask']
        
        # Add metadata
        inputs['example_id'] = example.id
        inputs['problem_text'] = example.problem
        inputs['solution_text'] = example.solution
        inputs['answer_text'] = str(example.answer)
        inputs['problem_type'] = example.problem_type
        inputs['difficulty'] = example.difficulty
        
        # Extract numerical answer if possible
        numerical_answer = example.extract_numerical_answer()
        inputs['numerical_answer'] = torch.tensor(numerical_answer if numerical_answer is not None else -999.0)
        
        # Squeeze batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                inputs[key] = inputs[key].squeeze(0)
        
        return inputs
    
    def _filter_example(self, example: MathExample) -> bool:
        """Filter examples based on quality criteria."""
        # Check problem type
        if self.config.problem_types and example.problem_type not in self.config.problem_types:
            return False
        
        # Check difficulty
        if self.config.difficulty_levels and example.difficulty not in self.config.difficulty_levels:
            return False
        
        # Validate answer if required
        if self.config.validate_answers:
            if not self._validate_answer(example):
                return False
        
        return True
    
    def _validate_answer(self, example: MathExample) -> bool:
        """Validate mathematical answer."""
        # Check if answer can be extracted
        numerical_answer = example.extract_numerical_answer()
        return numerical_answer is not None
    
    def _parse_latex(self, text: str) -> str:
        """Parse LaTeX expressions in text."""
        if not self.config.parse_latex or not HAS_SYMPY:
            return text
        
        # Simple LaTeX parsing - can be extended
        text = text.replace('\\frac', 'frac')
        text = text.replace('\\sqrt', 'sqrt')
        text = text.replace('\\cdot', '*')
        text = text.replace('\\times', '*')
        
        return text
    
    def get_few_shot_episodes(self, num_episodes: int = None) -> List[Dict[str, List[MathExample]]]:
        """Generate few-shot learning episodes for mathematical reasoning."""
        if num_episodes is None:
            num_episodes = self.config.num_tasks
        
        episodes = []
        
        # Group examples by problem type for more coherent episodes
        type_groups = defaultdict(list)
        for example in self.examples:
            type_groups[example.problem_type].append(example)
        
        for _ in range(num_episodes):
            # Choose a problem type for this episode
            if type_groups:
                problem_type = random.choice(list(type_groups.keys()))
                available_examples = list(type_groups[problem_type])
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
                'problem_type': problem_type if type_groups else 'mixed'
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
                    collated[key] = torch.stack([item[key] for item in batch])
                elif isinstance(first_item[key], str):
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = [item[key] for item in batch]
            
            return collated
        else:
            return batch


class GSM8KDataset(BaseMathDataset):
    """Grade School Math 8K (GSM8K) dataset implementation."""
    
    def __init__(self, config: MathDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load GSM8K data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for GSM8K loading")
        
        logger.info("Loading GSM8K dataset...")
        
        try:
            dataset = load_dataset("gsm8k", "main", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to MathExample format
            for i, item in enumerate(dataset):
                # Extract steps from solution
                solution = item['answer']
                steps = self._extract_steps_from_solution(solution)
                
                # Extract final answer
                answer = self._extract_final_answer(solution)
                
                example = MathExample(
                    id=f"gsm8k_{i}",
                    problem=item['question'],
                    solution=solution,
                    answer=answer,
                    steps=steps,
                    problem_type="arithmetic",
                    difficulty="elementary",
                    metadata={'source': 'gsm8k'}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} GSM8K examples")
            
        except Exception as e:
            logger.error(f"Failed to load GSM8K dataset: {e}")
            self.examples = []
    
    def _extract_steps_from_solution(self, solution: str) -> List[str]:
        """Extract step-by-step reasoning from GSM8K solution."""
        # GSM8K solutions often have steps separated by periods or newlines
        steps = []
        
        # Split by sentences and clean up
        sentences = re.split(r'[.!?]\s+', solution)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and not sentence.startswith('####'):
                steps.append(sentence)
        
        return steps
    
    def _extract_final_answer(self, solution: str) -> str:
        """Extract final numerical answer from GSM8K solution."""
        # GSM8K format: #### ANSWER
        match = re.search(r'####\s*(.+)', solution)
        if match:
            return match.group(1).strip()
        
        # Fallback: extract last number
        numbers = re.findall(r'[\d,]+\.?\d*', solution)
        return numbers[-1] if numbers else "0"


class MATHDataset(BaseMathDataset):
    """MATH dataset implementation (competition mathematics)."""
    
    def __init__(self, config: MathDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load MATH dataset."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for MATH dataset loading")
        
        logger.info("Loading MATH dataset...")
        
        try:
            dataset = load_dataset("competition_math", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to MathExample format
            for i, item in enumerate(dataset):
                # Parse LaTeX if enabled
                problem = self._parse_latex(item['problem']) if self.config.parse_latex else item['problem']
                solution = self._parse_latex(item['solution']) if self.config.parse_latex else item['solution']
                
                # Extract steps from solution
                steps = self._extract_steps_from_math_solution(solution)
                
                example = MathExample(
                    id=f"math_{i}",
                    problem=problem,
                    solution=solution,
                    answer=item['solution'].split('\n')[-1] if '\n' in item['solution'] else item['solution'],
                    steps=steps,
                    problem_type=item['type'].lower(),
                    difficulty=item['level'],
                    metadata={'source': 'math', 'type': item['type'], 'level': item['level']}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} MATH examples")
            
        except Exception as e:
            logger.error(f"Failed to load MATH dataset: {e}")
            self.examples = []
    
    def _extract_steps_from_math_solution(self, solution: str) -> List[str]:
        """Extract steps from MATH dataset solution."""
        steps = []
        
        # Split by common mathematical reasoning indicators
        patterns = [
            r'\n\n',  # Double newlines
            r'(?:Step \d+[:.]\s*)',  # Explicit steps
            r'(?:First,|Next,|Then,|Finally,)',  # Sequence words
            r'(?:Since|Because|Therefore|Thus)',  # Logical connectors
        ]
        
        # Try to split intelligently
        current_text = solution
        for pattern in patterns:
            parts = re.split(pattern, current_text)
            if len(parts) > 1:
                steps = [part.strip() for part in parts if part.strip()]
                break
        
        # Fallback: split by sentences
        if not steps:
            steps = [s.strip() for s in re.split(r'[.!?]\s+', solution) if s.strip()]
        
        return steps


class MathQADataset(BaseMathDataset):
    """MathQA dataset implementation."""
    
    def __init__(self, config: MathDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load MathQA data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for MathQA loading")
        
        logger.info("Loading MathQA dataset...")
        
        try:
            dataset = load_dataset("math_qa", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to MathExample format
            for i, item in enumerate(dataset):
                # MathQA has multiple choice options
                options_text = "\n".join([f"({chr(97+j)}) {opt}" for j, opt in enumerate(item['options'])])
                problem_with_options = f"{item['Problem']}\n\nOptions:\n{options_text}"
                
                example = MathExample(
                    id=f"mathqa_{i}",
                    problem=problem_with_options,
                    solution=item['Rationale'],
                    answer=item['correct'],
                    steps=self._extract_steps_from_rationale(item['Rationale']),
                    problem_type=item.get('category', 'general').lower(),
                    difficulty="medium",
                    metadata={
                        'source': 'mathqa',
                        'options': item['options'],
                        'correct_option': item['correct'],
                        'category': item.get('category', 'general')
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} MathQA examples")
            
        except Exception as e:
            logger.error(f"Failed to load MathQA dataset: {e}")
            self.examples = []
    
    def _extract_steps_from_rationale(self, rationale: str) -> List[str]:
        """Extract steps from MathQA rationale."""
        # MathQA rationales often have step indicators
        steps = []
        
        # Look for numbered steps or bullet points
        step_patterns = [
            r'(?:Step \d+[:.]\s*)([^.]+)',
            r'(?:\d+[.)]\s*)([^.]+)',
            r'(?:•\s*)([^.]+)',
            r'(?:- \s*)([^.]+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, rationale, re.IGNORECASE)
            if matches:
                steps = [match.strip() for match in matches]
                break
        
        # Fallback: split by sentences
        if not steps:
            steps = [s.strip() for s in re.split(r'[.!?]\s+', rationale) if s.strip()]
        
        return steps


class AQuADataset(BaseMathDataset):
    """AQuA (Algebra Question Answering) dataset implementation."""
    
    def __init__(self, config: MathDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load AQuA data."""
        logger.info("Loading AQuA dataset...")
        
        # AQuA is typically loaded from JSON files
        # This is a placeholder implementation
        try:
            # Would load from local AQuA files
            # For now, create some synthetic examples
            self._create_synthetic_examples()
            
            logger.info(f"Loaded {len(self.examples)} AQuA examples")
            
        except Exception as e:
            logger.error(f"Failed to load AQuA dataset: {e}")
            self.examples = []
    
    def _create_synthetic_examples(self):
        """Create synthetic algebraic examples."""
        synthetic_problems = [
            {
                'problem': 'If x + 5 = 12, what is the value of x?',
                'solution': 'To solve x + 5 = 12, subtract 5 from both sides: x = 12 - 5 = 7',
                'answer': '7',
                'type': 'algebra'
            },
            {
                'problem': 'A rectangle has length 8 and width 3. What is its area?',
                'solution': 'Area of rectangle = length × width = 8 × 3 = 24',
                'answer': '24',
                'type': 'geometry'
            },
            {
                'problem': 'What is 15% of 80?',
                'solution': '15% of 80 = 0.15 × 80 = 12',
                'answer': '12',
                'type': 'arithmetic'
            }
        ]
        
        for i, prob in enumerate(synthetic_problems):
            example = MathExample(
                id=f"aqua_synthetic_{i}",
                problem=prob['problem'],
                solution=prob['solution'],
                answer=prob['answer'],
                steps=[prob['solution']],
                problem_type=prob['type'],
                difficulty="medium",
                metadata={'source': 'aqua_synthetic'}
            )
            self.examples.append(example)


class CustomMathDataset(BaseMathDataset):
    """Custom mathematical reasoning dataset."""
    
    def __init__(self, config: MathDatasetConfig, data_path: str):
        super().__init__(config)
        self.data_path = Path(data_path)
        self.load_data()
    
    def load_data(self):
        """Load custom math data from file."""
        logger.info(f"Loading custom math data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            # Expect data in format: List[Dict] with keys: id, problem, solution, answer
            for item in data:
                example = MathExample(
                    id=item.get('id', f"custom_{len(self.examples)}"),
                    problem=item['problem'],
                    solution=item['solution'],
                    answer=item['answer'],
                    steps=item.get('steps', []),
                    problem_type=item.get('problem_type', 'general'),
                    difficulty=item.get('difficulty', 'medium'),
                    metadata=item.get('metadata', {})
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} custom math examples")
            
        except Exception as e:
            logger.error(f"Failed to load custom math data: {e}")
            self.examples = []


class MathDatasetFactory:
    """Factory for creating mathematical reasoning datasets."""
    
    DATASETS = {
        'gsm8k': GSM8KDataset,
        'math': MATHDataset,
        'mathqa': MathQADataset,
        'aqua': AQuADataset,
        'custom': CustomMathDataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_name: str, config: MathDatasetConfig = None, **kwargs) -> BaseMathDataset:
        """Create a mathematical reasoning dataset."""
        if config is None:
            config = MathDatasetConfig(dataset_name=dataset_name)
        
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
        """List all available math datasets."""
        return list(cls.DATASETS.keys())


# Utility functions
def evaluate_math_predictions(predictions: List[str], 
                            ground_truths: List[str],
                            tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Evaluate mathematical predictions with numerical accuracy.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        tolerance: Numerical tolerance for floating point comparison
        
    Returns:
        Dictionary of evaluation metrics
    """
    def extract_number(text: str) -> Optional[float]:
        """Extract numerical value from text."""
        if isinstance(text, (int, float)):
            return float(text)
        
        # Remove common formatting
        text = str(text).replace(',', '').replace(', ''').strip()
        
        # Try direct conversion first
        try:
            return float(text)
        except ValueError:
            pass
        
        # Extract number with regex
        patterns = [
            r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)',
            r'([-+]?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    exact_matches = 0
    numerical_matches = 0
    valid_pairs = 0
    
    for pred, truth in zip(predictions, ground_truths):
        valid_pairs += 1
        
        # Exact string match
        if str(pred).strip().lower() == str(truth).strip().lower():
            exact_matches += 1
            numerical_matches += 1
            continue
        
        # Numerical comparison
        pred_num = extract_number(pred)
        truth_num = extract_number(truth)
        
        if pred_num is not None and truth_num is not None:
            if abs(pred_num - truth_num) <= tolerance:
                numerical_matches += 1
    
    return {
        'exact_accuracy': exact_matches / valid_pairs * 100 if valid_pairs > 0 else 0,
        'numerical_accuracy': numerical_matches / valid_pairs * 100 if valid_pairs > 0 else 0,
        'num_examples': valid_pairs
    }


def create_math_dataloader(dataset_name: str,
                          split: str = "train", 
                          batch_size: int = 4,
                          max_examples: int = None,
                          **kwargs) -> DataLoader:
    """Convenience function to create math DataLoader."""
    config = MathDatasetConfig(
        dataset_name=dataset_name,
        split=split,
        max_examples=max_examples,
        **kwargs
    )
    
    dataset = MathDatasetFactory.create_dataset(dataset_name, config)
    return dataset.create_dataloader(batch_size=batch_size)


def create_few_shot_math_tasks(dataset: BaseMathDataset,
                              num_tasks: int = 100,
                              shots_per_task: int = 5) -> List[Dict[str, Any]]:
    """Create few-shot math tasks for meta-learning."""
    dataset.config.shots_per_task = shots_per_task
    dataset.config.num_tasks = num_tasks
    
    return dataset.get_few_shot_episodes(num_tasks)