"""
Code generation and understanding datasets for AdaptiveScale Networks.

This module provides implementations for various code-related datasets
including HumanEval, MBPP, CodeContests, and others. It handles code
generation, completion, and understanding tasks.
"""

import ast
import json
import logging
import random
import re
import subprocess
import tempfile
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
class CodeExample:
    """Single code example with metadata."""
    
    id: str
    prompt: str
    canonical_solution: str
    test_cases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    entry_point: Optional[str] = None
    difficulty: str = "medium"
    language: str = "python"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'id': self.id,
            'prompt': self.prompt,
            'canonical_solution': self.canonical_solution,
            'test_cases': self.test_cases,
            'description': self.description,
            'entry_point': self.entry_point,
            'difficulty': self.difficulty,
            'language': self.language,
            'metadata': self.metadata
        }
    
    def validate_solution(self, solution: str, timeout: int = 5) -> Dict[str, Any]:
        """
        Validate a solution against test cases.
        
        Args:
            solution: Code solution to validate
            timeout: Timeout in seconds for execution
            
        Returns:
            Dictionary with validation results
        """
        if not self.test_cases:
            return {'valid': None, 'error': 'No test cases available'}
        
        try:
            # Create temporary file with solution and test cases
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write solution
                f.write(solution + '\n\n')
                
                # Write test cases
                for test_case in self.test_cases:
                    f.write(test_case + '\n')
                
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Clean up
            Path(temp_file).unlink()
            
            if result.returncode == 0:
                return {'valid': True, 'error': None, 'output': result.stdout}
            else:
                return {'valid': False, 'error': result.stderr, 'output': result.stdout}
                
        except subprocess.TimeoutExpired:
            return {'valid': False, 'error': 'Execution timeout', 'output': ''}
        except Exception as e:
            return {'valid': False, 'error': str(e), 'output': ''}
    
    def extract_function_signature(self) -> Optional[str]:
        """Extract function signature from canonical solution."""
        try:
            tree = ast.parse(self.canonical_solution)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Reconstruct function signature
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)
                    return f"def {node.name}({', '.join(args)}):"
        except:
            pass
        
        # Fallback: regex extraction
        match = re.search(r'def\s+(\w+)\s*\([^)]*\):', self.canonical_solution)
        return match.group(0) if match else None


@dataclass
class CodeDatasetConfig:
    """Configuration for code datasets."""
    
    # Dataset settings
    dataset_name: str = "humaneval"
    split: str = "test"  # Many code datasets only have test split
    max_examples: Optional[int] = None
    
    # Tokenization settings
    tokenizer_name: str = "gpt2"
    max_seq_length: int = 1024
    max_solution_length: int = 512
    
    # Code processing settings
    include_docstrings: bool = True
    include_comments: bool = True
    format_code: bool = True
    validate_syntax: bool = True
    
    # Few-shot settings
    shots_per_task: int = 3
    query_shots_per_task: int = 5
    num_tasks: int = 50
    
    # Execution settings
    allow_execution: bool = False
    execution_timeout: int = 5
    
    # Filtering
    min_solution_length: int = 10
    max_solution_length_filter: int = 2000
    
    # Language settings
    languages: List[str] = field(default_factory=lambda: ["python"])
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


class BaseCodeDataset(Dataset):
    """Base class for code datasets."""
    
    def __init__(self, config: CodeDatasetConfig):
        self.config = config
        self.examples = []
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if HAS_TRANSFORMERS:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Add special tokens for code
            special_tokens = ["<code>", "</code>", "<test>", "</test>", "<docstring>", "</docstring>"]
            self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
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
    
    def _tokenize_example(self, example: CodeExample) -> Dict[str, torch.Tensor]:
        """Tokenize a code example."""
        # Format prompt for code generation
        prompt_text = self._format_prompt(example)
        
        # Format full solution
        full_text = f"{prompt_text}\n{example.canonical_solution}"
        
        # Tokenize full text
        inputs = self.tokenizer(
            full_text,
            max_length=self.config.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize just the prompt for generation
        prompt_inputs = self.tokenizer(
            prompt_text,
            max_length=self.config.max_seq_length // 2,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize solution for training
        solution_inputs = self.tokenizer(
            example.canonical_solution,
            max_length=self.config.max_solution_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Add components
        inputs['prompt_input_ids'] = prompt_inputs['input_ids']
        inputs['prompt_attention_mask'] = prompt_inputs['attention_mask']
        inputs['solution_input_ids'] = solution_inputs['input_ids']
        inputs['solution_attention_mask'] = solution_inputs['attention_mask']
        
        # Add metadata
        inputs['example_id'] = example.id
        inputs['prompt_text'] = prompt_text
        inputs['solution_text'] = example.canonical_solution
        inputs['entry_point'] = example.entry_point or ""
        inputs['difficulty'] = example.difficulty
        inputs['language'] = example.language
        
        # Function signature extraction
        signature = example.extract_function_signature()
        inputs['function_signature'] = signature or ""
        
        # Squeeze batch dimension
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor) and inputs[key].dim() > 1:
                inputs[key] = inputs[key].squeeze(0)
        
        return inputs
    
    def _format_prompt(self, example: CodeExample) -> str:
        """Format the prompt for code generation."""
        prompt_parts = []
        
        # Add description if available
        if example.description and self.config.include_docstrings:
            prompt_parts.append(f'"""\n{example.description}\n"""')
        
        # Add the main prompt
        prompt_parts.append(example.prompt)
        
        # Join and format
        formatted_prompt = "\n".join(prompt_parts)
        
        if self.config.format_code:
            formatted_prompt = self._format_code_block(formatted_prompt)
        
        return formatted_prompt
    
    def _format_code_block(self, code: str) -> str:
        """Format code block with proper indentation."""
        lines = code.split('\n')
        formatted_lines = []
        
        for line in lines:
            # Basic formatting - can be enhanced
            stripped = line.strip()
            if stripped:
                # Preserve some indentation for Python
                if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except', 'finally:')):
                    formatted_lines.append(stripped)
                elif any(line.startswith('    ' + keyword) for keyword in ['return', 'print', 'pass']):
                    formatted_lines.append(f"    {stripped}")
                else:
                    formatted_lines.append(stripped)
            else:
                formatted_lines.append('')
        
        return '\n'.join(formatted_lines)
    
    def _filter_example(self, example: CodeExample) -> bool:
        """Filter examples based on quality criteria."""
        # Check solution length
        solution_len = len(example.canonical_solution)
        if solution_len < self.config.min_solution_length or solution_len > self.config.max_solution_length_filter:
            return False
        
        # Check language
        if self.config.languages and example.language not in self.config.languages:
            return False
        
        # Validate syntax if required
        if self.config.validate_syntax and example.language == "python":
            try:
                ast.parse(example.canonical_solution)
            except SyntaxError:
                return False
        
        return True
    
    def get_few_shot_episodes(self, num_episodes: int = None) -> List[Dict[str, List[CodeExample]]]:
        """Generate few-shot learning episodes for code tasks."""
        if num_episodes is None:
            num_episodes = self.config.num_tasks
        
        episodes = []
        
        # Group by difficulty for coherent episodes
        difficulty_groups = defaultdict(list)
        for example in self.examples:
            difficulty_groups[example.difficulty].append(example)
        
        for _ in range(num_episodes):
            # Choose a difficulty level
            if difficulty_groups:
                difficulty = random.choice(list(difficulty_groups.keys()))
                available_examples = list(difficulty_groups[difficulty])
            else:
                available_examples = list(self.examples)
            
            random.shuffle(available_examples)
            
            # Split into support and query
            support_size = self.config.shots_per_task
            query_size = self.config.query_shots_per_task
            total_needed = support_size + query_size
            
            if len(available_examples) < total_needed:
                # Fall back to all examples
                available_examples = list(self.examples)
                random.shuffle(available_examples)
                
                if len(available_examples) < total_needed:
                    continue
            
            support_examples = available_examples[:support_size]
            query_examples = available_examples[support_size:support_size + query_size]
            
            episodes.append({
                'support': support_examples,
                'query': query_examples,
                'difficulty': difficulty if difficulty_groups else 'mixed'
            })
        
        return episodes
    
    def create_dataloader(self, batch_size: int = 2, shuffle: bool = True, **kwargs) -> DataLoader:
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


class HumanEvalDataset(BaseCodeDataset):
    """HumanEval dataset implementation."""
    
    def __init__(self, config: CodeDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load HumanEval data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for HumanEval loading")
        
        logger.info("Loading HumanEval dataset...")
        
        try:
            dataset = load_dataset("openai_humaneval", split="test", cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to CodeExample format
            for item in dataset:
                # Extract test cases from test string
                test_cases = self._extract_test_cases(item['test'])
                
                example = CodeExample(
                    id=item['task_id'],
                    prompt=item['prompt'],
                    canonical_solution=item['canonical_solution'],
                    test_cases=test_cases,
                    entry_point=item['entry_point'],
                    difficulty=self._classify_difficulty(item['prompt']),
                    language="python",
                    metadata={'source': 'humaneval'}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} HumanEval examples")
            
        except Exception as e:
            logger.error(f"Failed to load HumanEval dataset: {e}")
            self.examples = []
    
    def _extract_test_cases(self, test_string: str) -> List[str]:
        """Extract individual test cases from test string."""
        test_cases = []
        
        # Split by assert statements
        assertions = re.findall(r'assert .+', test_string)
        test_cases.extend(assertions)
        
        # Also include any other test patterns
        checks = re.findall(r'check\(.+\)', test_string)
        test_cases.extend(checks)
        
        return test_cases
    
    def _classify_difficulty(self, prompt: str) -> str:
        """Classify problem difficulty based on prompt content."""
        prompt_lower = prompt.lower()
        
        # Simple heuristics for difficulty classification
        if any(keyword in prompt_lower for keyword in ['sort', 'simple', 'basic', 'count']):
            return "easy"
        elif any(keyword in prompt_lower for keyword in ['algorithm', 'optimize', 'complex', 'recursive']):
            return "hard"
        else:
            return "medium"


class MBPPDataset(BaseCodeDataset):
    """Mostly Basic Python Problems (MBPP) dataset implementation."""
    
    def __init__(self, config: CodeDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load MBPP data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for MBPP loading")
        
        logger.info("Loading MBPP dataset...")
        
        try:
            dataset = load_dataset("mbpp", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to CodeExample format
            for item in dataset:
                # MBPP has test_list with test cases
                test_cases = item.get('test_list', [])
                
                example = CodeExample(
                    id=f"mbpp_{item['task_id']}",
                    prompt=item['text'],
                    canonical_solution=item['code'],
                    test_cases=test_cases,
                    description=item['text'],  # MBPP uses text as description
                    difficulty=self._classify_mbpp_difficulty(item['text']),
                    language="python",
                    metadata={'source': 'mbpp', 'task_id': item['task_id']}
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} MBPP examples")
            
        except Exception as e:
            logger.error(f"Failed to load MBPP dataset: {e}")
            self.examples = []
    
    def _classify_mbpp_difficulty(self, text: str) -> str:
        """Classify MBPP problem difficulty."""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ['write a function', 'simple', 'basic']):
            return "easy"
        elif any(keyword in text_lower for keyword in ['algorithm', 'efficient', 'complex']):
            return "hard"
        else:
            return "medium"


class CodeContestsDataset(BaseCodeDataset):
    """CodeContests dataset implementation."""
    
    def __init__(self, config: CodeDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load CodeContests data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for CodeContests loading")
        
        logger.info("Loading CodeContests dataset...")
        
        try:
            dataset = load_dataset("deepmind/code_contests", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to CodeExample format
            for i, item in enumerate(dataset):
                # CodeContests has multiple solutions, use the first Python one
                python_solutions = [sol for sol in item['solutions']['solution'] 
                                  if sol and 'python' in item['solutions']['language'][item['solutions']['solution'].index(sol)].lower()]
                
                if not python_solutions:
                    continue
                
                canonical_solution = python_solutions[0]
                
                # Extract test cases from public and private tests
                test_cases = []
                if item.get('public_tests'):
                    for test_input, test_output in zip(item['public_tests']['input'], item['public_tests']['output']):
                        # Format as assertion (simplified)
                        test_cases.append(f"# Test: input={test_input}, expected_output={test_output}")
                
                example = CodeExample(
                    id=f"codecontests_{i}",
                    prompt=item['description'],
                    canonical_solution=canonical_solution,
                    test_cases=test_cases,
                    description=item['description'],
                    difficulty=item.get('difficulty', 'medium'),
                    language="python",
                    metadata={
                        'source': 'codecontests',
                        'contest_id': item.get('contest_id'),
                        'problem_id': item.get('problem_id')
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} CodeContests examples")
            
        except Exception as e:
            logger.error(f"Failed to load CodeContests dataset: {e}")
            self.examples = []


class APPSDataset(BaseCodeDataset):
    """APPS (Automated Programming Progress Standard) dataset implementation."""
    
    def __init__(self, config: CodeDatasetConfig):
        super().__init__(config)
        self.load_data()
    
    def load_data(self):
        """Load APPS data."""
        if not HAS_DATASETS:
            raise ImportError("datasets package required for APPS loading")
        
        logger.info("Loading APPS dataset...")
        
        try:
            dataset = load_dataset("codeparrot/apps", split=self.config.split, cache_dir=self.config.cache_dir)
            
            if self.config.max_examples:
                dataset = dataset.select(range(min(self.config.max_examples, len(dataset))))
            
            # Convert to CodeExample format
            for item in dataset:
                # APPS has multiple solutions, use the first one
                solutions = item.get('solutions', [])
                if not solutions or not solutions[0]:
                    continue
                
                canonical_solution = solutions[0]
                
                # Extract test cases from input/output
                test_cases = []
                if item.get('input_output'):
                    try:
                        io_data = json.loads(item['input_output'])
                        inputs = io_data.get('inputs', [])
                        outputs = io_data.get('outputs', [])
                        
                        for inp, out in zip(inputs, outputs):
                            test_cases.append(f"# Test: input={inp}, expected_output={out}")
                    except:
                        pass
                
                example = CodeExample(
                    id=f"apps_{item.get('problem_id', len(self.examples))}",
                    prompt=item['question'],
                    canonical_solution=canonical_solution,
                    test_cases=test_cases,
                    description=item['question'],
                    difficulty=item.get('difficulty', 'medium'),
                    language="python",
                    metadata={
                        'source': 'apps',
                        'problem_id': item.get('problem_id'),
                        'contest': item.get('contest')
                    }
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} APPS examples")
            
        except Exception as e:
            logger.error(f"Failed to load APPS dataset: {e}")
            self.examples = []


class CustomCodeDataset(BaseCodeDataset):
    """Custom code dataset for loading from local files."""
    
    def __init__(self, config: CodeDatasetConfig, data_path: str):
        super().__init__(config)
        self.data_path = Path(data_path)
        self.load_data()
    
    def load_data(self):
        """Load custom code data from file."""
        logger.info(f"Loading custom code data from {self.data_path}")
        
        try:
            if self.data_path.suffix == '.json':
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
            
            # Expect data in format: List[Dict] with keys: id, prompt, solution, test_cases
            for item in data:
                example = CodeExample(
                    id=item.get('id', f"custom_{len(self.examples)}"),
                    prompt=item['prompt'],
                    canonical_solution=item['solution'],
                    test_cases=item.get('test_cases', []),
                    description=item.get('description'),
                    entry_point=item.get('entry_point'),
                    difficulty=item.get('difficulty', 'medium'),
                    language=item.get('language', 'python'),
                    metadata=item.get('metadata', {})
                )
                
                if self._filter_example(example):
                    self.examples.append(example)
            
            logger.info(f"Loaded {len(self.examples)} custom code examples")
            
        except Exception as e:
            logger.error(f"Failed to load custom code data: {e}")
            self.examples = []


class CodeDatasetFactory:
    """Factory for creating code datasets."""
    
    DATASETS = {
        'humaneval': HumanEvalDataset,
        'mbpp': MBPPDataset,
        'codecontests': CodeContestsDataset,
        'apps': APPSDataset,
        'custom': CustomCodeDataset
    }
    
    @classmethod
    def create_dataset(cls, dataset_name: str, config: CodeDatasetConfig = None, **kwargs) -> BaseCodeDataset:
        """Create a code dataset."""
        if config is None:
            config = CodeDatasetConfig(dataset_name=dataset_name)
        
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
        """List all available code datasets."""
        return list(cls.DATASETS.keys())


# Utility functions
def evaluate_code_predictions(predictions: List[str],
                            examples: List[CodeExample],
                            timeout: int = 5) -> Dict[str, float]:
    """
    Evaluate code predictions by executing them against test cases.
    
    Args:
        predictions: List of predicted code solutions
        examples: List of corresponding CodeExample objects
        timeout: Execution timeout in seconds
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_examples = len(predictions)
    syntax_correct = 0
    execution_correct = 0
    test_passed = 0
    
    for pred, example in zip(predictions, examples):
        # Check syntax
        try:
            ast.parse(pred)
            syntax_correct += 1
        except SyntaxError:
            continue
        
        # Try execution
        try:
            validation_result = example.validate_solution(pred, timeout)
            if validation_result['valid'] is not None:
                execution_correct += 1
                if validation_result['valid']:
                    test_passed += 1
        except Exception:
            pass
    
    return {
        'syntax_accuracy': syntax_correct / total_examples * 100 if total_examples > 0 else 0,
        'execution_accuracy': execution_correct / total_examples * 100 if total_examples > 0 else 0,
        'test_pass_rate': test_passed / total_examples * 100 if total_examples > 0 else 0,
        'total_examples': total_examples
    }


def create_code_dataloader(dataset_name: str,
                          split: str = "test",
                          batch_size: int = 2,
                          max_examples: int = None,
                          **kwargs) -> DataLoader:
    """Convenience function to create code DataLoader."""
    config = CodeDatasetConfig(
        dataset_name=dataset_name,
        split=split,
        max_examples=max_examples,
        **kwargs
    )
    
    dataset = CodeDatasetFactory.create_dataset(dataset_name, config)
    return dataset.create_dataloader(batch_size=batch_size)


def create_few_shot_code_tasks(dataset: BaseCodeDataset,
                              num_tasks: int = 50,
                              shots_per_task: int = 3) -> List[Dict[str, Any]]:
    """Create few-shot code tasks for meta-learning."""
    dataset.config.shots_per_task = shots_per_task
    dataset.config.num_tasks = num_tasks
    
    return dataset.get_few_shot_episodes(num_tasks)


def extract_code_metrics(code: str) -> Dict[str, Any]:
    """
    Extract various metrics from code.
    
    Args:
        code: Source code string
        
    Returns:
        Dictionary of code metrics
    """
    try:
        tree = ast.parse(code)
        
        # Count different node types
        function_count = 0
        class_count = 0
        loop_count = 0
        condition_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
            elif isinstance(node, ast.ClassDef):
                class_count += 1
            elif isinstance(node, (ast.For, ast.While)):
                loop_count += 1
            elif isinstance(node, ast.If):
                condition_count += 1
        
        # Basic metrics
        lines = code.split('\n')
        loc = len([line for line in lines if line.strip()])
        blank_lines = len([line for line in lines if not line.strip()])
        
        return {
            'lines_of_code': loc,
            'blank_lines': blank_lines,
            'total_lines': len(lines),
            'function_count': function_count,
            'class_count': class_count,
            'loop_count': loop_count,
            'condition_count': condition_count,
            'complexity_score': function_count + loop_count + condition_count
        }
        
    except SyntaxError:
        return {
            'lines_of_code': len(code.split('\n')),
            'syntax_error': True
        }