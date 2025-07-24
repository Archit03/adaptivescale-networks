"""
Data preprocessing module for AdaptiveScale Networks.

This module provides comprehensive preprocessing functionality for various task types
including question answering, mathematical reasoning, logical reasoning, and code generation.
"""

import logging
import re
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

# NLP preprocessing
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Transformers
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

# Math processing
import sympy
from sympy import symbols, sympify, latex

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Tokenization settings
    tokenizer_name: str = "gpt2"
    max_sequence_length: int = 512
    max_input_length: int = 400
    max_target_length: int = 128
    padding: str = "max_length"
    truncation: bool = True
    
    # Text cleaning
    clean_text: bool = True
    normalize_whitespace: bool = True
    remove_special_chars: bool = False
    lowercase: bool = False
    
    # Task-specific settings
    task_formats: Dict[str, str] = field(default_factory=lambda: {
        'qa': 'Question: {context}\nQ: {question}\nA: {answer}',
        'math': 'Problem: {question}\nSolution: {answer}',
        'reasoning': 'Question: {question}\nChoices: {choices}\nAnswer: {answer}',
        'code': 'Problem: {problem}\nCode: {code}'
    })
    
    # Mathematical preprocessing
    normalize_math_expressions: bool = True
    extract_numbers: bool = True
    convert_units: bool = True
    
    # Code preprocessing
    remove_comments: bool = True
    normalize_code_style: bool = True
    extract_imports: bool = True
    
    # Data augmentation
    enable_augmentation: bool = False
    augmentation_ratio: float = 0.2
    
    # Few-shot settings
    few_shot_template: str = "Examples:\n{examples}\n\nQuestion: {question}\nAnswer:"
    few_shot_separator: str = "\n\n"
    
    # Quality filtering
    min_length: int = 10
    max_length: int = 2048
    filter_duplicates: bool = True
    
    # Caching
    cache_preprocessed: bool = True
    cache_dir: Path = field(default_factory=lambda: Path("cache/preprocessing"))


class TextCleaner:
    """Text cleaning utilities."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
        # Initialize NLTK components
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
        
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return str(text)
        
        # Basic cleaning
        if self.config.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters if specified
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s\.\?\!,;:]', '', text)
        
        # Lowercase if specified
        if self.config.lowercase:
            text = text.lower()
        
        # Remove extra punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """Normalize numbers in text."""
        # Convert written numbers to digits
        number_words = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1000',
            'million': '1000000', 'billion': '1000000000'
        }
        
        for word, digit in number_words.items():
            text = re.sub(rf'\b{word}\b', digit, text, flags=re.IGNORECASE)
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text."""
        entities = {
            'numbers': [],
            'dates': [],
            'urls': [],
            'emails': []
        }
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        entities['numbers'] = numbers
        
        # Extract dates (simple patterns)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        entities['dates'] = dates
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        entities['urls'] = urls
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        entities['emails'] = emails
        
        return entities


class MathPreprocessor:
    """Mathematical expression preprocessing."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def normalize_math_expression(self, text: str) -> str:
        """Normalize mathematical expressions."""
        # Replace common mathematical symbols
        replacements = {
            '×': '*',
            '÷': '/',
            '−': '-',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '√': 'sqrt',
            'π': 'pi',
            '∞': 'infinity'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Normalize fractions
        text = re.sub(r'(\d+)/(\d+)', r'(\1/\2)', text)
        
        # Normalize powers
        text = re.sub(r'(\w+)\^(\w+)', r'\1**\2', text)
        
        return text
    
    def extract_mathematical_components(self, text: str) -> Dict[str, Any]:
        """Extract mathematical components from text."""
        components = {
            'equations': [],
            'expressions': [],
            'numbers': [],
            'variables': [],
            'operators': []
        }
        
        # Extract equations (containing =)
        equations = re.findall(r'[^=]*=[^=]*', text)
        components['equations'] = equations
        
        # Extract numbers
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        components['numbers'] = [float(n) for n in numbers]
        
        # Extract variables (single letters)
        variables = re.findall(r'\b[a-zA-Z]\b', text)
        components['variables'] = list(set(variables))
        
        # Extract operators
        operators = re.findall(r'[+\-*/^=<>≤≥≠]', text)
        components['operators'] = list(set(operators))
        
        return components
    
    def parse_math_problem(self, problem: str) -> Dict[str, Any]:
        """Parse a mathematical word problem."""
        parsed = {
            'problem_type': 'unknown',
            'key_numbers': [],
            'operations': [],
            'units': [],
            'question_type': 'unknown'
        }
        
        # Extract numbers with potential units
        number_pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]*)'
        matches = re.findall(number_pattern, problem)
        
        for number, unit in matches:
            parsed['key_numbers'].append(float(number))
            if unit:
                parsed['units'].append(unit)
        
        # Identify operations based on keywords
        operation_keywords = {
            'add': ['add', 'plus', 'sum', 'total', 'altogether', 'combined'],
            'subtract': ['subtract', 'minus', 'difference', 'less', 'fewer', 'remove'],
            'multiply': ['multiply', 'times', 'product', 'each', 'per'],
            'divide': ['divide', 'split', 'share', 'each', 'average', 'rate']
        }
        
        problem_lower = problem.lower()
        for operation, keywords in operation_keywords.items():
            if any(keyword in problem_lower for keyword in keywords):
                parsed['operations'].append(operation)
        
        # Identify question type
        if 'how many' in problem_lower:
            parsed['question_type'] = 'count'
        elif 'what is' in problem_lower or 'calculate' in problem_lower:
            parsed['question_type'] = 'calculation'
        elif 'find' in problem_lower:
            parsed['question_type'] = 'find'
        
        return parsed


class CodePreprocessor:
    """Code preprocessing utilities."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def clean_code(self, code: str) -> str:
        """Clean and normalize code."""
        if not isinstance(code, str):
            return str(code)
        
        # Remove comments if specified
        if self.config.remove_comments:
            # Remove single-line comments
            code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
            # Remove multi-line comments
            code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
            code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        
        # Normalize whitespace
        code = re.sub(r'\n\s*\n', '\n\n', code)  # Multiple blank lines to double
        code = re.sub(r'[ \t]+$', '', code, flags=re.MULTILINE)  # Trailing whitespace
        
        return code.strip()
    
    def extract_code_components(self, code: str) -> Dict[str, Any]:
        """Extract components from code."""
        components = {
            'imports': [],
            'functions': [],
            'classes': [],
            'variables': [],
            'docstrings': []
        }
        
        # Extract imports
        import_pattern = r'^(?:from\s+\S+\s+)?import\s+.+$'
        imports = re.findall(import_pattern, code, flags=re.MULTILINE)
        components['imports'] = imports
        
        # Extract function definitions
        func_pattern = r'^def\s+(\w+)\s*\([^)]*\):'
        functions = re.findall(func_pattern, code, flags=re.MULTILINE)
        components['functions'] = functions
        
        # Extract class definitions
        class_pattern = r'^class\s+(\w+)(?:\([^)]*\))?:'
        classes = re.findall(class_pattern, code, flags=re.MULTILINE)
        components['classes'] = classes
        
        # Extract docstrings
        docstring_pattern = r'"""(.*?)"""'
        docstrings = re.findall(docstring_pattern, code, flags=re.DOTALL)
        components['docstrings'] = docstrings
        
        return components
    
    def normalize_code_style(self, code: str) -> str:
        """Normalize code style (basic)."""
        # Add spaces around operators
        code = re.sub(r'([=+\-*/<>!])([^\s=])', r'\1 \2', code)
        code = re.sub(r'([^\s=])([=+\-*/<>!])', r'\1 \2', code)
        
        # Normalize spacing after commas
        code = re.sub(r',([^\s])', r', \1', code)
        
        # Normalize spacing in function calls
        code = re.sub(r'\(\s+', '(', code)
        code = re.sub(r'\s+\)', ')', code)
        
        return code


class TaskSpecificPreprocessor:
    """Task-specific preprocessing for different domains."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.text_cleaner = TextCleaner(config)
        self.math_preprocessor = MathPreprocessor(config)
        self.code_preprocessor = CodePreprocessor(config)
    
    def preprocess_qa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess question answering samples."""
        processed = sample.copy()
        
        # Clean context and question
        if 'context' in sample:
            processed['context'] = self.text_cleaner.clean_text(sample['context'])
        
        if 'question' in sample:
            processed['question'] = self.text_cleaner.clean_text(sample['question'])
        
        # Process answers
        if 'answer' in sample:
            if isinstance(sample['answer'], str):
                processed['answer'] = self.text_cleaner.clean_text(sample['answer'])
            elif isinstance(sample['answer'], dict):
                # Handle SQuAD-style answers
                if 'text' in sample['answer']:
                    processed['answer']['text'] = self.text_cleaner.clean_text(sample['answer']['text'])
        
        # Create formatted input
        context = processed.get('context', '')
        question = processed.get('question', '')
        answer = processed.get('answer', '')
        
        if isinstance(answer, dict):
            answer = answer.get('text', '')
        
        processed['formatted_input'] = self.config.task_formats['qa'].format(
            context=context,
            question=question,
            answer=answer
        )
        
        return processed
    
    def preprocess_math(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess mathematical reasoning samples."""
        processed = sample.copy()
        
        # Clean question
        if 'question' in sample:
            question = self.text_cleaner.clean_text(sample['question'])
            if self.config.normalize_math_expressions:
                question = self.math_preprocessor.normalize_math_expression(question)
            processed['question'] = question
            
            # Extract mathematical components
            processed['math_components'] = self.math_preprocessor.extract_mathematical_components(question)
            processed['parsed_problem'] = self.math_preprocessor.parse_math_problem(question)
        
        # Process answer
        if 'answer' in sample:
            answer = str(sample['answer']).strip()
            if self.config.normalize_math_expressions:
                answer = self.math_preprocessor.normalize_math_expression(answer)
            processed['answer'] = answer
        
        # Create formatted input
        question = processed.get('question', '')
        answer = processed.get('answer', '')
        
        processed['formatted_input'] = self.config.task_formats['math'].format(
            question=question,
            answer=answer
        )
        
        return processed
    
    def preprocess_reasoning(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess logical reasoning samples."""
        processed = sample.copy()
        
        # Clean question
        if 'question' in sample:
            processed['question'] = self.text_cleaner.clean_text(sample['question'])
        
        # Process choices
        choices_text = ""
        if 'choices' in sample:
            choices = sample['choices']
            if isinstance(choices, list):
                choices_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            elif isinstance(choices, dict):
                choices_text = "\n".join([f"{key}) {value}" for key, value in choices.items()])
            processed['choices_text'] = choices_text
        
        # Process answer
        if 'answer' in sample:
            answer = str(sample['answer']).strip()
            # Convert answer to letter format if it's a number
            if answer.isdigit():
                answer_idx = int(answer)
                if 0 <= answer_idx < 26:
                    answer = chr(65 + answer_idx)
            processed['answer'] = answer
        
        # Create formatted input
        question = processed.get('question', '')
        choices = processed.get('choices_text', '')
        answer = processed.get('answer', '')
        
        processed['formatted_input'] = self.config.task_formats['reasoning'].format(
            question=question,
            choices=choices,
            answer=answer
        )
        
        return processed
    
    def preprocess_code(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess code generation samples."""
        processed = sample.copy()
        
        # Clean problem description
        if 'problem' in sample:
            processed['problem'] = self.text_cleaner.clean_text(sample['problem'])
        
        # Process code
        if 'code' in sample:
            code = self.code_preprocessor.clean_code(sample['code'])
            if self.config.normalize_code_style:
                code = self.code_preprocessor.normalize_code_style(code)
            processed['code'] = code
            
            # Extract code components
            processed['code_components'] = self.code_preprocessor.extract_code_components(code)
        
        # Create formatted input
        problem = processed.get('problem', '')
        code = processed.get('code', '')
        
        processed['formatted_input'] = self.config.task_formats['code'].format(
            problem=problem,
            code=code
        )
        
        return processed


class DataPreprocessor:
    """Main data preprocessing class."""
    
    def __init__(self, config: PreprocessingConfig = None):
        if config is None:
            config = PreprocessingConfig()
        
        self.config = config
        
        # Initialize tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer {config.tokenizer_name}: {e}")
            self.tokenizer = None
        
        # Initialize task-specific preprocessor
        self.task_preprocessor = TaskSpecificPreprocessor(config)
        
        # Create cache directory
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DataPreprocessor with tokenizer: {config.tokenizer_name}")
    
    def preprocess_sample(self, sample: Dict[str, Any], task_type: str) -> Dict[str, Any]:
        """
        Preprocess a single sample based on task type.
        
        Args:
            sample: Input sample
            task_type: Type of task (qa, math, reasoning, code)
            
        Returns:
            Preprocessed sample
        """
        # Apply task-specific preprocessing
        if task_type == 'qa':
            processed = self.task_preprocessor.preprocess_qa(sample)
        elif task_type == 'math':
            processed = self.task_preprocessor.preprocess_math(sample)
        elif task_type == 'reasoning':
            processed = self.task_preprocessor.preprocess_reasoning(sample)
        elif task_type == 'code':
            processed = self.task_preprocessor.preprocess_code(sample)
        else:
            processed = sample.copy()
            logger.warning(f"Unknown task type: {task_type}")
        
        # Add task type to processed sample
        processed['task_type'] = task_type
        
        # Tokenize if tokenizer is available
        if self.tokenizer is not None:
            processed.update(self._tokenize_sample(processed))
        
        return processed
    
    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, BatchEncoding]:
        """Tokenize a processed sample."""
        tokenized = {}
        
        # Get the formatted input
        text_input = sample.get('formatted_input', '')
        
        # Tokenize input
        if text_input:
            encoding = self.tokenizer(
                text_input,
                max_length=self.config.max_sequence_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=None
            )
            
            tokenized.update({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask']
            })
            
            if 'token_type_ids' in encoding:
                tokenized['token_type_ids'] = encoding['token_type_ids']
        
        # Tokenize target separately if available
        if 'answer' in sample:
            answer = sample['answer']
            if isinstance(answer, dict):
                answer = answer.get('text', '')
            answer = str(answer)
            
            target_encoding = self.tokenizer(
                answer,
                max_length=self.config.max_target_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors=None
            )
            
            tokenized.update({
                'target_ids': target_encoding['input_ids'],
                'target_attention_mask': target_encoding['attention_mask']
            })
        
        return tokenized
    
    def preprocess_dataset(self, dataset: List[Dict[str, Any]], task_type: str) -> List[Dict[str, Any]]:
        """
        Preprocess an entire dataset.
        
        Args:
            dataset: List of samples
            task_type: Type of task
            
        Returns:
            List of preprocessed samples
        """
        logger.info(f"Preprocessing {len(dataset)} samples for task: {task_type}")
        
        # Check cache
        cache_key = f"{task_type}_{len(dataset)}_{hash(str(self.config))}"
        cache_path = self.cache_dir / f"{cache_key}.json"
        
        if self.config.cache_preprocessed and cache_path.exists():
            logger.info(f"Loading preprocessed data from cache: {cache_path}")
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Preprocess samples
        processed_samples = []
        for i, sample in enumerate(dataset):
            try:
                processed = self.preprocess_sample(sample, task_type)
                
                # Quality filtering
                if self._passes_quality_filter(processed):
                    processed_samples.append(processed)
                
                if (i + 1) % 1000 == 0:
                    logger.info(f"Processed {i + 1}/{len(dataset)} samples")
                    
            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue
        
        # Remove duplicates if specified
        if self.config.filter_duplicates:
            processed_samples = self._remove_duplicates(processed_samples)
        
        # Cache results
        if self.config.cache_preprocessed:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(processed_samples, f)
                logger.info(f"Cached preprocessed data to: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        logger.info(f"Preprocessing complete: {len(processed_samples)} samples")
        return processed_samples
    
    def _passes_quality_filter(self, sample: Dict[str, Any]) -> bool:
        """Check if sample passes quality filters."""
        # Check length constraints
        formatted_input = sample.get('formatted_input', '')
        if len(formatted_input) < self.config.min_length:
            return False
        if len(formatted_input) > self.config.max_length:
            return False
        
        # Check for required fields
        if 'task_type' not in sample:
            return False
        
        # Task-specific quality checks
        task_type = sample['task_type']
        
        if task_type == 'qa':
            return 'question' in sample and 'answer' in sample
        elif task_type == 'math':
            return 'question' in sample and 'answer' in sample
        elif task_type == 'reasoning':
            return 'question' in sample and 'answer' in sample
        elif task_type == 'code':
            return 'problem' in sample and 'code' in sample
        
        return True
    
    def _remove_duplicates(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate samples based on formatted input."""
        seen = set()
        unique_samples = []
        
        for sample in samples:
            formatted_input = sample.get('formatted_input', '')
            if formatted_input not in seen:
                seen.add(formatted_input)
                unique_samples.append(sample)
        
        logger.info(f"Removed {len(samples) - len(unique_samples)} duplicate samples")
        return unique_samples
    
    def create_few_shot_examples(self, samples: List[Dict[str, Any]], 
                                num_examples: int = 3) -> List[Dict[str, Any]]:
        """
        Create few-shot examples from samples.
        
        Args:
            samples: List of preprocessed samples
            num_examples: Number of examples per few-shot prompt
            
        Returns:
            List of few-shot formatted samples
        """
        if len(samples) < num_examples:
            logger.warning(f"Not enough samples ({len(samples)}) for {num_examples} examples")
            return samples
        
        few_shot_samples = []
        
        for i in range(num_examples, len(samples)):
            # Get examples and current sample
            examples = samples[i-num_examples:i]
            current_sample = samples[i].copy()
            
            # Format examples
            example_texts = []
            for example in examples:
                formatted = example.get('formatted_input', '')
                if formatted:
                    example_texts.append(formatted)
            
            examples_text = self.config.few_shot_separator.join(example_texts)
            
            # Create few-shot prompt
            question = self._extract_question_from_formatted(current_sample.get('formatted_input', ''))
            
            few_shot_input = self.config.few_shot_template.format(
                examples=examples_text,
                question=question
            )
            
            current_sample['few_shot_input'] = few_shot_input
            current_sample['original_input'] = current_sample.get('formatted_input', '')
            current_sample['formatted_input'] = few_shot_input
            
            few_shot_samples.append(current_sample)
        
        logger.info(f"Created {len(few_shot_samples)} few-shot examples")
        return few_shot_samples
    
    def _extract_question_from_formatted(self, formatted_input: str) -> str:
        """Extract question part from formatted input."""
        # Simple extraction based on common patterns
        if 'Question:' in formatted_input:
            question_part = formatted_input.split('Question:')[1]
            if 'Answer:' in question_part:
                question_part = question_part.split('Answer:')[0]
            return question_part.strip()
        elif 'Problem:' in formatted_input:
            question_part = formatted_input.split('Problem:')[1]
            if 'Solution:' in question_part or 'Code:' in question_part:
                question_part = question_part.split('Solution:')[0].split('Code:')[0]
            return question_part.strip()
        
        return formatted_input
    
    def get_preprocessing_stats(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        if not samples:
            return {}
        
        stats = {
            'total_samples': len(samples),
            'task_distribution': defaultdict(int),
            'length_stats': {
                'min': float('inf'),
                'max': 0,
                'mean': 0,
                'std': 0
            },
            'tokenization_stats': {},
            'quality_stats': {}
        }
        
        # Task distribution
        for sample in samples:
            task_type = sample.get('task_type', 'unknown')
            stats['task_distribution'][task_type] += 1
        
        # Length statistics
        lengths = []
        for sample in samples:
            formatted_input = sample.get('formatted_input', '')
            length = len(formatted_input)
            lengths.append(length)
            stats['length_stats']['min'] = min(stats['length_stats']['min'], length)
            stats['length_stats']['max'] = max(stats['length_stats']['max'], length)
        
        if lengths:
            stats['length_stats']['mean'] = np.mean(lengths)
            stats['length_stats']['std'] = np.std(lengths)
        
        # Tokenization statistics
        if self.tokenizer is not None:
            token_lengths = []
            for sample in samples:
                if 'input_ids' in sample:
                    token_length = len(sample['input_ids'])
                    token_lengths.append(token_length)
            
            if token_lengths:
                stats['tokenization_stats'] = {
                    'min_tokens': min(token_lengths),
                    'max_tokens': max(token_lengths),
                    'mean_tokens': np.mean(token_lengths),
                    'std_tokens': np.std(token_lengths),
                    'vocab_size': self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 'unknown'
                }
        
        # Quality statistics
        has_answer = sum(1 for sample in samples if 'answer' in sample)
        has_context = sum(1 for sample in samples if 'context' in sample)
        
        stats['quality_stats'] = {
            'samples_with_answer': has_answer,
            'samples_with_context': has_context,
            'answer_coverage': has_answer / len(samples) if samples else 0,
            'context_coverage': has_context / len(samples) if samples else 0
        }
        
        return dict(stats)
    
    def save_preprocessed_data(self, samples: List[Dict[str, Any]], 
                              output_path: Union[str, Path]) -> None:
        """Save preprocessed data to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_path.suffix == '.json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(samples, f, indent=2, ensure_ascii=False)
            elif output_path.suffix == '.jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            else:
                raise ValueError(f"Unsupported file format: {output_path.suffix}")
            
            logger.info(f"Saved {len(samples)} preprocessed samples to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save preprocessed data: {e}")
            raise
    
    def load_preprocessed_data(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load preprocessed data from file."""
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        try:
            if input_path.suffix == '.json':
                with open(input_path, 'r', encoding='utf-8') as f:
                    samples = json.load(f)
            elif input_path.suffix == '.jsonl':
                samples = []
                with open(input_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            else:
                raise ValueError(f"Unsupported file format: {input_path.suffix}")
            
            logger.info(f"Loaded {len(samples)} preprocessed samples from {input_path}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load preprocessed data: {e}")
            raise


class DataAugmentor:
    """Data augmentation utilities."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.text_cleaner = TextCleaner(config)
    
    def augment_qa_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment a QA sample."""
        augmented = [sample]  # Include original
        
        if not self.config.enable_augmentation:
            return augmented
        
        # Paraphrase question
        if 'question' in sample:
            paraphrased_question = self._paraphrase_text(sample['question'])
            if paraphrased_question != sample['question']:
                aug_sample = sample.copy()
                aug_sample['question'] = paraphrased_question
                aug_sample['augmentation_type'] = 'question_paraphrase'
                augmented.append(aug_sample)
        
        # Synonym replacement
        if 'context' in sample:
            context_with_synonyms = self._replace_synonyms(sample['context'])
            if context_with_synonyms != sample['context']:
                aug_sample = sample.copy()
                aug_sample['context'] = context_with_synonyms
                aug_sample['augmentation_type'] = 'synonym_replacement'
                augmented.append(aug_sample)
        
        return augmented
    
    def augment_math_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment a math sample."""
        augmented = [sample]  # Include original
        
        if not self.config.enable_augmentation:
            return augmented
        
        # Number variation
        if 'question' in sample:
            varied_question = self._vary_numbers(sample['question'])
            if varied_question != sample['question']:
                aug_sample = sample.copy()
                aug_sample['question'] = varied_question
                aug_sample['augmentation_type'] = 'number_variation'
                # Note: This would require recalculating the answer
                augmented.append(aug_sample)
        
        return augmented
    
    def augment_code_sample(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Augment a code sample."""
        augmented = [sample]  # Include original
        
        if not self.config.enable_augmentation:
            return augmented
        
        # Variable name changes
        if 'code' in sample:
            renamed_code = self._rename_variables(sample['code'])
            if renamed_code != sample['code']:
                aug_sample = sample.copy()
                aug_sample['code'] = renamed_code
                aug_sample['augmentation_type'] = 'variable_rename'
                augmented.append(aug_sample)
        
        return augmented
    
    def _paraphrase_text(self, text: str) -> str:
        """Simple paraphrasing (placeholder for more sophisticated methods)."""
        # This is a simple implementation
        # In practice, you might use models like T5 or paraphrasing APIs
        
        paraphrases = {
            'what is': 'what\'s',
            'how many': 'how much',
            'find the': 'determine the',
            'calculate': 'compute',
            'solve': 'find'
        }
        
        paraphrased = text
        for original, replacement in paraphrases.items():
            if original in text.lower():
                paraphrased = re.sub(original, replacement, text, flags=re.IGNORECASE)
                break
        
        return paraphrased
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with synonyms."""
        synonyms = {
            'big': 'large',
            'small': 'little',
            'fast': 'quick',
            'slow': 'sluggish',
            'good': 'excellent',
            'bad': 'poor'
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word.lower() in synonyms:
                words[i] = synonyms[word.lower()]
        
        return ' '.join(words)
    
    def _vary_numbers(self, text: str) -> str:
        """Vary numbers in text slightly."""
        def vary_number(match):
            num = float(match.group())
            # Vary by ±20%
            variation = np.random.uniform(-0.2, 0.2)
            new_num = num * (1 + variation)
            
            # Keep as integer if original was integer
            if match.group().find('.') == -1:
                return str(int(round(new_num)))
            else:
                return f"{new_num:.2f}"
        
        return re.sub(r'\d+(?:\.\d+)?', vary_number, text)
    
    def _rename_variables(self, code: str) -> str:
        """Rename variables in code."""
        # Simple variable renaming
        variable_map = {
            'x': 'a',
            'y': 'b',
            'i': 'idx',
            'j': 'jdx',
            'temp': 'tmp',
            'result': 'res'
        }
        
        renamed_code = code
        for old_var, new_var in variable_map.items():
            # Use word boundaries to avoid partial matches
            pattern = rf'\b{old_var}\b'
            renamed_code = re.sub(pattern, new_var, renamed_code)
        
        return renamed_code


class PreprocessingPipeline:
    """Complete preprocessing pipeline."""
    
    def __init__(self, config: PreprocessingConfig = None):
        if config is None:
            config = PreprocessingConfig()
        
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.augmentor = DataAugmentor(config) if config.enable_augmentation else None
        
        logger.info("Initialized PreprocessingPipeline")
    
    def process_dataset(self, dataset: List[Dict[str, Any]], 
                       task_type: str,
                       output_path: Optional[Union[str, Path]] = None,
                       return_stats: bool = True) -> Dict[str, Any]:
        """
        Process a complete dataset through the preprocessing pipeline.
        
        Args:
            dataset: Raw dataset
            task_type: Task type
            output_path: Optional output path to save processed data
            return_stats: Whether to return processing statistics
            
        Returns:
            Dictionary containing processed data and optional statistics
        """
        logger.info(f"Starting preprocessing pipeline for {len(dataset)} {task_type} samples")
        
        # Step 1: Basic preprocessing
        processed_samples = self.preprocessor.preprocess_dataset(dataset, task_type)
        
        # Step 2: Data augmentation
        if self.augmentor is not None:
            logger.info("Applying data augmentation")
            augmented_samples = []
            
            for sample in processed_samples:
                if task_type == 'qa':
                    aug_samples = self.augmentor.augment_qa_sample(sample)
                elif task_type == 'math':
                    aug_samples = self.augmentor.augment_math_sample(sample)
                elif task_type == 'code':
                    aug_samples = self.augmentor.augment_code_sample(sample)
                else:
                    aug_samples = [sample]
                
                augmented_samples.extend(aug_samples)
            
            processed_samples = augmented_samples
            logger.info(f"Augmentation complete: {len(processed_samples)} samples")
        
        # Step 3: Final quality check and filtering
        final_samples = []
        for sample in processed_samples:
            if self.preprocessor._passes_quality_filter(sample):
                final_samples.append(sample)
        
        logger.info(f"Final dataset size: {len(final_samples)} samples")
        
        # Step 4: Save processed data if path provided
        if output_path is not None:
            self.preprocessor.save_preprocessed_data(final_samples, output_path)
        
        # Step 5: Generate statistics
        results = {'processed_data': final_samples}
        
        if return_stats:
            stats = self.preprocessor.get_preprocessing_stats(final_samples)
            results['statistics'] = stats
            
            # Log key statistics
            logger.info(f"Processing Statistics:")
            logger.info(f"  - Total samples: {stats['total_samples']}")
            logger.info(f"  - Task distribution: {dict(stats['task_distribution'])}")
            logger.info(f"  - Average length: {stats['length_stats']['mean']:.1f} chars")
            if 'tokenization_stats' in stats:
                logger.info(f"  - Average tokens: {stats['tokenization_stats'].get('mean_tokens', 0):.1f}")
        
        return results
    
    def create_few_shot_dataset(self, dataset: List[Dict[str, Any]], 
                               num_examples: int = 3,
                               output_path: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
        """
        Create few-shot learning dataset.
        
        Args:
            dataset: Preprocessed dataset
            num_examples: Number of examples per prompt
            output_path: Optional output path
            
        Returns:
            Few-shot formatted dataset
        """
        logger.info(f"Creating few-shot dataset with {num_examples} examples per prompt")
        
        few_shot_samples = self.preprocessor.create_few_shot_examples(dataset, num_examples)
        
        if output_path is not None:
            self.preprocessor.save_preprocessed_data(few_shot_samples, output_path)
        
        return few_shot_samples
    
    def validate_preprocessing(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate preprocessed samples.
        
        Args:
            samples: Preprocessed samples
            
        Returns:
            Validation results
        """
        validation_results = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'invalid_samples': 0,
            'errors': []
        }
        
        for i, sample in enumerate(samples):
            try:
                # Check required fields
                if 'task_type' not in sample:
                    validation_results['errors'].append(f"Sample {i}: Missing task_type")
                    continue
                
                if 'formatted_input' not in sample:
                    validation_results['errors'].append(f"Sample {i}: Missing formatted_input")
                    continue
                
                # Check tokenization if available
                if self.preprocessor.tokenizer is not None:
                    if 'input_ids' not in sample:
                        validation_results['errors'].append(f"Sample {i}: Missing tokenization")
                        continue
                
                validation_results['valid_samples'] += 1
                
            except Exception as e:
                validation_results['errors'].append(f"Sample {i}: {str(e)}")
                validation_results['invalid_samples'] += 1
        
        validation_results['valid_ratio'] = (
            validation_results['valid_samples'] / validation_results['total_samples']
            if validation_results['total_samples'] > 0 else 0
        )
        
        logger.info(f"Validation complete: {validation_results['valid_samples']}/{validation_results['total_samples']} valid samples")
        
        return validation_results


# Utility functions
def load_raw_dataset(file_path: Union[str, Path], 
                    format: str = 'auto') -> List[Dict[str, Any]]:
    """
    Load raw dataset from file.
    
    Args:
        file_path: Path to dataset file
        format: File format ('json', 'jsonl', 'csv', 'auto')
        
    Returns:
        List of samples
    """
    file_path = Path(file_path)
    
    if format == 'auto':
        format = file_path.suffix[1:]  # Remove the dot
    
    try:
        if format == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and 'data' in data:
                    return data['data']
                else:
                    return [data]
        
        elif format == 'jsonl':
            samples = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
            return samples
        
        elif format == 'csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    except Exception as e:
        logger.error(f"Failed to load dataset from {file_path}: {e}")
        raise


def preprocess_huggingface_dataset(dataset_name: str, 
                                  task_type: str,
                                  config: PreprocessingConfig = None,
                                  split: str = 'train',
                                  max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Preprocess a Hugging Face dataset.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        task_type: Task type for preprocessing
        config: Preprocessing configuration
        split: Dataset split to use
        max_samples: Maximum number of samples to process
        
    Returns:
        Preprocessed samples
    """
    try:
        from datasets import load_dataset
        
        # Load dataset
        dataset = load_dataset(dataset_name, split=split)
        
        # Convert to list and limit samples if specified
        samples = list(dataset)
        if max_samples is not None:
            samples = samples[:max_samples]
        
        # Initialize preprocessing pipeline
        pipeline = PreprocessingPipeline(config)
        
        # Process dataset
        results = pipeline.process_dataset(samples, task_type)
        
        return results['processed_data']
    
    except Exception as e:
        logger.error(f"Failed to preprocess Hugging Face dataset {dataset_name}: {e}")
        raise


# Example usage and testing
if __name__ == "__main__":
    # Example preprocessing configuration
    config = PreprocessingConfig(
        tokenizer_name="gpt2",
        max_sequence_length=512,
        enable_augmentation=True,
        cache_preprocessed=True
    )
    
    # Example QA sample
    qa_sample = {
        "context": "The quick brown fox jumps over the lazy dog.",
        "question": "What does the fox do?",
        "answer": "jumps over the lazy dog"
    }
    
    # Example math sample
    math_sample = {
        "question": "If John has 5 apples and gives away 2, how many does he have left?",
        "answer": "3"
    }
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Process samples
    processed_qa = preprocessor.preprocess_sample(qa_sample, 'qa')
    processed_math = preprocessor.preprocess_sample(math_sample, 'math')
    
    print("QA Sample:")
    print(f"Original: {qa_sample}")
    print(f"Processed: {processed_qa['formatted_input']}")
    print()
    
    print("Math Sample:")
    print(f"Original: {math_sample}")
    print(f"Processed: {processed_math['formatted_input']}")
    print()
    
    # Test preprocessing pipeline
    pipeline = PreprocessingPipeline(config)
    results = pipeline.process_dataset([qa_sample, math_sample], 'qa')
    
    print("Pipeline Results:")
    print(f"Processed {len(results['processed_data'])} samples")
    if 'statistics' in results:
        print(f"Statistics: {results['statistics']}")