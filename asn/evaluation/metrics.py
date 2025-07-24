"""
Metrics module for AdaptiveScale Networks evaluation.

This module provides comprehensive metrics for evaluating different task types
including QA, mathematical reasoning, logical reasoning, and code generation.
"""

import logging
import re
import string
import ast
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path

try:
    from rouge import Rouge
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    HAS_NLTK = True
    # Download required NLTK data
    try:
        nltk.download('punkt', quiet=True)
    except:
        pass
except ImportError:
    HAS_NLTK = False

try:
    import difflib
    HAS_DIFFLIB = True
except ImportError:
    HAS_DIFFLIB = False

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Container for metric results."""
    value: float
    details: Dict[str, Any] = None
    confidence_interval: Tuple[float, float] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class MetricsCalculator:
    """Base class for metrics calculation."""
    
    def __init__(self, task_type: str = "general"):
        self.task_type = task_type
        
    def compute_all_metrics(self, predictions: List[str], 
                          ground_truths: Union[List[str], List[List[str]]], 
                          **kwargs) -> Dict[str, MetricResult]:
        """
        Compute all relevant metrics for the task type.
        
        Args:
            predictions: List of predicted outputs
            ground_truths: List of ground truth outputs (can be lists for multiple references)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of metric results
        """
        if self.task_type == "qa":
            return QAMetrics.compute_metrics(predictions, ground_truths, **kwargs)
        elif self.task_type == "math":
            return MathMetrics.compute_metrics(predictions, ground_truths, **kwargs)
        elif self.task_type == "reasoning":
            return ReasoningMetrics.compute_metrics(predictions, ground_truths, **kwargs)
        elif self.task_type == "code":
            return CodeMetrics.compute_metrics(predictions, ground_truths, **kwargs)
        else:
            # General metrics
            return {
                'exact_match': compute_exact_match(predictions, ground_truths),
                'f1_score': compute_f1_score(predictions, ground_truths)
            }


class QAMetrics:
    """Metrics for Question Answering tasks."""
    
    @staticmethod
    def normalize_answer(answer: str) -> str:
        """Normalize answer for comparison."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(answer))))
    
    @staticmethod
    def compute_exact_match(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> MetricResult:
        """Compute exact match accuracy."""
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truths):
            if isinstance(truth, list):
                # Multiple possible answers
                pred_normalized = QAMetrics.normalize_answer(pred)
                is_correct = any(pred_normalized == QAMetrics.normalize_answer(t) for t in truth)
            else:
                pred_normalized = QAMetrics.normalize_answer(pred)
                truth_normalized = QAMetrics.normalize_answer(truth)
                is_correct = pred_normalized == truth_normalized
            
            if is_correct:
                correct += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        return MetricResult(
            value=accuracy,
            details={
                'correct': correct,
                'total': total,
                'accuracy_decimal': accuracy / 100
            }
        )
    
    @staticmethod
    def compute_f1_score(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> MetricResult:
        """Compute F1 score."""
        f1_scores = []
        
        for pred, truth in zip(predictions, ground_truths):
            if isinstance(truth, list):
                # Take maximum F1 among all possible answers
                max_f1 = max(QAMetrics._single_f1_score(pred, t) for t in truth)
                f1_scores.append(max_f1)
            else:
                f1_scores.append(QAMetrics._single_f1_score(pred, truth))
        
        avg_f1 = np.mean(f1_scores) * 100
        
        return MetricResult(
            value=avg_f1,
            details={
                'individual_scores': f1_scores,
                'std_dev': np.std(f1_scores) * 100,
                'f1_decimal': avg_f1 / 100
            }
        )
    
    @staticmethod
    def _single_f1_score(prediction: str, ground_truth: str) -> float:
        """Compute F1 score between two strings."""
        pred_tokens = QAMetrics.normalize_answer(prediction).split()
        truth_tokens = QAMetrics.normalize_answer(ground_truth).split()
        
        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        common_tokens = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common_tokens.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        return f1
    
    @staticmethod
    def compute_metrics(predictions: List[str], ground_truths: List[Union[str, List[str]]], 
                       **kwargs) -> Dict[str, MetricResult]:
        """Compute all QA metrics."""
        metrics = {
            'exact_match': QAMetrics.compute_exact_match(predictions, ground_truths),
            'f1_score': QAMetrics.compute_f1_score(predictions, ground_truths)
        }
        
        # Add ROUGE scores if available
        if HAS_ROUGE:
            rouge_scores = compute_rouge_scores(predictions, ground_truths)
            metrics.update(rouge_scores)
        
        return metrics


class MathMetrics:
    """Metrics for Mathematical Reasoning tasks."""
    
    @staticmethod
    def extract_numerical_answer(text: str) -> Optional[float]:
        """Extract numerical answer from text."""
        if isinstance(text, (int, float)):
            return float(text)
        
        # Clean the text
        text = str(text).strip()
        
        # Remove common mathematical notation
        text = text.replace('$', '').replace(',', '').replace('%', '')
        
        # Try direct conversion first
        try:
            return float(text)
        except ValueError:
            pass
        
        # Extract number patterns
        patterns = [
            r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?',  # Scientific notation
            r'[-+]?\d+/\d+',  # Fractions
            r'[-+]?\d+',  # Integers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                try:
                    # Handle fractions
                    if '/' in matches[-1]:
                        num, den = matches[-1].split('/')
                        return float(num) / float(den)
                    else:
                        return float(matches[-1])
                except (ValueError, ZeroDivisionError):
                    continue
        
        return None
    
    @staticmethod
    def compute_numerical_accuracy(predictions: List[str], ground_truths: List[str], 
                                 tolerance: float = 1e-6) -> MetricResult:
        """Compute numerical accuracy with tolerance."""
        correct = 0
        valid_pairs = 0
        extraction_failures = 0
        
        for pred, truth in zip(predictions, ground_truths):
            pred_num = MathMetrics.extract_numerical_answer(pred)
            truth_num = MathMetrics.extract_numerical_answer(truth)
            
            if pred_num is None or truth_num is None:
                extraction_failures += 1
                continue
            
            valid_pairs += 1
            if abs(pred_num - truth_num) <= tolerance:
                correct += 1
        
        accuracy = correct / valid_pairs * 100 if valid_pairs > 0 else 0
        
        return MetricResult(
            value=accuracy,
            details={
                'correct': correct,
                'valid_pairs': valid_pairs,
                'extraction_failures': extraction_failures,
                'total': len(predictions),
                'tolerance': tolerance
            }
        )
    
    @staticmethod
    def compute_exact_match(predictions: List[str], ground_truths: List[str]) -> MetricResult:
        """Compute exact string match."""
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truths):
            if str(pred).strip().lower() == str(truth).strip().lower():
                correct += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        return MetricResult(
            value=accuracy,
            details={'correct': correct, 'total': total}
        )
    
    @staticmethod
    def compute_metrics(predictions: List[str], ground_truths: List[str], 
                       tolerance: float = 1e-6, **kwargs) -> Dict[str, MetricResult]:
        """Compute all math metrics."""
        return {
            'exact_match': MathMetrics.compute_exact_match(predictions, ground_truths),
            'numerical_accuracy': MathMetrics.compute_numerical_accuracy(
                predictions, ground_truths, tolerance
            )
        }


class ReasoningMetrics:
    """Metrics for Logical Reasoning tasks."""
    
    @staticmethod
    def normalize_choice_answer(answer: str, choices: List[str] = None) -> str:
        """Normalize multiple choice answer."""
        answer = str(answer).strip().upper()
        
        # Handle letter choices (A, B, C, D)
        if len(answer) == 1 and 'A' <= answer <= 'Z':
            return answer
        
        # Handle number choices (0, 1, 2, 3) -> (A, B, C, D)
        if answer.isdigit():
            num = int(answer)
            if 0 <= num <= 25:
                return chr(ord('A') + num)
        
        # Try to match against choices if provided
        if choices:
            answer_lower = answer.lower()
            for i, choice in enumerate(choices):
                if choice.lower().strip() == answer_lower:
                    return chr(ord('A') + i)
        
        return answer
    
    @staticmethod
    def compute_choice_accuracy(predictions: List[str], ground_truths: List[str], 
                              choices_list: List[List[str]] = None) -> MetricResult:
        """Compute accuracy for multiple choice questions."""
        correct = 0
        total = len(predictions)
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            choices = choices_list[i] if choices_list and i < len(choices_list) else None
            
            pred_normalized = ReasoningMetrics.normalize_choice_answer(pred, choices)
            truth_normalized = ReasoningMetrics.normalize_choice_answer(truth, choices)
            
            if pred_normalized == truth_normalized:
                correct += 1
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        return MetricResult(
            value=accuracy,
            details={'correct': correct, 'total': total}
        )
    
    @staticmethod
    def compute_partial_accuracy(predictions: List[str], ground_truths: List[str], 
                               choices_list: List[List[str]] = None) -> MetricResult:
        """Compute partial credit for reasoning tasks."""
        partial_scores = []
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truths)):
            choices = choices_list[i] if choices_list and i < len(choices_list) else None
            
            pred_normalized = ReasoningMetrics.normalize_choice_answer(pred, choices)
            truth_normalized = ReasoningMetrics.normalize_choice_answer(truth, choices)
            
            if pred_normalized == truth_normalized:
                partial_scores.append(1.0)
            elif choices and len(choices) > 1:
                # Random guess baseline
                random_prob = 1.0 / len(choices)
                partial_scores.append(random_prob)
            else:
                partial_scores.append(0.0)
        
        avg_score = np.mean(partial_scores) * 100
        
        return MetricResult(
            value=avg_score,
            details={
                'individual_scores': partial_scores,
                'perfect_matches': sum(1 for s in partial_scores if s == 1.0)
            }
        )
    
    @staticmethod
    def compute_metrics(predictions: List[str], ground_truths: List[str], 
                       choices_list: List[List[str]] = None, **kwargs) -> Dict[str, MetricResult]:
        """Compute all reasoning metrics."""
        return {
            'accuracy': ReasoningMetrics.compute_choice_accuracy(
                predictions, ground_truths, choices_list
            ),
            'partial_accuracy': ReasoningMetrics.compute_partial_accuracy(
                predictions, ground_truths, choices_list
            )
        }


class CodeMetrics:
    """Metrics for Code Generation tasks."""
    
    @staticmethod
    def check_syntax_validity(code: str, language: str = "python") -> bool:
        """Check if code has valid syntax."""
        if language.lower() == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        else:
            # For other languages, basic heuristics
            return len(code.strip()) > 0 and not code.strip().startswith("Error")
    
    @staticmethod
    def execute_code_tests(code: str, test_cases: List[str], timeout: int = 5) -> Dict[str, Any]:
        """Execute code against test cases."""
        if not test_cases:
            return {'passed': 0, 'total': 0, 'execution_error': 'No test cases provided'}
        
        try:
            # Create temporary file with code and tests
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code + '\n\n')
                for test_case in test_cases:
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
                return {
                    'passed': len(test_cases),
                    'total': len(test_cases),
                    'execution_error': None,
                    'output': result.stdout
                }
            else:
                return {
                    'passed': 0,
                    'total': len(test_cases),
                    'execution_error': result.stderr,
                    'output': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {
                'passed': 0,
                'total': len(test_cases),
                'execution_error': 'Execution timeout',
                'output': ''
            }
        except Exception as e:
            return {
                'passed': 0,
                'total': len(test_cases),
                'execution_error': str(e),
                'output': ''
            }
    
    @staticmethod
    def compute_syntax_accuracy(predictions: List[str], language: str = "python") -> MetricResult:
        """Compute syntax accuracy."""
        valid_count = 0
        total = len(predictions)
        
        for pred in predictions:
            if CodeMetrics.check_syntax_validity(pred, language):
                valid_count += 1
        
        accuracy = valid_count / total * 100 if total > 0 else 0
        
        return MetricResult(
            value=accuracy,
            details={'valid_count': valid_count, 'total': total}
        )
    
    @staticmethod
    def compute_execution_accuracy(predictions: List[str], test_cases_list: List[List[str]], 
                                 timeout: int = 5) -> MetricResult:
        """Compute execution accuracy against test cases."""
        total_passed = 0
        total_tests = 0
        execution_failures = 0
        
        for pred, test_cases in zip(predictions, test_cases_list):
            if not CodeMetrics.check_syntax_validity(pred):
                execution_failures += 1
                total_tests += len(test_cases) if test_cases else 1
                continue
            
            result = CodeMetrics.execute_code_tests(pred, test_cases, timeout)
            total_passed += result['passed']
            total_tests += result['total']
            
            if result['execution_error']:
                execution_failures += 1
        
        accuracy = total_passed / total_tests * 100 if total_tests > 0 else 0
        
        return MetricResult(
            value=accuracy,
            details={
                'total_passed': total_passed,
                'total_tests': total_tests,
                'execution_failures': execution_failures
            }
        )
    
    @staticmethod
    def compute_code_similarity(predictions: List[str], ground_truths: List[str]) -> MetricResult:
        """Compute code similarity using string similarity."""
        if not HAS_DIFFLIB:
            return MetricResult(value=0.0, details={'error': 'difflib not available'})
        
        similarities = []
        
        for pred, truth in zip(predictions, ground_truths):
            # Normalize code (remove extra whitespace, etc.)
            pred_norm = ' '.join(pred.split())
            truth_norm = ' '.join(truth.split())
            
            # Compute sequence similarity
            similarity = difflib.SequenceMatcher(None, pred_norm, truth_norm).ratio()
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) * 100
        
        return MetricResult(
            value=avg_similarity,
            details={
                'individual_similarities': similarities,
                'std_dev': np.std(similarities) * 100
            }
        )
    
    @staticmethod
    def compute_metrics(predictions: List[str], ground_truths: List[str] = None,
                       test_cases_list: List[List[str]] = None, 
                       language: str = "python", **kwargs) -> Dict[str, MetricResult]:
        """Compute all code metrics."""
        metrics = {
            'syntax_accuracy': CodeMetrics.compute_syntax_accuracy(predictions, language)
        }
        
        if test_cases_list:
            metrics['execution_accuracy'] = CodeMetrics.compute_execution_accuracy(
                predictions, test_cases_list
            )
        
        if ground_truths:
            metrics['code_similarity'] = CodeMetrics.compute_code_similarity(
                predictions, ground_truths
            )
        
        return metrics


class CompressionMetrics:
    """Metrics for model compression evaluation."""
    
    @staticmethod
    def compute_compression_ratio(original_params: int, compressed_params: int) -> MetricResult:
        """Compute compression ratio."""
        ratio = compressed_params / original_params if original_params > 0 else 1.0
        compression_factor = 1.0 / ratio if ratio > 0 else float('inf')
        
        return MetricResult(
            value=ratio,
            details={
                'original_params': original_params,
                'compressed_params': compressed_params,
                'compression_factor': compression_factor,
                'parameter_reduction': original_params - compressed_params,
                'reduction_percentage': (1 - ratio) * 100
            }
        )
    
    @staticmethod
    def compute_speedup(original_time: float, compressed_time: float) -> MetricResult:
        """Compute inference speedup."""
        speedup = original_time / compressed_time if compressed_time > 0 else float('inf')
        
        return MetricResult(
            value=speedup,
            details={
                'original_time': original_time,
                'compressed_time': compressed_time,
                'time_reduction': original_time - compressed_time,
                'efficiency_gain': (1 - compressed_time / original_time) * 100 if original_time > 0 else 0
            }
        )
    
    @staticmethod
    def compute_metrics(original_params: int, compressed_params: int, 
                       original_time: float = None, compressed_time: float = None,
                       **kwargs) -> Dict[str, MetricResult]:
        """Compute all compression metrics."""
        metrics = {
            'compression_ratio': CompressionMetrics.compute_compression_ratio(
                original_params, compressed_params
            )
        }
        
        if original_time is not None and compressed_time is not None:
            metrics['speedup'] = CompressionMetrics.compute_speedup(
                original_time, compressed_time
            )
        
        return metrics


# Utility functions
def compute_exact_match(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> MetricResult:
    """Compute exact match accuracy."""
    return QAMetrics.compute_exact_match(predictions, ground_truths)


def compute_f1_score(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> MetricResult:
    """Compute F1 score."""
    return QAMetrics.compute_f1_score(predictions, ground_truths)


def compute_rouge_scores(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> Dict[str, MetricResult]:
    """Compute ROUGE scores."""
    if not HAS_ROUGE:
        return {
            'rouge_1': MetricResult(0.0, {'error': 'ROUGE not available'}),
            'rouge_2': MetricResult(0.0, {'error': 'ROUGE not available'}),
            'rouge_l': MetricResult(0.0, {'error': 'ROUGE not available'})
        }
    
    rouge = Rouge()
    
    # Prepare references (ROUGE expects single strings)
    processed_preds = []
    processed_refs = []
    
    for pred, truth in zip(predictions, ground_truths):
        if isinstance(truth, list):
            # Use first reference for ROUGE
            truth = truth[0] if truth else ""
        
        # Ensure non-empty strings
        pred = pred if pred.strip() else "empty"
        truth = truth if truth.strip() else "empty"
        
        processed_preds.append(pred)
        processed_refs.append(truth)
    
    try:
        scores = rouge.get_scores(processed_preds, processed_refs, avg=True)
        
        return {
            'rouge_1': MetricResult(
                value=scores['rouge-1']['f'] * 100,
                details=scores['rouge-1']
            ),
            'rouge_2': MetricResult(
                value=scores['rouge-2']['f'] * 100,
                details=scores['rouge-2']
            ),
            'rouge_l': MetricResult(
                value=scores['rouge-l']['f'] * 100,
                details=scores['rouge-l']
            )
        }
    except Exception as e:
        error_result = MetricResult(0.0, {'error': str(e)})
        return {
            'rouge_1': error_result,
            'rouge_2': error_result,
            'rouge_l': error_result
        }


def compute_bleu_score(predictions: List[str], ground_truths: List[Union[str, List[str]]]) -> MetricResult:
    """Compute BLEU score."""
    if not HAS_NLTK:
        return MetricResult(0.0, {'error': 'NLTK not available'})
    
    smoothing_function = SmoothingFunction().method1
    bleu_scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        if isinstance(truth, list):
            references = [ref.split() for ref in truth]
        else:
            references = [truth.split()]
        
        hypothesis = pred.split()
        
        if not hypothesis or not any(references):
            bleu_scores.append(0.0)
            continue
        
        try:
            score = sentence_bleu(references, hypothesis, smoothing_function=smoothing_function)
            bleu_scores.append(score)
        except:
            bleu_scores.append(0.0)
    
    avg_bleu = np.mean(bleu_scores) * 100
    
    return MetricResult(
        value=avg_bleu,
        details={
            'individual_scores': bleu_scores,
            'std_dev': np.std(bleu_scores) * 100
        }
    )


def compute_code_similarity(predictions: List[str], ground_truths: List[str]) -> MetricResult:
    """Compute code similarity."""
    return CodeMetrics.compute_code_similarity(predictions, ground_truths)


def bootstrap_confidence_interval(scores: List[float], confidence: float = 0.95, 
                                num_samples: int = 1000) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    if not scores:
        return (0.0, 0.0)
    
    np.random.seed(42)  # For reproducibility
    bootstrap_scores = []
    
    for _ in range(num_samples):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_scores.append(np.mean(sample))
    
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_scores, lower_percentile)
    upper_bound = np.percentile(bootstrap_scores, upper_percentile)
    
    return (lower_bound, upper_bound)


def statistical_significance_test(scores1: List[float], scores2: List[float], 
                                alpha: float = 0.05) -> Dict[str, Any]:
    """Perform statistical significance test between two sets of scores."""
    try:
        from scipy import stats
        
        # Perform paired t-test if same length, independent t-test otherwise
        if len(scores1) == len(scores2):
            statistic, p_value = stats.ttest_rel(scores1, scores2)
            test_type = "paired_ttest"
        else:
            statistic, p_value = stats.ttest_ind(scores1, scores2)
            test_type = "independent_ttest"
        
        return {
            'test_type': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha,
            'mean_diff': np.mean(scores1) - np.mean(scores2),
            'effect_size': (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
        }
    except ImportError:
        return {'error': 'scipy not available for statistical tests'}
    except Exception as e:
        return {'error': str(e)}