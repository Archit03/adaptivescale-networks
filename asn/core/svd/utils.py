"""
SVD Utilities for AdaptiveScale Networks.

This module provides utility functions for SVD operations, compression analysis,
and efficient matrix operations used throughout the SVD decomposition pipeline.
"""

import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def analyze_singular_values(matrix: torch.Tensor, plot: bool = False) -> Dict[str, Any]:
    """
    Analyze singular value distribution of a matrix.
    
    Args:
        matrix: Input matrix to analyze
        plot: Whether to create visualization plots
        
    Returns:
        Dictionary containing singular value analysis
    """
    # Perform SVD
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    
    # Basic statistics
    total_energy = (S ** 2).sum().item()
    normalized_sv = S / S[0]  # Normalize by largest singular value
    
    # Compute cumulative energy
    cumulative_energy = torch.cumsum(S ** 2, dim=0)
    energy_ratios = cumulative_energy / total_energy
    
    # Find effective rank (95% energy)
    effective_rank_95 = torch.searchsorted(energy_ratios, 0.95).item() + 1
    effective_rank_99 = torch.searchsorted(energy_ratios, 0.99).item() + 1
    
    # Compute decay rate (fit exponential)
    log_sv = torch.log(S + 1e-8)
    ranks = torch.arange(1, len(S) + 1, dtype=torch.float32)
    
    # Linear regression for decay rate
    n = len(log_sv)
    if n > 1:
        sum_x = ranks.sum()
        sum_y = log_sv.sum()
        sum_xy = (ranks * log_sv).sum()
        sum_x2 = (ranks ** 2).sum()
        
        decay_rate = -(n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        decay_rate = decay_rate.item()
    else:
        decay_rate = 0.0
    
    # Condition number
    condition_number = (S[0] / S[-1]).item() if len(S) > 0 and S[-1] > 1e-10 else float('inf')
    
    # Rank estimation based on numerical rank
    numerical_rank = (S > S[0] * 1e-12).sum().item()
    
    analysis = {
        'singular_values': S.cpu().numpy(),
        'normalized_singular_values': normalized_sv.cpu().numpy(),
        'energy_ratios': energy_ratios.cpu().numpy(),
        'total_energy': total_energy,
        'effective_rank_95': effective_rank_95,
        'effective_rank_99': effective_rank_99,
        'numerical_rank': numerical_rank,
        'condition_number': condition_number,
        'decay_rate': decay_rate,
        'matrix_shape': matrix.shape,
        'max_possible_rank': min(matrix.shape)
    }
    
    # Create plots if requested
    if plot:
        analysis['plots'] = _create_singular_value_plots(S, energy_ratios)
    
    return analysis


def _create_singular_value_plots(singular_values: torch.Tensor, 
                                energy_ratios: torch.Tensor) -> Dict[str, Any]:
    """Create visualization plots for singular value analysis."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Singular values
        axes[0, 0].semilogy(singular_values.cpu().numpy())
        axes[0, 0].set_title('Singular Values')
        axes[0, 0].set_xlabel('Index')
        axes[0, 0].set_ylabel('Singular Value')
        axes[0, 0].grid(True)
        
        # Plot 2: Normalized singular values
        normalized_sv = singular_values / singular_values[0]
        axes[0, 1].semilogy(normalized_sv.cpu().numpy())
        axes[0, 1].set_title('Normalized Singular Values')
        axes[0, 1].set_xlabel('Index')
        axes[0, 1].set_ylabel('Normalized Value')
        axes[0, 1].grid(True)
        
        # Plot 3: Cumulative energy
        axes[1, 0].plot(energy_ratios.cpu().numpy())
        axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95% energy')
        axes[1, 0].axhline(y=0.99, color='orange', linestyle='--', label='99% energy')
        axes[1, 0].set_title('Cumulative Energy')
        axes[1, 0].set_xlabel('Rank')
        axes[1, 0].set_ylabel('Energy Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Energy per singular value
        energy_per_sv = (singular_values ** 2).cpu().numpy()
        axes[1, 1].bar(range(len(energy_per_sv)), energy_per_sv)
        axes[1, 1].set_title('Energy per Singular Value')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Energy')
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        return {'figure': fig, 'axes': axes}
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")
        return {}


def compute_compression_metrics(original_shape: Tuple[int, int], rank: int, 
                               has_bias: bool = False) -> Dict[str, float]:
    """
    Compute compression metrics for SVD decomposition.
    
    Args:
        original_shape: Shape of original matrix (m, n)
        rank: SVD rank
        has_bias: Whether layer has bias
        
    Returns:
        Dictionary of compression metrics
    """
    m, n = original_shape
    
    # Original parameters
    original_params = m * n
    if has_bias:
        original_params += m  # Add bias parameters
    
    # Compressed parameters (U: m×r, S: r, Vt: r×n)
    compressed_params = m * rank + rank + rank * n
    if has_bias:
        compressed_params += m  # Bias is not compressed
    
    # Metrics
    compression_ratio = compressed_params / original_params
    compression_factor = 1.0 / compression_ratio
    parameter_savings = original_params - compressed_params
    savings_percentage = (parameter_savings / original_params) * 100
    
    # Memory savings (assuming float32)
    memory_savings_mb = parameter_savings * 4 / (1024 * 1024)
    
    # Rank efficiency
    max_rank = min(m, n)
    rank_efficiency = rank / max_rank
    
    return {
        'original_params': original_params,
        'compressed_params': compressed_params,
        'compression_ratio': compression_ratio,
        'compression_factor': compression_factor,
        'parameter_savings': parameter_savings,
        'savings_percentage': savings_percentage,
        'memory_savings_mb': memory_savings_mb,
        'rank': rank,
        'max_rank': max_rank,
        'rank_efficiency': rank_efficiency
    }


def find_optimal_rank(matrix: torch.Tensor, target_compression: float = 0.5,
                     energy_threshold: float = 0.95, 
                     method: str = 'compression') -> Dict[str, Any]:
    """
    Find optimal rank for SVD decomposition based on different criteria.
    
    Args:
        matrix: Input matrix
        target_compression: Target compression ratio
        energy_threshold: Energy preservation threshold
        method: Selection method ('compression', 'energy', 'elbow', 'hybrid')
        
    Returns:
        Dictionary with optimal rank and analysis
    """
    # Perform SVD
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    m, n = matrix.shape
    max_rank = min(m, n)
    
    if method == 'compression':
        # Find rank that achieves target compression
        for rank in range(1, max_rank + 1):
            metrics = compute_compression_metrics((m, n), rank)
            if metrics['compression_ratio'] <= target_compression:
                optimal_rank = rank
                break
        else:
            optimal_rank = max_rank
            
    elif method == 'energy':
        # Find rank that preserves target energy
        total_energy = (S ** 2).sum()
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        energy_ratios = cumulative_energy / total_energy
        
        optimal_rank = torch.searchsorted(energy_ratios, energy_threshold).item() + 1
        optimal_rank = min(optimal_rank, max_rank)
        
    elif method == 'elbow':
        # Use elbow method on singular value curve
        optimal_rank = _find_elbow_point(S)
        
    elif method == 'hybrid':
        # Combine energy and compression constraints
        energy_rank = find_optimal_rank(matrix, energy_threshold=energy_threshold, 
                                      method='energy')['optimal_rank']
        compression_rank = find_optimal_rank(matrix, target_compression=target_compression,
                                           method='compression')['optimal_rank']
        
        # Take the more conservative (smaller) rank
        optimal_rank = min(energy_rank, compression_rank)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Compute metrics for optimal rank
    metrics = compute_compression_metrics((m, n), optimal_rank)
    
    # Compute reconstruction error
    U_opt = U[:, :optimal_rank]
    S_opt = S[:optimal_rank]
    Vt_opt = Vt[:optimal_rank, :]
    
    reconstructed = torch.mm(U_opt * S_opt.unsqueeze(0), Vt_opt)
    reconstruction_error = torch.norm(matrix - reconstructed) / torch.norm(matrix)
    
    return {
        'optimal_rank': optimal_rank,
        'method': method,
        'reconstruction_error': reconstruction_error.item(),
        'compression_metrics': metrics,
        'singular_values': S[:optimal_rank].cpu().numpy()
    }


def _find_elbow_point(singular_values: torch.Tensor) -> int:
    """Find elbow point in singular value curve using second derivative."""
    if len(singular_values) < 3:
        return len(singular_values)
    
    # Compute second derivative
    sv_np = singular_values.cpu().numpy()
    first_diff = np.diff(sv_np)
    second_diff = np.diff(first_diff)
    
    # Find point of maximum curvature
    elbow_idx = np.argmax(np.abs(second_diff)) + 2  # +2 due to double diff
    
    return min(elbow_idx, len(singular_values))


def benchmark_svd_methods(matrix: torch.Tensor, rank: int, 
                         methods: List[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different SVD methods for speed and accuracy.
    
    Args:
        matrix: Input matrix
        rank: Target rank
        methods: List of methods to benchmark
        
    Returns:
        Benchmark results
    """
    if methods is None:
        methods = ['torch_svd', 'randomized_svd']
    
    results = {}
    
    # Reference solution using full SVD
    start_time = time.time()
    U_ref, S_ref, Vt_ref = torch.linalg.svd(matrix, full_matrices=False)
    U_ref, S_ref, Vt_ref = U_ref[:, :rank], S_ref[:rank], Vt_ref[:rank, :]
    ref_time = time.time() - start_time
    ref_reconstruction = torch.mm(U_ref * S_ref.unsqueeze(0), Vt_ref)
    
    for method in methods:
        if method == 'torch_svd':
            # Standard PyTorch SVD
            start_time = time.time()
            U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
            U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
            elapsed_time = time.time() - start_time
            
        elif method == 'randomized_svd':
            # Randomized SVD
            start_time = time.time()
            U, S, Vt = randomized_svd(matrix, rank)
            elapsed_time = time.time() - start_time
            
        else:
            logger.warning(f"Unknown SVD method: {method}")
            continue
        
        # Compute reconstruction and error
        reconstruction = torch.mm(U * S.unsqueeze(0), Vt)
        
        # Error metrics
        frobenius_error = torch.norm(matrix - reconstruction) / torch.norm(matrix)
        relative_error = torch.norm(reconstruction - ref_reconstruction) / torch.norm(ref_reconstruction)
        
        results[method] = {
            'time_seconds': elapsed_time,
            'speedup': ref_time / elapsed_time if elapsed_time > 0 else float('inf'),
            'frobenius_error': frobenius_error.item(),
            'relative_error': relative_error.item(),
            'singular_values': S.cpu().numpy()
        }
    
    return results


def randomized_svd(matrix: torch.Tensor, rank: int, n_oversamples: int = 10,
                  n_power_iterations: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD algorithm for efficient low-rank approximation.
    
    Args:
        matrix: Input matrix [m, n]
        rank: Target rank
        n_oversamples: Number of oversamples for accuracy
        n_power_iterations: Number of power iterations for stability
        
    Returns:
        U, S, Vt: SVD components
    """
    m, n = matrix.shape
    effective_rank = min(rank + n_oversamples, min(m, n))
    
    # Stage A: Find orthonormal basis Q
    omega = torch.randn(n, effective_rank, device=matrix.device, dtype=matrix.dtype)
    Y = torch.mm(matrix, omega)
    
    # Power iterations for better approximation of dominant subspace
    for _ in range(n_power_iterations):
        Y = torch.mm(matrix, torch.mm(matrix.t(), Y))
    
    # QR factorization
    Q, _ = torch.linalg.qr(Y)
    
    # Stage B: Compute SVD of smaller matrix
    B = torch.mm(Q.t(), matrix)
    U_tilde, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Recover U
    U = torch.mm(Q, U_tilde)
    
    # Truncate to desired rank
    return U[:, :rank], S[:rank], Vt[:rank, :]


def adaptive_randomized_svd(matrix: torch.Tensor, tolerance: float = 1e-6,
                           max_rank: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adaptive randomized SVD that automatically determines rank.
    
    Args:
        matrix: Input matrix
        tolerance: Error tolerance for rank selection
        max_rank: Maximum allowed rank
        
    Returns:
        U, S, Vt: SVD components
    """
    m, n = matrix.shape
    if max_rank is None:
        max_rank = min(m, n) // 2
    
    # Start with small rank and increase until tolerance is met
    rank = min(10, max_rank)
    
    while rank <= max_rank:
        U, S, Vt = randomized_svd(matrix, rank)
        
        # Compute reconstruction error
        reconstruction = torch.mm(U * S.unsqueeze(0), Vt)
        error = torch.norm(matrix - reconstruction) / torch.norm(matrix)
        
        if error.item() <= tolerance:
            break
        
        # Increase rank
        rank = min(rank * 2, max_rank)
    
    return U, S, Vt


def hierarchical_svd(matrix: torch.Tensor, levels: int = 3) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Perform hierarchical SVD decomposition at multiple levels.
    
    Args:
        matrix: Input matrix
        levels: Number of hierarchical levels
        
    Returns:
        Dictionary mapping level to SVD components
    """
    m, n = matrix.shape
    max_rank = min(m, n)
    
    results = {}
    
    # Define ranks for each level (exponentially increasing)
    base_rank = max(1, max_rank // (2 ** levels))
    
    for level in range(levels):
        rank = min(base_rank * (2 ** level), max_rank)
        
        # Perform SVD
        U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
        U_level = U[:, :rank]
        S_level = S[:rank]
        Vt_level = Vt[:rank, :]
        
        # Compute metrics
        reconstruction = torch.mm(U_level * S_level.unsqueeze(0), Vt_level)
        error = torch.norm(matrix - reconstruction) / torch.norm(matrix)
        compression_metrics = compute_compression_metrics((m, n), rank)
        
        results[level] = {
            'U': U_level,
            'S': S_level,
            'Vt': Vt_level,
            'rank': rank,
            'reconstruction_error': error.item(),
            'compression_metrics': compression_metrics
        }
    
    return results


def progressive_svd_update(U_old: torch.Tensor, S_old: torch.Tensor, Vt_old: torch.Tensor,
                          update_matrix: torch.Tensor, learning_rate: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Progressively update SVD decomposition with new information.
    
    Args:
        U_old, S_old, Vt_old: Previous SVD components
        update_matrix: Update to incorporate
        learning_rate: Learning rate for update
        
    Returns:
        Updated U, S, Vt components
    """
    # Reconstruct previous matrix
    W_old = torch.mm(U_old * S_old.unsqueeze(0), Vt_old)
    
    # Apply update
    W_new = W_old + learning_rate * update_matrix
    
    # Recompute SVD
    U_new, S_new, Vt_new = torch.linalg.svd(W_new, full_matrices=False)
    
    # Maintain same rank
    rank = len(S_old)
    return U_new[:, :rank], S_new[:rank], Vt_new[:rank, :]


def svd_based_initialization(shape: Tuple[int, int], rank: int, 
                           initialization: str = 'xavier') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Initialize SVD components with proper scaling.
    
    Args:
        shape: Target matrix shape (m, n)
        rank: SVD rank
        initialization: Initialization method
        
    Returns:
        Initialized U, S, Vt components
    """
    m, n = shape
    
    if initialization == 'xavier':
        # Xavier/Glorot initialization
        scale = math.sqrt(2.0 / (m + n))
        U = torch.randn(m, rank) * scale
        S = torch.ones(rank) * scale
        Vt = torch.randn(rank, n) * scale
        
    elif initialization == 'he':
        # He initialization
        scale = math.sqrt(2.0 / m)
        U = torch.randn(m, rank) * scale
        S = torch.ones(rank) * scale
        Vt = torch.randn(rank, n) * scale
        
    elif initialization == 'lecun':
        # LeCun initialization
        scale = math.sqrt(1.0 / m)
        U = torch.randn(m, rank) * scale
        S = torch.ones(rank) * scale
        Vt = torch.randn(rank, n) * scale
        
    else:
        # Random initialization
        U = torch.randn(m, rank)
        S = torch.ones(rank)
        Vt = torch.randn(rank, n)
    
    # Orthogonalize U and Vt
    U, _ = torch.linalg.qr(U)
    Vt_t, _ = torch.linalg.qr(Vt.t())
    Vt = Vt_t.t()
    
    return U, S, Vt


def compute_svd_gradient(U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor,
                        grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients with respect to SVD components.
    
    Args:
        U, S, Vt: SVD components
        grad_output: Gradient from downstream
        
    Returns:
        Gradients for U, S, Vt
    """
    # This is a simplified version - full implementation requires careful handling
    # of the SVD manifold and proper differentiation
    
    # Gradient w.r.t. reconstructed matrix
    # W = U @ diag(S) @ Vt
    # grad_W = grad_output
    
    # Chain rule application
    grad_Vt = torch.mm((U * S.unsqueeze(0)).t(), grad_output)
    grad_U = torch.mm(grad_output, (S.unsqueeze(-1) * Vt).t())
    grad_S = torch.sum(U * torch.mm(grad_output, Vt.t()), dim=0)
    
    return grad_U, grad_S, grad_Vt


def validate_svd_decomposition(U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor,
                              original_matrix: torch.Tensor = None, 
                              tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Validate SVD decomposition for correctness.
    
    Args:
        U, S, Vt: SVD components
        original_matrix: Original matrix (if available)
        tolerance: Numerical tolerance
        
    Returns:
        Validation results
    """
    results = {'valid': True, 'errors': []}
    
    # Check orthogonality of U
    U_orthogonal = torch.allclose(torch.mm(U.t(), U), torch.eye(U.shape[1], device=U.device), atol=tolerance)
    if not U_orthogonal:
        results['valid'] = False
        results['errors'].append('U is not orthogonal')
    
    # Check orthogonality of Vt
    Vt_orthogonal = torch.allclose(torch.mm(Vt, Vt.t()), torch.eye(Vt.shape[0], device=Vt.device), atol=tolerance)
    if not Vt_orthogonal:
        results['valid'] = False
        results['errors'].append('Vt is not orthogonal')
    
    # Check singular values are non-negative and sorted
    S_sorted = torch.allclose(S, torch.sort(S, descending=True)[0], atol=tolerance)
    S_nonneg = torch.all(S >= -tolerance)
    
    if not S_sorted:
        results['valid'] = False
        results['errors'].append('Singular values are not sorted in descending order')
    
    if not S_nonneg:
        results['valid'] = False
        results['errors'].append('Singular values are not non-negative')
    
    # Check reconstruction if original matrix provided
    if original_matrix is not None:
        reconstructed = torch.mm(U * S.unsqueeze(0), Vt)
        reconstruction_error = torch.norm(original_matrix - reconstructed) / torch.norm(original_matrix)
        
        results['reconstruction_error'] = reconstruction_error.item()
        
        if reconstruction_error > tolerance * 10:  # Allow some numerical error
            results['valid'] = False
            results['errors'].append(f'High reconstruction error: {reconstruction_error.item():.2e}')
    
    # Additional checks
    results['orthogonality_error_U'] = torch.norm(torch.mm(U.t(), U) - torch.eye(U.shape[1], device=U.device)).item()
    results['orthogonality_error_Vt'] = torch.norm(torch.mm(Vt, Vt.t()) - torch.eye(Vt.shape[0], device=Vt.device)).item()
    results['condition_number'] = (S.max() / S.min()).item() if S.min() > 1e-12 else float('inf')
    
    return results


def create_svd_summary_report(decomposition_results: Dict[str, Any], 
                             output_path: str = None) -> str:
    """
    Create a comprehensive summary report of SVD decomposition results.
    
    Args:
        decomposition_results: Results from SVD decomposition
        output_path: Path to save report (optional)
        
    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SVD DECOMPOSITION SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    if 'overall_stats' in decomposition_results:
        stats = decomposition_results['overall_stats']
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-" * 40)
        
        for key, value in stats.items():
            if isinstance(value, float):
                report_lines.append(f"{key:30s}: {value:.6f}")
            else:
                report_lines.append(f"{key:30s}: {value}")
        report_lines.append("")
    
    # Layer-wise analysis
    if 'compression_stats' in decomposition_results:
        report_lines.append("LAYER-WISE COMPRESSION ANALYSIS:")
        report_lines.append("-" * 50)
        
        for scale, scale_stats in decomposition_results['compression_stats'].items():
            report_lines.append(f"\nScale: {scale}")
            report_lines.append("  " + "-" * 40)
            
            for layer_name, layer_stats in scale_stats.items():
                report_lines.append(f"  Layer: {layer_name}")
                report_lines.append(f"    Rank: {layer_stats['rank']}")
                report_lines.append(f"    Compression Ratio: {layer_stats['compression_ratio']:.4f}")
                report_lines.append(f"    Parameter Savings: {layer_stats.get('parameter_savings', 'N/A')}")
                report_lines.append("")
    
    # Memory analysis
    if 'layer_info' in decomposition_results and '_summary' in decomposition_results['layer_info']:
        summary = decomposition_results['layer_info']['_summary']
        report_lines.append("MEMORY ANALYSIS:")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Parameters: {summary['total_params']:,}")
        report_lines.append(f"Decomposable Parameters: {summary['decomposable_params']:,}")
        report_lines.append(f"Decomposable Ratio: {summary['decomposable_ratio']:.2%}")
        report_lines.append("")
    
    report = "\n".join(report_lines)
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"SVD summary report saved to {output_path}")
    
    return report