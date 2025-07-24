"""
SVD Decomposer for AdaptiveScale Networks.

This module provides multi-scale singular value decomposition for neural network
parameter compression and adaptation. It supports hierarchical decomposition,
adaptive rank selection, and efficient reconstruction.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict, OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class SVDConfig:
    """Configuration for SVD decomposition."""
    
    # SVD scales for multi-scale decomposition
    svd_scales: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7])
    
    # Target layers for decomposition
    target_layers: List[str] = field(default_factory=lambda: ["c_fc", "c_proj", "c_attn"])
    
    # Rank selection parameters
    max_rank_ratio: float = 0.8
    min_rank: int = 1
    adaptive_rank_selection: bool = True
    rank_selection_metric: str = "reconstruction_error"  # reconstruction_error, frobenius_norm, singular_values
    
    # SVD parameters
    use_randomized_svd: bool = False
    svd_driver: str = "gesvd"  # gesvd, gesvda, gesdd
    numerical_stability_eps: float = 1e-8
    
    # Compression settings
    target_compression_ratio: float = 0.5
    preserve_energy_ratio: float = 0.90
    
    # Hierarchical decomposition
    use_hierarchical: bool = True
    hierarchy_levels: int = 3
    
    # Performance optimization
    use_fp16: bool = False
    chunk_size: int = 1000  # For large matrices
    parallel_decomposition: bool = True


class SVDLayer(nn.Module):
    """
    SVD-decomposed layer that replaces original linear layers.
    
    Represents a linear layer as W = U @ S @ V^T where:
    - U: Left singular vectors [out_features, rank]
    - S: Singular values [rank]
    - V^T: Right singular vectors [rank, in_features]
    """
    
    def __init__(self, original_weight: torch.Tensor, rank: int, bias: Optional[torch.Tensor] = None):
        super().__init__()
        
        self.original_shape = original_weight.shape
        self.rank = rank
        
        # Perform SVD decomposition
        U, S, Vt = self._decompose_weight(original_weight, rank)
        
        # Store as parameters
        self.U = nn.Parameter(U)
        self.S = nn.Parameter(S)
        self.Vt = nn.Parameter(Vt)
        
        # Bias handling
        if bias is not None:
            self.bias = nn.Parameter(bias.clone())
        else:
            self.register_parameter('bias', None)
        
        # Compression statistics
        original_params = original_weight.numel() + (bias.numel() if bias is not None else 0)
        compressed_params = U.numel() + S.numel() + Vt.numel() + (bias.numel() if bias is not None else 0)
        self.compression_ratio = compressed_params / original_params
        print(f"SVDLayer initialized: rank={rank}, compression_ratio={self.compression_ratio:.4f}")
        print(f"Original shape: {self.original_shape}, Compressed shape: ({self.rank}, {self.original_shape[1]})")
        print(f"Original params: {original_params}, Compressed params: {compressed_params}")
        print(f"Rank ratio: {self.rank / min(self.original_shape):.4f}")
        

    def _decompose_weight(self, weight: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decompose weight matrix using SVD."""
        # Ensure weight is 2D
        if weight.dim() > 2:
            original_shape = weight.shape
            weight = weight.view(weight.shape[0], -1)
        
        # Perform SVD
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        
        # Truncate to desired rank
        rank = min(rank, min(U.shape[0], Vt.shape[0]))
        U = U[:, :rank].contiguous()
        S = S[:rank].contiguous()
        Vt = Vt[:rank, :].contiguous()
        
        return U, S, Vt
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through SVD-decomposed layer."""
        # Reconstruct weight: W = U @ diag(S) @ Vt
        # But we can compute more efficiently: (U @ diag(S) @ Vt) @ x = U @ (diag(S) @ (Vt @ x))
        
        # x: [batch_size, ..., in_features]
        # Reshape input if needed
        input_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, x.shape[-1])
        
        # Forward pass: x -> Vt @ x -> S * (Vt @ x) -> U @ (S * (Vt @ x))
        x = torch.mm(x, self.Vt.t())  # [batch, rank]
        x = x * self.S.unsqueeze(0)   # [batch, rank]
        x = torch.mm(x, self.U.t())   # [batch, out_features]
        
        # Add bias if present
        if self.bias is not None:
            x = x + self.bias
        
        # Reshape output back to original shape
        if len(input_shape) > 2:
            x = x.view(*input_shape[:-1], -1)
        
        return x
    
    def reconstruct_weight(self) -> torch.Tensor:
        """Reconstruct the original weight matrix."""
        return torch.mm(self.U * self.S.unsqueeze(0), self.Vt)
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        return {
            'compression_ratio': self.compression_ratio,
            'rank': self.rank,
            'original_params': self.original_shape[0] * self.original_shape[1],
            'compressed_params': self.U.numel() + self.S.numel() + self.Vt.numel(),
            'rank_ratio': self.rank / min(self.original_shape)
        }


class MultiScaleSVDDecomposer:
    """
    Multi-scale SVD decomposer for neural networks.
    
    Performs hierarchical SVD decomposition at multiple scales to enable
    adaptive parameter compression and fine-grained control over model capacity.
    """
    
    def __init__(self, config: SVDConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Decomposition cache
        self.decomposition_cache = {}
        self.compression_stats = defaultdict(dict)
        self.layer_info = {}
        
        logger.info(f"Initialized MultiScaleSVDDecomposer with scales: {config.svd_scales}")
    
    def decompose_model(self, model: nn.Module, target_scale: float = None) -> Dict[str, Any]:
        """
        Decompose model layers using multi-scale SVD.
        
        Args:
            model: Neural network model to decompose
            target_scale: Target compression scale (if None, use config scales)
            
        Returns:
            Dictionary containing decomposed weights and statistics
        """
        if target_scale is not None:
            scales = [target_scale]
        else:
            scales = self.config.svd_scales
        
        decomposed_weights = {}
        total_stats = defaultdict(list)
        
        # Analyze model structure
        self._analyze_model_structure(model)
        
        # Decompose at each scale
        for scale in scales:
            logger.info(f"Decomposing model at scale {scale}")
            
            scale_weights = {}
            scale_stats = {}
            
            for name, module in model.named_modules():
                if self._should_decompose_layer(name, module):
                    # Get target rank for this layer and scale
                    target_rank = self._compute_target_rank(module, scale)
                    
                    # Decompose layer
                    decomposed = self._decompose_layer(module, target_rank, name)
                    
                    if decomposed is not None:
                        scale_weights[name] = decomposed
                        scale_stats[name] = decomposed.get_compression_stats()
                        
                        # Update global stats
                        for key, value in scale_stats[name].items():
                            total_stats[key].append(value)
            
            decomposed_weights[scale] = scale_weights
            self.compression_stats[scale] = scale_stats
        
        # Compute overall statistics
        overall_stats = self._compute_overall_stats(total_stats)
        
        return {
            'decomposed_weights': decomposed_weights,
            'compression_stats': self.compression_stats,
            'overall_stats': overall_stats,
            'layer_info': self.layer_info
        }
    
    def _analyze_model_structure(self, model: nn.Module):
        """Analyze model structure to determine decomposition strategy."""
        self.layer_info.clear()
        
        total_params = 0
        decomposable_params = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                weight_shape = module.weight.shape
                param_count = module.weight.numel()
                
                self.layer_info[name] = {
                    'type': 'Linear',
                    'weight_shape': weight_shape,
                    'param_count': param_count,
                    'has_bias': module.bias is not None,
                    'decomposable': self._should_decompose_layer(name, module)
                }
                
                total_params += param_count
                if self.layer_info[name]['decomposable']:
                    decomposable_params += param_count
        
        self.layer_info['_summary'] = {
            'total_params': total_params,
            'decomposable_params': decomposable_params,
            'decomposable_ratio': decomposable_params / total_params if total_params > 0 else 0
        }
        
        logger.info(f"Model analysis: {total_params:,} total params, "
                   f"{decomposable_params:,} decomposable ({100*decomposable_params/total_params:.1f}%)")
    
    def _should_decompose_layer(self, name: str, module: nn.Module) -> bool:
        """Determine if a layer should be decomposed."""
        if not isinstance(module, nn.Linear):
            return False
        
        # Check if layer name matches target patterns
        if self.config.target_layers:
            return any(target in name for target in self.config.target_layers)
        
        # Default: decompose all linear layers with sufficient parameters
        return module.weight.numel() > 100  # Minimum parameter threshold
    
    def _compute_target_rank(self, module: nn.Module, scale: float) -> int:
        """Compute target rank for a layer given compression scale."""
        if not isinstance(module, nn.Linear):
            return None
        
        weight_shape = module.weight.shape
        max_possible_rank = min(weight_shape)
        
        if self.config.adaptive_rank_selection:
            # Use adaptive rank selection based on singular value analysis
            target_rank = self._adaptive_rank_selection(module.weight, scale)
        else:
            # Simple rank based on scale
            target_rank = max(self.config.min_rank, int(max_possible_rank * scale))
        
        # Apply constraints
        target_rank = max(self.config.min_rank, target_rank)
        target_rank = min(int(max_possible_rank * self.config.max_rank_ratio), target_rank)
        
        return target_rank
    
    def _adaptive_rank_selection(self, weight: torch.Tensor, scale: float) -> int:
        """Adaptively select rank based on singular value analysis."""
        # Perform SVD to analyze singular values
        U, S, Vt = torch.linalg.svd(weight, full_matrices=False)
        
        if self.config.rank_selection_metric == "singular_values":
            # Select rank based on singular value energy
            total_energy = (S ** 2).sum()
            cumulative_energy = torch.cumsum(S ** 2, dim=0)
            energy_ratios = cumulative_energy / total_energy
            
            target_energy = self.config.preserve_energy_ratio * scale
            target_rank = torch.searchsorted(energy_ratios, target_energy).item() + 1
            
        elif self.config.rank_selection_metric == "reconstruction_error":
            # Select rank based on reconstruction error threshold
            max_rank = min(weight.shape)
            errors = []
            
            for r in range(1, min(max_rank, 50)):  # Sample ranks for efficiency
                U_r = U[:, :r]
                S_r = S[:r]
                Vt_r = Vt[:r, :]
                
                reconstructed = torch.mm(U_r * S_r.unsqueeze(0), Vt_r)
                error = torch.norm(weight - reconstructed) / torch.norm(weight)
                errors.append(error.item())
            
            # Find rank where error is acceptable
            target_error = 1.0 - scale * self.config.preserve_energy_ratio
            target_rank = next((i + 1 for i, err in enumerate(errors) if err <= target_error), len(errors))
            
        else:  # frobenius_norm
            # Select rank based on Frobenius norm preservation
            total_norm = torch.norm(S)
            cumulative_norm = torch.sqrt(torch.cumsum(S ** 2, dim=0))
            norm_ratios = cumulative_norm / total_norm
            
            target_ratio = scale * self.config.preserve_energy_ratio
            target_rank = torch.searchsorted(norm_ratios, target_ratio).item() + 1
        
        return max(1, target_rank)
    
    def _decompose_layer(self, module: nn.Module, target_rank: int, layer_name: str) -> Optional[SVDLayer]:
        """Decompose a single layer."""
        if not isinstance(module, nn.Linear) or target_rank is None:
            return None
        
        try:
            # Create SVD layer
            svd_layer = SVDLayer(
                original_weight=module.weight.data,
                rank=target_rank,
                bias=module.bias.data if module.bias is not None else None
            )
            
            logger.debug(f"Decomposed layer {layer_name}: "
                        f"shape {module.weight.shape} -> rank {target_rank} "
                        f"(compression: {svd_layer.compression_ratio:.3f})")
            
            return svd_layer
            
        except Exception as e:
            logger.error(f"Failed to decompose layer {layer_name}: {e}")
            return None
    
    def _compute_overall_stats(self, stats: Dict[str, List]) -> Dict[str, float]:
        """Compute overall compression statistics."""
        if not stats:
            return {}
        
        overall = {}
        for key, values in stats.items():
            if values:
                overall[f'avg_{key}'] = np.mean(values)
                overall[f'std_{key}'] = np.std(values)
                overall[f'min_{key}'] = np.min(values)
                overall[f'max_{key}'] = np.max(values)
        
        # Aggregate compression ratio
        if 'compression_ratio' in stats:
            overall['overall_compression_ratio'] = np.mean(stats['compression_ratio'])
            overall['compression_factor'] = 1.0 / overall['overall_compression_ratio']
        
        return overall
    
    def apply_decomposition(self, model: nn.Module, decomposed_weights: Dict[str, SVDLayer], 
                          scale: float = None) -> nn.Module:
        """
        Apply SVD decomposition to model by replacing layers.
        
        Args:
            model: Original model
            decomposed_weights: Decomposed layer weights
            scale: Scale to apply (if None, use first available)
            
        Returns:
            Model with SVD-decomposed layers
        """
        if scale is None:
            scale = list(decomposed_weights.keys())[0]
        
        if scale not in decomposed_weights:
            raise ValueError(f"Scale {scale} not found in decomposed weights")
        
        # Create a copy of the model
        decomposed_model = type(model)(model.config) if hasattr(model, 'config') else model
        decomposed_model.load_state_dict(model.state_dict())
        
        # Replace layers with SVD decomposed versions
        scale_weights = decomposed_weights[scale]
        
        for name, svd_layer in scale_weights.items():
            # Navigate to the module
            module_path = name.split('.')
            parent_module = decomposed_model
            
            for path_part in module_path[:-1]:
                parent_module = getattr(parent_module, path_part)
            
            # Replace the module
            setattr(parent_module, module_path[-1], svd_layer)
            
            logger.debug(f"Replaced layer {name} with SVD decomposition")
        
        logger.info(f"Applied SVD decomposition at scale {scale} to model")
        return decomposed_model
    
    def reconstruct_weights(self, decomposed_weights: Dict[str, SVDLayer]) -> Dict[str, torch.Tensor]:
        """Reconstruct original weights from SVD decomposition."""
        reconstructed = {}
        
        for name, svd_layer in decomposed_weights.items():
            reconstructed[name] = svd_layer.reconstruct_weight()
        
        return reconstructed
    
    def compute_reconstruction_error(self, original_model: nn.Module, 
                                   decomposed_weights: Dict[str, SVDLayer]) -> Dict[str, float]:
        """Compute reconstruction error for each layer."""
        errors = {}
        
        for name, svd_layer in decomposed_weights.items():
            # Get original weight
            module_path = name.split('.')
            original_module = original_model
            
            for path_part in module_path:
                original_module = getattr(original_module, path_part)
            
            original_weight = original_module.weight.data
            reconstructed_weight = svd_layer.reconstruct_weight()
            
            # Compute various error metrics
            frobenius_error = torch.norm(original_weight - reconstructed_weight).item()
            relative_error = frobenius_error / torch.norm(original_weight).item()
            max_error = torch.max(torch.abs(original_weight - reconstructed_weight)).item()
            
            errors[name] = {
                'frobenius_error': frobenius_error,
                'relative_error': relative_error,
                'max_error': max_error
            }
        
        return errors
    
    def get_memory_savings(self, original_model: nn.Module) -> Dict[str, float]:
        """Compute memory savings from decomposition."""
        original_params = sum(p.numel() for p in original_model.parameters())
        
        savings_by_scale = {}
        
        for scale, scale_stats in self.compression_stats.items():
            total_original = 0
            total_compressed = 0
            
            for layer_stats in scale_stats.values():
                total_original += layer_stats['original_params']
                total_compressed += layer_stats['compressed_params']
            
            # Account for non-decomposed parameters
            non_decomposed_params = original_params - total_original
            total_compressed += non_decomposed_params
            
            memory_ratio = total_compressed / original_params
            memory_savings = 1.0 - memory_ratio
            
            savings_by_scale[scale] = {
                'original_params': original_params,
                'compressed_params': total_compressed,
                'memory_ratio': memory_ratio,
                'memory_savings': memory_savings,
                'savings_mb': (original_params - total_compressed) * 4 / (1024 * 1024)  # Assuming float32
            }
        
        return savings_by_scale
    
    def save_decomposition(self, filepath: str, decomposed_weights: Dict, metadata: Dict = None):
        """Save SVD decomposition to file."""
        save_dict = {
            'decomposed_weights': decomposed_weights,
            'compression_stats': dict(self.compression_stats),
            'config': self.config,
            'layer_info': self.layer_info
        }
        
        if metadata:
            save_dict['metadata'] = metadata
        
        torch.save(save_dict, filepath)
        logger.info(f"Saved SVD decomposition to {filepath}")
    
    def load_decomposition(self, filepath: str) -> Dict[str, Any]:
        """Load SVD decomposition from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.compression_stats = defaultdict(dict, checkpoint['compression_stats'])
        self.layer_info = checkpoint['layer_info']
        
        logger.info(f"Loaded SVD decomposition from {filepath}")
        return checkpoint


# Utility functions for SVD operations
def randomized_svd(matrix: torch.Tensor, rank: int, n_oversamples: int = 10, 
                   n_power_iterations: int = 2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomized SVD for large matrices.
    
    Args:
        matrix: Input matrix [m, n]
        rank: Target rank
        n_oversamples: Number of oversamples for accuracy
        n_power_iterations: Number of power iterations
        
    Returns:
        U, S, Vt: SVD components
    """
    m, n = matrix.shape
    effective_rank = min(rank + n_oversamples, min(m, n))
    
    # Random projection
    omega = torch.randn(n, effective_rank, device=matrix.device, dtype=matrix.dtype)
    Q = torch.mm(matrix, omega)
    
    # Power iterations for better approximation
    for _ in range(n_power_iterations):
        Q = torch.mm(matrix, torch.mm(matrix.t(), Q))
    
    # QR decomposition
    Q, _ = torch.linalg.qr(Q)
    
    # Project matrix
    B = torch.mm(Q.t(), matrix)
    
    # SVD of smaller matrix
    U_tilde, S, Vt = torch.linalg.svd(B, full_matrices=False)
    
    # Reconstruct U
    U = torch.mm(Q, U_tilde)
    
    # Truncate to desired rank
    return U[:, :rank], S[:rank], Vt[:rank, :]


def compute_svd_rank_for_compression(singular_values: torch.Tensor, 
                                    target_compression: float) -> int:
    """
    Compute optimal rank for target compression ratio.
    
    Args:
        singular_values: Singular values in descending order
        target_compression: Target compression ratio (0 < ratio < 1)
        
    Returns:
        Optimal rank
    """
    m, n = len(singular_values), len(singular_values)  # Assume square for simplicity
    original_params = m * n
    
    for rank in range(1, len(singular_values) + 1):
        compressed_params = rank * (m + n)
        compression_ratio = compressed_params / original_params
        
        if compression_ratio <= target_compression:
            return rank
    
    return len(singular_values)


def energy_based_rank_selection(singular_values: torch.Tensor, 
                               energy_threshold: float = 0.95) -> int:
    """
    Select rank based on energy preservation.
    
    Args:
        singular_values: Singular values in descending order
        energy_threshold: Fraction of energy to preserve
        
    Returns:
        Selected rank
    """
    total_energy = (singular_values ** 2).sum()
    cumulative_energy = torch.cumsum(singular_values ** 2, dim=0)
    energy_ratios = cumulative_energy / total_energy
    
    # Find first rank that preserves required energy
    rank = torch.searchsorted(energy_ratios, energy_threshold).item() + 1
    return min(rank, len(singular_values))


def batch_svd_decomposition(matrices: List[torch.Tensor], ranks: List[int], 
                           use_randomized: bool = False) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Perform SVD decomposition on a batch of matrices.
    
    Args:
        matrices: List of matrices to decompose
        ranks: Target ranks for each matrix
        use_randomized: Whether to use randomized SVD
        
    Returns:
        List of (U, S, Vt) tuples
    """
    results = []
    
    for matrix, rank in zip(matrices, ranks):
        if use_randomized and min(matrix.shape) > 100:
            U, S, Vt = randomized_svd(matrix, rank)
        else:
            U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
            U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
        
        results.append((U, S, Vt))
    
    return results