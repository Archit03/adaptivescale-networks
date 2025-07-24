"""
Uncertainty Estimation implementations for AdaptiveScale Networks.

This module provides various uncertainty estimation methods including
Bayesian uncertainty, ensemble uncertainty, and MC dropout.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty estimators."""
    
    # Model architecture
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 1
    num_layers: int = 3
    
    # Dropout settings
    dropout_rate: float = 0.1
    mc_dropout_samples: int = 10
    
    # Bayesian settings
    prior_std: float = 1.0
    posterior_rho_init: float = -3.0
    kl_weight: float = 1.0
    
    # Ensemble settings
    ensemble_size: int = 5
    ensemble_diversity_weight: float = 0.1
    
    # Temperature scaling
    temperature: float = 1.0
    calibration_samples: int = 1000
    
    # Uncertainty thresholds
    epistemic_threshold: float = 0.5
    aleatoric_threshold: float = 0.3
    uncertainty_threshold: float = 0.5
    
    # Training parameters
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    num_epochs: int = 100


class BaseUncertaintyEstimator(ABC, nn.Module):
    """Base class for all uncertainty estimators."""
    
    def __init__(self, config: UncertaintyConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    @abstractmethod
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation."""
        pass
    
    @abstractmethod
    def estimate_uncertainty(self, x: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty for given input."""
        pass
    
    def compute_confidence(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Compute confidence from uncertainty estimates."""
        return torch.exp(-uncertainty)
    
    def is_uncertain(self, uncertainty: torch.Tensor) -> torch.Tensor:
        """Determine if predictions are uncertain based on threshold."""
        return uncertainty > self.config.uncertainty_threshold


class MCDropoutUncertainty(BaseUncertaintyEstimator):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Estimates uncertainty by performing multiple forward passes
    with dropout enabled at test time.
    """
    
    def __init__(self, config: UncertaintyConfig):
        super().__init__(config)
        
        # Build network with dropout
        layers = []
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout_rate))
        
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
        
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized MCDropoutUncertainty with {self._count_parameters()} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with MC dropout uncertainty.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if not return_uncertainty:
            # Single forward pass without uncertainty
            self.network.eval()
            output = self.network(x)
            return {'predictions': output, 'epistemic_uncertainty': None, 'aleatoric_uncertainty': None}
        
        return self.estimate_uncertainty(x)
    
    def estimate_uncertainty(self, x: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty using MC dropout.
        
        Args:
            x: Input tensor
            num_samples: Number of MC samples
            
        Returns:
            Dictionary containing uncertainty estimates
        """
        if num_samples is None:
            num_samples = self.config.mc_dropout_samples
        
        # Enable dropout for uncertainty estimation
        self.network.train()
        
        # Collect predictions from multiple forward passes
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.network(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        pred_variance = predictions.var(dim=0)
        epistemic_uncertainty = pred_variance.mean(dim=-1)  # Model uncertainty
        
        # For MC dropout, aleatoric uncertainty is not directly available
        # We approximate it as a learned parameter or set to zero
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': mean_pred,
            'prediction_samples': predictions,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_variance': pred_variance
        }


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty."""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0, 
                 posterior_rho_init: float = -3.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), posterior_rho_init))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), posterior_rho_init))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.weight_mu, 0, 0.1)
        nn.init.normal_(self.bias_mu, 0, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with weight sampling."""
        # Sample weights and biases
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        
        weight_eps = torch.randn_like(self.weight_mu)
        bias_eps = torch.randn_like(self.bias_mu)
        
        weight = self.weight_mu + weight_std * weight_eps
        bias = self.bias_mu + bias_std * bias_eps
        
        return F.linear(x, weight, bias)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence between posterior and prior."""
        # Weight KL divergence
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        weight_var = weight_std ** 2
        weight_kl = 0.5 * (
            torch.log(self.prior_std ** 2 / weight_var) - 1 + 
            weight_var / (self.prior_std ** 2) + 
            (self.weight_mu ** 2) / (self.prior_std ** 2)
        ).sum()
        
        # Bias KL divergence
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        bias_var = bias_std ** 2
        bias_kl = 0.5 * (
            torch.log(self.prior_std ** 2 / bias_var) - 1 + 
            bias_var / (self.prior_std ** 2) + 
            (self.bias_mu ** 2) / (self.prior_std ** 2)
        ).sum()
        
        return weight_kl + bias_kl


class BayesianUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    Bayesian Neural Network for uncertainty estimation.
    
    Uses Bayesian linear layers to capture both epistemic and aleatoric uncertainty.
    """
    
    def __init__(self, config: UncertaintyConfig):
        super().__init__(config)
        
        # Build Bayesian network
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(BayesianLinear(
            config.input_dim, config.hidden_dim, 
            config.prior_std, config.posterior_rho_init
        ))
        
        # Hidden layers
        for _ in range(config.num_layers - 1):
            self.layers.append(BayesianLinear(
                config.hidden_dim, config.hidden_dim,
                config.prior_std, config.posterior_rho_init
            ))
        
        # Output layer (mean)
        self.output_layer = BayesianLinear(
            config.hidden_dim, config.output_dim,
            config.prior_std, config.posterior_rho_init
        )
        
        # Aleatoric uncertainty layer (log variance)
        self.aleatoric_layer = BayesianLinear(
            config.hidden_dim, config.output_dim,
            config.prior_std, config.posterior_rho_init
        )
        
        logger.info(f"Initialized BayesianUncertaintyEstimator with {self._count_parameters()} parameters")
    
    def _count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Bayesian uncertainty.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if not return_uncertainty:
            # Single forward pass
            h = x
            for layer in self.layers:
                h = F.relu(layer(h))
            
            mean = self.output_layer(h)
            log_var = self.aleatoric_layer(h)
            
            return {
                'predictions': mean,
                'log_variance': log_var,
                'epistemic_uncertainty': None,
                'aleatoric_uncertainty': None
            }
        
        return self.estimate_uncertainty(x)
    
    def estimate_uncertainty(self, x: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty using Bayesian sampling.
        
        Args:
            x: Input tensor
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Dictionary containing uncertainty estimates
        """
        if num_samples is None:
            num_samples = 10  # Default for Bayesian networks
        
        # Collect predictions from multiple forward passes
        predictions = []
        log_variances = []
        
        for _ in range(num_samples):
            h = x
            for layer in self.layers:
                h = F.relu(layer(h))
            
            mean = self.output_layer(h)
            log_var = self.aleatoric_layer(h)
            
            predictions.append(mean)
            log_variances.append(log_var)
        
        predictions = torch.stack(predictions, dim=0)  # [num_samples, batch_size, output_dim]
        log_variances = torch.stack(log_variances, dim=0)
        
        # Compute uncertainties
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0).mean(dim=-1)  # Model uncertainty
        
        # Aleatoric uncertainty (data uncertainty)
        mean_log_var = log_variances.mean(dim=0)
        aleatoric_uncertainty = torch.exp(mean_log_var).mean(dim=-1)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': mean_pred,
            'prediction_samples': predictions,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'log_variance': mean_log_var
        }
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence of all Bayesian layers."""
        kl_div = 0.0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl_div += layer.kl_divergence()
        
        # Add output layers
        kl_div += self.output_layer.kl_divergence()
        kl_div += self.aleatoric_layer.kl_divergence()
        
        return kl_div
    
    def elbo_loss(self, x: torch.Tensor, y: torch.Tensor, num_batches: int = 1) -> torch.Tensor:
        """
        Compute Evidence Lower Bound (ELBO) loss.
        
        Args:
            x: Input tensor
            y: Target tensor
            num_batches: Number of batches (for KL weight scaling)
            
        Returns:
            ELBO loss
        """
        # Forward pass
        output = self.forward(x, return_uncertainty=False)
        predictions = output['predictions']
        log_var = output['log_variance']
        
        # Likelihood term (negative log likelihood)
        precision = torch.exp(-log_var)
        likelihood = 0.5 * (precision * (y - predictions) ** 2 + log_var + math.log(2 * math.pi))
        likelihood_loss = likelihood.sum()
        
        # KL divergence term
        kl_div = self.kl_divergence()
        
        # ELBO = likelihood + KL (scaled by number of batches)
        elbo = likelihood_loss + self.config.kl_weight * kl_div / num_batches
        
        return elbo


class EnsembleUncertaintyEstimator(BaseUncertaintyEstimator):
    """
    Ensemble-based uncertainty estimation.
    
    Uses multiple models to estimate uncertainty through prediction disagreement.
    """
    
    def __init__(self, config: UncertaintyConfig):
        super().__init__(config)
        
        # Create ensemble of models
        self.ensemble = nn.ModuleList()
        for i in range(config.ensemble_size):
            # Create individual model
            layers = []
            layers.append(nn.Linear(config.input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            
            for _ in range(config.num_layers - 1):
                layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(config.dropout_rate))
            
            layers.append(nn.Linear(config.hidden_dim, config.output_dim))
            
            model = nn.Sequential(*layers)
            self.ensemble.append(model)
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1) * config.temperature)
        
        logger.info(f"Initialized EnsembleUncertaintyEstimator with {config.ensemble_size} models")
    
    def forward(self, x: torch.Tensor, return_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with ensemble uncertainty.
        
        Args:
            x: Input tensor
            return_uncertainty: Whether to compute uncertainty
            
        Returns:
            Dictionary containing predictions and uncertainty estimates
        """
        if not return_uncertainty:
            # Average predictions from all models
            predictions = []
            for model in self.ensemble:
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    predictions.append(pred)
            
            mean_pred = torch.stack(predictions, dim=0).mean(dim=0)
            return {'predictions': mean_pred, 'epistemic_uncertainty': None, 'aleatoric_uncertainty': None}
        
        return self.estimate_uncertainty(x)
    
    def estimate_uncertainty(self, x: torch.Tensor, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty using ensemble disagreement.
        
        Args:
            x: Input tensor
            num_samples: Not used for ensemble (determined by ensemble size)
            
        Returns:
            Dictionary containing uncertainty estimates
        """
        # Collect predictions from all ensemble members
        predictions = []
        for model in self.ensemble:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [ensemble_size, batch_size, output_dim]
        
        # Compute statistics
        mean_pred = predictions.mean(dim=0)
        pred_variance = predictions.var(dim=0)
        epistemic_uncertainty = pred_variance.mean(dim=-1)  # Ensemble disagreement
        
        # For ensemble methods, aleatoric uncertainty is not directly available
        # We can estimate it using the average variance within each model's predictions
        aleatoric_uncertainty = torch.zeros_like(epistemic_uncertainty)
        
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'predictions': mean_pred,
            'prediction_samples': predictions,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'prediction_variance': pred_variance,
            'ensemble_std': pred_variance.sqrt().mean(dim=-1)
        }
    
    def calibrate_temperature(self, val_loader, criterion=nn.MSELoss()):
        """
        Calibrate temperature parameter using validation data.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss criterion
        """
        self.eval()
        
        # Collect predictions and targets
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.forward(x, return_uncertainty=False)
                all_logits.append(output['predictions'])
                all_targets.append(y)
        
        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval_temp():
            optimizer.zero_grad()
            scaled_logits = all_logits / self.temperature
            loss = criterion(scaled_logits, all_targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_temp)
        
        logger.info(f"Calibrated temperature: {self.temperature.item():.4f}")
    
    def diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute diversity loss to encourage ensemble diversity.
        
        Args:
            x: Input tensor
            
        Returns:
            Diversity loss
        """
        predictions = []
        for model in self.ensemble:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [ensemble_size, batch_size, output_dim]
        
        # Pairwise diversity loss
        diversity_loss = 0.0
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Encourage different predictions
                similarity = F.cosine_similarity(predictions[i], predictions[j], dim=-1)
                diversity_loss += similarity.mean()
        
        # Normalize by number of pairs
        num_pairs = len(predictions) * (len(predictions) - 1) / 2
        diversity_loss = diversity_loss / num_pairs
        
        return diversity_loss


class UncertaintyEstimator:
    """
    Main interface for uncertainty estimation.
    
    Provides a unified interface for different uncertainty estimation methods.
    """
    
    def __init__(self, estimator_type: str = 'bayesian', config: UncertaintyConfig = None):
        if config is None:
            config = UncertaintyConfig()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize estimator based on type
        if estimator_type.lower() == 'mc_dropout':
            self.estimator = MCDropoutUncertainty(config)
        elif estimator_type.lower() == 'bayesian':
            self.estimator = BayesianUncertaintyEstimator(config)
        elif estimator_type.lower() == 'ensemble':
            self.estimator = EnsembleUncertaintyEstimator(config)
        else:
            raise ValueError(f"Unknown estimator type: {estimator_type}")
        
        self.estimator.to(self.device)
        self.estimator_type = estimator_type
        
        # Training setup
        if estimator_type.lower() == 'bayesian':
            # Use Adam for Bayesian networks
            self.optimizer = torch.optim.Adam(
                self.estimator.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        else:
            # Use Adam for other methods
            self.optimizer = torch.optim.Adam(
                self.estimator.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            )
        
        # Calibration
        self.is_calibrated = False
        
        logger.info(f"Initialized UncertaintyEstimator with {estimator_type} estimator")
    
    def predict_with_uncertainty(self, x: torch.Tensor, return_all: bool = False) -> Dict[str, Any]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            return_all: Whether to return all uncertainty components
            
        Returns:
            Dictionary containing predictions and uncertainty information
        """
        self.estimator.eval()
        
        with torch.no_grad():
            x = x.to(self.device)
            results = self.estimator.estimate_uncertainty(x)
            
            # Compute confidence
            confidence = self.estimator.compute_confidence(results['total_uncertainty'])
            
            # Determine if predictions are uncertain
            is_uncertain = self.estimator.is_uncertain(results['total_uncertainty'])
            
            output = {
                'predictions': results['predictions'].cpu(),
                'total_uncertainty': results['total_uncertainty'].cpu(),
                'confidence': confidence.cpu(),
                'is_uncertain': is_uncertain.cpu()
            }
            
            if return_all:
                output.update({
                    'epistemic_uncertainty': results['epistemic_uncertainty'].cpu(),
                    'aleatoric_uncertainty': results['aleatoric_uncertainty'].cpu(),
                })
                
                if 'prediction_samples' in results:
                    output['prediction_samples'] = results['prediction_samples'].cpu()
            
            return output
    
    def train_step(self, x: torch.Tensor, y: torch.Tensor, num_batches: int = 1) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            x: Input tensor
            y: Target tensor
            num_batches: Number of batches (for Bayesian KL scaling)
            
        Returns:
            Dictionary of training metrics
        """
        self.estimator.train()
        self.optimizer.zero_grad()
        
        x, y = x.to(self.device), y.to(self.device)
        
        if self.estimator_type == 'bayesian':
            # Use ELBO loss for Bayesian networks
            loss = self.estimator.elbo_loss(x, y, num_batches)
            total_loss = loss
        else:
            # Use standard MSE loss for other methods
            output = self.estimator(x, return_uncertainty=False)
            predictions = output['predictions']
            mse_loss = F.mse_loss(predictions, y)
            total_loss = mse_loss
            
            # Add diversity loss for ensemble
            if self.estimator_type == 'ensemble':
                diversity_loss = self.estimator.diversity_loss(x)
                total_loss += self.config.ensemble_diversity_weight * diversity_loss
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.estimator.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            if self.estimator_type != 'bayesian':
                output = self.estimator(x, return_uncertainty=False)
                predictions = output['predictions']
                mse = F.mse_loss(predictions, y).item()
                mae = F.l1_loss(predictions, y).item()
            else:
                output = self.estimator(x, return_uncertainty=False)
                predictions = output['predictions']
                mse = F.mse_loss(predictions, y).item()
                mae = F.l1_loss(predictions, y).item()
        
        metrics = {
            'total_loss': total_loss.item(),
            'mse': mse,
            'mae': mae
        }
        
        return metrics
    
    def calibrate(self, val_loader):
        """Calibrate the uncertainty estimator using validation data."""
        if self.estimator_type == 'ensemble':
            self.estimator.calibrate_temperature(val_loader)
            self.is_calibrated = True
            logger.info("Uncertainty estimator calibrated")
        else:
            logger.warning(f"Calibration not implemented for {self.estimator_type}")
    
    def save_checkpoint(self, filepath: str, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'estimator_state_dict': self.estimator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'estimator_type': self.estimator_type,
            'metrics': metrics,
            'is_calibrated': self.is_calibrated
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.estimator.load_state_dict(checkpoint['estimator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_calibrated = checkpoint.get('is_calibrated', False)
        logger.info(f"Loaded checkpoint from {filepath}")
        return checkpoint


# Utility functions for uncertainty analysis
def compute_calibration_error(predictions: torch.Tensor, uncertainties: torch.Tensor, 
                             targets: torch.Tensor, n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration error metrics.
    
    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        targets: Ground truth targets
        n_bins: Number of bins for calibration plot
        
    Returns:
        Dictionary of calibration metrics
    """
    # Convert uncertainty to confidence
    confidences = torch.exp(-uncertainties)
    errors = torch.abs(predictions - targets)
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = 1.0 - errors[in_bin].mean()  # Convert error to accuracy
            bin_count = in_bin.sum()
            
            bin_confidences.append(bin_confidence.item())
            bin_accuracies.append(bin_accuracy.item())
            bin_counts.append(bin_count.item())
    
    # Compute Expected Calibration Error (ECE)
    ece = 0.0
    total_samples = len(predictions)
    
    for conf, acc, count in zip(bin_confidences, bin_accuracies, bin_counts):
        ece += (count / total_samples) * abs(conf - acc)
    
    # Compute Maximum Calibration Error (MCE)
    if bin_confidences and bin_accuracies:
        mce = max(abs(conf - acc) for conf, acc in zip(bin_confidences, bin_accuracies))
    else:
        mce = 0.0
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'bin_confidences': bin_confidences,
        'bin_accuracies': bin_accuracies,
        'bin_counts': bin_counts
    }


def uncertainty_decomposition(predictions: torch.Tensor, targets: torch.Tensor, 
                             epistemic: torch.Tensor, aleatoric: torch.Tensor) -> Dict[str, float]:
    """
    Decompose uncertainty into epistemic and aleatoric components.
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        epistemic: Epistemic uncertainty estimates
        aleatoric: Aleatoric uncertainty estimates
        
    Returns:
        Dictionary of uncertainty decomposition metrics
    """
    total_uncertainty = epistemic + aleatoric
    errors = (predictions - targets) ** 2
    
    # Correlation between uncertainties and errors
    epistemic_correlation = torch.corrcoef(torch.stack([epistemic, errors]))[0, 1].item()
    aleatoric_correlation = torch.corrcoef(torch.stack([aleatoric, errors]))[0, 1].item()
    total_correlation = torch.corrcoef(torch.stack([total_uncertainty, errors]))[0, 1].item()
    
    return {
        'epistemic_mean': epistemic.mean().item(),
        'aleatoric_mean': aleatoric.mean().item(),
        'total_mean': total_uncertainty.mean().item(),
        'epistemic_std': epistemic.std().item(),
        'aleatoric_std': aleatoric.std().item(),
        'total_std': total_uncertainty.std().item(),
        'epistemic_error_correlation': epistemic_correlation,
        'aleatoric_error_correlation': aleatoric_correlation,
        'total_error_correlation': total_correlation,
        'epistemic_ratio': (epistemic.mean() / total_uncertainty.mean()).item(),
        'aleatoric_ratio': (aleatoric.mean() / total_uncertainty.mean()).item()
    }