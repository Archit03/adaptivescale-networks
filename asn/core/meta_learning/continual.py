"""
Continual Learning implementations for AdaptiveScale Networks.

This module provides various continual learning algorithms including
Elastic Weight Consolidation (EWC), PackNet, and Progressive Networks.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ContinualConfig:
    """Configuration for continual learning."""
    
    # EWC parameters
    ewc_lambda: float = 1000.0
    gamma: float = 1.0  # Forgetting factor for online EWC
    online_ewc: bool = True
    fisher_estimation_samples: int = 1000
    
    # Memory parameters
    memory_strength: float = 0.5
    memory_size: int = 1000
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs_per_task: int = 10
    
    # Regularization
    l2_reg: float = 0.0
    dropout_rate: float = 0.1
    
    # Logging
    log_interval: int = 10


class EWCLoss(nn.Module):
    """
    Elastic Weight Consolidation loss.
    
    Prevents catastrophic forgetting by adding a quadratic penalty
    on changes to important parameters.
    """
    
    def __init__(self, ewc_lambda: float = 1000.0):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher_information = {}
        self.optimal_params = {}
        
    def register_task(self, model: nn.Module, fisher_information: Dict[str, torch.Tensor]):
        """
        Register a new task by storing Fisher information and optimal parameters.
        
        Args:
            model: The model after training on the task
            fisher_information: Fisher information matrix for each parameter
        """
        self.fisher_information = {}
        self.optimal_params = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in fisher_information:
                self.fisher_information[name] = fisher_information[name].clone()
                self.optimal_params[name] = param.data.clone()
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty.
        
        Args:
            model: Current model
            
        Returns:
            EWC penalty term
        """
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                penalty += (self.fisher_information[name] * 
                           (param - self.optimal_params[name]).pow(2)).sum()
        
        return self.ewc_lambda * penalty


class OnlineEWC(nn.Module):
    """
    Online Elastic Weight Consolidation.
    
    Updates Fisher information incrementally as new tasks are encountered.
    """
    
    def __init__(self, ewc_lambda: float = 1000.0, gamma: float = 1.0):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.gamma = gamma
        self.fisher_information = {}
        self.optimal_params = {}
        self.task_count = 0
        
    def update_fisher_and_params(self, model: nn.Module, 
                                fisher_information: Dict[str, torch.Tensor]):
        """
        Update Fisher information and optimal parameters online.
        
        Args:
            model: Current model
            fisher_information: New Fisher information
        """
        self.task_count += 1
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in fisher_information:
                if name not in self.fisher_information:
                    # First task
                    self.fisher_information[name] = fisher_information[name].clone()
                    self.optimal_params[name] = param.data.clone()
                else:
                    # Update with exponential moving average
                    self.fisher_information[name] = (
                        self.gamma * self.fisher_information[name] + 
                        fisher_information[name]
                    )
                    self.optimal_params[name] = param.data.clone()
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute online EWC penalty."""
        penalty = 0.0
        
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                penalty += (self.fisher_information[name] * 
                           (param - self.optimal_params[name]).pow(2)).sum()
        
        return self.ewc_lambda * penalty


class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation for continual learning.
    
    Implements both standard EWC and online EWC variants.
    """
    
    def __init__(self, model: nn.Module, config: ContinualConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
        
        # Initialize EWC loss
        if config.online_ewc:
            self.ewc_loss = OnlineEWC(config.ewc_lambda, config.gamma)
        else:
            self.ewc_loss = EWCLoss(config.ewc_lambda)
        
        # Task history
        self.task_history = []
        self.current_task = 0
        
        logger.info(f"Initialized EWC with config: {config}")
    
    def compute_fisher_information(self, dataloader, num_samples: int = None) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher information matrix for current task.
        
        Args:
            dataloader: DataLoader for current task
            num_samples: Number of samples to use (if None, use all)
            
        Returns:
            Dictionary of Fisher information for each parameter
        """
        if num_samples is None:
            num_samples = self.config.fisher_estimation_samples
        
        self.model.eval()
        fisher_information = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_information[name] = torch.zeros_like(param)
        
        sample_count = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            if sample_count >= num_samples:
                break
                
            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)
            
            # Forward pass
            output = self.model(data)
            loss = F.log_softmax(output, dim=1)
            
            # Sample labels from model's prediction
            sampled_y = torch.multinomial(F.softmax(output, dim=1), 1).squeeze()
            
            # Compute loss for sampled labels
            loss = F.nll_loss(loss, sampled_y)
            
            # Compute gradients
            self.model.zero_grad()
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_information[name] += param.grad.pow(2) * batch_size
            
            sample_count += batch_size
        
        # Normalize by number of samples
        for name in fisher_information:
            fisher_information[name] /= sample_count
        
        logger.info(f"Computed Fisher information using {sample_count} samples")
        return fisher_information
    
    def consolidate_task(self, dataloader):
        """
        Consolidate knowledge from current task.
        
        Args:
            dataloader: DataLoader for current task
        """
        logger.info(f"Consolidating task {self.current_task}")
        
        # Compute Fisher information
        fisher_info = self.compute_fisher_information(dataloader)
        
        # Update EWC loss
        if isinstance(self.ewc_loss, OnlineEWC):
            self.ewc_loss.update_fisher_and_params(self.model, fisher_info)
        else:
            self.ewc_loss.register_task(self.model, fisher_info)
        
        # Store task information
        self.task_history.append({
            'task_id': self.current_task,
            'model_state': copy.deepcopy(self.model.state_dict()),
            'fisher_info': fisher_info
        })
        
        self.current_task += 1
    
    def train_task(self, dataloader, optimizer, num_epochs: int = None):
        """
        Train on a new task with EWC regularization.
        
        Args:
            dataloader: DataLoader for current task
            optimizer: Optimizer for training
            num_epochs: Number of epochs to train
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs_per_task
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_ewc_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = self.model(data)
                task_loss = F.cross_entropy(output, target)
                
                # EWC penalty
                ewc_penalty = self.ewc_loss(self.model) if self.current_task > 0 else 0.0
                
                # Total loss
                total_loss = task_loss + ewc_penalty
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += task_loss.item()
                if isinstance(ewc_penalty, torch.Tensor):
                    epoch_ewc_loss += ewc_penalty.item()
                
                if batch_idx % self.config.log_interval == 0:
                    logger.debug(f"Task {self.current_task}, Epoch {epoch}, "
                               f"Batch {batch_idx}: Loss={task_loss.item():.4f}, "
                               f"EWC={ewc_penalty if isinstance(ewc_penalty, float) else ewc_penalty.item():.4f}")
            
            avg_loss = epoch_loss / len(dataloader)
            avg_ewc_loss = epoch_ewc_loss / len(dataloader)
            
            logger.info(f"Task {self.current_task}, Epoch {epoch}: "
                       f"Avg Loss={avg_loss:.4f}, Avg EWC Loss={avg_ewc_loss:.4f}")
    
    def evaluate_task(self, dataloader, task_id: int = None) -> Dict[str, float]:
        """
        Evaluate model on a specific task.
        
        Args:
            dataloader: DataLoader for evaluation
            task_id: Task ID for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                total_loss += loss.item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        logger.info(f"Task {task_id if task_id is not None else 'current'}: "
                   f"Accuracy={accuracy:.2f}%, Loss={avg_loss:.4f}")
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def compute_forgetting(self, task_dataloaders: List) -> Dict[str, float]:
        """
        Compute forgetting metrics across all previous tasks.
        
        Args:
            task_dataloaders: List of dataloaders for all tasks
            
        Returns:
            Dictionary of forgetting metrics
        """
        if len(self.task_history) < 2:
            return {'average_forgetting': 0.0, 'forgetting_per_task': []}
        
        forgetting_per_task = []
        
        for i, task_data in enumerate(self.task_history[:-1]):  # Exclude current task
            # Load model state from when this task was learned
            original_state = task_data['model_state']
            current_state = self.model.state_dict()
            
            # Evaluate with original model
            self.model.load_state_dict(original_state)
            original_metrics = self.evaluate_task(task_dataloaders[i], task_id=i)
            
            # Evaluate with current model
            self.model.load_state_dict(current_state)
            current_metrics = self.evaluate_task(task_dataloaders[i], task_id=i)
            
            # Compute forgetting
            forgetting = original_metrics['accuracy'] - current_metrics['accuracy']
            forgetting_per_task.append(forgetting)
            
            logger.info(f"Task {i} forgetting: {forgetting:.2f}%")
        
        average_forgetting = np.mean(forgetting_per_task)
        
        return {
            'average_forgetting': average_forgetting,
            'forgetting_per_task': forgetting_per_task,
            'max_forgetting': max(forgetting_per_task) if forgetting_per_task else 0.0,
            'min_forgetting': min(forgetting_per_task) if forgetting_per_task else 0.0
        }


class ContinualLearner:
    """
    Main continual learning interface that orchestrates different algorithms.
    """
    
    def __init__(self, model: nn.Module, config: ContinualConfig, algorithm: str = 'ewc'):
        self.model = model
        self.config = config
        self.algorithm = algorithm.lower()
        self.device = next(model.parameters()).device
        
        # Initialize specific algorithm
        if self.algorithm == 'ewc':
            self.learner = ElasticWeightConsolidation(model, config)
        else:
            raise ValueError(f"Unknown continual learning algorithm: {algorithm}")
        
        # Training history
        self.training_history = []
        self.evaluation_history = []
        
        logger.info(f"Initialized ContinualLearner with algorithm: {algorithm}")
    
    def learn_task(self, train_dataloader, val_dataloader=None, task_name: str = None):
        """
        Learn a new task.
        
        Args:
            train_dataloader: Training data for the task
            val_dataloader: Validation data for the task
            task_name: Optional task name for logging
        """
        task_id = len(self.training_history)
        task_name = task_name or f"Task_{task_id}"
        
        logger.info(f"Learning {task_name} (ID: {task_id})")
        
        # Create optimizer for this task
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_reg
        )
        
        # Train on the task
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        self.learner.train_task(train_dataloader, optimizer, self.config.num_epochs_per_task)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            training_time = 0.0
        
        # Evaluate on training data
        train_metrics = self.learner.evaluate_task(train_dataloader, task_id)
        
        # Evaluate on validation data if provided
        val_metrics = {}
        if val_dataloader is not None:
            val_metrics = self.learner.evaluate_task(val_dataloader, task_id)
        
        # Consolidate knowledge
        self.learner.consolidate_task(train_dataloader)
        
        # Store training history
        task_info = {
            'task_id': task_id,
            'task_name': task_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'num_parameters': sum(p.numel() for p in self.model.parameters())
        }
        
        self.training_history.append(task_info)
        
        logger.info(f"Completed learning {task_name}: "
                   f"Train Acc={train_metrics['accuracy']:.2f}%, "
                   f"Val Acc={val_metrics.get('accuracy', 'N/A')}, "
                   f"Time={training_time:.2f}s")
    
    def evaluate_all_tasks(self, task_dataloaders: List, task_names: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate model on all learned tasks.
        
        Args:
            task_dataloaders: List of dataloaders for all tasks
            task_names: Optional list of task names
            
        Returns:
            Dictionary of evaluation results
        """
        if task_names is None:
            task_names = [f"Task_{i}" for i in range(len(task_dataloaders))]
        
        results = {}
        all_accuracies = []
        
        for i, (dataloader, name) in enumerate(zip(task_dataloaders, task_names)):
            metrics = self.learner.evaluate_task(dataloader, task_id=i)
            results[name] = metrics
            all_accuracies.append(metrics['accuracy'])
        
        # Compute summary statistics
        results['summary'] = {
            'average_accuracy': np.mean(all_accuracies),
            'accuracy_std': np.std(all_accuracies),
            'min_accuracy': min(all_accuracies),
            'max_accuracy': max(all_accuracies),
            'num_tasks': len(all_accuracies)
        }
        
        # Compute forgetting if applicable
        if len(task_dataloaders) > 1:
            forgetting_metrics = self.learner.compute_forgetting(task_dataloaders)
            results['forgetting'] = forgetting_metrics
        
        self.evaluation_history.append(results)
        
        logger.info(f"Evaluation summary: Avg Acc={results['summary']['average_accuracy']:.2f}%, "
                   f"Forgetting={results.get('forgetting', {}).get('average_forgetting', 0.0):.2f}%")
        
        return results
    
    def save_checkpoint(self, filepath: str):
        """Save continual learning checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'algorithm': self.algorithm,
            'training_history': self.training_history,
            'evaluation_history': self.evaluation_history,
            'learner_state': {
                'task_history': self.learner.task_history,
                'current_task': self.learner.current_task
            }
        }
        
        # Save EWC-specific state
        if hasattr(self.learner, 'ewc_loss'):
            checkpoint['ewc_state'] = {
                'fisher_information': self.learner.ewc_loss.fisher_information,
                'optimal_params': self.learner.ewc_loss.optimal_params
            }
            if hasattr(self.learner.ewc_loss, 'task_count'):
                checkpoint['ewc_state']['task_count'] = self.learner.ewc_loss.task_count
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved continual learning checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load continual learning checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint['training_history']
        self.evaluation_history = checkpoint['evaluation_history']
        
        # Restore learner state
        if 'learner_state' in checkpoint:
            self.learner.task_history = checkpoint['learner_state']['task_history']
            self.learner.current_task = checkpoint['learner_state']['current_task']
        
        # Restore EWC state
        if 'ewc_state' in checkpoint and hasattr(self.learner, 'ewc_loss'):
            self.learner.ewc_loss.fisher_information = checkpoint['ewc_state']['fisher_information']
            self.learner.ewc_loss.optimal_params = checkpoint['ewc_state']['optimal_params']
            if 'task_count' in checkpoint['ewc_state']:
                self.learner.ewc_loss.task_count = checkpoint['ewc_state']['task_count']
        
        logger.info(f"Loaded continual learning checkpoint from {filepath}")


# Utility functions for continual learning
def compute_plasticity_stability_dilemma(accuracies_over_time: List[List[float]]) -> Dict[str, float]:
    """
    Compute metrics related to the plasticity-stability dilemma.
    
    Args:
        accuracies_over_time: List of accuracy lists, where each inner list
                             contains accuracies for all tasks at a given time
    
    Returns:
        Dictionary of plasticity and stability metrics
    """
    if len(accuracies_over_time) < 2:
        return {'plasticity': 0.0, 'stability': 0.0, 'balance': 0.0}
    
    # Plasticity: ability to learn new tasks
    final_accuracies = accuracies_over_time[-1]
    plasticity = np.mean(final_accuracies)
    
    # Stability: ability to retain old knowledge
    forgetting_per_task = []
    for task_idx in range(len(accuracies_over_time[0]) - 1):  # Exclude last task
        initial_acc = accuracies_over_time[task_idx + 1][task_idx]  # Accuracy when task was learned
        final_acc = accuracies_over_time[-1][task_idx]  # Final accuracy
        forgetting = initial_acc - final_acc
        forgetting_per_task.append(forgetting)
    
    stability = 100.0 - np.mean(forgetting_per_task) if forgetting_per_task else 100.0
    
    # Balance between plasticity and stability
    balance = 2 * (plasticity * stability) / (plasticity + stability) if (plasticity + stability) > 0 else 0.0
    
    return {
        'plasticity': plasticity,
        'stability': stability,
        'balance': balance,
        'average_forgetting': np.mean(forgetting_per_task) if forgetting_per_task else 0.0
    }