"""
Optimizer and Learning Rate Scheduler
======================================

AdamW optimizer with cosine annealing and warmup.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

import torch
import torch.optim as optim
import math
from typing import Optional

logger = get_logger(__name__)


class CosineAnnealingWarmup:
    """
    Cosine annealing learning rate scheduler with linear warmup.
    
    Schedule:
        1. Warmup: Linear increase from 0 to max_lr (0 to warmup_iterations)
        2. Cosine annealing: Decrease from max_lr to min_lr (warmup_iterations to total_iterations)
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        min_lr: float,
        warmup_iterations: int,
        total_iterations: int,
    ):
        logger.info("Initializing CosineAnnealingWarmup scheduler")
        logger.info(f"Parameters: max_lr={max_lr}, min_lr={min_lr}, warmup_iterations={warmup_iterations}, total_iterations={total_iterations}")

        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iterations = warmup_iterations
        self.total_iterations = total_iterations
        self.current_iteration = 0

    def step(self):
        """Update learning rate."""
        self.current_iteration += 1
        lr = self._compute_lr()

        logger.debug(f"CosineAnnealingWarmup step {self.current_iteration}/{self.total_iterations}: LR = {lr:.6f}")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _compute_lr(self) -> float:
        """Compute learning rate for current iteration."""
        if self.current_iteration < self.warmup_iterations:
            # Linear warmup
            lr = self.max_lr * (self.current_iteration / self.warmup_iterations)
        else:
            # Cosine annealing
            progress = (self.current_iteration - self.warmup_iterations) / (
                self.total_iterations - self.warmup_iterations
            )
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        
        return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class StepLRWithWarmup:
    """
    Step learning rate scheduler with warmup.
    
    Decreases LR by factor at specified milestones.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        max_lr: float,
        warmup_iterations: int,
        milestones: list,
        gamma: float = 0.1,
    ):
        logger.info("Initializing StepLRWithWarmup scheduler")
        logger.info(f"Parameters: max_lr={max_lr}, warmup_iterations={warmup_iterations}, milestones={milestones}, gamma={gamma}")

        self.optimizer = optimizer
        self.max_lr = max_lr
        self.warmup_iterations = warmup_iterations
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.current_iteration = 0

    def step(self):
        """Update learning rate."""
        self.current_iteration += 1
        lr = self._compute_lr()

        logger.debug(f"StepLRWithWarmup step {self.current_iteration}: LR = {lr:.6f}")

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        """Compute learning rate."""
        if self.current_iteration < self.warmup_iterations:
            # Warmup
            return self.max_lr * (self.current_iteration / self.warmup_iterations)
        else:
            # Step decay
            lr = self.max_lr
            for milestone in self.milestones:
                if self.current_iteration >= milestone:
                    lr *= self.gamma
            return lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = 'adamw',
    learning_rate: float = 1e-3,
    weight_decay: float = 2e-4,
    betas: tuple = (0.9, 0.999),
    **kwargs
) -> optim.Optimizer:
    logger.info(f"Creating optimizer: {optimizer_name}")
    logger.info(f"Parameters: lr={learning_rate}, weight_decay={weight_decay}, betas={betas}")

    if optimizer_name.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        logger.info("Created AdamW optimizer")
    elif optimizer_name.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            weight_decay=weight_decay,
        )
        logger.info("Created Adam optimizer")
    elif optimizer_name.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        logger.info("Created SGD optimizer with momentum=0.9")
    else:
        logger.error(f"Unknown optimizer: {optimizer_name}")
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'cosine',
    max_lr: Optional[float] = None,
    min_lr: float = 1e-6,
    warmup_iterations: int = 2000,
    total_iterations: int = 100000,
    milestones: Optional[list] = None,
    **kwargs
) -> object:
    if max_lr is None:
        max_lr = optimizer.param_groups[0]['lr']

    logger.info(f"Creating scheduler: {scheduler_name}")

    if scheduler_name.lower() == 'cosine':
        scheduler = CosineAnnealingWarmup(
            optimizer=optimizer,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_iterations=warmup_iterations,
            total_iterations=total_iterations,
        )
        logger.info("Created CosineAnnealingWarmup scheduler")
    elif scheduler_name.lower() == 'step':
        if milestones is None:
            milestones = [60000, 80000]
        scheduler = StepLRWithWarmup(
            optimizer=optimizer,
            max_lr=max_lr,
            warmup_iterations=warmup_iterations,
            milestones=milestones,
        )
        logger.info("Created StepLRWithWarmup scheduler")
    elif scheduler_name.lower() == 'none':
        # No scheduling
        logger.info("No scheduler created (scheduler_name='none')")
        scheduler = None
    else:
        logger.error(f"Unknown scheduler: {scheduler_name}")
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


class GradientClipper:
    def __init__(
        self,
        max_norm: float = 3.0,
        norm_type: float = 2.0,
    ):
        logger.info(f"Initializing GradientClipper with max_norm={max_norm}, norm_type={norm_type}")
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: torch.nn.Module) -> float:
        parameters = [p for p in model.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        # clip_grad_norm_ returns the total norm BEFORE clipping (as a tensor).
        # Using its return value avoids computing the norm twice (which is
        # what the previous version did — manual norm + clip_grad_norm_).
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.max_norm, self.norm_type
        )

        # Convert to Python float (single sync, only once per step).
        if isinstance(total_norm, torch.Tensor):
            grad_norm = total_norm.item()
        else:
            grad_norm = float(total_norm)

        return grad_norm


class OptimizerState:
    """
    Optimizer state manager for checkpointing.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        scheduler: Optional[object] = None,
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        state = {
            'optimizer': self.optimizer.state_dict(),
        }
        
        if self.scheduler is not None:
            state['scheduler'] = {
                'current_iteration': self.scheduler.current_iteration,
            }
        
        return state
    
    def load_state_dict(self, state: dict):
        """Load state from checkpoint."""
        self.optimizer.load_state_dict(state['optimizer'])
        
        if self.scheduler is not None and 'scheduler' in state:
            self.scheduler.current_iteration = state['scheduler']['current_iteration']


def get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float = 2e-4,
    no_decay_bias: bool = True,
) -> list:
    if not no_decay_bias:
        return [{'params': model.parameters(), 'weight_decay': weight_decay}]
    
    # Separate parameters
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # No decay for bias and batch norm
        if 'bias' in name or 'bn' in name or 'norm' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    parameter_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    return parameter_groups