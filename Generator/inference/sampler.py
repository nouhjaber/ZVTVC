"""
ODE Samplers for Flow Matching Inference
Implements different numerical solvers for the ODE: dx/dt = v(x, t, conditions)
"""

import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod


class ODESampler(ABC):
    """
    Base class for ODE samplers.

    Solves: dx/dt = v(x, t, conditions)
    From t=0 (noise) to t=1 (data)
    """

    def __init__(self, num_steps: int = 10):
        """
        Args:
            num_steps: Number of discretization steps
        """
        self.num_steps = num_steps

    @abstractmethod
    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: float,
        dt: float,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        pass

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        if device is None:
            device = content.device

        # Initialize with Gaussian noise
        x = torch.randn(shape, device=device)

        # Time step
        dt = 1.0 / self.num_steps

        # Solve ODE from t=0 to t=1
        for i in range(self.num_steps):
            t = i / self.num_steps

            # Take ODE step
            x = self.step(
                model=model,
                x=x,
                t=t,
                dt=dt,
                content=content,
                prosody=prosody,
                timbre=timbre
            )

        return x


class EulerSampler(ODESampler):
    """
    Euler method (first-order).

    x_{n+1} = x_n + dt * v(x_n, t_n)

    Simple and fast, good enough for flow matching with 10 steps.
    """

    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: float,
        dt: float,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        """Euler step"""
        batch_size = x.shape[0]

        # Create time tensor
        t_tensor = torch.full((batch_size,), t, device=x.device)

        # Predict velocity
        v = model(x, t_tensor, content, prosody, timbre)

        # Euler update
        x_next = x + dt * v

        return x_next


class MidpointSampler(ODESampler):
    """
    Midpoint method (second-order, RK2).

    k1 = v(x_n, t_n)
    k2 = v(x_n + 0.5*dt*k1, t_n + 0.5*dt)
    x_{n+1} = x_n + dt * k2

    More accurate than Euler, but requires 2 model calls per step.
    Good alternative: 5 midpoint steps ≈ 10 Euler steps in quality.
    """

    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: float,
        dt: float,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        """Midpoint step"""
        batch_size = x.shape[0]

        # First evaluation at t_n
        t_tensor = torch.full((batch_size,), t, device=x.device)
        k1 = model(x, t_tensor, content, prosody, timbre)

        # Second evaluation at t_n + 0.5*dt
        x_mid = x + 0.5 * dt * k1
        t_mid = t + 0.5 * dt
        t_mid_tensor = torch.full((batch_size,), t_mid, device=x.device)
        k2 = model(x_mid, t_mid_tensor, content, prosody, timbre)

        # Update using midpoint
        x_next = x + dt * k2

        return x_next


class HeunSampler(ODESampler):
    """
    Heun's method (second-order, RK2 variant).

    k1 = v(x_n, t_n)
    k2 = v(x_n + dt*k1, t_n + dt)
    x_{n+1} = x_n + 0.5*dt*(k1 + k2)

    Also called improved Euler method.
    Similar accuracy to midpoint, different evaluation points.
    """

    def step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: float,
        dt: float,
        content: torch.Tensor,
        prosody: torch.Tensor,
        timbre: torch.Tensor
    ) -> torch.Tensor:
        """Heun step"""
        batch_size = x.shape[0]

        # First evaluation at t_n
        t_tensor = torch.full((batch_size,), t, device=x.device)
        k1 = model(x, t_tensor, content, prosody, timbre)

        # Second evaluation at t_n + dt
        x_next_euler = x + dt * k1
        t_next = t + dt
        t_next = min(t_next, 1.0)  # Clamp to [0, 1]
        t_next_tensor = torch.full((batch_size,), t_next, device=x.device)
        k2 = model(x_next_euler, t_next_tensor, content, prosody, timbre)

        # Heun update (average of slopes)
        x_next = x + 0.5 * dt * (k1 + k2)

        return x_next


def get_sampler(method: str = 'euler', num_steps: int = 10) -> ODESampler:
    method = method.lower()

    if method == 'euler':
        return EulerSampler(num_steps=num_steps)
    elif method == 'midpoint':
        return MidpointSampler(num_steps=num_steps)
    elif method == 'heun':
        return HeunSampler(num_steps=num_steps)
    else:
        raise ValueError(f"Unknown sampler method: {method}. Choose from: euler, midpoint, heun")