"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim
        self.layers = []
        for dim in hidden_dims:
            x = nn.Linear(in_features=input_dim, out_features=dim)
            self.layers.append(x)
            activation = nn.ReLU()
            self.layers.append(activation)
            input_dim = dim
        out = nn.Linear(in_features=input_dim, out_features=chunk_size * action_dim)
        self.layers.append(out)
        self.net = nn.Sequential(*self.layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        action_result = self.net(state)
        result = action_result.reshape(action_chunk.shape)
        loss = torch.sum((result - action_chunk) ** 2) / action_chunk.shape[0]
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        action_result = self.net(state)
        result = action_result.reshape(state.shape[0], self.chunk_size, self.action_dim)
        return result


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        input_dim = state_dim + (chunk_size * action_dim) + 1
        self.layers = []
        for dim in hidden_dims:
            x = nn.Linear(in_features=input_dim, out_features=dim)
            self.layers.append(x)
            activation = nn.ReLU()
            self.layers.append(activation)
            input_dim = dim
        out = nn.Linear(in_features=input_dim, out_features=chunk_size * action_dim)
        self.layers.append(out)
        self.net = nn.Sequential(*self.layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        noise = torch.randn_like(action_chunk)
        tau = torch.rand(action_chunk.shape[0], 1, device=action_chunk.device)
        tau = tau.unsqueeze(-1)
        noisy_action = tau * action_chunk + (1 - tau) * noise
        target_velocity = action_chunk - noise
        noisy_action = noisy_action.reshape(
            noisy_action.shape[0], noisy_action.shape[-1] * noisy_action.shape[-2]
        )
        tau = tau.reshape(tau.shape[0], -1)
        inputs = torch.concat([state, noisy_action, tau], dim=1)
        predicted_velocity = self.net(inputs)
        result = predicted_velocity.reshape(
            state.shape[0], self.chunk_size, self.action_dim
        )
        loss = torch.sum((result - target_velocity) ** 2) / action_chunk.shape[0]
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        noise = torch.randn(
            state.shape[0], self.chunk_size, self.action_dim, device=state.device
        )
        delta = 1 / num_steps
        u = 0
        noise = noise.reshape(state.shape[0], self.chunk_size * self.action_dim)
        for _ in range(num_steps):
            tau = torch.tensor(u, device=state.device).reshape(state.shape[0], 1)
            inputs = torch.concat([state, noise, tau], dim=1)
            pred_v = self.net(inputs)
            u = u + delta
            noise = noise + delta * pred_v
        result = noise.reshape(state.shape[0], self.chunk_size, self.action_dim)
        return result


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
