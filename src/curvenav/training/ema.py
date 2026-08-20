"""Exponential moving average with one fused update over model parameters."""

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        named_parameters = tuple(
            (name, parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        self.decay = decay
        self.num_updates = 0
        self._parameter_names = tuple(name for name, _ in named_parameters)
        self._parameters = tuple(parameter for _, parameter in named_parameters)
        self.shadow = {
            name: parameter.clone() for name, parameter in named_parameters
        }

    @torch.no_grad()
    def update(self) -> None:
        self.num_updates += 1
        effective_decay = min(
            self.decay,
            (1.0 + self.num_updates) / (10.0 + self.num_updates),
        )
        torch._foreach_lerp_(
            tuple(self.shadow.values()),
            self._parameters,
            1.0 - effective_decay,
        )

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        named_parameters = tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        if tuple(name for name, _ in named_parameters) != self._parameter_names:
            raise ValueError("EMA state does not match model parameters")
        for name, parameter in named_parameters:
            parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        decay = float(state["decay"])
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("EMA shadow state must be a dictionary")
        if tuple(shadow) != self._parameter_names:
            raise ValueError("EMA state does not match model parameters")
        self.decay = decay
        self.num_updates = int(state["num_updates"])
        self.shadow = {
            name: value.detach().to(device=parameter.device, dtype=parameter.dtype).clone()
            for (name, value), parameter in zip(shadow.items(), self._parameters)
        }
