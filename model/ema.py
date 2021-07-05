"""Pytorch EMA

Implementation class for Exponential Moving Averages(EMA).

Based on Tensorflow Implementation:
https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/training/moving_averages.py 
"""
import copy
import torch
from typing import Iterable


class EMA:
    """
    Implementation of Exponential Moving Averages(EMA) for model parameters.

    Args:
        parameters: model parameters.
                    Iterable(torch.nn.Parameter)
        decay_rate: decay rate for moving average.
                    Value is between 0. and 1.0
        num_updates: Number of updates to adjust decay_rate.
    """
    def __init__(self, parameters: Iterable[torch.nn.Parameter],
                 decay_rate: float,
                 num_updates: int = None) -> None:
        assert 0.0 <= decay_rate <= 1.0, \
               "Decay rate should be in range [0, 1]"
        parameters = list(parameters)
        self.decay_rate = decay_rate
        self.num_updates = num_updates
        self.shadow_params = [p.clone().detach() for p in parameters
                              if p.requires_grad]
        self.saved_params = parameters

    def update(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Update the EMA of parametes with current decay rate.
        """
        self.num_updates += 1
        if self.num_updates is not None:
            decay_rate = min(self.decay_rate,
                             (1+self.num_updates)/(10+self.num_updates))
        else:
            decay_rate = self.decay_rate

        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for shadow, param in zip(self.shadow_params, parameters):
                tmp = shadow - param
                tmp.mul_(1-decay_rate)
                shadow.sub_(tmp)

    def store(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Save the model parameters.
        """
        parameters = list(parameters)
        self.saved_params = [p.clone().detach()
                             for p in parameters
                             if p.requires_grad]

    def copy(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy the EMA parmeters to model parameters.
        """
        parameters = list(parameters)
        if len(parameters) != len(self.shadow_params):
            raise ValueError(
                "Number of parameters passed and number of shadow "
                "parameters does not match."
            )
        for param, shadow in zip(parameters, self.shadow_params):
            if param.requires_grad:
                param.data.copy_(shadow.data)

    def copy_back(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """
        Copy the saved parameters to model parameters.
        """
        parameters = list(parameters)
        if self.saved_params is None:
            raise ValueError(
                "No saved parameters found."
            )

        if len(parameters) != len(self.saved_params):
            raise ValueError(
                "Number of parameters does not match with "
                "number of saved parameters."
            )

        for saved_param, param in zip(self.saved_params, parameters):
            if param.requires_grad:
                param.data.copy_(saved_param.data)

    def to(self, device='cpu', dtype=None) -> None:
        self.shadow_params = [p.to(device=device, dtype=dtype) for p
                              in self.shadow_params]
        if self.saved_params is not None:
            self.saved_params = [p.to(device=device, dtype=dtype) for p
                                 in self.saved_params]

    def state_dict(self) -> dict:
        return {
            "decay_rate": self.decay_rate,
            "num_updates": self.num_updates,
            "shadow_params": self.shadow_params,
            "saved_params": self.saved_params
        }

    def load_state_dict(self, state_dict: dict) -> None:
        state_dict = copy.deepcopy(state_dict)
        self.decay_rate = state_dict['decay_rate']
        assert 0.0 <= self.decay_rate <= 1.0, \
               "Decay rate should be in range [0, 1]"

        self.num_updates = state_dict['num_updates']
        assert self.num_updates is None or isinstance(self.num_updates, int),\
            "num updates should either be None or int."

        def validate_params(params1, params2=None):
            assert isinstance(params1, list), "Parameters must be list."
            for param in params1:
                assert isinstance(param, torch.Tensor), \
                    "Each parameter much be torch tensor."

            if params2 is not None:
                if len(params1) != len(params2):
                    raise ValueError("Parameter length mismatch.")

        self.shadow_params = state_dict['shadow_params']
        validate_params(self.shadow_params)

        self.saved_params = state_dict['saved_params']
        validate_params(self.saved_params, self.shadow_params)
