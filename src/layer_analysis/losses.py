"""
Loss Functions for Gradient Analysis.

This module provides configurable loss functions for gradient analysis,
allowing different gradient behaviors to be tested.
"""

from abc import ABC, abstractmethod
from typing import Union, List

import torch
import torch.nn.functional as F


class GradientLoss(ABC):
    """Base class for gradient analysis loss functions.

    Parameters
    ----------
    None

    Attributes
    ----------
    name : str
        Name of this loss function.

    Notes
    -----
    Subclasses must implement the `name` property and `__call__` method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this loss function."""
        pass

    @abstractmethod
    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Compute loss given output and input tensors.

        Parameters
        ----------
        output : torch.Tensor or tuple
            Model output tensor(s).
        input : torch.Tensor
            Model input tensor.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        pass


class SumLoss(GradientLoss):
    """Sum of all outputs (original behavior).

    This ensures all output elements contribute to gradients, but produces
    only positive gradients which may not test negative gradient flow.
    """

    @property
    def name(self) -> str:
        return "sum"

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Compute sum of all output elements.

        Parameters
        ----------
        output : torch.Tensor or tuple
            Model output tensor(s).
        input : torch.Tensor
            Model input tensor (unused).

        Returns
        -------
        torch.Tensor
            Sum of all outputs.
        """
        if isinstance(output, tuple):
            total = 0
            for o in output:
                if isinstance(o, torch.Tensor):
                    total = total + o.sum()
            return total
        return output.sum()


class MSERandomTargetLoss(GradientLoss):
    """MSE to random target - balanced +/- gradient signals.

    This produces both positive and negative gradients, providing
    a more realistic training scenario.
    """

    @property
    def name(self) -> str:
        return "mse_random"

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss to a random target.

        Parameters
        ----------
        output : torch.Tensor or tuple
            Model output tensor(s).
        input : torch.Tensor
            Model input tensor (unused for target generation).

        Returns
        -------
        torch.Tensor
            MSE loss value.
        """
        if isinstance(output, tuple):
            output = output[0]
        target = torch.randn_like(output)
        return F.mse_loss(output, target)


class ReconstructionLoss(GradientLoss):
    """MSE(output, input) - tests identity-mapping capability.

    This loss is useful for testing whether a layer can learn to
    approximate an identity mapping, which is important for residual
    connections and stable training.
    """

    @property
    def name(self) -> str:
        return "reconstruction"

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Compute MSE loss between output and input.

        Parameters
        ----------
        output : torch.Tensor or tuple
            Model output tensor(s).
        input : torch.Tensor
            Model input tensor (used as target).

        Returns
        -------
        torch.Tensor
            MSE reconstruction loss, or sum loss if shapes mismatch.
        """
        if isinstance(output, tuple):
            output = output[0]
        if output.shape == input.shape:
            return F.mse_loss(output, input)
        # Fallback: use sum if shapes don't match
        return output.sum()


class OutputVarianceLoss(GradientLoss):
    """Negative output variance - detects representational collapse.

    This loss encourages the model to produce diverse outputs.
    A negative variance is used so that maximizing variance
    corresponds to minimizing the loss.

    Notes
    -----
    If the output variance is very small, this may indicate
    representational collapse where the model produces nearly
    constant outputs.
    """

    @property
    def name(self) -> str:
        return "variance"

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Compute negative variance of output.

        Parameters
        ----------
        output : torch.Tensor or tuple
            Model output tensor(s).
        input : torch.Tensor
            Model input tensor (unused).

        Returns
        -------
        torch.Tensor
            Negative variance of output.
        """
        if isinstance(output, tuple):
            output = output[0]
        return -output.var()


# Registry for CLI lookup
LOSS_REGISTRY = {
    "sum": SumLoss,
    "mse_random": MSERandomTargetLoss,
    "reconstruction": ReconstructionLoss,
    "variance": OutputVarianceLoss,
}

DEFAULT_LOSS = "reconstruction"


def get_loss(name: str) -> GradientLoss:
    """Get loss function by name.

    Parameters
    ----------
    name : str
        Name of the loss function.

    Returns
    -------
    GradientLoss
        Instantiated loss function.

    Raises
    ------
    ValueError
        If the loss name is not in the registry.

    Examples
    --------
    >>> loss = get_loss("reconstruction")
    >>> loss.name
    'reconstruction'
    """
    if name not in LOSS_REGISTRY:
        valid = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Valid options: {valid}")
    return LOSS_REGISTRY[name]()


def get_losses(names: Union[str, List[str]]) -> List[GradientLoss]:
    """Get one or more loss functions by name(s).

    Parameters
    ----------
    names : str or List[str]
        Loss function name(s).

    Returns
    -------
    List[GradientLoss]
        List of instantiated loss functions.

    Examples
    --------
    >>> losses = get_losses(["sum", "reconstruction"])
    >>> [l.name for l in losses]
    ['sum', 'reconstruction']
    """
    if isinstance(names, str):
        names = [names]
    return [get_loss(name) for name in names]


def list_losses() -> List[str]:
    """List all available loss function names.

    Returns
    -------
    List[str]
        Names of available loss functions.
    """
    return list(LOSS_REGISTRY.keys())
