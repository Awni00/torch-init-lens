"""
Input Tensor Generators for Module Analysis.

This module provides utilities for generating test input tensors with
various distributions for module initialization analysis.
"""

from typing import Dict, Callable, Optional, Tuple, List, OrderedDict
from collections import OrderedDict as ODict
import torch
import torch.nn.functional as F


class InputGenerator:
    """Factory for creating test input tensors.

    Provides methods for generating input tensors from various distributions
    to test module behavior under different input conditions.

    Parameters
    ----------
    batch_size : int
        Default batch size for generated tensors.
    device : torch.device
        Device for generated tensors.
    dtype : torch.dtype
        Data type for generated tensors.
    seed : Optional[int]
        Random seed for reproducibility.

    Examples
    --------
    >>> gen = InputGenerator(batch_size=2, device=torch.device("cuda"))
    >>> x = gen.make_normal(2, 16, 64)
    >>> x.shape
    torch.Size([2, 16, 64])
    """

    def __init__(
        self,
        batch_size: int = 2,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
    ):
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")
        self.dtype = dtype

        if seed is not None:
            torch.manual_seed(seed)

    def make_normal(
        self,
        *shape: int,
        mean: float = 0.0,
        std: float = 1.0,
    ) -> torch.Tensor:
        """Generate tensor from standard normal distribution.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.
        mean : float
            Mean of the distribution.
        std : float
            Standard deviation of the distribution.

        Returns
        -------
        torch.Tensor
            Random normal tensor.
        """
        x = torch.randn(*shape, device=self.device, dtype=self.dtype)
        return x * std + mean

    def make_uniform(
        self,
        *shape: int,
        low: float = -1.0,
        high: float = 1.0,
    ) -> torch.Tensor:
        """Generate tensor from uniform distribution.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.
        low : float
            Lower bound of the distribution.
        high : float
            Upper bound of the distribution.

        Returns
        -------
        torch.Tensor
            Random uniform tensor.
        """
        x = torch.rand(*shape, device=self.device, dtype=self.dtype)
        return x * (high - low) + low

    def make_sparse(
        self,
        *shape: int,
        sparsity: float = 0.9,
    ) -> torch.Tensor:
        """Generate sparse tensor with given sparsity level.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.
        sparsity : float
            Fraction of zeros (0.9 = 90% zeros).

        Returns
        -------
        torch.Tensor
            Sparse tensor with approximately (1-sparsity) fraction of non-zeros.
        """
        x = torch.randn(*shape, device=self.device, dtype=self.dtype)
        mask = torch.rand(*shape, device=self.device) > sparsity
        return x * mask

    def make_large_magnitude(
        self,
        *shape: int,
        scale: float = 1000.0,
    ) -> torch.Tensor:
        """Generate tensor with large magnitude values.

        Useful for testing numerical stability with large inputs.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.
        scale : float
            Scaling factor for the values.

        Returns
        -------
        torch.Tensor
            Large magnitude random tensor.
        """
        return torch.randn(*shape, device=self.device, dtype=self.dtype) * scale

    def make_small_magnitude(
        self,
        *shape: int,
        scale: float = 1e-4,
    ) -> torch.Tensor:
        """Generate tensor with small magnitude values.

        Useful for testing numerical precision with small inputs.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.
        scale : float
            Scaling factor for the values.

        Returns
        -------
        torch.Tensor
            Small magnitude random tensor.
        """
        return torch.randn(*shape, device=self.device, dtype=self.dtype) * scale

    def make_one_hot(
        self,
        batch_size: int,
        seq_len: int,
        n_classes: int,
    ) -> torch.Tensor:
        """Generate one-hot encoded patterns.

        Parameters
        ----------
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        n_classes : int
            Number of classes (embedding dimension).

        Returns
        -------
        torch.Tensor
            One-hot encoded tensor of shape (batch_size, seq_len, n_classes).
        """
        indices = torch.randint(
            0,
            n_classes,
            (batch_size, seq_len),
            device=self.device,
        )
        return F.one_hot(indices, n_classes).to(self.dtype)

    def make_orthogonal(
        self,
        batch_size: int,
        dim: int,
        n_vectors: int = 1,
    ) -> torch.Tensor:
        """Generate orthogonal vectors via QR decomposition.

        Parameters
        ----------
        batch_size : int
            Batch size.
        dim : int
            Dimension of each vector.
        n_vectors : int
            Number of orthogonal vectors per batch.

        Returns
        -------
        torch.Tensor
            Orthogonal vectors of shape (batch_size, n_vectors, dim) or
            (batch_size, dim) if n_vectors=1.
        """
        # Generate random matrices
        if n_vectors >= dim:
            n_vectors = dim

        x = torch.randn(batch_size, dim, dim, device=self.device, dtype=self.dtype)
        q, _ = torch.linalg.qr(x)

        if n_vectors == 1:
            return q[:, :, 0]  # (batch_size, dim)
        return q[:, :, :n_vectors].transpose(1, 2)  # (batch_size, n_vectors, dim)

    def make_correlated(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        rank: int = 4,
    ) -> torch.Tensor:
        """Generate low-rank correlated tensor.

        Produces tensors where features are correlated through a low-rank
        projection, useful for testing behavior with structured inputs.

        Parameters
        ----------
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        d_model : int
            Model dimension.
        rank : int
            Rank of the correlation structure.

        Returns
        -------
        torch.Tensor
            Low-rank correlated tensor of shape (batch_size, seq_len, d_model).
        """
        base = torch.randn(
            batch_size,
            seq_len,
            rank,
            device=self.device,
            dtype=self.dtype,
        )
        proj = torch.randn(rank, d_model, device=self.device, dtype=self.dtype)
        return base @ proj

    def make_alternating(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
    ) -> torch.Tensor:
        """Generate tensor with alternating sign pattern.

        Useful for testing handling of high-frequency patterns.

        Parameters
        ----------
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        d_model : int
            Model dimension.

        Returns
        -------
        torch.Tensor
            Tensor with alternating signs along sequence dimension.
        """
        x = torch.randn(
            batch_size,
            seq_len,
            d_model,
            device=self.device,
            dtype=self.dtype,
        )
        signs = torch.tensor(
            [1, -1] * ((seq_len + 1) // 2),
            device=self.device,
            dtype=self.dtype,
        )[:seq_len]
        return x * signs.view(1, -1, 1)

    def make_identity_like(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
    ) -> torch.Tensor:
        """Generate tensor with diagonal/identity-like structure.

        Creates patterns where each position has a distinct signature,
        useful for testing positional handling.

        Parameters
        ----------
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        d_model : int
            Model dimension.

        Returns
        -------
        torch.Tensor
            Tensor with diagonal patterns per position.
        """
        x = torch.zeros(
            batch_size,
            seq_len,
            d_model,
            device=self.device,
            dtype=self.dtype,
        )
        for i in range(min(seq_len, d_model)):
            x[:, i, i] = 1.0
        return x

    def make_constant(
        self,
        *shape: int,
        value: float = 1.0,
    ) -> torch.Tensor:
        """Generate tensor with constant values.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.
        value : float
            Constant value to fill.

        Returns
        -------
        torch.Tensor
            Constant-filled tensor.
        """
        return torch.full(shape, value, device=self.device, dtype=self.dtype)

    def make_zeros(self, *shape: int) -> torch.Tensor:
        """Generate tensor of zeros.

        Parameters
        ----------
        *shape : int
            Shape of the tensor.

        Returns
        -------
        torch.Tensor
            Zero tensor.
        """
        return torch.zeros(*shape, device=self.device, dtype=self.dtype)

    def make_arange(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Generate tensor with sequential values.

        Parameters
        ----------
        batch_size : int
            Batch size.
        seq_len : int
            Sequence length.
        d_model : int
            Model dimension.
        normalize : bool
            If True, normalize values to [0, 1] range.

        Returns
        -------
        torch.Tensor
            Tensor with sequential values.
        """
        x = torch.arange(
            seq_len * d_model,
            device=self.device,
            dtype=self.dtype,
        ).reshape(1, seq_len, d_model)
        x = x.expand(batch_size, -1, -1).clone()

        if normalize:
            x = x / (seq_len * d_model)

        return x

    def generate(
        self,
        distribution: str,
        shape: Tuple[int, ...],
        **kwargs,
    ) -> torch.Tensor:
        """Generate tensor from specified distribution.

        Parameters
        ----------
        distribution : str
            Name of the distribution. One of:
            "normal", "uniform", "sparse", "large_magnitude", "small_magnitude",
            "one_hot", "orthogonal", "correlated", "alternating", "identity_like",
            "constant", "zeros", "arange".
        shape : Tuple[int, ...]
            Shape of the tensor.
        **kwargs
            Additional arguments for the generator method.

        Returns
        -------
        torch.Tensor
            Generated tensor.

        Raises
        ------
        ValueError
            If distribution name is not recognized.
        """
        generator_map = self.get_generator_map()

        if distribution not in generator_map:
            raise ValueError(
                f"Unknown distribution: {distribution}. "
                f"Available: {list(generator_map.keys())}"
            )

        generator = generator_map[distribution]

        # Handle different generator signatures
        if distribution in ["one_hot", "orthogonal", "correlated", "alternating"]:
            if len(shape) >= 3:
                return generator(shape[0], shape[1], shape[2], **kwargs)
            elif len(shape) == 2:
                return generator(shape[0], shape[1], **kwargs)
        elif distribution == "identity_like":
            return generator(shape[0], shape[1], shape[2], **kwargs)
        elif distribution == "arange":
            return generator(shape[0], shape[1], shape[2], **kwargs)

        return generator(*shape, **kwargs)

    def get_generator_map(self) -> Dict[str, Callable]:
        """Get mapping of distribution names to generator methods.

        Returns
        -------
        Dict[str, Callable]
            Dictionary mapping distribution names to methods.
        """
        return {
            "normal": self.make_normal,
            "uniform": self.make_uniform,
            "sparse": self.make_sparse,
            "large_magnitude": self.make_large_magnitude,
            "small_magnitude": self.make_small_magnitude,
            "one_hot": self.make_one_hot,
            "orthogonal": self.make_orthogonal,
            "correlated": self.make_correlated,
            "alternating": self.make_alternating,
            "identity_like": self.make_identity_like,
            "constant": self.make_constant,
            "zeros": self.make_zeros,
            "arange": self.make_arange,
        }

    @classmethod
    def get_available_distributions(cls) -> List[str]:
        """Get list of available distribution names.

        Returns
        -------
        List[str]
            Names of available distributions.
        """
        return [
            "normal",
            "uniform",
            "sparse",
            "large_magnitude",
            "small_magnitude",
            "one_hot",
            "orthogonal",
            "correlated",
            "alternating",
            "identity_like",
            "constant",
            "zeros",
            "arange",
        ]

    def generate_inputs(
        self,
        input_shapes: Dict[str, Tuple[int, ...]],
        distribution: str = "normal",
        **kwargs,
    ) -> "ODict[str, torch.Tensor]":
        """Generate multiple named input tensors.

        Parameters
        ----------
        input_shapes : Dict[str, Tuple[int, ...]]
            Mapping of parameter names to shapes.
        distribution : str
            Distribution to use for all tensors.
        **kwargs
            Additional arguments for the generator method.

        Returns
        -------
        OrderedDict[str, torch.Tensor]
            Mapping of parameter names to generated tensors,
            preserving order for positional argument unpacking.

        Examples
        --------
        >>> gen = InputGenerator(device=torch.device("cuda"))
        >>> inputs = gen.generate_inputs(
        ...     {"x": (2, 16, 64), "symbols": (2, 16, 64)},
        ...     distribution="normal"
        ... )
        >>> inputs["x"].shape
        torch.Size([2, 16, 64])
        """
        result: ODict[str, torch.Tensor] = ODict()
        for name, shape in input_shapes.items():
            result[name] = self.generate(distribution, shape, **kwargs)
        return result
