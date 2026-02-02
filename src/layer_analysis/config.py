"""
Configuration for Module Initialization Analysis.

This module defines the AnalysisConfig dataclass that controls all analysis
parameters and toggles.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
import torch


@dataclass
class AnalysisConfig:
    """Configuration for module initialization analysis.

    Parameters
    ----------
    input_shapes : Dict[str, Tuple[int, ...]]
        Mapping of input parameter names to their shapes.
        For single-input modules, use {"x": (batch, seq, d_model)}.
        For multi-input modules (e.g., query/key/value), use
        {"query": (2, 8, 32), "key": (2, 8, 32), "value": (2, 8, 32)}.
    input_shape : Optional[Tuple[int, ...]]
        Backwards-compatible alias for a single input shape. If provided, it
        overrides input_shapes.
    device : str
        Device string ("cuda" or "cpu").
    dtype : str
        Data type string ("float32", "float16", "bfloat16").
    n_samples : int
        Number of samples for statistical estimation.
    input_distributions : List[str]
        Input distributions to test for operator norm estimation.

    Analysis Toggles
    ----------------
    run_parameter_norms : bool
        Run parameter norm analysis (Frobenius, operator, sup, std).
    run_operator_norm : bool
        Run effective operator norm estimation.
    run_gradient_analysis : bool
        Run gradient existence and norm checks.
    run_spectral_analysis : bool
        Run spectral analysis (eigenvalues, condition number).
    run_rank_analysis : bool
        Run effective rank analysis.
    run_lipschitz : bool
        Run Lipschitz constant estimation.
    run_activation_stats : bool
        Run activation statistics (mean/std/max).
    run_gradient_ratios : bool
        Run layer-wise gradient ratio analysis.
    run_precision_checks : bool
        Run numerical precision checks.
    run_weight_distribution : bool
        Run weight distribution visualization.
    gradient_loss_fn : str or List[str]
        Loss function(s) for gradient analysis. Options: "sum", "mse_random",
        "reconstruction", "variance". Default is "reconstruction".

    Tolerance Parameters
    --------------------
    gradient_existence_atol : float
        Absolute tolerance for gradient existence check.
    numerical_precision_atol : float
        Absolute tolerance for numerical precision checks.
    svd_tolerance : float
        Tolerance for SVD convergence and rank estimation.
    """

    # Input configuration - supports multiple named inputs
    input_shape: Optional[Tuple[int, ...]] = None
    input_shapes: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: {"x": (2, 16, 64)}
    )
    device: str = "cuda"
    dtype: str = "float32"

    # Estimation parameters
    n_samples: int = 100
    input_distributions: List[str] = field(
        default_factory=lambda: [
            "normal",
            "sparse",
            "large_magnitude",
            "small_magnitude",
            "one_hot",
            "orthogonal",
            "correlated",
            "alternating",
        ]
    )

    # Analysis toggles
    run_parameter_norms: bool = True
    run_operator_norm: bool = True
    run_gradient_analysis: bool = True
    run_spectral_analysis: bool = True
    run_rank_analysis: bool = True
    run_lipschitz: bool = True
    run_activation_stats: bool = True
    run_gradient_ratios: bool = True
    run_precision_checks: bool = True
    run_weight_distribution: bool = True

    # Gradient analysis loss function(s)
    gradient_loss_fn: Union[str, List[str]] = "reconstruction"

    # Tolerance parameters
    gradient_existence_atol: float = 1e-8
    numerical_precision_atol: float = 1e-5
    svd_tolerance: float = 1e-6

    # Spectral analysis parameters
    top_k_eigenvalues: int = 10
    max_params_for_full_svd: int = 10000

    # Lipschitz estimation parameters
    lipschitz_n_iterations: int = 20

    # Visualization parameters
    histogram_bins: int = 50
    save_plots: bool = True
    plot_format: str = "png"
    plot_dpi: int = 150

    def __post_init__(self) -> None:
        """Normalize input shape configuration.

        Returns
        -------
        None
            This method mutates the config in-place.
        """
        if self.input_shape is not None:
            self.input_shapes = {"x": self.input_shape}
        if not self.input_shapes:
            self.input_shapes = {"x": (2, 16, 64)}
        if self.input_shape is None:
            self.input_shape = next(iter(self.input_shapes.values()))

    @property
    def primary_input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the first/primary input tensor."""
        if not self.input_shapes:
            return (2, 16, 64)
        return next(iter(self.input_shapes.values()))

    @property
    def batch_size(self) -> int:
        """Extract batch size from primary input shape."""
        shape = self.primary_input_shape
        return shape[0] if len(shape) > 0 else 2

    @property
    def seq_len(self) -> int:
        """Extract sequence length from primary input shape (if applicable)."""
        shape = self.primary_input_shape
        if len(shape) >= 2:
            return shape[1]
        return 1

    @property
    def d_model(self) -> int:
        """Extract model dimension from primary input shape (if applicable)."""
        shape = self.primary_input_shape
        if len(shape) >= 3:
            return shape[2]
        elif len(shape) >= 2:
            return shape[1]
        return shape[0]

    def get_torch_device(self) -> torch.device:
        """Convert device string to torch.device.

        Returns
        -------
        torch.device
            PyTorch device object.
        """
        if self.device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(self.device)

    def get_torch_dtype(self) -> torch.dtype:
        """Convert dtype string to torch.dtype.

        Returns
        -------
        torch.dtype
            PyTorch data type.
        """
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float64": torch.float64,
        }
        return dtype_map.get(self.dtype, torch.float32)

    def get_enabled_analyses(self) -> List[str]:
        """Get list of enabled analysis types.

        Returns
        -------
        List[str]
            Names of enabled analyses.
        """
        analyses = []
        if self.run_parameter_norms:
            analyses.append("parameter_norms")
        if self.run_operator_norm:
            analyses.append("operator_norm")
        if self.run_gradient_analysis:
            analyses.append("gradient_analysis")
        if self.run_spectral_analysis:
            analyses.append("spectral_analysis")
        if self.run_rank_analysis:
            analyses.append("rank_analysis")
        if self.run_lipschitz:
            analyses.append("lipschitz")
        if self.run_activation_stats:
            analyses.append("activation_stats")
        if self.run_gradient_ratios:
            analyses.append("gradient_ratios")
        if self.run_precision_checks:
            analyses.append("precision_checks")
        if self.run_weight_distribution:
            analyses.append("weight_distribution")
        return analyses

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns
        -------
        dict
            Configuration as dictionary.
        """
        return {
            "input_shapes": {k: list(v) for k, v in self.input_shapes.items()},
            "device": self.device,
            "dtype": self.dtype,
            "n_samples": self.n_samples,
            "input_distributions": self.input_distributions,
            "enabled_analyses": self.get_enabled_analyses(),
            "gradient_loss_fn": self.gradient_loss_fn,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AnalysisConfig":
        """Create config from dictionary.

        Parameters
        ----------
        d : dict
            Configuration dictionary.

        Returns
        -------
        AnalysisConfig
            Configuration object.
        """
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
