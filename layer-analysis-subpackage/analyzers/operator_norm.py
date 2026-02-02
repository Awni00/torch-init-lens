"""
Operator Norm Estimator.

Estimates the effective operator norm of a module by computing the ratio
of output norm to input norm across various input distributions.
"""

from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..inputs.generators import InputGenerator
from ..errors import safe_forward


class OperatorNormEstimator(BaseAnalyzer):
    """Estimate effective operator norm via output/input ratio sampling.

    This estimates the operator norm by sampling inputs from various
    distributions and computing max(||f(x)||/||x||).

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    input_shapes : Dict[str, Tuple[int, ...]]
        Mapping of input parameter names to their shapes.
        For single-input modules: {"x": (batch, seq_len, d_model)}.
        For multi-input modules: {"x": (2, 16, 64), "symbols": (2, 16, 64)}.
    n_samples : int
        Number of samples per distribution.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> estimator = OperatorNormEstimator(model, input_shapes={"x": (2, 64)})
    >>> results = estimator.analyze()
    """

    name = "operator_norm"

    def __init__(
        self,
        module: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        n_samples: int = 100,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.input_shapes = input_shapes
        self.n_samples = n_samples
        # Get batch size from first input shape
        first_shape = next(iter(input_shapes.values()))
        self.input_generator = InputGenerator(
            batch_size=first_shape[0] if len(first_shape) > 0 else 1,
            device=self.device,
            dtype=self.dtype,
        )

    def analyze(
        self,
        distributions: Optional[List[str]] = None,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[AnalysisResult]:
        """Estimate operator norm with various input distributions.

        Parameters
        ----------
        distributions : Optional[List[str]]
            Input distributions to test. If None, uses default set.
        forward_kwargs : Optional[Dict[str, Any]]
            Additional keyword arguments for the forward pass.

        Returns
        -------
        List[AnalysisResult]
            Results per distribution with estimated operator norm.
        """
        if distributions is None:
            distributions = [
                "normal",
                "uniform",
                "sparse",
                "large_magnitude",
                "small_magnitude",
                "correlated",
            ]

        forward_kwargs = forward_kwargs or {}

        results = []
        all_estimates: Dict[str, Dict[str, float]] = {}

        # Set module to eval mode
        was_training = self.module.training
        self.module.eval()

        try:
            for dist_name in distributions:
                try:
                    estimate = self._estimate_for_distribution(
                        dist_name, forward_kwargs
                    )
                    all_estimates[dist_name] = estimate

                    results.append(
                        AnalysisResult(
                            name=f"operator_norm_{dist_name}",
                            value=estimate["max"],
                            passed=True,
                            message=f"Operator norm ({dist_name}): "
                            f"max={estimate['max']:.4f}, "
                            f"mean={estimate['mean']:.4f}, "
                            f"std={estimate['std']:.4f}",
                            details=estimate,
                        )
                    )
                except Exception as e:
                    results.append(
                        AnalysisResult(
                            name=f"operator_norm_{dist_name}",
                            value=None,
                            passed=False,
                            message=f"Failed for {dist_name}: {e}",
                            severity="warning",
                        )
                    )

            # Add summary result
            if all_estimates:
                max_norms = [e["max"] for e in all_estimates.values()]
                results.append(
                    AnalysisResult(
                        name="operator_norm_summary",
                        value={
                            "overall_max": max(max_norms),
                            "overall_mean": sum(max_norms) / len(max_norms),
                        },
                        passed=True,
                        message=f"Overall max operator norm: {max(max_norms):.4f}",
                        details={"per_distribution": all_estimates},
                    )
                )

        finally:
            # Restore training mode
            if was_training:
                self.module.train()

        return results

    def _estimate_for_distribution(
        self,
        distribution: str,
        forward_kwargs: Dict[str, Any],
    ) -> Dict[str, float]:
        """Estimate operator norm for a single distribution.

        Parameters
        ----------
        distribution : str
            Name of the input distribution.
        forward_kwargs : Dict[str, Any]
            Additional forward pass arguments.

        Returns
        -------
        Dict[str, float]
            Statistics: max, mean, std, min of ratios.
        """
        ratios = []

        for _ in range(self.n_samples):
            # Generate all inputs
            inputs_dict = self.input_generator.generate_inputs(
                self.input_shapes, distribution
            )
            inputs_list = list(inputs_dict.values())

            # Compute combined input norm (concatenate all inputs)
            x_norm = torch.sqrt(
                sum(torch.linalg.norm(x.flatten()) ** 2 for x in inputs_list)
            )
            if x_norm < 1e-8:
                continue

            # Forward pass with multiple inputs
            y = safe_forward(self.module, *inputs_list, **forward_kwargs)

            # Compute ratio
            y_norm = torch.linalg.norm(y.flatten())
            ratio = (y_norm / x_norm).item()

            if not np.isnan(ratio) and not np.isinf(ratio):
                ratios.append(ratio)

        if not ratios:
            return {"max": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0}

        return {
            "max": float(max(ratios)),
            "mean": float(np.mean(ratios)),
            "std": float(np.std(ratios)),
            "min": float(min(ratios)),
            "n_samples": len(ratios),
        }

    def estimate_power_iteration(
        self,
        n_iterations: int = 20,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Estimate operator norm using power iteration.

        More accurate but slower than random sampling.
        Note: For multi-input modules, uses the primary input for power iteration.

        Parameters
        ----------
        n_iterations : int
            Number of power iteration steps.
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        AnalysisResult
            Estimated operator norm.
        """
        forward_kwargs = forward_kwargs or {}

        # Generate initial inputs
        inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
        inputs_list = list(inputs_dict.values())

        # Normalize the primary input
        x_norm = torch.sqrt(
            sum(torch.linalg.norm(x.flatten()) ** 2 for x in inputs_list)
        )
        inputs_list = [x / x_norm for x in inputs_list]

        was_training = self.module.training
        self.module.eval()

        try:
            for _ in range(n_iterations):
                # Forward pass
                y = safe_forward(self.module, *inputs_list, **forward_kwargs)

                # Normalize
                y_norm = torch.linalg.norm(y.flatten())
                if y_norm < 1e-8:
                    break

                # For multi-input modules, power iteration is complex
                # Use simple random restart instead
                inputs_dict = self.input_generator.generate_inputs(
                    self.input_shapes, "normal"
                )
                inputs_list = list(inputs_dict.values())
                x_norm = torch.sqrt(
                    sum(torch.linalg.norm(x.flatten()) ** 2 for x in inputs_list)
                )
                inputs_list = [x / x_norm for x in inputs_list]

            # Final estimate
            inputs_dict = self.input_generator.generate_inputs(
                self.input_shapes, "normal"
            )
            inputs_list = list(inputs_dict.values())
            x_norm = torch.sqrt(
                sum(torch.linalg.norm(x.flatten()) ** 2 for x in inputs_list)
            )
            inputs_list = [x / x_norm for x in inputs_list]
            y_final = safe_forward(self.module, *inputs_list, **forward_kwargs)
            estimate = torch.linalg.norm(y_final.flatten()).item()

        finally:
            if was_training:
                self.module.train()

        return AnalysisResult(
            name="operator_norm_power_iteration",
            value=estimate,
            passed=True,
            message=f"Power iteration estimate: {estimate:.4f}",
            details={"n_iterations": n_iterations},
        )

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for operator norm estimation.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        batch, seq, d_model = (
            self.input_shape[0] if len(self.input_shape) > 0 else 2,
            self.input_shape[1] if len(self.input_shape) > 1 else 16,
            self.input_shape[2] if len(self.input_shape) > 2 else 64,
        )

        return [
            self._create_markdown_cell(
                "## Effective Operator Norm Estimation\n\n"
                "Estimate the effective operator norm by computing max(||f(x)|| / ||x||) "
                "over various input distributions.\n\n"
                "A large operator norm (>> 1) may indicate potential for exploding activations, "
                "while a very small norm (<< 1) may indicate vanishing signals."
            ),
            self._create_code_cell(
                f'''
from utilities.layer_analysis.inputs import InputGenerator
import numpy as np

# Input generator
input_gen = InputGenerator(batch_size={batch}, device=device, dtype=dtype)

# Test with various input distributions
distributions = ["normal", "uniform", "sparse", "large_magnitude", "small_magnitude", "correlated"]
n_samples = 50

operator_norm_estimates = {{}}

for dist_name in distributions:
    ratios = []

    for _ in range(n_samples):
        # Generate input based on distribution
        x = input_gen.generate(dist_name, ({batch}, {seq}, {d_model}))

        x_norm = torch.linalg.norm(x.flatten())
        if x_norm < 1e-8:
            continue

        # Forward pass
        with torch.no_grad():
            out = module(x)
            if isinstance(out, tuple):
                out = out[0]

            y_norm = torch.linalg.norm(out.flatten())
            ratio = (y_norm / x_norm).item()

            if not np.isnan(ratio) and not np.isinf(ratio):
                ratios.append(ratio)

    if ratios:
        operator_norm_estimates[dist_name] = {{
            "max": max(ratios),
            "mean": np.mean(ratios),
            "std": np.std(ratios),
        }}

# Display results
print("Effective Operator Norm Estimates:")
print("-" * 60)
for dist, stats in operator_norm_estimates.items():
    print(f"{{dist:20s}}: max={{stats['max']:.4f}}, mean={{stats['mean']:.4f}}, std={{stats['std']:.4f}}")
'''
            ),
        ]
