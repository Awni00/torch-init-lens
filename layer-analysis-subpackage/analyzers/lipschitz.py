"""
Lipschitz Constant Estimator.

Estimates the Lipschitz constant of a module, which bounds how much
the output can change relative to input perturbations.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..inputs.generators import InputGenerator
from ..errors import safe_forward


class LipschitzEstimator(BaseAnalyzer):
    """Estimate Lipschitz constant of a module.

    The Lipschitz constant L bounds the output change:
    ||f(x) - f(y)|| <= L * ||x - y||

    A large Lipschitz constant indicates high sensitivity to input
    perturbations, which can affect training stability.

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    input_shapes : Dict[str, Tuple[int, ...]]
        Dictionary mapping input names to their shapes.
    n_iterations : int
        Number of iterations for power iteration estimation.
    n_samples : int
        Number of random samples for empirical estimation.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> estimator = LipschitzEstimator(model, input_shapes={"x": (2, 64)})
    >>> results = estimator.analyze()
    """

    name = "lipschitz"

    def __init__(
        self,
        module: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        n_iterations: int = 20,
        n_samples: int = 50,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.input_shapes = input_shapes
        self.n_iterations = n_iterations
        self.n_samples = n_samples
        first_shape = next(iter(input_shapes.values()))
        self.input_generator = InputGenerator(
            batch_size=first_shape[0] if len(first_shape) > 0 else 1,
            device=self.device,
            dtype=self.dtype,
        )

    def analyze(
        self,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[AnalysisResult]:
        """Estimate Lipschitz constant.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        List[AnalysisResult]
            Lipschitz constant estimates.
        """
        forward_kwargs = forward_kwargs or {}
        results = []

        was_training = self.module.training
        self.module.eval()

        try:
            # Empirical estimation via random sampling
            empirical_result = self._estimate_empirical(forward_kwargs)
            results.append(empirical_result)

            # Power iteration estimation (gradient-based)
            power_result = self._estimate_power_iteration(forward_kwargs)
            results.append(power_result)

            # Weight-based upper bound (product of weight norms)
            weight_bound = self._compute_weight_bound()
            results.append(weight_bound)

            # Summary
            estimates = [
                r.value
                for r in results
                if r.passed and isinstance(r.value, (int, float))
            ]
            if estimates:
                results.append(
                    AnalysisResult(
                        name="lipschitz_summary",
                        value={
                            "empirical": empirical_result.value,
                            "power_iteration": power_result.value,
                            "weight_bound": weight_bound.value,
                            "best_estimate": min(estimates),
                        },
                        passed=True,
                        message=f"Best Lipschitz estimate: {min(estimates):.4f}",
                    )
                )

        finally:
            if was_training:
                self.module.train()

        return results

    def _estimate_empirical(
        self,
        forward_kwargs: Dict[str, Any],
    ) -> AnalysisResult:
        """Estimate Lipschitz via random perturbations.

        Parameters
        ----------
        forward_kwargs : Dict[str, Any]
            Forward pass arguments.

        Returns
        -------
        AnalysisResult
            Empirical Lipschitz estimate.
        """
        ratios = []
        perturbation_scale = 0.01

        for _ in range(self.n_samples):
            # Generate base inputs and perturbations
            inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            inputs_list = list(inputs_dict.values())
            
            delta_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            delta_list = [d * perturbation_scale for d in delta_dict.values()]
            inputs_perturbed = [x + d for x, d in zip(inputs_list, delta_list)]

            # Compute outputs
            y1 = safe_forward(self.module, *inputs_list, **forward_kwargs)
            y2 = safe_forward(self.module, *inputs_perturbed, **forward_kwargs)

            # Compute ratio (sum of perturbation norms for multi-input)
            input_diff = sum(torch.linalg.norm(d.flatten()).item() for d in delta_list)
            output_diff = torch.linalg.norm((y2 - y1).flatten()).item()

            if input_diff > 1e-10:
                ratio = output_diff / input_diff
                if not np.isnan(ratio) and not np.isinf(ratio):
                    ratios.append(ratio)

        if not ratios:
            return AnalysisResult(
                name="lipschitz_empirical",
                value=None,
                passed=False,
                message="Could not estimate empirically",
                severity="warning",
            )

        return AnalysisResult(
            name="lipschitz_empirical",
            value=float(max(ratios)),
            passed=True,
            message=f"Empirical Lipschitz: max={max(ratios):.4f}, "
            f"mean={np.mean(ratios):.4f}",
            details={
                "max": float(max(ratios)),
                "mean": float(np.mean(ratios)),
                "std": float(np.std(ratios)),
                "n_samples": len(ratios),
            },
        )

    def _estimate_power_iteration(
        self,
        forward_kwargs: Dict[str, Any],
    ) -> AnalysisResult:
        """Estimate Lipschitz via power iteration on Jacobian.

        Uses gradient-based approach to approximate largest singular
        value of the Jacobian.

        Parameters
        ----------
        forward_kwargs : Dict[str, Any]
            Forward pass arguments.

        Returns
        -------
        AnalysisResult
            Power iteration Lipschitz estimate.
        """
        try:
            # Initialize random vectors for each input
            first_shape = next(iter(self.input_shapes.values()))
            v = torch.randn(
                *first_shape,
                device=self.device,
                dtype=self.dtype,
            )
            v = v / torch.linalg.norm(v.flatten())

            for _ in range(self.n_iterations):
                v.requires_grad_(True)

                # Forward pass
                inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
                inputs_list = list(inputs_dict.values())
                for inp in inputs_list:
                    inp.requires_grad_(True)

                y = self.module(*inputs_list, **forward_kwargs)
                if isinstance(y, tuple):
                    y = y[0]

                # Compute Jacobian-vector product (forward mode)
                # J @ v where J is the Jacobian (use first input for gradient)
                jvp = torch.autograd.grad(
                    y,
                    inputs_list[0],
                    grad_outputs=v if v.shape == y.shape else torch.ones_like(y),
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Normalize
                jvp_norm = torch.linalg.norm(jvp.flatten())
                if jvp_norm < 1e-10:
                    break

                v = jvp.detach() / jvp_norm

            # Final estimate
            sigma = jvp_norm.item()

            return AnalysisResult(
                name="lipschitz_power_iteration",
                value=sigma,
                passed=True,
                message=f"Power iteration Lipschitz: {sigma:.4f}",
                details={"n_iterations": self.n_iterations},
            )

        except Exception as e:
            return AnalysisResult(
                name="lipschitz_power_iteration",
                value=None,
                passed=False,
                message=f"Power iteration failed: {e}",
                severity="warning",
            )

    def _compute_weight_bound(self) -> AnalysisResult:
        """Compute upper bound from product of weight norms.

        For a network f = L_n ... L_1, Lipschitz(f) <= prod(||W_i||).

        Returns
        -------
        AnalysisResult
            Weight-based Lipschitz upper bound.
        """
        spectral_norms = []

        for name, param in self.module.named_parameters():
            if param.dim() < 2:
                continue

            weight = param.data.float().reshape(param.size(0), -1)
            try:
                spectral_norm = torch.linalg.matrix_norm(weight, ord=2).item()
                spectral_norms.append(spectral_norm)
            except Exception:
                pass

        if not spectral_norms:
            return AnalysisResult(
                name="lipschitz_weight_bound",
                value=None,
                passed=False,
                message="No weight matrices found",
                severity="warning",
            )

        # Product of spectral norms
        bound = float(np.prod(spectral_norms))

        return AnalysisResult(
            name="lipschitz_weight_bound",
            value=bound,
            passed=True,
            message=f"Weight-based upper bound: {bound:.4f}",
            details={
                "spectral_norms": spectral_norms,
                "n_weights": len(spectral_norms),
            },
        )

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for Lipschitz estimation.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        first_shape = next(iter(self.input_shapes.values()))
        shape_str = ", ".join(str(d) for d in first_shape)

        return [
            self._create_markdown_cell(
                "## Lipschitz Constant Estimation\n\n"
                "The Lipschitz constant L bounds output change relative to input:\n"
                "||f(x) - f(y)|| <= L * ||x - y||\n\n"
                "Methods:\n"
                "- **Empirical**: Sample random perturbations and measure max ratio\n"
                "- **Weight bound**: Product of spectral norms (upper bound)\n\n"
                "Large Lipschitz constant may indicate sensitivity to input perturbations."
            ),
            self._create_code_cell(
                f'''
import numpy as np

# Empirical Lipschitz estimation
n_samples = 50
perturbation_scale = 0.01
ratios = []

for _ in range(n_samples):
    # Generate base input and perturbation
    x = torch.randn({shape_str}, device=device, dtype=dtype)
    delta = torch.randn({shape_str}, device=device, dtype=dtype) * perturbation_scale
    x_perturbed = x + delta

    # Compute outputs
    with torch.no_grad():
        y1 = module(x)
        y2 = module(x_perturbed)
        if isinstance(y1, tuple):
            y1 = y1[0]
        if isinstance(y2, tuple):
            y2 = y2[0]

    # Compute ratio
    input_diff = torch.linalg.norm(delta.flatten()).item()
    output_diff = torch.linalg.norm((y2 - y1).flatten()).item()

    if input_diff > 1e-10:
        ratios.append(output_diff / input_diff)

# Weight-based upper bound
spectral_norms = []
for name, param in module.named_parameters():
    if param.dim() >= 2:
        weight = param.data.float().reshape(param.size(0), -1)
        spectral_norms.append(torch.linalg.matrix_norm(weight, ord=2).item())

weight_bound = np.prod(spectral_norms) if spectral_norms else float('inf')

print("Lipschitz Constant Estimation:")
print("-" * 60)
print(f"Empirical estimate: max={{max(ratios):.4f}}, mean={{np.mean(ratios):.4f}}")
print(f"Weight-based upper bound: {{weight_bound:.4f}}")
'''
            ),
        ]
