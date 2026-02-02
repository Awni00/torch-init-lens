"""
Numerical Precision Checker.

Checks for potential numerical precision issues such as near-zero
divisions, overflow risks, and NaN/Inf values.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..inputs.generators import InputGenerator
from ..errors import safe_forward


class NumericalPrecisionChecker(BaseAnalyzer):
    """Check for numerical precision issues.

    Detects potential problems:
    - NaN or Inf values in outputs
    - Near-zero values that could cause division issues
    - Overflow risks from large values
    - Underflow risks from very small values

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    input_shapes : Dict[str, Tuple[int, ...]]
        Mapping of input names to their shapes.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> checker = NumericalPrecisionChecker(model, input_shapes={"x": (2, 64)})
    >>> results = checker.analyze()
    """

    name = "numerical_precision"

    def __init__(
        self,
        module: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.input_shapes = input_shapes
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
        """Run numerical precision checks.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        List[AnalysisResult]
            Numerical precision check results.
        """
        forward_kwargs = forward_kwargs or {}
        results = []

        was_training = self.module.training
        self.module.eval()

        try:
            # Check parameters
            param_result = self._check_parameters()
            results.append(param_result)

            # Check outputs with normal inputs
            normal_result = self._check_normal_inputs(forward_kwargs)
            results.append(normal_result)

            # Check outputs with extreme inputs
            extreme_result = self._check_extreme_inputs(forward_kwargs)
            results.append(extreme_result)

            # Check gradients
            grad_result = self._check_gradient_precision(forward_kwargs)
            results.append(grad_result)

            # Summary
            all_passed = all(r.passed for r in results)
            results.append(
                AnalysisResult(
                    name="precision_summary",
                    value=all_passed,
                    passed=all_passed,
                    message="All precision checks passed" if all_passed else "Some precision issues detected",
                    severity="info" if all_passed else "warning",
                )
            )

        finally:
            if was_training:
                self.module.train()

        return results

    def _check_parameters(self) -> AnalysisResult:
        """Check parameters for numerical issues.

        Returns
        -------
        AnalysisResult
            Parameter check result.
        """
        issues = []
        param_stats = {}

        for name, param in self.module.named_parameters():
            p = param.data.float()

            stats = {
                "has_nan": bool(torch.isnan(p).any().item()),
                "has_inf": bool(torch.isinf(p).any().item()),
                "max_abs": p.abs().max().item(),
                "min_abs": p[p != 0].abs().min().item() if (p != 0).any() else 0,
                "zero_count": (p == 0).sum().item(),
            }
            param_stats[name] = stats

            if stats["has_nan"]:
                issues.append(f"{name}: contains NaN")
            if stats["has_inf"]:
                issues.append(f"{name}: contains Inf")
            if stats["max_abs"] > 1e6:
                issues.append(f"{name}: very large values ({stats['max_abs']:.2e})")
            if stats["min_abs"] < 1e-10 and stats["min_abs"] > 0:
                issues.append(f"{name}: very small non-zero values ({stats['min_abs']:.2e})")

        return AnalysisResult(
            name="parameter_precision",
            value=len(issues) == 0,
            passed=len(issues) == 0,
            message="Parameters OK" if not issues else f"Issues: {', '.join(issues[:3])}",
            details={"issues": issues, "stats": param_stats},
            severity="error" if issues else "info",
        )

    def _check_normal_inputs(
        self,
        forward_kwargs: Dict[str, Any],
    ) -> AnalysisResult:
        """Check outputs with normal inputs.

        Parameters
        ----------
        forward_kwargs : Dict[str, Any]
            Forward pass arguments.

        Returns
        -------
        AnalysisResult
            Normal input check result.
        """
        issues = []

        inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
        inputs_list = list(inputs_dict.values())

        with torch.no_grad():
            out = safe_forward(self.module, *inputs_list, **forward_kwargs)
            if isinstance(out, tuple):
                out = out[0]

            out_float = out.float()

            if torch.isnan(out_float).any():
                issues.append("NaN in output")
            if torch.isinf(out_float).any():
                issues.append("Inf in output")

            max_val = out_float.abs().max().item()
            if max_val > 1e6:
                issues.append(f"Very large output values ({max_val:.2e})")

        return AnalysisResult(
            name="normal_input_precision",
            value=len(issues) == 0,
            passed=len(issues) == 0,
            message="Normal inputs OK" if not issues else f"Issues: {', '.join(issues)}",
            details={"issues": issues, "max_output": max_val},
            severity="error" if issues else "info",
        )

    def _check_extreme_inputs(
        self,
        forward_kwargs: Dict[str, Any],
    ) -> AnalysisResult:
        """Check outputs with extreme inputs.

        Parameters
        ----------
        forward_kwargs : Dict[str, Any]
            Forward pass arguments.

        Returns
        -------
        AnalysisResult
            Extreme input check result.
        """
        issues = []
        test_results = {}

        test_cases = [
            ("large", "large_magnitude", {"scale": 100.0}),
            ("small", "small_magnitude", {"scale": 1e-6}),
            ("zeros", "zeros", {}),
        ]

        for name, distribution, kwargs in test_cases:
            try:
                inputs_dict = self.input_generator.generate_inputs(
                    self.input_shapes, distribution, **kwargs
                )
                inputs_list = list(inputs_dict.values())

                with torch.no_grad():
                    out = safe_forward(self.module, *inputs_list, **forward_kwargs)
                    if isinstance(out, tuple):
                        out = out[0]

                    out_float = out.float()

                    has_nan = torch.isnan(out_float).any().item()
                    has_inf = torch.isinf(out_float).any().item()
                    max_val = out_float.abs().max().item()

                    test_results[name] = {
                        "has_nan": has_nan,
                        "has_inf": has_inf,
                        "max_value": max_val,
                    }

                    if has_nan:
                        issues.append(f"{name} input: NaN in output")
                    if has_inf:
                        issues.append(f"{name} input: Inf in output")

            except Exception as e:
                issues.append(f"{name} input: {str(e)[:50]}")
                test_results[name] = {"error": str(e)}

        return AnalysisResult(
            name="extreme_input_precision",
            value=len(issues) == 0,
            passed=len(issues) == 0,
            message="Extreme inputs OK" if not issues else f"Issues: {', '.join(issues[:3])}",
            details={"test_results": test_results, "issues": issues},
            severity="warning" if issues else "info",
        )

    def _check_gradient_precision(
        self,
        forward_kwargs: Dict[str, Any],
    ) -> AnalysisResult:
        """Check gradient precision.

        Parameters
        ----------
        forward_kwargs : Dict[str, Any]
            Forward pass arguments.

        Returns
        -------
        AnalysisResult
            Gradient precision check result.
        """
        issues = []
        grad_stats = {}

        was_training = self.module.training
        self.module.train()
        self.module.zero_grad()

        try:
            inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            inputs_list = list(inputs_dict.values())
            for inp in inputs_list:
                inp.requires_grad_(True)

            # Call module directly (not safe_forward which uses no_grad)
            out = self.module(*inputs_list, **forward_kwargs)
            if isinstance(out, tuple):
                out = out[0]

            loss = out.sum()
            loss.backward()

            for name, param in self.module.named_parameters():
                if param.grad is None:
                    continue

                g = param.grad.float()

                stats = {
                    "has_nan": bool(torch.isnan(g).any().item()),
                    "has_inf": bool(torch.isinf(g).any().item()),
                    "max_abs": g.abs().max().item(),
                    "mean_abs": g.abs().mean().item(),
                }
                grad_stats[name] = stats

                if stats["has_nan"]:
                    issues.append(f"{name}: gradient contains NaN")
                if stats["has_inf"]:
                    issues.append(f"{name}: gradient contains Inf")
                if stats["max_abs"] > 1e6:
                    issues.append(f"{name}: very large gradient ({stats['max_abs']:.2e})")

            self.module.zero_grad()

        finally:
            if not was_training:
                self.module.eval()

        return AnalysisResult(
            name="gradient_precision",
            value=len(issues) == 0,
            passed=len(issues) == 0,
            message="Gradients OK" if not issues else f"Issues: {', '.join(issues[:3])}",
            details={"grad_stats": grad_stats, "issues": issues},
            severity="error" if issues else "info",
        )

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for precision checking.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        shapes_repr = repr(self.input_shapes)

        return [
            self._create_markdown_cell(
                "## Numerical Precision Checks\n\n"
                "Check for potential numerical issues:\n"
                "- NaN or Inf values\n"
                "- Very large values (overflow risk)\n"
                "- Very small values (underflow risk)\n"
                "- Division by near-zero"
            ),
            self._create_code_cell(
                f'''
# Check parameters
print("Parameter Precision Check:")
print("-" * 60)
for name, param in module.named_parameters():
    p = param.data.float()
    has_nan = torch.isnan(p).any().item()
    has_inf = torch.isinf(p).any().item()
    max_abs = p.abs().max().item()

    status = "OK"
    if has_nan:
        status = "NaN DETECTED"
    elif has_inf:
        status = "Inf DETECTED"
    elif max_abs > 1e6:
        status = f"LARGE VALUES ({{max_abs:.2e}})"

    print(f"  {{name}}: {{status}}")

# Check with various inputs
print("\\nOutput Precision Check:")
print("-" * 60)

input_shapes = {shapes_repr}

def make_inputs(input_shapes, generator_fn):
    return [generator_fn(shape, device=device, dtype=dtype) for shape in input_shapes.values()]

test_cases = [
    ("normal", lambda shape, **kw: torch.randn(shape, **kw)),
    ("large", lambda shape, **kw: torch.randn(shape, **kw) * 100),
    ("small", lambda shape, **kw: torch.randn(shape, **kw) * 1e-6),
    ("zeros", lambda shape, **kw: torch.zeros(shape, **kw)),
]

for name, gen_fn in test_cases:
    inputs = make_inputs(input_shapes, gen_fn)
    with torch.no_grad():
        out = module(*inputs)
        if isinstance(out, tuple):
            out = out[0]

        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        max_val = out.abs().max().item()

        status = "OK"
        if has_nan:
            status = "NaN"
        elif has_inf:
            status = "Inf"
        elif max_val > 1e6:
            status = f"Large ({{max_val:.2e}})"

        print(f"  {{name:10s}}: {{status}}")
'''
            ),
        ]
