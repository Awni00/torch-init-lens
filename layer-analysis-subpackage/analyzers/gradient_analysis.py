"""
Gradient Analyzer.

Analyzes gradient existence, norms, and completeness to ensure all
parameters are properly connected to the computation graph.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple, Union
import torch
import torch.nn as nn

from .base import BaseAnalyzer, AnalysisResult
from ..inputs.generators import InputGenerator
from ..errors import GradientError
from ..losses import GradientLoss, get_losses, DEFAULT_LOSS


class _LegacyLossWrapper(GradientLoss):
    """Wrapper for legacy callable loss functions."""

    def __init__(self, fn: Callable):
        self._fn = fn

    @property
    def name(self) -> str:
        return "legacy"

    def __call__(self, output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        # Legacy functions may only take output
        import inspect
        sig = inspect.signature(self._fn)
        if len(sig.parameters) >= 2:
            return self._fn(output, input)
        return self._fn(output)


class GradientAnalyzer(BaseAnalyzer):
    """Analyze gradient existence, norms, and completeness.

    Verifies that:
    - All parameters receive gradients during backpropagation
    - Gradient magnitudes are within reasonable ranges
    - No parameters are disconnected from the computation graph

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    input_shapes : Dict[str, Tuple[int, ...]]
        Dictionary mapping input names to their shapes.
    loss_fn : str, List[str], GradientLoss, List[GradientLoss], or Callable, optional
        Loss function(s) for gradient analysis. Can be:
        - A string name (e.g., "reconstruction", "sum", "mse_random", "variance")
        - A list of string names to run analysis with multiple losses
        - A GradientLoss instance
        - A list of GradientLoss instances
        - A callable (output, input) -> scalar loss (legacy)
        Default is "reconstruction".
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> analyzer = GradientAnalyzer(model, input_shapes={"x": (2, 64)})
    >>> results = analyzer.analyze()

    >>> # With multiple loss functions
    >>> analyzer = GradientAnalyzer(
    ...     model,
    ...     input_shapes={"x": (2, 64)},
    ...     loss_fn=["reconstruction", "sum"]
    ... )
    >>> results = analyzer.analyze()
    """

    name = "gradient_analysis"

    def __init__(
        self,
        module: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        loss_fn: Optional[Union[str, List[str], GradientLoss, List[GradientLoss], Callable]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.input_shapes = input_shapes
        self.loss_functions = self._resolve_losses(loss_fn)
        first_shape = next(iter(input_shapes.values()))
        self.input_generator = InputGenerator(
            batch_size=first_shape[0] if len(first_shape) > 0 else 1,
            device=self.device,
            dtype=self.dtype,
        )

    def _resolve_losses(
        self,
        loss_fn: Optional[Union[str, List[str], GradientLoss, List[GradientLoss], Callable]]
    ) -> List[GradientLoss]:
        """Resolve loss function specification to list of GradientLoss instances.

        Parameters
        ----------
        loss_fn : str, List[str], GradientLoss, List[GradientLoss], Callable, or None
            Loss function specification.

        Returns
        -------
        List[GradientLoss]
            List of resolved loss functions.
        """
        if loss_fn is None:
            return get_losses(DEFAULT_LOSS)
        if isinstance(loss_fn, str):
            return get_losses(loss_fn)
        if isinstance(loss_fn, list) and all(isinstance(x, str) for x in loss_fn):
            return get_losses(loss_fn)
        if isinstance(loss_fn, GradientLoss):
            return [loss_fn]
        if isinstance(loss_fn, list) and all(isinstance(x, GradientLoss) for x in loss_fn):
            return loss_fn
        if callable(loss_fn):
            # Legacy callable support - wrap in a simple GradientLoss
            return [_LegacyLossWrapper(loss_fn)]
        raise TypeError(f"Invalid loss_fn type: {type(loss_fn)}")

    def analyze(
        self,
        check_existence: bool = True,
        check_norms: bool = True,
        check_completeness: bool = True,
        check_grad_param_ratio: bool = True,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[AnalysisResult]:
        """Run gradient analysis.

        Parameters
        ----------
        check_existence : bool
            Check that gradients exist for all parameters.
        check_norms : bool
            Compute gradient norms.
        check_completeness : bool
            Verify all parameters have non-None gradients.
        check_grad_param_ratio : bool
            Compute gradient-to-parameter norm ratios.
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        List[AnalysisResult]
            Gradient analysis results.
        """
        forward_kwargs = forward_kwargs or {}
        results = []

        # Run analysis for each loss function
        for loss_fn in self.loss_functions:
            loss_results = self._analyze_with_loss(
                loss_fn,
                check_existence=check_existence,
                check_norms=check_norms,
                check_completeness=check_completeness,
                check_grad_param_ratio=check_grad_param_ratio,
                forward_kwargs=forward_kwargs,
            )
            results.extend(loss_results)

        return results

    def _analyze_with_loss(
        self,
        loss_fn: GradientLoss,
        check_existence: bool,
        check_norms: bool,
        check_completeness: bool,
        check_grad_param_ratio: bool,
        forward_kwargs: Dict[str, Any],
    ) -> List[AnalysisResult]:
        """Run gradient analysis with a specific loss function.

        Parameters
        ----------
        loss_fn : GradientLoss
            Loss function to use.
        check_existence : bool
            Check that gradients exist.
        check_norms : bool
            Compute gradient norms.
        check_completeness : bool
            Check gradient completeness.
        check_grad_param_ratio : bool
            Compute gradient-to-parameter ratios.
        forward_kwargs : Dict[str, Any]
            Additional forward pass arguments.

        Returns
        -------
        List[AnalysisResult]
            Analysis results for this loss function.
        """
        results = []
        loss_name = loss_fn.name

        # Set to training mode for gradient computation
        was_training = self.module.training
        self.module.train()

        try:
            # Zero existing gradients
            self.module.zero_grad()

            # Create inputs
            inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            inputs_list = list(inputs_dict.values())
            primary_input = inputs_list[0]
            for inp in inputs_list:
                inp.requires_grad_(True)

            # Forward pass
            out = self.module(*inputs_list, **forward_kwargs)

            # Compute loss using the GradientLoss interface
            loss = loss_fn(out, primary_input)

            # Backward pass
            loss.backward()

            # Check gradient existence
            if check_existence:
                existence_result = self._check_gradient_existence(loss_name)
                results.append(existence_result)

            # Compute gradient norms
            if check_norms:
                norm_result = self._compute_gradient_norms(loss_name)
                results.append(norm_result)

            # Check completeness
            if check_completeness:
                completeness_result = self._check_gradient_completeness(loss_name)
                results.append(completeness_result)

            # Compute gradient-to-parameter ratios
            if check_grad_param_ratio:
                ratio_result = self._compute_grad_to_param_ratios(loss_name)
                results.append(ratio_result)

            # Check input gradients
            input_grad_norms = {}
            for i, inp in enumerate(inputs_list):
                if inp.grad is not None:
                    input_grad_norms[f"input_{i}"] = torch.linalg.norm(inp.grad.flatten()).item()
            if input_grad_norms:
                total_norm = sum(input_grad_norms.values())
                results.append(
                    AnalysisResult(
                        name=f"input_gradient_{loss_name}",
                        value=total_norm,
                        passed=True,
                        message=f"[{loss_name}] Input gradient norms: {input_grad_norms}",
                        details={"per_input_norms": input_grad_norms, "loss_fn": loss_name},
                    )
                )

            # Zero gradients after analysis
            self.module.zero_grad()

        finally:
            # Restore training mode
            if not was_training:
                self.module.eval()

        return results

    def _check_gradient_existence(self, loss_name: str = "") -> AnalysisResult:
        """Check that all parameters have gradients.

        Parameters
        ----------
        loss_name : str
            Name of the loss function used (for result naming).

        Returns
        -------
        AnalysisResult
            Result indicating whether all gradients exist.
        """
        missing_grads = []
        params_with_grads = []

        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    missing_grads.append(name)
                else:
                    params_with_grads.append(name)

        passed = len(missing_grads) == 0
        prefix = f"[{loss_name}] " if loss_name else ""
        message = (
            f"{prefix}All {len(params_with_grads)} parameters have gradients"
            if passed
            else f"{prefix}{len(missing_grads)} parameters missing gradients"
        )

        result_name = f"gradient_existence_{loss_name}" if loss_name else "gradient_existence"
        return AnalysisResult(
            name=result_name,
            value=passed,
            passed=passed,
            message=message,
            details={
                "missing_gradients": missing_grads,
                "params_with_grads": len(params_with_grads),
                "loss_fn": loss_name,
            },
            severity="error" if not passed else "info",
        )

    def _compute_gradient_norms(self, loss_name: str = "") -> AnalysisResult:
        """Compute gradient norms for all parameters.

        Parameters
        ----------
        loss_name : str
            Name of the loss function used (for result naming).

        Returns
        -------
        AnalysisResult
            Result with per-parameter gradient norms.
        """
        gradient_info: Dict[str, Dict[str, float]] = {}

        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                gradient_info[name] = {
                    "has_grad": False,
                    "norm": None,
                }
            else:
                grad = param.grad.float()
                gradient_info[name] = {
                    "has_grad": True,
                    "norm": torch.linalg.norm(grad.flatten()).item(),
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.abs().max().item(),
                    "min": grad.abs().min().item(),
                }

        # Compute statistics
        norms = [
            info["norm"]
            for info in gradient_info.values()
            if info.get("norm") is not None
        ]

        result_name = f"gradient_norms_{loss_name}" if loss_name else "gradient_norms"
        prefix = f"[{loss_name}] " if loss_name else ""

        if not norms:
            return AnalysisResult(
                name=result_name,
                value=None,
                passed=False,
                message=f"{prefix}No gradient norms computed",
                severity="warning",
                details={"loss_fn": loss_name},
            )

        return AnalysisResult(
            name=result_name,
            value={
                "mean": sum(norms) / len(norms),
                "max": max(norms),
                "min": min(norms),
            },
            passed=True,
            message=f"{prefix}Gradient norms: mean={sum(norms)/len(norms):.6f}, "
            f"max={max(norms):.6f}",
            details={"per_parameter": gradient_info, "loss_fn": loss_name},
        )

    def _check_gradient_completeness(self, loss_name: str = "") -> AnalysisResult:
        """Ensure all trainable parameters receive meaningful gradients.

        Parameters
        ----------
        loss_name : str
            Name of the loss function used (for result naming).

        Returns
        -------
        AnalysisResult
            Result indicating gradient completeness.
        """
        zero_grads = []
        tiny_grads = []
        normal_grads = []

        tolerance = 1e-10

        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            if param.grad is None:
                zero_grads.append(name)
            else:
                grad_norm = torch.linalg.norm(param.grad.flatten()).item()
                if grad_norm == 0:
                    zero_grads.append(name)
                elif grad_norm < tolerance:
                    tiny_grads.append(name)
                else:
                    normal_grads.append(name)

        passed = len(zero_grads) == 0
        warnings = len(tiny_grads) > 0

        prefix = f"[{loss_name}] " if loss_name else ""
        message_parts = [f"{prefix}{len(normal_grads)} params with normal gradients"]
        if zero_grads:
            message_parts.append(f"{len(zero_grads)} with zero/missing gradients")
        if tiny_grads:
            message_parts.append(f"{len(tiny_grads)} with tiny gradients")

        result_name = f"gradient_completeness_{loss_name}" if loss_name else "gradient_completeness"
        return AnalysisResult(
            name=result_name,
            value=passed,
            passed=passed,
            message=", ".join(message_parts),
            details={
                "zero_gradients": zero_grads,
                "tiny_gradients": tiny_grads,
                "normal_gradients": normal_grads,
                "loss_fn": loss_name,
            },
            severity="error" if not passed else ("warning" if warnings else "info"),
        )

    def _compute_grad_to_param_ratios(self, loss_name: str = "") -> AnalysisResult:
        """Compute gradient norm / parameter norm for each parameter.

        This ratio indicates how much each parameter will change relative
        to its current magnitude during a single gradient step (before
        learning rate scaling).

        Parameters
        ----------
        loss_name : str
            Name of the loss function used (for result naming).

        Returns
        -------
        AnalysisResult
            Result with gradient-to-parameter ratios.

        Notes
        -----
        Interpretation of ratios:
        - Ratio >> 1: Parameter will change drastically (potential instability)
        - Ratio << 1: Slow learning for this parameter
        - Ratio ~ 1: Balanced update magnitude
        """
        ratios: Dict[str, float] = {}
        ratio_values = []

        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                continue

            param_norm = torch.linalg.norm(param.data.flatten()).item()
            grad_norm = torch.linalg.norm(param.grad.flatten()).item()

            if param_norm > 1e-10:
                ratio = grad_norm / param_norm
            else:
                ratio = float('inf') if grad_norm > 0 else 0.0

            ratios[name] = ratio
            if ratio != float('inf'):
                ratio_values.append(ratio)

        result_name = f"grad_param_ratio_{loss_name}" if loss_name else "grad_param_ratio"
        prefix = f"[{loss_name}] " if loss_name else ""

        if not ratio_values:
            return AnalysisResult(
                name=result_name,
                value=None,
                passed=False,
                message=f"{prefix}No gradient-to-parameter ratios computed",
                severity="warning",
                details={"loss_fn": loss_name},
            )

        mean_ratio = sum(ratio_values) / len(ratio_values)
        max_ratio = max(ratio_values)
        min_ratio = min(ratio_values)

        # Check for potential issues
        has_large_ratio = max_ratio > 10.0
        has_tiny_ratio = min_ratio < 0.001

        passed = not has_large_ratio
        severity = "info"
        if has_large_ratio:
            severity = "warning"
        if has_tiny_ratio and not has_large_ratio:
            severity = "info"  # Tiny ratios are less concerning

        return AnalysisResult(
            name=result_name,
            value={
                "mean": mean_ratio,
                "max": max_ratio,
                "min": min_ratio,
            },
            passed=passed,
            message=f"{prefix}Grad/param ratios: mean={mean_ratio:.6f}, max={max_ratio:.6f}, min={min_ratio:.6f}",
            details={
                "per_parameter": ratios,
                "loss_fn": loss_name,
            },
            severity=severity,
        )

    def analyze_gradient_flow(
        self,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Analyze gradient flow through the network layers.

        Computes gradient statistics at each layer to detect
        vanishing or exploding gradients.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        AnalysisResult
            Layer-wise gradient flow statistics.
        """
        forward_kwargs = forward_kwargs or {}

        # Set to training mode
        was_training = self.module.training
        self.module.train()
        self.module.zero_grad()

        try:
            # Create inputs
            inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            inputs_list = list(inputs_dict.values())
            primary_input = inputs_list[0]
            for inp in inputs_list:
                inp.requires_grad_(True)

            # Forward pass
            out = self.module(*inputs_list, **forward_kwargs)
            loss_fn = self.loss_functions[0]  # Use first loss function
            loss = loss_fn(out, primary_input)
            loss.backward()

            # Collect gradient statistics by layer
            layer_gradients = {}
            for name, param in self.module.named_parameters():
                if param.grad is None:
                    continue

                # Extract layer name (first part before ".")
                layer = name.split(".")[0]
                if layer not in layer_gradients:
                    layer_gradients[layer] = []

                layer_gradients[layer].append(
                    torch.linalg.norm(param.grad.flatten()).item()
                )

            # Compute per-layer statistics
            layer_stats = {}
            for layer, norms in layer_gradients.items():
                layer_stats[layer] = {
                    "mean": sum(norms) / len(norms),
                    "max": max(norms),
                    "min": min(norms),
                }

            self.module.zero_grad()

        finally:
            if not was_training:
                self.module.eval()

        return AnalysisResult(
            name="gradient_flow",
            value=layer_stats,
            passed=True,
            message=f"Analyzed gradient flow through {len(layer_stats)} layers",
            details={"layer_statistics": layer_stats},
        )

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for gradient analysis.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        shape_str = ", ".join(str(d) for d in self.input_shape)

        return [
            self._create_markdown_cell(
                "## Gradient Analysis\n\n"
                "Verify gradient existence and compute gradient norms to assess "
                "training stability.\n\n"
                "Key checks:\n"
                "- All parameters receive gradients\n"
                "- No vanishing gradients (zero norms)\n"
                "- Reasonable gradient magnitudes"
            ),
            self._create_code_cell(
                f'''
# Create input requiring gradients
x = torch.randn({shape_str}, device=device, dtype=dtype, requires_grad=True)

# Forward pass
module.train()
out = module(x)
if isinstance(out, tuple):
    out = out[0]

# Backward pass with scalar loss
loss = out.sum()
loss.backward()

# Check gradient existence and norms
gradient_info = {{}}
missing_grads = []

for name, param in module.named_parameters():
    if param.grad is None:
        missing_grads.append(name)
        gradient_info[name] = {{"has_grad": False, "norm": None}}
    else:
        grad_norm = torch.linalg.norm(param.grad.flatten()).item()
        gradient_info[name] = {{
            "has_grad": True,
            "norm": grad_norm,
            "mean": param.grad.mean().item(),
            "std": param.grad.std().item(),
            "max": param.grad.abs().max().item(),
        }}

# Report
print("Gradient Analysis:")
print("-" * 60)

if missing_grads:
    print(f"WARNING: {{len(missing_grads)}} parameters missing gradients:")
    for name in missing_grads:
        print(f"  - {{name}}")
else:
    print("All parameters have gradients.")

print("\\nGradient Norms:")
for name, info in gradient_info.items():
    if info["has_grad"]:
        print(f"  {{name:40s}}: norm={{info['norm']:.6f}}")

# Reset
module.zero_grad()
module.eval()
'''
            ),
        ]
