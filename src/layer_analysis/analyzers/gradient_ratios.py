"""
Gradient Ratio Analyzer.

Analyzes layer-wise gradient ratios to detect vanishing or exploding
gradients through the network.
"""

from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..inputs.generators import InputGenerator
from ..losses import GradientLoss, get_losses, DEFAULT_LOSS


class GradientRatioAnalyzer(BaseAnalyzer):
    """Analyze layer-wise gradient ratios.

    Computes the ratio of gradient norms between successive layers
    to detect:
    - Vanishing gradients (ratio << 1)
    - Exploding gradients (ratio >> 1)

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    input_shapes : Dict[str, Tuple[int, ...]]
        Dictionary mapping input names to their shapes.
    loss_fn : str or GradientLoss, optional
        Loss function for gradient analysis. Default is "reconstruction".
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 16))
    >>> analyzer = GradientRatioAnalyzer(model, input_shapes={"x": (2, 64)})
    >>> results = analyzer.analyze()
    """

    name = "gradient_ratios"

    def __init__(
        self,
        module: nn.Module,
        input_shapes: Dict[str, Tuple[int, ...]],
        loss_fn: Optional[Union[str, GradientLoss]] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.input_shapes = input_shapes
        self.loss_fn = self._resolve_loss(loss_fn)
        first_shape = next(iter(input_shapes.values()))
        self.input_generator = InputGenerator(
            batch_size=first_shape[0] if len(first_shape) > 0 else 1,
            device=self.device,
            dtype=self.dtype,
        )

    def _resolve_loss(
        self,
        loss_fn: Optional[Union[str, GradientLoss]]
    ) -> GradientLoss:
        """Resolve loss function specification to GradientLoss instance.

        Parameters
        ----------
        loss_fn : str, GradientLoss, or None
            Loss function specification.

        Returns
        -------
        GradientLoss
            Resolved loss function.
        """
        if loss_fn is None:
            return get_losses(DEFAULT_LOSS)[0]
        if isinstance(loss_fn, str):
            return get_losses(loss_fn)[0]
        if isinstance(loss_fn, GradientLoss):
            return loss_fn
        raise TypeError(f"Invalid loss_fn type: {type(loss_fn)}")

    def analyze(
        self,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[AnalysisResult]:
        """Analyze gradient ratios between layers.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        List[AnalysisResult]
            Gradient ratio analysis results.
        """
        forward_kwargs = forward_kwargs or {}
        results = []

        was_training = self.module.training
        self.module.train()
        self.module.zero_grad()

        try:
            # Forward and backward pass
            inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            inputs_list = list(inputs_dict.values())
            primary_input = inputs_list[0]
            for inp in inputs_list:
                inp.requires_grad_(True)

            out = self.module(*inputs_list, **forward_kwargs)

            loss = self.loss_fn(out, primary_input)
            loss.backward()

            # Collect gradient norms by layer
            layer_grad_norms = self._collect_layer_gradients()

            if len(layer_grad_norms) < 2:
                return [
                    AnalysisResult(
                        name="gradient_ratios",
                        value=None,
                        passed=True,
                        message="Not enough layers for ratio analysis",
                        severity="warning",
                    )
                ]

            # Compute ratios between successive layers
            layer_names = list(layer_grad_norms.keys())
            ratios = {}

            for i in range(len(layer_names) - 1):
                current = layer_names[i]
                next_layer = layer_names[i + 1]

                current_norm = layer_grad_norms[current]
                next_norm = layer_grad_norms[next_layer]

                if current_norm > 1e-10:
                    ratio = next_norm / current_norm
                    ratios[f"{current}->{next_layer}"] = ratio

                    # Check for issues
                    if ratio < 0.1:
                        severity = "warning"
                        message = f"Vanishing: {ratio:.4f}"
                        passed = False
                    elif ratio > 10:
                        severity = "warning"
                        message = f"Exploding: {ratio:.4f}"
                        passed = False
                    else:
                        severity = "info"
                        message = f"Ratio: {ratio:.4f}"
                        passed = True

                    results.append(
                        AnalysisResult(
                            name=f"grad_ratio_{current}_{next_layer}",
                            value=ratio,
                            passed=passed,
                            message=message,
                            severity=severity,
                        )
                    )

            # Summary
            if ratios:
                ratio_values = list(ratios.values())
                results.append(
                    AnalysisResult(
                        name="gradient_ratio_summary",
                        value={
                            "mean_ratio": float(np.mean(ratio_values)),
                            "max_ratio": float(max(ratio_values)),
                            "min_ratio": float(min(ratio_values)),
                            "std_ratio": float(np.std(ratio_values)),
                        },
                        passed=bool(0.1 < np.mean(ratio_values) < 10),
                        message=f"Mean gradient ratio: {np.mean(ratio_values):.4f}",
                        details={
                            "layer_norms": layer_grad_norms,
                            "ratios": ratios,
                        },
                    )
                )

            self.module.zero_grad()

        finally:
            if not was_training:
                self.module.eval()

        return results

    def _collect_layer_gradients(self) -> Dict[str, float]:
        """Collect gradient norms grouped by layer.

        Returns
        -------
        Dict[str, float]
            Layer name -> total gradient norm.
        """
        layer_grads: Dict[str, List[float]] = {}

        for name, param in self.module.named_parameters():
            if param.grad is None:
                continue

            # Extract layer name (first part)
            parts = name.split(".")
            layer = parts[0] if len(parts) > 1 else name

            grad_norm = torch.linalg.norm(param.grad.flatten()).item()

            if layer not in layer_grads:
                layer_grads[layer] = []
            layer_grads[layer].append(grad_norm)

        # Compute total norm per layer
        return {
            layer: float(np.sqrt(sum(n**2 for n in norms)))
            for layer, norms in layer_grads.items()
        }

    def analyze_gradient_flow(
        self,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AnalysisResult:
        """Analyze gradient flow through all parameters.

        Provides a detailed view of gradient distribution across
        the entire network.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.

        Returns
        -------
        AnalysisResult
            Detailed gradient flow analysis.
        """
        forward_kwargs = forward_kwargs or {}

        was_training = self.module.training
        self.module.train()
        self.module.zero_grad()

        try:
            inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
            inputs_list = list(inputs_dict.values())
            primary_input = inputs_list[0]
            for inp in inputs_list:
                inp.requires_grad_(True)

            out = self.module(*inputs_list, **forward_kwargs)

            loss = self.loss_fn(out, primary_input)
            loss.backward()

            # Collect all gradient norms
            all_norms = []
            param_norms = {}

            for name, param in self.module.named_parameters():
                if param.grad is not None:
                    norm = torch.linalg.norm(param.grad.flatten()).item()
                    all_norms.append(norm)
                    param_norms[name] = norm

            self.module.zero_grad()

            if not all_norms:
                return AnalysisResult(
                    name="gradient_flow",
                    value=None,
                    passed=False,
                    message="No gradients found",
                    severity="error",
                )

            # Compute statistics
            return AnalysisResult(
                name="gradient_flow",
                value={
                    "mean": float(np.mean(all_norms)),
                    "std": float(np.std(all_norms)),
                    "max": float(max(all_norms)),
                    "min": float(min(all_norms)),
                    "range": float(max(all_norms) / (min(all_norms) + 1e-10)),
                },
                passed=True,
                message=f"Gradient flow: mean={np.mean(all_norms):.6f}, "
                f"range={max(all_norms)/(min(all_norms)+1e-10):.2f}x",
                details={"per_parameter": param_norms},
            )

        finally:
            if not was_training:
                self.module.eval()

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for gradient ratio analysis.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        inputs_code_lines = []
        input_var_names = []
        for name, shape in self.input_shapes.items():
            shape_str = ", ".join(str(d) for d in shape)
            inputs_code_lines.append(
                f"{name} = torch.randn({shape_str}, device=device, dtype=dtype, requires_grad=True)"
            )
            input_var_names.append(name)
        inputs_code = "\n".join(inputs_code_lines)
        inputs_call = ", ".join(input_var_names)

        return [
            self._create_markdown_cell(
                "## Layer-wise Gradient Ratios\n\n"
                "Analyze the ratio of gradient norms between successive layers.\n\n"
                "- Ratio << 1: Potential vanishing gradients\n"
                "- Ratio >> 1: Potential exploding gradients\n"
                "- Ratio ~ 1: Healthy gradient flow"
            ),
            self._create_code_cell(
                f'''
import numpy as np

# Forward and backward pass
module.train()
module.zero_grad()

{inputs_code}
out = module({inputs_call})
if isinstance(out, tuple):
    out = out[0]
out.sum().backward()

# Collect gradient norms by layer
layer_grads = {{}}
for name, param in module.named_parameters():
    if param.grad is None:
        continue
    layer = name.split(".")[0]
    grad_norm = torch.linalg.norm(param.grad.flatten()).item()
    if layer not in layer_grads:
        layer_grads[layer] = []
    layer_grads[layer].append(grad_norm)

# Total norm per layer
layer_norms = {{
    layer: np.sqrt(sum(n**2 for n in norms))
    for layer, norms in layer_grads.items()
}}

# Compute ratios
layer_names = list(layer_norms.keys())
print("Gradient Ratios:")
print("-" * 60)
for i in range(len(layer_names) - 1):
    current = layer_names[i]
    next_layer = layer_names[i + 1]
    ratio = layer_norms[next_layer] / (layer_norms[current] + 1e-10)
    status = "OK" if 0.1 < ratio < 10 else "WARNING"
    print(f"{{current}} -> {{next_layer}}: {{ratio:.4f}} [{{status}}]")

module.zero_grad()
module.eval()
'''
            ),
        ]
