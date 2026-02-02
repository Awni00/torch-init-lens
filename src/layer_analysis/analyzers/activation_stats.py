"""
Activation Statistics Analyzer.

Analyzes activation statistics (mean, std, max) during forward passes
to detect potential issues like saturation or dead neurons.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from .base import BaseAnalyzer, AnalysisResult
from ..inputs.generators import InputGenerator


class ActivationStatsAnalyzer(BaseAnalyzer):
    """Analyze activation statistics during forward pass.

    Hooks into intermediate layers to collect activation statistics,
    useful for detecting:
    - Vanishing activations (near-zero values)
    - Exploding activations (very large values)
    - Dead neurons (always zero)
    - Saturation (values at extremes)

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    input_shapes : Dict[str, Tuple[int, ...]]
        Dictionary mapping input names to their shapes.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
    >>> analyzer = ActivationStatsAnalyzer(model, input_shapes={"x": (2, 64)})
    >>> results = analyzer.analyze()
    """

    name = "activation_stats"

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
        self._hooks = []
        self._activation_stats: Dict[str, Dict[str, float]] = {}

    def analyze(
        self,
        forward_kwargs: Optional[Dict[str, Any]] = None,
        n_samples: int = 10,
    ) -> List[AnalysisResult]:
        """Analyze activation statistics.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional forward pass arguments.
        n_samples : int
            Number of forward passes to average statistics.

        Returns
        -------
        List[AnalysisResult]
            Activation statistics per layer.
        """
        forward_kwargs = forward_kwargs or {}
        results = []

        was_training = self.module.training
        self.module.eval()

        try:
            # Register hooks
            self._register_hooks()

            # Run forward passes
            for _ in range(n_samples):
                inputs_dict = self.input_generator.generate_inputs(self.input_shapes, "normal")
                inputs_list = list(inputs_dict.values())
                with torch.no_grad():
                    self.module(*inputs_list, **forward_kwargs)

            # Average statistics
            averaged_stats = self._average_statistics(n_samples)

            # Create results per layer
            for layer_name, stats in averaged_stats.items():
                # Check for issues
                issues = []
                if stats["mean_abs"] < 1e-6:
                    issues.append("near-zero activations")
                if stats["max"] > 1e6:
                    issues.append("very large activations")
                if stats.get("zero_fraction", 0) > 0.9:
                    issues.append("many dead neurons")

                passed = len(issues) == 0

                results.append(
                    AnalysisResult(
                        name=f"activation_{layer_name}",
                        value=stats,
                        passed=passed,
                        message=f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                        f"max={stats['max']:.4f}"
                        + (f" [{', '.join(issues)}]" if issues else ""),
                        details=stats,
                        severity="warning" if not passed else "info",
                    )
                )

            # Summary
            if averaged_stats:
                all_means = [s["mean_abs"] for s in averaged_stats.values()]
                all_stds = [s["std"] for s in averaged_stats.values()]

                results.append(
                    AnalysisResult(
                        name="activation_summary",
                        value={
                            "mean_activation": sum(all_means) / len(all_means),
                            "mean_std": sum(all_stds) / len(all_stds),
                            "n_layers": len(averaged_stats),
                        },
                        passed=True,
                        message=f"Analyzed {len(averaged_stats)} layers",
                        details={"per_layer": averaged_stats},
                    )
                )

        finally:
            # Remove hooks
            self._remove_hooks()
            self._activation_stats.clear()
            if was_training:
                self.module.train()

        return results

    def _register_hooks(self) -> None:
        """Register forward hooks on all layers."""
        for name, layer in self.module.named_modules():
            if name:  # Skip root
                hook = layer.register_forward_hook(
                    self._make_hook(name)
                )
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook function for a specific layer.

        Parameters
        ----------
        name : str
            Layer name.

        Returns
        -------
        Callable
            Hook function.
        """
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if not isinstance(output, torch.Tensor):
                return

            output_float = output.detach().float()

            stats = {
                "mean": output_float.mean().item(),
                "std": output_float.std().item(),
                "max": output_float.max().item(),
                "min": output_float.min().item(),
                "mean_abs": output_float.abs().mean().item(),
                "zero_fraction": (output_float == 0).float().mean().item(),
            }

            if name not in self._activation_stats:
                self._activation_stats[name] = []
            self._activation_stats[name].append(stats)

        return hook

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _average_statistics(self, n_samples: int) -> Dict[str, Dict[str, float]]:
        """Average statistics across samples.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Averaged statistics per layer.
        """
        averaged = {}
        for name, stats_list in self._activation_stats.items():
            if not stats_list:
                continue

            averaged[name] = {
                key: sum(s[key] for s in stats_list) / len(stats_list)
                for key in stats_list[0].keys()
            }

        return averaged

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for activation statistics.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        # Build input generation code for multiple inputs
        input_gen_lines = []
        input_names = []
        for name, shape in self.input_shapes.items():
            shape_str = ", ".join(str(d) for d in shape)
            input_gen_lines.append(f"    {name} = torch.randn({shape_str}, device=device, dtype=dtype)")
            input_names.append(name)
        input_gen_code = "\n".join(input_gen_lines)
        input_args = ", ".join(input_names)

        return [
            self._create_markdown_cell(
                "## Activation Statistics\n\n"
                "Analyze activation values at each layer during forward pass.\n\n"
                "Potential issues to detect:\n"
                "- Very small activations (vanishing signals)\n"
                "- Very large activations (exploding signals)\n"
                "- High fraction of zeros (dead neurons)"
            ),
            self._create_code_cell(
                f'''
# Collect activation statistics using hooks
activation_stats = {{}}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            return

        out = output.detach().float()
        activation_stats[name] = {{
            "mean": out.mean().item(),
            "std": out.std().item(),
            "max": out.max().item(),
            "min": out.min().item(),
            "mean_abs": out.abs().mean().item(),
            "zero_fraction": (out == 0).float().mean().item(),
        }}
    return hook

# Register hooks
hooks = []
for name, layer in module.named_modules():
    if name:
        hooks.append(layer.register_forward_hook(make_hook(name)))

# Forward pass
with torch.no_grad():
{input_gen_code}
    module({input_args})

# Remove hooks
for hook in hooks:
    hook.remove()

# Display results
print("Activation Statistics:")
print("-" * 80)
print(f"{{'Layer':<40}} {{'Mean':>10}} {{'Std':>10}} {{'Max':>10}} {{'Zero%':>10}}")
print("-" * 80)
for name, stats in activation_stats.items():
    short_name = name[:38]
    print(f"{{short_name:<40}} {{stats['mean']:>10.4f}} {{stats['std']:>10.4f}} "
          f"{{stats['max']:>10.4f}} {{stats['zero_fraction']*100:>9.1f}}%")
'''
            ),
        ]
