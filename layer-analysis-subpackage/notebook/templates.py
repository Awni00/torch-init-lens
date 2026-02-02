"""
Notebook Cell Templates.

This module provides templates for generating notebook cells for each
type of module analysis.
"""

from typing import List, Dict, Any, Tuple, Optional, Type
import textwrap
import json

import torch.nn as nn


def _markdown_cell(content: str) -> Dict[str, Any]:
    """Create a markdown notebook cell.

    Parameters
    ----------
    content : str
        Markdown content.

    Returns
    -------
    Dict[str, Any]
        Notebook cell dictionary.
    """
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.split("\n")],
    }


def _code_cell(source: str) -> Dict[str, Any]:
    """Create a code notebook cell.

    Parameters
    ----------
    source : str
        Python code.

    Returns
    -------
    Dict[str, Any]
        Notebook cell dictionary.
    """
    source = textwrap.dedent(source).strip()
    return {
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")],
        "outputs": [],
        "execution_count": None,
    }


def get_title_cells(
    module_identifier: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    device: str,
    dtype: str,
) -> List[Dict[str, Any]]:
    """Generate title and introduction cells.

    Parameters
    ----------
    module_identifier : str
        Module identifier (e.g., "models.abstractor@RelationalAttention").
    input_shapes : Dict[str, Tuple[int, ...]]
        Input tensor shapes.
    device : str
        Device string.
    dtype : str
        Data type string.

    Returns
    -------
    List[Dict[str, Any]]
        Title cells.
    """
    module_path, class_name = module_identifier.split("@")

    shapes_str = "\n".join([f"- {k}: {v}" for k, v in input_shapes.items()])

    return [
        _markdown_cell(
            f"# Module Initialization Analysis: {class_name}\n\n"
            f"**Module**: `{module_identifier}`\n\n"
            f"**Configuration**:\n"
            f"{shapes_str}\n"
            f"- Device: {device}\n"
            f"- Dtype: {dtype}\n\n"
            "This notebook analyzes the initialization properties of the module "
            "to verify stability and identify potential issues."
        ),
    ]


def get_setup_cells(
    module_path: str,
    class_name: str,
) -> List[Dict[str, Any]]:
    """Generate setup and import cells.

    Parameters
    ----------
    module_path : str
        Python module path.
    class_name : str
        Class name to import.

    Returns
    -------
    List[Dict[str, Any]]
        Setup cells.
    """
    return [
        _markdown_cell("## Setup"),
        _code_cell(
            f'''
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(os.getcwd())
while not (project_root / "models").exists() and project_root.parent != project_root:
    project_root = project_root.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import the module to analyze
from {module_path} import {class_name}

# Set random seed for reproducibility
torch.manual_seed(42)

print(f"PyTorch version: {{torch.__version__}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
'''
        ),
    ]


def get_module_instantiation_cells(
    module_path: str,
    class_name: str,
    input_shapes: Dict[str, Tuple[int, ...]],
    device: str,
    dtype: str,
    module_kwargs: Optional[Dict[str, Any]] = None,
    module_class: Optional[Type[nn.Module]] = None,
) -> List[Dict[str, Any]]:
    """Generate module instantiation cells.

    Parameters
    ----------
    module_path : str
        Python module path.
    class_name : str
        Class name.
    input_shapes : Dict[str, Tuple[int, ...]]
        Input shapes.
    device : str
        Device string.
    dtype : str
        Data type string.
    module_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for module instantiation.
    module_class : Optional[Type[nn.Module]]
        Module class for signature-aware default merging.

    Returns
    -------
    List[Dict[str, Any]]
        Instantiation cells.
    """
    module_kwargs = module_kwargs or {}

    # Use the first input shape for default merging (e.g. d_model derivation)
    primary_shape = next(iter(input_shapes.values()))

    # Build config using merge_kwargs if module_class is available
    if module_class is not None:
        from ..introspection import merge_kwargs
        module_config_dict = merge_kwargs(module_class, module_kwargs, primary_shape)
    else:
        # Fallback for backwards compatibility
        from ..introspection import load_defaults
        defaults = load_defaults()
        module_config_dict = {
            "d_model": primary_shape[2] if len(primary_shape) > 2 else 64,
            **{k: v for k, v in defaults.items()},
            **module_kwargs,
        }
    module_config_str = "{\n" + "\n".join(
        f'    {k!r}: {v!r},' for k, v in module_config_dict.items()
    ) + "\n}"

    input_shapes_str = json.dumps(input_shapes, indent=4)

    return [
        _markdown_cell(
            f"## Module Instantiation\n\n"
            f"Create an instance of `{class_name}` with the configured parameters.\n\n"
            "**Note**: Modify `INPUT_SHAPES` and `MODULE_CONFIG` below to adjust the analysis."
        ),
        _code_cell(
            f'''
# === CONFIGURATION (modify these as needed) ===
INPUT_SHAPES = {input_shapes_str}
MODULE_CONFIG = {module_config_str}

# Derived configuration
device = torch.device("{device}" if torch.cuda.is_available() else "cpu")
dtype = torch.{dtype}

# Instantiate module
module = {class_name}(**MODULE_CONFIG).to(device=device, dtype=dtype)

# Set to eval mode for analysis
module.eval();

# Print module info
print(f"Module: {{type(module).__name__}}")
print(f"Device: {{device}}")
print(f"Dtype: {{dtype}}")
print(f"Input shapes: {{INPUT_SHAPES}}")
print(f"Total parameters: {{sum(p.numel() for p in module.parameters()):,}}")
print(f"Trainable parameters: {{sum(p.numel() for p in module.parameters() if p.requires_grad):,}}")
print()

# List parameters
print("Parameters:")
for name, param in module.named_parameters():
    print(f"  {{name}}: {{tuple(param.shape)}}")
'''
        ),
    ]


def get_summary_cells() -> List[Dict[str, Any]]:
    """Generate summary cells.

    Returns
    -------
    List[Dict[str, Any]]
        Summary cells.
    """
    return [
        _markdown_cell(
            "## Summary\n\n"
            "Aggregate the analysis results and provide overall assessment."
        ),
        _code_cell(
            '''
# Aggregate summary
print("=" * 70)
print("INITIALIZATION ANALYSIS SUMMARY")
print("=" * 70)

# Parameter summary
total_params = sum(p.numel() for p in module.parameters())
trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
print(f"\\nParameters: {total_params:,} total, {trainable_params:,} trainable")

# Norm summary
if 'param_norms' in dir():
    frob_norms = [v["frobenius"] for v in param_norms.values()]
    print(f"\\nParameter Norms:")
    print(f"  Frobenius: mean={np.mean(frob_norms):.4f}, max={np.max(frob_norms):.4f}")

# Operator norm summary
if 'operator_norm_estimates' in dir():
    max_norms = [v["max"] for v in operator_norm_estimates.values()]
    print(f"\\nOperator Norm Estimates:")
    print(f"  Max: {max(max_norms):.4f}")

# Gradient summary
if 'gradient_info' in dir():
    grad_norms = [v["norm"] for v in gradient_info.values() if v.get("norm") is not None]
    if grad_norms:
        print(f"\\nGradient Norms:")
        print(f"  Mean: {np.mean(grad_norms):.6f}, Max: {np.max(grad_norms):.6f}")

# Spectral summary
if 'spectral_info' in dir():
    cond_numbers = [v["condition_number"] for v in spectral_info.values()
                    if v.get("condition_number", float("inf")) < float("inf")]
    if cond_numbers:
        print()
        print(f"Condition Numbers:")
        print(f"  Mean: {np.mean(cond_numbers):.2f}, Max: {np.max(cond_numbers):.2f}")

print("=" * 70)
print("Analysis complete!")
'''
        ),
    ]


def get_all_analysis_cells(
    input_shapes: Dict[str, Tuple[int, ...]],
    device: str,
    dtype: str,
    gradient_loss_fn: str = "reconstruction",
) -> Dict[str, List[Dict[str, Any]]]:
    """Get all analysis cell templates organized by analysis type.

    Parameters
    ----------
    input_shapes : Dict[str, Tuple[int, ...]]
        Input shapes.
    device : str
        Device string.
    dtype : str
        Data type string.
    gradient_loss_fn : str
        Loss function for gradient analysis. Default is "reconstruction".

    Returns
    -------
    Dict[str, List[Dict[str, Any]]]
        Dictionary mapping analysis names to cell lists.
    """
    # Use first loss function if multiple provided
    if isinstance(gradient_loss_fn, list):
        gradient_loss_fn = gradient_loss_fn[0] if gradient_loss_fn else "reconstruction"

    return {
        "parameter_norms": _get_parameter_norm_cells(),
        "operator_norm": _get_operator_norm_cells(),
        "gradient_analysis": _get_gradient_cells(loss_fn=gradient_loss_fn),
        "spectral_analysis": _get_spectral_cells(),
        "rank_analysis": _get_rank_cells(),
        "lipschitz": _get_lipschitz_cells(),
        "activation_stats": _get_activation_cells(),
        "gradient_ratios": _get_gradient_ratio_cells(loss_fn=gradient_loss_fn),
        "precision_checks": _get_precision_cells(),
        "weight_distribution": _get_weight_distribution_cells(),
    }


def _get_parameter_norm_cells() -> List[Dict[str, Any]]:
    """Get parameter norm analysis cells."""
    return [
        _markdown_cell(
            "## Parameter Norm Analysis\n\n"
            "Analyze parameter norms to assess initialization scale:\n"
            "- **Frobenius norm**: Overall magnitude (L2 norm of flattened tensor)\n"
            "- **Operator norm**: Spectral norm (largest singular value)\n"
            "- **Sup-norm**: Maximum absolute value\n"
            "- **Std**: Standard deviation of values"
        ),
        _code_cell(
            '''
# Compute parameter norms
param_norms = {}
for name, param in module.named_parameters():
    param_data = param.data.float()

    frobenius = torch.linalg.norm(param_data.flatten()).item()
    sup_norm = param_data.abs().max().item()
    std = param_data.std().item()
    mean = param_data.mean().item()

    if param_data.dim() >= 2:
        reshaped = param_data.reshape(param_data.size(0), -1)
        operator_norm = torch.linalg.matrix_norm(reshaped, ord=2).item()
    else:
        operator_norm = frobenius

    param_norms[name] = {
        "frobenius": frobenius,
        "operator": operator_norm,
        "sup": sup_norm,
        "std": std,
        "mean": mean,
        "shape": list(param.shape),
        "numel": param.numel(),
    }

# Display results
try:
    import pandas as pd
    df = pd.DataFrame(param_norms).T
    df = df.round(6)
    print("Parameter Norm Analysis:")
    display(df)
except ImportError:
    print("Parameter Norm Analysis:")
    for name, norms in param_norms.items():
        print(f"  {name}: frob={norms['frobenius']:.4f}, op={norms['operator']:.4f}")
'''
        ),
        _code_cell(
            '''
# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, norm_type in zip(axes.flatten(), ["frobenius", "operator", "sup", "std"]):
    values = [v[norm_type] for v in param_norms.values()]
    names = list(param_norms.keys())

    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.split(".")[-1][:20] for n in names], fontsize=8)
    ax.set_title(f"{norm_type.title()} Norm")
    ax.set_xlabel("Norm Value")

plt.tight_layout()
plt.show()
'''
        ),
    ]

def _get_operator_norm_cells() -> List[Dict[str, Any]]:
    """Get operator norm estimation cells."""
    return [
        _markdown_cell(
            "## Effective Operator Norm Estimation\n\n"
            "Estimate the effective operator norm by computing max(||f(x)|| / ||x||) "
            "over various input distributions."
        ),
        _code_cell(
            f'''
# Operator norm estimation
distributions = ["normal", "uniform", "sparse", "large_magnitude", "small_magnitude"]
n_samples = 50

operator_norm_estimates = {{}}

for dist_name in distributions:
    ratios = []

    for _ in range(n_samples):
        inputs = {{}}
        total_input_norm_sq = 0.0

        for name, shape in INPUT_SHAPES.items():
            if dist_name == "normal":
                x = torch.randn(*shape, device=device, dtype=dtype)
            elif dist_name == "uniform":
                x = torch.rand(*shape, device=device, dtype=dtype) * 2 - 1
            elif dist_name == "sparse":
                x = torch.randn(*shape, device=device, dtype=dtype)
                x = x * (torch.rand_like(x) > 0.9)
            elif dist_name == "large_magnitude":
                x = torch.randn(*shape, device=device, dtype=dtype) * 100
            elif dist_name == "small_magnitude":
                x = torch.randn(*shape, device=device, dtype=dtype) * 1e-4
            else:
                x = torch.randn(*shape, device=device, dtype=dtype)

            inputs[name] = x
            total_input_norm_sq += torch.linalg.norm(x.flatten())**2

        input_norm = torch.sqrt(torch.tensor(total_input_norm_sq))
        if input_norm < 1e-8:
            continue

        with torch.no_grad():
            out = module(**inputs)
            if isinstance(out, tuple):
                out = out[0]

            y_norm = torch.linalg.norm(out.flatten())
            ratio = (y_norm / input_norm).item()

            if not np.isnan(ratio) and not np.isinf(ratio):
                ratios.append(ratio)

    if ratios:
        operator_norm_estimates[dist_name] = {{
            "max": max(ratios),
            "mean": np.mean(ratios),
            "std": np.std(ratios),
        }}

print("Effective Operator Norm Estimates:")
print("-" * 60)
for dist, stats in operator_norm_estimates.items():
    print(f"{{dist:20s}}: max={{stats['max']:.4f}}, mean={{stats['mean']:.4f}}")
'''
        ),
    ]

def _get_gradient_cells(loss_fn: str = "reconstruction") -> List[Dict[str, Any]]:
    """Get gradient analysis cells.

    Parameters
    ----------
    loss_fn : str
        Loss function to use. Options: "sum", "mse_random", "reconstruction", "variance".
    """
    # Generate loss computation code based on loss function
    loss_code = _get_loss_code(loss_fn)

    return [
        _markdown_cell(
            "## Gradient Analysis\n\n"
            f"Verify gradient existence and compute gradient norms using **{loss_fn}** loss.\n\n"
            "Available loss functions:\n"
            "- `sum`: Sum of all outputs (ensures all elements contribute)\n"
            "- `mse_random`: MSE to random target (balanced +/- gradients)\n"
            "- `reconstruction`: MSE(output, input) (tests identity-mapping)\n"
            "- `variance`: Negative output variance (detects collapse)\n"
        ),
        _code_cell(
            f'''
import torch.nn.functional as F

# Gradient analysis with {loss_fn} loss
inputs = {{}}
for name, shape in INPUT_SHAPES.items():
    inputs[name] = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

module.train()
out = module(**inputs)
if isinstance(out, tuple):
    out = out[0]

# Compute loss ({loss_fn})
{loss_code}
loss.backward()

gradient_info = {{}}
missing_grads = []
grad_param_ratios = {{}}

for name, param in module.named_parameters():
    if param.grad is None:
        missing_grads.append(name)
        gradient_info[name] = {{"has_grad": False, "norm": None}}
    else:
        grad_norm = torch.linalg.norm(param.grad.flatten()).item()
        param_norm = torch.linalg.norm(param.data.flatten()).item()
        gradient_info[name] = {{
            "has_grad": True,
            "norm": grad_norm,
            "mean": param.grad.mean().item(),
            "max": param.grad.abs().max().item(),
        }}
        # Compute gradient-to-parameter ratio
        if param_norm > 1e-10:
            grad_param_ratios[name] = grad_norm / param_norm
        else:
            grad_param_ratios[name] = float('inf') if grad_norm > 0 else 0.0

print("Gradient Analysis (loss: {loss_fn}):")
print("-" * 60)
if missing_grads:
    print(f"WARNING: {{len(missing_grads)}} parameters missing gradients:")
    for name in missing_grads:
        print(f"  - {{name}}")
else:
    print("All parameters have gradients.")

print("Gradient Norms:")
for name, info in gradient_info.items():
    if info["has_grad"]:
        print(f"  {{name:40s}}: {{info['norm']:.6f}}")

print("Gradient-to-Parameter Ratios:")
for name, ratio in grad_param_ratios.items():
    status = "OK" if 0.001 < ratio < 10 else "WARNING"
    print(f"  {{name:40s}}: {{ratio:.6f}} [{{status}}]")

module.zero_grad()
module.eval();
'''
        ),
    ]

def _get_loss_code(loss_fn: str) -> str:
    """Generate loss computation code for the specified loss function.

    Parameters
    ----------
    loss_fn : str
        Loss function name.

    Returns
    -------
    str
        Python code for computing the loss.
    """
    if loss_fn == "sum":
        return "loss = out.sum()"
    elif loss_fn == "mse_random":
        return "target = torch.randn_like(out)\nloss = F.mse_loss(out, target)"
    elif loss_fn == "reconstruction":
        # Check if output matches any input shape for reconstruction
        return (
            "loss = None\n"
            "for x in inputs.values():\n"
            "    if out.shape == x.shape:\n"
            "        loss = F.mse_loss(out, x)\n"
            "        break\n"
            "if loss is None:\n"
            "    loss = out.sum()  # Fallback to sum if shapes mismatch"
        )
    elif loss_fn == "variance":
        return "loss = -out.var()"
    else:
        return "loss = out.sum()  # fallback to sum"

def _get_spectral_cells() -> List[Dict[str, Any]]:
    """Get spectral analysis cells."""
    return [
        _markdown_cell(
            "## Spectral Analysis\n\n"
            "Compute spectral properties of weight matrices."
        ),
        _code_cell(
            '''
spectral_info = {}

for name, param in module.named_parameters():
    if param.dim() < 2:
        continue

    weight = param.data.float().reshape(param.size(0), -1)

    try:
        S = torch.linalg.svdvals(weight)
        spectral_info[name] = {
            "spectral_radius": S[0].item(),
            "min_singular": S[-1].item(),
            "condition_number": (S[0] / S[-1]).item() if S[-1] > 1e-10 else float("inf"),
            "effective_rank": (S > 1e-6).sum().item(),
        }
    except Exception as e:
        print(f"SVD failed for {name}: {e}")

print("Spectral Analysis:")
print("-" * 60)
for name, info in spectral_info.items():
    print(f"{name}:")
    print(f"  Spectral radius: {info['spectral_radius']:.4f}")
    print(f"  Condition number: {info['condition_number']:.2f}")
    print(f"  Effective rank: {info['effective_rank']}")
'''
        ),
    ]

def _get_rank_cells() -> List[Dict[str, Any]]:
    """Get rank analysis cells."""
    return [
        _markdown_cell("## Effective Rank Analysis\n\nAnalyze effective rank of weight matrices."),
        _code_cell(
            '''
rank_info = {}

for name, param in module.named_parameters():
    if param.dim() < 2:
        continue

    weight = param.data.float().reshape(param.size(0), -1)
    theoretical_max = min(weight.shape)

    S = torch.linalg.svdvals(weight)
    effective_rank = (S > 1e-6).sum().item()

    rank_info[name] = {
        "effective_rank": effective_rank,
        "theoretical_max": theoretical_max,
        "ratio": effective_rank / theoretical_max,
    }

print("Rank Analysis:")
print("-" * 60)
for name, info in rank_info.items():
    print(f"{name}: {info['effective_rank']}/{info['theoretical_max']} ({info['ratio']:.1%})")
'''
        ),
    ]

def _get_lipschitz_cells() -> List[Dict[str, Any]]:
    """Get Lipschitz estimation cells."""
    return [
        _markdown_cell("## Lipschitz Constant Estimation\n\nEstimate the Lipschitz constant."),
        _code_cell(
            f'''
n_samples = 50
perturbation_scale = 0.01
ratios = []

for _ in range(n_samples):
    inputs = {{}}
    deltas = {{}}
    total_delta_norm_sq = 0.0

    for name, shape in INPUT_SHAPES.items():
        x = torch.randn(*shape, device=device, dtype=dtype)
        delta = torch.randn(*shape, device=device, dtype=dtype) * perturbation_scale
        inputs[name] = x
        deltas[name] = delta
        total_delta_norm_sq += torch.linalg.norm(delta.flatten())**2

    inputs_perturbed = {{k: v + deltas[k] for k, v in inputs.items()}}

    with torch.no_grad():
        y1 = module(**inputs)
        y2 = module(**inputs_perturbed)
        if isinstance(y1, tuple):
            y1, y2 = y1[0], y2[0]

    input_diff = torch.sqrt(torch.tensor(total_delta_norm_sq)).cpu().item()
    output_diff = torch.linalg.norm((y2 - y1).flatten()).cpu().item()

    if input_diff > 1e-10:
        ratios.append(output_diff / input_diff)

    # Clean up large tensors
    del inputs, deltas, inputs_perturbed, y1, y2
    torch.cuda.empty_cache()

print("Lipschitz Estimation:")
print(f"  Empirical max: {{max(ratios):.4f}}")
print(f"  Empirical mean: {{np.mean(ratios):.4f}}")
'''
        ),
    ]

def _get_activation_cells() -> List[Dict[str, Any]]:
    """Get activation statistics cells."""
    return [
        _markdown_cell("## Activation Statistics\n\nAnalyze activation values at each layer."),
        _code_cell(
            f'''
activation_stats = {{}}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, torch.Tensor):
            out = output.detach().float()
            activation_stats[name] = {{
                "mean": out.mean().item(),
                "std": out.std().item(),
                "max": out.max().item(),
            }}
    return hook

hooks = []
for name, layer in module.named_modules():
    if name:
        hooks.append(layer.register_forward_hook(make_hook(name)))

with torch.no_grad():
    inputs = {{k: torch.randn(*s, device=device, dtype=dtype) for k, s in INPUT_SHAPES.items()}}
    module(**inputs)

for hook in hooks:
    hook.remove()

print("Activation Statistics:")
for name, stats in list(activation_stats.items())[:10]:
    print(f"  {{name}}: mean={{stats['mean']:.4f}}, std={{stats['std']:.4f}}")
'''
        ),
    ]

def _get_gradient_ratio_cells(loss_fn: str = "reconstruction") -> List[Dict[str, Any]]:
    """Get gradient ratio cells.

    Parameters
    ----------
    loss_fn : str
        Loss function to use.
    """
    loss_code = _get_loss_code(loss_fn)

    return [
        _markdown_cell(f"## Gradient Ratios\n\nAnalyze gradient flow between layers using **{loss_fn}** loss."),
        _code_cell(
            f'''
import torch.nn.functional as F

module.train()
module.zero_grad()

inputs = {{}}
for name, shape in INPUT_SHAPES.items():
    inputs[name] = torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)

out = module(**inputs)
if isinstance(out, tuple):
    out = out[0]

# Compute loss ({loss_fn})
{loss_code}
loss.backward()

layer_grads = {{}}
for name, param in module.named_parameters():
    if param.grad is None:
        continue
    layer = name.split(".")[0]
    if layer not in layer_grads:
        layer_grads[layer] = []
    layer_grads[layer].append(torch.linalg.norm(param.grad.flatten()).item())

layer_norms = {{layer: np.sqrt(sum(n**2 for n in norms)) for layer, norms in layer_grads.items()}}

print("Gradient Ratios (loss: {loss_fn}):")
layers = list(layer_norms.keys())
for i in range(len(layers) - 1):
    ratio = layer_norms[layers[i+1]] / (layer_norms[layers[i]] + 1e-10)
    status = "OK" if 0.1 < ratio < 10 else "WARNING"
    print(f"  {{layers[i]}} -> {{layers[i+1]}}: {{ratio:.4f}} [{{status}}]")

module.zero_grad()
module.eval();
'''
        ),
    ]

def _get_precision_cells() -> List[Dict[str, Any]]:
    """Get numerical precision cells."""
    return [
        _markdown_cell("## Numerical Precision\n\nCheck for numerical issues."),
        _code_cell(
            f'''
print("Numerical Precision Checks:")
print("-" * 60)

# Check parameters
for name, param in module.named_parameters():
    p = param.data.float()
    if torch.isnan(p).any():
        print(f"  WARNING: {{name}} contains NaN")
    if torch.isinf(p).any():
        print(f"  WARNING: {{name}} contains Inf")

# Check output
with torch.no_grad():
    inputs = {{k: torch.randn(*s, device=device, dtype=dtype) for k, s in INPUT_SHAPES.items()}}
    out = module(**inputs)
    if isinstance(out, tuple):
        out = out[0]

    if torch.isnan(out).any():
        print("  WARNING: Output contains NaN")
    elif torch.isinf(out).any():
        print("  WARNING: Output contains Inf")
    else:
        print("  Output OK (no NaN/Inf)")
'''
        ),
    ]

def _get_weight_distribution_cells() -> List[Dict[str, Any]]:
    """Get weight distribution cells."""
    return [
        _markdown_cell("## Weight Distributions\n\nVisualize weight value distributions."),
        _code_cell(
            '''
params_to_plot = [(n, p) for n, p in module.named_parameters() if p.dim() >= 2][:4]

if params_to_plot:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, param) in zip(axes, params_to_plot):
        values = param.data.cpu().float().flatten().numpy()
        ax.hist(values, bins=50, density=True, alpha=0.7)
        ax.axvline(values.mean(), color='r', linestyle='--', label=f'Mean: {values.mean():.4f}')
        ax.set_title(name.split(".")[-1][:20])
        ax.legend()

    plt.tight_layout()
    plt.show()
'''
        ),
    ]
