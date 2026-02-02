"""
Parameter Norm Analyzer.

Analyzes parameter norms including Frobenius norm, operator norm,
sup-norm, and standard deviation.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from .base import BaseAnalyzer, AnalysisResult


class ParameterNormAnalyzer(BaseAnalyzer):
    """Analyze parameter norms to assess initialization scale.

    Computes various norm metrics for each parameter in the module:
    - Frobenius norm: L2 norm of flattened tensor
    - Operator norm: Spectral norm (largest singular value)
    - Sup-norm: Maximum absolute value
    - Standard deviation: Spread of parameter values
    - Mean: Average parameter value

    Parameters
    ----------
    module : nn.Module
        The module to analyze.
    include_buffers : bool
        Whether to include buffers in the analysis.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> analyzer = ParameterNormAnalyzer(model)
    >>> results = analyzer.analyze()
    >>> for r in results:
    ...     print(f"{r.name}: {r.value:.4f}")
    """

    name = "parameter_norms"

    def __init__(
        self,
        module: nn.Module,
        include_buffers: bool = False,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.include_buffers = include_buffers

    def analyze(
        self,
        norm_types: Optional[List[str]] = None,
    ) -> List[AnalysisResult]:
        """Compute parameter norms.

        Parameters
        ----------
        norm_types : Optional[List[str]]
            Norm types to compute. Options: "frobenius", "operator", "sup", "std", "mean".
            If None, computes all.

        Returns
        -------
        List[AnalysisResult]
            Analysis results with per-parameter breakdown.
        """
        if norm_types is None:
            norm_types = ["frobenius", "operator", "sup", "std", "mean"]

        results = []
        param_norms: Dict[str, Dict[str, float]] = {}

        # Collect parameters
        params = list(self.module.named_parameters())
        if self.include_buffers:
            params.extend(list(self.module.named_buffers()))

        if not params:
            return [
                AnalysisResult(
                    name="parameter_norms",
                    value=None,
                    passed=True,
                    message="No parameters to analyze",
                    severity="warning",
                )
            ]

        # Compute norms for each parameter
        for name, param in params:
            param_data = param.data.float()  # Ensure float32 for stability
            norms: Dict[str, float] = {}

            if "frobenius" in norm_types:
                norms["frobenius"] = self.compute_frobenius_norm(param_data)

            if "operator" in norm_types:
                norms["operator"] = self.compute_operator_norm(param_data)

            if "sup" in norm_types:
                norms["sup"] = self.compute_sup_norm(param_data)

            if "std" in norm_types:
                norms["std"] = self.compute_std(param_data)

            if "mean" in norm_types:
                norms["mean"] = self.compute_mean(param_data)

            norms["numel"] = param.numel()
            norms["shape"] = tuple(param.shape)

            param_norms[name] = norms

        # Create per-parameter results showing all norm types
        for param_name, norms in param_norms.items():
            # Build compact message showing all norm types for this parameter
            norm_parts = []
            for norm_type in norm_types:
                if norm_type in norms:
                    norm_parts.append(f"{norm_type}={norms[norm_type]:.4f}")
            
            message = ", ".join(norm_parts)
            
            results.append(
                AnalysisResult(
                    name=f"param_{param_name}",
                    value={
                        norm_type: norms[norm_type]
                        for norm_type in norm_types
                        if norm_type in norms
                    },
                    passed=True,
                    message=message,
                    details={
                        "shape": norms["shape"],
                        "numel": norms["numel"],
                    },
                )
            )

        # Add summary result
        results.append(
            AnalysisResult(
                name="parameter_norms_summary",
                value=param_norms,
                passed=True,
                message=f"Analyzed {len(param_norms)} parameters",
                details={
                    "total_params": sum(n["numel"] for n in param_norms.values()),
                    "parameter_names": list(param_norms.keys()),
                },
            )
        )

        return results

    def compute_frobenius_norm(self, param: torch.Tensor) -> float:
        """Compute Frobenius norm (L2 norm of flattened tensor).

        Parameters
        ----------
        param : torch.Tensor
            Parameter tensor.

        Returns
        -------
        float
            Frobenius norm value.
        """
        return torch.linalg.norm(param.flatten()).item()

    def compute_operator_norm(self, param: torch.Tensor) -> float:
        """Compute operator (spectral) norm.

        For 2D matrices, this is the largest singular value.
        For other tensors, reshapes to 2D first.

        Parameters
        ----------
        param : torch.Tensor
            Parameter tensor.

        Returns
        -------
        float
            Operator norm value.
        """
        if param.dim() < 2:
            # For 1D, operator norm equals Frobenius norm
            return self.compute_frobenius_norm(param)

        # Reshape to 2D: (first_dim, product of rest)
        reshaped = param.reshape(param.size(0), -1)

        try:
            # Use matrix_norm for 2D
            return torch.linalg.matrix_norm(reshaped, ord=2).item()
        except Exception:
            # Fallback to SVD
            try:
                s = torch.linalg.svdvals(reshaped)
                return s[0].item()
            except Exception:
                return float("nan")

    def compute_sup_norm(self, param: torch.Tensor) -> float:
        """Compute sup-norm (maximum absolute value).

        Parameters
        ----------
        param : torch.Tensor
            Parameter tensor.

        Returns
        -------
        float
            Sup-norm value.
        """
        return param.abs().max().item()

    def compute_std(self, param: torch.Tensor) -> float:
        """Compute standard deviation of parameter values.

        Parameters
        ----------
        param : torch.Tensor
            Parameter tensor.

        Returns
        -------
        float
            Standard deviation value.
        """
        return param.std().item()

    def compute_mean(self, param: torch.Tensor) -> float:
        """Compute mean of parameter values.

        Parameters
        ----------
        param : torch.Tensor
            Parameter tensor.

        Returns
        -------
        float
            Mean value.
        """
        return param.mean().item()

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for parameter norm analysis.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        return [
            self._create_markdown_cell(
                "## Parameter Norm Analysis\n\n"
                "Analyze parameter norms to assess initialization scale:\n"
                "- **Frobenius norm**: Overall magnitude (L2 norm of flattened tensor)\n"
                "- **Operator norm**: Spectral norm (largest singular value)\n"
                "- **Sup-norm**: Maximum absolute value\n"
                "- **Std**: Standard deviation of values\n"
                "- **Mean**: Average parameter value"
            ),
            self._create_code_cell(
                '''
# Compute parameter norms
param_norms = {}
for name, param in module.named_parameters():
    param_data = param.data.float()  # Ensure float32 for numerical stability

    frobenius = torch.linalg.norm(param_data.flatten()).item()
    sup_norm = param_data.abs().max().item()
    std = param_data.std().item()
    mean = param_data.mean().item()

    # Operator norm (for 2D weights)
    if param_data.dim() >= 2:
        reshaped = param_data.reshape(param_data.size(0), -1)
        operator_norm = torch.linalg.matrix_norm(reshaped, ord=2).item()
    else:
        operator_norm = frobenius  # 1D case

    param_norms[name] = {
        "frobenius": frobenius,
        "operator": operator_norm,
        "sup": sup_norm,
        "std": std,
        "mean": mean,
        "shape": list(param.shape),
        "numel": param.numel(),
    }

# Display results as table
try:
    import pandas as pd
    df = pd.DataFrame(param_norms).T
    df = df.round(6)
    print("Parameter Norm Analysis:")
    print(df.to_string())
except ImportError:
    print("Parameter Norm Analysis:")
    for name, norms in param_norms.items():
        print(f"  {name}: frob={norms['frobenius']:.4f}, "
              f"op={norms['operator']:.4f}, sup={norms['sup']:.4f}")
'''
            ),
            self._create_code_cell(
                '''
# Visualization: Parameter norm distribution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, norm_type in zip(axes.flatten(), ["frobenius", "operator", "sup", "std"]):
    values = [v[norm_type] for v in param_norms.values()]
    names = list(param_norms.keys())

    ax.barh(range(len(values)), values)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.split(".")[-1] for n in names], fontsize=8)
    ax.set_title(f"{norm_type.title()} Norm")
    ax.set_xlabel("Norm Value")

plt.tight_layout()
plt.savefig("parameter_norms.png", dpi=150, bbox_inches="tight")
plt.show()
'''
            ),
        ]
