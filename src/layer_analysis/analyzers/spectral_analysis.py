"""
Spectral Analyzer.

Analyzes spectral properties of weight matrices including eigenvalues,
spectral radius, and condition number.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from .base import BaseAnalyzer, AnalysisResult
from ..errors import safe_svd, NumericalInstabilityError


class SpectralAnalyzer(BaseAnalyzer):
    """Spectral analysis: eigenvalues, spectral radius, condition number.

    Computes spectral properties for 2D weight matrices:
    - Singular values
    - Spectral radius (largest singular value)
    - Condition number (ratio of max to min singular value)
    - Top-k eigenvalues

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    max_params_for_full_svd : int
        Maximum parameter count for full SVD. Larger matrices use
        randomized methods.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> analyzer = SpectralAnalyzer(model)
    >>> results = analyzer.analyze()
    """

    name = "spectral_analysis"

    def __init__(
        self,
        module: nn.Module,
        max_params_for_full_svd: int = 10000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.max_params_for_full_svd = max_params_for_full_svd

    def analyze(
        self,
        compute_eigenvalues: bool = True,
        compute_spectral_radius: bool = True,
        compute_condition_number: bool = True,
        top_k_eigenvalues: int = 10,
    ) -> List[AnalysisResult]:
        """Run spectral analysis on weight matrices.

        Parameters
        ----------
        compute_eigenvalues : bool
            Compute top-k singular values.
        compute_spectral_radius : bool
            Compute spectral radius (max singular value).
        compute_condition_number : bool
            Compute condition number.
        top_k_eigenvalues : int
            Number of singular values to compute.

        Returns
        -------
        List[AnalysisResult]
            Spectral analysis results per weight matrix.
        """
        results = []
        spectral_info: Dict[str, Dict[str, Any]] = {}

        # Get 2D weights
        weights = self._get_2d_weights(self.module)

        if not weights:
            return [
                AnalysisResult(
                    name="spectral_analysis",
                    value=None,
                    passed=True,
                    message="No 2D weight matrices to analyze",
                    severity="warning",
                )
            ]

        for name, weight in weights:
            try:
                info = self._analyze_weight(
                    weight,
                    compute_eigenvalues,
                    compute_spectral_radius,
                    compute_condition_number,
                    top_k_eigenvalues,
                )
                spectral_info[name] = info
            except NumericalInstabilityError as e:
                spectral_info[name] = {"error": str(e)}

        # Create per-weight results
        for name, info in spectral_info.items():
            if "error" in info:
                results.append(
                    AnalysisResult(
                        name=f"spectral_{name}",
                        value=None,
                        passed=False,
                        message=f"Spectral analysis failed: {info['error']}",
                        severity="warning",
                    )
                )
            else:
                results.append(
                    AnalysisResult(
                        name=f"spectral_{name}",
                        value=info,
                        passed=True,
                        message=f"Spectral radius: {info.get('spectral_radius', 0):.4f}, "
                        f"Condition: {info.get('condition_number', 0):.2f}",
                        details=info,
                    )
                )

        # Add summary
        valid_infos = [i for i in spectral_info.values() if "error" not in i]
        if valid_infos:
            radii = [i["spectral_radius"] for i in valid_infos]
            cond_numbers = [
                i["condition_number"]
                for i in valid_infos
                if i["condition_number"] < float("inf")
            ]

            results.append(
                AnalysisResult(
                    name="spectral_summary",
                    value={
                        "max_spectral_radius": max(radii),
                        "mean_spectral_radius": sum(radii) / len(radii),
                        "max_condition_number": max(cond_numbers) if cond_numbers else float("inf"),
                    },
                    passed=True,
                    message=f"Analyzed {len(valid_infos)} weight matrices",
                    details={"per_weight": spectral_info},
                )
            )

        return results

    def _analyze_weight(
        self,
        weight: torch.Tensor,
        compute_eigenvalues: bool,
        compute_spectral_radius: bool,
        compute_condition_number: bool,
        top_k: int,
    ) -> Dict[str, Any]:
        """Analyze a single weight matrix.

        Parameters
        ----------
        weight : torch.Tensor
            Weight matrix (2D or reshaped).
        compute_eigenvalues : bool
            Whether to compute eigenvalues.
        compute_spectral_radius : bool
            Whether to compute spectral radius.
        compute_condition_number : bool
            Whether to compute condition number.
        top_k : int
            Number of singular values to return.

        Returns
        -------
        Dict[str, Any]
            Spectral properties.
        """
        # Reshape to 2D
        weight_2d = weight.float().reshape(weight.size(0), -1)

        info: Dict[str, Any] = {
            "shape": tuple(weight.shape),
            "reshaped": tuple(weight_2d.shape),
        }

        # Compute SVD
        U, S, Vh = safe_svd(weight_2d)

        if S is not None and S.numel() > 0:
            if compute_spectral_radius:
                info["spectral_radius"] = S[0].item()

            if compute_condition_number:
                min_sv = S[-1].item() if S.numel() > 1 else S[0].item()
                if min_sv > 1e-10:
                    info["condition_number"] = (S[0] / min_sv).item()
                else:
                    info["condition_number"] = float("inf")

            if compute_eigenvalues:
                k = min(top_k, S.numel())
                info["top_singular_values"] = S[:k].tolist()
                info["effective_rank"] = (S > 1e-6).sum().item()

        return info

    def compute_spectral_radius(self, weight: torch.Tensor) -> float:
        """Compute spectral radius of a weight matrix.

        Parameters
        ----------
        weight : torch.Tensor
            Weight matrix.

        Returns
        -------
        float
            Spectral radius (largest singular value).
        """
        weight_2d = weight.float().reshape(weight.size(0), -1)
        _, S, _ = safe_svd(weight_2d)
        return S[0].item() if S is not None and S.numel() > 0 else 0.0

    def compute_condition_number(self, weight: torch.Tensor) -> float:
        """Compute condition number of a weight matrix.

        Parameters
        ----------
        weight : torch.Tensor
            Weight matrix.

        Returns
        -------
        float
            Condition number (sigma_max / sigma_min).
        """
        weight_2d = weight.float().reshape(weight.size(0), -1)
        _, S, _ = safe_svd(weight_2d)

        if S is None or S.numel() == 0:
            return float("inf")

        min_sv = S[-1].item()
        if min_sv > 1e-10:
            return (S[0] / min_sv).item()
        return float("inf")

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for spectral analysis.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        return [
            self._create_markdown_cell(
                "## Spectral Analysis\n\n"
                "Compute spectral properties of weight matrices:\n"
                "- **Singular values**: Measure of matrix \"size\" in different directions\n"
                "- **Spectral radius**: Maximum singular value (determines amplification)\n"
                "- **Condition number**: Ratio of max to min singular value "
                "(measures numerical sensitivity)"
            ),
            self._create_code_cell(
                '''
# Spectral analysis for 2D weight matrices
spectral_info = {}

for name, param in module.named_parameters():
    if param.dim() < 2:
        continue

    weight = param.data.float().reshape(param.size(0), -1)

    try:
        # Compute singular values
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        spectral_info[name] = {
            "spectral_radius": S[0].item(),
            "min_singular": S[-1].item(),
            "condition_number": (S[0] / S[-1]).item() if S[-1] > 1e-10 else float("inf"),
            "top_5_singular": S[:5].tolist(),
            "effective_rank": (S > 1e-6).sum().item(),
        }
    except Exception as e:
        print(f"SVD failed for {name}: {e}")

# Display results
print("Spectral Analysis:")
print("-" * 60)
for name, info in spectral_info.items():
    print(f"\\n{name}:")
    print(f"  Spectral radius: {info['spectral_radius']:.6f}")
    print(f"  Min singular:    {info['min_singular']:.6f}")
    print(f"  Condition number: {info['condition_number']:.2f}")
    print(f"  Effective rank:   {info['effective_rank']}")
'''
            ),
            self._create_code_cell(
                '''
# Visualization: Singular value distribution
import matplotlib.pyplot as plt
import numpy as np

# Select a few representative weights
weights_to_plot = []
for name, param in module.named_parameters():
    if param.dim() >= 2 and param.numel() > 100:
        weights_to_plot.append((name, param))
        if len(weights_to_plot) >= 4:
            break

if weights_to_plot:
    fig, axes = plt.subplots(1, len(weights_to_plot), figsize=(4*len(weights_to_plot), 4))
    if len(weights_to_plot) == 1:
        axes = [axes]

    for ax, (name, param) in zip(axes, weights_to_plot):
        weight = param.data.float().reshape(param.size(0), -1)
        S = torch.linalg.svdvals(weight)

        ax.semilogy(S.cpu().numpy())
        ax.set_title(name.split(".")[-1], fontsize=10)
        ax.set_xlabel("Singular Value Index")
        ax.set_ylabel("Singular Value (log scale)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("singular_values.png", dpi=150, bbox_inches="tight")
    plt.show()
'''
            ),
        ]
