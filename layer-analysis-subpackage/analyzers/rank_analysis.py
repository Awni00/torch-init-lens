"""
Rank Analyzer.

Analyzes effective rank of weight matrices to detect potentially
low-rank or degenerate initializations.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAnalyzer, AnalysisResult
from ..errors import safe_svd


class RankAnalyzer(BaseAnalyzer):
    """Analyze effective rank of weight matrices.

    Effective rank measures how many singular values significantly
    contribute to the matrix, providing insight into whether the
    initialization is accidentally low-rank.

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    tolerance : float
        Threshold for considering singular values as non-zero.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> analyzer = RankAnalyzer(model)
    >>> results = analyzer.analyze()
    """

    name = "rank_analysis"

    def __init__(
        self,
        module: nn.Module,
        tolerance: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.tolerance = tolerance

    def analyze(self) -> List[AnalysisResult]:
        """Analyze effective rank of all weight matrices.

        Returns
        -------
        List[AnalysisResult]
            Rank analysis results.
        """
        results = []
        rank_info: Dict[str, Dict[str, Any]] = {}

        weights = self._get_2d_weights(self.module)

        if not weights:
            return [
                AnalysisResult(
                    name="rank_analysis",
                    value=None,
                    passed=True,
                    message="No 2D weight matrices to analyze",
                    severity="warning",
                )
            ]

        for name, weight in weights:
            try:
                info = self._analyze_rank(weight)
                rank_info[name] = info

                # Check if rank is suspiciously low
                theoretical_max = min(weight.size(0), weight.reshape(weight.size(0), -1).size(1))
                is_low_rank = info["effective_rank"] < theoretical_max * 0.5

                results.append(
                    AnalysisResult(
                        name=f"rank_{name}",
                        value=info["effective_rank"],
                        passed=not is_low_rank,
                        message=f"Effective rank: {info['effective_rank']}/{theoretical_max} "
                        f"({info['rank_ratio']:.1%})",
                        details=info,
                        severity="warning" if is_low_rank else "info",
                    )
                )
            except Exception as e:
                results.append(
                    AnalysisResult(
                        name=f"rank_{name}",
                        value=None,
                        passed=False,
                        message=f"Rank analysis failed: {e}",
                        severity="warning",
                    )
                )

        # Summary
        valid_infos = [i for i in rank_info.values() if "effective_rank" in i]
        if valid_infos:
            ratios = [i["rank_ratio"] for i in valid_infos]
            results.append(
                AnalysisResult(
                    name="rank_summary",
                    value={
                        "mean_rank_ratio": float(np.mean(ratios)),
                        "min_rank_ratio": float(min(ratios)),
                    },
                    passed=min(ratios) > 0.5,
                    message=f"Mean rank ratio: {np.mean(ratios):.1%}",
                    details={"per_weight": rank_info},
                )
            )

        return results

    def _analyze_rank(self, weight: torch.Tensor) -> Dict[str, Any]:
        """Analyze rank of a single weight matrix.

        Parameters
        ----------
        weight : torch.Tensor
            Weight matrix.

        Returns
        -------
        Dict[str, Any]
            Rank information.
        """
        weight_2d = weight.float().reshape(weight.size(0), -1)
        theoretical_max = min(weight_2d.shape)

        _, S, _ = safe_svd(weight_2d)

        if S is None or S.numel() == 0:
            return {
                "effective_rank": 0,
                "theoretical_max": theoretical_max,
                "rank_ratio": 0.0,
            }

        # Effective rank using tolerance
        effective_rank = (S > self.tolerance).sum().item()

        # Stable rank (squared Frobenius norm / squared spectral norm)
        frobenius_sq = (S ** 2).sum().item()
        spectral_sq = (S[0] ** 2).item()
        stable_rank = frobenius_sq / spectral_sq if spectral_sq > 0 else 0

        # Nuclear rank (sum of singular values / max singular value)
        nuclear_rank = S.sum().item() / S[0].item() if S[0] > 0 else 0

        # Entropy-based effective rank
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
        entropy_rank = np.exp(entropy)

        return {
            "effective_rank": effective_rank,
            "theoretical_max": theoretical_max,
            "rank_ratio": effective_rank / theoretical_max if theoretical_max > 0 else 0,
            "stable_rank": stable_rank,
            "nuclear_rank": nuclear_rank,
            "entropy_rank": entropy_rank,
            "shape": tuple(weight.shape),
        }

    def compute_stable_rank(self, weight: torch.Tensor) -> float:
        """Compute stable rank of a weight matrix.

        Stable rank = ||W||_F^2 / ||W||_2^2

        Parameters
        ----------
        weight : torch.Tensor
            Weight matrix.

        Returns
        -------
        float
            Stable rank value.
        """
        weight_2d = weight.float().reshape(weight.size(0), -1)
        frobenius_sq = torch.linalg.norm(weight_2d, ord="fro") ** 2
        spectral_sq = torch.linalg.matrix_norm(weight_2d, ord=2) ** 2
        return (frobenius_sq / spectral_sq).item() if spectral_sq > 0 else 0

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for rank analysis.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        return [
            self._create_markdown_cell(
                "## Effective Rank Analysis\n\n"
                "Analyze the effective rank of weight matrices:\n"
                "- **Effective rank**: Number of singular values above threshold\n"
                "- **Stable rank**: ||W||_F^2 / ||W||_2^2 (robust to noise)\n"
                "- **Entropy rank**: exp(entropy of normalized singular values)\n\n"
                "Low rank relative to theoretical max may indicate degenerate initialization."
            ),
            self._create_code_cell(
                '''
import numpy as np

# Rank analysis for 2D weight matrices
rank_info = {}
tolerance = 1e-6

for name, param in module.named_parameters():
    if param.dim() < 2:
        continue

    weight = param.data.float().reshape(param.size(0), -1)
    theoretical_max = min(weight.shape)

    try:
        S = torch.linalg.svdvals(weight)

        # Effective rank
        effective_rank = (S > tolerance).sum().item()

        # Stable rank
        frobenius_sq = (S ** 2).sum().item()
        spectral_sq = (S[0] ** 2).item()
        stable_rank = frobenius_sq / spectral_sq if spectral_sq > 0 else 0

        # Entropy rank
        S_norm = S / S.sum()
        entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()
        entropy_rank = np.exp(entropy)

        rank_info[name] = {
            "effective_rank": effective_rank,
            "theoretical_max": theoretical_max,
            "rank_ratio": effective_rank / theoretical_max,
            "stable_rank": stable_rank,
            "entropy_rank": entropy_rank,
        }
    except Exception as e:
        print(f"Rank analysis failed for {name}: {e}")

# Display results
print("Rank Analysis:")
print("-" * 70)
print(f"{'Parameter':<30} {'Eff Rank':>10} {'Max':>6} {'Ratio':>8} {'Stable':>8}")
print("-" * 70)
for name, info in rank_info.items():
    short_name = name.split(".")[-1][:28]
    print(f"{short_name:<30} {info['effective_rank']:>10} {info['theoretical_max']:>6} "
          f"{info['rank_ratio']:>8.1%} {info['stable_rank']:>8.2f}")
'''
            ),
        ]
