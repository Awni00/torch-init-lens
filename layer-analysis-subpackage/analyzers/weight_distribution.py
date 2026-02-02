"""
Weight Distribution Analyzer.

Analyzes and visualizes the distribution of weight values to assess
initialization quality.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
import numpy as np

from .base import BaseAnalyzer, AnalysisResult


class WeightDistributionAnalyzer(BaseAnalyzer):
    """Analyze weight value distributions.

    Provides statistical analysis and visualization of weight
    distributions to verify initialization quality.

    Parameters
    ----------
    module : nn.Module
        Module to analyze.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.

    Examples
    --------
    >>> model = nn.Linear(64, 32)
    >>> analyzer = WeightDistributionAnalyzer(model)
    >>> results = analyzer.analyze()
    """

    name = "weight_distribution"

    def __init__(
        self,
        module: nn.Module,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)

    def analyze(
        self,
        n_bins: int = 50,
    ) -> List[AnalysisResult]:
        """Analyze weight distributions.

        Parameters
        ----------
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        List[AnalysisResult]
            Weight distribution analysis results.
        """
        results = []
        dist_info: Dict[str, Dict[str, Any]] = {}

        for name, param in self.module.named_parameters():
            info = self._analyze_distribution(param.data, n_bins)
            dist_info[name] = info

            # Check for potential issues
            issues = []
            if abs(info["mean"]) > 0.1:
                issues.append(f"non-zero mean ({info['mean']:.4f})")
            if info["std"] > 1.0:
                issues.append(f"large std ({info['std']:.4f})")
            if info["skewness"] > 1.0 or info["skewness"] < -1.0:
                issues.append(f"skewed ({info['skewness']:.2f})")
            if info["kurtosis"] > 5.0:
                issues.append(f"heavy tails ({info['kurtosis']:.2f})")

            results.append(
                AnalysisResult(
                    name=f"dist_{name}",
                    value=info,
                    passed=len(issues) == 0,
                    message=f"mean={info['mean']:.4f}, std={info['std']:.4f}"
                    + (f" [{', '.join(issues)}]" if issues else ""),
                    details=info,
                    severity="warning" if issues else "info",
                )
            )

        # Summary
        if dist_info:
            all_means = [i["mean"] for i in dist_info.values()]
            all_stds = [i["std"] for i in dist_info.values()]

            results.append(
                AnalysisResult(
                    name="distribution_summary",
                    value={
                        "mean_of_means": float(np.mean(all_means)),
                        "mean_of_stds": float(np.mean(all_stds)),
                        "std_of_means": float(np.std(all_means)),
                        "n_parameters": len(dist_info),
                    },
                    passed=True,
                    message=f"Analyzed {len(dist_info)} parameter distributions",
                    details={"per_parameter": dist_info},
                )
            )

        return results

    def _analyze_distribution(
        self,
        weights: torch.Tensor,
        n_bins: int,
    ) -> Dict[str, Any]:
        """Analyze distribution of a single weight tensor.

        Parameters
        ----------
        weights : torch.Tensor
            Weight tensor.
        n_bins : int
            Number of histogram bins.

        Returns
        -------
        Dict[str, Any]
            Distribution statistics.
        """
        w = weights.float().flatten()

        # Basic statistics
        mean = w.mean().item()
        std = w.std().item()
        median = w.median().item()
        min_val = w.min().item()
        max_val = w.max().item()

        # Higher moments
        centered = w - mean
        if std > 1e-10:
            normalized = centered / std
            skewness = (normalized ** 3).mean().item()
            kurtosis = (normalized ** 4).mean().item() - 3  # Excess kurtosis
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Percentiles
        percentiles = {
            "p1": torch.quantile(w, 0.01).item(),
            "p5": torch.quantile(w, 0.05).item(),
            "p25": torch.quantile(w, 0.25).item(),
            "p75": torch.quantile(w, 0.75).item(),
            "p95": torch.quantile(w, 0.95).item(),
            "p99": torch.quantile(w, 0.99).item(),
        }

        # Histogram
        hist = torch.histc(w, bins=n_bins)
        hist = hist / hist.sum()  # Normalize

        return {
            "mean": mean,
            "std": std,
            "median": median,
            "min": min_val,
            "max": max_val,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "percentiles": percentiles,
            "histogram": hist.tolist(),
            "shape": tuple(weights.shape),
            "numel": weights.numel(),
        }

    def compare_to_expected(
        self,
        expected_mean: float = 0.0,
        expected_std: Optional[float] = None,
    ) -> List[AnalysisResult]:
        """Compare distributions to expected initialization.

        Parameters
        ----------
        expected_mean : float
            Expected mean (typically 0).
        expected_std : Optional[float]
            Expected standard deviation. If None, uses fan_in based estimate.

        Returns
        -------
        List[AnalysisResult]
            Comparison results.
        """
        results = []

        for name, param in self.module.named_parameters():
            w = param.data.float().flatten()

            actual_mean = w.mean().item()
            actual_std = w.std().item()

            # Compute expected std if not provided
            if expected_std is None:
                if param.dim() >= 2:
                    fan_in = param.size(1) * np.prod(param.shape[2:]) if param.dim() > 2 else param.size(1)
                    exp_std = 1.0 / np.sqrt(fan_in)
                else:
                    exp_std = 1.0
            else:
                exp_std = expected_std

            mean_diff = abs(actual_mean - expected_mean)
            std_ratio = actual_std / exp_std if exp_std > 0 else float('inf')

            passed = mean_diff < 0.1 and 0.5 < std_ratio < 2.0

            results.append(
                AnalysisResult(
                    name=f"init_check_{name}",
                    value={
                        "actual_mean": actual_mean,
                        "expected_mean": expected_mean,
                        "actual_std": actual_std,
                        "expected_std": exp_std,
                        "std_ratio": std_ratio,
                    },
                    passed=passed,
                    message=f"mean diff={mean_diff:.4f}, std ratio={std_ratio:.2f}",
                    severity="warning" if not passed else "info",
                )
            )

        return results

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Generate notebook cells for weight distribution analysis.

        Returns
        -------
        List[Dict[str, Any]]
            Notebook cells.
        """
        return [
            self._create_markdown_cell(
                "## Weight Distribution Analysis\n\n"
                "Analyze the distribution of weight values to verify initialization.\n\n"
                "Good initialization typically has:\n"
                "- Mean close to 0\n"
                "- Std scaled appropriately for layer size\n"
                "- Symmetric distribution (low skewness)\n"
                "- Light tails (kurtosis ~ 0)"
            ),
            self._create_code_cell(
                '''
import numpy as np

# Analyze weight distributions
dist_info = {}

for name, param in module.named_parameters():
    w = param.data.float().flatten()

    mean = w.mean().item()
    std = w.std().item()

    # Higher moments
    centered = w - mean
    if std > 1e-10:
        normalized = centered / std
        skewness = (normalized ** 3).mean().item()
        kurtosis = (normalized ** 4).mean().item() - 3
    else:
        skewness = kurtosis = 0

    dist_info[name] = {
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "min": w.min().item(),
        "max": w.max().item(),
    }

# Display results
print("Weight Distribution Analysis:")
print("-" * 80)
print(f"{'Parameter':<35} {'Mean':>10} {'Std':>10} {'Skew':>8} {'Kurt':>8}")
print("-" * 80)
for name, info in dist_info.items():
    short_name = name.split(".")[-1][:33]
    print(f"{short_name:<35} {info['mean']:>10.4f} {info['std']:>10.4f} "
          f"{info['skewness']:>8.2f} {info['kurtosis']:>8.2f}")
'''
            ),
            self._create_code_cell(
                '''
# Visualization: Weight distribution histograms
import matplotlib.pyplot as plt

# Select representative weights
params_to_plot = []
for name, param in module.named_parameters():
    if param.dim() >= 2 and param.numel() > 100:
        params_to_plot.append((name, param))
        if len(params_to_plot) >= 4:
            break

if params_to_plot:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, param) in zip(axes, params_to_plot):
        values = param.data.cpu().float().flatten().numpy()

        ax.hist(values, bins=50, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(values.mean(), color='r', linestyle='--',
                   label=f'Mean: {values.mean():.4f}')
        ax.axvline(0, color='k', linestyle='-', alpha=0.3)

        ax.set_title(name.split(".")[-1], fontsize=10)
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("weight_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()
'''
            ),
        ]
