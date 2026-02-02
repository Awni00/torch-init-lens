"""
Analysis Runner.

This module provides the AnalysisRunner class that orchestrates all
analyzers and produces comprehensive analysis reports.
"""

from typing import List, Dict, Any, Optional, Type, Tuple
from pathlib import Path
import json
import torch
import torch.nn as nn

from .config import AnalysisConfig
from .introspection import load_module_class, merge_kwargs, validate_kwargs, infer_device
from .inputs.generators import InputGenerator
from .analyzers.base import AnalysisResult
from .analyzers.parameter_norms import ParameterNormAnalyzer
from .analyzers.operator_norm import OperatorNormEstimator
from .analyzers.gradient_analysis import GradientAnalyzer
from .analyzers.spectral_analysis import SpectralAnalyzer
from .analyzers.rank_analysis import RankAnalyzer
from .analyzers.lipschitz import LipschitzEstimator
from .analyzers.activation_stats import ActivationStatsAnalyzer
from .analyzers.gradient_ratios import GradientRatioAnalyzer
from .analyzers.numerical_precision import NumericalPrecisionChecker
from .analyzers.weight_distribution import WeightDistributionAnalyzer


def color_text(text: str, color: str) -> str:
    """Colorize text using ANSI escape codes.

    Parameters
    ----------
    text : str
        Text to colorize.
    color : str
        Color name ('green', 'orange', 'red').

    Returns
    -------
    str
        Colorized text.
    """
    colors = {
        "green": "\033[92m",
        "orange": "\033[93m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }
    code = colors.get(color, "")
    reset = colors["reset"] if code else ""
    return f"{code}{text}{reset}"


class AnalysisRunner:
    """Orchestrates module initialization analysis.

    Parameters
    ----------
    module_identifier : str
        Module identifier (e.g., "models.abstractor@RelationalAttention").
    config : AnalysisConfig
        Analysis configuration.
    module_kwargs : Optional[Dict[str, Any]]
        Keyword arguments for module instantiation.

    Examples
    --------
    >>> config = AnalysisConfig(input_shape=(2, 16, 64))
    >>> runner = AnalysisRunner("models.abstractor@DualAttention", config)
    >>> results = runner.run_all()
    >>> runner.print_summary(results)
    """

    def __init__(
        self,
        module_identifier: str,
        config: AnalysisConfig,
        module_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.module_identifier = module_identifier
        self.config = config
        self.module_kwargs = module_kwargs or {}

        # Load module class
        self.module_class = load_module_class(module_identifier)

        # Create module instance
        self.module = self._create_module()

        # Get device and dtype
        self.device = config.get_torch_device()
        self.dtype = config.get_torch_dtype()

        # Create input generator
        self.input_generator = InputGenerator(
            batch_size=config.batch_size,
            device=self.device,
            dtype=self.dtype,
        )

    def _create_module(self) -> nn.Module:
        """Create and configure the module instance.

        Returns
        -------
        nn.Module
            Configured module instance.
        """
        # Get input shape for d_model derivation
        input_shape = self.config.input_shapes.get("x")

        # Merge defaults, derived values, and user kwargs
        kwargs = merge_kwargs(self.module_class, self.module_kwargs, input_shape)

        # Validate all required parameters are present and no invalid ones
        validate_kwargs(self.module_class, kwargs, user_kwargs=self.module_kwargs)

        # Create module
        module = self.module_class(**kwargs)
        module = module.to(
            device=self.config.get_torch_device(),
            dtype=self.config.get_torch_dtype()
        )
        module.eval()
        return module

    def run_all(
        self,
        forward_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[AnalysisResult]]:
        """Run all enabled analyses.

        Parameters
        ----------
        forward_kwargs : Optional[Dict[str, Any]]
            Additional arguments for forward passes.

        Returns
        -------
        Dict[str, List[AnalysisResult]]
            Results organized by analysis type.
        """
        forward_kwargs = forward_kwargs or {}
        results: Dict[str, List[AnalysisResult]] = {}

        if self.config.run_parameter_norms:
            results["parameter_norms"] = self._run_parameter_norms()

        if self.config.run_operator_norm:
            results["operator_norm"] = self._run_operator_norm(forward_kwargs)

        if self.config.run_gradient_analysis:
            results["gradient_analysis"] = self._run_gradient_analysis(forward_kwargs)

        if self.config.run_spectral_analysis:
            results["spectral_analysis"] = self._run_spectral_analysis()

        if self.config.run_rank_analysis:
            results["rank_analysis"] = self._run_rank_analysis()

        if self.config.run_lipschitz:
            results["lipschitz"] = self._run_lipschitz(forward_kwargs)

        if self.config.run_activation_stats:
            results["activation_stats"] = self._run_activation_stats(forward_kwargs)

        if self.config.run_gradient_ratios:
            results["gradient_ratios"] = self._run_gradient_ratios(forward_kwargs)

        if self.config.run_precision_checks:
            results["precision_checks"] = self._run_precision_checks(forward_kwargs)

        if self.config.run_weight_distribution:
            results["weight_distribution"] = self._run_weight_distribution()

        return results

    def _run_parameter_norms(self) -> List[AnalysisResult]:
        """Run parameter norm analysis."""
        analyzer = ParameterNormAnalyzer(
            self.module,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze()

    def _run_operator_norm(self, forward_kwargs: Dict[str, Any]) -> List[AnalysisResult]:
        """Run operator norm estimation."""
        analyzer = OperatorNormEstimator(
            self.module,
            input_shapes=self.config.input_shapes,
            n_samples=self.config.n_samples,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(
            distributions=self.config.input_distributions,
            forward_kwargs=forward_kwargs,
        )

    def _run_gradient_analysis(self, forward_kwargs: Dict[str, Any]) -> List[AnalysisResult]:
        """Run gradient analysis."""
        analyzer = GradientAnalyzer(
            self.module,
            input_shapes=self.config.input_shapes,
            loss_fn=self.config.gradient_loss_fn,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(forward_kwargs=forward_kwargs)

    def _run_spectral_analysis(self) -> List[AnalysisResult]:
        """Run spectral analysis."""
        analyzer = SpectralAnalyzer(
            self.module,
            max_params_for_full_svd=self.config.max_params_for_full_svd,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(top_k_eigenvalues=self.config.top_k_eigenvalues)

    def _run_rank_analysis(self) -> List[AnalysisResult]:
        """Run rank analysis."""
        analyzer = RankAnalyzer(
            self.module,
            tolerance=self.config.svd_tolerance,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze()

    def _run_lipschitz(self, forward_kwargs: Dict[str, Any]) -> List[AnalysisResult]:
        """Run Lipschitz estimation."""
        analyzer = LipschitzEstimator(
            self.module,
            input_shapes=self.config.input_shapes,
            n_iterations=self.config.lipschitz_n_iterations,
            n_samples=self.config.n_samples,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(forward_kwargs=forward_kwargs)

    def _run_activation_stats(self, forward_kwargs: Dict[str, Any]) -> List[AnalysisResult]:
        """Run activation statistics analysis."""
        analyzer = ActivationStatsAnalyzer(
            self.module,
            input_shapes=self.config.input_shapes,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(forward_kwargs=forward_kwargs)

    def _run_gradient_ratios(self, forward_kwargs: Dict[str, Any]) -> List[AnalysisResult]:
        """Run gradient ratio analysis."""
        # Use the first loss function from the config for gradient ratios
        loss_fn = self.config.gradient_loss_fn
        if isinstance(loss_fn, list):
            loss_fn = loss_fn[0] if loss_fn else "reconstruction"
        analyzer = GradientRatioAnalyzer(
            self.module,
            input_shapes=self.config.input_shapes,
            loss_fn=loss_fn,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(forward_kwargs=forward_kwargs)

    def _run_precision_checks(self, forward_kwargs: Dict[str, Any]) -> List[AnalysisResult]:
        """Run numerical precision checks."""
        analyzer = NumericalPrecisionChecker(
            self.module,
            input_shapes=self.config.input_shapes,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(forward_kwargs=forward_kwargs)

    def _run_weight_distribution(self) -> List[AnalysisResult]:
        """Run weight distribution analysis."""
        analyzer = WeightDistributionAnalyzer(
            self.module,
            device=self.device,
            dtype=self.dtype,
        )
        return analyzer.analyze(n_bins=self.config.histogram_bins)

    def print_summary(self, results: Dict[str, List[AnalysisResult]]) -> None:
        """Print a summary of the analysis results.

        Parameters
        ----------
        results : Dict[str, List[AnalysisResult]]
            Results from run_all().
        """
        print("=" * 70)
        print(f"MODULE INITIALIZATION ANALYSIS: {self.module_identifier}")
        print("=" * 70)

        total_params = sum(p.numel() for p in self.module.parameters())
        trainable_params = sum(
            p.numel() for p in self.module.parameters() if p.requires_grad
        )
        print(f"\nParameters: {total_params:,} total, {trainable_params:,} trainable")
        print(f"Device: {self.device}, Dtype: {self.dtype}")
        print(f"Input shapes: {self.config.input_shapes}")

        passed_count = 0
        failed_count = 0
        warning_count = 0

        for analysis_type, analysis_results in results.items():
            print(f"\n--- {analysis_type.upper().replace('_', ' ')} ---")

            for result in analysis_results:
                if result.passed:
                    status = color_text("PASS", "green")
                    passed_count += 1
                else:
                    if result.severity == "warning":
                        status = color_text("WARN", "orange")
                        warning_count += 1
                    else:
                        status = color_text("FAIL", "red")
                        failed_count += 1

                print(f"  [{status}] {result.name}: {result.message}")

        print("\n" + "=" * 70)
        print(f"SUMMARY: {passed_count} passed, {warning_count} warnings, {failed_count} failed")
        print("=" * 70)

    def save_json(
        self,
        results: Dict[str, List[AnalysisResult]],
        path: Path,
    ) -> None:
        """Save results to JSON file.

        Parameters
        ----------
        results : Dict[str, List[AnalysisResult]]
            Results from run_all().
        path : Path
            Output path.
        """
        output = {
            "module": self.module_identifier,
            "config": self.config.to_dict(),
            "results": {
                analysis_type: [r.to_dict() for r in analysis_results]
                for analysis_type, analysis_results in results.items()
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    def save_markdown(
        self,
        results: Dict[str, List[AnalysisResult]],
        path: Path,
    ) -> None:
        """Save results to Markdown file.

        Parameters
        ----------
        results : Dict[str, List[AnalysisResult]]
            Results from run_all().
        path : Path
            Output path.
        """
        lines = [
            f"# Module Initialization Analysis: {self.module_identifier}",
            "",
            "## Configuration",
            f"- Input shapes: {self.config.input_shapes}",
            f"- Device: {self.config.device}",
            f"- Dtype: {self.config.dtype}",
            "",
        ]

        for analysis_type, analysis_results in results.items():
            lines.append(f"## {analysis_type.replace('_', ' ').title()}")
            lines.append("")

            for result in analysis_results:
                status = "PASS" if result.passed else ("WARN" if result.severity == "warning" else "FAIL")
                lines.append(f"- **[{status}]** {result.name}: {result.message}")

            lines.append("")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write("\n".join(lines))


def run_analysis(
    module_identifier: str,
    input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    device: str = "cuda",
    dtype: str = "float32",
    module_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[AnalysisResult]]:
    """Convenience function to run analysis.

    Parameters
    ----------
    module_identifier : str
        Module identifier.
    input_shapes : Optional[Dict[str, Tuple[int, ...]]]
        Mapping of input parameter names to shapes.
        Defaults to {"x": (2, 16, 64)} for single-input modules.
    device : str
        Device string.
    dtype : str
        Data type string.
    module_kwargs : Optional[Dict[str, Any]]
        Module instantiation kwargs.

    Returns
    -------
    Dict[str, List[AnalysisResult]]
        Analysis results.
    """
    if input_shapes is None:
        input_shapes = {"x": (2, 16, 64)}

    config = AnalysisConfig(
        input_shapes=input_shapes,
        device=device,
        dtype=dtype,
    )

    runner = AnalysisRunner(module_identifier, config, module_kwargs=module_kwargs)
    return runner.run_all()
