"""
Base Analyzer Interface.

This module defines the abstract base class for all analyzers and the
AnalysisResult dataclass for storing analysis results.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn


@dataclass
class AnalysisResult:
    """Container for a single analysis result.

    Parameters
    ----------
    name : str
        Name of the analysis (e.g., "frobenius_norm", "gradient_check").
    value : Any
        The computed value. Can be a scalar, tensor, dict, or list.
    passed : bool
        Whether the check passed (for validation analyses). True by default.
    message : str
        Human-readable description of the result.
    details : Dict[str, Any]
        Additional metadata (per-parameter breakdown, statistics, etc.).
    severity : str
        Severity level: "info", "warning", "error". Default is "info".

    Examples
    --------
    >>> result = AnalysisResult(
    ...     name="frobenius_norm",
    ...     value=1.234,
    ...     passed=True,
    ...     message="Parameter norm within expected range",
    ...     details={"param_name": "weight", "shape": (64, 64)},
    ... )
    """

    name: str
    value: Any
    passed: bool = True
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary.

        Returns
        -------
        Dict[str, Any]
            Result as dictionary with serializable values.
        """
        # Convert tensor values to Python types
        value = self.value
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.tolist()

        # Process details
        details = {}
        for k, v in self.details.items():
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    details[k] = v.item()
                elif v.numel() <= 100:
                    details[k] = v.tolist()
                else:
                    details[k] = f"Tensor({tuple(v.shape)})"
            else:
                details[k] = v

        return {
            "name": self.name,
            "value": value,
            "passed": self.passed,
            "message": self.message,
            "details": details,
            "severity": self.severity,
        }

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"AnalysisResult({self.name}: {status}, value={self.value})"


class BaseAnalyzer(ABC):
    """Abstract base class for module analyzers.

    All analyzer implementations should inherit from this class and
    implement the `analyze` method.

    Parameters
    ----------
    module : nn.Module
        The PyTorch module to analyze.
    device : Optional[torch.device]
        Device for computations. If None, inferred from module.
    dtype : torch.dtype
        Data type for tensors.

    Attributes
    ----------
    module : nn.Module
        The module being analyzed.
    device : torch.device
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.
    """

    # Class-level name for the analyzer type
    name: str = "base"

    def __init__(
        self,
        module: nn.Module,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.module = module
        self.device = device or self._infer_device(module)
        self.dtype = dtype

    @abstractmethod
    def analyze(self, **kwargs) -> List[AnalysisResult]:
        """Run the analysis and return results.

        This method should be implemented by subclasses to perform the
        specific analysis.

        Parameters
        ----------
        **kwargs
            Analysis-specific parameters.

        Returns
        -------
        List[AnalysisResult]
            List of analysis results.
        """
        pass

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Return notebook cells for this analysis.

        Override in subclasses to provide notebook cell templates.

        Returns
        -------
        List[Dict[str, Any]]
            List of notebook cells (markdown and code).
        """
        return []

    def get_summary(self, results: List[AnalysisResult]) -> str:
        """Generate a text summary of the analysis results.

        Parameters
        ----------
        results : List[AnalysisResult]
            Results from the analyze() method.

        Returns
        -------
        str
            Human-readable summary.
        """
        lines = [f"=== {self.name.upper()} ANALYSIS ==="]

        for result in results:
            status = "PASS" if result.passed else "FAIL"
            lines.append(f"[{status}] {result.name}: {result.message}")

        return "\n".join(lines)

    @staticmethod
    def _infer_device(module: nn.Module) -> torch.device:
        """Infer device from module parameters or buffers.

        Parameters
        ----------
        module : nn.Module
            Module to inspect.

        Returns
        -------
        torch.device
            Device of the first parameter/buffer, or CPU if none found.
        """
        try:
            return next(module.parameters()).device
        except StopIteration:
            pass

        try:
            return next(module.buffers()).device
        except StopIteration:
            pass

        return torch.device("cpu")

    @staticmethod
    def _get_2d_weights(
        module: nn.Module,
    ) -> List[tuple[str, torch.Tensor]]:
        """Get all 2D weight parameters from the module.

        Parameters
        ----------
        module : nn.Module
            Module to inspect.

        Returns
        -------
        List[tuple[str, torch.Tensor]]
            List of (name, weight) pairs for 2D parameters.
        """
        weights = []
        for name, param in module.named_parameters():
            if param.dim() >= 2:
                weights.append((name, param.data))
        return weights

    @staticmethod
    def _format_shape(shape: tuple) -> str:
        """Format tensor shape for display.

        Parameters
        ----------
        shape : tuple
            Tensor shape.

        Returns
        -------
        str
            Formatted shape string.
        """
        return f"({', '.join(str(d) for d in shape)})"

    def _create_markdown_cell(self, content: str) -> Dict[str, Any]:
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

    def _create_code_cell(self, source: str) -> Dict[str, Any]:
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
        import textwrap

        source = textwrap.dedent(source).strip()
        return {
            "cell_type": "code",
            "metadata": {},
            "source": [line + "\n" for line in source.split("\n")],
            "outputs": [],
            "execution_count": None,
        }


class CompositeAnalyzer(BaseAnalyzer):
    """Analyzer that combines multiple analyzers.

    Useful for running a suite of analyses on a module.

    Parameters
    ----------
    module : nn.Module
        The module to analyze.
    analyzers : List[BaseAnalyzer]
        List of analyzers to run.
    device : Optional[torch.device]
        Device for computations.
    dtype : torch.dtype
        Data type for tensors.
    """

    name = "composite"

    def __init__(
        self,
        module: nn.Module,
        analyzers: List[BaseAnalyzer],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(module, device, dtype)
        self.analyzers = analyzers

    def analyze(self, **kwargs) -> List[AnalysisResult]:
        """Run all analyzers and aggregate results.

        Parameters
        ----------
        **kwargs
            Arguments passed to each analyzer.

        Returns
        -------
        List[AnalysisResult]
            Combined results from all analyzers.
        """
        all_results = []
        for analyzer in self.analyzers:
            try:
                results = analyzer.analyze(**kwargs)
                all_results.extend(results)
            except Exception as e:
                all_results.append(
                    AnalysisResult(
                        name=f"{analyzer.name}_error",
                        value=str(e),
                        passed=False,
                        message=f"Analyzer {analyzer.name} failed: {e}",
                        severity="error",
                    )
                )
        return all_results

    def get_notebook_cells(self) -> List[Dict[str, Any]]:
        """Get notebook cells from all analyzers.

        Returns
        -------
        List[Dict[str, Any]]
            Combined notebook cells.
        """
        cells = []
        for analyzer in self.analyzers:
            cells.extend(analyzer.get_notebook_cells())
        return cells
