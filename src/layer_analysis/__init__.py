"""
Module Initialization Analysis Package

A self-contained sub-package for analyzing PyTorch module initialization properties.
Provides reusable analysis utilities and CLI tools for generating analysis notebooks.

Usage:
    # CLI usage
    python -m layer_analysis analyze torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32
    python -m layer_analysis generate-notebook torch.nn@LayerNorm --input-shape 2,16 -o analysis.ipynb

    # Programmatic usage
    from layer_analysis import AnalysisRunner, AnalysisConfig
    from layer_analysis.analyzers import ParameterNormAnalyzer

    config = AnalysisConfig(input_shape=(2, 16), device="cpu")
    runner = AnalysisRunner(
        "torch.nn@Linear",
        config,
        module_kwargs={"in_features": 16, "out_features": 32},
    )
    results = runner.run_all()
"""

from .config import AnalysisConfig
from .introspection import load_module_class, get_module_signature, detect_module_type
from .runner import AnalysisRunner, run_analysis
from .errors import (
    ModuleAnalysisError,
    ModuleLoadError,
    IncompatibleModuleError,
    NumericalInstabilityError,
    GradientError,
)

__version__ = "0.1.0"

__all__ = [
    # Config
    "AnalysisConfig",
    "AnalysisRunner",
    "run_analysis",
    # Introspection
    "load_module_class",
    "get_module_signature",
    "detect_module_type",
    # Errors
    "ModuleAnalysisError",
    "ModuleLoadError",
    "IncompatibleModuleError",
    "NumericalInstabilityError",
    "GradientError",
]
