"""
Module Initialization Analysis Package

A self-contained sub-package for analyzing PyTorch module initialization properties.
Provides reusable analysis utilities and CLI tools for generating analysis notebooks.

Usage:
    # CLI usage
    python -m utilities.layer_analysis analyze models.abstractor@RelationalAttention
    python -m utilities.layer_analysis generate-notebook models.abstractor@DualAttention -o analysis.ipynb

    # Programmatic usage
    from utilities.layer_analysis import AnalysisRunner, AnalysisConfig
    from utilities.layer_analysis.analyzers import ParameterNormAnalyzer

    config = AnalysisConfig(input_shape=(2, 16, 64), device="cuda")
    runner = AnalysisRunner("models.abstractor@DualAttention", config)
    results = runner.run_all()
"""

from .config import AnalysisConfig
from .introspection import load_module_class, get_module_signature, detect_module_type
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
