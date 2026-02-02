"""
Notebook Generator.

This module provides the NotebookGenerator class for generating Jupyter
notebooks for module initialization analysis.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from ..config import AnalysisConfig
from ..introspection import load_module_class, merge_kwargs, validate_kwargs
from .templates import (
    get_title_cells,
    get_setup_cells,
    get_module_instantiation_cells,
    get_summary_cells,
    get_all_analysis_cells,
)


class NotebookGenerator:
    """Generate Jupyter notebooks for module initialization analysis.

    Parameters
    ----------
    module_identifier : str
        Module identifier (e.g., "torch.nn@Linear").
    config : AnalysisConfig
        Analysis configuration.
    module_kwargs : Optional[Dict[str, Any]]
        Additional keyword arguments for module instantiation.

    Examples
    --------
    >>> from layer_analysis import AnalysisConfig
    >>> config = AnalysisConfig(input_shape=(2, 16))
    >>> generator = NotebookGenerator(
    ...     "torch.nn@Linear",
    ...     config,
    ...     module_kwargs={"in_features": 16, "out_features": 32},
    ... )
    >>> generator.generate(Path("analysis.ipynb"))
    """

    def __init__(
        self,
        module_identifier: str,
        config: AnalysisConfig,
        module_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if "@" not in module_identifier:
            raise ValueError(
                f"Invalid module identifier: {module_identifier}. "
                "Expected format: 'module.path@ClassName'"
            )

        self.module_identifier = module_identifier
        self.config = config
        self.module_kwargs = module_kwargs or {}
        self.module_path, self.class_name = module_identifier.split("@", 1)

        self._validate_module_kwargs()

    def _validate_module_kwargs(self) -> None:
        """Validate that all required module kwargs are provided.

        Raises
        ------
        MissingModuleKwargsError
            If required kwargs are missing.
        InvalidModuleKwargsError
            If invalid kwargs are provided.
        """
        cls = load_module_class(self.module_identifier)
        # Use primary input shape for default merging (e.g. d_model derivation)
        input_shape = self.config.input_shape
        kwargs = merge_kwargs(cls, self.module_kwargs, input_shape)
        validate_kwargs(cls, kwargs, user_kwargs=self.module_kwargs)

    def generate(
        self,
        output_path: Path,
        skip_analyses: Optional[List[str]] = None,
    ) -> None:
        """Generate and save the notebook.

        Parameters
        ----------
        output_path : Path
            Path to save the notebook.
        skip_analyses : Optional[List[str]]
            Analysis types to skip.
        """
        skip_analyses = skip_analyses or []

        notebook = self._create_notebook_structure()
        cells = []

        # Title and introduction
        cells.extend(
            get_title_cells(
                self.module_identifier,
                self.config.input_shapes,
                self.config.device,
                self.config.dtype,
            )
        )

        # Setup and imports
        cells.extend(get_setup_cells(self.module_path, self.class_name))

        # Module instantiation
        cells.extend(
            get_module_instantiation_cells(
                self.module_path,
                self.class_name,
                self.config.input_shapes,
                self.config.device,
                self.config.dtype,
                self.module_kwargs,
                module_class=load_module_class(self.module_identifier),
            )
        )

        # Analysis sections
        analysis_cells = get_all_analysis_cells(
            self.config.input_shapes,
            self.config.device,
            self.config.dtype,
            gradient_loss_fn=self.config.gradient_loss_fn,
        )

        analysis_map = {
            "parameter_norms": self.config.run_parameter_norms,
            "operator_norm": self.config.run_operator_norm,
            "gradient_analysis": self.config.run_gradient_analysis,
            "spectral_analysis": self.config.run_spectral_analysis,
            "rank_analysis": self.config.run_rank_analysis,
            "lipschitz": self.config.run_lipschitz,
            "activation_stats": self.config.run_activation_stats,
            "gradient_ratios": self.config.run_gradient_ratios,
            "precision_checks": self.config.run_precision_checks,
            "weight_distribution": self.config.run_weight_distribution,
        }

        for name, enabled in analysis_map.items():
            if enabled and name not in skip_analyses and name in analysis_cells:
                cells.extend(analysis_cells[name])

        # Summary
        cells.extend(get_summary_cells())

        notebook["cells"] = cells

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)

    def _create_notebook_structure(self) -> Dict[str, Any]:
        """Create the basic notebook structure.

        Returns
        -------
        Dict[str, Any]
            Empty notebook structure.
        """
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.10.0",
                    "mimetype": "text/x-python",
                    "file_extension": ".py",
                },
            },
            "cells": [],
        }

    def generate_custom(
        self,
        output_path: Path,
        analyses: List[str],
        include_setup: bool = True,
        include_summary: bool = True,
    ) -> None:
        """Generate a notebook with custom analysis selection.

        Parameters
        ----------
        output_path : Path
            Path to save the notebook.
        analyses : List[str]
            List of analysis types to include.
        include_setup : bool
            Whether to include setup cells.
        include_summary : bool
            Whether to include summary cells.
        """
        notebook = self._create_notebook_structure()
        cells = []

        # Title
        cells.extend(
            get_title_cells(
                self.module_identifier,
                self.config.input_shapes,
                self.config.device,
                self.config.dtype,
            )
        )

        if include_setup:
            cells.extend(get_setup_cells(self.module_path, self.class_name))
            cells.extend(
                get_module_instantiation_cells(
                    self.module_path,
                    self.class_name,
                    self.config.input_shapes,
                    self.config.device,
                    self.config.dtype,
                    self.module_kwargs,
                    module_class=load_module_class(self.module_identifier),
                )
            )

        # Selected analyses
        analysis_cells = get_all_analysis_cells(
            self.config.input_shapes,
            self.config.device,
            self.config.dtype,
            gradient_loss_fn=self.config.gradient_loss_fn,
        )

        for analysis in analyses:
            if analysis in analysis_cells:
                cells.extend(analysis_cells[analysis])

        if include_summary:
            cells.extend(get_summary_cells())

        notebook["cells"] = cells

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(notebook, f, indent=2)


def create_analysis_notebook(
    module_identifier: str,
    output_path: Path,
    input_shape: tuple = (2, 16, 64),
    device: str = "cuda",
    dtype: str = "float32",
    skip_analyses: Optional[List[str]] = None,
) -> None:
    """Convenience function to create an analysis notebook.

    Parameters
    ----------
    module_identifier : str
        Module identifier.
    output_path : Path
        Output path for notebook.
    input_shape : tuple
        Input tensor shape.
    device : str
        Device string.
    dtype : str
        Data type string.
    skip_analyses : Optional[List[str]]
        Analyses to skip.
    """
    config = AnalysisConfig(
        input_shape=input_shape,
        device=device,
        dtype=dtype,
    )

    generator = NotebookGenerator(module_identifier, config)
    generator.generate(output_path, skip_analyses=skip_analyses)
