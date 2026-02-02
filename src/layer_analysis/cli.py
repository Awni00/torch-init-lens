"""
Command-Line Interface for Module Initialization Analysis.

Usage:
    # Single-input module
    python -m layer_analysis analyze torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32

    # Multi-input module (e.g., MultiheadAttention expects query/key/value)
    python -m layer_analysis analyze torch.nn@MultiheadAttention \
        --input-shapes query:2,8,32 key:2,8,32 value:2,8,32 \
        --module-kwargs embed_dim=32 num_heads=4

    # Module with custom kwargs
    python -m layer_analysis analyze torch.nn@LayerNorm \
        --input-shape 2,16 --module-kwargs normalized_shape=16

    # Generate notebook (default output: tests/layer_analysis/{module_name}.ipynb)
    python -m layer_analysis generate-notebook torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32

    # Generate Python script
    python -m layer_analysis generate-notebook torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32 --output-format py -o analysis.py
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .config import AnalysisConfig
from .runner import AnalysisRunner
from .introspection import load_module_class, get_input_signature
from .notebook.generator import NotebookGenerator
from .notebook.python_generator import PythonScriptGenerator


def validate_input_shapes(module_class, input_shapes: Dict[str, Any]) -> None:
    """Validate that input shapes match the module's forward signature.

    Parameters
    ----------
    module_class : Type[nn.Module]
        The module class to check.
    input_shapes : Dict[str, Any]
        The provided input shapes.

    Raises
    ------
    ValueError
        If required inputs are missing.
    """
    sig = get_input_signature(module_class)
    if sig is None:
        return

    # Filter out parameters that have defaults or are variable args
    required_args = {
        name for name, info in sig.items()
        if info.get('required', False)
        and info['kind'] not in ('VAR_POSITIONAL', 'VAR_KEYWORD')
    }

    provided_args = set(input_shapes.keys())
    missing = required_args - provided_args

    if missing:
        raise ValueError(
            f"Input signature mismatch for {module_class.__name__}.\\n"
            f"Missing required input shapes for: {', '.join(sorted(missing))}.\\n"
            f"Required inputs: {sorted(required_args)}\\n"
            f"Provided inputs: {sorted(provided_args)}"
        )

def execute_notebook(notebook_path: Path) -> None:
    """Execute a Jupyter notebook in-place.

    Parameters
    ----------
    notebook_path : Path
        Path to the notebook file.
    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    print(f"Executing notebook: {notebook_path}...")

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
    except Exception as e:
        print(f"Error executing notebook: {e}", file=sys.stderr)
        raise

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("Notebook execution completed.")



def parse_input_shape(shape_str: str) -> tuple:
    """Parse input shape from comma-separated string.

    Parameters
    ----------
    shape_str : str
        Shape as comma-separated ints (e.g., "2,16,64").

    Returns
    -------
    tuple
        Shape tuple.
    """
    return tuple(int(x.strip()) for x in shape_str.split(','))


def parse_input_shapes(shapes_list: List[str]) -> dict:
    """Parse multiple input shapes from name:shape format.

    Parameters
    ----------
    shapes_list : List[str]
        List of strings in "name:shape" format (e.g., ["x:2,16,64", "symbols:2,16,64"]).

    Returns
    -------
    dict
        Dictionary mapping parameter names to shape tuples.

    Examples
    --------
    >>> parse_input_shapes(["x:2,16,64", "symbols:2,16,64"])
    {'x': (2, 16, 64), 'symbols': (2, 16, 64)}
    """
    result = {}
    for item in shapes_list:
        if ":" not in item:
            raise ValueError(
                f"Invalid input shape format: '{item}'. "
                "Expected 'name:shape' (e.g., 'x:2,16,64')."
            )
        name, shape_str = item.split(":", 1)
        result[name.strip()] = parse_input_shape(shape_str)
    return result


def parse_module_kwargs(kwargs_list: List[str]) -> Dict[str, Any]:
    """Parse module kwargs from key=value strings.

    Parameters
    ----------
    kwargs_list : List[str]
        List of strings in "key=value" format.

    Returns
    -------
    Dict[str, Any]
        Dictionary of parsed kwargs.
    """
    kwargs = {}
    for item in kwargs_list:
        if "=" not in item:
            raise ValueError(f"Invalid kwarg format: {item}. Expected key=value.")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to parse as int or float, else string
        try:
            if "." in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            # Handle booleans
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            # Otherwise keep as string

        kwargs[key] = value
    return kwargs


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="layer_analysis",
        description="Module Initialization Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single-input module with custom shape
    python -m layer_analysis analyze torch.nn@Linear --input-shape 4,32 --module-kwargs in_features=32 out_features=64

    # Multi-input module (e.g., MultiheadAttention expects query/key/value)
    python -m layer_analysis analyze torch.nn@MultiheadAttention \\
        --input-shapes query:2,8,32 key:2,8,32 value:2,8,32 \\
        --module-kwargs embed_dim=32 num_heads=4

    # Module with custom kwargs
    python -m layer_analysis analyze torch.nn@LayerNorm \\
        --input-shape 2,16 --module-kwargs normalized_shape=16

    # Generate Jupyter notebook (default output: tests/layer_analysis/{module_name}.ipynb)
    python -m layer_analysis generate-notebook torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32

    # Generate Python script
    python -m layer_analysis generate-notebook torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32 --output-format py -o analysis.py
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- analyze command ---
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run analysis on a module and print results",
    )
    _add_common_args(analyze_parser)
    analyze_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file for results (JSON or Markdown based on extension)",
    )

    # --- generate-notebook command ---
    notebook_parser = subparsers.add_parser(
        "generate-notebook",
        help="Generate analysis notebook or script",
    )
    _add_common_args(notebook_parser)
    notebook_parser.add_argument(
        "--output-format",
        type=str,
        default="ipynb",
        choices=["ipynb", "py"],
        help="Output format: Jupyter notebook (ipynb) or Python script (py)",
    )
    notebook_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: tests/layer_analysis/{module_name}.{ext})",
    )
    notebook_parser.add_argument(
        "--skip",
        type=str,
        nargs="*",
        default=[],
        help="Analysis types to skip",
    )
    notebook_parser.add_argument(
        "--run",
        action="store_true",
        help="Run the notebook after generation",
    )

    # --- list-analyses command ---
    subparsers.add_parser(
        "list-analyses",
        help="List available analysis types",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to add arguments to.
    """
    parser.add_argument(
        "module",
        type=str,
        help="Module identifier (e.g., 'torch.nn@Linear')",
    )
    # Support both single shape (backward compat) and multiple shapes
    shape_group = parser.add_mutually_exclusive_group()
    shape_group.add_argument(
        "--input-shape",
        type=str,
        default=None,
        help="Single input shape as comma-separated ints (batch,seq,dim). "
        "For single-input modules only.",
    )
    shape_group.add_argument(
        "--input-shapes",
        type=str,
        nargs="+",
        default=None,
        help="Multiple input shapes as 'name:shape' pairs. "
        "Example: --input-shapes x:2,16,64 symbols:2,16,64",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for analysis",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for tensors",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples for statistical estimation",
    )
    parser.add_argument(
        "--module-kwargs",
        type=str,
        nargs="*",
        default=[],
        help="Additional keyword arguments for module initialization (key=value). E.g. n_symbols=100",
    )
    parser.add_argument(
        "--loss-fn",
        type=str,
        nargs="*",
        default=["reconstruction"],
        help="Loss function(s) for gradient analysis. "
        "Options: sum, mse_random, reconstruction, variance. Default: reconstruction",
    )


def _get_loss_fn(args: argparse.Namespace):
    """Extract loss function(s) from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    str or List[str]
        Loss function name(s).
    """
    loss_fn = getattr(args, 'loss_fn', ['reconstruction'])
    if len(loss_fn) == 1:
        return loss_fn[0]
    return loss_fn


def _get_input_shapes(args: argparse.Namespace) -> dict:
    """Extract input shapes from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    dict
        Dictionary mapping parameter names to shape tuples.
    """
    if args.input_shapes is not None:
        return parse_input_shapes(args.input_shapes)
    elif args.input_shape is not None:
        return {"x": parse_input_shape(args.input_shape)}
    else:
        # Default
        return {"x": (2, 16, 64)}


def run_analyze(args: argparse.Namespace) -> int:
    """Execute the analyze command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    input_shapes = _get_input_shapes(args)

    # Validate input shapes
    try:
        module_class = load_module_class(args.module)
        validate_input_shapes(module_class, input_shapes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    config = AnalysisConfig(
        input_shapes=input_shapes,
        device=args.device,
        dtype=args.dtype,
        n_samples=args.n_samples,
        gradient_loss_fn=_get_loss_fn(args),
    )

    module_kwargs = parse_module_kwargs(args.module_kwargs)

    try:
        runner = AnalysisRunner(args.module, config, module_kwargs=module_kwargs)
        results = runner.run_all()

        if args.output:
            if args.output.suffix == ".json":
                runner.save_json(results, args.output)
                print(f"Results saved to {args.output}")
            else:
                runner.save_markdown(results, args.output)
                print(f"Results saved to {args.output}")
        else:
            runner.print_summary(results)

        # Return non-zero if any failures
        failed = sum(
            1
            for analysis_results in results.values()
            for r in analysis_results
            if not r.passed and r.severity == "error"
        )
        return 1 if failed > 0 else 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run_generate_notebook(args: argparse.Namespace) -> int:
    """Execute the generate-notebook command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    input_shapes = _get_input_shapes(args)

    # Validate input shapes
    try:
        module_class = load_module_class(args.module)
        validate_input_shapes(module_class, input_shapes)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    config = AnalysisConfig(
        input_shapes=input_shapes,
        device=args.device,
        dtype=args.dtype,
        n_samples=args.n_samples,
        gradient_loss_fn=_get_loss_fn(args),
    )

    module_kwargs = parse_module_kwargs(args.module_kwargs)

    if args.output is None:
        module_name = args.module.split("@", 1)[-1]
        extension = "ipynb" if args.output_format == "ipynb" else "py"
        args.output = Path("tests") / "layer_analysis" / f"{module_name}.{extension}"

    if args.output.exists():
        response = input(f"Warning: {args.output} already exists. Overwrite? [y/N] ").strip().lower()
        if response not in ("y", "yes"):
            print("Aborted.")
            return 1

    try:
        if args.output_format == "ipynb":
            generator = NotebookGenerator(args.module, config, module_kwargs=module_kwargs)
            generator.generate(args.output, skip_analyses=args.skip)
        else:
            generator = PythonScriptGenerator(args.module, config, module_kwargs=module_kwargs)
            generator.generate(args.output, skip_analyses=args.skip)

        print(f"Generated {args.output_format} file: {args.output}")

        if args.run and args.output_format == "ipynb":
            execute_notebook(args.output)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def run_list_analyses(args: argparse.Namespace) -> int:
    """Execute the list-analyses command.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.

    Returns
    -------
    int
        Exit code.
    """
    analyses = [
        ("parameter_norms", "Compute Frobenius, operator, sup-norm, std for all parameters"),
        ("operator_norm", "Estimate effective operator norm via output/input ratio"),
        ("gradient_analysis", "Check gradient existence and compute gradient norms"),
        ("spectral_analysis", "Compute singular values, spectral radius, condition number"),
        ("rank_analysis", "Compute effective rank of weight matrices"),
        ("lipschitz", "Estimate Lipschitz constant"),
        ("activation_stats", "Analyze activation statistics (mean, std, max)"),
        ("gradient_ratios", "Analyze layer-wise gradient ratios"),
        ("precision_checks", "Check for NaN, Inf, numerical issues"),
        ("weight_distribution", "Analyze and visualize weight distributions"),
    ]

    print("Available Analysis Types:")
    print("-" * 70)
    for name, description in analyses:
        print(f"  {name:<25} {description}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point.

    Parameters
    ----------
    argv : Optional[List[str]]
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        return run_analyze(args)
    elif args.command == "generate-notebook":
        return run_generate_notebook(args)
    elif args.command == "list-analyses":
        return run_list_analyses(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
