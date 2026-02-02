# Module Initialization Analysis

A toolkit for analyzing PyTorch module initialization properties, including parameter norms, gradients, spectral properties, and numerical stability.

## Purpose

This package helps verify that newly implemented modules are well-behaved at initialization by checking:

- **Parameter norms**: Frobenius, operator, sup-norm, standard deviation
- **Operator norm**: Effective input-output ratio across various input distributions
- **Gradient analysis**: Gradient existence, norms, and completeness
- **Spectral analysis**: Singular values, spectral radius, condition number
- **Rank analysis**: Effective rank of weight matrices
- **Lipschitz estimation**: Empirical and weight-based bounds
- **Activation statistics**: Mean, std, max of intermediate activations
- **Gradient ratios**: Layer-wise gradient flow analysis
- **Numerical precision**: NaN/Inf checks, extreme input handling

## CLI Usage

```bash
# Single-input module
python -m layer_analysis analyze torch.nn@Linear \
    --input-shape 2,16 --module-kwargs in_features=16 out_features=32

# Multi-input module (e.g., MultiheadAttention expects query/key/value)
python -m layer_analysis analyze torch.nn@MultiheadAttention \
    --input-shapes query:2,8,32 key:2,8,32 value:2,8,32 \
    --module-kwargs embed_dim=32 num_heads=4

# Module with custom kwargs
python -m layer_analysis analyze torch.nn@LayerNorm \
    --input-shape 2,16 --module-kwargs normalized_shape=16

# Generate Jupyter notebook for interactive analysis
python -m layer_analysis generate-notebook torch.nn@LayerNorm \
    --input-shape 2,16 --module-kwargs normalized_shape=16 \
    -o analysis.ipynb

# Generate and run Jupyter notebook
python -m layer_analysis generate-notebook torch.nn@Linear \
    --input-shape 2,16 --module-kwargs in_features=16 out_features=32 --run

# Generate Python script
python -m layer_analysis generate-notebook torch.nn@Linear \
    --input-shape 2,16 --module-kwargs in_features=16 out_features=32 \
    --output-format py -o analysis.py

# List available analyses
python -m layer_analysis list-analyses
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--input-shape` | Single input shape as `batch,seq,dim` (e.g., `2,16,64`) |
| `--input-shapes` | Multiple inputs as `name:shape` pairs (e.g., `x:2,16,64 symbols:2,16,64`) |
| `--module-kwargs` | Additional module init kwargs as `key=value` pairs (e.g. `n_symbols=100`) |
| `--loss-fn` | Loss function(s) for gradient analysis (default: `reconstruction`). Options: `sum`, `mse_random`, `reconstruction`, `variance` |
| `--device` | Device for analysis: `cuda` or `cpu` (default: `cuda`) |
| `--dtype` | Data type: `float32`, `float16`, `bfloat16` (default: `float32`) |
| `--n-samples` | Number of samples for statistical estimation (default: `50`) |
| `-o, --output` | Output file (`.json` or `.md` for analyze, required for generate-notebook) |
| `--run` | Execute the notebook after generation (generate-notebook only) |

## Programmatic Usage

```python
from layer_analysis.runner import run_analysis, AnalysisRunner
from layer_analysis.config import AnalysisConfig

# Quick analysis with convenience function
results = run_analysis(
    "torch.nn@Linear",
    input_shapes={"x": (2, 16)},
    device="cpu",
    module_kwargs={"in_features": 16, "out_features": 32},
)

# Full control with AnalysisRunner
config = AnalysisConfig(
    input_shapes={"x": (2, 16)},
    device="cpu",
    dtype="float32",
    n_samples=100,
    run_spectral_analysis=True,
    run_lipschitz=False,  # Disable specific analyses
)

runner = AnalysisRunner(
    "torch.nn@Linear",
    config,
    module_kwargs={"in_features": 16, "out_features": 32},
)
results = runner.run_all()
runner.print_summary(results)

# Save results
runner.save_json(results, "results.json")
runner.save_markdown(results, "results.md")
```

## Package Structure

```
src/layer_analysis/
├── __init__.py
├── __main__.py          # Entry point for python -m
├── cli.py               # Command-line interface
├── config.py            # AnalysisConfig dataclass
├── runner.py            # AnalysisRunner orchestrator
├── introspection.py     # Module loading and inspection utilities
├── errors.py            # Custom exceptions and safe utilities
├── analyzers/           # Individual analysis implementations
│   ├── base.py              # BaseAnalyzer abstract class
│   ├── parameter_norms.py   # Parameter norm analysis
│   ├── operator_norm.py     # Operator norm estimation
│   ├── gradient_analysis.py # Gradient checks
│   ├── spectral_analysis.py # SVD and spectral properties
│   ├── rank_analysis.py     # Effective rank
│   ├── lipschitz.py         # Lipschitz estimation
│   ├── activation_stats.py  # Activation statistics
│   ├── gradient_ratios.py   # Layer-wise gradient ratios
│   ├── numerical_precision.py # NaN/Inf checks
│   └── weight_distribution.py # Weight distribution analysis
├── inputs/              # Input tensor generation
│   └── generators.py        # InputGenerator class
└── notebook/            # Notebook/script generation
    ├── generator.py         # Jupyter notebook generator
    └── python_generator.py  # Python script generator
```

## Interpreting Results

Each analysis returns `AnalysisResult` objects with:
- `name`: Analysis identifier
- `value`: Computed metric(s)
- `passed`: Whether the check passed
- `message`: Human-readable description
- `severity`: `"info"`, `"warning"`, or `"error"`

### Key Metrics to Watch

| Metric | Healthy Range | Concern |
|--------|---------------|---------|
| Operator norm | 0.5 - 2.0 | >> 1 may cause exploding activations |
| Condition number | < 1000 | High values indicate ill-conditioning |
| Gradient ratios | 0.1 - 10 | Outside range suggests vanishing/exploding gradients |
| Effective rank | > 50% | Low rank may indicate redundant parameters |
