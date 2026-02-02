# Layer Analysis (torch-init-lens)

A self-contained toolkit for analyzing PyTorch module initialization properties, including parameter norms, gradients, spectral properties, and numerical stability. The package is architecture-agnostic: you can point it at any `torch.nn.Module` (or custom module on your PYTHONPATH) using a simple `module.path@ClassName` identifier.

## Installation

```bash
pip install layer-analysis
```

Optional extras:

```bash
# For notebook generation/execution
pip install "layer-analysis[notebook]"

# For plotting helpers in analyzers
pip install "layer-analysis[plots]"

# For development
pip install "layer-analysis[dev]"
```

## Quickstart

### CLI

```bash
# Single-input module
python -m layer_analysis analyze torch.nn@Linear \
  --input-shape 2,16 --module-kwargs in_features=16 out_features=32

# Multi-input module (e.g., MultiheadAttention expects query/key/value)
python -m layer_analysis analyze torch.nn@MultiheadAttention \
  --input-shapes query:2,8,32 key:2,8,32 value:2,8,32 \
  --module-kwargs embed_dim=32 num_heads=4

# Generate a notebook
python -m layer_analysis generate-notebook torch.nn@LayerNorm \
  --input-shape 2,16 --module-kwargs normalized_shape=16 -o analysis.ipynb

# List available analyses
python -m layer_analysis list-analyses
```

### Programmatic

```python
from layer_analysis import AnalysisConfig, AnalysisRunner

config = AnalysisConfig(
    input_shapes={"x": (2, 16)},
    device="cpu",
)

runner = AnalysisRunner(
    "torch.nn@Linear",
    config,
    module_kwargs={"in_features": 16, "out_features": 32},
)
results = runner.run_all()
runner.print_summary(results)
```

## Design choices

- **Architecture-agnostic module loading:** uses `module.path@ClassName` to load any `torch.nn.Module`, including your own modules if they are importable.
- **Self-contained analysis pipeline:** all analyzers (norms, gradients, spectral stats, Lipschitz bounds, etc.) live in a single package with minimal external assumptions.
- **Safe defaults with configurable toggles:** you can disable heavy analyses or adjust sampling parameters via `AnalysisConfig`.
- **Notebook and script generation:** the CLI can produce Jupyter notebooks or standalone scripts for reproducible analysis.

## Project layout

```
.
├── src/layer_analysis/        # Package source
├── docs/                      # Extended docs
├── examples/                  # Scripts and notebooks
├── tests/                     # Unit tests
└── .github/workflows/         # CI/CD automation
```

## Examples

See the `examples/` directory for runnable scripts and notebook demos. The notebooks are designed to be executed after installing the package and optional notebook dependencies.

## Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
