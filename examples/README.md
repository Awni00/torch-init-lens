# Examples

This directory contains runnable examples and notebook demos for `layer_analysis`.

## Scripts

- `basic_usage.py` — programmatic analysis of a standard `torch.nn.Linear` module.
- `custom_module.py` — a tiny custom module that you can analyze via CLI.

## Running a custom module

From the repository root:

```bash
python -m layer_analysis analyze examples.custom_module@TinyMLP \
  --input-shape 2,16 --module-kwargs in_features=16 hidden_features=32 out_features=8
```

## Notebooks

The `notebooks/` folder includes a `quickstart.ipynb` demo that mirrors the basic usage script. Install the optional notebook dependencies before running:

```bash
pip install "layer-analysis[notebook]"
```
