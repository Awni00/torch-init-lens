"""
Main entry point for running layer_analysis as a package.

Usage:
    python -m layer_analysis analyze torch.nn@Linear --input-shape 2,16 --module-kwargs in_features=16 out_features=32
    python -m layer_analysis generate-notebook torch.nn@LayerNorm --input-shape 2,16 -o analysis.ipynb
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
