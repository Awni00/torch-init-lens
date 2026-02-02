"""
Main entry point for running layer_analysis as a package.

Usage:
    python -m utilities.layer_analysis analyze models.abstractor@RelationalAttention
    python -m utilities.layer_analysis generate-notebook models.abstractor@DualAttention -o analysis.ipynb
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
