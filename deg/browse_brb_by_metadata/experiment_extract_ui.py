from __future__ import annotations

# Canonical implementation stays in the top-level module for now.
# This file exists so browse-by-experiment code can import from a stable metadata package path.

from deg.experiment_extract_ui import render_experiment_extract_ui

__all__ = ["render_experiment_extract_ui"]

