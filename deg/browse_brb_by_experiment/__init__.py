"""
Browse BRB-seq by experiment (precomputed results + experiment-scoped navigation).

This package holds Streamlit UI code for:
- Experiment selection and navigation
- Browsing precomputed DEG results stored in MongoDB/S3
"""

from deg.browse_brb_by_experiment.experiment_browser import run_experiment_browser
from deg.browse_brb_by_experiment.precomputed_results_browser import run_precomputed_results_browser

__all__ = ["run_experiment_browser", "run_precomputed_results_browser"]

