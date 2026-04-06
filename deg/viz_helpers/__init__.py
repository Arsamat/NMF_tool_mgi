"""
DEG result visualizations (volcano, MA, heatmap, GSEA, research pipeline).

Public entrypoint: ``render_deg_results_and_visualizations``.
"""

from deg.viz_helpers.common import (
    DEG_API_URL,
    MAX_FDR,
    MIN_LFC,
    ensure_research_session,
    fig_to_html,
    valid_display_label,
)
from deg.viz_helpers.deg_viz import render_deg_results_and_visualizations
from deg.viz_helpers.heatmap import METADATA_SORT_KEYS

# Backward-compatible names used by ``precomputed_results_browser_helpers`` etc.
_fig_to_html = fig_to_html
_valid_display_label = valid_display_label

__all__ = [
    "DEG_API_URL",
    "MAX_FDR",
    "METADATA_SORT_KEYS",
    "MIN_LFC",
    "_fig_to_html",
    "_valid_display_label",
    "ensure_research_session",
    "fig_to_html",
    "render_deg_results_and_visualizations",
    "valid_display_label",
]
