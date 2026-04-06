"""
Browse BRB-seq by metadata (interactive filtering, group assignment, de novo DEG).

This package holds Streamlit UI code for:
- Metadata filtering / retrieval
- Group assignment (Group A / Group B)
- Submitting to backend for de novo DEG
"""

from deg.browse_brb_by_metadata.group_selection import run_group_selection

__all__ = ["run_group_selection"]

