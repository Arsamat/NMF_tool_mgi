"""
Helper functions for DEG group selection: add to group, remove from group, clear group.
"""
import streamlit as st
import pandas as pd

SAMPLE_COL = "SampleName"


def add_to_group(edited, group_key, other_group_key, sample_col=SAMPLE_COL):
    """Add selected samples from the data_editor to group_key; clear selections and rerun."""
    selected = edited[edited["Select"]].get(sample_col, pd.Series(dtype=object))
    to_add = selected.dropna().astype(str).tolist()
    if not to_add:
        return
    current = set(st.session_state[group_key])
    other_current = set(st.session_state[other_group_key])
    dup = [s for s in to_add if s in current]
    if dup:
        st.session_state["deg_dup_warning"] = (
            f"Duplicate samples skipped (already in {group_key}): {', '.join(dup)}. "
            "Those additions were not made."
        )
    dup = [s for s in to_add if s in other_current]
    if dup:
        st.session_state["deg_dup_warning"] = (
            f"Duplicate samples skipped (already in {other_group_key}): {', '.join(dup)}. "
            "Those additions were not made."
        )
    for s in to_add:
        if s not in current and s not in other_current:
            st.session_state[group_key].append(s)
    # Clear all selections after adding
    df = st.session_state["deg_metadata_df"]
    if "Select" in df.columns:
        df = df.copy()
        df["Select"] = False
        st.session_state["deg_metadata_df"] = df
    st.session_state["deg_editor_reset"] = st.session_state.get("deg_editor_reset", 0) + 1
    st.rerun()


def remove_from_group(group_key, sample):
    """Remove one sample from the group and rerun."""
    st.session_state[group_key] = [s for s in st.session_state[group_key] if s != sample]
    st.rerun()


def clear_group(group_key):
    """Clear the group and reset Select checkboxes; rerun."""
    st.session_state[group_key] = []
    df = st.session_state["deg_metadata_df"]
    if "Select" in df.columns:
        df = df.copy()
        df["Select"] = False
        st.session_state["deg_metadata_df"] = df
    st.session_state["deg_editor_reset"] = st.session_state.get("deg_editor_reset", 0) + 1
    st.rerun()
