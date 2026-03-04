"""
Helper functions for DEG group selection: add to group, remove from group, clear group,
authentication, and EC2 wake-up (Lambda) logic.
"""
import streamlit as st
import pandas as pd
import time
import requests

SAMPLE_COL = "SampleName"

# --- Authentication (same pattern as brb_data_pages/extract_counts_frontend) ---


def ensure_auth_session():
    """Initialize session state for authentication."""
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False


def authenticate():
    """Show password UI and set st.session_state['authenticated'] on success. Same as extract_counts_frontend."""
    title_placeholder = st.empty()
    help_placeholder = st.empty()
    password_input_placeholder = st.empty()
    button_placeholder = st.empty()
    success_placeholder = st.empty()

    if not st.session_state["authenticated"]:
        with title_placeholder:
            st.title("If you are part of MGI you can access DEG analysis")
        with help_placeholder:
            with st.expander("**⚠️ Read if You Need Help With Password**"):
                st.write("To request or get an updated password contact developers.")
                st.write("**Azamat Khanbabaev** azamat@wustl.edu")
                st.write("**Aura Ferreiro** alferreiro@wustl.edu")
        with password_input_placeholder:
            user_password = st.text_input(
                "Enter the application password:", type="password", key="deg_pwd_input"
            )
        check_password = (
            True
            if user_password == st.secrets.get("PASSWORD", "")
            else False
        )
        with button_placeholder:
            if st.button("Authenticate", key="deg_auth_btn") or user_password:
                if check_password:
                    st.session_state["authenticated"] = True
                    password_input_placeholder.empty()
                    button_placeholder.empty()
                    success_placeholder.success("Authentication Successful!")
                    st.balloons()
                    time.sleep(1)
                    success_placeholder.empty()
                    title_placeholder.empty()
                    help_placeholder.empty()
                    st.rerun()
                else:
                    st.error("❌ Incorrect Password. Please Try Again.")


# --- EC2 wake-up via Lambda (same pattern as extract_counts_frontend) ---


def ensure_ec2_wake_session(api_url: str, lambda_url: str):
    """Set session state for EC2 wake-up: API URL, Lambda URL, and readiness flags."""
    st.session_state.setdefault("deg_api_url", api_url)
    st.session_state.setdefault("deg_lambda_url", lambda_url)
    st.session_state.setdefault("deg_ec2_start_triggered", False)
    st.session_state.setdefault("deg_fastapi_ready", False)


def start_ec2_once():
    """Fire Lambda once to wake EC2. Idempotent: only runs if deg_ec2_start_triggered is False."""
    if st.session_state.get("deg_ec2_start_triggered"):
        return
    try:
        requests.post(
            st.session_state["deg_lambda_url"],
            json={},
            timeout=10,
        )
        st.session_state["deg_ec2_start_triggered"] = True
    except Exception as e:
        st.warning(f"Could not wake compute node: {e}")


def check_health(health_url: str):
    """GET health_url and set deg_fastapi_ready in session state."""
    try:
        r = requests.get(health_url, timeout=2)
        st.session_state["deg_fastapi_ready"] = r.ok
    except Exception:
        st.session_state["deg_fastapi_ready"] = False


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
