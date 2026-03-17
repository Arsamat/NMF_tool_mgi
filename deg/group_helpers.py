"""
Helper functions for DEG group selection: add to group, remove from group, clear group,
authentication, and EC2 wake-up (Lambda) logic.
"""
import streamlit as st
import pandas as pd
import time
import requests
from dotenv import load_dotenv
from anthropic import Anthropic
import json
import os

SAMPLE_COL = "SampleName"
load_dotenv()

# --- Authentication (same pattern as brb_data_pages/extract_counts_frontend) ---


def natural_language_to_filters(user_description: str, metadata_schema: dict) -> dict:
    """
    Convert natural language description to structured filter dictionary.

    Args:
        user_description: User's description of desired samples
        metadata_schema: Schema with columns and unique_values

    Returns:
        Dictionary of filters to apply
    """
    import os

    #api_key = os.getenv("ANTHROPIC_API_KEY")
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    if not api_key:
        st.error("ANTHROPIC_API_KEY not set. Natural language filtering unavailable.")
        return {}

    # Format schema for Claude
    unique_values = metadata_schema.get("unique_values", {})
    schema_text = "Available columns and their values:\n"
    to_skip = ["OriginalName", "SampleName", "Replicate", "NumUnqDedupedReads", "Barcode", "Index1", "Index2", "CountsTable", "Library"]
    for col, values in unique_values.items():
        if col in to_skip:
            continue
        if col != SAMPLE_COL:
            # Limit display to first 20 values to keep prompt manageable
            vals_str = ", ".join([str(v) for v in sorted(values)[:30]])
            if len(values) > 20:
                vals_str += f", ... and {len(values) - 20} more"
            schema_text += f"- {col}: {vals_str}\n"

    prompt = f"""You are a metadata filtering assistant. Convert natural language descriptions to JSON filters.

{schema_text}

User request: "{user_description}"

RULES:
1. Handle OR logic: "WT or TDP43" → {{"Genotype": ["WT", "TDP43"]}}
2. Handle exclusions: "all timepoints excluding 96 hr" means ALL timepoint values EXCEPT 96
3. Match case-insensitively but return exact case from schema above
4. Look through the schema values to match the keywords in the user request.
5. For dose specifications (e.g., "Rotenone 1uM"), use Treatment + Dose as separate columns (e.g., {{"Treatment": ["Rotenone"], "Dose": ["1uM"]}})
6. Qualifier terms like "mutant", "WT-type", "control" modify but don't change the base name.
   "vehicle control" = Treatment: Vehicle; "rotenone treated" = Treatment: Rotenone (ignore "control" and "treated" as descriptors)
7. Cell type mapping: recognize biological cell type names:
   - "motor neurons" or "iMNs" → iMN
   - "cortical neurons" or "iCNs" → iCN
   - "sensory neurons" or "iSNs" → iSN
   Only add CellType if user explicitly mentions a cell type.
8. AND/OR logic in requests: "vehicle AND rotenone" or "vehicle or rotenone" both mean Treatment: ["Vehicle", "Rotenone"]
8. Return ONLY valid JSON: {{"ColumnName": ["value1", "value2"]}}
9. Return {{}} only if completely unrelated to schema

EXAMPLES:
- "WT or TDP43 treated with Rotenone at all timepoints excluding 96 hr"
  → {{"Genotype": ["WT", "TDP43"], "Treatment": ["Rotenone"], "Timepoint": [6, 12, 16, 24, 48, 72]}}
- "PINK1 or LATS2KO at 24 or 48 hours"
  → {{"Genotype": ["PINK1", "LATS2KO"], "Timepoint": [24, 48]}}
- "A549 cells with Rotenone but not TRULI"
  → {{"CellType": ["A549"], "Treatment": ["Rotenone"]}}
- "MFN2 mutant neurons treated with Rotenone 1uM"
  → {{"Genotype": ["MFN2"], "Treatment": ["Rotenone"], "Dose": ["1uM"]}}
- "Vehicle control and rotenone treated samples in WT motor neurons"
  → {{"Genotype": ["WT"], "CellType": ["iMN"], "Treatment": ["Vehicle", "Rotenone"]}}

CRITICAL: When user says "excluding" or "except", compute the complement: include ALL available values MINUS the excluded ones.
Return only JSON, no explanation."""

    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        response_text = message.content[0].text.strip()

        # Parse JSON response
        filters = json.loads(response_text)

        # Convert numeric values to integers where appropriate
        if isinstance(filters, dict):
            for col, values in filters.items():
                if isinstance(values, list):
                    filters[col] = [int(v) if isinstance(v, float) and v.is_integer() else v for v in values]

        return filters if isinstance(filters, dict) else {}
    except Exception as e:
        st.error(f"Error generating filters: {e}")
        return {}


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


def add_samples_to_group(samples: list, group_key: str, other_group_key: str):
    """Add a pre-computed list of sample names to group_key, skipping duplicates."""
    if not samples:
        return
    current = set(st.session_state[group_key])
    other_current = set(st.session_state[other_group_key])
    dup_self = [s for s in samples if s in current]
    dup_other = [s for s in samples if s in other_current]
    warnings = []
    if dup_self:
        warnings.append(f"Already in {group_key}: {', '.join(dup_self)}")
    if dup_other:
        warnings.append(f"Already in {other_group_key}: {', '.join(dup_other)}")
    if warnings:
        st.session_state["deg_dup_warning"] = " | ".join(warnings)
    added = []
    for s in samples:
        if s not in current and s not in other_current:
            st.session_state[group_key].append(s)
            added.append(s)
    return added


def _resolve_filters(filters: dict, metadata_df: pd.DataFrame) -> list:
    """Apply a filter dict to metadata_df and return matching SampleName values."""
    filtered_df = metadata_df.copy()
    for col, values in filters.items():
        if col in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col].isin(values)]
    return (
        filtered_df[SAMPLE_COL].dropna().astype(str).tolist()
        if SAMPLE_COL in filtered_df.columns
        else []
    )


def _coerce_filters(filters: dict) -> dict:
    """Coerce float-integers (e.g. 24.0 → 24) in filter value lists."""
    for col, values in filters.items():
        if isinstance(values, list):
            filters[col] = [
                int(v) if isinstance(v, float) and v.is_integer() else v
                for v in values
            ]
    return filters


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences that Claude sometimes wraps around JSON."""
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    return text


def natural_language_group_assignment(user_query: str, metadata_df: pd.DataFrame, schema: dict) -> dict:
    """
    Parse a natural language query and populate one or both comparison groups.

    Supports two modes determined by the query itself:
    - Single-group: "Add WT Rotenone samples to Comparison Group"
    - Dual-group comparison: "Compare C9orf72 iMNs against WT controls"

    Args:
        user_query: Natural language description of the desired group assignment.
        metadata_df: The currently loaded metadata DataFrame.
        schema: Schema dict with 'unique_values' mapping column → value list.

    Returns:
        {
            "group_a": {"filters": dict, "samples": list[str]},
            "group_b": {"filters": dict, "samples": list[str]},
            "error": str or None
        }
        Either group may have empty filters/samples if the query targets only one group.
    """
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    if not api_key:
        empty = {"filters": {}, "samples": []}
        return {"group_a": empty, "group_b": empty, "error": "ANTHROPIC_API_KEY not set."}

    unique_values = schema.get("unique_values", {})
    to_skip = ["OriginalName", "SampleName", "Replicate", "NumUnqDedupedReads",
               "Barcode", "Index1", "Index2", "CountsTable", "Library"]
    schema_text = "Available columns and their values:\n"
    for col, values in unique_values.items():
        if col in to_skip:
            continue
        vals_str = ", ".join([str(v) for v in sorted(values)[:30]])
        if len(values) > 30:
            vals_str += f", ... and {len(values) - 30} more"
        schema_text += f"- {col}: {vals_str}\n"

    prompt = f"""You are a sample group assignment assistant for a DEG (Differential Expression) analysis tool.
The tool has two groups:
- group_a = Comparison Group (the experimental / treatment / mutant samples)
- group_b = Reference Group (the control / wild-type / baseline samples)

{schema_text}

User request: "{user_query}"

TASK: Parse the request and return filters for group_a, group_b, or both.

RULES:
1. If the query mentions only one group (e.g. "Add X to Comparison Group"), populate only that group's filters and leave the other as {{}}.
2. If the query implies a comparison (keywords: "compare", "versus", "vs", "against", "compared to"),
   populate BOTH group_a and group_b from the two sides of the comparison.
   - The mutant / treated / experimental side → group_a
   - The wild-type / vehicle / control / reference side → group_b
3. For single-group queries, use "group_a" for: Comparison Group / Group A / treatment / experimental / mutant.
   Use "group_b" for: Reference Group / Group B / control / wild-type / WT / baseline / reference.
4. OR logic within a group: "WT or TDP43" → {{"Genotype": ["WT", "TDP43"]}}
5. Exclusions: "all timepoints except 96h" → include ALL available values minus excluded ones.
6. Match values case-insensitively but return exact case from the schema.
7. Cell type shorthands: "motor neurons"/"iMNs" → iMN, "cortical neurons"/"iCNs" → iCN, "sensory neurons" or "iSNs" → iSN
 - If the user doesn't mention a cell type, don't add it to the filters.
8. Dose specs: "Rotenone 1uM" → {{"Treatment": ["Rotenone"], "Dose": ["1uM"]}}

Return ONLY valid JSON with exactly this structure (both keys always present, either may have empty filters):
{{
  "group_a": {{"filters": {{"ColumnName": ["value1"]}}}},
  "group_b": {{"filters": {{"ColumnName": ["value1"]}}}}
}}

EXAMPLES:
- "Add WT Rotenone samples to Comparison Group"
  → {{"group_a": {{"filters": {{"Genotype": ["WT"], "Treatment": ["Rotenone"]}}}}, "group_b": {{"filters": {{}}}}}}

- "Put TDP43 Vehicle controls in Reference Group"
  → {{"group_a": {{"filters": {{}}}}, "group_b": {{"filters": {{"Genotype": ["TDP43"], "Treatment": ["Vehicle"]}}}}}}

- "Compare C9orf72 mutant iMNs against wild-type controls"
  → {{"group_a": {{"filters": {{"Genotype": ["C9orf72"], "CellType": ["iMN"]}}}}, "group_b": {{"filters": {{"Genotype": ["WT"], "CellType": ["iMN"]}}}}}}

- "Compare Rotenone vs Vehicle in WT iMNs at 24h"
  → {{"group_a": {{"filters": {{"Treatment": ["Rotenone"], "Genotype": ["WT"], "CellType": ["iMN"], "Timepoint": [24]}}}}, "group_b": {{"filters": {{"Treatment": ["Vehicle"], "Genotype": ["WT"], "CellType": ["iMN"], "Timepoint": [24]}}}}}}

Return only JSON, no explanation."""

    try:
        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        response_text = _strip_code_fences(message.content[0].text.strip())
        result = json.loads(response_text)

        raw_a = result.get("group_a", {}).get("filters", {})
        raw_b = result.get("group_b", {}).get("filters", {})
        filters_a = _coerce_filters(raw_a)
        filters_b = _coerce_filters(raw_b)

        samples_a = _resolve_filters(filters_a, metadata_df) if filters_a else []
        samples_b = _resolve_filters(filters_b, metadata_df) if filters_b else []

        return {
            "group_a": {"filters": filters_a, "samples": samples_a},
            "group_b": {"filters": filters_b, "samples": samples_b},
            "error": None,
        }

    except Exception as e:
        empty = {"filters": {}, "samples": []}
        return {"group_a": empty, "group_b": empty, "error": str(e)}
