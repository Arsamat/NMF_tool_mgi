"""Research LLM pipeline UI for DEG results."""

import io

import requests
import streamlit as st

from deg.viz_helpers.common import DEG_API_URL, ensure_research_session


def render_research_section(deg_df, state_prefix: str, widget_prefix: str):
    st.divider()
    st.subheader("AI-Powered Research Insights")
    st.caption(
        "Generate disease associations, drug response predictions, and research hypotheses using Claude LLM."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state[f"{state_prefix}research_disease_context"] = st.text_input(
            "Provide disease context:",
            key=f"{widget_prefix}disease_input",
        )
    with col2:
        st.session_state[f"{state_prefix}research_tissue"] = st.text_input(
            "Tissue type:",
            key=f"{widget_prefix}tissue_input",
        )
    with col3:
        st.session_state[f"{state_prefix}research_num_genes"] = st.number_input(
            "Number of top DE genes to use for analysis:",
            min_value=1,
            max_value=100,
            value=st.session_state.get(f"{state_prefix}research_num_genes", 10),
            key=f"{widget_prefix}num_genes_input",
        )
    with col4:
        st.session_state[f"{state_prefix}comparison_description"] = st.text_input(
            "Describe the context of the DEG comparison:",
            key=f"{widget_prefix}comparison_description_input",
        )

    if st.button("Run Research Pipeline", type="primary", key=f"{widget_prefix}research_run"):
        ensure_research_session(state_prefix)
        with st.spinner("Running research pipeline… generating LLM predictions (this may take 1–2 minutes)"):
            try:
                api_url = st.session_state.get("deg_api_url", DEG_API_URL)
                deg_csv_bytes = deg_df.to_csv(index=False).encode()
                files = {"deg_table": ("deg_results.csv", io.BytesIO(deg_csv_bytes), "text/csv")}
                data = {
                    "disease_context": st.session_state[f"{state_prefix}research_disease_context"],
                    "tissue": st.session_state[f"{state_prefix}research_tissue"],
                    "num_genes": st.session_state[f"{state_prefix}research_num_genes"],
                    "comparison_description": st.session_state[f"{state_prefix}comparison_description"],
                }
                resp = requests.post(
                    f"{api_url}deg_with_research/",
                    files=files,
                    data=data,
                    timeout=300,
                )
                resp.raise_for_status()
                st.session_state[f"{state_prefix}research_results"] = resp.json()
                st.success("Research pipeline completed successfully!")
                st.rerun()
            except requests.exceptions.RequestException as e:
                try:
                    err_msg = e.response.json().get("detail", str(e)) if e.response is not None else str(e)
                except Exception:
                    err_msg = str(e)
                st.error(f"Pipeline failed: {err_msg}")
            except Exception as e:
                st.error(f"Error: {e}")

    research_results = st.session_state.get(f"{state_prefix}research_results")
    if research_results is None:
        return

    st.markdown("---")

    if "deg_summary" in research_results:
        with st.expander("Analysis Summary", expanded=True):
            summary_text = research_results["deg_summary"]
            st.write(summary_text)
            st.download_button(
                "Download analysis summary (TXT)",
                summary_text,
                "deg_analysis_summary.txt",
                "text/plain",
                key=f"{widget_prefix}summary_download",
            )

    if "biological_context" in research_results:
        with st.expander("Biological Context"):
            bio_text = research_results["biological_context"]
            st.text(bio_text)
            st.download_button(
                "Download biological context (TXT)",
                bio_text,
                "deg_biological_context.txt",
                "text/plain",
                key=f"{widget_prefix}bio_download",
            )

    if "disease_predictions" in research_results:
        pred = research_results["disease_predictions"]
        with st.expander("Disease Association Predictions"):
            pred_text = pred.get("predictions", "")
            st.write(pred_text)
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"Model: {pred.get('model', 'Unknown')}")
            with col_b:
                st.caption(f"Tokens used: {pred.get('tokens_used', 'N/A')}")
            st.download_button(
                "Download disease predictions (TXT)",
                pred_text,
                "deg_disease_predictions.txt",
                "text/plain",
                key=f"{widget_prefix}disease_download",
            )

    if "drug_predictions" in research_results:
        pred = research_results["drug_predictions"]
        with st.expander("Drug Response Predictions"):
            pred_text = pred.get("predictions", "")
            st.write(pred_text)
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"Model: {pred.get('model', 'Unknown')}")
            with col_b:
                st.caption(f"Tokens used: {pred.get('tokens_used', 'N/A')}")
            st.download_button(
                "Download drug predictions (TXT)",
                pred_text,
                "deg_drug_predictions.txt",
                "text/plain",
                key=f"{widget_prefix}drug_download",
            )

    if "research_hypotheses" in research_results:
        hyp = research_results["research_hypotheses"]
        with st.expander("Novel Research Hypotheses"):
            hyp_text = hyp.get("hypotheses", "")
            st.write(hyp_text)
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption(f"Model: {hyp.get('model', 'Unknown')}")
            with col_b:
                st.caption(f"Tokens used: {hyp.get('tokens_used', 'N/A')}")
            st.download_button(
                "Download research hypotheses (TXT)",
                hyp_text,
                "deg_research_hypotheses.txt",
                "text/plain",
                key=f"{widget_prefix}hypotheses_download",
            )
