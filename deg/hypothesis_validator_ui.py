"""
Streamlit UI for displaying hypothesis validation results.

Shows novelty assessments and similar papers for research hypotheses.
"""
import streamlit as st
import pandas as pd
import requests
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed


def display_hypothesis_validation_results(validation_results: Dict):
    """
    Display hypothesis validation results in Streamlit.

    Args:
        validation_results: Dict from /validate_hypotheses/ endpoint containing:
            - status: "success" or error
            - total_hypotheses: number of hypotheses
            - results: list of validation dicts
            - summary: novelty counts
    """
    if "error" in validation_results and validation_results["error"]:
        st.error(f"Validation error: {validation_results['error']}")
        return

    # Support both embedding-based and Claude-based responses
    status = validation_results.get("status")
    if status is not None and status != "success":
        st.warning("No validation results available")
        return

    st.markdown("---")
    st.subheader("📚 Hypothesis Literature Validation")
    st.caption("Each hypothesis is validated against PubMed literature to assess novelty")

    # Display summary if present (embedding- or Claude-based)
    summary = validation_results.get("summary", {})
    if summary:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                "Total",
                summary.get("high_novelty", 0)
                + summary.get("moderate_novelty", 0)
                + summary.get("low_novelty", 0),
            )
        with col2:
            st.metric("🟢 High Novelty", summary.get("high_novelty", 0))
        with col3:
            st.metric("🟡 Moderate", summary.get("moderate_novelty", 0))
        with col4:
            st.metric("🔴 Low Novelty", summary.get("low_novelty", 0))

    # Display detailed results
    results = validation_results.get("results", [])

    if not results:
        st.info("No validation results to display")
        return

    # Explain novelty / confidence levels once, using backend definition
    st.markdown("**Confidence / novelty levels (backend definition):**")
    st.markdown(
        "- **HIGH**: The specific mechanism or relationship proposed has not been published.\n"
        "- **MODERATE-HIGH**: The hypothesis extends existing findings in a clearly new direction.\n"
        "- **MODERATE**: Related work exists but the hypothesis has a distinct angle or context.\n"
        "- **LOW**: Very similar work has already been published."
    )

    for idx, result in enumerate(results, 1):
        with st.expander(
            f"Hypothesis {idx}: {result.get('novelty_assessment', 'Unknown')}"
        ):
            # Hypothesis text
            st.markdown("**Hypothesis:**")
            st.write(result.get("hypothesis", ""))

            # Novelty & confidence
            novelty = result.get("novelty_assessment", "Unknown")
            confidence = result.get("confidence", "MEDIUM")

            # Embedding-based responses include a similarity score; Claude-based do not.
            max_sim = result.get("max_similarity")

            # Color code based on novelty
            if "HIGH" in novelty:
                st.success(f"✅ {novelty}" + (f" (Max similarity: {max_sim:.3f})" if max_sim is not None else ""))
            elif "MODERATE-HIGH" in novelty:
                st.info(f"ℹ️ {novelty}" + (f" (Max similarity: {max_sim:.3f})" if max_sim is not None else ""))
            elif "MODERATE" in novelty:
                st.warning(f"⚠️ {novelty}" + (f" (Max similarity: {max_sim:.3f})" if max_sim is not None else ""))
            else:
                st.error(f"❌ {novelty}" + (f" (Max similarity: {max_sim:.3f})" if max_sim is not None else ""))

            st.caption(f"Confidence: {confidence}")

            # Embedding-based extras
            keywords = result.get("search_keywords")
            similar_papers = result.get("similar_papers")
            papers_searched = result.get("papers_searched")

            if keywords:
                st.markdown("**Search Keywords:**")
                st.write(", ".join(keywords))

            if similar_papers:
                st.markdown(
                    f"**Top Similar Papers**"
                    + (f" ({papers_searched} papers searched):" if papers_searched is not None else ":")
                )

                for paper_idx, paper in enumerate(similar_papers, 1):
                    with st.container():
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.markdown(f"**{paper_idx}. {paper['title']}**")
                            st.caption(f"PMID: {paper['pmid']}")
                            st.write(paper["abstract"])
                        with col2:
                            similarity = paper.get("similarity_score", None)
                            if similarity is not None:
                                st.metric("Similarity", f"{similarity:.3f}")
                        st.divider()

            # Claude-based extras
            novelty_explanation = result.get("novelty_explanation")
            novel_aspects = result.get("novel_aspects")
            related_work = result.get("related_work")
            key_papers = result.get("key_papers")
            searches_performed = result.get("searches_performed")

            if novelty_explanation:
                st.markdown("**Explanation:**")
                st.write(novelty_explanation)

            if novel_aspects:
                st.markdown("**Novel Aspects:**")
                st.write(novel_aspects)

            if related_work:
                st.markdown("**Related Work (summary):**")
                st.write(related_work)

            if key_papers:
                st.markdown("**Key Papers (Claude selection):**")
                for i, paper in enumerate(key_papers, 1):
                    st.markdown(f"**{i}. {paper.get('title', 'N/A')}**")
                    pmid = paper.get("pmid")
                    if pmid:
                        st.caption(f"PMID: {pmid}")
                    relevance = paper.get("relevance")
                    if relevance:
                        st.write(f"Relevance: {relevance}")
                    st.divider()

            if searches_performed:
                st.markdown("**PubMed searches performed:**")
                st.write(", ".join(searches_performed))

            # Allow user to download a text summary for this hypothesis
            hyp_text = result.get("hypothesis", "")

            # Build Claude-based extras for the download
            key_papers_lines = []
            if key_papers:
                for kp in key_papers:
                    title = kp.get("title", "N/A")
                    pmid = kp.get("pmid", "N/A")
                    relevance = kp.get("relevance", "")
                    line = f"- {title} (PMID: {pmid})"
                    if relevance:
                        line += f" | Relevance: {relevance}"
                    key_papers_lines.append(line)

            searches_line = ""
            if searches_performed:
                searches_line = ", ".join(searches_performed)

            download_content = (
                f"Hypothesis {idx}\n"
                f"Novelty level: {novelty}\n"
                f"Confidence: {confidence}\n\n"
                f"Hypothesis text:\n{hyp_text}\n\n"
                f"Explanation:\n{novelty_explanation or ''}\n\n"
                f"Novel aspects:\n{novel_aspects or ''}\n\n"
                f"Related work (summary):\n{related_work or ''}\n\n"
                f"Key papers (Claude selection):\n"
                f"{chr(10).join(key_papers_lines) if key_papers_lines else 'None'}\n\n"
                f"PubMed searches performed:\n{searches_line or 'None'}\n"
            )
            st.download_button(
                label="Download this hypothesis summary",
                data=download_content,
                file_name=f"hypothesis_{idx}_summary.txt",
                mime="text/plain",
                key=f"deg_hypothesis_download_{idx}",
            )


def run_hypothesis_validation_from_text(api_url: str, hypothesis_text: str, max_papers: int = 10):
    """
    Run hypothesis validation via API.

    Args:
        api_url: Base API URL (e.g., "http://localhost:8000/")
        hypothesis_text: Raw text with hypotheses (numbered format)
        max_papers: Max papers to search per hypothesis

    Returns:
        Validation results dict or error dict
    """
    try:
        # Parse hypotheses from text
        hypotheses = parse_hypotheses_from_text(hypothesis_text)

        if not hypotheses:
            return {"error": "Could not parse hypotheses from text"}

        endpoint = "/validate_hypotheses_claude/"

        # Single hypothesis: simple call
        if len(hypotheses) == 1:
            response = requests.post(
                f"{api_url}{endpoint}",
                json={"hypotheses": hypotheses, "max_papers": max_papers},
                timeout=300,
            )
            if response.status_code != 200:
                return {
                    "error": f"API error: {response.status_code}",
                    "details": response.text,
                }
            return response.json()

        # Multiple hypotheses: call backend in parallel, one hypothesis per request
        def _call_single(hyp: str):
            resp = requests.post(
                f"{api_url}{endpoint}",
                json={"hypotheses": [hyp], "max_papers": max_papers},
                timeout=300,
            )
            return resp

        max_workers = min(4, len(hypotheses))
        responses = [None] * len(hypotheses)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_call_single, hyp): idx
                for idx, hyp in enumerate(hypotheses)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception as e:
                    return {"error": f"Request failed for hypothesis {idx + 1}: {e}"}

        # Check responses for HTTP errors
        for idx, resp in enumerate(responses):
            if resp is None:
                return {"error": f"No response for hypothesis {idx + 1}"}
            if resp.status_code != 200:
                return {
                    "error": f"API error for hypothesis {idx + 1}: {resp.status_code}",
                    "details": resp.text,
                }

        # Aggregate individual responses into a single combined structure
        combined_results = []
        for resp in responses:
            data = resp.json()
            single_results = data.get("results", [])
            if single_results:
                combined_results.append(single_results[0])

        if not combined_results:
            return {"error": "No validation results returned from API"}

        summary = {
            "high_novelty": sum(
                1
                for r in combined_results
                if "HIGH" in r.get("novelty_assessment", "")
            ),
            "moderate_novelty": sum(
                1
                for r in combined_results
                if "MODERATE" in r.get("novelty_assessment", "")
            ),
            "low_novelty": sum(
                1
                for r in combined_results
                if "LOW" in r.get("novelty_assessment", "")
            ),
        }

        return {
            "status": "success",
            "total_hypotheses": len(combined_results),
            "results": combined_results,
            "summary": summary,
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Validation error: {str(e)}"}


def parse_hypotheses_from_text(text: str) -> List[str]:
    """
    Parse multiple hypotheses from free-form text.

    Supports formats like:
      - "Hypothesis 1: ... Hypothesis 2: ..."
      - "Hypothesis 1: ...\nHypothesis 2: ..."
    """
    import re

    # Normalize whitespace a bit
    cleaned = text.strip()
    if not cleaned:
        return []

    # Primary pattern: split on "Hypothesis N:" (case-insensitive)
    pattern = r"(?i)Hypothesis\s+\d+\s*:.*?(?=Hypothesis\s+\d+\s*:|$)"
    matches = re.findall(pattern, cleaned, flags=re.S)

    hypotheses: List[str] = []
    for m in matches:
        hyp = m.strip()
        if len(hyp) > 20:
            hypotheses.append(hyp)

    # Fallback: if we only got one long chunk, try splitting on sentences containing "Hypothesis"
    if len(hypotheses) <= 1:
        alt_chunks = []
        for part in re.split(r"(?i)(?=Hypothesis\s+\d+\s*:)", cleaned):
            part = part.strip()
            if part and len(part) > 20:
                alt_chunks.append(part)
        if len(alt_chunks) > 1:
            hypotheses = alt_chunks

    return hypotheses


def validate_and_display(api_url: str, hypothesis_text: str, max_papers: int = 10):
    """
    Convenience function: validate hypotheses and display results.

    Args:
        api_url: Base API URL
        hypothesis_text: Raw hypothesis text
        max_papers: Max papers per hypothesis
    """
    with st.spinner("Validating hypotheses against literature..."):
        results = run_hypothesis_validation_from_text(
            api_url, hypothesis_text, max_papers
        )

    # Persist full results in session state for downstream use
    st.session_state["deg_hypothesis_validation_results"] = results
    st.session_state["deg_hypothesis_validation_text"] = hypothesis_text

    # Always display from session state to keep a single source of truth
    display_hypothesis_validation_results(
        st.session_state["deg_hypothesis_validation_results"]
    )
