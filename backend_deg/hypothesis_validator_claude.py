"""
Hypothesis novelty validator using Claude as a true agent with PubMed tool access.

Claude controls the entire search strategy: it formulates queries, calls the
search_pubmed tool, reads returned abstracts, decides if more searches are needed,
and provides a final reasoned novelty assessment.
"""
import requests
import logging
import xml.etree.ElementTree as ET
import json
from typing import List, Dict, Optional
from anthropic import Anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── PubMed tool implementation ────────────────────────────────────────────────
# This function is executed by us when Claude requests the tool.
# Claude never calls NCBI directly — we do it on its behalf.


def _pubmed_search(query: str, max_results: int = 8) -> str:
    """
    Execute a PubMed search and return formatted results as a string for Claude.

    Args:
        query: PubMed search query (Claude-formulated)
        max_results: Max papers to return

    Returns:
        Formatted string with papers (PMID, title, abstract) or error message
    """
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    headers = {"User-Agent": "MGI-HypothesisValidator/1.0"}

    try:
        # Step 1: Get PMIDs
        r = requests.get(
            f"{BASE_URL}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "tool": "mgi_hypothesis_validator",
                "email": "research@example.com",
            },
            headers=headers,
            timeout=10,
        )
        r.raise_for_status()
        pmids = r.json().get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return f"No papers found for query: '{query}'"

        # Step 2: Fetch abstracts via XML
        r = requests.get(
            f"{BASE_URL}/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "rettype": "abstract",
                "retmode": "xml",
                "tool": "mgi_hypothesis_validator",
                "email": "research@example.com",
            },
            headers=headers,
            timeout=15,
        )
        r.raise_for_status()

        papers = []
        root = ET.fromstring(r.content)
        for article in root.findall(".//PubmedArticle"):
            pmid_elem = article.find(".//PMID")
            title_elem = article.find(".//ArticleTitle")
            abstract_elem = article.find(".//AbstractText")

            pmid = pmid_elem.text if pmid_elem is not None else None
            title = title_elem.text if title_elem is not None else None
            abstract = abstract_elem.text if abstract_elem is not None else None

            if pmid and title and abstract:
                papers.append(
                    {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract[:600],
                    }
                )

        if not papers:
            return f"Found {len(pmids)} PMIDs but could not fetch abstracts."

        # Format as readable text for Claude
        lines = [f"Found {len(papers)} papers for '{query}':\n"]
        for i, p in enumerate(papers, 1):
            lines.append(
                f"[{i}] PMID: {p['pmid']}\n"
                f"    Title: {p['title']}\n"
                f"    Abstract: {p['abstract']}\n"
            )

        return "\n".join(lines)

    except requests.exceptions.RequestException as e:
        return f"PubMed API error: {e}"
    except ET.ParseError as e:
        return f"XML parse error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


# ── Tool definition passed to Claude ─────────────────────────────────────────

PUBMED_TOOL = {
    "name": "search_pubmed",
    "description": (
        "Search PubMed for biomedical research papers. Returns paper titles, PMIDs, "
        "and abstracts. Use this to find existing literature related to a research "
        "hypothesis so you can assess its novelty. You may call this tool multiple "
        "times with different queries to cover different aspects of the hypothesis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "PubMed search query. Use specific biomedical terms: gene names, "
                    "protein names, disease names, molecular mechanisms. Supports AND/OR. "
                    "Examples: 'TDP-43 AND autophagy AND ALS', 'miR-146a neuroinflammation'"
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Max papers to return. Default 8, maximum 15.",
                "default": 8,
            },
        },
        "required": ["query"],
    },
}


class HypothesisValidatorClaude:
    """
    Validate hypotheses using Claude as an agent with PubMed tool access.

    Claude formulates its own search queries, reads papers, decides when to
    search again with different terms, and produces a final reasoned assessment.
    """

    SYSTEM_PROMPT = """You are an expert biomedical researcher specializing in assessing the novelty of research hypotheses.

Your task is to determine whether a given hypothesis proposes something genuinely new, or whether it overlaps significantly with existing published work.

Instructions:
1. Analyze the hypothesis to identify its core claims: key genes, proteins, pathways, mechanisms, or disease contexts.
2. Use the search_pubmed tool to find relevant literature. Formulate specific, targeted queries.
3. Read the returned abstracts carefully.
4. If the results are insufficient or miss key aspects of the hypothesis, search again with different or more specific terms.
5. Once you have enough information, provide your final assessment.

Novelty levels:
- HIGH: The specific mechanism or relationship proposed has not been published.
- MODERATE-HIGH: The hypothesis extends existing findings in a clearly new direction.
- MODERATE: Related work exists but the hypothesis has a distinct angle or context.
- LOW: Very similar work has already been published.

When you are done searching, respond with ONLY a JSON object in this exact format:
{
    "novelty_level": "HIGH|MODERATE-HIGH|MODERATE|LOW",
    "novelty_explanation": "One clear sentence summarizing why this level was chosen",
    "novel_aspects": "What is genuinely new or unexplored about this hypothesis",
    "related_work": "The top-5 most similar existing papers and how they relate",
    "key_papers": [
        {"pmid": "12345678", "title": "Paper title", "relevance": "Why this paper matters"}
    ],
    "confidence": "HIGH|MEDIUM|LOW",
    "searches_performed": ["query1", "query2"]
}

Do not include any text outside the JSON."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            model: Claude model. Opus recommended for best reasoning quality.
            api_key: Anthropic API key. Uses ANTHROPIC_API_KEY env var if not provided.
        """
        self.model = model
        self.client = Anthropic(api_key=api_key)
        logger.info(f"Claude agent initialized: {model}")

    def validate_hypothesis(
        self,
        hypothesis: str,
        max_papers_per_search: int = 8,
        max_iterations: int = 10,
    ) -> Dict:
        """
        Validate a single hypothesis via the Claude agent + PubMed tool loop.

        The agent loop runs until Claude stops calling tools (stop_reason == "end_turn"),
        at which point we parse Claude's final JSON assessment.

        Args:
            hypothesis: Research hypothesis text
            max_papers_per_search: Max papers returned per tool call (capped at 15)
            max_iterations: Max agentic loop iterations (default 10, prevents infinite loops)

        Returns:
            Dict with novelty_assessment, novelty_explanation, novel_aspects,
            related_work, key_papers, confidence, searches_performed
        """
        logger.info(f"Starting agent validation: {hypothesis[:80]}...")

        messages = [
            {
                "role": "user",
                "content": (
                    "Please assess the novelty of the following research hypothesis "
                    "by searching PubMed for related work:\n\n" + hypothesis
                ),
            }
        ]

        searches_performed: List[str] = []
        iteration = 0

        # Agentic loop
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Agent loop iteration {iteration}/{max_iterations}")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                system=self.SYSTEM_PROMPT,
                tools=[PUBMED_TOOL],
                messages=messages,
            )

            logger.info(f"stop_reason: {response.stop_reason}")

            # Add Claude's response to conversation history
            messages.append({"role": "assistant", "content": response.content})

            # Claude is done — parse final JSON assessment
            if response.stop_reason == "end_turn":
                final_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        final_text = block.text.strip()

                try:
                    # Strip markdown code fences if present
                    if "```json" in final_text:
                        final_text = (
                            final_text.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in final_text:
                        final_text = final_text.split("```")[1].split("```")[0].strip()

                    analysis = json.loads(final_text)

                    return {
                        "hypothesis": hypothesis,
                        "novelty_assessment": analysis.get(
                            "novelty_level", "UNKNOWN"
                        ),
                        "novelty_explanation": analysis.get(
                            "novelty_explanation", ""
                        ),
                        "novel_aspects": analysis.get("novel_aspects", ""),
                        "related_work": analysis.get("related_work", ""),
                        "key_papers": analysis.get("key_papers", []),
                        "confidence": analysis.get("confidence", "MEDIUM"),
                        "searches_performed": analysis.get(
                            "searches_performed", searches_performed
                        ),
                    }

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(
                        f"Failed to parse Claude's final response: {e}\n{final_text}"
                    )
                    return {
                        "hypothesis": hypothesis,
                        "novelty_assessment": "UNKNOWN",
                        "novelty_explanation": f"Could not parse response: {e}",
                        "novel_aspects": "",
                        "related_work": "",
                        "key_papers": [],
                        "confidence": "LOW",
                        "searches_performed": searches_performed,
                        "error": str(e),
                    }

            # Max tokens reached — ask Claude for a final JSON assessment without more tool calls
            if response.stop_reason == "max_tokens":
                logger.warning("Reached max_tokens limit. Asking Claude for final summary without further tool use.")

                try:
                    # Ask Claude to summarize based on the conversation so far, with tools disabled
                    final_response = self.client.messages.create(
                        model=self.model,
                        max_tokens=2000,
                        system=self.SYSTEM_PROMPT,
                        tools=[],
                        messages=messages
                        + [
                            {
                                "role": "user",
                                "content": (
                                    "Based on the literature and reasoning above, "
                                    "now provide your FINAL JSON assessment in the required format. "
                                    "Do NOT call any tools."
                                ),
                            }
                        ],
                    )

                    final_text = ""
                    for block in final_response.content:
                        if hasattr(block, "text"):
                            final_text = block.text.strip()

                    if not final_text:
                        raise ValueError("Empty final response after max_tokens.")

                    # Strip markdown code fences if present
                    if "```json" in final_text:
                        final_text = (
                            final_text.split("```json")[1].split("```")[0].strip()
                        )
                    elif "```" in final_text:
                        final_text = final_text.split("```")[1].split("```")[0].strip()

                    analysis = json.loads(final_text)

                    return {
                        "hypothesis": hypothesis,
                        "novelty_assessment": analysis.get(
                            "novelty_level", "UNKNOWN"
                        ),
                        "novelty_explanation": analysis.get(
                            "novelty_explanation", ""
                        ),
                        "novel_aspects": analysis.get("novel_aspects", ""),
                        "related_work": analysis.get("related_work", ""),
                        "key_papers": analysis.get("key_papers", []),
                        "confidence": analysis.get("confidence", "MEDIUM"),
                        "searches_performed": analysis.get(
                            "searches_performed", searches_performed
                        ),
                        "warning": "Assessment completed after hitting initial token limit.",
                    }
                except Exception as e:
                    logger.error(
                        f"Failed to obtain final summary after max_tokens: {e}"
                    )
                    return {
                        "hypothesis": hypothesis,
                        "novelty_assessment": "UNKNOWN",
                        "novelty_explanation": (
                            "Assessment stopped due to token limit and a "
                            "follow‑up summary request also failed."
                        ),
                        "novel_aspects": "",
                        "related_work": "",
                        "key_papers": [],
                        "confidence": "LOW",
                        "searches_performed": searches_performed,
                        "warning": "Partial assessment - token limit reached twice",
                    }

            # Claude called a tool — execute it and return results
            if response.stop_reason == "tool_use":
                tool_results = []

                for block in response.content:
                    if getattr(block, "type", None) != "tool_use":
                        continue

                    if block.name == "search_pubmed":
                        query = block.input.get("query", "")
                        max_r = min(
                            block.input.get("max_results", max_papers_per_search), 15
                        )

                        logger.info(f"Claude → search_pubmed('{query}', max={max_r})")
                        searches_performed.append(query)

                        result_text = _pubmed_search(query, max_results=max_r)

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            }
                        )

                # Return tool results to Claude so it can continue
                messages.append({"role": "user", "content": tool_results})

        # Iteration limit reached — ask Claude for a final JSON assessment without more tool calls
        logger.warning(f"Reached max iterations ({max_iterations}). Requesting final summary without further tool use.")
        try:
            final_response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=self.SYSTEM_PROMPT,
                tools=[],
                messages=messages
                + [
                    {
                        "role": "user",
                        "content": (
                            "You have reached the iteration limit for tool calls. "
                            "Based on the literature and reasoning so far, "
                            "now provide your FINAL JSON assessment in the required format. "
                            "Do NOT call any tools."
                        ),
                    }
                ],
            )

            final_text = ""
            for block in final_response.content:
                if hasattr(block, "text"):
                    final_text = block.text.strip()

            if not final_text:
                raise ValueError("Empty final response after iteration limit.")

            if "```json" in final_text:
                final_text = final_text.split("```json")[1].split("```")[0].strip()
            elif "```" in final_text:
                final_text = final_text.split("```")[1].split("```")[0].strip()

            analysis = json.loads(final_text)

            return {
                "hypothesis": hypothesis,
                "novelty_assessment": analysis.get("novelty_level", "UNKNOWN"),
                "novelty_explanation": analysis.get("novelty_explanation", ""),
                "novel_aspects": analysis.get("novel_aspects", ""),
                "related_work": analysis.get("related_work", ""),
                "key_papers": analysis.get("key_papers", []),
                "confidence": analysis.get("confidence", "MEDIUM"),
                "searches_performed": analysis.get(
                    "searches_performed", searches_performed
                ),
                "warning": (
                    f"Assessment completed after reaching the iteration limit "
                    f"({max_iterations})."
                ),
            }
        except Exception as e:
            logger.error(
                f"Failed to obtain final summary after iteration limit: {e}"
            )
            return {
                "hypothesis": hypothesis,
                "novelty_assessment": "UNKNOWN",
                "novelty_explanation": (
                    "Assessment stopped after reaching the iteration limit and a "
                    "follow‑up summary request also failed."
                ),
                "novel_aspects": "",
                "related_work": "",
                "key_papers": [],
                "confidence": "LOW",
                "searches_performed": searches_performed,
                "warning": f"Stopped after {max_iterations} iterations",
            }

    def validate_multiple(
        self,
        hypotheses: List[str],
        max_papers_per_search: int = 8,
        max_iterations: int = 20,
    ) -> List[Dict]:
        """
        Validate a list of hypotheses sequentially.

        Args:
            hypotheses: List of hypothesis strings
            max_papers_per_search: Max papers per PubMed tool call
            max_iterations: Max agentic loop iterations per hypothesis

        Returns:
            List of validation result dicts
        """
        results: List[Dict] = []
        for i, hyp in enumerate(hypotheses, 1):
            logger.info(f"\n=== Hypothesis {i}/{len(hypotheses)} ===")
            result = self.validate_hypothesis(
                hyp,
                max_papers_per_search=max_papers_per_search,
                max_iterations=max_iterations,
            )
            results.append(result)
        return results

