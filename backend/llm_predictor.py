"""
Uses Claude LLM to generate biological predictions from DEG analysis results
and biological context.
"""
from typing import List, Dict, Optional
from anthropic import Anthropic
import os
import json


class BiomedicalPredictor:
    """Uses Claude to generate biomedical predictions."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-6"

    def predict_disease_associations(
        self,
        genes: List[Dict],
        biological_context: str,
        disease_context: str = None,
        tissue: str = None
    ) -> Dict:
        """
        Predict how genes could affect diseases based on expression changes.

        Args:
            genes: List of dicts with gene, log2fc, padj
            biological_context: Context string from LiteratureRetriever
            disease_context: The disease being studied
            tissue: The tissue being studied

        Returns:
            Dict with predictions, mechanisms, and confidence scores
        """

        # Format gene data for the prompt
        gene_list = "\n".join([
            f"- {g['gene']}: log2FC={g['log2fc']:.2f} (padj={g['padj']:.2e})"
            for g in genes
        ])

        prompt = f"""
You are an expert biomedical researcher analyzing differential gene expression (DEG) results.

{biological_context}

DIFFERENTIAL EXPRESSION RESULTS:
{gene_list}

Disease Context: {disease_context or 'General analysis'}
Tissue: {tissue or 'Not specified'}

Based on these genes and their expression changes, provide a focused analysis:

1. DISEASE ASSOCIATIONS (max 3-4 diseases):
   - Primary disease associations and confidence level (high/moderate/low)
   - Mechanism explanation for the top disease

2. TREATMENT POSSIBILITIES (max 3-4 options):
   - Most relevant drugs/therapeutic approaches
   - Brief rationale for each

3. PATHWAY ANALYSIS (max 2-3 pathways):
   - Main biological pathways affected
   - Brief description of interconnections

4. EXPERIMENTAL VALIDATION (1-2 suggestions):
   - Most critical experiments to validate top predictions

CRITICAL INSTRUCTIONS:
- Do NOT include gene identity summary tables or comprehensive gene lists.
- Only mention specific genes when directly supporting your main findings.
- Strictly respect the maximum number of items in each section (3-4, 2-3, 1-2).
- Focus exclusively on the most important, high-confidence results.
- Only mention treatments/diseases with strong scientific evidence.
- Prioritize findings over comprehensive coverage - less is more.
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text

        return {
            "predictions": response_text,
            "model": self.model,
            "tokens_used": message.usage.output_tokens
        }

    def predict_drug_response(
        self,
        genes: List[Dict],
        biological_context: str,
        disease_context: str = None
    ) -> Dict:
        """
        Predict potential drug response based on gene expression changes.
        """

        gene_list = "\n".join([
            f"- {g['gene']}: log2FC={g['log2fc']:.2f}"
            for g in genes
        ])

        prompt = f"""
You are a biomedical informatics expert specializing in pharmacogenomics.

{biological_context}

DIFFERENTIAL EXPRESSION IN {disease_context or 'Patient Sample'}:
{gene_list}

Provide focused drug response predictions:

1. DRUG SENSITIVITY (max 3-4 drugs):
   - Most relevant drugs this expression profile is sensitive/resistant to
   - Brief mechanism for each

2. REPURPOSING OPPORTUNITIES (max 2-3):
   - Currently approved drugs that could be repurposed
   - Brief rationale

3. BIOMARKERS (max 2 genes):
   - Top genes suitable for treatment selection biomarkers

4. DRUG INTERACTIONS (1-2 key interactions):
   - Most critical gene-drug interactions to consider

CRITICAL INSTRUCTIONS:
- Do NOT include gene identity summary tables or gene lists.
- Only mention specific genes when directly relevant to drug response predictions.
- Strictly respect the maximum number of items in each section (3-4, 2-3, 1-2).
- Focus exclusively on high-confidence, actionable predictions.
- Only include drugs/interactions with strong clinical or mechanistic evidence.
- Be specific and concise - prioritize quality over comprehensiveness.
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text

        return {
            "predictions": response_text,
            "model": self.model,
            "tokens_used": message.usage.output_tokens
        }

    def generate_research_hypothesis(
        self,
        genes: List[Dict],
        biological_context: str,
        disease_context: str = None
    ) -> Dict:
        """
        Generate novel research hypotheses from DEG results.
        """

        gene_list = "\n".join([
            f"- {g['gene']}: log2FC={g['log2fc']:.2f} ({g.get('direction', 'unknown')})"
            for g in genes
        ])

        prompt = f"""
You are a creative biomedical researcher brainstorming novel hypotheses.

{biological_context}

DIFFERENTIAL EXPRESSION RESULTS:
{gene_list}

Disease/Context: {disease_context or 'Unknown'}

Propose 2-3 novel, scientifically grounded research hypotheses:

For each hypothesis:
1. Clear hypothesis statement
2. Biological rationale (brief)
3. Key experiment to test it
4. Clinical relevance/impact

CRITICAL INSTRUCTIONS:
- Do NOT include gene identity summary tables or comprehensive gene annotations.
- Only mention genes in your hypotheses when they directly support the proposed mechanism.
- Generate exactly 2-3 hypotheses - no more, no fewer.
- Focus on non-obvious connections and novel angles, not obvious associations.
- All hypotheses must be scientifically plausible and testable.
- Be concise but specific - each hypothesis should be clear enough for researchers to pursue.
- Avoid redundancy - ensure each hypothesis is distinct and adds new insight.
"""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2500,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        response_text = message.content[0].text

        return {
            "hypotheses": response_text,
            "model": self.model,
            "tokens_used": message.usage.output_tokens
        }
