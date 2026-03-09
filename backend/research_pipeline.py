"""
Orchestrates the complete DEG analysis to research insights pipeline.
Combines DEG analysis, literature context retrieval, and LLM predictions.
"""
import logging
import pandas as pd
from typing import Dict, List, Optional
import tempfile
import subprocess
import os

from deg_parser import DEGAnalysis, DEGResult
from literature_retriever import LiteratureRetriever
from llm_predictor import BiomedicalPredictor

logger = logging.getLogger(__name__)


class ResearchPipeline:
    """Complete pipeline for DEG analysis to research insights."""

    def __init__(self):
        """Initialize pipeline components."""
        self.literature_retriever = LiteratureRetriever()
        self.predictor = None

        try:
            self.predictor = BiomedicalPredictor()
            logger.info("BiomedicalPredictor initialized successfully")
        except ValueError as e:
            logger.warning(f"BiomedicalPredictor not available: {e}")
            logger.warning("LLM predictions will not be available without API key")

    def run_deg_analysis(
        self,
        counts_df: pd.DataFrame,
        group_a: List[str],
        group_b: List[str]
    ) -> pd.DataFrame:
        """
        Run DEG analysis using R script (limma/edgeR).

        Args:
            counts_df: DataFrame with gene counts (first column = gene ID, rest = samples)
            group_a: Sample names for group A
            group_b: Sample names for group B

        Returns:
            DataFrame with DEG results
        """
        # Build metadata
        meta_rows = []
        for s in group_a:
            meta_rows.append({"SampleName": s, "Group": "GroupA"})
        for s in group_b:
            meta_rows.append({"SampleName": s, "Group": "GroupB"})
        metadata_df = pd.DataFrame(meta_rows)

        # Ensure counts has the required samples
        gene_col = "Geneid" if "Geneid" in counts_df.columns else counts_df.columns[0]
        available_samples = [c for c in counts_df.columns if c != gene_col]
        requested = set(metadata_df["SampleName"])
        missing = requested - set(available_samples)
        if missing:
            raise ValueError(f"Samples not found in counts: {missing}")

        # Subset counts to requested samples
        count_cols = [gene_col] + list(metadata_df["SampleName"])
        counts_subset = counts_df[[c for c in count_cols if c in counts_df.columns]].copy()
        counts_subset = counts_subset.rename(columns={gene_col: "gene"})

        # Find R script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        r_script = os.path.join(script_dir, "deg_analysis.R")
        if not os.path.isfile(r_script):
            raise FileNotFoundError(f"DEG R script not found: {r_script}")

        # Run R analysis
        with tempfile.TemporaryDirectory() as tmpdir:
            count_path = os.path.join(tmpdir, "counts.csv")
            meta_path = os.path.join(tmpdir, "metadata.csv")
            out_path = os.path.join(tmpdir, "deg_results.csv")

            counts_subset.to_csv(count_path, index=False)
            metadata_df.to_csv(meta_path, index=False)

            r_args = [r_script, count_path, meta_path, out_path, "human"]

            logger.info(f"Running R DEG analysis...")
            result = subprocess.run(
                r_args,
                cwd=script_dir,
                timeout=300,
            )

            if result.returncode != 0:
                raise RuntimeError("DEG analysis failed. Check logs for R output.")

            if not os.path.isfile(out_path):
                raise RuntimeError("DEG script did not produce output file")

            deg_df = pd.read_csv(out_path)
            logger.info(f"DEG analysis complete. Found {len(deg_df)} genes.")

            return deg_df

    def parse_deg_results(
        self,
        deg_df: pd.DataFrame,
        disease_context: str = None,
        tissue: str = None
    ) -> DEGAnalysis:
        """
        Parse DEG results from DataFrame.

        Args:
            deg_df: DataFrame with DEG results (must include columns for gene, log2fc, padj)
            disease_context: Disease/condition being studied
            tissue: Tissue type being studied

        Returns:
            DEGAnalysis object
        """
        deg_analysis = DEGAnalysis(disease_context=disease_context, tissue=tissue)
        deg_analysis.from_dataframe(deg_df)
        return deg_analysis

    def get_biological_context(self, deg_analysis: DEGAnalysis, num_genes: int = 10) -> str:
        """
        Retrieve biological context for top genes.

        Args:
            deg_analysis: DEGAnalysis object
            num_genes: Number of top genes to get context for

        Returns:
            Formatted biological context string
        """
        top_genes = deg_analysis.get_top_genes(num_genes, by="padj")
        gene_names = [deg.gene for deg in top_genes]

        logger.info(f"Retrieving biological context for {len(gene_names)} genes")
        context = self.literature_retriever.generate_context_for_llm(gene_names)
        return context

    def predict_disease_associations(
        self,
        deg_analysis: DEGAnalysis,
        num_genes: int = 10
    ) -> Optional[Dict]:
        """Generate disease association predictions."""
        if not self.predictor:
            logger.warning("Predictor not available - skipping disease predictions")
            return None

        top_genes = deg_analysis.get_top_genes(num_genes, by="padj")
        genes_data = [deg.to_dict() for deg in top_genes]

        context = self.get_biological_context(deg_analysis, num_genes)

        logger.info("Generating disease association predictions")
        predictions = self.predictor.predict_disease_associations(
            genes=genes_data,
            biological_context=context,
            disease_context=deg_analysis.disease_context,
            tissue=deg_analysis.tissue
        )

        return predictions

    def predict_drug_response(
        self,
        deg_analysis: DEGAnalysis,
        num_genes: int = 10
    ) -> Optional[Dict]:
        """Generate drug response predictions."""
        if not self.predictor:
            logger.warning("Predictor not available - skipping drug predictions")
            return None

        top_genes = deg_analysis.get_top_genes(num_genes, by="padj")
        genes_data = [deg.to_dict() for deg in top_genes]

        context = self.get_biological_context(deg_analysis, num_genes)

        logger.info("Generating drug response predictions")
        predictions = self.predictor.predict_drug_response(
            genes=genes_data,
            biological_context=context,
            disease_context=deg_analysis.disease_context
        )

        return predictions

    def generate_hypotheses(
        self,
        deg_analysis: DEGAnalysis,
        num_genes: int = 10
    ) -> Optional[Dict]:
        """Generate novel research hypotheses."""
        if not self.predictor:
            logger.warning("Predictor not available - skipping hypothesis generation")
            return None

        top_genes = deg_analysis.get_top_genes(num_genes, by="padj")
        genes_data = [deg.to_dict() for deg in top_genes]

        context = self.get_biological_context(deg_analysis, num_genes)

        logger.info("Generating research hypotheses")
        hypotheses = self.predictor.generate_research_hypothesis(
            genes=genes_data,
            biological_context=context,
            disease_context=deg_analysis.disease_context
        )

        return hypotheses

    def full_analysis(
        self,
        deg_df: pd.DataFrame,
        disease_context: str = None,
        tissue: str = None,
        num_genes: int = 10
    ) -> Dict:
        """
        Run complete analysis pipeline from pre-computed DEG results.

        Args:
            deg_df: DEG results DataFrame (must include columns: gene, log2fc, padj)
            disease_context: Disease/condition being studied
            tissue: Tissue type being studied
            num_genes: Number of top genes to analyze

        Returns:
            Dictionary with all analysis results
        """
        # Parse DEG results
        logger.info("Parsing DEG results...")
        deg_analysis = self.parse_deg_results(deg_df, disease_context, tissue)

        results = {
            "status": "success",
            "deg_summary": deg_analysis.summary(),
            "total_genes": len(deg_analysis.degs),
            "disease_context": disease_context,
            "tissue": tissue,
            "deg_data": deg_analysis.to_dict()
        }

        # Get biological context
        logger.info("Retrieving biological context...")
        context = self.get_biological_context(deg_analysis, num_genes)
        results["biological_context"] = context

        # Run predictions if predictor is available
        if self.predictor:
            try:
                logger.info("Running LLM predictions...")

                # Disease associations
                disease_pred = self.predict_disease_associations(deg_analysis, num_genes)
                if disease_pred:
                    results["disease_predictions"] = disease_pred

                # Drug response
                drug_pred = self.predict_drug_response(deg_analysis, num_genes)
                if drug_pred:
                    results["drug_predictions"] = drug_pred

                # Research hypotheses
                hyp = self.generate_hypotheses(deg_analysis, num_genes)
                if hyp:
                    results["research_hypotheses"] = hyp

            except Exception as e:
                logger.error(f"Error during LLM predictions: {e}")
                results["prediction_error"] = str(e)
        else:
            results["note"] = "LLM predictions not available (no API key)"

        return results
