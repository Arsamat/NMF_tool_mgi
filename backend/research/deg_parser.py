"""
Parser for Differential Expression Gene (DEG) analysis results.
"""
import pandas as pd
from typing import List, Dict
import json


class DEGResult:
    """Represents a single DEG result."""
    def __init__(self, gene: str, log2fc: float, padj: float, pvalue: float = None):
        self.gene = gene
        self.log2fc = log2fc
        self.padj = padj
        self.pvalue = pvalue
        self.direction = "upregulated" if log2fc > 0 else "downregulated"

    def to_dict(self):
        return {
            "gene": self.gene,
            "log2fc": self.log2fc,
            "padj": self.padj,
            "pvalue": self.pvalue,
            "direction": self.direction
        }

    def __repr__(self):
        return f"DEG({self.gene}, FC={self.log2fc:.2f}, padj={self.padj:.2e})"


class DEGAnalysis:
    """Container for DEG analysis results."""

    def __init__(self, disease_context: str = None, tissue: str = None):
        self.degs: List[DEGResult] = []
        self.disease_context = disease_context
        self.tissue = tissue

    def add_deg(self, deg: DEGResult):
        self.degs.append(deg)

    def filter_by_padj(self, threshold: float = 0.05) -> List[DEGResult]:
        """Filter DEGs by adjusted p-value threshold."""
        return [deg for deg in self.degs if deg.padj < threshold]

    def filter_by_lfc(self, threshold: float = 1.0) -> List[DEGResult]:
        """Filter DEGs by log2 fold change threshold."""
        return [deg for deg in self.degs if abs(deg.log2fc) > threshold]

    def get_top_genes(self, n: int = 10, by: str = "padj") -> List[DEGResult]:
        """Get top N genes sorted by specified metric."""
        if by == "padj":
            return sorted(self.degs, key=lambda x: x.padj)[:n]
        elif by == "log2fc":
            return sorted(self.degs, key=lambda x: abs(x.log2fc), reverse=True)[:n]
        elif by == "combined":
            # Simple ranking: combine statistical significance and fold change
            scored = [(deg, abs(deg.log2fc) * (-1 * (len(str(deg.padj))))) for deg in self.degs]
            return [deg for deg, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n]]
        return self.degs[:n]

    def from_csv(self, filepath: str) -> 'DEGAnalysis':
        """Load DEG results from CSV file."""
        df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            # Prefer Ensembl IDs when available because downstream context
            # retrieval (e.g. OpenTargets) currently works most reliably
            # with Ensembl identifiers. Fall back to other common columns.
            gene = (
                row.get('ENSEMBLID')
                or row.get('gene')
                or row.get('Gene')
                or row.get('gene_name')
                or row.get('SYMBOL')
            )
            log2fc = row.get('log2fc') or row.get('log2FoldChange') or row.get('logFC')
            padj = row.get('padj') or row.get('adj.P.Val') or row.get('adjusted_pvalue')
            pvalue = row.get('pvalue') or row.get('pval') or row.get('p_value')

            if gene and log2fc is not None and padj is not None:
                deg = DEGResult(
                    gene=str(gene),
                    log2fc=float(log2fc),
                    padj=float(padj),
                    pvalue=float(pvalue) if pvalue else None
                )
                self.add_deg(deg)

        return self

    def from_dataframe(self, df: pd.DataFrame, gene_col: str = None) -> 'DEGAnalysis':
        """Load DEG results from pandas DataFrame."""
        for _, row in df.iterrows():
            # Try to find gene column
            gene = None
            if gene_col and gene_col in row:
                gene = row[gene_col]
            else:
                # Fall back to common names
                for col in ['ENSEMBLID', 'gene', 'Gene', 'gene_name', 'SYMBOL']:
                    if col in row:
                        gene = row[col]
                        break

            # Try to find logfc
            log2fc = None
            for col in ['log2fc', 'log2FoldChange', 'logFC']:
                if col in row:
                    log2fc = row[col]
                    break

            # Try to find padj
            padj = None
            for col in ['padj', 'adj.P.Val', 'adjusted_pvalue']:
                if col in row:
                    padj = row[col]
                    break

            pvalue = row.get('pvalue') or row.get('pval') or row.get('p_value')

            if gene and log2fc is not None and padj is not None:
                deg = DEGResult(
                    gene=str(gene),
                    log2fc=float(log2fc),
                    padj=float(padj),
                    pvalue=float(pvalue) if pvalue else None
                )
                self.add_deg(deg)

        return self

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "disease_context": self.disease_context,
            "tissue": self.tissue,
            "num_genes": len(self.degs),
            "genes": [deg.to_dict() for deg in self.degs]
        }

    def summary(self) -> str:
        """Generate summary of DEG analysis."""
        upregulated = sum(1 for deg in self.degs if deg.direction == "upregulated")
        downregulated = sum(1 for deg in self.degs if deg.direction == "downregulated")

        summary = f"""
DEG Analysis Summary:
- Disease/Condition: {self.disease_context or 'Unknown'}
- Tissue: {self.tissue or 'Unknown'}
- Total significant genes: {len(self.degs)}
- Upregulated: {upregulated}
- Downregulated: {downregulated}
- Top genes by adjusted p-value:
"""
        for deg in self.get_top_genes(5, by="padj"):
            summary += f"  • {deg.gene}: log2FC={deg.log2fc:.2f}, padj={deg.padj:.2e}\n"

        return summary
