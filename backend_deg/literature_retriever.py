"""
Retrieves biological context and literature information for genes.
Uses OpenTargets API only (no mock data).
Free, no authentication required.
"""
from typing import Dict, List, Optional

import logging

from opentargets_retriever import OpenTargetsAPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneKnowledge:
    """Stores biological knowledge about a gene."""
    def __init__(self, gene: str):
        self.gene = gene
        self.known_functions = []
        self.disease_associations = []
        self.drug_targets = []
        self.pathway_associations = []
        self.protein_interactions = []

    def add_function(self, function: str, source: str = "literature"):
        self.known_functions.append({"description": function, "source": source})

    def add_disease(self, disease: str, evidence_strength: str = "moderate", pmids: List[str] = None):
        self.disease_associations.append({
            "disease": disease,
            "evidence_strength": evidence_strength,
            "pmids": pmids or []
        })

    def add_drug(self, drug: str, mechanism: str, status: str = "approved", source: str = "OpenTargets"):
        self.drug_targets.append({
            "drug": drug,
            "mechanism": mechanism,
            "status": status,
            "source": source
        })

    def add_pathway(self, pathway: str, database: str = "KEGG"):
        self.pathway_associations.append({"pathway": pathway, "database": database})

    def to_dict(self):
        return {
            "gene": self.gene,
            "known_functions": self.known_functions,
            "disease_associations": self.disease_associations,
            "drug_targets": self.drug_targets,
            "pathway_associations": self.pathway_associations,
            "protein_interactions": self.protein_interactions
        }


class LiteratureRetriever:
    """Retrieves biological context for genes from OpenTargets API only."""

    def __init__(self, use_cache: bool = True):
        """Initialize literature retriever with OpenTargets API."""
        self.opentargets = OpenTargetsAPI(use_cache=use_cache)
        logger.info("LiteratureRetriever initialized (OpenTargets only, no mock data)")

    def _get_from_opentargets(self, gene_id: str) -> Optional[GeneKnowledge]:
        """
        Get gene knowledge from OpenTargets API.
        gene_id can be Ensembl ID (e.g. ENSG00000141510) or symbol (e.g. TP53).
        """
        try:
            diseases_data = self.opentargets.get_diseases_for_gene(gene_id)
            if not diseases_data:
                return None

            knowledge = GeneKnowledge(gene_id)
            diseases = self.opentargets.parse_disease_response(diseases_data)
            for disease_name, evidence_strength, _score in diseases:
                knowledge.add_disease(disease_name, evidence_strength)

            drugs_data = self.opentargets.get_drugs_for_gene(gene_id)
            if drugs_data:
                drugs = self.opentargets.parse_drug_response(drugs_data)
                for drug_info in drugs:
                    knowledge.add_drug(
                        drug_info["drug"],
                        drug_info["mechanism"],
                        drug_info["status"],
                        "OpenTargets"
                    )

            if knowledge.disease_associations or knowledge.drug_targets:
                knowledge.add_function(
                    "Disease and drug associations from OpenTargets",
                    source="OpenTargets"
                )
            return knowledge

        except Exception as e:
            logger.warning(f"Error retrieving from OpenTargets for {gene_id}: {e}")
            return None

    def get_gene_knowledge(self, gene: str) -> Optional[GeneKnowledge]:
        """
        Retrieve knowledge for a specific gene.
        gene can be Ensembl ID (ENSG...) or HGNC symbol (e.g. TP53).
        """
        if not gene or not str(gene).strip():
            return None
        gene_id = str(gene).strip()
        knowledge = self._get_from_opentargets(gene_id)
        if knowledge:
            logger.info(f"Retrieved {gene_id} from OpenTargets")
        else:
            logger.debug(f"No OpenTargets data for {gene_id}")
        return knowledge

    def batch_retrieve(self, genes: List[str]) -> Dict[str, GeneKnowledge]:
        """Retrieve knowledge for multiple genes from OpenTargets."""
        results = {}
        for gene in genes:
            knowledge = self.get_gene_knowledge(gene)
            if knowledge:
                results[gene] = knowledge
        return results

    def generate_context_for_llm(self, genes: List[str]) -> str:
        """Generate formatted context string for LLM input from OpenTargets data."""
        context = "BIOLOGICAL CONTEXT:\n\n"
        knowledge_base = self.batch_retrieve(genes)

        for gene in genes:
            if gene in knowledge_base:
                kb = knowledge_base[gene]
                context += f"Gene: {gene}\n"
                if kb.known_functions:
                    context += f"  Functions: {'; '.join([f['description'] for f in kb.known_functions])}\n"
                if kb.disease_associations:
                    context += "  Disease associations:\n"
                    for disease in kb.disease_associations:
                        context += f"    - {disease['disease']} ({disease['evidence_strength']} evidence)\n"
                if kb.drug_targets:
                    context += "  Drug targets:\n"
                    for drug in kb.drug_targets:
                        context += f"    - {drug['drug']} ({drug['mechanism']}, {drug['status']})\n"
                if kb.pathway_associations:
                    context += f"  Pathways: {'; '.join([p['pathway'] for p in kb.pathway_associations])}\n"
                context += "\n"
            else:
                context += f"Gene: {gene}\n  (No data from OpenTargets for this identifier)\n\n"

        return context
