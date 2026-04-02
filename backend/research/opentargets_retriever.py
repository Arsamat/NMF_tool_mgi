"""
OpenTargets API integration for disease-gene associations.
Provides real biomedical data from OpenTargets platform.
Free, no authentication required.

Documentation: https://www.opentargets.org
API Docs: https://docs.targetvalidation.org/
"""

import requests
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenTargetsAPI:
    """Client for OpenTargets GraphQL API - real disease-gene associations."""

    # Updated to current OpenTargets v4 GraphQL endpoint
    API_URL = "https://api.platform.opentargets.org/api/v4/graphql"
    CACHE_FILE = Path("opentargets_cache.json")

    def __init__(self, use_cache: bool = True, rate_limit: float = 0.3):
        """
        Initialize OpenTargets API client.

        Args:
            use_cache: Whether to cache API responses
            rate_limit: Delay between requests (seconds)

        Note: OpenTargets is completely free and requires no authentication.
        """
        self.use_cache = use_cache
        self.rate_limit = rate_limit
        self.cache = self._load_cache() if use_cache else {}
        self.last_request_time = 0
        logger.info("OpenTargets API initialized (free, no auth required)")

    def _load_cache(self) -> Dict:
        """Load cached API responses."""
        if self.CACHE_FILE.exists():
            try:
                with open(self.CACHE_FILE, "r") as f:
                    logger.info("Loaded OpenTargets cache from file")
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}

    def _save_cache(self):
        """Save cache to file."""
        if self.use_cache:
            try:
                with open(self.CACHE_FILE, "w") as f:
                    json.dump(self.cache, f, indent=2)
                    logger.debug("Saved OpenTargets cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def _respect_rate_limit(self):
        """Respect API rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _build_disease_query(self, ensembl_id: str, max_results: int = 100) -> str:
        """
        Build GraphQL query for disease associations.

        Note: modern OpenTargets GraphQL returns results under
        `associatedDiseases { count rows { ... } }`.
        """
        return f"""
        query {{
            target(ensemblId: "{ensembl_id}") {{
                id
                approvedSymbol
                approvedName
                associatedDiseases {{
                    count
                    rows {{
                        disease {{
                            id
                            name
                            description
                        }}
                        score
                    }}
                }}
            }}
        }}
        """

    def _build_drug_query(self, ensembl_id: str, max_results: int = 50) -> str:
        """
        Build GraphQL query for drug targets.

        Modern schema exposes `knownDrugs(cursor, size, freeTextQuery)`.
        We follow the official example query shape.
        """
        return f"""
        query KnownDrugsQuery {{
            target(ensemblId: "{ensembl_id}") {{
                knownDrugs {{
                    count
                    rows {{
                        phase
                        status
                        urls {{
                            name
                            url
                        }}
                        disease {{
                            id
                            name
                        }}
                        drug {{
                            id
                            name
                            mechanismsOfAction {{
                                rows {{
                                    actionType
                                    targets {{
                                        id
                                    }}
                                }}
                            }}
                        }}
                        drugType
                        phase
                        mechanismOfAction
                    }}
                }}
            }}
        }}
        """

    def _get_ensembl_id(self, gene_symbol: str) -> Optional[str]:
        """
        Resolve a gene symbol like 'TP53' to an Ensembl ID using
        OpenTargets search. Caches results to minimize round-trips.
        """
        # If the caller already passed an Ensembl ID, just use it
        if gene_symbol.upper().startswith("ENSG"):
            return gene_symbol

        cache_key = f"ensembl:{gene_symbol}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        query = f"""
        query {{
            search(queryString: "{gene_symbol}", entityNames: [TARGET]) {{
                total
                hits {{
                    id
                    name
                }}
            }}
        }}
        """

        try:
            self._respect_rate_limit()
            response = requests.post(
                self.API_URL,
                json={"query": query},
                timeout=10,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                logger.warning(
                    f"OpenTargets search API returned {response.status_code} "
                    f"for symbol {gene_symbol}"
                )
                return None

            data = response.json()
            if "errors" in data:
                logger.warning(f"GraphQL search error for {gene_symbol}: {data['errors']}")
                return None

            search_data = data.get("data", {}).get("search")
            if not search_data:
                return None

            hits = search_data.get("hits") or []
            if not hits:
                logger.warning(f"No Ensembl ID found for gene symbol {gene_symbol}")
                return None

            ensembl_id = hits[0].get("id")
            if not ensembl_id:
                logger.warning(f"Search hit for {gene_symbol} missing id")
                return None

            if self.use_cache:
                self.cache[cache_key] = ensembl_id
                self._save_cache()

            return ensembl_id

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to resolve Ensembl ID for {gene_symbol}: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error processing Ensembl ID lookup for {gene_symbol}: {e}")
            return None

    def get_diseases_for_gene(self, gene_symbol: str, max_results: int = 100) -> List[Dict]:
        """
        Get diseases associated with a gene from OpenTargets.

        Args:
            gene_symbol: Gene symbol or Ensembl ID (e.g. "TP53" or "ENSG00000141510")
            max_results: Maximum number of diseases to retrieve

        Returns:
            List of disease associations with scores
        """
        cache_key = f"diseases:{gene_symbol}"

        # Check cache first
        if cache_key in self.cache:
            logger.debug(f"Using cached data for {gene_symbol}")
            return self.cache[cache_key]

        try:
            # Resolve gene symbol to Ensembl ID expected by OpenTargets
            ensembl_id = self._get_ensembl_id(gene_symbol)
            if not ensembl_id:
                logger.warning(f"Could not resolve Ensembl ID for {gene_symbol}")
                return []

            self._respect_rate_limit()

            # Build and execute GraphQL query
            query = self._build_disease_query(ensembl_id, max_results)
            response = requests.post(
                self.API_URL,
                json={"query": query},
                timeout=10,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()

                # Check for errors in GraphQL response
                if "errors" in data:
                    logger.warning(f"GraphQL error for {gene_symbol}: {data['errors']}")
                    return []

                # Extract diseases from response
                if "data" in data and "target" in data["data"]:
                    target = data["data"]["target"]
                    if target and "associatedDiseases" in target:
                        assoc = target["associatedDiseases"] or {}
                        rows = assoc.get("rows") or []
                        logger.info(
                            f"Retrieved {len(rows)} diseases for {gene_symbol} from OpenTargets"
                        )

                        # Cache the result
                        if self.use_cache:
                            self.cache[cache_key] = rows
                            self._save_cache()

                        return rows

                logger.warning(f"No data returned for {gene_symbol} from OpenTargets")
                return []

            else:
                logger.warning(f"OpenTargets API returned {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query OpenTargets for {gene_symbol}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error processing OpenTargets response for {gene_symbol}: {e}")
            return []

    def get_drugs_for_gene(self, gene_symbol: str, max_results: int = 50) -> List[Dict]:
        """
        Get drugs targeting a gene from OpenTargets.

        Args:
            gene_symbol: Gene symbol or Ensembl ID
            max_results: Maximum number of drugs to retrieve

        Returns:
            List of drug targets with mechanisms and clinical phases
        """
        cache_key = f"drugs:{gene_symbol}"

        if cache_key in self.cache:
            logger.debug(f"Using cached drug data for {gene_symbol}")
            return self.cache[cache_key]

        try:
            # Resolve gene symbol to Ensembl ID expected by OpenTargets
            ensembl_id = self._get_ensembl_id(gene_symbol)
            if not ensembl_id:
                logger.warning(f"Could not resolve Ensembl ID for {gene_symbol}")
                return []

            self._respect_rate_limit()

            query = self._build_drug_query(ensembl_id, max_results)
            response = requests.post(
                self.API_URL,
                json={"query": query},
                timeout=10,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()

                if "errors" in data:
                    logger.warning(f"GraphQL error for {gene_symbol}: {data['errors']}")
                    return []

                if "data" in data and "target" in data["data"]:
                    target = data["data"]["target"]
                    if target and "knownDrugs" in target:
                        kd = target["knownDrugs"] or {}
                        rows = kd.get("rows") or []
                        logger.info(
                            f"Retrieved {len(rows)} drugs for {gene_symbol} from OpenTargets"
                        )

                        if self.use_cache:
                            self.cache[cache_key] = rows
                            self._save_cache()

                        return rows

                logger.debug(f"No drugs found for {gene_symbol}")
                return []

            else:
                logger.warning(f"OpenTargets API returned {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to query OpenTargets drugs for {gene_symbol}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error processing OpenTargets drug response for {gene_symbol}: {e}")
            return []

    def parse_disease_response(
        self, diseases_data: List[Dict]
    ) -> List[Tuple[str, str, float]]:
        """
        Parse OpenTargets disease response into standard format.

        Args:
            diseases_data: Raw disease list from API

        Returns:
            List of (disease_name, evidence_strength, score) tuples
        """
        diseases = []

        for disease in diseases_data:
            try:
                disease_name = disease.get("disease", {}).get("name", "Unknown")
                score = float(disease.get("score", 0))

                # Convert score to evidence strength
                if score >= 0.7:
                    evidence_strength = "very strong"
                elif score >= 0.5:
                    evidence_strength = "strong"
                elif score >= 0.3:
                    evidence_strength = "moderate"
                elif score >= 0.1:
                    evidence_strength = "weak"
                else:
                    evidence_strength = "minimal"

                diseases.append((disease_name, evidence_strength, score))

            except (KeyError, TypeError, ValueError) as e:
                logger.debug(f"Error parsing disease entry: {e}")
                continue

        return diseases

    def parse_drug_response(self, drugs_data: List[Dict]) -> List[Dict]:
        """
        Parse OpenTargets drug response into standard format.

        Args:
            drugs_data: Raw drug list from API

        Returns:
            List of dicts with drug information
        """
        drugs = []

        for drug in drugs_data:
            try:
                drug_info = drug.get("drug") or {}
                name = (
                    drug.get("drugName")
                    or drug.get("prefName")
                    or drug.get("label")
                    or drug_info.get("name")
                    or "Unknown"
                )

                mechanism = drug.get("mechanismOfAction")
                if not mechanism:
                    moa = (drug_info.get("mechanismsOfAction") or {}).get("rows") or []
                    if moa:
                        mechanism = moa[0].get("actionType") or "Unknown"
                if not mechanism:
                    mechanism = "Unknown"

                phase = drug.get("status") or drug.get("phase") or drug.get("clinicalPhase") or "unknown"
                if isinstance(phase, (int, float)):
                    phase_str = f"phase {int(phase)}"
                else:
                    phase_str = str(phase)

                disease_info = drug.get("disease")
                if isinstance(disease_info, dict):
                    indication = disease_info.get("name", "Unknown")
                else:
                    indication = "Unknown"

                drug_dict = {
                    "drug": name,
                    "mechanism": mechanism,
                    "status": phase_str.lower(),
                    "indication": indication,
                }
                drugs.append(drug_dict)

            except (KeyError, TypeError) as e:
                logger.debug(f"Error parsing drug entry: {e}")
                continue

        return drugs

    def clear_cache(self):
        """Clear the cache file."""
        if self.CACHE_FILE.exists():
            self.CACHE_FILE.unlink()
            self.cache = {}
            logger.info("Cleared OpenTargets cache")
