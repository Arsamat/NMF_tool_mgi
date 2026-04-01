from typing import List

from fastapi.responses import JSONResponse

from hypothesis_validator_claude import HypothesisValidatorClaude


def validate_hypotheses_claude_service(
    hypotheses: List[str],
    max_papers: int,
) -> JSONResponse:
    """
    Shared service logic for the /validate_hypotheses_claude/ endpoint.

    Runs the Claude-based hypothesis validator and returns a JSONResponse
    with results and a novelty summary.
    """
    if not hypotheses:
        return JSONResponse(
            status_code=200,
            content={"error": "No hypotheses provided", "results": []},
        )

    if len(hypotheses) > 10:
        return JSONResponse(
            status_code=200,
            content={"error": "Maximum 10 hypotheses per request", "results": []},
        )

    validator = HypothesisValidatorClaude()
    results = validator.validate_multiple(
        hypotheses,
        max_papers_per_search=max_papers,
    )

    summary = {
        "high_novelty": sum(
            1 for r in results if "HIGH" in r.get("novelty_assessment", "")
        ),
        "moderate_novelty": sum(
            1 for r in results if "MODERATE" in r.get("novelty_assessment", "")
        ),
        "low_novelty": sum(
            1 for r in results if "LOW" in r.get("novelty_assessment", "")
        ),
    }

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "total_hypotheses": len(results),
            "results": results,
            "summary": summary,
        },
    )

