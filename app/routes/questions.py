from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.retrieval import find_candidate_patterns, pattern_to_dict
from app.llm import analyze_problem
from app.schemas import ProblemSubmission, PatternAnalysis
from app.models import Attempt
import json

router = APIRouter(prefix="/api", tags=["questions"])

def _attach_urls(result: dict, matched_pattern) -> dict:
    """The LLM returns plain title strings for similar_problems - we never
    let it invent URLs. Instead, look up each title's real URL from the
    matched Pattern's own example_problems, so every link is guaranteed to
    be one we actually stored, not something the LLM guessed.

    Guarded against missing URLs at every step: a pattern with no examples,
    an example row where url is still null (not yet re-ingested), or a
    title the LLM returned that doesn't exactly match anything on file -
    all of these should degrade to a plain title with no link, never a crash.
    """
    titles = result.get("similar_problems") or []

    if not matched_pattern:
        result["similar_problems"] = [{"title": t, "url": None} for t in titles]
        return result

    url_by_title = {
        ep.title: getattr(ep, "url", None)
        for ep in (matched_pattern.example_problems or [])
    }
    result["similar_problems"] = [
        {"title": t, "url": url_by_title.get(t) or None}
        for t in titles
    ]
    return result

@router.post("/analyze", response_model=PatternAnalysis)
def analyze(submission: ProblemSubmission, db: Session = Depends(get_db)):
    if not submission.problem_statement.strip():
        raise HTTPException(status_code=400, detail="Problem statement cannot be empty.")

    candidates = find_candidate_patterns(db, submission.problem_statement)
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No patterns in the database yet - run `python -m app.ingest` first.",
        )

    candidate_dicts = [pattern_to_dict(pattern, distance) for pattern, distance in candidates]
    result = analyze_problem(submission.problem_statement, candidate_dicts)

    matched_pattern_obj = next(
        (p for p, _ in candidates if p.name == result.get("pattern_name")),
        None,
    ) if result.get("matched") else None
    result = _attach_urls(result, matched_pattern_obj)

    # Log the attempt - this is what later phases (mastery tracking, spaced
    # repetition) will aggregate over, so we capture it from day one.
    matched_pattern_id = next(
        (pattern.id for pattern, _ in candidates if pattern.name == result.get("pattern_name")),
        None,
    )
    attempt = Attempt(
        problem_statement=submission.problem_statement,
        matched_pattern_id=matched_pattern_id,
        llm_response=json.dumps(result),
    )
    db.add(attempt)
    db.commit()

    return PatternAnalysis(**result)
