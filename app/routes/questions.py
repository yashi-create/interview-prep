from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.retrieval import find_candidate_patterns, pattern_to_dict
from app.llm import analyze_problem
from app.schemas import ProblemSubmission, PatternAnalysis
from app.models import Attempt
import json

router = APIRouter(prefix="/api", tags=["questions"])


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
