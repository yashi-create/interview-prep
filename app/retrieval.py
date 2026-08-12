from sqlalchemy.orm import Session
from sqlalchemy import select

from app.models import Pattern
from app.embeddings import embed_text


MIN_K = 3
MAX_K = 8


def find_candidate_patterns(
    db: Session, problem_statement: str, k: int | None = None
) -> list[tuple[Pattern, float]]:
    """Embed the incoming problem statement and return the top-k closest
    patterns by cosine distance, each paired with its distance score.

    Deliberately does NOT collapse this to a single "best" match - a forced
    top-1 pick silently returns a bad match when nothing in the library is
    actually close (e.g. a genuinely novel problem). Returning candidates
    with distances lets the caller (or the LLM) judge fit quality instead of
    trusting whichever pattern happens to be nearest in vector space.

    k defaults to scaling with the size of the library - a fixed k=3 was fine
    at 5 patterns but starts silently excluding the correct pattern from the
    candidate set once the library grows past a handful of entries. Capped at
    MAX_K so the LLM prompt doesn't balloon as the library gets large.
    """
    if k is None:
        total_patterns = db.query(Pattern).count()
        k = min(MAX_K, max(MIN_K, total_patterns // 3))

    query_vector = embed_text(problem_statement)

    stmt = (
        select(Pattern, Pattern.embedding.cosine_distance(query_vector).label("distance"))
        .order_by("distance")
        .limit(k)
    )
    results = db.execute(stmt).all()
    return [(row[0], row[1]) for row in results]


def pattern_to_dict(pattern: Pattern, distance: float | None = None) -> dict:
    """Serialize a Pattern (with its example problems) into the plain dict
    shape the LLM prompt expects - keeps the LLM layer decoupled from the ORM.
    Includes the retrieval distance when available, so the LLM can see how
    close the vector match actually was rather than treating every candidate
    as equally strong.
    """
    return {
        "name": pattern.name,
        "retrieval_distance": round(distance, 4) if distance is not None else None,
        "description": pattern.description,
        "signal_phrases": pattern.signal_phrases,
        "core_technique": pattern.core_technique,
        "common_pitfalls": pattern.common_pitfalls,
        "additional_considerations": pattern.additional_considerations,
        "example_problems": [
            {"title": ep.title, "why_it_fits": ep.why_it_fits}
            for ep in pattern.example_problems
        ],
    }