"""
Seed/refresh the patterns table from data/patterns/*.json (one file per
pattern, so adding or editing a pattern is a single-file diff instead of a
merge-conflict-prone shared array).

Usage:
    python -m app.ingest
"""
import json
from pathlib import Path

from sqlalchemy import text

from app.database import Base, engine, SessionLocal
from app.models import Pattern, ExampleProblem
from app.embeddings import embed_text

DATA_DIR = Path(__file__).parent.parent / "data" / "patterns"


def ensure_pgvector_extension():
    """pgvector must be enabled on the Postgres instance before the vector
    column can be created. Supabase supports this via a plain SQL command."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()


def build_embedding_text(pattern: dict) -> str:
    """What we embed matters: description + signal phrases gives the vector
    a strong sense of 'what kinds of problems trigger this pattern', which is
    exactly what we're matching new problem statements against."""
    signals = ", ".join(pattern["signal_phrases"])
    return f"{pattern['name']}. {pattern['description']} Common signals: {signals}."


def load_patterns_data() -> list[dict]:
    """Each pattern lives in its own file so adding/editing one pattern is a
    single-file change to review, not an edit to one large shared array."""
    return [json.loads(path.read_text()) for path in sorted(DATA_DIR.glob("*.json"))]


def run():
    ensure_pgvector_extension()
    Base.metadata.create_all(bind=engine)

    patterns_data = load_patterns_data()

    db = SessionLocal()
    try:
        for p in patterns_data:
            embedding_text = build_embedding_text(p)
            vector = embed_text(embedding_text)

            existing = db.query(Pattern).filter_by(name=p["name"]).first()
            if existing:
                existing.description = p["description"]
                existing.signal_phrases = p["signal_phrases"]
                existing.core_technique = p["core_technique"]
                existing.common_pitfalls = p["common_pitfalls"]
                existing.additional_considerations = p["additional_considerations"]
                existing.embedding = vector
                # replace wholesale rather than diffing - cascade="all, delete-orphan"
                # on the relationship handles dropping the old rows.
                existing.example_problems = [
                    ExampleProblem(title=ep["title"], why_it_fits=ep["why_it_fits"])
                    for ep in p["example_problems"]
                ]
                print(f"Updated '{p['name']}'.")
                continue

            pattern = Pattern(
                name=p["name"],
                description=p["description"],
                signal_phrases=p["signal_phrases"],
                core_technique=p["core_technique"],
                common_pitfalls=p["common_pitfalls"],
                additional_considerations=p["additional_considerations"],
                embedding=vector,
            )
            db.add(pattern)
            db.flush()  # get pattern.id before adding children

            for ep in p["example_problems"]:
                db.add(
                    ExampleProblem(
                        pattern_id=pattern.id,
                        title=ep["title"],
                        why_it_fits=ep["why_it_fits"],
                    )
                )

            print(f"Seeded '{p['name']}' with {len(p['example_problems'])} example problems.")

        db.commit()
        print("Ingestion complete.")
    finally:
        db.close()


if __name__ == "__main__":
    run()
