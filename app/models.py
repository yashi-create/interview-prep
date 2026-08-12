from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.database import Base

# all-MiniLM-L6-v2 produces 384-dimensional embeddings
EMBEDDING_DIM = 384


class Pattern(Base):
    """A DSA pattern, e.g. 'Two Pointers' or 'Sliding Window'.

    The `embedding` column holds a vector representation of the pattern's
    description + signal phrases, so a new problem statement can be matched
    to the pattern it belongs to via similarity search.
    """
    __tablename__ = "patterns"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text, nullable=False)
    signal_phrases = Column(JSON, nullable=False)   # list[str]
    core_technique = Column(Text, nullable=False)
    common_pitfalls = Column(Text, nullable=False)
    additional_considerations = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)

    example_problems = relationship(
        "ExampleProblem", back_populates="pattern", cascade="all, delete-orphan"
    )


class ExampleProblem(Base):
    """A concrete problem illustrating a pattern, with a note on *why* it fits."""
    __tablename__ = "example_problems"

    id = Column(Integer, primary_key=True)
    pattern_id = Column(Integer, ForeignKey("patterns.id"), nullable=False)
    title = Column(String, nullable=False)
    why_it_fits = Column(Text, nullable=False)

    pattern = relationship("Pattern", back_populates="example_problems")


class Attempt(Base):
    """A user's submission of a problem statement for pattern analysis.

    Kept from day one (even in the DSA-only phase) so later phases can build
    weak-area tracking and spaced repetition on top of real history instead
    of retrofitting it.
    """
    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True)
    problem_statement = Column(Text, nullable=False)
    matched_pattern_id = Column(Integer, ForeignKey("patterns.id"), nullable=True)
    llm_response = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    matched_pattern = relationship("Pattern")
