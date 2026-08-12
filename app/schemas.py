from pydantic import BaseModel


class ProblemSubmission(BaseModel):
    problem_statement: str


class PatternAnalysis(BaseModel):
    pattern_name: str
    why_this_pattern: str
    core_technique: str
    similar_problems: list[str]
    additional_considerations: str
    confidence_note: str
