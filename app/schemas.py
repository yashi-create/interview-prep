from pydantic import BaseModel


class ProblemSubmission(BaseModel):
    problem_statement: str

class SimilarProblem(BaseModel):
    title: str
    url: str | None = None    

class PatternAnalysis(BaseModel):
    pattern_name: str
    why_this_pattern: str
    core_technique: str
    similar_problems: list[SimilarProblem]
    additional_considerations: str
    confidence_note: str
