from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.database import get_db
from app.retrieval import find_candidate_patterns, pattern_to_dict
from app.llm import analyze_problem
from app.models import Attempt
from app.routes import questions
import json

app = FastAPI(title="Interview Prep Copilot - DSA Pattern Recognition")

app.include_router(questions.router)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/analyze-form", response_class=HTMLResponse)
def analyze_form(
    request: Request,
    problem_statement: str = Form(...),
    db: Session = Depends(get_db),
):
    """HTMX-driven endpoint: returns an HTML fragment, not JSON, so the
    frontend can stay a plain server-rendered page with no JS build step."""
    candidates = find_candidate_patterns(db, problem_statement)
    if not candidates:
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "error": "No patterns loaded yet - run the ingestion script first."},
        )

    candidate_dicts = [pattern_to_dict(pattern, distance) for pattern, distance in candidates]
    result = analyze_problem(problem_statement, candidate_dicts)

    matched_pattern_id = next(
        (pattern.id for pattern, _ in candidates if pattern.name == result.get("pattern_name")),
        None,
    )
    attempt = Attempt(
        problem_statement=problem_statement,
        matched_pattern_id=matched_pattern_id,
        llm_response=json.dumps(result),
    )
    db.add(attempt)
    db.commit()

    return templates.TemplateResponse("result.html", {"request": request, "result": result, "error": None})
