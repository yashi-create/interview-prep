from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from app.database import get_db
from app.retrieval import find_candidate_patterns, pattern_to_dict
from app.llm import analyze_problem
from app.models import Attempt
from app.routes import questions
import json

app = FastAPI(title="Interview Prep Copilot - DSA Pattern Recognition")

# /api/analyze is meant to be called from other platforms (a mobile app, a
# separate frontend) - there's no cookie/session auth here for a permissive
# origin policy to put at risk, so a wide-open policy is the simple choice.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(questions.router)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
def home(request: Request):
    # Render's health check pings "/" with HEAD, not GET - a GET-only route
    # 405s that check, so Render never marks the deploy healthy.
    return templates.TemplateResponse("home.html", {"request": request, "active": "home"})


@app.get("/analyze", response_class=HTMLResponse)
def analyze_page(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "active": "analyze"})


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
