import os
import json
from groq import Groq

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are an interview-prep assistant that identifies the underlying \
pattern behind a DSA problem. You are given several CANDIDATE PATTERNS retrieved by \
vector similarity search, each with a distance score (lower = more similar). \
Vector similarity is a rough first pass, not ground truth - it can retrieve a \
pattern that merely SOUNDS similar in wording while being the wrong fit conceptually. \
Your job is to actually reason about which candidate (if any) correctly explains the \
problem's solution approach, not just default to whichever has the lowest distance.

Rules:
- You must ground every claim in the reference material of whichever pattern you pick. \
Do not invent example problems, pitfalls, or techniques not present in that pattern's \
reference material.
- If NONE of the candidates genuinely fit - the retrieved patterns are all a poor \
conceptual match for how this problem is actually solved - say so explicitly rather \
than forcing the closest one. It is better to report "no confident match" than to give \
a confidently wrong answer.
- Judge fit by whether the candidate's core_technique and signal_phrases actually \
describe how you'd solve this specific problem - not by surface wording overlap.

Respond ONLY with valid JSON matching this schema, no markdown fences, no preamble:
{
  "matched": boolean (true if one candidate genuinely fits, false if none do),
  "pattern_name": string (the matched pattern's name, or "" if matched is false),
  "why_this_pattern": string (2-3 sentences explaining what in the problem signals this \
pattern - or, if matched is false, 2-3 sentences explaining why none of the candidates fit \
and what the problem actually requires instead),
  "core_technique": string (the reusable technique, grounded in the reference - empty if matched is false),
  "similar_problems": [string] (problem titles from the matched pattern's reference material - empty if matched is false),
  "additional_considerations": string (edge cases / optimizations / pitfalls specific to this problem, \
grounded in the reference material - empty if matched is false),
  "confidence_note": string (one honest sentence on match quality, e.g. "Strong match - the \
sorted-array + pair-sum signal is explicit." or "No candidate fits well - this problem needs a \
pattern not yet in the reference library.")
}"""


def analyze_problem(problem_statement: str, candidates: list[dict]) -> dict:
    """Given a problem statement and several candidate patterns (retrieved via
    vector similarity, each with a distance score), ask the LLM to reason about
    which one actually fits - or report that none do - rather than blindly
    explaining whatever came back as the nearest vector match.
    """
    candidates_text = json.dumps(candidates, indent=2)

    user_prompt = f"""PROBLEM STATEMENT:
{problem_statement}

CANDIDATE PATTERNS (retrieved by vector similarity, ordered nearest-first - \
lower distance means more similar wording, NOT necessarily correct fit):
{candidates_text}

Decide which candidate, if any, correctly explains how this problem is solved."""

    client = _get_client()
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,  # low temperature - we want grounded, consistent output, not creativity
    )

    raw = completion.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Model occasionally wraps in markdown fences despite instructions - strip and retry once
        cleaned = raw.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)