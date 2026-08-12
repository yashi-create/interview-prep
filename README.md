# Interview Prep Copilot — Phase 1: DSA Pattern Recognition

A RAG-powered tool that takes a DSA problem statement and identifies the
underlying pattern it belongs to (Two Pointers, Sliding Window, Binary Search
on Answer, BFS/DFS, DP), explains *why*, shows other problems that share the
same pattern, and flags the edge cases / optimizations that tend to trip
people up — grounded in a curated reference library, not free-form LLM
generation.

**Why grounded, not just "ask the LLM":** every claim in the response is
required to trace back to a specific retrieved reference document (pattern
description, common pitfalls, example problems). This is the same
retrieval-grounding approach production RAG systems use to keep hallucination
rate down — the model explains a pattern using material we actually stored
and retrieved, not just what it "remembers."

## Architecture

```
Problem statement
      │
      ▼
 embed (sentence-transformers, local, free)
      │
      ▼
 pgvector similarity search  →  best-matching Pattern + its example problems
      │
      ▼
 LLM (Groq, free tier) generates a grounded explanation
 using ONLY the retrieved pattern's reference material
      │
      ▼
 Structured JSON response → rendered in the UI
```

Every attempt is logged (`attempts` table) — this is deliberate groundwork
for Phase 2 (weak-area tracking + spaced repetition), so nothing needs to be
retrofitted later.

## Stack (entirely free-tier)

| Piece | Choice | Why |
|---|---|---|
| Backend | FastAPI | Typed, async-ready, close mental model to Spring Boot |
| DB + vector store | Supabase Postgres (pgvector) | Free tier, pgvector built in |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs locally on CPU, zero API cost |
| LLM | Groq API (Llama 3.3 70B) | Free tier, fast inference |
| Frontend | Jinja2 + HTMX | No JS build step, one deployable service |
| Hosting | Render (free web service) | Deploys straight from GitHub |

## Setup

### 1. Create a free Supabase project
- [supabase.com](https://supabase.com) → New project
- Go to **Project Settings → Database → Connection string** (use the
  "URI" / pooler connection string)
- pgvector is available by default — the ingestion script enables the
  extension automatically (`CREATE EXTENSION IF NOT EXISTS vector`)

### 2. Get a free Groq API key
- [console.groq.com/keys](https://console.groq.com/keys) → create a key

### 3. Local setup
```bash
git clone <your-repo-url>
cd interview-prep
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: paste your DATABASE_URL and GROQ_API_KEY
```

### 4. Seed the pattern library
```bash
python -m app.ingest
```
This creates the tables, enables pgvector, and embeds+stores the seed
patterns from `data/patterns/*.json` (one file per pattern). Re-run any time
you add or edit a pattern file - it upserts by name, so edits to an existing
pattern's wording take effect on re-ingestion instead of being skipped.

### 5. Run locally
```bash
uvicorn app.main:app --reload
```
Visit `http://localhost:8000`.

## Deploying for free

1. Push this repo to GitHub.
2. [render.com](https://render.com) → New → Web Service → connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add `DATABASE_URL` and `GROQ_API_KEY` as environment variables in Render's
   dashboard (don't commit `.env`).
6. Deploy. Free tier spins down after inactivity — first request after idle
   takes ~30s to wake up. Fine for a portfolio demo; worth mentioning as a
   known trade-off if asked in an interview.

## Extending the pattern library

Add a new file to `data/patterns/` (e.g. `data/patterns/two-heaps.json`)
following the existing shape, then re-run `python -m app.ingest`. No code
changes needed — the embedding and retrieval logic is entirely data-driven.
One file per pattern keeps each addition/edit a single-file diff to review
instead of a shared-array merge conflict. Retrieval's candidate count (`k`)
scales automatically with how many pattern files exist, up to a cap, so
growing the library doesn't quietly shrink recall.

## Roadmap (future phases)

- **Phase 2:** System design primers (caching, sharding, load balancing) as
  a second retrieval domain, topic-scoped.
- **Phase 3:** Mastery tracking — aggregate `attempts` by pattern, surface a
  weak-topic dashboard.
- **Phase 4:** Spaced repetition scheduling (APScheduler) — resurface
  patterns you struggled with after N days instead of randomly.
- **Phase 5:** Adaptive difficulty — escalate/de-escalate question difficulty
  based on tracked performance.
