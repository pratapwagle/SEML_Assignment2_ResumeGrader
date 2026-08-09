# Group 179 — AI Resume Screening System

**Course:** AIMLCZG546 · Software Engineering for Machine Learning  
**Assignment:** II — Implementation, code quality, API design, and QA  
**Group:** 179

Production-style ML application for resume screening: modular training and
inference, FastAPI service, Streamlit recruiter UI, automated tests, and
quality metrics. This directory is the **submission package** — clone or
download as ZIP and run from this folder root.

---

## What to submit

| Deliverable | Path | Notes |
|-------------|------|--------|
| Source + tests | this repository root | Modular Python packages |
| Architecture | [`architecture.md`](architecture.md) | API spec, structure, diagrams |
| Report (PDF/Word) | [`report/179.pdf`](report/179.pdf), [`report/179.docx`](report/179.docx) | Taxila written submission |
| Notebook (required name) | [`179.ipynb`](179.ipynb) | Research vs production demo |
| Executed notebook | [`notebooks/179_executed.ipynb`](notebooks/179_executed.ipynb) | Outputs preserved |
| Report + QA evidence | [`report/`](report/) | DOCX/PDF, pytest, lint, metrics snapshot |

Upload to Taxila as directed by the course (typically `179.pdf` / `179.docx`).
Use **this folder** (or a git archive of it) as the source ZIP.

---

## Repository layout (summary)

```text
submission/
├── architecture.md      # System architecture & API contract
├── README.md            # This file
├── 179.ipynb            # Assignment-required notebook name
├── app/                 # REST API (FastAPI)
├── ml/                  # Data, ingestion, preprocess, train, infer
├── services/            # Domain orchestration, scoring, audit
├── quality/             # Model & data quality metrics
├── tests/               # pytest suite
├── models/              # Trained joblib artifact
├── notebooks/           # Executed notebook evidence
└── report/              # 179.docx/pdf + lint/pytest/metrics evidence
```

Full module map, sequence diagrams, and OpenAPI-style contracts:
**[architecture.md](architecture.md)**.

**Component diagram (Objective 1):** Mermaid source
[`docs/component_diagram.mmd`](docs/component_diagram.mmd), rendered PNG
[`docs/component_diagram.png`](docs/component_diagram.png) (also embedded in
`report/179.docx` §2 as Figure 1b).

---

## Production vs notebook

| Layer | Role |
|-------|------|
| `ml/`, `services/`, `app/`, `quality/` | **Production system** — run, test, deploy |
| `179.ipynb` | **Course notebook** — research prototype vs production `clean_text` |
| `notebooks/179_executed.ipynb` | Same notebook with saved execution outputs |

You do not need Jupyter to run the application.

---

## Quick start

```bash
# From this directory (submission/)
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -r requirements.txt
python train.py
pytest -q
python run_api.py
```

- API docs: http://127.0.0.1:8000/docs  
- Health: http://127.0.0.1:8000/health  
- Recruiter UI: `streamlit run streamlit_app.py`

### Example prediction

```bash
curl -X POST http://127.0.0.1:8000/v1/predictions \
  -H "Content-Type: application/json" \
  -d "{\"candidate_name\":\"Demo\",\"resume_text\":\"Python machine learning statistics NLP and model evaluation experience.\"}"
```

### Code quality gates

```bash
black --check .
isort --check-only .
flake8 .
```

---

## REST API (overview)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Readiness + version |
| `POST` | `/predict` | Inference alias |
| `POST` | `/v1/predictions` | Versioned inference |
| `GET` | `/metrics` | Training quality snapshot |
| `GET` | `/docs` | OpenAPI UI |

Full request/response schemas and status codes: **[architecture.md §5](architecture.md)**.

---

## Verified quality snapshot

Recorded under `report/` (regenerate with `python train.py` and `pytest`):

- Tests: 35 passed  
- Model accuracy / weighted F1 / top-3: 1.000 (small synthetic set; assignment evidence)  
- Multiclass Brier: ~0.725 (monitored)  
- Data schema valid; missing required values: 0  
- Black, isort, Flake8: pass (see `report/lint_*.txt`)

Metrics are **not** a production-performance claim; the dataset is deliberately small to demonstrate engineering workflow and gates.

---

## Group

| BITS ID | Name | Qualitative Task Description | Percentage |
|---------|------|----------------------------|------------|
| 2025AB05113 | Prashant | Implemented ML training & preprocessing pipelines, built FastAPI REST endpoints, authored automated Pytest suite, and drafted initial report sections. | 25% |
| 2025AA05032 | Prathap Wagle | Designed system architecture, integrated Streamlit UI & application layer, authored main report, added code comments, and validated submission readiness. | 25% |
| 2024AC05999 | Prasanna R T | Conducted test design review, performed functional validation of candidate scoring workflows, and verified edge-case handling. | 25% |
| 2024AC05914 | Pranav Mehrotra | Reviewed quality assurance evidence, conducted code audit reviews, and validated final submission package integrity. | 25% |

> Fill quantitative contribution percentages (sum to 100) in the notebook and report before final Taxila upload if still blank.

---

## Git / ZIP hygiene

```bash
# Create a clean archive of this package only
git archive -o Group179_Assignment_II.zip HEAD
# or zip this folder after excluding .venv and caches (see .gitignore)
```

Do **not** commit: `.venv/`, `__pycache__/`, `.pytest_cache/`, `logs/`, local audit CSV under `data/`.
