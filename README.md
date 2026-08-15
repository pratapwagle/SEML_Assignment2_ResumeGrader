# Group 179 - AI Resume Screening System

**Course:** AIMLCZG546 · Software Engineering for Machine Learning 
**Assignment:** II - Implementation, code quality, API design, and QA 
**Group:** 179

Production-style ML application for resume screening: modular training and
inference, FastAPI service, Streamlit recruiter UI, automated tests, and
quality metrics. This directory is the **179 package** — clone or
download as ZIP and run from this folder root.

---

## Deliverables

| Deliverable | Path | Notes |
|-------------|------|--------|
| Source + tests | this repository root | Modular Python packages |
| Architecture | [`architecture.md`](architecture.md) | API spec, structure, diagrams |
| Report (PDF/Word) | [`report/179.pdf`](report/179.pdf), [`report/179.docx`](report/179.docx) | Taxila written report |
| Notebook (required name) | [`179.ipynb`](179.ipynb) | Research/demo path; production in `ml/`, `app/`, `services/` |
| Report + QA evidence | [`report/`](report/) | DOCX/PDF, pytest, lint, metrics snapshot |

Upload to Taxila as directed by the course (typically `179.pdf` / `179.docx`).
Use **this folder** (or a git archive of it) as the source ZIP.

---

## Repository layout (production code)

Run everything from this folder (`179/`). Research notebook `179.ipynb` is
kept out of the tree below; it is not required to run the system.

```text
179/
├── app/                      # FastAPI transport
│   ├── __init__.py
│   ├── api.py                # /health, /predict, /v1/predictions, /metrics
│   └── schemas.py            # Pydantic request/response contracts
├── ml/                       # ML lifecycle
│   ├── __init__.py
│   ├── data.py               # dataset + schema validation
│   ├── ingestion.py          # TXT / PDF / DOCX extraction
│   ├── preprocessing.py      # clean_text
│   ├── features.py           # FeatureEngineer (TF-IDF)
│   ├── trainer.py            # train, quality gates, persist
│   └── predictor.py          # load artifact + ranked inference
├── services/                 # domain layer (API and Streamlit share this)
│   ├── __init__.py
│   ├── application.py        # validate → score → audit
│   ├── scoring.py            # explanation + advisory decision note
│   └── audit.py              # metadata only; no raw resume text
├── quality/
│   ├── __init__.py
│   ├── data_metrics.py
│   └── model_metrics.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_continuity.py
│   ├── test_data_quality.py
│   ├── test_features.py
│   ├── test_inference.py
│   ├── test_preprocessing.py
│   ├── test_quality_metrics.py
│   └── test_training.py
├── config.py
├── logging_config.py
├── train.py                  # offline training
├── run_api.py                # uvicorn on 127.0.0.1:8000
└── streamlit_app.py          # recruiter UI
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
| `ml/`, `services/`, `app/`, `quality/` | **Production system** - run, test, deploy |
| `179.ipynb` | Research/demo notebook; production code in `ml/`, `services/`, `app/` |

You do not need Jupyter to run the application.

---

## Quick start

```bash
# From this directory (179/)
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
- Recruiter UI: `streamlit run streamlit_app.py` (candidate name, TXT/PDF/DOCX upload or paste, explanation, audit history)

### Docker

From this `179/` folder. A single container now starts **both** the API and the recruiter UI.

```bash
docker build -t group179-resume-screening:latest .
docker run -d --name 179 --restart unless-stopped -p 8000:8000 -p 8501:8501 group179-resume-screening:latest
```

- API: http://127.0.0.1:8000/docs
- Recruiter UI: http://127.0.0.1:8501

`--restart unless-stopped` brings both back after a reboot or Docker restart. Publish **both** ports or the UI will be unreachable.

API and recruiter UI as two services (same URLs):

```bash
docker compose up -d --build
```

Compose also uses `restart: unless-stopped`. Stop with `docker compose down`.

### Kubernetes (kind, 3 pods)

Three pods in namespace `resume-screening` on a local [kind](https://kind.sigs.k8s.io/) cluster:

| Pod | Role | Port |
|-----|------|------|
| `resume-api` | FastAPI inference | 8000 |
| `resume-ui` | Streamlit recruiter UI | 8501 |
| `resume-gateway` | nginx front door | 80 → NodePort 30080 |

```bash
docker build -t group179-resume-screening:latest .
kind create cluster --config k8s/kind-cluster.yaml
kind load docker-image group179-resume-screening:latest --name resume-screening
kubectl config use-context kind-resume-screening
kubectl apply -k k8s
kubectl -n resume-screening get pods
```

- Recruiter UI: http://127.0.0.1:30080/
- API docs: http://127.0.0.1:30080/docs
- API health: http://127.0.0.1:30080/health

Tear down with `kubectl delete -k k8s` and `kind delete cluster --name resume-screening`.

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

This assignment API is unauthenticated. Authentication and authorization are
out of scope here and would be required before production use.

---

## Verified quality snapshot

Recorded under `report/` (regenerate with `python train.py` and `pytest`):

- Tests: 44 collected / 43 functions (one parametrized case) 
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
| 2025AA05032 | Prathap Wagle | Designed system architecture, integrated Streamlit UI & application layer, authored main report, added code comments, and validated package readiness. | 25% |
| 2024AC05999 | Prasanna R T | Conducted test design review, performed functional validation of candidate scoring workflows, and verified edge-case handling. | 25% |
| 2024AC05914 | Pranav Mehrotra | Reviewed quality assurance evidence, conducted code audit reviews, and validated final package integrity. | 25% |


