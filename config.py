from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "resume_classifier.joblib"
METRICS_PATH = PROJECT_ROOT / "report" / "metrics_snapshot.json"
AUDIT_PATH = PROJECT_ROOT / "data" / "resume_screening_audit.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "resume_screening.log"
MODEL_VERSION = "2.1.0"
ROLE_LABELS = (
    "Backend Developer",
    "Data Engineer",
    "Data Scientist",
    "DevOps Engineer",
    "Frontend Developer",
    "QA Engineer",
)
