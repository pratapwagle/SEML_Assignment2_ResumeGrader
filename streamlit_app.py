"""Streamlit recruiter UI preserved from Assignment I.

Uses the same ``ResumeScreeningApplication`` path as the REST API so scoring
and audit behaviour cannot drift between interfaces.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from config import AUDIT_PATH, METRICS_PATH, MODEL_PATH
from logging_config import configure_logging
from ml.ingestion import extract_resume_text
from ml.predictor import ModelPredictor
from services.application import ResumeScreeningApplication, ResumeSubmission
from services.audit import AuditRepository
from services.scoring import ResumeScoringService

configure_logging()


@st.cache_resource
def load_application() -> ResumeScreeningApplication:
    """Build and cache the shared application service for the Streamlit process."""
    service = ResumeScoringService(ModelPredictor(MODEL_PATH))
    return ResumeScreeningApplication(service, AuditRepository(AUDIT_PATH))


def main() -> None:
    """Render upload/paste workflow, results, audit history, and metrics sidebar."""
    st.set_page_config(page_title="AI Resume Screening - Group 179", layout="wide")
    st.title("AI Resume Screening System")
    st.caption(
        "Assignment I recruiter workflow, powered by the modular Assignment II "
        "inference service."
    )

    if not MODEL_PATH.exists():
        st.error("Model artifact is missing. Run `python train.py` first.")
        st.stop()

    application = load_application()
    left, right = st.columns([1.1, 0.9])
    with left:
        candidate_name = st.text_input("Candidate name", value="Candidate")
        uploaded_file = st.file_uploader(
            "Upload resume",
            type=["txt", "pdf", "docx"],
            help="Maximum upload size enforced by the ingestion module: 5 MB.",
        )
        pasted_text = st.text_area("Or paste resume text", height=220)

        extracted_text = ""
        if uploaded_file is not None:
            try:
                extracted_text = extract_resume_text(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                )
                st.success(
                    f"Extracted {len(extracted_text)} characters from "
                    f"{uploaded_file.name}."
                )
            except ValueError as exc:
                st.error(str(exc))

        if st.button("Analyze resume", type="primary"):
            resume_text = pasted_text.strip() or extracted_text
            if not candidate_name.strip():
                st.error("Candidate name must not be blank.")
            elif not resume_text:
                st.error("Upload a resume or paste resume text.")
            else:
                try:
                    source = "text input" if pasted_text.strip() else "upload"
                    result = application.submit(
                        ResumeSubmission(
                            candidate_name=candidate_name,
                            source=source,
                            resume_text=resume_text,
                        )
                    )
                    st.session_state["latest_result"] = result
                except ValueError as exc:
                    st.error(str(exc))

    with right:
        st.subheader("Latest result")
        result = st.session_state.get("latest_result")
        if result:
            st.metric("Predicted role", result["predicted_role"])
            st.metric("Confidence", f"{result['confidence']:.3f}")
            st.write(result["explanation"])
            st.info(result["decision_note"])
            st.dataframe(
                pd.DataFrame(result["role_ranking"]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Analyze a resume to see the recommendation.")

    st.subheader("Audit history")
    history = application.repository.list_results()
    if history.empty:
        st.caption("No analyses have been stored.")
    else:
        st.dataframe(
            history.sort_values("timestamp_utc", ascending=False),
            hide_index=True,
            width="stretch",
        )

    if METRICS_PATH.exists():
        import json

        snapshot = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        model_quality = snapshot["model_quality"]
        st.sidebar.header("Verified model quality")
        st.sidebar.metric("Accuracy", f"{model_quality['accuracy']:.3f}")
        st.sidebar.metric("Weighted F1", f"{model_quality['weighted_f1']:.3f}")


if __name__ == "__main__":
    main()
