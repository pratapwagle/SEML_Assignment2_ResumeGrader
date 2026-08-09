from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"resume_text", "job_role"}


@dataclass(frozen=True)
class DataQualityReport:
    row_count: int
    missing_values: int
    duplicate_rows: int
    schema_valid: bool
    class_count: int


def build_training_data() -> pd.DataFrame:
    rows = [
        (
            "python machine learning pandas sklearn model training classification "
            "regression statistics nlp",
            "Data Scientist",
        ),
        (
            "tensorflow pytorch feature engineering experimentation evaluation "
            "computer vision deep learning",
            "Data Scientist",
        ),
        (
            "predictive modeling feature selection hypothesis testing data analysis "
            "notebooks python",
            "Data Scientist",
        ),
        (
            "nlp transformers sentiment analysis model deployment experimentation "
            "metrics",
            "Data Scientist",
        ),
        (
            "sql spark airflow kafka warehouse etl pipelines data lake orchestration",
            "Data Engineer",
        ),
        (
            "bigquery snowflake ingestion batch streaming quality checks schema "
            "evolution",
            "Data Engineer",
        ),
        (
            "data pipelines orchestration warehousing airflow spark dbt sql batch "
            "processing",
            "Data Engineer",
        ),
        (
            "kafka streaming ingestion schema registry cloud warehouse lakehouse "
            "data engineering",
            "Data Engineer",
        ),
        (
            "java spring boot rest api microservices backend docker kubernetes",
            "Backend Developer",
        ),
        (
            "nodejs express graphql authentication mongodb scalable backend "
            "development",
            "Backend Developer",
        ),
        (
            "backend api caching redis authentication service integration java spring",
            "Backend Developer",
        ),
        (
            "rest endpoints message queues backend architecture nodejs express "
            "postgres",
            "Backend Developer",
        ),
        (
            "selenium playwright test automation regression api testing jira quality",
            "QA Engineer",
        ),
        (
            "test cases defect tracking automation framework performance testing "
            "quality assurance",
            "QA Engineer",
        ),
        (
            "qa automation api validation selenium test plans bug triage regression "
            "suite",
            "QA Engineer",
        ),
        (
            "playwright ui testing smoke testing exploratory testing defect logging",
            "QA Engineer",
        ),
        (
            "react angular html css javascript ui ux responsive frontend typescript",
            "Frontend Developer",
        ),
        (
            "frontend accessibility component library figma design system web "
            "performance",
            "Frontend Developer",
        ),
        (
            "responsive ui frontend state management react typescript accessibility "
            "css",
            "Frontend Developer",
        ),
        (
            "component design system figma handoff frontend performance optimization",
            "Frontend Developer",
        ),
        (
            "aws terraform kubernetes docker ci cd observability linux grafana",
            "DevOps Engineer",
        ),
        (
            "jenkins github actions infrastructure as code monitoring incident "
            "response cloud",
            "DevOps Engineer",
        ),
        (
            "devops release automation infrastructure monitoring kubernetes terraform "
            "cloud reliability",
            "DevOps Engineer",
        ),
        (
            "container orchestration ci cd observability logging linux automation aws",
            "DevOps Engineer",
        ),
    ]
    frame = pd.DataFrame(rows, columns=["resume_text", "job_role"])
    logger.info("Created demonstration dataset with %d rows", len(frame))
    return frame


def validate_training_data(frame: pd.DataFrame) -> DataQualityReport:
    if not isinstance(frame, pd.DataFrame):
        logger.error("Training data must be a pandas DataFrame")
        raise TypeError("training data must be a pandas DataFrame")

    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        logger.error(
            "Schema validation failed; missing columns: %s",
            sorted(missing_columns),
        )
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    missing = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())
    duplicates = int(frame.duplicated().sum())
    if missing:
        logger.warning("Dataset contains %d missing values", missing)
        raise ValueError("Training data contains missing required values")
    if duplicates:
        logger.warning("Dataset contains %d duplicate rows", duplicates)

    class_counts = frame["job_role"].value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        logger.error("Each of at least two classes must contain two or more rows")
        raise ValueError(
            "Training data requires at least two classes with two rows each"
        )

    report = DataQualityReport(
        row_count=len(frame),
        missing_values=missing,
        duplicate_rows=duplicates,
        schema_valid=True,
        class_count=int(len(class_counts)),
    )
    logger.info("Data quality validation completed: %s", report)
    return report
