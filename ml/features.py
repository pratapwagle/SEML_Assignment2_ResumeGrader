"""Feature engineering for resume text classification.

Isolates the TF-IDF representation so training and inference share one
sklearn-compatible transformer instead of constructing the vectorizer inline.
"""

from __future__ import annotations

import logging
from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """TF-IDF feature step used by the training and inference pipeline.

    Args:
        ngram_range: Character or word n-gram range passed to TF-IDF.
        min_df: Minimum document frequency for a term to be kept.
        max_features: Optional vocabulary cap.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_features: int | None = None,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features

    def fit(self, texts: Iterable[str], y=None) -> "FeatureEngineer":
        """Fit the TF-IDF vocabulary on cleaned resume texts."""
        self.vectorizer_ = TfidfVectorizer(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
        )
        self.vectorizer_.fit(texts)
        logger.info(
            "Fitted FeatureEngineer with %d terms",
            len(self.vectorizer_.vocabulary_),
        )
        return self

    def transform(self, texts: Iterable[str]):
        """Transform cleaned resume texts into a TF-IDF matrix."""
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("FeatureEngineer must be fitted before transform")
        return self.vectorizer_.transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return the fitted vocabulary in sklearn order."""
        if not hasattr(self, "vectorizer_"):
            raise RuntimeError("FeatureEngineer must be fitted before reading names")
        return self.vectorizer_.get_feature_names_out().tolist()
