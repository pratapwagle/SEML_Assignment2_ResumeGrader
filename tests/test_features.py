from ml.data import build_training_data
from ml.features import FeatureEngineer
from ml.preprocessing import clean_text
from ml.trainer import build_pipeline


def test_feature_engineer_fit_transform_shape():
    texts = build_training_data()["resume_text"].map(clean_text)
    engineer = FeatureEngineer(ngram_range=(1, 2), min_df=1).fit(texts)
    matrix = engineer.transform(texts)
    assert matrix.shape[0] == len(texts)
    assert matrix.shape[1] > 0
    assert len(engineer.get_feature_names()) == matrix.shape[1]


def test_feature_engineer_is_deterministic():
    texts = build_training_data()["resume_text"].map(clean_text)
    first = FeatureEngineer().fit(texts).get_feature_names()
    second = FeatureEngineer().fit(texts).get_feature_names()
    assert first == second


def test_feature_engineer_handles_unseen_tokens():
    engineer = FeatureEngineer().fit(["python machine learning"])
    matrix = engineer.transform(["unknowntoken xyzabc"])
    assert matrix.shape == (1, len(engineer.get_feature_names()))


def test_training_pipeline_uses_feature_engineer_step():
    pipeline = build_pipeline()
    assert list(pipeline.named_steps)[0] == "features"
    assert isinstance(pipeline.named_steps["features"], FeatureEngineer)
