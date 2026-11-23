"""Preprocessing and schema tests."""
import numpy as np

from src.config import CLASS_MAPPING, FEATURES
from src.features.preprocessing import (
    AstronomyFeatureTransformer,
    clean_data,
    create_features,
    encode_classes,
    prepare_sample,
)


def test_clean_data_removes_sentinel_and_rerun_id(sample_raw_data):
    data = sample_raw_data.copy()
    data.loc[0, "u"] = -9999
    cleaned = clean_data(data)
    assert len(cleaned) == len(data) - 1
    assert "rerun_ID" not in cleaned.columns
    assert not (cleaned[["u", "g", "r", "i", "z"]] == -9999).any().any()


def test_create_features_uses_notebook_formulas(sample_raw_data):
    result = create_features(sample_raw_data)
    first = sample_raw_data.iloc[0]
    assert result.iloc[0]["u_g"] == first["u"] - first["g"]
    assert result.iloc[0]["g_r"] == first["g"] - first["r"]
    assert result.iloc[0]["r_i"] == first["r"] - first["i"]
    assert result.iloc[0]["i_z"] == first["i"] - first["z"]
    expected = np.sign(first["redshift"]) * np.log1p(abs(first["redshift"]))
    assert result.iloc[0]["signed_log_redshift"] == expected


def test_transformer_returns_exact_feature_order(sample_raw_data):
    transformer = AstronomyFeatureTransformer().fit(sample_raw_data)
    assert list(transformer.transform(sample_raw_data).columns) == FEATURES
    assert list(transformer.get_feature_names_out()) == FEATURES


def test_class_mapping_matches_notebook(sample_raw_data):
    encoded, mapping = encode_classes(sample_raw_data["class"])
    assert mapping == CLASS_MAPPING
    assert set(encoded.unique()) == {0, 1, 2}


def test_sample_and_split_are_stratified(sample_raw_data):
    featured = create_features(clean_data(sample_raw_data))
    split = prepare_sample(featured, sample_size=60, random_state=42)
    assert (split.sample_size, len(split.X_train), len(split.X_test)) == (60, 48, 12)
    assert set(split.y_train.unique()) == {0, 1, 2}
    assert set(split.y_test.unique()) == {0, 1, 2}
