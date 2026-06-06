"""Model training and evaluation entry point."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from src.inference.model_wrapper import DecodedClassifier
from src.training.features import add_text_features, build_vectorizer
from src.components.model_component import build_model_candidates, save_model_artifacts
from src.components.evaluation import evaluate_predictions
from src.utils.data import detect_review_schema, load_reviews_csv, prepare_dataset, save_json_report
from src.utils.logger import get_logger
from src.utils.paths import DEFAULT_DATASET, MODELS_DIR, REPORTS_DIR, TRAINED_MODEL_PATH, VECTORIZER_PATH, ensure_project_dirs


LOGGER = get_logger(__name__)


def train_models(dataset_path: str | Path = DEFAULT_DATASET) -> dict[str, Any]:
    """Train candidate sentiment classifiers and save the best estimator."""

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    ensure_project_dirs()
    raw_df = load_reviews_csv(dataset_path)
    text_col, rating_col = detect_review_schema(raw_df)
    df = add_text_features(prepare_dataset(raw_df, text_col, rating_col))
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    x_train, x_test, y_train_raw, y_test_raw = train_test_split(
        df["clean_text"],
        df["sentiment"],
        test_size=0.2,
        random_state=42,
        stratify=df["sentiment"],
    )
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    vectorizer = build_vectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    models = build_model_candidates()

    results: dict[str, Any] = {}
    best_model = None
    best_f1 = -1.0
    best_name = ""

    for name, model in models.items():
        LOGGER.info("Training %s", name)
        model.fit(x_train_vec, y_train)
        predictions = model.predict(x_test_vec)
        model_results = evaluate_predictions(y_test, predictions, target_names=label_encoder.classes_)
        results[name] = model_results

        if model_results["f1"] > best_f1:
            best_name = name
            best_model = model
            best_f1 = model_results["f1"]

    if best_model is None:
        raise RuntimeError("No model completed training successfully.")

    save_model_artifacts(best_model, label_encoder, vectorizer)
    report = {"best_model": best_name, "best_weighted_f1": best_f1, "results": results}
    save_json_report(report, REPORTS_DIR / "model_report.json")
    LOGGER.info("Saved best model to %s", MODELS_DIR)
    return report


if __name__ == "__main__":
    print(train_models())
