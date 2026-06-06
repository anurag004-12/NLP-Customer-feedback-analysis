"""Model building and persistence components."""

from __future__ import annotations

import pickle
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from src.components.evaluation import evaluate_predictions
from src.exceptions import ModelTrainingError
from src.utils.paths import MODELS_DIR, TRAINED_MODEL_PATH, VECTORIZER_PATH


def build_model_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
    }

    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=80,
            max_depth=5,
            eval_metric="mlogloss",
            random_state=42,
        )
    except ImportError:
        pass

    return candidates


def select_best_model(results: dict[str, Any]) -> str:
    if not results:
        raise ModelTrainingError("No models were trained successfully.")
    return max(results, key=lambda key: results[key].get("f1", 0.0))


def save_model_artifacts(model: Any, label_encoder: LabelEncoder, vectorizer: Any) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINED_MODEL_PATH.open("wb") as model_file:
        pickle.dump((model, label_encoder), model_file)
    with VECTORIZER_PATH.open("wb") as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)
