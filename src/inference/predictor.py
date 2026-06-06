"""Sentiment prediction service."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.preprocessing.text_cleaner import TextPreprocessor
from src.utils.paths import TRAINED_MODEL_PATH, VECTORIZER_PATH


class SentimentPredictor:
    """Load saved model artifacts and predict review sentiment."""

    def __init__(
        self,
        model_path: str | Path = TRAINED_MODEL_PATH,
        vectorizer_path: str | Path = VECTORIZER_PATH,
    ) -> None:
        self.preprocessor = TextPreprocessor()
        with Path(model_path).open("rb") as model_file:
            loaded = pickle.load(model_file)
        with Path(vectorizer_path).open("rb") as vectorizer_file:
            self.vectorizer = pickle.load(vectorizer_file)

        if isinstance(loaded, tuple) and len(loaded) == 2:
            self.model, self.label_encoder = loaded
        else:
            self.model = loaded
            self.label_encoder = None

    def predict(self, review: str) -> dict[str, float | str]:
        """Predict sentiment and confidence for one review."""
        clean_text = self.preprocessor.clean(review)
        features = self.vectorizer.transform([clean_text])
        raw_prediction = self.model.predict(features)[0]
        if self.label_encoder is not None:
            prediction = str(self.label_encoder.inverse_transform([raw_prediction])[0])
        else:
            prediction = str(raw_prediction)

        confidence = 1.0
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
            confidence = float(max(probabilities))
        return {"sentiment": prediction, "confidence": confidence}

