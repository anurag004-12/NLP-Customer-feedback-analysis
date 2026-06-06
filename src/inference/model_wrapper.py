"""Pickle-safe model wrappers."""

from __future__ import annotations

from typing import Any


class DecodedClassifier:
    """Wrap a classifier so predictions return original string labels."""

    def __init__(self, model: Any, label_encoder: Any) -> None:
        self.model = model
        self.label_encoder = label_encoder

    def predict(self, features: Any) -> Any:
        """Return decoded class labels."""
        return self.label_encoder.inverse_transform(self.model.predict(features))

    def predict_proba(self, features: Any) -> Any:
        """Delegate class probability prediction."""
        return self.model.predict_proba(features)

