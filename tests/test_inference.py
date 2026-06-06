import pickle
import tempfile
import unittest
from pathlib import Path

from src.inference.predictor import SentimentPredictor


class DummyVectorizer:
    def transform(self, texts):
        return texts


class DummyModel:
    def predict(self, features):
        return ["Positive"]

    def predict_proba(self, features):
        return [[0.05, 0.9, 0.05]]


class InferenceTest(unittest.TestCase):
    def test_predictor_loads_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"
            vectorizer_path = Path(temp_dir) / "vectorizer.pkl"
            model_path.write_bytes(pickle.dumps(DummyModel()))
            vectorizer_path.write_bytes(pickle.dumps(DummyVectorizer()))
            result = SentimentPredictor(model_path, vectorizer_path).predict("Great service")
            self.assertEqual(result["sentiment"], "Positive")
            self.assertAlmostEqual(result["confidence"], 0.9)


if __name__ == "__main__":
    unittest.main()

