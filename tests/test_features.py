import unittest

import pandas as pd

from src.training.features import add_text_features


class FeatureEngineeringTest(unittest.TestCase):
    def test_add_text_features(self) -> None:
        df = pd.DataFrame({"review_text": ["Excellent delivery", "Worst refund delay"]})
        result = add_text_features(df)
        self.assertIn("clean_text", result.columns)
        self.assertIn("review_length", result.columns)
        self.assertIn("sentiment_score", result.columns)
        self.assertGreater(result.loc[0, "sentiment_score"], result.loc[1, "sentiment_score"])


if __name__ == "__main__":
    unittest.main()

