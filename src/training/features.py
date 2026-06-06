"""Feature engineering utilities."""

from __future__ import annotations

import pandas as pd

from src.preprocessing.text_cleaner import TextPreprocessor


def add_text_features(df: pd.DataFrame, text_col: str = "review_text") -> pd.DataFrame:
    """Add cleaned text, review length, and simple sentiment score features."""
    preprocessor = TextPreprocessor()
    featured = df.copy()
    featured["clean_text"] = preprocessor.transform(featured[text_col].fillna("").tolist())
    featured["review_length"] = featured[text_col].fillna("").astype(str).str.split().str.len()
    positive_terms = {"good", "great", "excellent", "love", "fast", "value", "happy", "perfect"}
    negative_terms = {"bad", "terrible", "poor", "late", "broken", "worst", "delay", "refund"}

    def score(text: str) -> int:
        tokens = set(text.split())
        return len(tokens & positive_terms) - len(tokens & negative_terms)

    featured["sentiment_score"] = featured["clean_text"].apply(score)
    return featured


def build_vectorizer():
    """Create the TF-IDF vectorizer with unigram and bigram features."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError("Install scikit-learn to build TF-IDF features.") from exc

    return TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2)

