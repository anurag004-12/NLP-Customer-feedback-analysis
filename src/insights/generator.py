"""Advanced NLP insight utilities."""

from __future__ import annotations

from collections import Counter

import pandas as pd


def extract_keywords(texts: list[str], top_n: int = 20) -> list[tuple[str, int]]:
    """Extract frequent keywords from cleaned reviews."""
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(token for token in str(text).split() if len(token) > 2)
    return counter.most_common(top_n)


def generate_business_insights(df: pd.DataFrame) -> str:
    """Create a readable executive summary from sentiment and keyword patterns."""
    sentiment_counts = df["sentiment"].value_counts(normalize=True)
    dominant = sentiment_counts.idxmax() if not sentiment_counts.empty else "Unknown"
    positives = df.loc[df["sentiment"] == "Positive", "clean_text"].tolist()
    negatives = df.loc[df["sentiment"] == "Negative", "clean_text"].tolist()
    praise = ", ".join(word for word, _ in extract_keywords(positives, 5)) or "limited positive themes"
    complaints = ", ".join(word for word, _ in extract_keywords(negatives, 5)) or "limited negative themes"
    return (
        f"The dominant sentiment is {dominant}. Customers most often praise {praise}. "
        f"Common complaint themes include {complaints}. These patterns suggest prioritizing "
        "service recovery for recurring pain points while reinforcing the strongest product and delivery experiences."
    )


def topic_modeling(texts: list[str], n_topics: int = 5) -> list[list[str]]:
    """Run LDA topic modeling when scikit-learn is installed."""
    try:
        from sklearn.decomposition import LatentDirichletAllocation
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise ImportError("Install scikit-learn to run topic modeling.") from exc

    vectorizer = CountVectorizer(max_features=1000, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(matrix)
    terms = vectorizer.get_feature_names_out()
    topics = []
    for topic in lda.components_:
        topics.append([terms[index] for index in topic.argsort()[-8:][::-1]])
    return topics
