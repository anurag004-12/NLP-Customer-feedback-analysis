"""Reusable customer-review text preprocessing."""

from __future__ import annotations

import html
import re
import string


DEFAULT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "my",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
}


class TextPreprocessor:
    """Clean and normalize review text for NLP models."""

    def __init__(self, remove_stopwords: bool = True) -> None:
        self.remove_stopwords = remove_stopwords
        self.stopwords = DEFAULT_STOPWORDS
        self.punctuation_table = str.maketrans("", "", string.punctuation)

    def clean(self, text: object) -> str:
        """Apply lowercasing, URL/HTML/emoji/punctuation removal, and lemmatization."""
        cleaned = "" if text is None else str(text)
        cleaned = html.unescape(cleaned.lower())
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        cleaned = re.sub(r"<.*?>", " ", cleaned)
        cleaned = self._remove_emoji(cleaned)
        cleaned = cleaned.translate(self.punctuation_table)
        tokens = [self._lemmatize(token) for token in cleaned.split()]
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        return re.sub(r"\s+", " ", " ".join(tokens)).strip()

    def transform(self, texts: list[object]) -> list[str]:
        """Clean a list of text values."""
        return [self.clean(text) for text in texts]

    @staticmethod
    def _remove_emoji(text: str) -> str:
        return re.sub(
            "["
            "\U0001f600-\U0001f64f"
            "\U0001f300-\U0001f5ff"
            "\U0001f680-\U0001f6ff"
            "\U0001f1e0-\U0001f1ff"
            "\U00002700-\U000027bf"
            "\U000024c2-\U0001f251"
            "]+",
            " ",
            text,
        )

    @staticmethod
    def _lemmatize(token: str) -> str:
        """Lightweight lemmatization fallback that works without external corpora."""
        if len(token) > 4 and token.endswith("ies"):
            return token[:-3] + "y"
        if len(token) > 5 and token.endswith("ing"):
            return token[:-3]
        if len(token) > 4 and token.endswith("ed"):
            return token[:-2]
        if len(token) > 3 and token.endswith("s"):
            return token[:-1]
        return token

