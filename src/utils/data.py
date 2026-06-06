"""Dataset loading, schema detection, and reporting utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

TEXT_CANDIDATES = (
    "reviewtext",
    "review_body",
    "review text",
    "review",
    "text",
    "comment",
    "content",
)
RATING_CANDIDATES = ("overall", "star_rating", "rating", "stars", "score")


def load_reviews_csv(path: str | Path) -> pd.DataFrame:
    """Load a reviews CSV with a robust parser fallback for malformed rows."""
    csv_path = Path(path)
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        LOGGER.warning("Default CSV parser failed: %s. Retrying with python engine.", exc)
        return pd.read_csv(csv_path, engine="python", on_bad_lines="skip")


def _normalize_column(name: str) -> str:
    return re.sub(r"[^a-z0-9_ ]+", "", str(name).lower()).strip()


def detect_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    """Detect a column from likely names using exact and partial matching."""
    normalized = {_normalize_column(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    for column in columns:
        norm = _normalize_column(column)
        if any(candidate in norm for candidate in candidates):
            return column

    raise ValueError(f"Could not detect column from candidates: {candidates}")


def detect_review_schema(df: pd.DataFrame) -> tuple[str, str]:
    """Return detected review text and rating columns."""
    columns = [str(column) for column in df.columns]
    return detect_column(columns, TEXT_CANDIDATES), detect_column(columns, RATING_CANDIDATES)


def parse_rating(value: Any) -> float | None:
    """Extract a numeric rating from raw values such as 'Rated 4 out of 5 stars'."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    match = re.search(r"([1-5](?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else None


def create_sentiment_label(rating: float | None) -> str | None:
    """Map star ratings to Negative, Neutral, or Positive."""
    if rating is None or pd.isna(rating):
        return None
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def prepare_dataset(df: pd.DataFrame, text_col: str, rating_col: str) -> pd.DataFrame:
    """Create a normalized modeling dataframe."""
    prepared = df.copy()
    prepared["review_text"] = prepared[text_col].fillna("").astype(str)
    prepared["rating_numeric"] = prepared[rating_col].apply(parse_rating)
    prepared["sentiment"] = prepared["rating_numeric"].apply(create_sentiment_label)
    prepared["review_length"] = prepared["review_text"].str.split().str.len()
    return prepared.dropna(subset=["sentiment"]).reset_index(drop=True)


def build_data_profile(df: pd.DataFrame, text_col: str, rating_col: str) -> dict[str, Any]:
    """Generate a compact data understanding report."""
    numeric_rating = df[rating_col].apply(parse_rating)
    review_lengths = df[text_col].fillna("").astype(str).str.split().str.len()
    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(map(str, df.columns)),
        "detected_text_column": text_col,
        "detected_rating_column": rating_col,
        "missing_values": df.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "rating_distribution": numeric_rating.value_counts(dropna=False).sort_index().astype(int).to_dict(),
        "sentiment_distribution": numeric_rating.apply(create_sentiment_label)
        .value_counts(dropna=False)
        .astype(int)
        .to_dict(),
        "review_length": {
            "mean": float(review_lengths.mean()),
            "median": float(review_lengths.median()),
            "min": int(review_lengths.min()),
            "max": int(review_lengths.max()),
        },
        "class_imbalance": numeric_rating.apply(create_sentiment_label).value_counts(normalize=True).to_dict(),
    }
    q1 = review_lengths.quantile(0.25)
    q3 = review_lengths.quantile(0.75)
    iqr = q3 - q1
    profile["review_length_outliers_iqr"] = int(((review_lengths < q1 - 1.5 * iqr) | (review_lengths > q3 + 1.5 * iqr)).sum())
    return profile


def save_json_report(report: dict[str, Any], path: str | Path) -> None:
    """Persist a report as formatted JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

