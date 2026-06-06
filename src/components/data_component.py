"""Data preparation component wrappers."""

from pathlib import Path
import pandas as pd

from src.utils.data import (
    build_data_profile,
    detect_review_schema,
    load_reviews_csv,
    prepare_dataset,
    save_json_report,
)


def load_dataset(path: str | Path) -> pd.DataFrame:
    return load_reviews_csv(path)


def detect_dataset_schema(df: pd.DataFrame) -> tuple[str, str]:
    return detect_review_schema(df)


def prepare_dataset_for_training(df: pd.DataFrame, text_col: str, rating_col: str) -> pd.DataFrame:
    return prepare_dataset(df, text_col, rating_col)


def build_profile(df: pd.DataFrame, text_col: str, rating_col: str) -> dict[str, object]:
    return build_data_profile(df, text_col, rating_col)


def save_report(report: dict[str, object], path: str | Path) -> None:
    save_json_report(report, path)
