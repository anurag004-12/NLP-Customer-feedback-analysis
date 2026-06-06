"""Orchestration engine for the customer feedback analysis pipeline."""

from __future__ import annotations

from src.components.data_component import (
    build_profile,
    detect_dataset_schema,
    load_dataset,
    prepare_dataset_for_training,
    save_report,
)
from src.insights.generator import generate_business_insights
from src.logger import get_logger
from src.training.features import add_text_features
from src.utils.paths import DEFAULT_DATASET, REPORTS_DIR, ensure_project_dirs
from src.visualization.eda import generate_eda_plots

LOGGER = get_logger(__name__)


def run_pipeline(skip_training: bool = False) -> None:
    """Run the full customer feedback analysis pipeline."""
    ensure_project_dirs()
    raw_df = load_dataset(DEFAULT_DATASET)
    text_col, rating_col = detect_dataset_schema(raw_df)

    profile = build_profile(raw_df, text_col, rating_col)
    save_report(profile, REPORTS_DIR / "data_profile.json")

    prepared = add_text_features(prepare_dataset_for_training(raw_df, text_col, rating_col))
    prepared.to_csv(REPORTS_DIR / "processed_reviews_sample.csv", index=False)
    generate_eda_plots(prepared)
    save_report({"summary": generate_business_insights(prepared)}, REPORTS_DIR / "insights_report.json")

    if skip_training:
        LOGGER.info("Skipping model training.")
        return

    from src.training.train_model import train_models

    train_models(DEFAULT_DATASET)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Customer feedback analysis pipeline")
    parser.add_argument("--skip-training", action="store_true", help="Run reports and EDA without training ML models.")
    args = parser.parse_args()
    run_pipeline(skip_training=args.skip_training)


if __name__ == "__main__":
    main()
