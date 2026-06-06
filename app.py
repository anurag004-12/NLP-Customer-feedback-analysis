"""Streamlit application for AI customer feedback analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.inference.predictor import SentimentPredictor
from src.insights.generator import generate_business_insights, topic_modeling
from src.training.features import add_text_features
from src.utils.data import detect_review_schema, load_reviews_csv, prepare_dataset
from src.utils.paths import DEFAULT_DATASET, PLOTS_DIR, REPORTS_DIR, TRAINED_MODEL_PATH, VECTORIZER_PATH


st.set_page_config(page_title="Customer Feedback Analysis", layout="wide")


@st.cache_data(show_spinner=False)
def load_prepared_data() -> pd.DataFrame:
    """Load and preprocess data for Streamlit views."""
    processed = REPORTS_DIR / "processed_reviews_sample.csv"
    if processed.exists():
        return pd.read_csv(processed)
    raw_df = load_reviews_csv(DEFAULT_DATASET)
    text_col, rating_col = detect_review_schema(raw_df)
    return add_text_features(prepare_dataset(raw_df, text_col, rating_col))


def show_plot(filename: str, caption: str) -> None:
    path = PLOTS_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info("Run `python run_pipeline.py --skip-training` to generate this chart.")


def dashboard(df: pd.DataFrame) -> None:
    st.title("AI Customer Feedback Analysis")
    total, positive, neutral, negative = st.columns(4)
    total.metric("Total Reviews", f"{len(df):,}")
    positive.metric("Positive", int((df["sentiment"] == "Positive").sum()))
    neutral.metric("Neutral", int((df["sentiment"] == "Neutral").sum()))
    negative.metric("Negative", int((df["sentiment"] == "Negative").sum()))

    left, right = st.columns(2)
    with left:
        show_plot("rating_distribution.png", "Rating Distribution")
    with right:
        show_plot("sentiment_distribution.png", "Sentiment Distribution")


def analytics(df: pd.DataFrame) -> None:
    st.title("Analytics")
    left, right = st.columns(2)
    with left:
        show_plot("word_cloud.png", "Word Cloud")
        show_plot("frequent_words.png", "Most Frequent Words")
    with right:
        show_plot("review_length_analysis.png", "Review Length Analysis")
        st.subheader("Topic Analysis")
        try:
            for index, terms in enumerate(topic_modeling(df["clean_text"].dropna().astype(str).tolist(), 5), start=1):
                st.write(f"Topic {index}: {', '.join(terms)}")
        except Exception as exc:
            st.info(f"Topic modeling requires scikit-learn. {exc}")


def prediction() -> None:
    st.title("Sentiment Prediction")
    review = st.text_area("Enter customer review", height=180)
    if st.button("Predict Sentiment", type="primary"):
        if not review.strip():
            st.warning("Enter a review before predicting.")
            return
        if not TRAINED_MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
            st.error("Model artifacts not found. Run `python -m src.training.train_model` first.")
            return
        result = SentimentPredictor().predict(review)
        st.metric("Predicted Sentiment", str(result["sentiment"]))
        st.progress(float(result["confidence"]))
        st.caption(f"Confidence: {float(result['confidence']):.2%}")


def insights(df: pd.DataFrame) -> None:
    st.title("Insights")
    report_path = REPORTS_DIR / "insights_report.json"
    if report_path.exists():
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        st.write(report.get("summary", ""))
    else:
        st.write(generate_business_insights(df))

    st.subheader("Common Complaints")
    st.dataframe(df[df["sentiment"] == "Negative"][["review_text", "rating_numeric"]].head(20), use_container_width=True)
    st.subheader("Common Praises")
    st.dataframe(df[df["sentiment"] == "Positive"][["review_text", "rating_numeric"]].head(20), use_container_width=True)


def main() -> None:
    df = load_prepared_data()
    page = st.sidebar.radio("Page", ["Dashboard", "Analytics", "Prediction", "Insights"])
    if page == "Dashboard":
        dashboard(df)
    elif page == "Analytics":
        analytics(df)
    elif page == "Prediction":
        prediction()
    else:
        insights(df)


if __name__ == "__main__":
    main()
