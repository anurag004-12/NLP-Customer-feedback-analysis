"""EDA chart generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.insights.generator import extract_keywords
from src.utils.logger import get_logger
from src.utils.paths import PLOTS_DIR


LOGGER = get_logger(__name__)


def generate_eda_plots(df: pd.DataFrame, output_dir: str | Path = PLOTS_DIR) -> list[Path]:
    """Create rating, sentiment, word frequency, word cloud, and length plots."""
    try:
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError:
        LOGGER.warning("matplotlib/wordcloud unavailable; skipping plot generation.")
        return []

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save_bar(series: pd.Series, title: str, filename: str, color: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        series.plot(kind="bar", ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    save_bar(df["rating_numeric"].value_counts().sort_index(), "Rating Distribution", "rating_distribution.png", "#2d6cdf")
    save_bar(df["sentiment"].value_counts(), "Sentiment Distribution", "sentiment_distribution.png", "#1b8a5a")

    keywords = pd.Series(dict(extract_keywords(df["clean_text"].tolist(), 20))).sort_values()
    if not keywords.empty:
        save_bar(keywords, "Most Frequent Words", "frequent_words.png", "#c15b3d")

    fig, ax = plt.subplots(figsize=(8, 5))
    df["review_length"].plot(kind="hist", bins=40, ax=ax, color="#6c5ce7")
    ax.set_title("Review Length Analysis")
    ax.set_xlabel("Words per review")
    fig.tight_layout()
    length_path = output / "review_length_analysis.png"
    fig.savefig(length_path, dpi=140)
    plt.close(fig)
    saved.append(length_path)

    all_text = " ".join(df["clean_text"].dropna().astype(str).tolist())
    if all_text.strip():
        wordcloud = WordCloud(width=1200, height=700, background_color="white").generate(all_text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        path = output / "word_cloud.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        saved.append(path)

    return saved

