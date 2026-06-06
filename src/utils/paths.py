"""Centralized project paths."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "assets"
PLOTS_DIR = ASSETS_DIR / "plots"
REPORTS_DIR = ASSETS_DIR / "reports"
DEFAULT_DATASET = DATA_DIR / "Amazon_Reviews.csv"
TRAINED_MODEL_PATH = MODELS_DIR / "trained_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.pkl"


def ensure_project_dirs() -> None:
    """Create runtime output directories."""
    for path in (DATA_DIR, MODELS_DIR, ASSETS_DIR, PLOTS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
