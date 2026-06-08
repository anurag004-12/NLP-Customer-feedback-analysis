# AI Customer Feedback Analysis

> Production-ready NLP pipeline for analyzing Amazon customer reviews — from raw CSV to trained models and an interactive Streamlit dashboard.

🚀 **Live Demo:** [nlp-customer-feedback-analysis.onrender.com](https://nlp-customer-feedback-analysis.onrender.com)

---

## Overview

This project automates the full sentiment analysis lifecycle: it auto-detects review and rating columns, cleans text, assigns sentiment labels from star ratings, generates EDA reports, trains and compares multiple ML classifiers, and surfaces insights through a multi-page Streamlit application.

**Detected columns for the included dataset:**

| Field | Column Name |
|-------|-------------|
| Review Text | `Review Text` |
| Star Rating | `Rating` |

---

## Project Structure

```text
customer-feedback-analysis/
├── app.py                    # Streamlit entry point
├── run_pipeline.py           # Pipeline runner script
├── setup.py
├── render.yaml               # Render deployment config
├── DEPLOY_ON_RENDER.md       # Render deployment guide
├── Dockerfile
├── requirements.txt
├── runtime.txt
├── .gitignore
├── assets/
│   ├── plots/                # Generated EDA plots
│   └── reports/              # Profiling reports
├── data/
│   └── Amazon_Reviews.csv    # Source dataset
├── models/                   # Saved model artifacts
├── notebooks/                # Exploratory notebooks
├── scripts/                  # Utility scripts
├── src/
│   ├── components/
│   ├── exceptions.py
│   ├── inference/
│   ├── insights/
│   ├── logger.py
│   ├── pipeline/
│   ├── preprocessing/
│   ├── training/
│   ├── visualization/
│   └── utils/
└── tests/
```

---

## Features

### Data Profiling
Automatic profiling covers missing values, duplicate rows, rating distribution, sentiment distribution, and review length statistics.

### Text Preprocessing
Reviews are cleaned through a multi-step pipeline: lowercasing → URL removal → HTML cleanup → emoji removal → punctuation removal → stopword removal → lemmatization → whitespace normalization.

### Sentiment Labeling
Star ratings are mapped to three sentiment classes:

| Stars | Label |
|-------|-------|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

### Processed Dataset Fields
Each processed record includes `review_text`, `rating_numeric`, `sentiment`, `review_length`, `clean_text`, and `sentiment_score`.

### EDA Plots
Five plots are generated automatically:
- `rating_distribution.png`
- `sentiment_distribution.png`
- `frequent_words.png`
- `review_length_analysis.png`
- `word_cloud.png`

### Model Training
TF-IDF features (unigrams + bigrams, up to 12,000 terms) are extracted from `clean_text`. Four classifiers are trained and compared:
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- XGBoost

### Business Insights
The insights module runs keyword extraction, LDA topic modeling, and generates natural-language summaries of key patterns in the review corpus.

### Streamlit App
Four interactive pages: **Dashboard**, **Analytics**, **Prediction**, and **Insights**.

---

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install -e .
```

---

## Usage

**Generate profiling reports, processed data, and plots (skip training):**
```bash
python run_pipeline.py --skip-training
```

**Train and save the best model:**
```bash
python -m src.training.train_model
```

**Launch the Streamlit app:**
```bash
streamlit run app.py
```

**Run tests:**
```bash
python -m unittest discover -s tests
```

---

## Deployment

### Render

For a full walkthrough see [DEPLOY_ON_RENDER.md]. The repo includes a `render.yaml` for one-click infrastructure-as-code deployment.

| Setting | Value |
|---------|-------|
| Build command | `pip install -r requirements.txt` |
| Start command | `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT` |

### Docker

```bash
docker build -t customer-feedback-analysis .
docker run -p 8501:8501 customer-feedback-analysis
```

Then open `http://localhost:8501` in your browser.

---

## Roadmap

- [ ] Transformer-based embeddings for semantic search and richer sentiment modeling
- [ ] Experiment tracking with MLflow
- [ ] Data drift monitoring for deployed review streams
- [ ] Authenticated admin workflows for on-demand model retraining