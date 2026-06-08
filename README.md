# AI Customer Feedback Analysis

Production-ready NLP project for analyzing Amazon customer reviews. It detects the review and rating columns automatically, cleans review text, creates sentiment labels from star ratings, produces EDA reports, trains multiple ML models, and serves insights through a Streamlit application.

## Dataset

The repository includes `data/Amazon_Reviews.csv`. The loader supports review columns named `reviewText`, `Rating`, `Review Title`.
For this dataset, the detected columns are:

- Text: `Review Text`
- Rating: `Rating`

## Architecture

```text
customer-feedback-analysis/
|-- app.py
|-- assets/
|   |-- plots/
|   `-- reports/
|-- data/
|   `-- Amazon_Reviews.csv
|-- models/
|-- notebooks/
|-- src/
|   |-- components/
|   |-- exceptions.py
|   |-- inference/
|   |-- logger.py
|   |-- pipeline/
|   |-- preprocessing/
|   |-- training/
|   |-- visualization/
|   |-- insights/
|   `-- utils/
|-- tests/
|-- requirements.txt
|-- runtime.txt
|-- Dockerfile
`-- README.md
```

## Features

- Data profiling: `missing_values`, `duplicate_rows`, `rating_distribution`, `sentiment_distribution`, `review_length`, and `review_length_outliers_iqr`.
- Preprocessing: lowercasing, URL removal, HTML cleanup, emoji removal, punctuation removal, stopword removal, lightweight lemmatization, and whitespace normalization.
- Sentiment labels: `Negative` for 1-2 stars, `Neutral` for 3 stars, and `Positive` for 4-5 stars.
- Generated dataset fields: `review_text`, `rating_numeric`, `sentiment`, `review_length`, `clean_text`, and `sentiment_score`.
- EDA plots: `rating_distribution.png`, `sentiment_distribution.png`, `frequent_words.png`, `review_length_analysis.png`, and `word_cloud.png`.
- Model features: TF-IDF unigram and bigram features from `clean_text`, with up to 12,000 terms.
- Model comparison: `Logistic Regression`, `Multinomial Naive Bayes`, `Random Forest`, and `XGBoost` when available.
- Insights: keyword extraction, LDA topic modeling, and generated business insight summaries.
- Streamlit pages: `Dashboard`, `Analytics`, `Prediction`, and `Insights`.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Usage

Generate profiling reports, processed data, and plots:

```bash
python run_pipeline.py --skip-training
```

Train and save the best model:

```bash
python -m src.training.train_model
```

Run the Streamlit app:

```bash
streamlit run app.py
```

Run tests:

```bash
python -m unittest discover -s tests
```

## Deployment

### Render (Recommended)

For detailed deployment instructions, see [DEPLOY_ON_RENDER.md](DEPLOY_ON_RENDER.md).

Quick start:
1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Create a new Web Service from this repository
4. Select **Docker** as runtime
5. Deploy automatically or via `render.yaml`

### Streamlit Community Cloud

1. Push this repository to GitHub.
2. Set `app.py` as the entry point.
3. Ensure `requirements.txt` and `runtime.txt` are committed.
4. Run `python run_pipeline.py` locally first if you want reports and model artifacts pre-generated.

### Docker (Local or Self-Hosted)

```bash
docker build -t customer-feedback-analysis .
docker run -p 8501:8501 customer-feedback-analysis
```

## Future Enhancements

- Add transformer embeddings for semantic search and better sentiment modeling.
- Add experiment tracking with MLflow.
- Add drift monitoring for deployed review streams.
- Add authenticated admin workflows for model retraining.
