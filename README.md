# AI Customer Feedback Analysis

Production-ready NLP project for analyzing Amazon customer reviews. It detects the review and rating columns automatically, cleans review text, creates sentiment labels from star ratings, produces EDA reports, trains multiple ML models, and serves insights through a Streamlit application.

## Dataset

The repository includes `data/Amazon_Reviews.csv`. The loader supports review columns named `reviewText`, `review_body`, `Review Text`, `review`, or similar, and rating columns named `overall`, `star_rating`, `rating`, or similar. For this dataset, the detected columns are:

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

- Data profiling: missing values, duplicates, outliers, class imbalance, review length.
- Preprocessing: lowercasing, URL removal, HTML cleanup, emoji removal, punctuation removal, stopword removal, lemmatization, whitespace normalization.
- Sentiment labels: 1-2 stars Negative, 3 stars Neutral, 4-5 stars Positive.
- EDA: rating distribution, sentiment distribution, frequent words, word cloud, review length.
- Feature engineering: TF-IDF, n-grams, review length, simple sentiment scores.
- Model comparison: Logistic Regression, Multinomial Naive Bayes, Random Forest, XGBoost when available.
- Advanced NLP: prediction, keyword extraction, LDA topic modeling, business insight generation, review-level summaries.
- Streamlit app: Dashboard, Analytics, Prediction, Insights.

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

Streamlit Community Cloud:

1. Push this repository to GitHub.
2. Set `app.py` as the entry point.
3. Ensure `requirements.txt` and `runtime.txt` are committed.
4. Run `python run_pipeline.py` locally first if you want reports and model artifacts pre-generated.

Render:

1. Create a new Web Service from the GitHub repository.
2. Use `pip install -r requirements.txt` as the build command.
3. Use `streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT` as the start command.

Docker:

```bash
docker build -t customer-feedback-analysis .
docker run -p 8501:8501 customer-feedback-analysis
```

## Future Enhancements

- Add transformer embeddings for semantic search and better sentiment modeling.
- Add experiment tracking with MLflow.
- Add drift monitoring for deployed review streams.
- Add authenticated admin workflows for model retraining.
