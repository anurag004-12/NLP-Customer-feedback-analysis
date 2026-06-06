import time
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.svm import LinearSVC

DATA_DIR = Path('data')
OUT_DIR = Path('assets/reports')
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FRAC = 0.1

def load_data():
    X = np.load(DATA_DIR / 'combined_features.npy')
    y = np.load(DATA_DIR / 'labels.npy', allow_pickle=True)
    return X, y

def to_numeric(y):
    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
    return pd.Series(y).map(label_map).values

def time_model(name, model, X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    t = time.perf_counter() - t0
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return {'Model': name, 'Train Accuracy': train_score, 'Test Accuracy': test_score, 'Train Time (s)': round(t, 2), 'Overfit Gap': train_score - test_score}

def main():
    X, y = load_data()
    y_num = to_numeric(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_num, test_size=0.2, random_state=42, stratify=y_num)

    if SAMPLE_FRAC < 1.0:
        n = max(1, int(len(X_train) * SAMPLE_FRAC))
        X_train = X_train[:n]
        y_train = y_train[:n]
        print(f'Using SAMPLE_FRAC={SAMPLE_FRAC}: Train set reduced to {X_train.shape}')

    results = []

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    results.append(time_model('Logistic Regression', lr, X_train, y_train, X_test, y_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    results.append(time_model('Random Forest', rf, X_train, y_train, X_test, y_test))

    # HistGradientBoosting
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=5, random_state=42)
    results.append(time_model('Gradient Boosting', gb, X_train, y_train, X_test, y_test))

    # Linear SVM
    svm = LinearSVC(max_iter=5000, random_state=42)
    results.append(time_model('SVM', svm, X_train, y_train, X_test, y_test))

    out_path = OUT_DIR / 'model_report_0.1.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('Benchmark complete. Results written to', out_path)

if __name__ == '__main__':
    main()
