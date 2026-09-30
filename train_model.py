"""
train_model.py
AeroFare Intelligence - Machine Learning Model Training & Benchmark Pipeline

Trains and compares multiple regression algorithms:
  1. Linear Regression (Baseline)
  2. Ridge Regression (L2 Regularized)
  3. Decision Tree Regressor
  4. Random Forest Regressor

Builds an end-to-end Scikit-Learn Pipeline with ColumnTransformer,
evaluates performance metrics (R2, MAE, RMSE), and serializes the best
model and benchmark metrics to the `models/` directory.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from dataset_loader import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "flight_price_pipeline.pkl")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Features to use for modeling
CATEGORICAL_FEATURES = [
    "airline",
    "source_city",
    "departure_time",
    "stops",
    "arrival_time",
    "destination_city",
    "class",
]
NUMERICAL_FEATURES = ["duration", "days_left"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "price"


def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer to encode categorical and scale numerical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
        ]
    )
    return preprocessor


def train_and_benchmark(sample_limit: int = 50000):
    """
    Loads dataset, executes train-test split, trains candidates,
    evaluates benchmarks, and exports artifacts.
    """
    print("=" * 60)
    print("   AeroFare Intelligence - Model Training & Benchmark")
    print("=" * 60)

    # 1. Load Data
    df, source_desc = load_dataset()
    print(f"[+] Loaded data from: {source_desc}")
    print(f"[+] Total records available: {len(df):,}")

    # If full Kaggle dataset (~300k rows), optionally sample for responsive training
    if len(df) > sample_limit:
        print(f"[i] Sampling {sample_limit:,} records for fast, balanced training...")
        df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # 2. Train-Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[+] Training samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 3. Candidate Models
    candidates = {
        "Linear Regression (Baseline)": LinearRegression(),
        "Ridge Regression (Alpha=1.0)": Ridge(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=12, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, max_depth=14, min_samples_split=4, random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_model_name = None
    best_r2 = -float("inf")
    trained_pipelines = {}

    preprocessor = build_preprocessor()

    print("\n--- Evaluating Candidate Algorithms ---")
    for name, regressor in candidates.items():
        print(f"\n[>] Training {name}...")
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("regressor", regressor),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        r2 = float(r2_score(y_test, y_pred))
        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape = float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)

        print(f"    R^2 Score: {r2:.4f}")
        print(f"    MAE      : INR {mae:,.2f}")
        print(f"    RMSE     : INR {rmse:,.2f}")
        print(f"    MAPE     : {mape:.2f}%")

        results[name] = {
            "r2_score": round(r2, 4),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
        }
        trained_pipelines[name] = pipeline

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name

    print("\n" + "=" * 60)
    print(f"[*] Best Model Selected: {best_model_name} (R^2 = {best_r2:.4f})")
    print("=" * 60)

    # 4. Feature Importance Extraction
    best_pipeline = trained_pipelines[best_model_name]
    feature_importances = []
    try:
        prep = best_pipeline.named_steps["preprocessor"]
        cat_encoder = prep.named_transformers_["cat"]
        cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_encoded_features = cat_features + NUMERICAL_FEATURES

        reg = best_pipeline.named_steps["regressor"]
        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
            feature_imp_df = (
                pd.DataFrame({"feature": all_encoded_features, "importance": importances})
                .sort_values(by="importance", ascending=False)
                .head(10)
            )
            feature_importances = feature_imp_df.to_dict(orient="records")
        elif hasattr(reg, "coef_"):
            coefs = np.abs(reg.coef_)
            feature_imp_df = (
                pd.DataFrame({"feature": all_encoded_features, "importance": coefs})
                .sort_values(by="importance", ascending=False)
                .head(10)
            )
            feature_importances = feature_imp_df.to_dict(orient="records")
    except Exception as e:
        print(f"[!] Note on feature importances: {e}")

    # 5. Retrain Best Pipeline on Full Dataset
    print(f"\n[+] Retraining final {best_model_name} on the full dataset...")
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", candidates[best_model_name]),
        ]
    )
    final_pipeline.fit(X, y)

    # 6. Save Artifacts
    joblib.dump(final_pipeline, MODEL_ARTIFACT_PATH)
    print(f"[OK] Best model pipeline saved to: {MODEL_ARTIFACT_PATH}")

    metadata = {
        "best_model": best_model_name,
        "dataset_source": source_desc,
        "total_records": len(df),
        "benchmark_metrics": results,
        "feature_importances": feature_importances,
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
        },
    }

    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Benchmark metrics saved to: {METRICS_JSON_PATH}")

    return metadata


if __name__ == "__main__":
    train_and_benchmark()
