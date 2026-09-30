"""
predict.py
AeroFare Intelligence - Prediction and Smart Booking Advisory Module

Provides inference functions for:
  - Single flight ticket price prediction with confidence boundaries
  - Dynamic "Best Time to Buy" recommendation algorithm
  - Days-left sensitivity curve generation for route simulation
  - Batch CSV inference processing
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_ARTIFACT_PATH = os.path.join(MODELS_DIR, "flight_price_pipeline.pkl")
METRICS_JSON_PATH = os.path.join(MODELS_DIR, "model_metrics.json")

# Cached pipeline instance
_CACHED_PIPELINE = None


def get_pipeline():
    """Loads and caches the trained ML pipeline."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is None:
        if not os.path.exists(MODEL_ARTIFACT_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_ARTIFACT_PATH}. "
                "Please run `python train_model.py` first."
            )
        _CACHED_PIPELINE = joblib.load(MODEL_ARTIFACT_PATH)
    return _CACHED_PIPELINE


def get_model_metadata() -> dict:
    """Reads saved model benchmark metadata from JSON."""
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def predict_flight_price(
    airline: str,
    source_city: str,
    departure_time: str,
    stops: str,
    arrival_time: str,
    destination_city: str,
    flight_class: str,
    duration: float,
    days_left: int,
) -> dict:
    """
    Predicts flight fare and returns an enriched result object
    with confidence margins and fare category.
    """
    pipeline = get_pipeline()

    input_df = pd.DataFrame(
        [
            {
                "airline": airline,
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "class": flight_class,
                "duration": float(duration),
                "days_left": int(days_left),
            }
        ]
    )

    raw_prediction = pipeline.predict(input_df)[0]
    predicted_price = int(max(1500, round(raw_prediction)))

    # Estimate realistic confidence intervals (+/- 5%)
    lower_bound = int(max(1200, round(predicted_price * 0.95)))
    upper_bound = int(round(predicted_price * 1.05))

    # Determine fare tier
    if flight_class == "Economy":
        if predicted_price < 4500:
            fare_tier = "Budget Economy"
        elif predicted_price <= 9000:
            fare_tier = "Standard Economy"
        else:
            fare_tier = "Peak Surge Economy"
    else:
        if predicted_price < 45000:
            fare_tier = "Standard Business"
        else:
            fare_tier = "Peak Luxury Business"

    return {
        "predicted_price": predicted_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "fare_tier": fare_tier,
    }


def get_booking_advice(days_left: int, flight_class: str, predicted_price: float) -> dict:
    """
    Evaluates pricing volatility and booking lead time to deliver
    an actionable booking recommendation (Buy Now vs Wait).
    """
    if days_left <= 3:
        status = "CRITICAL SURGE"
        recommendation = "🚨 Buy Now Immediately"
        reason = (
            f"Departure is in only {days_left} day(s). Airlines enforce aggressive last-minute "
            "surge pricing in the final 72 hours. Ticket fares will rise or sell out rapidly."
        )
        badge_style = "error"
        action = "BUY_NOW"
    elif days_left <= 10:
        status = "HIGH DEMAND WINDOW"
        recommendation = "⚡ Strongly Recommended to Book"
        reason = (
            f"You are within {days_left} days of departure. Seat inventory is constricting, "
            "and fare algorithms begin progressive rate hikes."
        )
        badge_style = "warning"
        action = "BUY_NOW"
    elif 11 <= days_left <= 30:
        status = "OPTIMAL BOOKING SWEET SPOT"
        recommendation = "✅ Fair Market Price - Lock It In"
        reason = (
            f"{days_left} days out is the statistical 'Sweet Spot' for domestic travel. "
            "Prices are balanced before late-booking price surges begin."
        )
        badge_style = "success"
        action = "FAIR_PRICE"
    else:
        status = "EARLY BIRD STABLE"
        recommendation = "⏳ Safe to Monitor or Book Early"
        reason = (
            f"With {days_left} days remaining, price volatility is low. You can safely "
            "track routes or book early for maximum schedule flexibility."
        )
        badge_style = "info"
        action = "MONITOR"

    return {
        "status": status,
        "recommendation": recommendation,
        "reason": reason,
        "badge_style": badge_style,
        "action": action,
    }


def generate_price_curve(base_params: dict) -> pd.DataFrame:
    """
    Generates simulated price trajectory across 1 to 50 days left
    for the exact route and configuration specified by the user.
    """
    pipeline = get_pipeline()
    days_range = list(range(1, 51, 1))

    records = []
    for d in days_range:
        row = base_params.copy()
        row["days_left"] = d
        records.append(row)

    sim_df = pd.DataFrame(records)
    predictions = pipeline.predict(sim_df)

    curve_df = pd.DataFrame(
        {
            "Days Before Flight": days_range,
            "Estimated Price (INR)": [int(max(1500, p)) for p in predictions],
        }
    )
    return curve_df


def batch_predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs vector batch prediction for an uploaded dataset.
    """
    pipeline = get_pipeline()
    required_cols = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "class",
        "duration",
        "days_left",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Uploaded dataset is missing required columns: {missing}")

    preds = pipeline.predict(df[required_cols])
    result_df = df.copy()
    result_df["predicted_price_inr"] = [int(max(1500, round(p))) for p in preds]

    # Add booking recommendation column
    result_df["booking_recommendation"] = [
        get_booking_advice(row["days_left"], row["class"], p)["recommendation"]
        for row, p in zip(df.to_dict("records"), result_df["predicted_price_inr"])
    ]

    return result_df


if __name__ == "__main__":
    # Test sample inference
    res = predict_flight_price(
        airline="Indigo",
        source_city="Delhi",
        departure_time="Morning",
        stops="zero",
        arrival_time="Afternoon",
        destination_city="Mumbai",
        flight_class="Economy",
        duration=2.15,
        days_left=2,
    )
    print("Test Single Flight Prediction:")
    print(res)
    advice = get_booking_advice(2, "Economy", res["predicted_price"])
    print("\nTest Booking Advice:")
    print(f"Status: {advice['status']}")
    print(f"Action: {advice['action']}")
