"""
==========================================================
Module: trajectory_prediction.py

Description:
------------
Machine Learning pipeline for Maritime Vessel
Trajectory Prediction.

Includes:
---------
1. Train/Test Split (GroupShuffleSplit)
2. Model Training
3. Prediction
4. Evaluation
5. Feature Importance
6. Model Save & Load
7. Prediction Export

Author: Arpit Mukherjee
Project:
Maritime Vessel Trajectory Prediction using Machine Learning
==========================================================
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupShuffleSplit

## Train/Test Split

def split_data(
    df: pd.DataFrame,
    feature_columns: list,
    target_columns: list,
    group_column: str = "MMSI",
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Perform vessel-level train/test split using GroupShuffleSplit.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            groups=df[group_column]
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]

    y_train = train_df[target_columns]
    y_test = test_df[target_columns]

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df
    )

## Train Model

def train_model(
    X_train,
    y_train,
    random_state=42
):
    """
    Train Random Forest model.
    """

    model = RandomForestRegressor(

        n_estimators=200,

        max_depth=20,

        random_state=random_state,

        n_jobs=-1

    )

    model.fit(
        X_train,
        y_train
    )

    return model

## Prediction

def predict(
    model,
    X_test
):
    """
    Generate predictions.
    """

    predictions = model.predict(
        X_test
    )

    prediction_df = pd.DataFrame(

        predictions,

        columns=[
            "Predicted_LAT",
            "Predicted_LON"
        ]

    )

    return prediction_df

## Evaluation

def evaluate_model(
    y_test,
    prediction_df
):
    """
    Evaluate prediction performance.
    """

    results = []

    targets = [

        ("Future_LAT", "Predicted_LAT"),

        ("Future_LON", "Predicted_LON")

    ]

    for actual, predicted in targets:

        mae = mean_absolute_error(
            y_test[actual],
            prediction_df[predicted]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test[actual],
                prediction_df[predicted]
            )
        )

        r2 = r2_score(
            y_test[actual],
            prediction_df[predicted]
        )

        results.append([

            actual,

            mae,

            rmse,

            r2

        ])

    metrics = pd.DataFrame(

        results,

        columns=[

            "Target",

            "MAE",

            "RMSE",

            "R²"

        ]

    )

    return metrics

## Feature Importance

def feature_importance(
    model,
    feature_columns
):
    """
    Generate feature importance.
    """

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        "Importance",

        ascending=False

    )

    return importance

## Save Model

def save_model(
    model,
    output_path
):
    """
    Save trained model.
    """

    Path(output_path).parent.mkdir(

        parents=True,

        exist_ok=True

    )

    joblib.dump(

        model,

        output_path

    )

    print("Model saved successfully.")

## Load Model

def load_model(
    model_path
):
    """
    Load trained model.
    """

    return joblib.load(
        model_path
    )

## Export Predictions

def export_predictions(
    predictions,
    output_path
):
    """
    Save prediction results.
    """

    predictions.to_csv(

        output_path,

        index=False

    )

    print("Predictions exported.")

## Full Pipeline

def run_pipeline(
    df,
    feature_columns,
    target_columns
):
    """
    Complete machine learning pipeline.
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df
    ) = split_data(

        df,

        feature_columns,

        target_columns

    )

    model = train_model(

        X_train,

        y_train

    )

    prediction_df = predict(

        model,

        X_test

    )

    metrics = evaluate_model(

        y_test,

        prediction_df

    )

    importance = feature_importance(

        model,

        feature_columns

    )

    return {

        "model": model,

        "metrics": metrics,

        "feature_importance": importance,

        "predictions": prediction_df,

        "X_test": X_test,

        "y_test": y_test,

        "train_df": train_df,

        "test_df": test_df

    }

## Example Usage

if __name__ == "__main__":

    print(
        "Trajectory Prediction Module"
    )