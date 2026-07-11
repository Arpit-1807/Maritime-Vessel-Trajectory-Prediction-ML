"""
==========================================================
Module: feature_engineering.py

Description:
------------
Feature engineering utilities for Maritime Vessel
Trajectory Prediction.

This module creates temporal, movement, historical,
and prediction target features from AIS data.

Author: Arpit Mukherjee
Project:
Maritime Vessel Trajectory Prediction using Machine Learning
==========================================================
"""

import pandas as pd
import numpy as np

## Sort Dataset

def sort_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort AIS data by vessel and timestamp.
    """

    df = df.copy()

    df["BaseDateTime"] = pd.to_datetime(
        df["BaseDateTime"]
    )

    df = (
        df
        .sort_values(
            ["MMSI", "BaseDateTime"]
        )
        .reset_index(drop=True)
    )

    return df

## Time Features 

def create_time_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create hour and month features.
    """

    df = df.copy()

    df["Hour"] = (
        df["BaseDateTime"]
        .dt.hour
    )

    df["Month"] = (
        df["BaseDateTime"]
        .dt.month
    )

    return df

## Time Difference

def create_time_difference(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate time difference (minutes)
    between consecutive AIS messages.
    """

    df = df.copy()

    df["Time_Diff_Min"] = (

        df.groupby("MMSI")["BaseDateTime"]

        .diff()

        .dt.total_seconds()

        / 60

    )

    df["Time_Diff_Min"] = (
        df["Time_Diff_Min"]
        .fillna(0)
    )

    return df

## Speed Category

def create_speed_category(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Categorize vessel speed.
    """

    df = df.copy()

    bins = [

        -1,

        0.5,

        5,

        15,

        30,

        np.inf

    ]

    labels = [

        "Stopped",

        "Slow",

        "Moderate",

        "Fast",

        "Very Fast"

    ]

    df["Speed_Category"] = pd.cut(

        df["SOG"],

        bins=bins,

        labels=labels

    )

    return df

## Movement Status

def create_movement_status(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create vessel movement status.
    """

    df = df.copy()

    df["Movement_Status"] = np.where(

        df["SOG"] < 0.5,

        "Stationary",

        "Moving"

    )

    return df

## Previous Features

def create_previous_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create lag features.
    """

    df = df.copy()

    df["Prev_SOG"] = (

        df.groupby("MMSI")["SOG"]

        .shift(1)

    )

    df["Prev_COG"] = (

        df.groupby("MMSI")["COG"]

        .shift(1)

    )

    df["Prev_Heading"] = (

        df.groupby("MMSI")["Heading"]

        .shift(1)

    )

    df[
        [

            "Prev_SOG",

            "Prev_COG",

            "Prev_Heading"

        ]

    ] = df[
        [

            "Prev_SOG",

            "Prev_COG",

            "Prev_Heading"

        ]

    ].fillna(0)

    return df

## Future Target 

def create_future_targets(
    df: pd.DataFrame,
    horizon: int = 5
) -> pd.DataFrame:
    """
    Create future prediction targets.

    Parameters
    ----------
    horizon : int

    Number of AIS messages ahead.
    """

    df = df.copy()

    df["Future_LAT"] = (

        df.groupby("MMSI")["LAT"]

        .shift(-horizon)

    )

    df["Future_LON"] = (

        df.groupby("MMSI")["LON"]

        .shift(-horizon)

    )

    df = df.dropna(

        subset=[

            "Future_LAT",

            "Future_LON"

        ]

    )

    return df

## Full Pipeline

def engineer_features(
    df: pd.DataFrame,
    horizon: int = 5
) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    """

    df = sort_dataset(df)

    df = create_time_features(df)

    df = create_time_difference(df)

    df = create_speed_category(df)

    df = create_movement_status(df)

    df = create_previous_features(df)

    df = create_future_targets(

        df,

        horizon=horizon

    )

    return df

## Feature List

def get_model_features():
    """
    Return model feature list.
    """

    return [

        "LAT",

        "LON",

        "SOG",

        "COG",

        "Heading",

        "Length",

        "Width",

        "Draft",

        "Hour",

        "Month",

        "Time_Diff_Min",

        "Prev_SOG",

        "Prev_COG",

        "Prev_Heading"

    ]

## Example 

if __name__ == "__main__":

    print(

        "Feature Engineering Module"

    )