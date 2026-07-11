"""
==========================================================
Module: anomaly_detection.py

Description:
------------
Detect operational anomalies in AIS vessel trajectory data.

Implemented Anomalies
---------------------
1. High-Speed Vessels
2. Sudden Acceleration
3. Sudden Heading Change
4. Long AIS Transmission Gap
5. GPS Position Jump

Author: Arpit Mukherjee
Project:
Maritime Vessel Trajectory Prediction using Machine Learning
==========================================================
"""

import pandas as pd
import numpy as np

## High Speed Detection

def detect_high_speed(
    df: pd.DataFrame,
    speed_limit: float = 30
) -> pd.DataFrame:
    """
    Detect vessels exceeding the speed threshold.

    Parameters
    ----------
    speed_limit : float
        Speed Over Ground (knots).
    """

    anomalies = df[df["SOG"] > speed_limit].copy()

    anomalies["Anomaly"] = "High Speed"

    return anomalies

## Sudden Acceleration

def detect_sudden_acceleration(
    df: pd.DataFrame,
    threshold: float = 10
) -> pd.DataFrame:
    """
    Detect sudden increases in vessel speed.
    """

    temp = df.copy()

    temp["Speed_Change"] = (
        temp.groupby("MMSI")["SOG"]
        .diff()
        .fillna(0)
    )

    anomalies = temp[
        temp["Speed_Change"] > threshold
    ].copy()

    anomalies["Anomaly"] = "Sudden Acceleration"

    return anomalies

## Sudden Heading Change

def detect_heading_change(
    df: pd.DataFrame,
    threshold: float = 90
) -> pd.DataFrame:
    """
    Detect abrupt heading changes.
    """

    temp = df.copy()

    temp["Heading_Change"] = (
        temp.groupby("MMSI")["Heading"]
        .diff()
        .abs()
        .fillna(0)
    )

    anomalies = temp[
        temp["Heading_Change"] > threshold
    ].copy()

    anomalies["Anomaly"] = "Heading Change"

    return anomalies

## Long AIS Gap

def detect_long_gap(
    df: pd.DataFrame,
    minutes: int = 60
) -> pd.DataFrame:
    """
    Detect long AIS transmission gaps.
    """

    anomalies = df[
        df["Time_Diff_Min"] > minutes
    ].copy()

    anomalies["Anomaly"] = "Long AIS Gap"

    return anomalies

## GPS Position Jump

def detect_position_jump(
    df: pd.DataFrame,
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Detect unusually large position changes
    between consecutive AIS messages.
    """

    temp = df.copy()

    temp["Prev_LAT"] = (
        temp.groupby("MMSI")["LAT"]
        .shift(1)
    )

    temp["Prev_LON"] = (
        temp.groupby("MMSI")["LON"]
        .shift(1)
    )

    temp["Jump_Distance"] = np.sqrt(
        (temp["LAT"] - temp["Prev_LAT"]) ** 2 +
        (temp["LON"] - temp["Prev_LON"]) ** 2
    )

    anomalies = temp[
        temp["Jump_Distance"] > threshold
    ].copy()

    anomalies["Anomaly"] = "GPS Position Jump"

    return anomalies

## Combine All Anomalies

def detect_all_anomalies(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Run every anomaly detector and combine
    the results.
    """

    results = [

        detect_high_speed(df),

        detect_sudden_acceleration(df),

        detect_heading_change(df),

        detect_long_gap(df),

        detect_position_jump(df)

    ]

    anomalies = pd.concat(
        results,
        ignore_index=True
    )

    return anomalies

## Summary Report

def anomaly_summary(
    anomalies: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate anomaly counts.
    """

    summary = (
        anomalies["Anomaly"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        "Anomaly Type",
        "Count"
    ]

    return summary

## Save Report

def save_anomaly_report(
    summary: pd.DataFrame,
    output_path: str
):
    """
    Save anomaly summary report.
    """

    summary.to_csv(
        output_path,
        index=False
    )

    print("Anomaly report saved.")

## Example Usage

if __name__ == "__main__":

    print(
        "Anomaly Detection Module"
    )