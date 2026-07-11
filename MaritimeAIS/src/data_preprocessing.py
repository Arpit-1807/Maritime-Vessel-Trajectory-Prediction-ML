"""
==========================================================
Module: data_processing.py

Description:
------------
Utilities for loading, validating, cleaning, and reporting
Automatic Identification System (AIS) vessel trajectory data.

Author: Arpit Mukherjee
Project:
Maritime Vessel Trajectory Prediction using Machine Learning
==========================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np

## Data Loading Function
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load AIS dataset from CSV.

    Parameters
    ----------
    file_path : str
        Path to CSV file.

    Returns
    -------
    pd.DataFrame
    """

    df = pd.read_csv(
        file_path,
        low_memory=False
    )

    print(f"Dataset Loaded Successfully")

    print(f"Rows : {len(df):,}")

    print(f"Columns : {df.shape[1]}")

    return df

## Coordinate Validation
def validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid latitude and longitude values.
    """

    df = df.copy()

    df = df[
        df["LAT"].between(-90, 90)
    ]

    df = df[
        df["LON"].between(-180, 180)
    ]

    return df

## Duplicate Removal
def remove_duplicates(
    df: pd.DataFrame
) -> tuple:

    """
    Remove duplicate rows and duplicate MMSI + Timestamp.
    """

    rows_before = len(df)

    duplicate_rows = df.duplicated().sum()

    df = df.drop_duplicates()

    duplicate_mmsi = (
        df
        .duplicated(
            subset=[
                "MMSI",
                "BaseDateTime"
            ]
        )
        .sum()
    )

    df = df.drop_duplicates(
        subset=[
            "MMSI",
            "BaseDateTime"
        ]
    )

    rows_after = len(df)

    report = {

        "Rows Before": rows_before,

        "Rows After": rows_after,

        "Rows Removed": rows_before - rows_after,

        "Duplicate Rows Removed": duplicate_rows,

        "Duplicate MMSI + Timestamp": duplicate_mmsi

    }

    return df, report

## Cleaning Pipeline
def clean_ais_data(
    df: pd.DataFrame
):

    """
    Complete AIS Cleaning Pipeline.
    """

    df = validate_coordinates(df)

    df, report = remove_duplicates(df)

    return df, report

## Cleaning Report
def generate_cleaning_report(
    report: dict
):

    """
    Convert report dictionary to DataFrame.
    """

    report_df = pd.DataFrame(

        report.items(),

        columns=[
            "Metric",
            "Value"
        ]

    )

    return report_df

## Save Report
def save_report(
    report_df: pd.DataFrame,
    output_path: str
):

    """
    Save report as CSV.
    """

    Path(output_path).parent.mkdir(

        parents=True,

        exist_ok=True

    )

    report_df.to_csv(

        output_path,

        index=False

    )

    print("Cleaning report saved.")

## Example Usage
if __name__ == "__main__":

    DATA_PATH = "data/raw/AIS_Data.csv"

    df = load_data(DATA_PATH)

    cleaned_df, report = clean_ais_data(df)

    report_df = generate_cleaning_report(report)

    print(report_df)