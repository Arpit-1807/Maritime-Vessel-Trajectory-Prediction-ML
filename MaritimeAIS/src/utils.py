"""
==========================================================
Module: utils.py

Description:
------------
Reusable utility functions for the Maritime Vessel
Trajectory Prediction project.

Includes
--------
1. Directory Management
2. CSV Export
3. Model Metrics Display
4. Random Seed
5. Execution Timer
6. Logging
7. DataFrame Information

Author: Arpit Mukherjee
Project:
Maritime Vessel Trajectory Prediction using Machine Learning
==========================================================
"""

from pathlib import Path
from datetime import datetime
import logging
import random
import time

import numpy as np
import pandas as pd

## Create Directories

def create_directory(path: str):
    """
    Create directory if it does not exist.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )

## Save CSV

def save_csv(
    df: pd.DataFrame,
    output_path: str
):
    """
    Save DataFrame to CSV.
    """

    Path(output_path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(f"Saved: {output_path}")

## Display DataFrame Summary

def dataframe_summary(
    df: pd.DataFrame
):
    """
    Print basic DataFrame information.
    """

    print("=" * 50)

    print("DataFrame Summary")

    print("=" * 50)

    print(f"Rows : {len(df):,}")

    print(f"Columns : {df.shape[1]}")

    print()

    print(df.dtypes)

    print()

    print(df.head())

## Missing Value Report

def missing_value_report(
    df: pd.DataFrame
):
    """
    Generate missing value report.
    """

    report = (

        df.isnull()

        .sum()

        .reset_index()

    )

    report.columns = [

        "Column",

        "Missing Values"

    ]

    report["Percentage"] = (

        report["Missing Values"]

        / len(df)

        * 100

    ).round(2)

    return report.sort_values(

        "Missing Values",

        ascending=False

    )

## Set Random Seed

def set_random_seed(
    seed: int = 42
):
    """
    Set reproducible random seed.
    """

    random.seed(seed)

    np.random.seed(seed)

## Execution Timer

class Timer:
    """
    Measure execution time.

    Example:

    with Timer():

        code
    """

    def __enter__(self):

        self.start = time.time()

        return self

    def __exit__(

        self,

        exc_type,

        exc_value,

        traceback

    ):

        elapsed = (

            time.time()

            - self.start

        )

        print(

            f"Execution Time: {elapsed:.2f} seconds"

        )

## Configure Logger

def configure_logger():

    """
    Configure project logger.
    """

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s"

    )

    return logging.getLogger(__name__)

## Model Metrics Display

def print_metrics(
    metrics_df: pd.DataFrame
):
    """
    Display model metrics.
    """

    print()

    print("=" * 60)

    print("Model Performance")

    print("=" * 60)

    print(metrics_df)

    print("=" * 60)

## Timestamp

def current_timestamp():
    """
    Return formatted timestamp.
    """

    return datetime.now().strftime(

        "%Y-%m-%d %H:%M:%S"

    )

## Example Usage

if __name__ == "__main__":

    print(

        current_timestamp()

    )