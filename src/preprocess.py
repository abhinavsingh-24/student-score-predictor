from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    """Load raw student performance data from a CSV file."""
    return pd.read_csv(path)


def add_study_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """Add study efficiency as score divided by hours studied.

    This function handles zero and missing study hours safely by setting
    study_efficiency to 0 when hours_studied is zero or missing.
    """
    df = df.copy()
    hours = df.get("hours_studied", pd.Series(dtype=float))
    score = df.get("score", pd.Series(dtype=float))
    df["study_efficiency"] = np.where(
        hours > 0,
        score / hours,
        0.0,
    )
    return df


def add_attendance_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add attendance categories based on attendance percentage."""
    df = df.copy()

    def bucket_attendance(value):
        if pd.isna(value):
            return "unknown"
        if value >= 90:
            return "high"
        if value >= 70:
            return "medium"
        return "low"

    df["attendance_category"] = df["attendance_pct"].apply(bucket_attendance)
    return df


def add_assignment_completion_rate(
    df: pd.DataFrame, max_assignments: Optional[int] = None
) -> pd.DataFrame:
    """Add assignment completion rate as a fraction of total possible assignments."""
    df = df.copy()
    assignments = df.get("assignments_completed", pd.Series(dtype=float))
    if max_assignments is None:
        max_value = assignments.max()
    else:
        max_value = max_assignments

    if pd.isna(max_value) or max_value <= 0:
        df["assignment_completion_rate"] = 0.0
    else:
        df["assignment_completion_rate"] = assignments / float(max_value)
    return df


def encode_categorical_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Encode categorical columns using one-hot encoding."""
    df = df.copy()
    if columns is None:
        columns = [col for col in df.columns if df[col].dtype == object or df[col].dtype.name == "category"]
    if not columns:
        return df
    df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)
    return df


def preprocess_dataset(
    input_path: Path,
    output_path: Path,
    encode_categorical: bool = True,
    max_assignments: Optional[int] = None,
) -> pd.DataFrame:
    """Load raw data, engineer features, optionally encode categories, and save processed data."""
    df = load_data(input_path)
    df = add_study_efficiency(df)
    df = add_attendance_category(df)
    df = add_assignment_completion_rate(df, max_assignments=max_assignments)
    if encode_categorical:
        df = encode_categorical_features(df, columns=["attendance_category"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    raw_path = Path("data/student_scores.csv")
    processed_path = Path("data/processed_student_scores.csv")
    preprocess_dataset(raw_path, processed_path)
    print(f"Saved processed dataset to {processed_path}")
