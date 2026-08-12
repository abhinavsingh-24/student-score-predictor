from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the student scores dataset from a CSV file."""
    return pd.read_csv(path)


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Return key dataset overview metrics."""
    overview = pd.DataFrame(
        {
            "shape": [df.shape],
            "missing_values": [df.isnull().sum().sum()],
            "data_types": [df.dtypes.astype(str).to_dict()],
            "summary_statistics": [df.describe().to_dict()],
        }
    )
    return overview


def print_summary(df: pd.DataFrame):
    """Display dataset summary information."""
    print("Dataset shape:", df.shape)
    print("\nMissing values by column:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nSummary statistics:")
    print(df.describe())


def save_figures(df: pd.DataFrame, out_dir: Path):
    """Save EDA figures to the output directory."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Score distribution histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df["score"], kde=True, bins=12, color="#4c72b0")
    plt.title("Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    score_hist_path = out_dir / "score_distribution.png"
    plt.savefig(score_hist_path, bbox_inches="tight")
    plt.close()

    # Hours studied vs score scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["hours_studied"], y=df["score"], hue=df["attendance_pct"], palette="viridis")
    plt.title("Hours Studied vs Score")
    plt.xlabel("Hours Studied")
    plt.ylabel("Score")
    scatter_hours_path = out_dir / "hours_vs_score.png"
    plt.savefig(scatter_hours_path, bbox_inches="tight")
    plt.close()

    # Attendance vs score scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df["attendance_pct"], y=df["score"], hue=df["previous_score"], palette="rocket")
    plt.title("Attendance vs Score")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Score")
    scatter_attendance_path = out_dir / "attendance_vs_score.png"
    plt.savefig(scatter_attendance_path, bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    corr_path = out_dir / "correlation_heatmap.png"
    plt.savefig(corr_path, bbox_inches="tight")
    plt.close()

    return {
        "score_distribution": score_hist_path,
        "hours_vs_score": scatter_hours_path,
        "attendance_vs_score": scatter_attendance_path,
        "correlation_heatmap": corr_path,
    }


def main():
    data_path = Path("data/student_scores.csv")
    output_dir = Path("reports/figures")
    df = load_dataset(data_path)
    print_summary(df)
    save_figures(df, output_dir)
    print(f"Saved EDA figures to {output_dir}")


if __name__ == "__main__":
    main()
