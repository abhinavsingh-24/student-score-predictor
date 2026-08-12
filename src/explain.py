from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_PATH = Path("models/best_student_score_model.joblib")
FEATURES = ["hours_studied", "attendance_pct", "previous_score", "assignments_completed"]
OUTPUT_PATH = Path("reports/feature_importance.png")


def load_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}")
    return joblib.load(path)


def get_feature_importance(model) -> pd.DataFrame:
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        estimator = model.named_steps["model"]
    else:
        estimator = model

    if hasattr(estimator, "feature_importances_"):
        importance_values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance_values = abs(estimator.coef_)
    else:
        raise ValueError("Model does not expose feature importances or coefficients")

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": importance_values,
    }).sort_values(by="importance", ascending=False)
    return importance_df


def save_importance_chart(importance_df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
    plt.title("Feature Importance for Student Score Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return output_path


def explain_top_features(importance_df: pd.DataFrame, top_n: int = 3) -> str:
    top_features = importance_df.head(top_n)
    explanations = [
        f"{row.feature} ({row.importance:.3f})" for _, row in top_features.iterrows()
    ]
    if len(explanations) == 1:
        return f"The prediction is most influenced by {explanations[0]}."
    if len(explanations) == 2:
        return f"The prediction is most influenced by {explanations[0]} and {explanations[1]}."
    return (
        "The prediction is most influenced by "
        + ", ".join(explanations[:-1])
        + f", and {explanations[-1]}."
    )


def main():
    model = load_model(MODEL_PATH)
    importance_df = get_feature_importance(model)
    chart_path = save_importance_chart(importance_df, OUTPUT_PATH)

    print("Feature importance rankings:")
    print(importance_df.to_string(index=False))
    print(f"Saved feature importance chart to {chart_path}")
    print(explain_top_features(importance_df))


if __name__ == "__main__":
    main()
