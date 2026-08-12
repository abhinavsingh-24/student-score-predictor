from pathlib import Path
import numpy as np
import pandas as pd


def generate_student_scores(n_students: int = 5000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    hours_studied = np.clip(rng.normal(loc=6, scale=2.5, size=n_students), 0, 15)
    attendance_pct = np.clip(rng.normal(loc=88, scale=8, size=n_students), 50, 100)
    previous_score = np.clip(rng.normal(loc=70, scale=15, size=n_students), 20, 100)
    assignments_completed = np.clip(rng.normal(loc=8, scale=2.5, size=n_students), 0, 12)
    sleep_hours = np.clip(rng.normal(loc=7, scale=1.2, size=n_students), 4, 10)
    internet_usage = np.clip(rng.normal(loc=3, scale=1.5, size=n_students), 0, 12)

    extracurricular_activity = rng.choice(["low", "medium", "high"], size=n_students, p=[0.4, 0.4, 0.2])
    family_support = rng.choice(["low", "medium", "high"], size=n_students, p=[0.25, 0.5, 0.25])

    base_score = (
        2.5 * hours_studied
        + 0.3 * attendance_pct
        + 0.4 * previous_score
        + 2.0 * assignments_completed
        + 1.0 * sleep_hours
    )
    activity_bonus = np.where(extracurricular_activity == "high", 3, np.where(extracurricular_activity == "medium", 1, 0))
    support_bonus = np.where(family_support == "high", 4, np.where(family_support == "medium", 2, 0))
    internet_penalty = np.where(internet_usage > 8, (internet_usage - 8) * 1.5, 0)

    noise = rng.normal(loc=0, scale=6, size=n_students)
    score = base_score + activity_bonus + support_bonus - internet_penalty + noise
    score = np.clip(score, 0, 100)

    df = pd.DataFrame({
        "hours_studied": np.round(hours_studied, 1),
        "attendance_pct": np.round(attendance_pct, 1),
        "previous_score": np.round(previous_score, 1),
        "assignments_completed": np.round(assignments_completed).astype(int),
        "sleep_hours": np.round(sleep_hours, 1),
        "internet_usage": np.round(internet_usage, 1),
        "extracurricular_activity": extracurricular_activity,
        "family_support": family_support,
        "score": np.round(score, 1),
    })
    return df


def save_dataset(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    df = generate_student_scores(5000, random_state=42)
    output_path = Path("data/student_scores_large.csv")
    save_dataset(df, output_path)
    print(f"Saved synthetic dataset to {output_path}")


if __name__ == "__main__":
    main()
