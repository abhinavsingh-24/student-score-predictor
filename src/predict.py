import argparse
from pathlib import Path

from joblib import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_student_score_model.joblib")
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument("--attendance", type=float, required=True)
    parser.add_argument("--previous", type=float, required=True)
    parser.add_argument("--assignments", type=float, required=True)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found at {model_path}. Run src/train.py first.")

    model = load(model_path)

    features = [[args.hours, args.attendance, args.previous, args.assignments]]
    pred = model.predict(features)[0]
    print(f"Predicted score: {pred:.1f}")


if __name__ == "__main__":
    main()
