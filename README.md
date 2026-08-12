# Student Score Predictor

Simple Python project that predicts student scores using Pandas and scikit-learn.

Getting started

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Train a model:

```bash
python src/train.py --data data/student_scores.csv --output models/best_student_score_model.joblib
```

3. Run the Streamlit dashboard:

```bash
streamlit run src/app.py
```

4. Predict from CLI:

```bash
python src/predict.py --hours 5 --attendance 90 --previous 72 --assignments 8
```

Files

- [README.md](README.md)
- [requirements.txt](requirements.txt)
- [data/student_scores.csv](data/student_scores.csv)
- [src/train.py](src/train.py)
- [src/predict.py](src/predict.py)
- [models/.gitkeep](models/.gitkeep)
