from pathlib import Path
import sys
from typing import Dict, Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split

src_root = Path(__file__).resolve()
while src_root.name != "src" and src_root.parent != src_root:
    src_root = src_root.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils import get_logger

logger = get_logger(__name__)


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compare a few regressors using cross-validated MAE (lower is better)."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    results = []
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        # use neg_mean_absolute_error so we can use cross_val_score
        scores = cross_val_score(model, X, y, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
        mae_scores = -scores
        results.append({"model": name, "mae_mean": mae_scores.mean(), "mae_std": mae_scores.std()})
    return pd.DataFrame(results).sort_values(by="mae_mean")


def train_and_select(X: pd.DataFrame, y: pd.Series, output_path: Path) -> Dict[str, Any]:
    """Train candidate models and save the best one by MAE on a holdout set."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    candidates = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }
    best_name = None
    best_score = np.inf
    best_model = None
    metrics = {}
    for name, model in candidates.items():
        logger.info(f"Training {name}")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        metrics[name] = {"mae": float(mae)}
        logger.info(f"{name} MAE: {mae:.3f}")
        if mae < best_score:
            best_score = mae
            best_name = name
            best_model = model
    if best_model is None:
        raise RuntimeError("No model was trained successfully")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, output_path)
    logger.info(f"Saved best model ({best_name}) to {output_path}")
    return {"best_model": best_name, "best_mae": float(best_score), "metrics": metrics}
