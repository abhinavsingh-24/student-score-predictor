from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

src_root = Path(__file__).resolve()
while src_root.name != "src" and src_root.parent != src_root:
    src_root = src_root.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils import get_logger

logger = get_logger(__name__)


def permutation_importance_report(model, X: pd.DataFrame, y: pd.Series, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Computing permutation importance")
    res = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
    importances = pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    importances.plot.bar()
    plt.title("Permutation Importances")
    plt.ylabel("Mean importance")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved permutation importance plot to {out_path}")
    return out_path


def partial_dependence_placeholder():
    # A placeholder function: for heavier PDP plots, consider sklearn.inspection.partial_dependence
    logger.info("Partial dependence analysis can be added here (sklearn >= 0.24)")
    return None
