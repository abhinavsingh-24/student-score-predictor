from pathlib import Path
import sys
from typing import List

import pandas as pd

src_root = Path(__file__).resolve()
while src_root.name != "src" and src_root.parent != src_root:
    src_root = src_root.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create and return a dataframe with additional engineered features."""
    df2 = df.copy()
    if "hours_studied" in df2.columns:
        df2["hours_squared"] = df2["hours_studied"] ** 2
    if "assignments_completed" in df2.columns and "hours_studied" in df2.columns:
        df2["assignments_per_hour"] = df2["assignments_completed"] / (df2["hours_studied"] + 1e-6)
    # example interaction
    if set(["attendance_pct", "previous_score"]).issubset(df2.columns):
        df2["att_prev_interaction"] = df2["attendance_pct"] * df2["previous_score"] / 100.0
    logger.info("Engineered features: %s", [c for c in df2.columns if c not in df.columns])
    return df2


def select_features(df: pd.DataFrame, candidates: List[str]) -> List[str]:
    """Return intersection of candidates present in df."""
    chosen = [c for c in candidates if c in df.columns]
    logger.info("Selected features: %s", chosen)
    return chosen
