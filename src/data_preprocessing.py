from pathlib import Path
import sys
from typing import Tuple

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

src_root = Path(__file__).resolve()
while src_root.name != "src" and src_root.parent != src_root:
    src_root = src_root.parent
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from utils import get_logger

logger = get_logger(__name__)


def load_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data: filling missing values and dropping duplicates")
    df = df.copy()
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=["number"]).columns
    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df


def preprocess_features(df: pd.DataFrame, features: list, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Preprocessing features: scaling numeric features")
    X = df[features].copy()
    y = df[target].copy()
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)
    return X_scaled, y


def save_processed(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved processed data to {path}")
