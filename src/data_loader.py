from pathlib import Path
from typing import Tuple

import pandas as pd


REQUIRED_COLUMNS = ("text", "label")


def _validate_columns(df: pd.DataFrame, file_path: Path) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s) {missing} in file: {file_path}"
        )


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file and validate expected resume dataset columns."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    _validate_columns(df, path)
    return df


def load_train_data(dataset_dir: str | Path = "Dataset") -> pd.DataFrame:
    """Load and return train.csv from the dataset directory."""
    dataset_path = Path(dataset_dir)
    return load_csv(dataset_path / "train.csv")


def load_test_data(dataset_dir: str | Path = "Dataset") -> pd.DataFrame:
    """Load and return test.csv from the dataset directory."""
    dataset_path = Path(dataset_dir)
    return load_csv(dataset_path / "test.csv")


def load_train_test_data(dataset_dir: str | Path = "Dataset") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and return both train and test datasets."""
    train_df = load_train_data(dataset_dir)
    test_df = load_test_data(dataset_dir)
    return train_df, test_df
