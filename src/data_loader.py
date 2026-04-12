from pathlib import Path # handling file paths in clean and cross-platform way
from typing import Tuple # typing hint for functions that return multiple values

import pandas as pd # for data manipulation and analysis


REQUIRED_COLUMNS = ("text", "label") # Define required columns for the dataset to ensure consistency and avoid errors during processing


def _validate_columns(df: pd.DataFrame, file_path: Path) -> None: # function donot rerturn anything it only checks validations
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
#     missing = []
# for col in REQUIRED_COLUMNS:
#     if col not in df.columns:
#         missing.append(col)
    if missing: # if missing list is not empty then raise error (alteast one column is missing)
        raise ValueError(
            f"Missing required column(s) {missing} in file: {file_path}"
        )


def load_csv(file_path: str | Path) -> pd.DataFrame: # input str or path, output dataframe
    """Load a CSV file and validate expected resume dataset columns."""
    path = Path(file_path) #convert string to path object i.e "Dataset/train.csv" to Path("Dataset/train.csv")
    if not path.exists():# checkfile path exist !
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path) #read csv file and store in dataframe
    _validate_columns(df, path)
    return df


def load_train_data(dataset_dir: str | Path = "Dataset") -> pd.DataFrame: # dataset default directory 
    """Load and return train.csv from the dataset directory."""
    dataset_path = Path(dataset_dir)
    return load_csv(dataset_path / "train.csv")


def load_test_data(dataset_dir: str | Path = "Dataset") -> pd.DataFrame:
    """Load and return test.csv from the dataset directory."""
    dataset_path = Path(dataset_dir)
    return load_csv(dataset_path / "test.csv")


def load_train_test_data(dataset_dir: str | Path = "Dataset") -> Tuple[pd.DataFrame, pd.DataFrame]: # returns 2 dataframes 
    """Load and return both train and test datasets."""
    train_df = load_train_data(dataset_dir)
    test_df = load_test_data(dataset_dir)
    return train_df, test_df
