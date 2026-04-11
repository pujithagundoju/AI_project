import re
import string
from typing import Iterable

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOPWORDS = set(ENGLISH_STOP_WORDS)
STEMMER = PorterStemmer()


def to_lower(text: str) -> str:
    """Convert text to lowercase."""
    return str(text).lower()


def remove_punctuation_and_digits(text: str) -> str:
    """Remove punctuation and digits; keep words and spaces."""
    text = str(text).translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str, stopwords: Iterable[str] | None = None) -> str:
    """Remove stopwords from text."""
    active_stopwords = set(stopwords) if stopwords is not None else STOPWORDS
    tokens = [token for token in str(text).split() if token not in active_stopwords]
    return " ".join(tokens)


def apply_stemming(text: str) -> str:
    """Apply Porter stemming token-wise."""
    return " ".join(STEMMER.stem(token) for token in str(text).split())


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for one text value."""
    if text is None:
        return ""

    text = to_lower(text)
    text = remove_punctuation_and_digits(text)
    text = remove_stopwords(text)
    text = apply_stemming(text)
    return text


def preprocess_series(text_series: pd.Series) -> pd.Series:
    """Apply preprocessing to a pandas Series."""
    return text_series.fillna("").astype(str).apply(preprocess_text)


def preprocess_dataframe(df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Return a copy with processed text column."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    processed = df.copy()
    processed[text_column] = preprocess_series(processed[text_column])
    return processed
