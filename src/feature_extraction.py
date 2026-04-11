from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfFeatureSet:
    vectorizer: TfidfVectorizer
    features: csr_matrix
    feature_names: list[str]


def create_tfidf_vectorizer(
    max_features: int | None = 5000,
    ngram_range: tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_df: float = 1.0,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for resume text."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
    )


def fit_tfidf_vectorizer(
    texts: Sequence[str] | pd.Series,
    **vectorizer_kwargs,
) -> TfidfFeatureSet:
    """Fit a TF-IDF vectorizer and return the matrix plus feature names."""
    vectorizer = create_tfidf_vectorizer(**vectorizer_kwargs)
    features = vectorizer.fit_transform(list(texts))
    feature_names = vectorizer.get_feature_names_out().tolist()
    return TfidfFeatureSet(vectorizer=vectorizer, features=features, feature_names=feature_names)


def transform_texts(
    vectorizer: TfidfVectorizer,
    texts: Sequence[str] | pd.Series,
) -> csr_matrix:
    """Transform texts using an already fitted TF-IDF vectorizer."""
    return vectorizer.transform(list(texts))


def extract_tfidf_features(
    df: pd.DataFrame,
    text_column: str = "Text",
    **vectorizer_kwargs,
) -> TfidfFeatureSet:
    """Fit TF-IDF on the specified text column of a DataFrame."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    return fit_tfidf_vectorizer(df[text_column].fillna("").astype(str), **vectorizer_kwargs)


def top_features_for_row(
    vectorizer: TfidfVectorizer,
    row_vector: csr_matrix,
    top_n: int = 10,
) -> list[tuple[str, float]]:
    """Return the top weighted TF-IDF terms for a single row."""
    dense_row = row_vector.toarray().ravel()
    if dense_row.size == 0:
        return []

    indices = np.argsort(dense_row)[::-1]
    top_indices = [index for index in indices if dense_row[index] > 0][:top_n]
    feature_names = vectorizer.get_feature_names_out()
    return [(feature_names[index], float(dense_row[index])) for index in top_indices]


def tfidf_summary(matrix: csr_matrix) -> dict[str, float | int]:
    """Return useful summary stats for a TF-IDF matrix."""
    total_entries = matrix.shape[0] * matrix.shape[1]
    non_zero_entries = matrix.nnz
    sparsity = 0.0 if total_entries == 0 else 1 - (non_zero_entries / total_entries)
    return {
        "rows": matrix.shape[0],
        "columns": matrix.shape[1],
        "non_zero_entries": non_zero_entries,
        "sparsity": round(float(sparsity), 4),
    }
