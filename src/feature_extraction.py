from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_texts(texts: Iterable[str]) -> list[str]:
    """Convert input texts to a clean list of strings."""
    return [str(text) for text in texts]


def create_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
    analyzer: str = "word",
    stop_words: str | list[str] | None = None,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with practical defaults for resume text."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        analyzer=analyzer,
        stop_words=stop_words,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype="float32",
    )


def fit_tfidf(
    texts: Iterable[str],
    vectorizer: TfidfVectorizer | None = None,
):
    """Fit a TF-IDF vectorizer and transform input texts."""
    active_vectorizer = vectorizer or create_tfidf_vectorizer()
    normalized_texts = _normalize_texts(texts)
    features = active_vectorizer.fit_transform(normalized_texts)
    return active_vectorizer, features


def transform_tfidf(vectorizer: TfidfVectorizer, texts: Iterable[str]):
    """Transform texts with a fitted TF-IDF vectorizer."""
    normalized_texts = _normalize_texts(texts)
    return vectorizer.transform(normalized_texts)


def extract_train_test_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    max_features: int = 5000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
    max_df: float = 1.0,
    analyzer: str = "word",
    stop_words: str | list[str] | None = None,
):
    """Fit on train text and transform both train and test text."""
    if text_column not in train_df.columns:
        raise ValueError(f"Column '{text_column}' not found in train DataFrame")
    if text_column not in test_df.columns:
        raise ValueError(f"Column '{text_column}' not found in test DataFrame")

    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        analyzer=analyzer,
        stop_words=stop_words,
    )
    train_texts = train_df[text_column].fillna("").astype(str)
    test_texts = test_df[text_column].fillna("").astype(str)

    vectorizer, x_train = fit_tfidf(train_texts, vectorizer)
    x_test = transform_tfidf(vectorizer, test_texts)
    return vectorizer, x_train, x_test


def rank_resumes_by_job_description(
    job_description: str,
    resume_texts: Sequence[str],
    vectorizer: TfidfVectorizer | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Rank resumes by cosine similarity against a job description."""
    if not resume_texts:
        raise ValueError("resume_texts cannot be empty")

    normalized_resumes = _normalize_texts(resume_texts)
    active_vectorizer, resume_matrix = fit_tfidf(normalized_resumes, vectorizer)

    job_vector = transform_tfidf(active_vectorizer, [job_description])
    similarity_scores = cosine_similarity(job_vector, resume_matrix).flatten()

    ranking = pd.DataFrame(
        {
            "resume_index": list(range(len(normalized_resumes))),
            "similarity_score": similarity_scores,
            "resume_text": normalized_resumes,
        }
    ).sort_values(by="similarity_score", ascending=False)

    ranking = ranking.reset_index(drop=True)
    if top_k is not None:
        ranking = ranking.head(top_k).reset_index(drop=True)
    return ranking
