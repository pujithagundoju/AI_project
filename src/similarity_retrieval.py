from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _normalize_texts(texts: Iterable[str]) -> list[str]:
    """Convert iterable text input into a clean string list."""
    return [str(text) for text in texts]


def create_retrieval_vectorizer(
    max_features: int = 10000,
    ngram_range: tuple[int, int] = (1, 3),
    min_df: int = 1,
    max_df: float = 0.95,
    analyzer: str = "word",
    stop_words: str | list[str] | None = "english",
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for vector space retrieval."""
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


def build_resume_vector_space(
    resume_texts: Sequence[str],
    vectorizer: TfidfVectorizer | None = None,
) -> tuple[TfidfVectorizer, any, list[str]]:
    """Fit TF-IDF on resume texts and return vectorizer, matrix, and normalized texts."""
    if len(resume_texts) == 0:
        raise ValueError("resume_texts cannot be empty")

    normalized_texts = _normalize_texts(resume_texts)
    active_vectorizer = vectorizer or create_retrieval_vectorizer()
    resume_matrix = active_vectorizer.fit_transform(normalized_texts)
    return active_vectorizer, resume_matrix, normalized_texts


def transform_job_description(job_description: str, vectorizer: TfidfVectorizer):
    """Transform one job description into vector space coordinates."""
    return vectorizer.transform([str(job_description)])


def compute_similarity_scores(job_vector, resume_matrix) -> np.ndarray:
    """Compute cosine similarity scores between JD vector and resume matrix."""
    return cosine_similarity(job_vector, resume_matrix).flatten()


def rank_resumes_by_similarity(
    job_description: str,
    resume_texts: Sequence[str],
    vectorizer: TfidfVectorizer | None = None,
    top_k: int | None = None,
) -> tuple[pd.DataFrame, TfidfVectorizer, any, any]:
    """Rank resumes by relevance score in a vector space model."""
    fitted_vectorizer, resume_matrix, normalized_texts = build_resume_vector_space(resume_texts, vectorizer=vectorizer)
    job_vector = transform_job_description(job_description, fitted_vectorizer)
    scores = compute_similarity_scores(job_vector, resume_matrix)

    ranking_df = pd.DataFrame(
        {
            "resume_index": list(range(len(normalized_texts))),
            "similarity_score": scores,
            "resume_text": normalized_texts,
        }
    ).sort_values(by="similarity_score", ascending=False).reset_index(drop=True)

    if top_k is not None:
        ranking_df = ranking_df.head(top_k).reset_index(drop=True)

    return ranking_df, fitted_vectorizer, job_vector, resume_matrix


def get_nonzero_tfidf_terms(vector, feature_names: np.ndarray, value_column: str = "tfidf") -> pd.DataFrame:
    """Return non-zero TF-IDF terms for one sparse vector."""
    dense_values = vector.toarray().ravel()
    mask = dense_values > 0
    return pd.DataFrame(
        {
            "term": feature_names[mask],
            value_column: dense_values[mask],
        }
    ).sort_values(by=value_column, ascending=False).reset_index(drop=True)


def explain_term_products(
    job_vector,
    resume_matrix,
    resume_texts: Sequence[str],
    feature_names: np.ndarray,
    top_terms_per_resume: int = 10,
) -> pd.DataFrame:
    """Explain relevance with JD-TFIDF x Resume-TFIDF term products."""
    job_dense = job_vector.toarray().ravel()
    resume_dense = resume_matrix.toarray()

    rows: list[pd.DataFrame] = []
    for idx, resume_text in enumerate(_normalize_texts(resume_texts)):
        products = resume_dense[idx] * job_dense
        mask = products > 0
        if not mask.any():
            continue

        product_df = pd.DataFrame(
            {
                "resume_index": idx,
                "resume_text": resume_text,
                "term": feature_names[mask],
                "jd_tfidf": job_dense[mask],
                "resume_tfidf": resume_dense[idx][mask],
                "product": products[mask],
            }
        ).sort_values(by="product", ascending=False)

        rows.append(product_df.head(top_terms_per_resume))

    if not rows:
        return pd.DataFrame(
            columns=["resume_index", "resume_text", "term", "jd_tfidf", "resume_tfidf", "product"]
        )

    return pd.concat(rows, ignore_index=True)


def retrieval_report(
    job_description: str,
    resume_texts: Sequence[str],
    top_k: int = 5,
    top_terms_per_resume: int = 10,
) -> dict[str, object]:
    """Return a complete information retrieval report for one JD and resume set."""
    ranking_df, vectorizer, job_vector, resume_matrix = rank_resumes_by_similarity(
        job_description=job_description,
        resume_texts=resume_texts,
        top_k=top_k,
    )
    feature_names = vectorizer.get_feature_names_out()

    report = {
        "vectorizer": vectorizer,
        "job_vector_shape": tuple(job_vector.shape),
        "resume_matrix_shape": tuple(resume_matrix.shape),
        "job_terms_df": get_nonzero_tfidf_terms(job_vector, feature_names, value_column="jd_tfidf"),
        "ranking_df": ranking_df,
        "products_df": explain_term_products(
            job_vector,
            resume_matrix,
            resume_texts=_normalize_texts(resume_texts),
            feature_names=feature_names,
            top_terms_per_resume=top_terms_per_resume,
        ),
    }
    return report
