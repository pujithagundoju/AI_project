from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeRetriever:
    """Vector-space retriever for resume text using TF-IDF + cosine similarity."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        max_df: float = 0.95,
        stop_words: str | list[str] | None = "english",
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
            sublinear_tf=True,
        )
        self._resume_texts: list[str] = []
        self._resume_matrix = None

    @staticmethod
    def _normalize_texts(texts: Iterable[str]) -> list[str]:
        return [str(text) for text in texts]

    def fit(self, resume_texts: Sequence[str]) -> "ResumeRetriever":
        """Build retrieval index from resume texts."""
        if not resume_texts:
            raise ValueError("resume_texts cannot be empty")

        self._resume_texts = self._normalize_texts(resume_texts)
        self._resume_matrix = self.vectorizer.fit_transform(self._resume_texts)
        return self

    def cosine_scores(self, query_text: str) -> pd.DataFrame:
        """Return cosine similarity score for each indexed resume."""
        if self._resume_matrix is None:
            raise ValueError("Retriever is not fitted. Call fit() first.")

        query_vector = self.vectorizer.transform([str(query_text)])
        scores = cosine_similarity(query_vector, self._resume_matrix).flatten()

        return pd.DataFrame(
            {
                "resume_index": list(range(len(self._resume_texts))),
                "cosine_similarity": scores,
            }
        )

    def rank(self, query_text: str, top_k: int | None = None) -> pd.DataFrame:
        """Return ranked resumes by relevance score for a query/job description."""
        score_df = self.cosine_scores(query_text)

        ranked = pd.DataFrame(
            {
                "resume_index": list(range(len(self._resume_texts))),
                "relevance_score": score_df["cosine_similarity"].values,
                "resume_text": self._resume_texts,
            }
        ).sort_values(by="relevance_score", ascending=False)

        ranked = ranked.reset_index(drop=True)
        if top_k is not None:
            ranked = ranked.head(top_k).reset_index(drop=True)
        return ranked


def cosine_similarity_between_texts(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts using TF-IDF."""
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    matrix = vectorizer.fit_transform([str(text_a), str(text_b)])
    score = cosine_similarity(matrix[0:1], matrix[1:2]).flatten()[0]
    return float(score)


def retrieve_from_dataframe(
    df: pd.DataFrame,
    query_text: str,
    text_column: str = "text",
    top_k: int = 10,
) -> pd.DataFrame:
    """Fit retriever on a DataFrame text column and return top-k matches."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")

    retriever = ResumeRetriever()
    retriever.fit(df[text_column].fillna("").astype(str).tolist())

    ranked = retriever.rank(query_text=query_text, top_k=top_k)
    ranked.insert(1, "source_row_index", ranked["resume_index"])
    return ranked


def precision_recall_f1_at_k(
    ranked_indices: Sequence[int],
    relevant_indices: set[int],
    k: int,
) -> dict[str, float]:
    """Compute Precision@k, Recall@k and F1@k for retrieval."""
    if k <= 0:
        raise ValueError("k must be >= 1")

    top_k = list(ranked_indices[:k])
    hits = sum(1 for idx in top_k if idx in relevant_indices)

    precision = hits / len(top_k) if top_k else 0.0
    recall = hits / len(relevant_indices) if relevant_indices else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision_at_k": float(precision),
        "recall_at_k": float(recall),
        "f1_at_k": float(f1),
        "hits": float(hits),
    }


def evaluate_retrieval_query(
    retriever: ResumeRetriever,
    query_text: str,
    relevant_indices: set[int],
    k: int = 10,
) -> dict[str, float]:
    """Evaluate a single retrieval query against known relevant resume indices."""
    ranked = retriever.rank(query_text=query_text)
    ranked_indices = ranked["resume_index"].tolist()
    return precision_recall_f1_at_k(ranked_indices, relevant_indices, k)
