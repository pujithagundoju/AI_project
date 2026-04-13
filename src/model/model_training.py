from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from ..feature_extraction import extract_train_test_features


def get_model_candidates(random_state: int = 42) -> dict[str, Any]:
    """Return resume-classification model candidates."""
    return {
        "ComplementNB": ComplementNB(alpha=0.2),
        "MultinomialNB": MultinomialNB(alpha=0.05),
        "LogisticRegression": LogisticRegression(
            C=8.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "LinearSVM": LinearSVC(
            C=2.0,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RidgeClassifier": RidgeClassifier(
            alpha=0.5,
            class_weight="balanced",
            random_state=random_state,
        ),
        "PassiveAggressive": PassiveAggressiveClassifier(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",
            algorithm="brute",
            weights="distance",
            n_jobs=-1,
        ),
    }


def train_model(model: Any, x_train, y_train):
    """Fit one model and return it."""
    model.fit(x_train, y_train)
    return model


def evaluate_model(model: Any, x_test, y_test) -> dict[str, float]:
    """Evaluate one model and return classification metrics."""
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    balanced_accuracy = balanced_accuracy_score(y_test, predictions)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_score": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def compare_models(
    x_train,
    y_train,
    x_test,
    y_test,
    models: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and compare all models, sorted by accuracy."""
    model_pool = models or get_model_candidates()
    trained_models: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []

    for model_name, model in model_pool.items():
        trained = train_model(model, x_train, y_train)
        metrics = evaluate_model(trained, x_test, y_test)
        trained_models[model_name] = trained

        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "precision_weighted": metrics["precision_weighted"],
                "recall_weighted": metrics["recall_weighted"],
                "f1_weighted": metrics["f1_weighted"],
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        by=["accuracy", "balanced_accuracy", "f1_score", "precision"],
        ascending=False,
    ).reset_index(drop=True)

    return results_df, trained_models


def get_best_model(
    x_train,
    y_train,
    x_test,
    y_test,
    models: dict[str, Any] | None = None,
) -> tuple[str, Any, pd.DataFrame]:
    """Return best model name, trained model, and full results."""
    results_df, trained_models = compare_models(x_train, y_train, x_test, y_test, models=models)
    best_name = str(results_df.iloc[0]["model"])
    return best_name, trained_models[best_name], results_df


def compare_model_feature_combinations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    feature_settings: list[dict[str, Any]] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare model performance across TF-IDF parameter combinations."""
    if label_column not in train_df.columns:
        raise ValueError(f"Column '{label_column}' not found in train DataFrame")
    if label_column not in test_df.columns:
        raise ValueError(f"Column '{label_column}' not found in test DataFrame")

    settings = feature_settings or [
        {
            "max_features": 10000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "analyzer": "word",
            "stop_words": "english",
        },
        {
            "max_features": 10000,
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.9,
            "analyzer": "word",
            "stop_words": "english",
        },
        {
            "max_features": 10000,
            "ngram_range": (3, 5),
            "min_df": 2,
            "max_df": 1.0,
            "analyzer": "char_wb",
            "stop_words": None,
        },
    ]

    y_train = train_df[label_column]
    y_test = test_df[label_column]

    rows: list[dict[str, Any]] = []
    for setting in settings:
        _, x_train, x_test = extract_train_test_features(
            train_df,
            test_df,
            text_column=text_column,
            max_features=int(setting["max_features"]),
            ngram_range=setting["ngram_range"],
            min_df=int(setting["min_df"]),
            max_df=float(setting["max_df"]),
            analyzer=str(setting["analyzer"]),
            stop_words=setting.get("stop_words", None),
        )

        results_df, _ = compare_models(
            x_train,
            y_train,
            x_test,
            y_test,
            models=get_model_candidates(random_state=random_state),
        )

        for _, row in results_df.iterrows():
            rows.append(
                {
                    "model": row["model"],
                    "accuracy": row["accuracy"],
                    "balanced_accuracy": row["balanced_accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                    "precision_weighted": row["precision_weighted"],
                    "recall_weighted": row["recall_weighted"],
                    "f1_weighted": row["f1_weighted"],
                    "max_features": setting["max_features"],
                    "ngram_range": setting["ngram_range"],
                    "min_df": setting["min_df"],
                    "max_df": setting["max_df"],
                    "analyzer": setting["analyzer"],
                    "stop_words": setting.get("stop_words", None),
                }
            )

    return pd.DataFrame(rows).sort_values(
        by=["accuracy", "balanced_accuracy", "f1_score", "precision"],
        ascending=False,
    ).reset_index(drop=True)


def get_best_model_feature_combination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    feature_settings: list[dict[str, Any]] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return the best model + TF-IDF setting by accuracy."""
    comparison_df = compare_model_feature_combinations(
        train_df=train_df,
        test_df=test_df,
        text_column=text_column,
        label_column=label_column,
        feature_settings=feature_settings,
        random_state=random_state,
    )

    best_row = comparison_df.iloc[0]
    return {
        "model": best_row["model"],
        "accuracy": float(best_row["accuracy"]),
        "balanced_accuracy": float(best_row["balanced_accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1_score": float(best_row["f1_score"]),
        "precision_weighted": float(best_row["precision_weighted"]),
        "recall_weighted": float(best_row["recall_weighted"]),
        "f1_weighted": float(best_row["f1_weighted"]),
        "max_features": int(best_row["max_features"]),
        "ngram_range": best_row["ngram_range"],
        "min_df": int(best_row["min_df"]),
        "max_df": float(best_row["max_df"]),
        "analyzer": str(best_row["analyzer"]),
        "stop_words": best_row["stop_words"],
    }
