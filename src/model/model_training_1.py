from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

from feature_extraction import extract_train_test_features


# -------------------------------------------------------
# Stronger Model Candidates (Higher Accuracy)
# -------------------------------------------------------

def get_model_candidates(random_state: int = 42) -> dict[str, Any]:
    """Return stronger models for resume classification."""

    return {

        "LinearSVM": LinearSVC(
            C=3.0,
            class_weight="balanced",
            random_state=random_state
        ),

        "LogisticRegression": LogisticRegression(
            C=8.0,
            max_iter=4000,
            class_weight="balanced",
            solver="lbfgs",
            n_jobs=-1,
            random_state=random_state,
        ),

        "RidgeClassifier": RidgeClassifier(
            alpha=0.8,
            class_weight="balanced",
            random_state=random_state
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            random_state=random_state
        ),

        "NaiveBayes": MultinomialNB(alpha=0.1),

    }


# -------------------------------------------------------
# Better TF-IDF Feature Settings
# -------------------------------------------------------

def get_feature_settings() -> list[dict[str, Any]]:

    return [

        {
            "max_features": 15000,
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.95,
            "analyzer": "word",
            "stop_words": "english",
        },

        {
            "max_features": 20000,
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.90,
            "analyzer": "word",
            "stop_words": "english",
        },

        {
            "max_features": 25000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.85,
            "analyzer": "word",
            "stop_words": "english",
        },

    ]


# -------------------------------------------------------
# Train Model
# -------------------------------------------------------

def train_model(model: Any, x_train, y_train):

    return model.fit(x_train, y_train)


# -------------------------------------------------------
# Evaluate Model
# -------------------------------------------------------

def evaluate_model(model: Any, x_test, y_test):

    predictions = model.predict(x_test)

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

        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),

        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_score": float(f1_macro),

        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),

    }


# -------------------------------------------------------
# Compare Models
# -------------------------------------------------------

def compare_models(
    x_train,
    y_train,
    x_test,
    y_test,
    models: dict[str, Any] | None = None
):

    model_pool = models or get_model_candidates()

    rows = []
    trained_models = {}

    for name, model in model_pool.items():

        start = perf_counter()

        trained = train_model(clone(model), x_train, y_train)

        elapsed = perf_counter() - start

        trained_models[name] = trained

        metrics = evaluate_model(trained, x_test, y_test)

        rows.append({
            "model": name,
            **metrics,
            "train_time_sec": round(float(elapsed), 4),
        })

    results_df = pd.DataFrame(rows).sort_values(
        by=["accuracy", "balanced_accuracy", "f1_score"],
        ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models


# -------------------------------------------------------
# Compare Model + Feature Settings
# -------------------------------------------------------

def compare_model_feature_combinations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    feature_settings: list[dict[str, Any]] | None = None,
    random_state: int = 42,
):

    settings = feature_settings or get_feature_settings()

    rows = []

    y_train = train_df[label_column]
    y_test = test_df[label_column]

    for setting in settings:

        _, x_train, x_test = extract_train_test_features(
            train_df,
            test_df,
            text_column=text_column,
            **setting
        )

        results_df, _ = compare_models(
            x_train,
            y_train,
            x_test,
            y_test,
            models=get_model_candidates(random_state=random_state)
        )

        for _, row in results_df.iterrows():

            rows.append({
                **row.to_dict(),
                **setting
            })

    return pd.DataFrame(rows).sort_values(
        by=["accuracy", "balanced_accuracy", "f1_score"],
        ascending=False
    ).reset_index(drop=True)


# -------------------------------------------------------
# Get Best Model
# -------------------------------------------------------

def get_best_model_feature_combination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
):

    best_row = compare_model_feature_combinations(
        train_df,
        test_df,
        text_column,
        label_column
    ).iloc[0]

    return best_row.to_dict()


# -------------------------------------------------------
# Main Runner
# -------------------------------------------------------

def run_compact_model_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
):

    results_df = compare_model_feature_combinations(
        train_df,
        test_df,
        text_column,
        label_column,
    )

    best_combo = results_df.iloc[0]

    return {

        "results_df": results_df,

        "top_5": results_df.head(5),

        "best_model": best_combo["model"],

        "best_accuracy": best_combo["accuracy"],

        "best_settings": best_combo.to_dict(),

    }