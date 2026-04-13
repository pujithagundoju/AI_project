from __future__ import annotations

from time import perf_counter
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from feature_extraction import extract_train_test_features


def get_model_candidates(random_state: int = 42) -> dict[str, Any]:
    """Return strong baseline and ensemble candidates for resume classification."""
    linear_svm = LinearSVC(C=2.5, class_weight="balanced", random_state=random_state)
    calibrated_svm = CalibratedClassifierCV(
        estimator=LinearSVC(C=2.5, class_weight="balanced", random_state=random_state),
        cv=3,
        method="sigmoid",
    )
    calibrated_ridge = CalibratedClassifierCV(
        estimator=RidgeClassifier(alpha=0.6, class_weight="balanced", random_state=random_state),
        cv=3,
        method="sigmoid",
    )
    strong_logistic = LogisticRegression(
        C=8.0,
        max_iter=6000,
        class_weight="balanced",
        solver="saga",
        random_state=random_state,
    )
    complement_nb = ComplementNB(alpha=0.12)

    return {
        "LinearSVM": linear_svm,
        "CalibratedLinearSVM": calibrated_svm,
        "RidgeClassifier": RidgeClassifier(alpha=0.6, class_weight="balanced", random_state=random_state),
        "CalibratedRidge": calibrated_ridge,
        "LogisticRegression": strong_logistic,
        "SGDLogLoss": SGDClassifier(
            loss="log_loss",
            alpha=2e-5,
            penalty="l2",
            max_iter=7000,
            tol=1e-4,
            class_weight="balanced",
            random_state=random_state,
        ),
        "PassiveAggressive": PassiveAggressiveClassifier(
            C=0.8,
            class_weight="balanced",
            max_iter=7000,
            tol=1e-4,
            random_state=random_state,
        ),
        "ComplementNB": complement_nb,
        "SoftVotingEnsemble": VotingClassifier(
            estimators=[
                ("lr", clone(strong_logistic)),
                ("svm", clone(calibrated_svm)),
                ("nb", clone(complement_nb)),
            ],
            voting="soft",
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def get_feature_settings() -> list[dict[str, Any]]:
    """Return TF-IDF settings with feature spaces above 10000 for stronger accuracy."""
    return [
        {
            "max_features": 12000,
            "ngram_range": (1, 2),
            "min_df": 1,
            "max_df": 0.95,
            "analyzer": "word",
            "stop_words": "english",
        },
        {
            "max_features": 15000,
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.9,
            "analyzer": "word",
            "stop_words": "english",
        },
        {
            "max_features": 20000,
            "ngram_range": (3, 5),
            "min_df": 2,
            "max_df": 1.0,
            "analyzer": "char_wb",
            "stop_words": None,
        },
    ]


def train_model(model: Any, x_train, y_train):
    """Fit one model and return it."""
    return model.fit(x_train, y_train)


def evaluate_model(model: Any, x_test, y_test) -> dict[str, float]:
    """Evaluate one model and return classification metrics."""
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


def compare_models(
    x_train,
    y_train,
    x_test,
    y_test,
    models: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train and compare candidate models by accuracy and tie-breaker metrics."""
    model_pool = models or get_model_candidates()
    rows: list[dict[str, Any]] = []
    trained_models: dict[str, Any] = {}

    for name, model in model_pool.items():
        start = perf_counter()
        trained = train_model(clone(model), x_train, y_train)
        elapsed = perf_counter() - start
        trained_models[name] = trained

        metrics = evaluate_model(trained, x_test, y_test)
        rows.append({"model": name, **metrics, "train_time_sec": round(float(elapsed), 4)})

    results_df = pd.DataFrame(rows).sort_values(
        by=["accuracy", "balanced_accuracy", "f1_score", "train_time_sec"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    return results_df, trained_models


def get_best_model(
    x_train,
    y_train,
    x_test,
    y_test,
    models: dict[str, Any] | None = None,
) -> tuple[str, Any, pd.DataFrame]:
    """Return best model name, fitted model, and comparison table."""
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
    """Compare model performance across multiple TF-IDF settings."""
    settings = feature_settings or get_feature_settings()
    if label_column not in train_df.columns or label_column not in test_df.columns:
        raise ValueError(f"Column '{label_column}' not found in train/test DataFrame")

    rows: list[dict[str, Any]] = []
    y_train, y_test = train_df[label_column], test_df[label_column]

    for setting in settings:
        _, x_train, x_test = extract_train_test_features(
            train_df,
            test_df,
            text_column=text_column,
            **setting,
        )
        results_df, _ = compare_models(
            x_train,
            y_train,
            x_test,
            y_test,
            models=get_model_candidates(random_state=random_state),
        )
        for _, row in results_df.iterrows():
            rows.append({**row.to_dict(), **setting})

    return pd.DataFrame(rows).sort_values(
        by=["accuracy", "balanced_accuracy", "f1_score", "train_time_sec"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def get_best_model_feature_combination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    feature_settings: list[dict[str, Any]] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Return the best model and TF-IDF setting combination."""
    best_row = compare_model_feature_combinations(
        train_df=train_df,
        test_df=test_df,
        text_column=text_column,
        label_column=label_column,
        feature_settings=feature_settings,
        random_state=random_state,
    ).iloc[0]

    return {
        k: best_row[k]
        for k in [
            "model",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1_score",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "train_time_sec",
            "max_features",
            "ngram_range",
            "min_df",
            "max_df",
            "analyzer",
            "stop_words",
        ]
    }


def run_compact_model_comparison(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    feature_settings: list[dict[str, Any]] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run comparison and return notebook-friendly summaries."""
    results_df = compare_model_feature_combinations(
        train_df=train_df,
        test_df=test_df,
        text_column=text_column,
        label_column=label_column,
        feature_settings=feature_settings,
        random_state=random_state,
    )
    top_overall = results_df[
        ["model", "accuracy", "balanced_accuracy", "f1_score", "train_time_sec", "ngram_range", "analyzer"]
    ].head(5)
    focus_models = results_df[
        results_df["model"].isin(["SoftVotingEnsemble", "CalibratedLinearSVM", "LogisticRegression"])
    ][
        ["model", "accuracy", "balanced_accuracy", "f1_score", "train_time_sec", "ngram_range", "analyzer"]
    ].head(4)
    best_combo = get_best_model_feature_combination(
        train_df=train_df,
        test_df=test_df,
        text_column=text_column,
        label_column=label_column,
        feature_settings=feature_settings,
        random_state=random_state,
    )

    return {
        "results_df": results_df,
        "top_overall": top_overall,
        "focus_models": focus_models,
        "randomforest_svm": focus_models,
        "best_combo": best_combo,
    }
