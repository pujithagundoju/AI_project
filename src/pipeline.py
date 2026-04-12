from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from data_loader import load_train_test_data
from feature_extraction import extract_train_test_features
from model_training import get_best_model
from preprocessing import preprocess_dataframe, preprocess_text
from similarity_retrieval import retrieval_report


def run_final_pipeline_result(
    dataset_dir: str = "Dataset",
    text_column: str = "text",
    label_column: str = "label",
    max_features: int = 10000,
    new_resume_texts: Sequence[str] | None = None,
    job_description: str | None = None,
    top_k: int = 5,
    top_n_models: int = 5,
) -> dict[str, Any]:
    """Use existing module functions and return one structured final result."""
    train_df, test_df = load_train_test_data(dataset_dir=dataset_dir)
    train_df = preprocess_dataframe(train_df, text_column=text_column)
    test_df = preprocess_dataframe(test_df, text_column=text_column)

    vectorizer, x_train, x_test = extract_train_test_features(
        train_df=train_df,
        test_df=test_df,
        text_column=text_column,
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        analyzer="word",
        stop_words="english",
    )

    best_model_name, best_model, results_df = get_best_model(
        x_train=x_train,
        y_train=train_df[label_column],
        x_test=x_test,
        y_test=test_df[label_column],
    )
    best_metrics_row = results_df.iloc[0].to_dict()
    top_models_df = results_df[
        ["model", "accuracy", "balanced_accuracy", "precision", "recall", "f1_score"]
    ].head(top_n_models)

    resumes = list(new_resume_texts) if new_resume_texts is not None else [
        "Data scientist with python, machine learning, statistics and NLP.",
        "Web developer with JavaScript, React, HTML and CSS.",
    ]
    processed_resumes = [preprocess_text(text) for text in resumes]
    predicted_labels = [str(label) for label in best_model.predict(vectorizer.transform(processed_resumes))]

    retrieval = None
    if job_description:
        retrieval = retrieval_report(
            job_description=preprocess_text(job_description),
            resume_texts=train_df[text_column].fillna("").astype(str).head(50).tolist(),
            top_k=top_k,
        )

    return {
        "classification": {
            "best_model_name": best_model_name,
            "best_model_metrics": {
                "accuracy": float(best_metrics_row["accuracy"]),
                "balanced_accuracy": float(best_metrics_row["balanced_accuracy"]),
                "precision": float(best_metrics_row["precision"]),
                "recall": float(best_metrics_row["recall"]),
                "f1_score": float(best_metrics_row["f1_score"]),
                "precision_weighted": float(best_metrics_row["precision_weighted"]),
                "recall_weighted": float(best_metrics_row["recall_weighted"]),
                "f1_weighted": float(best_metrics_row["f1_weighted"]),
            },
            "top_models": top_models_df.to_dict("records"),
            "results": results_df,
            "x_train_shape": tuple(x_train.shape),
            "x_test_shape": tuple(x_test.shape),
        },
        "predictions": {
            "inputs": resumes,
            "labels": predicted_labels,
        },
        "retrieval": retrieval,
    }


# Backward-compatible name for existing code.
run_final_combined_pipeline = run_final_pipeline_result


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module=r"sklearn\.feature_extraction\.text")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"sklearn\.utils\.deprecation")

    project_root = Path(__file__).resolve().parent.parent
    output = run_final_pipeline_result(
        dataset_dir=str(project_root / "Dataset"),
        job_description="Looking for python, machine learning, NLP and data science skills.",
        top_k=5,
    )

    top_model_rows = output["classification"]["results"][["model", "accuracy"]].head(5)
    print("Top Accuracies")
    for _, row in top_model_rows.iterrows():
        print(f"{row['model']}: {float(row['accuracy']):.6f}")

    print("Top Similarity Scores")
    if output["retrieval"] is not None and not output["retrieval"]["ranking_df"].empty:
        top_similarity_rows = output["retrieval"]["ranking_df"][["resume_index", "similarity_score"]].head(5)
        for _, row in top_similarity_rows.iterrows():
            print(f"resume_index={int(row['resume_index'])}: {float(row['similarity_score']):.6f}")
    else:
        print("N/A")
