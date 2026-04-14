from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

# Make src imports work when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
	sys.path.insert(0, str(SRC_PATH))

warnings.filterwarnings("ignore")

from data_loader import load_train_test_data
from model_training import compare_model_feature_combinations
from preprocessing import preprocess_dataframe
from similarity_retrieval import ResumeRetriever


def run_pipeline(
	dataset_dir: str | Path = "Dataset",
	query_text: str = "Looking for a data scientist with Python, machine learning, NLP, and model training experience.",
	top_k: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Run the full pipeline and return top-k accuracy and similarity tables."""
	train_df, test_df = load_train_test_data(dataset_dir)

	train_df = preprocess_dataframe(train_df, text_column="text")
	test_df = preprocess_dataframe(test_df, text_column="text")

	# Best model accuracy results across available model + TF-IDF combinations.
	model_results = compare_model_feature_combinations(
		train_df=train_df,
		test_df=test_df,
		text_column="text",
		label_column="label",
	)
	top_accuracy = model_results[["model", "accuracy"]].head(top_k).reset_index(drop=True)

	# Similarity retrieval against the preprocessed train resumes.
	retriever = ResumeRetriever(max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95)
	retriever.fit(train_df["text"].fillna("").astype(str).tolist())
	similarity_df = retriever.rank(query_text=query_text, top_k=top_k)
	similarity_df["label"] = similarity_df["resume_index"].map(train_df["label"])
	top_similarity = similarity_df[["label", "relevance_score"]].reset_index(drop=True)

	return top_accuracy, top_similarity


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run end-to-end pipeline.",
	)
	parser.add_argument("--dataset-dir", default="Dataset", help="Path to dataset directory")
	parser.add_argument(
		"--query",
		default="Looking for a data scientist with Python, machine learning, NLP, and model training experience.",
		help="Job description / query for similarity retrieval",
	)
	parser.add_argument("--top-k", type=int, default=5, help="Number of top rows to print")
	args = parser.parse_args()

	top_accuracy, top_similarity = run_pipeline(
		dataset_dir=args.dataset_dir,
		query_text=args.query,
		top_k=args.top_k,
	)

	print(top_accuracy.to_string(index=False))
	print(top_similarity.to_string(index=False))


if __name__ == "__main__":
	main()
