import glob
import sys

import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError


sys.path.append(".")
from utils import BASE_PATH, FINAL_CSV_FILE, GITHUB_SHA, REPO_ID, collate_csv  # noqa: E402


def has_previous_benchmark() -> str:
    csv_path = None
    try:
        csv_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FINAL_CSV_FILE)
    except EntryNotFoundError:
        csv_path = None
    return csv_path


def filter_float(value):
    if isinstance(value, str):
        return float(value.split()[0])
    return value


def push_to_hf_dataset():
    all_csvs = sorted(glob.glob(f"{BASE_PATH}/*.csv"))
    collate_csv(all_csvs, FINAL_CSV_FILE)

    # If there's an existing benchmark file, we should report the changes.
    csv_path = has_previous_benchmark()
    if csv_path is not None:
        current_results = pd.read_csv(FINAL_CSV_FILE)
        previous_results = pd.read_csv(csv_path)

        numeric_columns = current_results.select_dtypes(include=["float64", "int64"]).columns
        numeric_columns = [
            c for c in numeric_columns if c not in ["batch_size", "num_inference_steps", "actual_gpu_memory (gbs)"]
        ]

        for column in numeric_columns:
            previous_results[column] = previous_results[column].map(lambda x: filter_float(x))

            # Calculate the percentage change
            current_results[column] = current_results[column].astype(float)
            previous_results[column] = previous_results[column].astype(float)
            percent_change = ((current_results[column] - previous_results[column]) / previous_results[column]) * 100

            # Format the values with '+' or '-' sign and append to original values
            current_results[column] = current_results[column].map(str) + percent_change.map(
                lambda x: f" ({'+' if x > 0 else ''}{x:.2f}%)"
            )
            # There might be newly added rows. So, filter out the NaNs.
            current_results[column] = current_results[column].map(lambda x: x.replace(" (nan%)", ""))

        # Overwrite the current result file.
        current_results.to_csv(FINAL_CSV_FILE, index=False)

    commit_message = f"upload from sha: {GITHUB_SHA}" if GITHUB_SHA is not None else "upload benchmark results"
    upload_file(
        repo_id=REPO_ID,
        path_in_repo=FINAL_CSV_FILE,
        path_or_fileobj=FINAL_CSV_FILE,
        repo_type="dataset",
        commit_message=commit_message,
    )


if __name__ == "__main__":
    push_to_hf_dataset()
