import os

import pandas as pd
import torch
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError


if torch.cuda.is_available():
    TOTAL_GPU_MEMORY = float(
        os.getenv("TOTAL_GPU_MEMORY", torch.cuda.get_device_properties(0).total_memory / (1024**3))
    )
else:
    raise

REPO_ID = "diffusers/benchmarks"


def has_previous_benchmark() -> str:
    from run_all import FINAL_CSV_FILENAME

    csv_path = None
    try:
        csv_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=FINAL_CSV_FILENAME)
    except EntryNotFoundError:
        csv_path = None
    return csv_path


def filter_float(value):
    if isinstance(value, str):
        return float(value.split()[0])
    return value


def push_to_hf_dataset():
    from run_all import FINAL_CSV_FILENAME, GITHUB_SHA

    # If there's an existing benchmark file, we should report the changes.
    csv_path = has_previous_benchmark()
    if csv_path is not None:
        current_results = pd.read_csv(FINAL_CSV_FILENAME)
        previous_results = pd.read_csv(csv_path)

        # identify the numeric columns we want to annotate
        numeric_columns = current_results.select_dtypes(include=["float64", "int64"]).columns

        # for each numeric column, append the old value in () if present
        for column in numeric_columns:
            # coerce any “x units” strings back to float
            prev_vals = previous_results[column].map(filter_float)
            # align indices in case rows were added/removed
            prev_vals = prev_vals.reindex(current_results.index)

            # build the new string: "current_value (previous_value)"
            curr_str = current_results[column].astype(str)
            prev_str = prev_vals.map(lambda x: f" ({x})" if pd.notnull(x) else "")

            current_results[column] = curr_str + prev_str

        # overwrite the CSV
        current_results.to_csv(FINAL_CSV_FILENAME, index=False)

    commit_message = f"upload from sha: {GITHUB_SHA}" if GITHUB_SHA is not None else "upload benchmark results"
    upload_file(
        repo_id=REPO_ID,
        path_in_repo=FINAL_CSV_FILENAME,
        path_or_fileobj=FINAL_CSV_FILENAME,
        repo_type="dataset",
        commit_message=commit_message,
    )


if __name__ == "__main__":
    push_to_hf_dataset()
