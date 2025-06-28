import os

import pandas as pd
from huggingface_hub import hf_hub_download, upload_file
from huggingface_hub.utils import EntryNotFoundError


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

    csv_path = has_previous_benchmark()
    if csv_path is not None:
        current_results = pd.read_csv(FINAL_CSV_FILENAME)
        previous_results = pd.read_csv(csv_path)

        numeric_columns = current_results.select_dtypes(include=["float64", "int64"]).columns

        for column in numeric_columns:
            # get previous values as floats, aligned to current index
            prev_vals = previous_results[column].map(filter_float).reindex(current_results.index)

            # get current values as floats
            curr_vals = current_results[column].astype(float)

            # stringify the current values
            curr_str = curr_vals.map(str)

            # build an appendage only when prev exists and differs
            append_str = prev_vals.where(prev_vals.notnull() & (prev_vals != curr_vals), other=pd.NA).map(
                lambda x: f" ({x})" if pd.notnull(x) else ""
            )

            # combine
            current_results[column] = curr_str + append_str
        os.remove(FINAL_CSV_FILENAME)
        current_results.to_csv(FINAL_CSV_FILENAME, index=False)

    commit_message = f"upload from sha: {GITHUB_SHA}" if GITHUB_SHA is not None else "upload benchmark results"
    upload_file(
        repo_id=REPO_ID,
        path_in_repo=FINAL_CSV_FILENAME,
        path_or_fileobj=FINAL_CSV_FILENAME,
        repo_type="dataset",
        commit_message=commit_message,
    )
    upload_file(
        repo_id="diffusers/benchmark-analyzer",
        path_in_repo=FINAL_CSV_FILENAME,
        path_or_fileobj=FINAL_CSV_FILENAME,
        repo_type="space",
        commit_message=commit_message,
    )


if __name__ == "__main__":
    push_to_hf_dataset()
