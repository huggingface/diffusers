import glob
import os
import sys

from huggingface_hub import upload_file


sys.path.append(".")
from benchmarks.utils import BASE_PATH, collate_csv  # noqa: E402


FINAL_CSV_FILE = "collated_results.csv"
REPO_ID = "diffusers/benchmarks"
GITHUB_SHA = os.getenv("GITHUB_SHA", None)


def push_to_hf_dataset():
    all_csvs = sorted(glob.glob(f"{BASE_PATH}/*.csv"))
    collate_csv(all_csvs, FINAL_CSV_FILE)

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
