import glob
import os
import subprocess

import pandas as pd

from diffusers.utils import logging


PATTERN = "benchmarking_*.py"
FINAL_CSV_FILENAME = "collated_results.csv"
GITHUB_SHA = os.getenv("GITHUB_SHA", None)


logger = logging.get_logger(__name__)


class SubprocessCallException(Exception):
    pass


# Taken from `test_examples_utils.py`
def run_command(command: list[str], return_stdout=False):
    """
    Runs `command` with `subprocess.check_output` and will potentially return the `stdout`. Will also properly capture
    if an error occurred while running `command`
    """
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout:
            if hasattr(output, "decode"):
                output = output.decode("utf-8")
            return output
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(
            f"Command `{' '.join(command)}` failed with the following error:\n\n{e.output.decode()}"
        ) from e


def run_scripts():
    python_files = sorted(glob.glob(PATTERN))

    for file in python_files:
        if file != "benchmarking_utils.py":
            logger.info(f"****** Running file: {file} ******")
            command = f"python {file}"
            try:
                run_command(command)
            except SubprocessCallException as e:
                logger.error(f"Error running {file}: {e}")
                continue


def merge_csvs():
    all_csvs = glob.glob("*.csv")
    final_df = pd.concat([pd.read_csv(f) for f in all_csvs]).reset_index(drop=True)
    if GITHUB_SHA:
        final_df["github_sha"] = GITHUB_SHA
    final_df.to_csv(FINAL_CSV_FILENAME)


if __name__ == "__main__":
    run_scripts()
    merge_csvs()
