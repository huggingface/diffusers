import glob
import logging
import os
import subprocess

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PATTERN = "benchmarking_*.py"
FINAL_CSV_FILENAME = "collated_results.csv"
GITHUB_SHA = os.getenv("GITHUB_SHA", None)


class SubprocessCallException(Exception):
    pass


def run_command(command: list[str], return_stdout=False):
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        if return_stdout and hasattr(output, "decode"):
            return output.decode("utf-8")
    except subprocess.CalledProcessError as e:
        raise SubprocessCallException(f"Command `{' '.join(command)}` failed with:\n{e.output.decode()}") from e


def merge_csvs(final_csv: str = "collated_results.csv"):
    all_csvs = glob.glob("*.csv")
    all_csvs = [f for f in all_csvs if f != final_csv]
    if not all_csvs:
        logger.info("No result CSVs found to merge.")
        return

    df_list = []
    for f in all_csvs:
        try:
            d = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            # If a file existed but was zero‐bytes or corrupted, skip it
            continue
        df_list.append(d)

    if not df_list:
        logger.info("All result CSVs were empty or invalid; nothing to merge.")
        return

    final_df = pd.concat(df_list, ignore_index=True)
    if GITHUB_SHA is not None:
        final_df["github_sha"] = GITHUB_SHA
    final_df.to_csv(final_csv, index=False)
    logger.info(f"Merged {len(all_csvs)} partial CSVs → {final_csv}.")


def run_scripts():
    python_files = sorted(glob.glob(PATTERN))
    python_files = [f for f in python_files if f != "benchmarking_utils.py"]

    for file in python_files:
        script_name = file.split(".py")[0].split("_")[-1]  # example: benchmarking_foo.py -> foo
        logger.info(f"\n****** Running file: {file} ******")

        partial_csv = f"{script_name}.csv"
        if os.path.exists(partial_csv):
            logger.info(f"Found {partial_csv}. Removing for safer numbers and duplication.")
            os.remove(partial_csv)

        command = ["python", file]
        try:
            run_command(command)
            logger.info(f"→ {file} finished normally.")
        except SubprocessCallException as e:
            logger.info(f"Error running {file}:\n{e}")
        finally:
            logger.info(f"→ Merging partial CSVs after {file} …")
            merge_csvs(final_csv=FINAL_CSV_FILENAME)

    logger.info(f"\nAll scripts attempted. Final collated CSV: {FINAL_CSV_FILENAME}")


if __name__ == "__main__":
    run_scripts()
