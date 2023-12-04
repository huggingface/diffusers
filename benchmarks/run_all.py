import glob
import subprocess


PATTERN = "benchmark_*.py"


def main():
    python_files = glob.glob(PATTERN)

    for file in python_files:
        print(f"Running {file}.")
        subprocess.run(["python", file])
        subprocess.run(["python", file, "--run_compile"])


if __name__ == "__main__":
    main()
