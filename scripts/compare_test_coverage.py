#!/usr/bin/env python3
"""
Compare test coverage between main and model-test-refactor branches
for the Flux transformer tests.

Usage:
    python scripts/compare_test_coverage.py
"""

import subprocess


TEST_FILE = "tests/models/transformers/test_models_transformer_flux.py"
BRANCHES = ["main", "model-test-refactor"]


def run_command(cmd, capture=True):
    """Run a shell command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=capture, text=True)
    return result.stdout, result.stderr, result.returncode


def get_current_branch():
    """Get the current git branch name."""
    stdout, _, _ = run_command("git branch --show-current")
    return stdout.strip()


def stash_changes():
    """Stash any uncommitted changes."""
    run_command("git stash")


def pop_stash():
    """Pop stashed changes."""
    run_command("git stash pop")


def checkout_branch(branch):
    """Checkout a git branch."""
    _, stderr, code = run_command(f"git checkout {branch}")
    if code != 0:
        print(f"Failed to checkout {branch}: {stderr}")
        return False
    return True


def collect_tests(test_file):
    """Collect tests from a test file and return test info."""
    cmd = f"python -m pytest {test_file} --collect-only -q 2>/dev/null"
    stdout, stderr, code = run_command(cmd)

    tests = []
    for line in stdout.strip().split("\n"):
        if "::" in line and not line.startswith("="):
            tests.append(line.strip())

    return tests


def run_tests_verbose(test_file):
    """Run tests and capture pass/skip/fail status."""
    cmd = f"python -m pytest {test_file} -v --tb=no 2>&1"
    stdout, _, _ = run_command(cmd)

    results = {"passed": [], "skipped": [], "failed": [], "errors": []}

    for line in stdout.split("\n"):
        if " PASSED" in line:
            test_name = line.split(" PASSED")[0].strip()
            results["passed"].append(test_name)
        elif " SKIPPED" in line:
            test_name = line.split(" SKIPPED")[0].strip()
            reason = ""
            if "SKIPPED" in line and "[" in line:
                reason = line.split("[")[-1].rstrip("]") if "[" in line else ""
            results["skipped"].append((test_name, reason))
        elif " FAILED" in line:
            test_name = line.split(" FAILED")[0].strip()
            results["failed"].append(test_name)
        elif " ERROR" in line:
            test_name = line.split(" ERROR")[0].strip()
            results["errors"].append(test_name)

    return results


def compare_results(main_results, pr_results):
    """Compare test results between branches."""
    print("\n" + "=" * 70)
    print("COVERAGE COMPARISON REPORT")
    print("=" * 70)

    print("\n## Test Counts")
    print(f"{'Category':<20} {'main':<15} {'PR':<15} {'Diff':<10}")
    print("-" * 60)

    for category in ["passed", "skipped", "failed", "errors"]:
        main_count = len(main_results[category])
        pr_count = len(pr_results[category])
        diff = pr_count - main_count
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        print(f"{category:<20} {main_count:<15} {pr_count:<15} {diff_str:<10}")

    main_tests = set(main_results["passed"] + [t[0] for t in main_results["skipped"]])
    pr_tests = set(pr_results["passed"] + [t[0] for t in pr_results["skipped"]])

    missing_in_pr = main_tests - pr_tests
    new_in_pr = pr_tests - main_tests

    if missing_in_pr:
        print("\n## Tests in main but MISSING in PR:")
        for test in sorted(missing_in_pr):
            print(f"  - {test}")

    if new_in_pr:
        print("\n## NEW tests in PR (not in main):")
        for test in sorted(new_in_pr):
            print(f"  + {test}")

    print("\n## Skipped Tests Comparison")
    main_skipped = {t[0]: t[1] for t in main_results["skipped"]}
    pr_skipped = {t[0]: t[1] for t in pr_results["skipped"]}

    newly_skipped = set(pr_skipped.keys()) - set(main_skipped.keys())
    no_longer_skipped = set(main_skipped.keys()) - set(pr_skipped.keys())

    if newly_skipped:
        print("\nNewly skipped in PR:")
        for test in sorted(newly_skipped):
            print(f"  - {test}: {pr_skipped.get(test, 'unknown reason')}")

    if no_longer_skipped:
        print("\nNo longer skipped in PR (now running):")
        for test in sorted(no_longer_skipped):
            print(f"  + {test}")

    if not newly_skipped and not no_longer_skipped:
        print("\nNo changes in skipped tests.")

    print("\n" + "=" * 70)


def main():
    original_branch = get_current_branch()
    print(f"Current branch: {original_branch}")

    results = {}

    print("Stashing uncommitted changes...")
    stash_changes()

    try:
        for branch in BRANCHES:
            print(f"\n--- Analyzing branch: {branch} ---")

            if not checkout_branch(branch):
                print(f"Skipping {branch}")
                continue

            print(f"Collecting and running tests from {TEST_FILE}...")
            results[branch] = run_tests_verbose(TEST_FILE)

            print(f"  Passed: {len(results[branch]['passed'])}")
            print(f"  Skipped: {len(results[branch]['skipped'])}")
            print(f"  Failed: {len(results[branch]['failed'])}")

        checkout_branch(original_branch)

        if "main" in results and "model-test-refactor" in results:
            compare_results(results["main"], results["model-test-refactor"])
        else:
            print("Could not compare - missing results from one or both branches")

    finally:
        print("\nRestoring stashed changes...")
        pop_stash()

        checkout_branch(original_branch)


if __name__ == "__main__":
    main()
