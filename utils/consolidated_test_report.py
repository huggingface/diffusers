#!/usr/bin/env python
import argparse
import glob
import os
import re
from datetime import date, datetime

from slack_sdk import WebClient
from tabulate import tabulate


MAX_LEN_MESSAGE = 3001  # slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--slack_channel_name", default="diffusers-ci-nightly")
parser.add_argument(
    "--reports_dir",
    default="reports",
    help="Directory containing test reports (will search recursively in all subdirectories)",
)
parser.add_argument("--output_file", default=None, help="Path to save the consolidated report (markdown format)")


def parse_stats_file(file_path):
    """Parse a stats file to extract test statistics."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

            # Extract the numbers using regex
            tests_pattern = r"collected (\d+) items"
            passed_pattern = r"(\d+) passed"
            failed_pattern = r"(\d+) failed"
            skipped_pattern = r"(\d+) skipped"
            xpassed_pattern = r"(\d+) xpassed"

            tests_match = re.search(tests_pattern, content)
            passed_match = re.search(passed_pattern, content)
            failed_match = re.search(failed_pattern, content)
            skipped_match = re.search(skipped_pattern, content)
            xpassed_match = re.search(xpassed_pattern, content)

            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            skipped = int(skipped_match.group(1)) if skipped_match else 0
            xpassed = int(xpassed_match.group(1)) if xpassed_match else 0

            # If tests_match exists, use it, otherwise calculate from passed/failed/skipped
            if tests_match:
                tests = int(tests_match.group(1))
            else:
                tests = passed + failed + skipped + xpassed

            # Extract timing information if available
            timing_pattern = r"slowest \d+ test durations[\s\S]*?\n([\s\S]*?)={70}"
            timing_match = re.search(timing_pattern, content, re.MULTILINE)
            slowest_tests = []

            if timing_match:
                timing_text = timing_match.group(1).strip()
                test_timing_lines = timing_text.split("\n")
                for line in test_timing_lines:
                    if line.strip():
                        # Format is typically: 10.37s call     tests/path/to/test.py::TestClass::test_method
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            time_str = parts[0]
                            test_path = " ".join(parts[2:])

                            # Skip entries with "< 0.05 secs were omitted" or similar
                            if "secs were omitted" in test_path:
                                continue

                            try:
                                time_seconds = float(time_str.rstrip("s"))
                                slowest_tests.append({"test": test_path, "duration": time_seconds})
                            except ValueError:
                                pass

            return {
                "tests": tests,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "slowest_tests": slowest_tests,
            }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {"tests": 0, "passed": 0, "failed": 0, "skipped": 0, "slowest_tests": []}


def parse_durations_file(file_path):
    """Parse a durations file to extract test timing information."""
    slowest_tests = []
    try:
        durations_file = file_path.replace("_stats.txt", "_durations.txt")
        if os.path.exists(durations_file):
            with open(durations_file, "r") as f:
                content = f.read()

                # Skip the header line
                for line in content.split("\n")[1:]:
                    if line.strip():
                        # Format is typically: 10.37s call     tests/path/to/test.py::TestClass::test_method
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            time_str = parts[0]
                            test_path = " ".join(parts[2:])

                            # Skip entries with "< 0.05 secs were omitted" or similar
                            if "secs were omitted" in test_path:
                                continue

                            try:
                                time_seconds = float(time_str.rstrip("s"))
                                slowest_tests.append({"test": test_path, "duration": time_seconds})
                            except ValueError:
                                # If time_str is not a valid float, it might be a different format
                                # For example, some pytest formats show "< 0.05s" or similar
                                if test_path.startswith("<") and "secs were omitted" in test_path:
                                    # Extract the time value from test_path if it's in the format "< 0.05 secs were omitted"
                                    try:
                                        # This handles entries where the time is in the test_path itself
                                        dur_match = re.search(r"(\d+(?:\.\d+)?)", test_path)
                                        if dur_match:
                                            time_seconds = float(dur_match.group(1))
                                            slowest_tests.append({"test": test_path, "duration": time_seconds})
                                    except ValueError:
                                        pass
    except Exception as e:
        print(f"Error parsing durations file {file_path.replace('_stats.txt', '_durations.txt')}: {e}")

    return slowest_tests


def parse_failures_file(file_path):
    """Parse a failures file to extract failed test details."""
    failures = []
    try:
        with open(file_path, "r") as f:
            content = f.read()

            # We don't need the base file name anymore as we're getting test paths from summary

            # Check if it's a short stack format
            if "============================= FAILURES SHORT STACK =============================" in content:
                # First, look for pytest-style failure headers with underscores and clean them up
                test_headers = re.findall(r"_{5,}\s+([^_\n]+?)\s+_{5,}", content)

                for test_name in test_headers:
                    test_name = test_name.strip()
                    # Make sure it's a valid test name (contains a dot and doesn't look like a number)
                    if "." in test_name and not test_name.replace(".", "").isdigit():
                        # For test names missing the full path, check if we can reconstruct it from failures_line.txt
                        # This is a best effort - we won't always have the line file available
                        if not test_name.endswith(".py") and "::" not in test_name and "/" not in test_name:
                            # Try to look for a corresponding line file
                            line_file = file_path.replace("_failures_short.txt", "_failures_line.txt")
                            if os.path.exists(line_file):
                                try:
                                    with open(line_file, "r") as lf:
                                        line_content = lf.read()
                                        # Look for test name in line file which might have the full path
                                        path_match = re.search(
                                            r"(tests/[\w/]+\.py::[^:]+::" + test_name.split(".")[-1] + ")",
                                            line_content,
                                        )
                                        if path_match:
                                            test_name = path_match.group(1)
                                except Exception:
                                    pass  # If we can't read the line file, just use what we have

                        failures.append(
                            {
                                "test": test_name,
                                "error": "Error occurred",
                                "original_test_name": test_name,  # Keep original for reference
                            }
                        )

                # If we didn't find any pytest-style headers, try other formats
                if not failures:
                    # Look for test names at the beginning of the file (in first few lines)
                    first_lines = content.split("\n")[:20]  # Look at first 20 lines
                    for line in first_lines:
                        # Look for test names in various formats
                        # Format: tests/file.py::TestClass::test_method
                        path_match = re.search(r"(tests/[\w/]+\.py::[\w\.]+::\w+)", line)
                        # Format: TestClass.test_method
                        class_match = re.search(r"([A-Za-z][A-Za-z0-9_]+\.[A-Za-z][A-Za-z0-9_]+)", line)

                        if path_match:
                            test_name = path_match.group(1)
                            failures.append(
                                {"test": test_name, "error": "Error occurred", "original_test_name": test_name}
                            )
                            break  # Found a full path, stop looking
                        elif class_match and "test" in line.lower():
                            test_name = class_match.group(1)
                            # Make sure it's likely a test name (contains test in method name)
                            if "test" in test_name.lower():
                                failures.append(
                                    {"test": test_name, "error": "Error occurred", "original_test_name": test_name}
                                )
            else:
                # Standard format - try to extract from standard pytest output
                failure_blocks = re.split(r"={70}", content)

                for block in failure_blocks:
                    if not block.strip():
                        continue

                    # Look for test paths in the format: path/to/test.py::TestClass::test_method
                    path_matches = re.findall(r"([\w/]+\.py::[\w\.]+::\w+)", block)
                    if path_matches:
                        for test_name in path_matches:
                            failures.append(
                                {"test": test_name, "error": "Error occurred", "original_test_name": test_name}
                            )
                    else:
                        # Try alternative format: TestClass.test_method
                        class_matches = re.findall(r"([A-Za-z][A-Za-z0-9_]+\.[A-Za-z][A-Za-z0-9_]+)", block)
                        for test_name in class_matches:
                            # Filter out things that don't look like test names
                            if (
                                not test_name.startswith(("e.g", "i.e", "etc."))
                                and not test_name.isdigit()
                                and "test" in test_name.lower()
                            ):
                                failures.append(
                                    {"test": test_name, "error": "Error occurred", "original_test_name": test_name}
                                )

    except Exception as e:
        print(f"Error parsing failures in {file_path}: {e}")

    return failures


def consolidate_reports(reports_dir):
    """Consolidate test reports from multiple test runs, including from subdirectories."""
    # Get all stats files, including those in subdirectories
    stats_files = glob.glob(f"{reports_dir}/**/*_stats.txt", recursive=True)

    results = {}
    total_stats = {"tests": 0, "passed": 0, "failed": 0, "skipped": 0}

    # Collect all slow tests across all test suites
    all_slow_tests = []

    # Process each stats file and its corresponding failures file
    for stats_file in stats_files:
        # Extract test suite name from filename (e.g., tests_pipeline_allegro_cuda_stats.txt -> pipeline_allegro_cuda)
        base_name = os.path.basename(stats_file).replace("_stats.txt", "")

        # Include parent directory in suite name if it's in a subdirectory
        rel_path = os.path.relpath(os.path.dirname(stats_file), reports_dir)
        if rel_path and rel_path != ".":
            # Remove 'test_reports' suffix from directory name if present
            dir_name = os.path.basename(rel_path)
            if dir_name.endswith("_test_reports"):
                dir_name = dir_name[:-13]  # Remove '_test_reports' suffix
            base_name = f"{dir_name}/{base_name}"

        # Parse stats
        stats = parse_stats_file(stats_file)

        # If no slowest tests found in stats file, try the durations file directly
        if not stats.get("slowest_tests"):
            stats["slowest_tests"] = parse_durations_file(stats_file)

        # Update total stats
        for key in ["tests", "passed", "failed", "skipped"]:
            total_stats[key] += stats[key]

        # Collect slowest tests with their suite name
        for slow_test in stats.get("slowest_tests", []):
            all_slow_tests.append({"test": slow_test["test"], "duration": slow_test["duration"], "suite": base_name})

        # Parse failures if there are any
        failures = []
        if stats["failed"] > 0:
            # First try to get test paths from summary_short.txt which has the best format
            summary_file = stats_file.replace("_stats.txt", "_summary_short.txt")
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, "r") as f:
                        content = f.read()
                        # Look for full lines with test path and error message: "FAILED test_path - error_msg"
                        failed_test_lines = re.findall(
                            r"FAILED\s+(tests/[\w/]+\.py::[A-Za-z0-9_\.]+::[A-Za-z0-9_]+)(?:\s+-\s+(.+))?", content
                        )

                        if failed_test_lines:
                            for match in failed_test_lines:
                                test_path = match[0]
                                error_msg = match[1] if len(match) > 1 and match[1] else "No error message"

                                failures.append({"test": test_path, "error": error_msg})
                except Exception as e:
                    print(f"Error parsing summary file: {e}")

            # If no failures found in summary, try other failure files
            if not failures:
                failure_patterns = ["_failures_short.txt", "_failures.txt", "_failures_line.txt", "_failures_long.txt"]

                for pattern in failure_patterns:
                    failures_file = stats_file.replace("_stats.txt", pattern)
                    if os.path.exists(failures_file):
                        failures = parse_failures_file(failures_file)
                        if failures:
                            break

                # No debug output needed

        # Store results for this test suite
        results[base_name] = {"stats": stats, "failures": failures}

    # Filter out entries with "secs were omitted"
    filtered_slow_tests = [test for test in all_slow_tests if "secs were omitted" not in test["test"]]

    # Sort all slow tests by duration (descending)
    filtered_slow_tests.sort(key=lambda x: x["duration"], reverse=True)

    # Get the number of slowest tests to show from environment variable or default to 10
    num_slowest_tests = int(os.environ.get("SHOW_SLOWEST_TESTS", "10"))
    top_slowest_tests = filtered_slow_tests[:num_slowest_tests] if filtered_slow_tests else []

    # Calculate additional duration statistics
    total_duration = sum(test["duration"] for test in all_slow_tests)

    # Calculate duration per suite
    suite_durations = {}
    for test in all_slow_tests:
        suite_name = test["suite"]
        if suite_name not in suite_durations:
            suite_durations[suite_name] = 0
        suite_durations[suite_name] += test["duration"]

    # Removed duration categories

    return {
        "total_stats": total_stats,
        "test_suites": results,
        "slowest_tests": top_slowest_tests,
        "duration_stats": {"total_duration": total_duration, "suite_durations": suite_durations},
    }


def generate_report(consolidated_data):
    """Generate a comprehensive markdown report from consolidated data."""
    report = []

    # Add report header
    report.append("# Diffusers Nightly Test Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Removed comparison section

    # Add summary section
    total = consolidated_data["total_stats"]
    report.append("## Summary")

    # Get duration stats if available
    duration_stats = consolidated_data.get("duration_stats", {})
    total_duration = duration_stats.get("total_duration", 0)

    summary_table = [
        ["Total Tests", total["tests"]],
        ["Passed", total["passed"]],
        ["Failed", total["failed"]],
        ["Skipped", total["skipped"]],
        ["Success Rate", f"{(total['passed'] / total['tests'] * 100):.2f}%" if total["tests"] > 0 else "N/A"],
        ["Total Duration", f"{total_duration:.2f}s" if total_duration else "N/A"],
    ]

    report.append(tabulate(summary_table, tablefmt="pipe"))
    report.append("")

    # Removed duration distribution section

    # Add test suites summary
    report.append("## Test Suites")

    # Include duration in test suites table if available
    suite_durations = consolidated_data.get("duration_stats", {}).get("suite_durations", {})

    if suite_durations:
        suites_table = [["Test Suite", "Tests", "Passed", "Failed", "Skipped", "Success Rate", "Duration (s)"]]
    else:
        suites_table = [["Test Suite", "Tests", "Passed", "Failed", "Skipped", "Success Rate"]]

    # Sort test suites by success rate (ascending - least successful first)
    sorted_suites = sorted(
        consolidated_data["test_suites"].items(),
        key=lambda x: (x[1]["stats"]["passed"] / x[1]["stats"]["tests"] * 100) if x[1]["stats"]["tests"] > 0 else 0,
        reverse=False,
    )

    for suite_name, suite_data in sorted_suites:
        stats = suite_data["stats"]
        success_rate = f"{(stats['passed'] / stats['tests'] * 100):.2f}%" if stats["tests"] > 0 else "N/A"

        if suite_durations:
            duration = suite_durations.get(suite_name, 0)
            suites_table.append(
                [
                    suite_name,
                    stats["tests"],
                    stats["passed"],
                    stats["failed"],
                    stats["skipped"],
                    success_rate,
                    f"{duration:.2f}",
                ]
            )
        else:
            suites_table.append(
                [suite_name, stats["tests"], stats["passed"], stats["failed"], stats["skipped"], success_rate]
            )

    report.append(tabulate(suites_table, headers="firstrow", tablefmt="pipe"))
    report.append("")

    # Add slowest tests section
    slowest_tests = consolidated_data.get("slowest_tests", [])
    if slowest_tests:
        report.append("## Slowest Tests")

        slowest_table = [["Rank", "Test", "Duration (s)", "Test Suite"]]
        for i, test in enumerate(slowest_tests, 1):
            # Skip entries that don't contain actual test names
            if "< 0.05 secs were omitted" in test["test"]:
                continue
            slowest_table.append([i, test["test"], f"{test['duration']:.2f}", test["suite"]])

        report.append(tabulate(slowest_table, headers="firstrow", tablefmt="pipe"))
        report.append("")

    # Add failures section if there are any
    failed_suites = [s for s in sorted_suites if s[1]["stats"]["failed"] > 0]

    if failed_suites:
        report.append("## Failures")

        # Group failures by module for cleaner organization
        failures_by_module = {}

        for suite_name, suite_data in failed_suites:
            # Extract failures data for this suite
            for failure in suite_data.get("failures", []):
                test_name = failure["test"]

                # If test name doesn't look like a full path, try to reconstruct it
                if not ("/" in test_name or "::" in test_name) and "." in test_name:
                    # For simple 'TestClass.test_method' format, try to get full path from suite name
                    # Form: tests_<suite>_cuda -> tests/<suite>/test_<suite>.py::TestClass::test_method
                    if suite_name.startswith("tests_") and "_cuda" in suite_name:
                        # Extract component name from suite
                        component = suite_name.replace("tests_", "").replace("_cuda", "")
                        if "." in test_name:
                            class_name, method_name = test_name.split(".", 1)
                            possible_path = f"tests/{component}/test_{component}.py::{class_name}::{method_name}"
                            # Use this constructed path if it seems reasonable
                            if "test_" in method_name:
                                test_name = possible_path

                # Extract module name from test name
                if "::" in test_name:
                    # For path/file.py::TestClass::test_method format
                    parts = test_name.split("::")
                    module_name = parts[-2] if len(parts) >= 2 else "Other"  # TestClass
                elif "." in test_name:
                    # For TestClass.test_method format
                    parts = test_name.split(".")
                    module_name = parts[0]  # TestClass
                else:
                    module_name = "Other"

                # Skip module names that don't look like class/module names
                if (
                    module_name.startswith(("e.g", "i.e", "etc"))
                    or module_name.replace(".", "").isdigit()
                    or len(module_name) < 3
                ):
                    module_name = "Other"

                # Add to the module group
                if module_name not in failures_by_module:
                    failures_by_module[module_name] = []

                # Prepend the suite name if the test name doesn't already have a full path
                if "/" not in test_name and suite_name not in test_name:
                    full_test_name = f"{suite_name}::{test_name}"
                else:
                    full_test_name = test_name

                # Add this failure to the module group
                failures_by_module[module_name].append(
                    {"test": full_test_name, "original_test": test_name, "error": failure["error"]}
                )

        # Create a list of failing tests for each module
        if failures_by_module:
            for module_name, failures in sorted(failures_by_module.items()):
                report.append(f"### {module_name}")

                # Put all failed tests in a single code block
                report.append("```")
                for failure in failures:
                    # Show test path and error message if available
                    if failure.get("error") and failure["error"] != "No error message":
                        report.append(f"{failure['test']} - {failure['error']}")
                    else:
                        report.append(failure["test"])
                report.append("```")

                report.append("")  # Add space between modules
        else:
            report.append("*No detailed failure information available*")
            report.append("")

    return "\n".join(report)


def create_test_groups_table(test_groups, total_tests, total_success_rate):
    """Create a table-like format for test groups showing total tests and success rate."""
    if not test_groups:
        return None

    # Sort by total test count (descending)
    sorted_groups = sorted(test_groups.items(), key=lambda x: x[1]["total"], reverse=True)

    # Create table lines
    table_lines = ["```"]
    table_lines.append("Test Results Summary")
    table_lines.append("-------------------")
    table_lines.append(f"Total Tests:  {total_tests:,}")
    table_lines.append(f"Success Rate: {total_success_rate}")
    table_lines.append("")
    table_lines.append("Category            | Total Tests | Failed | Success Rate")
    table_lines.append("------------------- | ----------- | ------ | ------------")

    # Add rows
    for category, stats in sorted_groups:
        # Pad category name to fixed width (19 chars)
        padded_cat = category[:19].ljust(19)  # Truncate if too long
        # Right-align counts
        padded_total = str(stats["total"]).rjust(11)
        padded_failed = str(stats["failed"]).rjust(6)
        # Calculate and format success rate
        if stats["total"] > 0:
            cat_success_rate = f"{((stats['total'] - stats['failed']) / stats['total'] * 100):.1f}%"
        else:
            cat_success_rate = "N/A"
        padded_rate = cat_success_rate.rjust(12)
        table_lines.append(f"{padded_cat} | {padded_total} | {padded_failed} | {padded_rate}")

    table_lines.append("```")

    total_failures = sum(stats["failed"] for stats in test_groups.values())
    return (
        f"*Test Groups Summary ({total_failures} {'failure' if total_failures == 1 else 'failures'}):*\n"
        + "\n".join(table_lines)
    )


def create_slack_payload(consolidated_data):
    """Create a concise Slack message payload from consolidated data."""
    total = consolidated_data["total_stats"]
    success_rate = f"{(total['passed'] / total['tests'] * 100):.2f}%" if total["tests"] > 0 else "N/A"

    # Determine emoji based on success rate
    if total["failed"] == 0:
        emoji = "✅"
    elif total["failed"] / total["tests"] < 0.1:
        emoji = "⚠️"
    else:
        emoji = "❌"

    # Create a more compact summary section
    summary = f"{emoji} *Diffusers Nightly Tests:* {success_rate} success ({total['passed']}/{total['tests']} tests"
    if total["skipped"] > 0:
        summary += f", {total['skipped']} skipped"
    summary += ")"

    # Create the test suites table in markdown format
    # Build the markdown table with proper alignment
    table_lines = []
    table_lines.append("```")

    # Sort test suites by success rate (ascending - least successful first)
    sorted_suites = sorted(
        consolidated_data["test_suites"].items(),
        key=lambda x: (x[1]["stats"]["passed"] / x[1]["stats"]["tests"] * 100) if x[1]["stats"]["tests"] > 0 else 0,
        reverse=False,
    )

    # Calculate max widths for proper alignment
    max_suite_name_len = max(len(suite_name) for suite_name, _ in sorted_suites) if sorted_suites else 10
    max_suite_name_len = max(max_suite_name_len, len("Test Suite"))  # Ensure header fits

    # Create header with proper spacing (only Tests, Failed, Success Rate)
    header = f"| {'Test Suite'.ljust(max_suite_name_len)} | {'Tests'.rjust(6)} | {'Failed'.rjust(6)} | {'Success Rate'.ljust(12)} |"
    separator = f"|:{'-' * max_suite_name_len}|{'-' * 7}:|{'-' * 7}:|:{'-' * 11}|"

    table_lines.append(header)
    table_lines.append(separator)

    # Add data rows with proper alignment
    for suite_name, suite_data in sorted_suites:
        stats = suite_data["stats"]
        suite_success_rate = f"{(stats['passed'] / stats['tests'] * 100):.2f}%" if stats["tests"] > 0 else "N/A"

        row = f"| {suite_name.ljust(max_suite_name_len)} | {str(stats['tests']).rjust(6)} | {str(stats['failed']).rjust(6)} | {suite_success_rate.ljust(12)} |"

        table_lines.append(row)

    table_lines.append("```")

    # Create the Slack payload with character limit enforcement
    payload = [
        {"type": "section", "text": {"type": "mrkdwn", "text": summary}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(table_lines)}},
    ]

    # Add action button
    if os.environ.get("GITHUB_RUN_ID"):
        run_id = os.environ["GITHUB_RUN_ID"]
        payload.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*<https://github.com/huggingface/diffusers/actions/runs/{run_id}|View full report on GitHub>*",
                },
            }
        )

    # Add date in more compact form
    payload.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "plain_text",
                    "text": f"Results for {date.today()}",
                },
            ],
        }
    )

    # Enforce 3001 character limit
    payload_text = str(payload)
    if len(payload_text) > MAX_LEN_MESSAGE:
        # Truncate table if payload is too long
        # Remove rows from the bottom until under limit
        original_table_lines = table_lines[:]
        while len(str(payload)) > MAX_LEN_MESSAGE and len(table_lines) > 3:  # Keep at least header and separator
            # Remove the last data row (but keep ``` at the end)
            table_lines.pop(-2)  # Remove second to last (last is the closing ```)

            # Recreate payload with truncated table
            payload[1] = {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(table_lines)}}

        # Add note if we had to truncate
        if len(table_lines) < len(original_table_lines):
            truncated_count = len(original_table_lines) - len(table_lines)
            table_lines.insert(-1, f"... {truncated_count} more test suites (truncated due to message limit)")
            payload[1] = {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(table_lines)}}

    return payload


def create_failed_tests_by_suite_ordered(consolidated_data):
    """Group failed tests by test suite, ordered by success rate (ascending)."""
    # Sort test suites by success rate (ascending - least successful first)
    sorted_suites = sorted(
        consolidated_data["test_suites"].items(),
        key=lambda x: (x[1]["stats"]["passed"] / x[1]["stats"]["tests"] * 100) if x[1]["stats"]["tests"] > 0 else 0,
        reverse=False,
    )

    failed_suite_tests = []

    # Process suites in order of success rate
    for suite_name, suite_data in sorted_suites:
        if suite_data["stats"]["failed"] > 0:
            suite_failures = []

            for failure in suite_data.get("failures", []):
                test_name = failure["test"]

                # Try to reconstruct full path if partial
                if "::" in test_name and "/" in test_name:
                    full_test_name = test_name
                elif "::" in test_name or "." in test_name:
                    if "/" not in test_name and suite_name not in test_name:
                        full_test_name = f"{suite_name}::{test_name}"
                    else:
                        full_test_name = test_name
                else:
                    full_test_name = f"{suite_name}::{test_name}"

                suite_failures.append(full_test_name)

            # Sort and deduplicate tests within the suite
            suite_failures = sorted(set(suite_failures))

            if suite_failures:
                failed_suite_tests.append(
                    {
                        "suite_name": suite_name,
                        "tests": suite_failures,
                        "success_rate": (suite_data["stats"]["passed"] / suite_data["stats"]["tests"] * 100)
                        if suite_data["stats"]["tests"] > 0
                        else 0,
                    }
                )

    return failed_suite_tests


def main(args):
    # Make sure reports directory exists
    if not os.path.isdir(args.reports_dir):
        print(f"Error: Reports directory '{args.reports_dir}' does not exist.")
        return

    # Consolidate reports
    consolidated_data = consolidate_reports(args.reports_dir)

    # Check if we found any test results
    if consolidated_data["total_stats"]["tests"] == 0:
        print(f"Warning: No test results found in '{args.reports_dir}' or its subdirectories.")

    # Generate markdown report
    report = generate_report(consolidated_data)

    # Save report to file if specified
    if args.output_file:
        # Create parent directories if they don't exist
        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(args.output_file, "w") as f:
            f.write(report)

        # Only print the report when saving to file
        print(report)

    # Send to Slack if token is available (optional, can be disabled)
    slack_token = os.environ.get("SLACK_API_TOKEN")
    if slack_token and args.slack_channel_name:
        payload = create_slack_payload(consolidated_data)

        try:
            client = WebClient(token=slack_token)
            # Send main message
            response = client.chat_postMessage(channel=f"#{args.slack_channel_name}", blocks=payload)
            print(f"Report sent to Slack channel: {args.slack_channel_name}")

            # Send failed tests as separate threaded replies grouped by test suite (ordered by success rate)
            total = consolidated_data["total_stats"]
            if total["failed"] > 0:
                failed_suites = create_failed_tests_by_suite_ordered(consolidated_data)
                for suite_info in failed_suites:
                    suite_name = suite_info["suite_name"]
                    suite_tests = suite_info["tests"]
                    success_rate = suite_info["success_rate"]
                    message_text = (
                        f"**{suite_name}** (Success Rate: {success_rate:.2f}%)\n```\n"
                        + "\n".join(suite_tests)
                        + "\n```"
                    )
                    client.chat_postMessage(
                        channel=f"#{args.slack_channel_name}",
                        thread_ts=response["ts"],  # Reply in thread
                        text=message_text,  # Use text instead of blocks for markdown
                    )
                print(f"Failed tests details sent as {len(failed_suites)} thread replies")
        except Exception as e:
            print(f"Error sending report to Slack: {e}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
