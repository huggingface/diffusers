#!/usr/bin/env python
import argparse
import glob
import json
import os
import re
import tempfile
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

from huggingface_hub import HfApi, login, snapshot_download, upload_file
from slack_sdk import WebClient
from tabulate import tabulate


MAX_LEN_MESSAGE = 2900  # slack endpoint has a limit of 3001 characters

parser = argparse.ArgumentParser()
parser.add_argument("--slack_channel_name", default="diffusers-ci-nightly")
parser.add_argument("--reports_dir", default="reports", help="Directory containing test reports")
parser.add_argument("--output_file", default=None, help="Path to save the consolidated report (markdown format)")
parser.add_argument("--hf_dataset_repo", default=None, help="Hugging Face dataset repository to store reports")
parser.add_argument("--upload_to_hub", action="store_true", help="Whether to upload the report to Hugging Face Hub")
parser.add_argument("--compare_with_previous", action="store_true", help="Compare with the previous report from Hub")


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
            
            tests_match = re.search(tests_pattern, content)
            passed_match = re.search(passed_pattern, content)
            failed_match = re.search(failed_pattern, content)
            skipped_match = re.search(skipped_pattern, content)
            
            tests = int(tests_match.group(1)) if tests_match else 0
            passed = int(passed_match.group(1)) if passed_match else 0
            failed = int(failed_match.group(1)) if failed_match else 0
            skipped = int(skipped_match.group(1)) if skipped_match else 0
            
            # Extract timing information if available
            timing_pattern = r"slowest \d+ test durations[\s\S]*?\n([\s\S]*?)={70}"
            timing_match = re.search(timing_pattern, content, re.MULTILINE)
            slowest_tests = []
            
            if timing_match:
                timing_text = timing_match.group(1).strip()
                test_timing_lines = timing_text.split('\n')
                for line in test_timing_lines:
                    if line.strip():
                        # Format is typically: 10.37s call     tests/path/to/test.py::TestClass::test_method
                        parts = line.strip().split()
                        if len(parts) >= 3:
                            time_str = parts[0]
                            test_path = ' '.join(parts[2:])
                            try:
                                time_seconds = float(time_str.rstrip('s'))
                                slowest_tests.append({
                                    "test": test_path,
                                    "duration": time_seconds
                                })
                            except ValueError:
                                pass
            
            return {
                "tests": tests,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "slowest_tests": slowest_tests
            }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return {
            "tests": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "slowest_tests": []
        }


def parse_failures_file(file_path):
    """Parse a failures file to extract failed test details."""
    failures = []
    try:
        with open(file_path, "r") as f:
            content = f.read()
            
            # Split content by test failure
            failure_blocks = re.split(r"_{80}", content)
            
            for block in failure_blocks:
                if not block.strip():
                    continue
                    
                # Extract test name and failure message
                test_match = re.search(r"([\w/]+\.py::[\w\.]+::\w+)", block)
                error_match = re.search(r"(E\s+.+?)(\n\n|\Z)", block, re.DOTALL)
                
                if test_match:
                    test_name = test_match.group(1)
                    error_msg = error_match.group(1).strip() if error_match else "Unknown error"
                    failures.append({
                        "test": test_name,
                        "error": error_msg
                    })
    except Exception as e:
        print(f"Error parsing failures in {file_path}: {e}")
    
    return failures


def consolidate_reports(reports_dir):
    """Consolidate test reports from multiple test runs."""
    # Get all stats files
    stats_files = glob.glob(f"{reports_dir}/*_stats.txt")
    
    results = {}
    total_stats = {
        "tests": 0,
        "passed": 0,
        "failed": 0,
        "skipped": 0
    }
    
    # Collect all slow tests across all test suites
    all_slow_tests = []
    
    # Process each stats file and its corresponding failures file
    for stats_file in stats_files:
        # Extract test suite name from filename (e.g., tests_pipeline_allegro_cuda_stats.txt -> pipeline_allegro_cuda)
        base_name = os.path.basename(stats_file).replace("_stats.txt", "")
        
        # Parse stats
        stats = parse_stats_file(stats_file)
        
        # Update total stats
        for key in ["tests", "passed", "failed", "skipped"]:
            total_stats[key] += stats[key]
        
        # Collect slowest tests with their suite name
        for slow_test in stats.get("slowest_tests", []):
            all_slow_tests.append({
                "test": slow_test["test"],
                "duration": slow_test["duration"],
                "suite": base_name
            })
        
        # Parse failures if there are any
        failures = []
        failures_file = stats_file.replace("_stats.txt", "_failures_short.txt")
        if os.path.exists(failures_file) and stats["failed"] > 0:
            failures = parse_failures_file(failures_file)
        
        # Store results for this test suite
        results[base_name] = {
            "stats": stats,
            "failures": failures
        }
    
    # Sort all slow tests by duration (descending)
    all_slow_tests.sort(key=lambda x: x["duration"], reverse=True)
    
    # Get the number of slowest tests to show from environment variable or default to 10
    num_slowest_tests = int(os.environ.get("SHOW_SLOWEST_TESTS", "10"))
    top_slowest_tests = all_slow_tests[:num_slowest_tests] if all_slow_tests else []
    
    return {
        "total_stats": total_stats,
        "test_suites": results,
        "slowest_tests": top_slowest_tests
    }


def generate_report(consolidated_data):
    """Generate a comprehensive markdown report from consolidated data."""
    report = []
    
    # Add report header
    report.append("# Diffusers Nightly Test Report")
    report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Add comparison section if available
    comparison = consolidated_data.get("comparison")
    previous_date = consolidated_data.get("previous_date")
    
    if comparison:
        # Determine comparison header based on previous date
        if previous_date:
            report.append(f"## New Failures Since {previous_date}")
        else:
            report.append("## New Failures")
        
        # New failures
        new_failures = comparison.get("new_failures", [])
        if new_failures:
            report.append(f"🔴 {len(new_failures)} new failing tests compared to previous report:\n")
            for i, test in enumerate(new_failures, 1):
                report.append(f"{i}. `{test}`")
            report.append("")
        else:
            report.append("No new test failures detected! 🎉\n")
    
    # Add summary section
    total = consolidated_data["total_stats"]
    report.append("## Summary")
    
    summary_table = [
        ["Total Tests", total["tests"]],
        ["Passed", total["passed"]],
        ["Failed", total["failed"]],
        ["Skipped", total["skipped"]],
        ["Success Rate", f"{(total['passed'] / total['tests'] * 100):.2f}%" if total["tests"] > 0 else "N/A"]
    ]
    
    report.append(tabulate(summary_table, tablefmt="pipe"))
    report.append("")
    
    # Add test suites summary
    report.append("## Test Suites")
    
    suites_table = [["Test Suite", "Tests", "Passed", "Failed", "Skipped", "Success Rate"]]
    
    # Sort test suites by number of failures (descending)
    sorted_suites = sorted(
        consolidated_data["test_suites"].items(),
        key=lambda x: x[1]["stats"]["failed"],
        reverse=True
    )
    
    for suite_name, suite_data in sorted_suites:
        stats = suite_data["stats"]
        success_rate = f"{(stats['passed'] / stats['tests'] * 100):.2f}%" if stats["tests"] > 0 else "N/A"
        suites_table.append([
            suite_name, 
            stats["tests"], 
            stats["passed"], 
            stats["failed"], 
            stats["skipped"],
            success_rate
        ])
    
    report.append(tabulate(suites_table, headers="firstrow", tablefmt="pipe"))
    report.append("")
    
    # Add slowest tests section
    slowest_tests = consolidated_data.get("slowest_tests", [])
    if slowest_tests:
        num_slowest = len(slowest_tests)
        report.append(f"## Slowest {num_slowest} Tests")
        
        slowest_table = [["Rank", "Test", "Duration (s)", "Test Suite"]]
        for i, test in enumerate(slowest_tests, 1):
            slowest_table.append([
                i,
                test["test"],
                f"{test['duration']:.2f}",
                test["suite"]
            ])
        
        report.append(tabulate(slowest_table, headers="firstrow", tablefmt="pipe"))
        report.append("")
    
    # Add failures section if there are any
    failed_suites = [s for s in sorted_suites if s[1]["stats"]["failed"] > 0]
    
    if failed_suites:
        report.append("## Failures")
        
        for suite_name, suite_data in failed_suites:
            if suite_data["failures"]:
                report.append(f"### {suite_name}")
                
                failures_table = [["Test", "Error"]]
                for failure in suite_data["failures"]:
                    # Truncate long error messages
                    error_msg = failure["error"]
                    if len(error_msg) > 100:
                        error_msg = error_msg[:97] + "..."
                    
                    failures_table.append([failure["test"], error_msg])
                
                report.append(tabulate(failures_table, headers="firstrow", tablefmt="pipe"))
                report.append("")
    
    return "\n".join(report)


def create_slack_payload(consolidated_data):
    """Create a Slack message payload from consolidated data."""
    total = consolidated_data["total_stats"]
    success_rate = f"{(total['passed'] / total['tests'] * 100):.2f}%" if total["tests"] > 0 else "N/A"
    
    # Determine emoji based on success rate
    if total["failed"] == 0:
        emoji = "✅"
    elif total["failed"] / total["tests"] < 0.1:
        emoji = "⚠️"
    else:
        emoji = "❌"
    
    payload = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"{emoji} Diffusers Nightly Test Report"
            }
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Total Tests:* {total['tests']}"},
                {"type": "mrkdwn", "text": f"*Passed:* {total['passed']}"},
                {"type": "mrkdwn", "text": f"*Failed:* {total['failed']}"},
                {"type": "mrkdwn", "text": f"*Skipped:* {total['skipped']}"},
                {"type": "mrkdwn", "text": f"*Success Rate:* {success_rate}"}
            ]
        }
    ]
    
    # Add new failures section if available
    comparison = consolidated_data.get("comparison")
    previous_date = consolidated_data.get("previous_date")
    
    if comparison and "new_failures" in comparison:
        new_failures = comparison["new_failures"]
        
        if previous_date:
            title = f"*New Failures Since {previous_date}:*"
        else:
            title = "*New Failures:*"
            
        if new_failures:
            message = f"{title}\n"
            for i, test in enumerate(new_failures[:10], 1):  # Limit to first 10
                message += f"{i}. `{test}`\n"
                
            if len(new_failures) > 10:
                message += f"_...and {len(new_failures) - 10} more_\n"
                
            payload.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": message}
            })
    
    # Add failed test suites summary
    failed_suites = [(name, data) for name, data in consolidated_data["test_suites"].items() 
                    if data["stats"]["failed"] > 0]
    
    if failed_suites:
        message = "*Failed Test Suites:*\n"
        for suite_name, suite_data in failed_suites:
            message += f"• {suite_name}: {suite_data['stats']['failed']} failed tests\n"
        
        if len(message) > MAX_LEN_MESSAGE:
            message = message[:MAX_LEN_MESSAGE] + "..."
        
        payload.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": message}
        })
    
    # Add slowest tests summary
    slowest_tests = consolidated_data.get("slowest_tests", [])
    if slowest_tests:
        # Take top 5 for Slack message to avoid clutter
        top5_slowest = slowest_tests[:5]
        
        slowest_message = "*Top 5 Slowest Tests:*\n"
        for i, test in enumerate(top5_slowest, 1):
            test_name = test["test"].split("::")[-1] if "::" in test["test"] else test["test"]
            slowest_message += f"{i}. `{test_name}` - {test['duration']:.2f}s ({test['suite']})\n"
        
        payload.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": slowest_message}
        })
    
    # Add action button
    if os.environ.get("GITHUB_RUN_ID"):
        payload.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": "*For more details:*"},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Check Action results", "emoji": True},
                "url": f"https://github.com/huggingface/diffusers/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
        })
    
    # Add date
    payload.append({
        "type": "context",
        "elements": [
            {
                "type": "plain_text",
                "text": f"Nightly test results for {date.today()}",
            },
        ],
    })
    
    return payload


def download_previous_report(repo_id):
    """Download the most recent report from the HF dataset repository."""
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download the repository content
            snapshot_download(repo_id=repo_id, local_dir=tmp_dir, repo_type="dataset")
            
            # Find the most recent report file
            report_files = glob.glob(os.path.join(tmp_dir, "report_*.json"))
            if not report_files:
                print("No previous reports found in the repository.")
                return None, None
            
            # Sort by date (assuming report_YYYY-MM-DD.json format)
            report_files.sort(reverse=True)
            latest_file = report_files[0]
            
            # Extract date from filename (report_YYYY-MM-DD.json)
            report_date = os.path.basename(latest_file).split('.')[0].split('_')[1]
            
            # Read the most recent report
            with open(latest_file, "r") as f:
                return json.load(f), report_date
    except Exception as e:
        print(f"Error downloading previous report: {e}")
        return None, None


def compare_reports(current_data, previous_data):
    """Compare current test results with previous ones to identify new failures."""
    if not previous_data:
        return {
            "new_failures": []
        }
    
    # Get current and previous failed tests
    current_failures = set()
    for suite_name, suite_data in current_data["test_suites"].items():
        for failure in suite_data["failures"]:
            current_failures.add(failure["test"])
    
    previous_failures = set()
    for suite_name, suite_data in previous_data["test_suites"].items():
        for failure in suite_data["failures"]:
            previous_failures.add(failure["test"])
    
    # Find new failures
    new_failures = current_failures - previous_failures
    
    return {
        "new_failures": list(new_failures)
    }


def upload_report_to_hub(data, report_text, repo_id):
    """Upload the report to the Hugging Face Hub dataset repository."""
    try:
        # Check if HF_TOKEN is available
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("HF_TOKEN environment variable not set. Cannot upload to Hub.")
            return False
        
        # Login to Hugging Face
        login(token=hf_token)
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate filename based on current date
            today = date.today().strftime("%Y-%m-%d")
            json_filename = f"report_{today}.json"
            md_filename = f"report_{today}.md"
            
            # Save report as JSON and Markdown
            json_path = os.path.join(tmp_dir, json_filename)
            md_path = os.path.join(tmp_dir, md_filename)
            
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            
            with open(md_path, "w") as f:
                f.write(report_text)
            
            # Upload files to Hub
            upload_file(
                path_or_fileobj=json_path,
                path_in_repo=json_filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            
            upload_file(
                path_or_fileobj=md_path,
                path_in_repo=md_filename,
                repo_id=repo_id,
                repo_type="dataset"
            )
            
            print(f"Report successfully uploaded to {repo_id}")
            return True
    except Exception as e:
        print(f"Error uploading report to Hub: {e}")
        return False


def main(args):
    # Consolidate reports
    consolidated_data = consolidate_reports(args.reports_dir)
    
    # Compare with previous report if requested
    comparison_data = None
    if args.compare_with_previous and args.hf_dataset_repo:
        previous_data, previous_date = download_previous_report(args.hf_dataset_repo)
        if previous_data:
            comparison_data = compare_reports(consolidated_data, previous_data)
            # Add comparison data and previous report date to consolidated data
            consolidated_data["comparison"] = comparison_data
            consolidated_data["previous_date"] = previous_date
    
    # Generate markdown report
    report = generate_report(consolidated_data)
    
    # Print report to stdout
    print(report)
    
    # Save report to file if specified
    if args.output_file:
        with open(args.output_file, "w") as f:
            f.write(report)
    
    # Upload to Hugging Face Hub if requested
    if args.upload_to_hub and args.hf_dataset_repo:
        upload_report_to_hub(consolidated_data, report, args.hf_dataset_repo)
    
    # Send to Slack if token is available
    slack_token = os.environ.get("SLACK_API_TOKEN")
    if slack_token and args.slack_channel_name:
        payload = create_slack_payload(consolidated_data)
        
        try:
            client = WebClient(token=slack_token)
            client.chat_postMessage(channel=f"#{args.slack_channel_name}", blocks=payload)
            print(f"Report sent to Slack channel: {args.slack_channel_name}")
        except Exception as e:
            print(f"Error sending report to Slack: {e}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)