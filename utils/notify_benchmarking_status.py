# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import requests


# Configuration
GITHUB_REPO = "huggingface/diffusers"
GITHUB_RUN_ID = os.getenv("GITHUB_RUN_ID")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def main(args):
    action_url = f"https://github.com/{GITHUB_REPO}/actions/runs/{GITHUB_RUN_ID}"
    if args.status == "success":
        hub_path = "https://huggingface.co/datasets/diffusers/benchmarks/blob/main/collated_results.csv"
        message = (
            "✅ New benchmark workflow successfully run.\n"
            f"🕸️ GitHub Action URL: {action_url}.\n"
            f"🤗 Check out the benchmarks here: {hub_path}."
        )
    else:
        message = (
            "❌ Something wrong happened in the benchmarking workflow.\n"
            f"Check out the GitHub Action to know more: {action_url}."
        )

    payload = {"text": message}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)

    if response.status_code == 200:
        print("Notification sent to Slack successfully.")
    else:
        print("Failed to send notification to Slack.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=str, default="success", choices=["success", "failure"])
    args = parser.parse_args()
    main(args)
