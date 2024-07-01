# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import os

import requests


# Configuration
LIBRARY_NAME = "diffusers"
GITHUB_REPO = "huggingface/diffusers"
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")


def check_pypi_for_latest_release(library_name):
    """Check PyPI for the latest release of the library."""
    response = requests.get(f"https://pypi.org/pypi/{library_name}/json")
    if response.status_code == 200:
        data = response.json()
        return data["info"]["version"]
    else:
        print("Failed to fetch library details from PyPI.")
        return None


def get_github_release_info(github_repo):
    """Fetch the latest release info from GitHub."""
    url = f"https://api.github.com/repos/{github_repo}/releases/latest"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return {"tag_name": data["tag_name"], "url": data["html_url"], "release_time": data["published_at"]}

    else:
        print("Failed to fetch release info from GitHub.")
        return None


def notify_slack(webhook_url, library_name, version, release_info):
    """Send a notification to a Slack channel."""
    message = (
        f"🚀 New release for {library_name} available: version **{version}** 🎉\n"
        f"📜 Release Notes: {release_info['url']}\n"
        f"⏱️ Release time: {release_info['release_time']}"
    )
    payload = {"text": message}
    response = requests.post(webhook_url, json=payload)

    if response.status_code == 200:
        print("Notification sent to Slack successfully.")
    else:
        print("Failed to send notification to Slack.")


def main():
    latest_version = check_pypi_for_latest_release(LIBRARY_NAME)
    release_info = get_github_release_info(GITHUB_REPO)
    parsed_version = release_info["tag_name"].replace("v", "")

    if latest_version and release_info and latest_version == parsed_version:
        notify_slack(SLACK_WEBHOOK_URL, LIBRARY_NAME, latest_version, release_info)
    else:
        print(f"{latest_version=}, {release_info=}, {parsed_version=}")
        raise ValueError("There were some problems.")


if __name__ == "__main__":
    main()
