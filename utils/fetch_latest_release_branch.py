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


import requests
from packaging.version import parse


# GitHub repository details
USER = "huggingface"
REPO = "diffusers"


def fetch_all_branches(user, repo):
    branches = []  # List to store all branches
    page = 1  # Start from first page
    while True:
        # Make a request to the GitHub API for the branches
        response = requests.get(
            f"https://api.github.com/repos/{user}/{repo}/branches",
            params={"page": page},
            timeout=60,
        )

        # Check if the request was successful
        if response.status_code == 200:
            # Add the branches from the current page to the list
            branches.extend([branch["name"] for branch in response.json()])

            # Check if there is a 'next' link for pagination
            if "next" in response.links:
                page += 1  # Move to the next page
            else:
                break  # Exit loop if there is no next page
        else:
            print("Failed to retrieve branches:", response.status_code)
            break

    return branches


def main():
    # Fetch all branches
    branches = fetch_all_branches(USER, REPO)

    # Filter branches.
    # print(f"Total branches: {len(branches)}")
    filtered_branches = []
    for branch in branches:
        if branch.startswith("v") and ("-release" in branch or "-patch" in branch):
            filtered_branches.append(branch)
            # print(f"Filtered: {branch}")

    sorted_branches = sorted(filtered_branches, key=lambda x: parse(x.split("-")[0][1:]), reverse=True)
    latest_branch = sorted_branches[0]
    # print(f"Latest branch: {latest_branch}")
    return latest_branch


if __name__ == "__main__":
    print(main())
