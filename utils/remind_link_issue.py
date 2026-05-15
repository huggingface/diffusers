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
"""
Script to remind PR authors to link an issue.

Behavior:
- Scans open, non-draft PRs.
- A PR is considered "linked" if GitHub's GraphQL `closingIssuesReferences` returns > 0
  (covers both `Fixes #N` keywords in the body and issues linked via the GitHub UI).
- If a PR is not linked, the script posts up to 3 reminder comments spaced 7 days apart.
- PRs labeled `no-issue-needed` and bot-authored PRs are skipped.
"""

import os
from datetime import datetime, timedelta, timezone

import requests
from github import Github


REPO = "huggingface/diffusers"
REMINDER_MARKER = "<!-- pr-link-issue-reminder -->"
REMINDER_INTERVAL = timedelta(days=7)
MAX_REMINDERS = 3
BYPASS_LABELS = {"no-issue-needed"}
CONTRIBUTION_GUIDE_URL = "https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#coding-with-ai-agents"

GRAPHQL_URL = "https://api.github.com/graphql"
GRAPHQL_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      closingIssuesReferences(first: 1) {
        totalCount
      }
    }
  }
}
"""


def has_linked_issue(token, owner, name, number):
    response = requests.post(
        GRAPHQL_URL,
        json={"query": GRAPHQL_QUERY, "variables": {"owner": owner, "name": name, "number": number}},
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["data"]["repository"]["pullRequest"]["closingIssuesReferences"]["totalCount"] > 0


def reminder_history(pr):
    reminders = [c for c in pr.get_issue_comments() if REMINDER_MARKER in (c.body or "")]
    reminders.sort(key=lambda c: c.created_at)
    return reminders


def reminder_body(author, count):
    remaining = MAX_REMINDERS - count
    lines = [
        REMINDER_MARKER,
        f"Hi @{author}, this PR does not appear to link an issue it fixes. "
        "If this PR addresses an existing issue, please add a closing keyword "
        "(e.g. `Fixes #1234`) to the PR description so the issue is linked. "
        f"See the [contribution guide]({CONTRIBUTION_GUIDE_URL}) for more details.",
        "",
        f"Reminder **{count}/{MAX_REMINDERS}**.",
    ]
    if remaining > 0:
        lines[-1] += (
            f" If no linked issue is added within {REMINDER_INTERVAL.days} days, "
            f"you will receive {remaining} more reminder(s)."
        )
    else:
        lines[-1] += (
            " This is the final reminder. If this PR intentionally does not fix "
            "a tracked issue, a maintainer can add the `no-issue-needed` label "
            "to bypass this check."
        )
    return "\n".join(lines)


def aware(ts):
    return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def main():
    token = os.environ["GITHUB_TOKEN"]
    g = Github(token)
    repo = g.get_repo(REPO)
    owner, name = REPO.split("/", 1)

    now = datetime.now(timezone.utc)

    for pr in repo.get_pulls(state="open"):
        if pr.draft:
            continue
        if pr.user is None:
            continue
        author = pr.user.login
        if not author or author.endswith("[bot]") or pr.user.type == "Bot":
            continue
        labels = {label.name for label in pr.labels}
        if labels & BYPASS_LABELS:
            continue
        if has_linked_issue(token, owner, name, pr.number):
            continue

        reminders = reminder_history(pr)
        count = len(reminders)

        if count == 0:
            pr.create_issue_comment(reminder_body(author, 1))
            continue

        if count >= MAX_REMINDERS:
            continue

        if now - aware(reminders[-1].created_at) < REMINDER_INTERVAL:
            continue

        pr.create_issue_comment(reminder_body(author, count + 1))


if __name__ == "__main__":
    main()
