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
- If a PR is not linked and no prior reminder is present, the script posts a single
  friendly reminder comment.
- PRs labeled `no-issue-needed` and bot-authored PRs are skipped.
- PRs authored by maintainers, users with write (or admin) access, and collaborators
  are skipped; the reminder only targets external contributors.
"""

import logging
import os
import re
from datetime import datetime, timedelta, timezone

import requests
from github import Github


logger = logging.getLogger(__name__)

REPO = "huggingface/diffusers"
REMINDER_MARKER = "<!-- pr-link-issue-reminder -->"
BYPASS_LABELS = {"no-issue-needed"}
LOOKBACK_DAYS = 2
# Collaborator permission levels that mark a PR author as a maintainer / writer /
# collaborator. Authors with any of these are skipped (the reminder is only for
# external contributors).
PRIVILEGED_PERMISSIONS = {"admin", "write", "maintain", "triage"}

# `author_association` values that mark the author as a maintainer / collaborator.
# These are available on the PR payload without needing extra token scopes.
PRIVILEGED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}

# A PR authored by the model/pipeline's own team does not need to link an issue.
# Matches a checked task-list item for the corresponding PR template checkbox.
AUTHOR_CHECKBOX_PATTERN = re.compile(
    r"-\s*\[\s*[xX]\s*\]\s*Are you the author \(or part of the team\) of the model/pipeline"
)
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
    data = payload.get("data")
    if not data:
        return False
    return data["repository"]["pullRequest"]["closingIssuesReferences"]["totalCount"] > 0


def author_checkbox_checked(pr):
    return bool(AUTHOR_CHECKBOX_PATTERN.search(pr.body or ""))


def has_existing_reminder(pr):
    return any(REMINDER_MARKER in (c.body or "") for c in pr.get_issue_comments())


def is_privileged_author(repo, pr, author):
    """Return True if the author is a maintainer, has write/admin access, or is a collaborator."""
    # `author_association` is on the PR payload and needs no extra token scope.
    association = (pr.raw_data or {}).get("author_association")
    if association in PRIVILEGED_ASSOCIATIONS:
        return True
    # Fall back to the collaborator-permission API to catch writers/collaborators
    # whose association is reported as CONTRIBUTOR/NONE on this particular PR.
    try:
        permission = repo.get_collaborator_permission(author)
    except Exception as e:
        # A 404 here means the user is not a collaborator at all (external contributor).
        logger.info("Could not resolve permission for @%s, treating as external: %s", author, e)
        return False
    return permission in PRIVILEGED_PERMISSIONS


def reminder_body(author):
    return (
        f"{REMINDER_MARKER}\n"
        f"Hi @{author}, thanks for the PR! It does not appear to link an issue it fixes. "
        "If this PR addresses an existing issue, please add a closing keyword "
        "(e.g. `Fixes #1234`) to the PR description so the issue is linked. "
        f"See the [contribution guide]({CONTRIBUTION_GUIDE_URL}) for more details. "
        "If this PR intentionally does not fix a tracked issue, a maintainer can "
        "add the `no-issue-needed` label to silence this reminder."
    )


def main():
    token = os.environ["GITHUB_TOKEN"]
    g = Github(token)
    repo = g.get_repo(REPO)
    owner, name = REPO.split("/", 1)
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    try:
        pulls = repo.get_pulls(state="open", sort="created", direction="desc")
        for pr in pulls:
            try:
                created_at = pr.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                # PRs are sorted newest-first, so once we cross the cutoff every
                # remaining PR is older too and we can stop paginating.
                if created_at < cutoff:
                    break
                if pr.draft:
                    continue
                if pr.user is None:
                    continue
                author = pr.user.login
                if not author or author.endswith("[bot]") or pr.user.type == "Bot":
                    continue
                if is_privileged_author(repo, pr, author):
                    continue
                labels = {label.name for label in pr.labels}
                if labels & BYPASS_LABELS:
                    continue
                if author_checkbox_checked(pr):
                    continue
                if has_linked_issue(token, owner, name, pr.number):
                    continue
                if has_existing_reminder(pr):
                    continue
                pr.create_issue_comment(reminder_body(author))
            except Exception as e:
                logger.warning("Skipping PR #%s: %s", getattr(pr, "number", "?"), e)
                continue
    except Exception as e:
        logger.error("Failed to fetch open PRs: %s", e)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
