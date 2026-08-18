# Copyright 2026 The HuggingFace Team. All rights reserved.
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
Script to remind PR authors to link an issue, and to escalate unresolved reminders.

Behavior:
- Scans open, non-draft PRs.
- A PR is considered "linked" if GitHub's GraphQL `closingIssuesReferences` returns > 0
  (covers both `Fixes #N` keywords in the body and issues linked via the GitHub UI).
- If a PR is not linked and has no reminder yet, the script posts a single friendly
  reminder comment warning that the PR may be auto-closed.
- If a PR only has an old-style reminder (posted before the auto-close notice existed),
  the script posts a single follow-up carrying the notice instead, so those PRs enter
  the same escalation path with the full warning window.
- Once the warning is `SLACK_DIGEST_DAYS` old and the PR is still not linked, the PR is
  included in a daily Slack digest so maintainers can rescue it (by linking an issue or
  adding the `no-issue-needed` label).
- Once the warning is `AUTOCLOSE_DAYS` old and the PR is still not linked, the PR is
  closed with an explanatory comment.
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
# Present only in comments that warn about auto-closure; the escalation clock starts
# from the bot comment carrying this marker.
AUTOCLOSE_MARKER = "<!-- pr-link-issue-autoclose -->"
# Login the reminder comments are authored under (the workflow's GITHUB_TOKEN).
BOT_LOGIN = "github-actions[bot]"
BYPASS_LABELS = {"no-issue-needed"}
# Days after the warning at which a still-unlinked PR enters the Slack rescue digest.
SLACK_DIGEST_DAYS = 7
# Days after the warning at which a still-unlinked PR is automatically closed.
AUTOCLOSE_DAYS = 10
# Upper bound on how far back to paginate open PRs; older PRs are left alone.
SCAN_LOOKBACK_DAYS = 30
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
        f"{AUTOCLOSE_MARKER}\n"
        f"Hi @{author}, thanks for the PR! It does not appear to link an issue it fixes. "
        "If this PR addresses an existing issue, please add a closing keyword "
        "(e.g. `Fixes #1234`) to the PR description so the issue is linked. "
        f"See the [contribution guide]({CONTRIBUTION_GUIDE_URL}) for more details. "
        "If this PR intentionally does not fix a tracked issue, a maintainer can "
        "add the `no-issue-needed` label to silence this reminder.\n\n"
        f"**Please note that PRs without a linked issue are likely to be automatically "
        f"closed {AUTOCLOSE_DAYS} days after this notice.**"
    )


def followup_body(author):
    return (
        f"{AUTOCLOSE_MARKER}\n"
        f"Hi @{author}, a follow-up on the reminder above: this PR still does not link "
        "an issue it fixes.\n\n"
        f"**Please note that PRs without a linked issue are likely to be automatically "
        f"closed {AUTOCLOSE_DAYS} days after this notice.** Adding a closing keyword "
        "(e.g. `Fixes #1234`) to the PR description, or a maintainer adding the "
        "`no-issue-needed` label, will prevent that."
    )


def autoclose_body():
    return (
        "This PR has been automatically closed because it does not link an issue and "
        f"the reminder above was not addressed within {AUTOCLOSE_DAYS} days. "
        "If this PR is still relevant, please link the issue it fixes "
        "(e.g. `Fixes #1234`) or ask a maintainer to add the `no-issue-needed` label, "
        "and it can be reopened."
    )


def send_slack_digest(webhook_url, mention_ids, at_risk):
    lines = []
    if mention_ids:
        lines.append("cc " + " ".join(f"<@{mid}>" for mid in mention_ids))
    lines.append(
        f"⚠️ {len(at_risk)} open PR(s) without a linked issue will be auto-closed soon. "
        "Link an issue or add the `no-issue-needed` label to rescue them:"
    )
    for pr, days_left in at_risk:
        lines.append(f"• <{pr.html_url}|#{pr.number} {pr.title}> by `{pr.user.login}` — closes in {days_left} day(s)")
    response = requests.post(webhook_url, json={"text": "\n".join(lines)}, timeout=30)
    response.raise_for_status()


def main():
    token = os.environ["GITHUB_TOKEN"]
    slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    # Comma-separated Slack member IDs (e.g. "U0123ABC,U0456DEF") pinged in the digest.
    mention_ids = [m.strip() for m in os.getenv("SLACK_MENTION_IDS", "").split(",") if m.strip()]
    g = Github(token)
    repo = g.get_repo(REPO)
    owner, name = REPO.split("/", 1)
    now = datetime.now(timezone.utc)
    scan_cutoff = now - timedelta(days=SCAN_LOOKBACK_DAYS)
    # (pr, days_left) pairs for the Slack rescue digest.
    at_risk = []

    try:
        pulls = repo.get_pulls(state="open", sort="created", direction="desc")
        for pr in pulls:
            try:
                created_at = pr.created_at
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                # PRs are sorted newest-first, so once we cross the scan cutoff every
                # remaining PR is older too and we can stop paginating.
                if created_at < scan_cutoff:
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
                comments = list(pr.get_issue_comments())
                # Only a marker comment authored by the bot itself starts the
                # escalation clock; a pasted marker in a user comment does not.
                warning = next(
                    (
                        c
                        for c in comments
                        if AUTOCLOSE_MARKER in (c.body or "") and c.user is not None and c.user.login == BOT_LOGIN
                    ),
                    None,
                )
                if warning is None:
                    # A PR with only an old-style reminder (without the auto-close
                    # notice) gets a follow-up carrying the notice; the escalation
                    # clock starts from whichever comment carries the marker.
                    already_reminded = any(REMINDER_MARKER in (c.body or "") for c in comments)
                    pr.create_issue_comment(followup_body(author) if already_reminded else reminder_body(author))
                    continue
                warned_at = warning.created_at
                if warned_at.tzinfo is None:
                    warned_at = warned_at.replace(tzinfo=timezone.utc)
                days_since_warning = (now - warned_at).days
                if days_since_warning >= AUTOCLOSE_DAYS:
                    pr.create_issue_comment(autoclose_body())
                    pr.edit(state="closed")
                elif days_since_warning >= SLACK_DIGEST_DAYS:
                    at_risk.append((pr, AUTOCLOSE_DAYS - days_since_warning))
            except Exception as e:
                logger.warning("Skipping PR #%s: %s", getattr(pr, "number", "?"), e)
                continue
    except Exception as e:
        logger.error("Failed to fetch open PRs: %s", e)
        raise

    if at_risk:
        if slack_webhook_url:
            send_slack_digest(slack_webhook_url, mention_ids, at_risk)
        else:
            logger.warning("SLACK_WEBHOOK_URL is not set; skipping digest for %d at-risk PR(s).", len(at_risk))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
