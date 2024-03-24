# Copyright 2024 The HuggingFace Team, the AllenNLP library authors. All rights reserved.
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
Script to close stale issue. Taken in part from the AllenNLP repository.
https://github.com/allenai/allennlp.
"""

import os
from datetime import datetime as dt
from datetime import timezone

from github import Github


LABELS_TO_EXEMPT = [
    "good first issue",
    "good second issue",
    "good difficult issue",
    "enhancement",
    "new pipeline/model",
    "new scheduler",
    "wip",
]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/diffusers")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        labels = [label.name.lower() for label in issue.get_labels()]
        if "stale" in labels:
            comments = sorted(issue.get_comments(), key=lambda i: i.created_at, reverse=True)
            last_comment = comments[0] if len(comments) > 0 else None
            if last_comment is not None and last_comment.user.login != "github-actions[bot]":
                # Opens the issue if someone other than Stalebot commented.
                issue.edit(state="open")
                issue.remove_from_labels("stale")
        elif (
            (dt.now(timezone.utc) - issue.updated_at).days > 23
            and (dt.now(timezone.utc) - issue.created_at).days >= 30
            and not any(label in LABELS_TO_EXEMPT for label in labels)
        ):
            # Post a Stalebot notification after 23 days of inactivity.
            issue.create_comment(
                "This issue has been automatically marked as stale because it has not had "
                "recent activity. If you think this still needs to be addressed "
                "please comment on this thread.\n\nPlease note that issues that do not follow the "
                "[contributing guidelines](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) "
                "are likely to be ignored."
            )
            issue.add_to_labels("stale")


if __name__ == "__main__":
    main()
