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
Tell Slack when a link in the docs to another website stops working.

The script takes three file paths:

1. The link check report from `doc-builder check-links --check-external --format list`.
2. The list of broken links from last week. This file is missing the first time.
3. Where to save this week's list of broken links for next week.

The script only posts about a broken link once. It stays quiet while the same
links are still broken. It posts once when all the links work again. It also
posts once if the check fails without naming a broken link, so someone can look
at the run log.

A link is matched by its file and URL, not its line number. Moving a broken link
to another line doesn't make it count as new.
"""

import json
import os
import re
import sys
import urllib.request


REPORT_LINE = re.compile(r"^(\S+?):\d+ - (https?://\S+)")
MAX_LISTED = 20
CHECK_FAILED = ("-", "check-failed")


def broken_links(report: str) -> set[tuple[str, str]]:
    """Return the file and URL of each broken link in the report.

    Links listed under "Could not verify" are left out. The check couldn't tell
    whether they work, so they don't fail it.
    """
    broken_section = report.split("Could not verify", 1)[0]
    return {(m.group(1), m.group(2)) for line in broken_section.splitlines() if (m := REPORT_LINE.match(line))}


def read_state(path: str) -> set[tuple[str, str]]:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {tuple(line.split(" ", 1)) for line in f.read().splitlines() if " " in line}


def write_state(path: str, links: set[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(f"{file} {url}\n" for file, url in sorted(links))


def escape(text: str) -> str:
    """Replace &, < and > so Slack shows them as plain text instead of formatting."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def format_links(links: set[tuple[str, str]]) -> str:
    lines = [f"• `{escape(file)}` {escape(url)}" for file, url in sorted(links)[:MAX_LISTED]]
    if len(links) > MAX_LISTED:
        lines.append(f"…and {len(links) - MAX_LISTED} more. The run summary lists all of them.")
    return "\n".join(lines)


def post_to_slack(webhook_url: str, text: str) -> None:
    request = urllib.request.Request(
        webhook_url, data=json.dumps({"text": text}).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


def main() -> None:
    report_path, previous_path, new_path = sys.argv[1:4]
    outcome = os.environ["OUTCOME"]
    run_url = os.environ["RUN_URL"]
    webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    with open(report_path, encoding="utf-8") as f:
        broken = broken_links(f.read())
    previous = read_state(previous_path)

    if outcome == "failure" and not broken:
        broken = previous | {CHECK_FAILED}
        text = f"The docs link check failed without listing a broken external link. <{run_url}|Run log>"
        if CHECK_FAILED in previous:
            print("The check failed again without listing a broken external link. Posting nothing.")
            write_state(new_path, broken)
            return
    elif outcome == "success" and previous:
        text = f"Every external link in the docs passes again. <{run_url}|Run>"
    elif new := broken - previous:
        text = (
            f"{len(new)} external link(s) in the docs broke since the last check. <{run_url}|Run>\n{format_links(new)}"
        )
    else:
        print(f"No newly broken links ({len(broken)} still broken). Posting nothing.")
        write_state(new_path, broken)
        return

    if not webhook_url:
        print(f"SLACK_WEBHOOK_URL is empty. Would have posted:\n{text}")
        write_state(new_path, previous)
        return

    post_to_slack(webhook_url, text)
    print(f"Posted to Slack:\n{text}")
    write_state(new_path, broken)


if __name__ == "__main__":
    main()
