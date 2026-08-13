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
"""Check that the agent guides and skills under `.ai/` still hold together.

Skills are installed outside a checkout, so a link that only resolves in the repo is a link an installed skill cannot
follow. This checks the three things that break when files move:

1. No relative link escapes its own directory — those cannot resolve once a skill is installed, so link to a stable URL
   instead.
2. A guide a skill cites as `references/<guide>.md` exists in `.ai/references/`, since that is where the installer
   copies it from.
3. Every other relative link resolves on disk, and each skill's frontmatter `name` matches its directory.

Run `python utils/check_ai.py`.
"""

import re
import sys
from pathlib import Path


AI_DIR = Path(__file__).parent.parent / ".ai"
REFERENCES_DIR = AI_DIR / "references"
LINK = re.compile(r"\]\(([^)]+)\)")
# Fenced code is prose to us — Python like `attn.to_out[0](hidden_states)` otherwise reads as a link.
CODE_FENCE = re.compile(r"^```.*?^```", re.M | re.S)
# Matches how the installer finds the guides a skill needs, so this checks exactly what gets copied.
CITATION = re.compile(r"(?<![./\w])references/([\w-]+\.md)")
FRONTMATTER_NAME = re.compile(r"^name:\s*(\S+)\s*$", re.M)


def main() -> int:
    problems: list[str] = []

    for path in sorted(AI_DIR.rglob("*.md")):
        where = path.relative_to(AI_DIR.parent)
        prose = CODE_FENCE.sub("", path.read_text())

        for link in LINK.findall(prose):
            target = link.split("#")[0]
            if not target or target.startswith(("http://", "https://")):
                continue
            if ".." in Path(target).parts:
                problems.append(f"{where}: '{link}' escapes its directory — link to a stable URL instead")
            elif path.name == "SKILL.md" and target.startswith("references/"):
                continue  # covered below, which also catches guides cited outside a markdown link
            elif not (path.parent / target).exists():
                problems.append(f"{where}: '{link}' does not exist")

        if path.name == "SKILL.md":
            for guide in sorted(set(CITATION.findall(prose))):
                if not (REFERENCES_DIR / guide).exists():
                    problems.append(f"{where}: cites 'references/{guide}', which is not in {REFERENCES_DIR.name}/")

    for skill_md in sorted(AI_DIR.glob("skills/*/SKILL.md")):
        declared = FRONTMATTER_NAME.search(skill_md.read_text())
        if declared is None:
            problems.append(f"{skill_md.relative_to(AI_DIR.parent)}: no 'name' in frontmatter")
        elif declared.group(1) != skill_md.parent.name:
            problems.append(
                f"{skill_md.relative_to(AI_DIR.parent)}: frontmatter name '{declared.group(1)}' "
                f"does not match directory '{skill_md.parent.name}'"
            )

    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
