"""
Suggest tests that the regular PR CI does not cover.

Reads the JSON list of a PR's changed files (as returned by
``GET /repos/{repo}/pulls/{number}/files``) from stdin, keeps the changes that
touch core library code under ``models/``, ``pipelines/``, ``modular_pipelines/``
and ``loaders/``, maps each to its mirrored test directory, and prints a Markdown
comment body suggesting the maintainer run those slow tests before merging. The
regular PR CI only runs fast CPU tests, so these are otherwise never exercised on
a PR.

The mirrored directories are the deterministic *candidates*. When ``HF_TOKEN``
is set, a model (via HF Inference Providers) prunes them to the subset the diff
actually exercises; its output is validated against the candidate list, so an
invented path can never leak into the suggestion. Any failure falls back to the
full candidate list, so the comment always posts.

Prints nothing (and exits 0) when the PR touches none of those source roots.
"""

import json
import os
import sys


# Source root -> mirrored test root. Only changes under these roots are considered.
ROOTS = {
    "src/diffusers/models/": "tests/models/",
    "src/diffusers/pipelines/": "tests/pipelines/",
    "src/diffusers/modular_pipelines/": "tests/modular_pipelines/",
}

# `loaders/` is flat but mixes LoRA, single-file and IP-adapter loaders, each with
# a different suite, so it can't be mirrored — map explicitly per file. Files not
# listed (generic helpers, textual inversion which has no dedicated test dir) are
# skipped rather than pointed at a too-broad suite.
LOADERS_ROOT = "src/diffusers/loaders/"
LOADER_TESTS = {
    "peft.py": "tests/lora/",
    "lora_base.py": "tests/lora/",
    "lora_pipeline.py": "tests/lora/",
    "lora_conversion_utils.py": "tests/lora/",
    "unet_loader_utils.py": "tests/lora/",
    "unet.py": "tests/lora/",
    "transformer_flux.py": "tests/lora/",
    "transformer_sd3.py": "tests/lora/",
    "single_file.py": "tests/single_file/",
    "single_file_model.py": "tests/single_file/",
    "single_file_utils.py": "tests/single_file/",
    "ip_adapter.py": "tests/pipelines/ip_adapters/",
}

# Hidden marker so the workflow can find its own sticky comment.
MARKER = "<!-- suggest-tests-bot -->"

# Default instruct model served through HF Inference Providers (overridable via HF_MODEL).
DEFAULT_MODEL = "Qwen/Qwen3.5-35B-A3B"

# Cap each file's patch so a large PR stays within the model's context window.
MAX_PATCH_CHARS = 4000


def test_dir_for(filename: str) -> str | None:
    """Map a changed source file to the test directory that mirrors it.

    ``src/diffusers/pipelines/flux/pipeline_flux.py`` -> ``tests/pipelines/flux/``
    ``src/diffusers/models/transformers/transformer_flux.py`` -> ``tests/models/transformers/``
    ``src/diffusers/models/attention.py`` -> ``tests/models/``
    ``src/diffusers/loaders/peft.py`` -> ``tests/lora/`` (explicit, see LOADER_TESTS)
    """
    if filename.startswith(LOADERS_ROOT):
        return LOADER_TESTS.get(filename[len(LOADERS_ROOT) :])

    for src_root, test_root in ROOTS.items():
        if not filename.startswith(src_root):
            continue
        relative = filename[len(src_root) :]
        head = relative.split("/", 1)
        # A file inside a sub-package maps to that sub-package's test directory;
        # a file directly under the root maps to the root test directory.
        return test_root + head[0] + "/" if len(head) > 1 else test_root
    return None


def prune_with_model(candidates: list[str], diff: str) -> list[str]:
    """Rank the candidate test directories by relevance to the diff, most relevant first.

    Returns an ordered subset of ``candidates``; any path the model invents is
    dropped, and an empty result falls back to all candidates.
    """
    from huggingface_hub import InferenceClient

    prompt = (
        "You are triaging which slow tests to run before merging a PR to the diffusers library.\n"
        "You are given the PR diff (core library changes only) and a list of candidate test "
        "directories derived from the changed files.\n"
        "Select the subset most worth running, ordered most relevant first. Choose ONLY from the "
        "candidate list, copy each path verbatim, and drop directories the diff does not "
        'meaningfully exercise.\n'
        'Respond with JSON only, of the form {"tests": ["tests/.../", ...]}.\n\n'
        f"Candidate test directories:\n" + "\n".join(candidates) + f"\n\nDiff:\n{diff}"
    )
    client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    completion = client.chat.completions.create(
        model=os.environ.get("HF_MODEL", DEFAULT_MODEL),
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    selected = json.loads(completion.choices[0].message.content)["tests"]
    allowed = set(candidates)
    pruned = [path for path in selected if path in allowed]
    return pruned or candidates


def main():
    pr_files = json.load(sys.stdin)
    changed = {f["filename"] for f in pr_files if f["filename"].endswith(".py")}

    candidates = set()
    for filename in changed:
        test_dir = test_dir_for(filename)
        if test_dir is None:
            continue
        # Keep the directory if it already exists on disk (base checkout) or the
        # PR itself adds tests there (a brand-new pipeline ships src + tests).
        if os.path.isdir(test_dir) or any(f.startswith(test_dir) for f in changed):
            candidates.add(test_dir)

    if not candidates:
        return

    candidates = sorted(candidates)
    test_dirs = candidates

    # Assemble the diff for just the source files that produced candidates, then let a
    # small model prune to the relevant subset. Any failure keeps the full candidate list.
    diff = "\n\n".join(
        f"--- {f['filename']}\n{f['patch'][:MAX_PATCH_CHARS]}"
        for f in pr_files
        if test_dir_for(f["filename"]) is not None and f.get("patch")
    )
    if os.environ.get("HF_TOKEN") and diff:
        try:
            test_dirs = prune_with_model(candidates, diff)
        except Exception as error:  # noqa: BLE001 — never let the model break the comment
            print(f"suggest_tests: model pruning failed, using all candidates ({error})", file=sys.stderr)

    bullets = "\n".join(f"* `{test_dir}`" for test_dir in test_dirs)
    print(
        f"""{MARKER}
Suggested slow tests to run _before_ merge:

{bullets}"""
    )


if __name__ == "__main__":
    main()
