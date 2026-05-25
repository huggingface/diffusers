# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Diffusers tests_fetcher (graph-based).

For each PR, walk the AST of every modified Python file to extract diffusers-internal imports, build a
forward dependency graph for the repo, invert it to a reverse map (file → tests transitively depending on
it), and select the impacted tests.

There is no automatic full-suite trigger. If a change is in territory the import graph can't see
correctly (dynamic dispatch via auto-mappings, lazy `_import_structure`, etc.), apply the `run-all-tests`
PR label or pass `--force_full_suite` to bypass selection.

Pipeline-specific note: diffusers' `__init__.py` files use the `_import_structure = {...}` lazy-loading
pattern paired with an `if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:` block containing real
`from .submodule import Class` statements. AST extraction sees the TYPE_CHECKING imports (since
`ast.walk` descends into `If` / `Try` blocks regardless of runtime conditions), so the import graph
mirrors the actual public API.

Stage 1 — diff: list modified Python files (vs. merge-base with main, or the previous commit on main).
    Docstring/comment-only changes are filtered out by content comparison.
Stage 2 — graph: parse every .py under `src/diffusers/` and `tests/` with `ast`, build the forward
    dependency map, transitively close it, then invert to the reverse map.
Stage 3 — select: for each modified file, look up `reverse_map[file]` to get impacted tests.
Stage 4 — bucket: group tests by top-level `tests/` folder for the CI matrix.

Usage:

```bash
python utils/tests_fetcher.py                         # PR mode: diff against main
python utils/tests_fetcher.py --diff_with_last_commit # main mode: diff against last commit
python utils/tests_fetcher.py --force_full_suite      # bypass selection, run everything
```
"""

import argparse
import ast
import collections
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
PATH_TO_DIFFUSERS = PATH_TO_REPO / "src/diffusers"
PATH_TO_TESTS = PATH_TO_REPO / "tests"

# Top-level test folders excluded from the matrix. `lora` runs as its own dedicated job in pr_tests.yml;
# `fixtures` is data, not tests.
MODULES_TO_IGNORE = {"fixtures", "lora"}

# ============================================================
# Generic helpers
# ============================================================


@contextmanager
def checkout_commit(repo: Repo, commit_id: str):
    """Check out `commit_id` for the duration of the block, restoring the prior HEAD on exit."""
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref
    try:
        repo.git.checkout(commit_id)
        yield
    finally:
        repo.git.checkout(current_head)


# ============================================================
# Diff detection
# ============================================================


def _strip_comments_and_docstrings(source: str) -> str:
    """Return source with all docstrings and comments removed via AST round-trip.

    Used by `diff_is_docstring_only` to detect diffs that are purely cosmetic.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    for node in ast.walk(tree):
        # Strip module/class/function docstrings.
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body.pop(0)
    return ast.unparse(tree)


def diff_is_docstring_only(repo: Repo, branching_point, filename: str) -> bool:
    """True if the diff in `filename` between `branching_point` and HEAD only changes docstrings/comments."""
    with checkout_commit(repo, branching_point):
        old_content = (PATH_TO_REPO / filename).read_text(encoding="utf-8")
    new_content = (PATH_TO_REPO / filename).read_text(encoding="utf-8")
    return _strip_comments_and_docstrings(old_content) == _strip_comments_and_docstrings(new_content)


def get_diff(repo: Repo, base_commit, commits) -> List[str]:
    """Return Python files changed between `commits` (branching point) and `base_commit` (HEAD)."""
    code_diff = []
    for commit in commits:
        for d in commit.diff(base_commit):
            paths = [p for p in (d.a_path, d.b_path) if p and p.endswith(".py")]
            if not paths:
                continue
            # Add/delete/rename: keep every changed path verbatim. Pure modification: skip if the diff is
            # docstring/comment-only.
            if d.change_type in ("A", "D") or d.a_path != d.b_path:
                code_diff.extend(paths)
            elif not diff_is_docstring_only(repo, commit, d.b_path):
                code_diff.append(d.b_path)
    return code_diff


def get_modified_python_files(diff_with_last_commit: bool = False) -> List[str]:
    """List Python files modified between HEAD and either main (default) or the previous commit."""
    repo = Repo(PATH_TO_REPO)
    if diff_with_last_commit:
        base_label = "previous commit"
        commits = repo.head.commit.parents
    else:
        upstream_main = repo.remotes.origin.refs.main
        base_label = f"merge-base with main ({upstream_main.commit})"
        commits = repo.merge_base(upstream_main, repo.head)
    print(f"Diffing HEAD ({repo.head.commit}) against {base_label}: {[str(c) for c in commits]}")
    return get_diff(repo, repo.head.commit, commits)


def get_all_tests() -> List[str]:
    """Top-level entries under `tests/` (folders + `tests/test_*.py`), used to expand a full-suite selection."""
    return sorted(
        f"tests/{p.name}"
        for p in PATH_TO_TESTS.iterdir()
        if "__pycache__" not in p.name and (p.is_dir() or p.name.startswith("test_"))
    )


# ============================================================
# AST-based import extraction
# ============================================================


def _resolve_import(module: Optional[str], level: int, importer_pkg: List[str]) -> Optional[List[str]]:
    """Resolve an `ImportFrom` node to a list of repo-rooted path parts.

    Args:
        module: the `X.Y` part of `from X.Y import Z` (None for `from . import Z`).
        level: number of leading dots (0 for absolute, 1+ for relative).
        importer_pkg: parts of the importing module's *package* (parent dir parts), e.g.
            `["src", "diffusers", "pipelines", "flux"]` for `pipelines/flux/pipeline_flux.py`.

    Returns:
        Path parts like `["src", "diffusers", "pipelines", "flux", "pipeline_flux"]` (no extension),
        or None if the import is external or can't be resolved.
    """
    if level == 0:
        if module is None or not (module == "diffusers" or module.startswith("diffusers.")):
            return None
        sub = module.split(".")[1:]
        return ["src", "diffusers", *sub]

    if level > len(importer_pkg):
        return None
    base = importer_pkg[: len(importer_pkg) - level + 1]
    if module:
        return [*base, *module.split(".")]
    return base


def _to_module_file(path_parts: List[str]) -> Optional[str]:
    """Resolve `path_parts` to either `<parts>.py` or `<parts>/__init__.py`. Returns repo-relative path."""
    candidate = PATH_TO_REPO.joinpath(*path_parts).with_suffix(".py")
    if candidate.is_file():
        return str(candidate.relative_to(PATH_TO_REPO))
    init = PATH_TO_REPO.joinpath(*path_parts) / "__init__.py"
    if init.is_file():
        return str(init.relative_to(PATH_TO_REPO))
    return None


def _iter_module_level_imports(node):
    """Yield `ImportFrom` nodes that execute at module load.

    Recurses into `If` / `Try` / `ClassDef` bodies (those run at import time) but stops at
    `FunctionDef` / `AsyncFunctionDef` / `Lambda` boundaries — imports inside function bodies are
    deferred runtime imports (e.g. lazy stubs in deprecation shims) and shouldn't count as dependencies.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    if isinstance(node, ast.ImportFrom):
        yield node
    for child in ast.iter_child_nodes(node):
        yield from _iter_module_level_imports(child)


def _extract_imports(module_file: str) -> List[Tuple[str, List[str]]]:
    """Parse `module_file` and return [(target_file, [imported_symbols]), ...] for diffusers-internal imports.

    Only module-level `from X import ...` statements are considered. Bare `import X`, `from X import *`,
    external imports (transformers, torch, stdlib), and imports inside function bodies are skipped.
    """
    abs_path = PATH_TO_REPO / module_file
    try:
        source = abs_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        return []

    importer_pkg = list(Path(module_file).parts[:-1])

    results: List[Tuple[str, List[str]]] = []
    for node in _iter_module_level_imports(tree):
        names = [alias.name for alias in node.names if alias.name != "*"]
        if not names:
            continue
        target_parts = _resolve_import(node.module, node.level, importer_pkg)
        if target_parts is None:
            continue
        target_file = _to_module_file(target_parts)
        if target_file is None:
            continue
        results.append((target_file, names))

    return results


# ============================================================
# Dependency graph
# ============================================================


def get_module_dependencies(module_file: str, cache: Dict[str, List[Tuple[str, List[str]]]]) -> List[str]:
    """Return source files `module_file` truly depends on, traversing inits to find the defining file.

    When an import lands on an `__init__.py`, walk its imports too, matching by symbol name to find the
    actual submodule that re-exports each requested symbol. This collapses
    `from diffusers import StableDiffusionPipeline` to `pipelines/stable_diffusion/pipeline_stable_diffusion.py`
    instead of the root init (which would over-select to almost every test).
    """
    if module_file not in cache:
        cache[module_file] = _extract_imports(module_file)

    dependencies: List[str] = []
    queue: List[Tuple[str, List[str]]] = list(cache[module_file])
    seen_inits: set = set()

    while queue:
        target, symbols = queue.pop(0)

        if not target.endswith("__init__.py"):
            dependencies.append(target)
            continue

        # Avoid cycles through inits importing each other.
        if target in seen_inits:
            dependencies.append(target)
            continue
        seen_inits.add(target)

        if target not in cache:
            cache[target] = _extract_imports(target)
        init_imports = cache[target]

        unresolved = list(symbols)
        for sub_target, sub_names in init_imports:
            matched = [s for s in unresolved if s in sub_names]
            if matched:
                queue.append((sub_target, matched))
                unresolved = [s for s in unresolved if s not in matched]

        if unresolved:
            # Symbol(s) couldn't be resolved through the init's TYPE_CHECKING imports — likely lazy-loaded
            # via `_import_structure` or defined directly in the init. Keep the init as the dep (coarse but
            # correct: changes to the init will trigger this module).
            dependencies.append(target)

    return list(set(dependencies))


def _merged_nested_deps(m: str, direct_deps: Dict[str, List[str]]) -> bool:
    """Pull each of m's deps' deps into m. Returns True if m grew.

    Skips `__init__.py` targets — they re-export the entire package surface, so expanding through
    them would pull in every diffusers symbol via the root init.
    """
    merged = False
    for d in list(direct_deps[m]):
        if d.endswith("__init__.py"):
            continue
        new_deps = set(direct_deps[d]) - set(direct_deps[m])
        if new_deps:
            direct_deps[m].extend(new_deps)
            merged = True
    return merged


def create_reverse_dependency_map() -> Dict[str, List[str]]:
    """Build the reverse dependency map: file → list of files that transitively depend on it.

    1. Compute direct deps for every .py under `src/diffusers/` and `tests/`.
    2. Transitively close (skipping inits during recursion to avoid pulling in the universe via the root init).
    3. Invert.
    """
    cache: Dict[str, List[Tuple[str, List[str]]]] = {}
    all_modules = [
        str(p.relative_to(PATH_TO_REPO))
        for p in list(PATH_TO_DIFFUSERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    ]
    direct_deps: Dict[str, List[str]] = {m: get_module_dependencies(m, cache) for m in all_modules}

    # Each pass propagates dependency info one level deeper. Loop until a full pass adds nothing.
    while any([_merged_nested_deps(m, direct_deps) for m in all_modules]):
        pass

    reverse_map: Dict[str, List[str]] = collections.defaultdict(list)
    for m in all_modules:
        for d in direct_deps[m]:
            reverse_map[d].append(m)

    # For inits, do the forward direction: editing an init impacts everything it re-exports.
    for init in [m for m in all_modules if m.endswith("__init__.py")]:
        deps = get_module_dependencies(init, cache)
        impacted = set(deps)
        for d in deps:
            if not d.endswith("__init__.py"):
                impacted.update(reverse_map.get(d, []))
        reverse_map[init] = sorted(impacted - {init})

    return dict(reverse_map)


# ============================================================
# Test selection
# ============================================================


def _bucket_for_matrix(test_paths: List[str]) -> Dict[str, List[str]]:
    """Group test paths by top-level folder under `tests/`. Files directly under `tests/` go to `common`."""
    test_map: Dict[str, List[str]] = collections.defaultdict(list)
    for p in test_paths:
        parts = p.split("/")
        if len(parts) < 2 or parts[0] != "tests":
            continue
        top = parts[1]
        if top in MODULES_TO_IGNORE:
            continue
        bucket = "common" if len(parts) == 2 else top
        test_map[bucket].append(p)
    return {k: sorted(set(v)) for k, v in test_map.items()}


def _is_test_file(path: str) -> bool:
    """True if `path` is a `tests/.../test_*.py` file (the kind pytest collects)."""
    return path.startswith("tests/") and Path(path).name.startswith("test_")


def fetch_tests_to_run(json_output_file: str, diff_with_last_commit: bool):
    """Determine the tests to run from the diff and write `test_map.json`."""
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    reverse_map = create_reverse_dependency_map()

    # Each modified file contributes itself (if it's a test) plus tests transitively impacted by it.
    selected = set()
    for f in modified_files:
        if _is_test_file(f):
            selected.add(f)
        selected.update(t for t in reverse_map.get(f, []) if _is_test_file(t))

    test_files_to_run = sorted(p for p in selected if (PATH_TO_REPO / p).exists())

    test_map = _bucket_for_matrix(test_files_to_run)
    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump({k: " ".join(v) for k, v in test_map.items()}, fp, ensure_ascii=False)


def _all_test_files() -> List[str]:
    """Enumerate every `tests/.../test_*.py` (used for the full-suite path)."""
    return sorted(
        str(p.relative_to(PATH_TO_REPO)) for p in PATH_TO_TESTS.glob("**/test_*.py") if "__pycache__" not in p.parts
    )


def _write_full_suite(json_output_file: str):
    """Schedule the entire test suite. Used by `--force_full_suite` and as exception fallback."""
    test_map = _bucket_for_matrix(_all_test_files())
    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump({k: " ".join(v) for k, v in test_map.items()}, fp, ensure_ascii=False)


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_output_file",
        type=str,
        default="test_map.json",
        help="Where to store the category → test paths matrix consumed by CI.",
    )
    parser.add_argument(
        "--diff_with_last_commit",
        action="store_true",
        help="Diff against the previous commit instead of main (use on main branch jobs)",
    )
    parser.add_argument(
        "--force_full_suite",
        action="store_true",
        help="Bypass selection and write outputs that schedule the entire test suite.",
    )
    args = parser.parse_args()

    if args.force_full_suite:
        print("Forcing full test suite.")
        _write_full_suite(args.json_output_file)
        raise SystemExit(0)

    repo = Repo(PATH_TO_REPO)
    diff_with_last_commit = args.diff_with_last_commit
    if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref == repo.refs.main:
        print("main branch detected, fetching tests against last commit.")
        diff_with_last_commit = True

    try:
        fetch_tests_to_run(args.json_output_file, diff_with_last_commit)
    except Exception as e:
        import traceback

        print(f"\nError when trying to grab the relevant tests: {e}\n")
        traceback.print_exc()
        print("\nRunning all tests.")
        _write_full_suite(args.json_output_file)
