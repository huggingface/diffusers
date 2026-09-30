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
correctly (dynamic dispatch via auto-mappings, lazy `_import_structure`, etc.), pass `--force_full_suite`
to bypass selection.

Pipeline-specific note: diffusers' `__init__.py` files use the `_import_structure = {...}` lazy-loading
pattern paired with an `if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:` block containing real
`from .submodule import Class` statements. The import walker descends into `If` / `Try` blocks regardless
of runtime conditions, so it sees the TYPE_CHECKING imports and the import graph mirrors the actual public
API.

Stage 1 — diff: list modified Python files (vs. merge-base with main, or the previous commit on main).
    Docstring/comment-only changes are filtered out by content comparison.
Stage 2 — graph: parse every .py under `src/diffusers/` and `tests/` with `ast`, build the forward
    dependency map, transitively close it, then invert to the reverse map.
Stage 3 — select: for each modified file, look up `reverse_map[file]` to get impacted tests.
Stage 4 — bucket: group tests by top-level `tests/` folder for the CI matrix. `tests/models` and
    `tests/pipelines` are split further into one job per feature mixin (via the pytest markers the
    mixins carry) so a large selection fans out instead of serialising in one job.
Stage 5 — report: print the import chain that selected each test (kept as a CI artifact) and write a
    markdown job summary of how many tests each modified file triggers.

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
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PATH_TO_REPO))
from tests.conftest import NON_CORE_MARKERS, marker_decorators  # noqa: E402


PATH_TO_DIFFUSERS = PATH_TO_REPO / "src/diffusers"
PATH_TO_TESTS = PATH_TO_REPO / "tests"

# `tests/models` and `tests/pipelines` compose one test class per feature mixin, and each mixin carries a
# pytest marker (see the `is_*` decorators in tests/testing_utils.py). Each group below becomes its own
# matrix job, selected with `pytest -m <expr>`, but only when a selected file composes a mixin carrying
# one of the group's markers. `core` is every test carrying none of `NON_CORE_MARKERS` (tests/conftest.py
# derives that set from the decorators and marks those tests at collection time). Non-core markers with no
# group here are gated on an accelerator or multi-GPU, so they get no CPU job rather than spinning up a
# runner to skip everything.
SPLIT_BY_FEATURE = {"models", "pipelines"}
FEATURE_GROUPS = {
    "lora": ["lora"],
    "attention": ["attention"],
    "memory": ["memory", "cpu_offload", "group_offload"],
    "cache": ["cache"],
    "ip_adapter": ["ip_adapter"],
    # Needs only an HF token (gated checkpoints), which the CPU job provides; skips on fork PRs.
    "single_file": ["single_file"],
}
FEATURE_MARKERS = [m for markers in FEATURE_GROUPS.values() for m in markers]
_unknown = sorted(set(FEATURE_MARKERS) - NON_CORE_MARKERS)
if _unknown:
    raise ValueError(
        f"FEATURE_GROUPS markers {_unknown} have no `is_*` decorator in tests/testing_utils.py, so their tests "
        "would also run in `core`. Add the decorator or drop the group."
    )
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
    return list(dict.fromkeys(code_diff))


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
    seen: set = set()

    while queue:
        target, symbols = queue.pop(0)

        if not target.endswith("__init__.py"):
            dependencies.append(target)
            continue

        # A symbol arriving at an init it already passed through means inits import each other in a cycle;
        # keep the init as the dep instead of looping. Tracked per (init, symbol) so a second statement
        # importing other symbols from the same package still resolves them to their defining modules.
        unseen = [s for s in symbols if (target, s) not in seen]
        if len(unseen) < len(symbols):
            dependencies.append(target)
        if not unseen:
            continue
        seen.update((target, s) for s in unseen)

        if target not in cache:
            cache[target] = _extract_imports(target)
        init_imports = cache[target]

        unresolved = unseen
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


def create_direct_dependency_map() -> Dict[str, List[str]]:
    """Direct (one-hop) deps of every .py under `src/diffusers/` and `tests/`: file → files it imports."""
    cache: Dict[str, List[Tuple[str, List[str]]]] = {}
    all_modules = [
        str(p.relative_to(PATH_TO_REPO))
        for p in list(PATH_TO_DIFFUSERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    ]
    return {m: get_module_dependencies(m, cache) for m in all_modules}


def create_reverse_dependency_map(direct_deps: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Build the reverse dependency map: file → list of files that transitively depend on it.

    1. Transitively close the direct deps (skipping inits during recursion to avoid pulling in the universe
       via the root init).
    2. Invert.
    """
    all_modules = list(direct_deps)
    one_hop_deps = direct_deps
    direct_deps = {m: list(deps) for m, deps in direct_deps.items()}

    # Each pass propagates dependency info one level deeper. Loop until a full pass adds nothing.
    changed = True
    while changed:
        changed = False
        for m in all_modules:
            if _merged_nested_deps(m, direct_deps):
                changed = True

    reverse_map: Dict[str, List[str]] = collections.defaultdict(list)
    for m in all_modules:
        for d in direct_deps[m]:
            reverse_map[d].append(m)

    # For inits, do the forward direction: editing an init impacts everything it re-exports.
    for init in [m for m in all_modules if m.endswith("__init__.py")]:
        deps = one_hop_deps[init]
        impacted = set(deps)
        for d in deps:
            if not d.endswith("__init__.py"):
                impacted.update(reverse_map.get(d, []))
        reverse_map[init] = sorted(impacted - {init})

    return dict(reverse_map)


def _import_chain(test_file: str, modified_files: List[str], direct_deps: Dict[str, List[str]]) -> Optional[List[str]]:
    """Shortest import chain from a modified file to `test_file`, or None if there is none.

    Walks `test_file`'s imports outward with the same rules as the transitive closure: an `__init__.py`
    reached as a dependency is not expanded further. A modified init is reached through the modules it
    re-exports, mirroring the forward direction `create_reverse_dependency_map` applies to inits.
    """
    targets = {f: f for f in modified_files}
    for f in modified_files:
        if f.endswith("__init__.py"):
            for d in direct_deps.get(f, []):
                if not d.endswith("__init__.py"):
                    targets.setdefault(d, f)

    parents: Dict[str, Optional[str]] = {test_file: None}
    queue = collections.deque([test_file])
    while queue:
        node = queue.popleft()
        if node in targets:
            chain = [] if targets[node] == node else [targets[node]]
            while node is not None:
                chain.append(node)
                node = parents[node]
            return chain
        if node.endswith("__init__.py") and node != test_file:
            continue
        for d in direct_deps.get(node, []):
            if d not in parents:
                parents[d] = node
                queue.append(d)
    return None


def _print_selection_report(test_files: List[str], modified_files: List[str], direct_deps: Dict[str, List[str]]):
    """Print, for every selected test, the import chain from the modified file that pulled it in."""
    print(f"\nSelected {len(test_files)} test files from {len(modified_files)} modified Python files.")
    print("Each entry shows the import chain from a modified file to the test (modified file first):\n")
    for test_file in test_files:
        chain = _import_chain(test_file, modified_files, direct_deps)
        if chain == [test_file]:
            reason = "modified directly"
        elif chain is None:
            reason = "no direct import chain found; selected through the reverse map"
        else:
            reason = " -> ".join(chain[:-1])
        print(f"{test_file}\n    {reason}")


def _tests_triggered_by(f: str, reverse_map: Dict[str, List[str]]) -> int:
    """Number of test files a change to `f` selects, counting `f` itself when it is a test."""
    triggered = {t for t in reverse_map.get(f, []) if _is_test_file(t)}
    if _is_test_file(f):
        triggered.add(f)
    return len(triggered)


def _write_summary(
    summary_file: str, modified_files: List[str], test_files: List[str], reverse_map: Dict[str, List[str]]
):
    """Write a short markdown report for the CI job summary: how many tests each modified file triggers.

    A file with outsized reach is the cue for a maintainer to open the per-test import chains in the artifact.
    """
    lines = [
        "## Test fetcher",
        "",
        f"Selected {len(test_files)} test files from {len(modified_files)} modified Python files. "
        "The import chain that selected each test is in the `test_fetched` artifact.",
        "",
        "| Modified file | Tests it triggers |",
        "|---|---|",
    ]
    # A modified file that pulls in nothing beyond itself adds no information; only files with reach get a row.
    triggered = {f: _tests_triggered_by(f, reverse_map) for f in modified_files}
    for f in sorted(modified_files, key=lambda f: (-triggered[f], f)):
        if triggered[f] > (1 if _is_test_file(f) else 0):
            lines.append(f"| `{f}` | {triggered[f]} |")
    lines.append(f"| **Distinct tests selected** | **{len(test_files)}** |")

    Path(summary_file).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        bucket = "common" if len(parts) == 2 else parts[1]
        test_map[bucket].append(p)
    return {k: sorted(set(v)) for k, v in test_map.items()}


def _base_names(node: ast.ClassDef) -> List[str]:
    return [
        b.id if isinstance(b, ast.Name) else b.attr for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))
    ]


def _mixin_markers(bucket: str, decorators: Dict[str, str]) -> Dict[str, List[str]]:
    """Map mixin class name → markers it carries, from the `@is_*` decorators on classes in
    `tests/<bucket>/testing_utils/`. Markers are inherited through pytest, so a mixin also carries those of
    any base mixin. Scoped per bucket because class names repeat across packages (the pipelines' legacy,
    unmarked `IPAdapterTesterMixin` vs. the models' marked one)."""
    bases: Dict[str, List[str]] = {}
    markers: Dict[str, set] = {}
    for module in (PATH_TO_TESTS / bucket / "testing_utils").glob("*.py"):
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            bases[node.name] = _base_names(node)
            markers[node.name] = {
                decorators[d.id] for d in node.decorator_list if isinstance(d, ast.Name) and d.id in decorators
            }

    changed = True
    while changed:
        changed = False
        for cls, cls_bases in bases.items():
            inherited = set().union(*(markers.get(b, set()) for b in cls_bases))
            if not inherited <= markers[cls]:
                markers[cls] |= inherited
                changed = True
    return {cls: sorted(m) for cls, m in markers.items() if m}


def _class_markers(test_file: str, mixin_markers: Dict[str, List[str]]) -> List[set]:
    """Markers of every test class in `test_file`, derived from the marked mixins it composes."""
    tree = ast.parse((PATH_TO_REPO / test_file).read_text(encoding="utf-8"))
    return [
        {m for base in _base_names(node) for m in mixin_markers.get(base, [])}
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    ]


def _feature_groups_for(paths: List[str], mixin_markers: Dict[str, List[str]]) -> List[str]:
    """Names of the feature groups (including `core`) that at least one test class in `paths` would land in."""
    class_markers = [m for p in paths for m in _class_markers(p, mixin_markers)]
    non_core = NON_CORE_MARKERS
    groups = []
    if any(not m & non_core for m in class_markers):
        groups.append("core")
    for group, markers in FEATURE_GROUPS.items():
        if any(m & set(markers) for m in class_markers):
            groups.append(group)
    # Function-style test files declare no class, so no marker: they are core tests.
    return groups or ["core"]


def _matrix_entries(test_map: Dict[str, List[str]]) -> List[Dict[str, str]]:
    """Expand buckets into CI matrix entries: `{"name", "paths", "markers"}`, one job each."""
    decorators = marker_decorators()
    entries = []
    for bucket, paths in sorted(test_map.items()):
        joined = " ".join(paths)
        if bucket not in SPLIT_BY_FEATURE:
            entries.append({"name": bucket, "paths": joined, "markers": ""})
            continue
        for group in _feature_groups_for(paths, _mixin_markers(bucket, decorators)):
            expr = "core" if group == "core" else " or ".join(FEATURE_GROUPS[group])
            entries.append({"name": f"{bucket}-{group}", "paths": joined, "markers": expr})
    return entries


def _write_matrix(json_output_file: str, test_files: List[str]):
    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump(_matrix_entries(_bucket_for_matrix(test_files)), fp, ensure_ascii=False)


def _is_test_file(path: str) -> bool:
    """True if `path` is a `tests/.../test_*.py` file (the kind pytest collects)."""
    return path.startswith("tests/") and Path(path).name.startswith("test_")


def fetch_tests_to_run(json_output_file: str, summary_file: str, diff_with_last_commit: bool):
    """Determine the tests to run from the diff, write `test_map.json` and the markdown job summary."""
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    direct_deps = create_direct_dependency_map()
    reverse_map = create_reverse_dependency_map(direct_deps)

    # Each modified file contributes itself (if it's a test) plus tests transitively impacted by it.
    selected = set()
    for f in modified_files:
        if _is_test_file(f):
            selected.add(f)
        selected.update(t for t in reverse_map.get(f, []) if _is_test_file(t))

    test_files_to_run = sorted(p for p in selected if (PATH_TO_REPO / p).exists())
    _print_selection_report(test_files_to_run, modified_files, direct_deps)
    _write_summary(summary_file, modified_files, test_files_to_run, reverse_map)
    _write_matrix(json_output_file, test_files_to_run)


def _all_test_files() -> List[str]:
    """Enumerate every `tests/.../test_*.py` (used for the full-suite path)."""
    return sorted(
        str(p.relative_to(PATH_TO_REPO)) for p in PATH_TO_TESTS.glob("**/test_*.py") if "__pycache__" not in p.parts
    )


def _write_full_suite(json_output_file: str):
    """Schedule the entire test suite. Used by `--force_full_suite` and as exception fallback."""
    _write_matrix(json_output_file, _all_test_files())


# ============================================================
# CLI
# ============================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_output_file",
        type=str,
        default="test_map.json",
        help="Where to store the list of matrix entries (name / paths / markers) consumed by CI.",
    )
    parser.add_argument(
        "--summary_output_file",
        type=str,
        default="test_fetcher_summary.md",
        help="Where to store the markdown report appended to the CI job summary.",
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
    if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref.name == "main":
        print("main branch detected, fetching tests against last commit.")
        diff_with_last_commit = True

    try:
        fetch_tests_to_run(args.json_output_file, args.summary_output_file, diff_with_last_commit)
    except Exception as e:
        import traceback

        print(f"\nError when trying to grab the relevant tests: {e}\n")
        traceback.print_exc()
        print("\nRunning all tests.")
        _write_full_suite(args.json_output_file)
