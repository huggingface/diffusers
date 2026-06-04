# Plan: `/diffusers-bot` — run GPU tests from a PR comment

## Goal

Let repository maintainers (admins + anyone with `write` access) kick off GPU tests on a
PR by commenting:

```
/diffusers-bot pytest tests/models/test_modeling_common.py -k "some_test"
```

The bot then:

1. Validates the commenter is authorized and the command is well-formed.
2. Triggers a GPU run on the same runner used by `pr_tests_gpu.yml`.
3. Posts a PR comment linking to the Actions run ("⏳ running…").
4. When the run finishes (success **or** failure), **edits that same comment** with the
   final status (✅ / ❌) and the link.

Works for PRs from forks (external contributors) as well as same-repo branches.

---

## Key design decisions & constraints

### 1. Trigger: `issue_comment`, not `pull_request_target`

We use:

```yaml
on:
  issue_comment:
    types: [created]
```

`issue_comment` workflows **always run from the workflow file on the repo's default
branch**, in the context of the base repo, with access to base-repo secrets and a
`GITHUB_TOKEN` that can write comments. A fork cannot alter the workflow logic that runs.
This is the correct, safe trigger for a bot command. (`github.event.issue.pull_request`
is set when the comment is on a PR — we gate on it.)

### 2. Authorization — must be write/admin, not just author_association

`claude_review.yml` gates on `author_association` (`MEMBER`/`OWNER`/`COLLABORATOR`). That is
a usable first filter, but `author_association` is **not** a reliable proxy for write
access (e.g. a contributor can be `CONTRIBUTOR` while still being on a team with write).

The robust check is the collaborator-permission API:

```bash
PERM=$(gh api "repos/${REPO}/collaborators/${COMMENTER}/permission" --jq '.permission')
# -> one of: admin, write, read, none
[[ "$PERM" == "admin" || "$PERM" == "write" ]] || exit_unauthorized
```

We do this in a **gate job** before anything touches the GPU. Unauthorized commenters get a
👎 reaction (or a short comment) and the workflow stops. This is the single most important
security control because the next step **executes untrusted fork code on our GPU runner** —
only a trusted maintainer may vouch for that.

### 3. Command parsing

**The authorization gate (above) is the trust boundary.** Only admins / write-access
maintainers can trigger this, which is the same trust level as pushing a workflow change or
running CI — so we do **not** need an elaborate pytest-arg allowlist or shell-metacharacter
filtering. A maintainer running an arbitrary `pytest` invocation on the runner is
explicitly allowed.

We keep only basic hygiene, not a security control:

- Pass the command through an `env:` variable rather than interpolating
  `${{ github.event.comment.body }}` straight into a `run:` script. This is to avoid YAML/
  shell breakage on quotes and special characters in a legitimate command — not to defend
  against the (already-trusted) commenter.
- Parsing: strip the leading `/diffusers-bot pytest` prefix; the remainder is the pytest
  argv, forwarded as-is to `pytest`.

(Optional, purely as a guardrail against typos — not security: a soft check that the
command targets `tests/`. Can be dropped if it gets in the way.)

### 4. Checking out fork PR code

Resolve the PR number from `github.event.issue.number`, then check out the PR **head**
(works for forks without needing fork credentials), exactly like `run_tests_from_a_pr.yml`:

```yaml
- uses: actions/checkout@v6
  with:
    ref: refs/pull/${{ needs.gate.outputs.pr_number }}/head
```

### 5. Secret minimization on the GPU job

The GPU job runs untrusted code, so it gets the **least** privilege:

- `permissions: contents: read` (no `pull-requests: write` here).
- Only the secret the tests actually need: `HF_TOKEN: ${{ secrets.DIFFUSERS_HF_HUB_READ_TOKEN }}`
  (a **read** token, same as `pr_tests_gpu.yml`).

All comment creation/editing happens in **separate ubuntu jobs** that never check out fork
code, so the `pull-requests: write` token is never exposed to untrusted code.

### 6. Comment lifecycle (create once, edit in place)

- **Gate job** creates the initial comment via `gh api` and captures its id:
  ```bash
  CID=$(gh api -X POST "repos/${REPO}/issues/${PR}/comments" \
        -f body="⏳ Running \`pytest …\` on GPU — [view run]($RUN_URL)" --jq '.id')
  echo "comment_id=$CID" >> "$GITHUB_OUTPUT"
  ```
  `RUN_URL = ${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}`.
- **Report job** (`needs: [gate, gpu-tests]`, `if: always()`) reads
  `needs.gpu-tests.result` (`success`/`failure`/`cancelled`/`skipped`) and **PATCHes** the
  same comment id:
  ```bash
  gh api -X PATCH "repos/${REPO}/issues/comments/${CID}" -f body="$FINAL_BODY"
  ```

We use `gh api` rather than `peter-evans/create-or-update-comment` to avoid adding a new
third-party action dependency (consistent with how the repo already drives comments via the
`gh` CLI in `claude_review.yml`).

---

## Workflow structure

New file: `.github/workflows/pr_comment_gpu_tests.yml`

```
on: issue_comment (created)

concurrency:
  group: diffusers-bot-${{ github.event.issue.number }}
  cancel-in-progress: true   # a newer command supersedes an in-flight one for the same PR

jobs:
  gate:                         # ubuntu, fast
    if: PR comment AND body starts with "/diffusers-bot pytest"
    permissions: { pull-requests: write }
    steps:
      - check commenter permission (admin|write) via gh api
      - parse pytest args (strip prefix; pass via env, no allowlist needed)
      - resolve PR head ref + whether cross-repo (fork)
      - 👀 react to the comment to acknowledge
      - create "⏳ running" comment, output comment_id
    outputs: { pr_number, pr_ref, pytest_args, comment_id, authorized }

  gpu-tests:                    # GPU runner, least privilege
    needs: gate
    if: needs.gate.outputs.authorized == 'true'
    runs-on: { group: aws-g4dn-2xlarge }
    container:
      image: diffusers/diffusers-pytorch-cuda
      options: --gpus all --shm-size "16gb" --ipc host
    permissions: { contents: read }
    env: { HF_TOKEN: secrets.DIFFUSERS_HF_HUB_READ_TOKEN }
    steps:
      - checkout refs/pull/<pr>/head
      - install deps (see "Dependency installation" below)
      - run: pytest $PYTEST_ARGS  (args via env; --make-reports for artifacts)
      - upload reports artifact (if: always())

  report:                       # ubuntu, edits the comment
    needs: [gate, gpu-tests]
    if: always() && needs.gate.outputs.comment_id != ''
    permissions: { pull-requests: write }
    steps:
      - compute status from needs.gpu-tests.result
      - PATCH the comment_id with ✅/❌/⚠️ + run link (+ short failure hint)
```

---

## Dependency installation (GPU job)

Mirror `pr_tests_gpu.yml`'s `torch_cuda_tests` install block, but pull in **all** the
training/test extras from `setup.py` so peft and the rest are present:

```bash
printf 'tokenizers<0.23.0\ntorch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\n' > "$UV_OVERRIDE"
uv pip install -e ".[quality,training,test]"      # training -> peft, accelerate, datasets, timm, …
                                                  # test     -> pytest, pytest-xdist, parameterized, …
uv pip install peft@git+https://github.com/huggingface/peft.git
uv pip uninstall accelerate && uv pip install -U accelerate@git+https://github.com/huggingface/accelerate.git
uv pip uninstall transformers huggingface_hub && UV_PRERELEASE=allow uv pip install -U transformers@git+https://github.com/huggingface/transformers.git
```

- `[training]` provides peft + the training stack; `[test]` provides the pytest toolchain —
  both as declared in `setup.py`.
- The git installs of peft/accelerate/transformers (and the `UV_OVERRIDE` pin) match
  `pr_tests_gpu.yml` so the environment is identical to the existing GPU PR tests.

---

## Edge cases handled

- **Comment is not on a PR** → `github.event.issue.pull_request` is null → job skipped.
- **Comment doesn't start with the command** → `if:` filters it out (no run spent).
- **Unauthorized commenter** → gate fails fast and posts a reply comment ("not authorized
  to run this"), no GPU time used.
- **Fork PR** → handled by `refs/pull/<n>/head` checkout; fork code never sees the
  comment-write token.
- **Run cancelled / superseded** (concurrency) → report job's `always()` patches the
  comment to ⚠️ cancelled.
- **Test job genuinely fails** → comment edited to ❌ with link to logs + reports artifact.

---

## Security summary (why this is safe to run fork code on a GPU)

| Risk | Mitigation |
|------|-----------|
| Fork modifies the bot logic | `issue_comment` runs the default-branch workflow file only |
| Untrusted user triggers GPU/secret use | hard `admin`/`write` permission check via API — **this is the trust boundary** |
| Untrusted code exfiltrates write token | GPU job has `contents: read` only; commenting isolated to separate jobs |
| Untrusted code abuses powerful secrets | only a **read** HF token is exposed, same as existing GPU PR tests |

The command string itself is **not** treated as an attack surface: only trusted
maintainers can issue it, so arbitrary `pytest` args are by design.

---

## Deliverables / implementation steps

1. Add `.github/workflows/pr_comment_gpu_tests.yml` implementing the 3-job structure above.
2. Reuse the dependency-install + override-pin steps per "Dependency installation" so the
   env matches `pr_tests_gpu.yml`.
3. **Remove `.github/workflows/run_tests_from_a_pr.yml`** — this comment-driven workflow
   supersedes that `workflow_dispatch`-based one.
4. (Optional follow-up) Document the command in `CONTRIBUTING`/maintainer notes: usage and
   who can run it.

### Resolved decisions

- **Command prefix**: `/diffusers-bot pytest …` exactly.
- **Allowed test scope**: any `pytest` invocation (trusted maintainer) — no allowlist.
- **Runner**: `aws-g4dn-2xlarge` (single GPU, matches existing PR GPU job).
- **Acknowledgement**: 👀 reaction on the triggering comment, plus the "⏳ running" comment.
- **Dependencies**: install `.[quality,training,test]` (peft + all training/test extras from
  `setup.py`) plus the git installs from `pr_tests_gpu.yml`.
```
