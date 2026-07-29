# Skill: Guided Contribution & Lifecycle Assistant

**Trigger:** Use this skill when a team member (PM, Developer, QA, or DevOps) needs to plan, build, test, or audit a scoped Diffusers change.

## Phase 0: Persona Routing & Initialization

1. Identify the user's role based on their prompt (e.g., "As a PM...", "Audit this diff for DevOps...").
2. If the role is unclear, pause and ask: *"Which perspective are we executing this from? (1. PM/Planning, 2. Developer/Implementation, 3. QA/Testing, 4. DevOps/Audit)"*.
3. Establish the terminal phase for the chosen persona:
   - **PMs:** Execute Phase 1 -> 2 -> 3. (Goal: Output a Feature Spec / Plan. Do not write code).
   - **Developers:** Execute the full Phase 1 through Phase 6. (Goal: End-to-end scaffolding).
   - **QA Engineers:** Skip to Phase 2, then execute Phase 5. (Goal: Read existing code and generate a robust test suite/edge cases. If a test suite already exists, provide any potential improvements).
   - **DevOps:** Skip to Phase 5 and Phase 6. (Goal: Audit an existing PR/diff for CI/CD compliance, security, and deployment risks. Only provide the brief for the DevOps persona, leave out all others.).

## Phase 1: Understand

1. Read the requested task.
2. Restate the problem in plain language tailored to the active persona.
3. Identify who benefits from the change.
4. Define clear acceptance criteria for the current phase.
5. List unanswered blocking questions.
6. Do not modify files.
7. If there are blocking questions, do not proceed to the next phase until the questions are answered.

## Phase 2: Explore

1. Read `.cursor/rules/1-agent-workflows.mdc` and follow its pointer to review `.ai/AGENTS.md`.
2. Read `.cursor/rules/2-testing-and-quality.mdc` and follow its pointer to review `.ai/testing.md`.
3. Read `.cursor/rules/3-review-and-guardrails.mdc` and follow its pointer to review `.ai/review-rules.md`.
4. Read any additional `.ai/` guidance relevant to the task.
5. Search the repository for analogous implementations.
6. Search for existing tests covering similar behavior.
7. Identify existing helpers and abstractions that can be reused.
8. Do not modify files.

## Phase 3: Plan

Produce a plan containing:

### Problem
What is currently missing, unclear, or incorrect?

### Business and user impact
Who benefits and what friction is reduced?

### Existing patterns
Which repository files demonstrate the preferred approach?

### Proposed approach
What is the smallest coherent change?

### Files affected
Which files are expected to change and why?

### Tests
Which tests should be added or updated, and what does each prove?

### Risks and assumptions
What could be incorrect, incompatible, slow, or incomplete?

### Deliberately excluded
What will not be handled by this change?

### Validation
What exact commands should be run?

*(Note: If the active persona is a PM, present this Plan as the final Feature Specification and STOP here.)*
Stop after producing the plan and wait for human approval.

## Phase 4: Implement

*(Note: Only execute this phase if the active persona is Developer.)*
After approval:

1. Restate the approved scope.
2. Identify the files that will be edited.
3. Implement only the approved change.
4. Follow existing repository patterns.
5. Add focused tests.
6. Avoid unrelated cleanup.
7. Do not claim validation passed before commands are run.

## Phase 5: Validate and review

*(Note: QA and DevOps personas focus heavily here based on their specific `.mdc` guardrails.)*

1. Inspect `git status`.
2. Inspect `git diff --stat`.
3. Inspect the complete Git diff.
4. Run the smallest relevant tests first.
5. Run applicable repository quality checks.
6. Review the diff against `.ai/review-rules.md`.
7. Separate:
   - blocking findings,
   - non-blocking findings,
   - unverified assumptions.
8. Correct only findings that remain within the approved scope.

## Phase 6: Handoff

Create a Change Brief with:

### Developer view
- What changed technically?
- Which existing pattern was followed?
- What follow-up work remains?

### QA view
- What behavior should be verified?
- Which edge cases matter?
- Which tests were actually run?

### PM view
- What user problem was solved?
- What changed from the user's perspective?
- What was deliberately excluded?

### DevOps view
- Were dependencies, configuration, runtime behavior, or deployment steps changed?
- Is a migration required?
- What is the rollback approach?
- What remains unverified?

Clearly label all validation as Verified, Not run, or Assumed.