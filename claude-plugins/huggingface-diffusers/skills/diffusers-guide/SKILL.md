---
name: diffusers-guide
description: Route Diffusers questions to the right pipeline, docs, examples, and next implementation steps. Use when users ask which Diffusers feature to use or where to start in the codebase.
---

Guide the user to the correct Diffusers path quickly.

## Workflow

1. Determine target outcome and modality (image, video, editing, training, optimization).
2. Recommend one primary route and one fallback route.
3. Provide one minimal runnable snippet or command.
4. Point to exact repo paths for deeper work.

## Routing map

- General onboarding: `README.md`, `docs/source/en/quicktour.md`.
- Inference techniques: `docs/source/en/using-diffusers/`.
- Training: `docs/source/en/training/` and related `examples/` folder.
- Optimization: `docs/source/en/optimization/`.
- Internals: `src/diffusers/` plus matching `tests/`.

## Quality bar

- Prefer existing examples over inventing new APIs.
- State required inputs explicitly (prompt/image/mask/control image).
- Mention device and dtype assumptions when giving code.
