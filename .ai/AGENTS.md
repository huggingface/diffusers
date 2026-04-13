# Diffusers — Agent Guide

## Coding style

Strive to write code as simple and explicit as possible.

- Minimize small helper/utility functions — inline the logic instead. A reader should be able to follow the full flow without jumping between functions.
- No defensive code or unused code paths — do not add fallback paths, safety checks, or configuration options "just in case". When porting from a research repo, delete training-time code paths, experimental flags, and ablation branches entirely — only keep the inference path you are actually integrating.
- Do not guess user intent and silently correct behavior. Make the expected inputs clear in the docstring, and raise a concise error for unsupported cases rather than adding complex fallback logic.

---

## Code formatting

- `make style` and `make fix-copies` should be run as the final step before opening a PR

### Copied Code

- Many classes are kept in sync with a source via a `# Copied from ...` header comment
- Do not edit a `# Copied from` block directly — run `make fix-copies` to propagate changes from the source
- Remove the header to intentionally break the link

### Models

- See [models.md](models.md) for model conventions, attention pattern, implementation rules, dependencies, and gotchas.
- See the [model-integration](./skills/model-integration/SKILL.md) skill for the full integration workflow, file structure, test setup, and other details.

### Pipelines & Schedulers

- Pipelines inherit from `DiffusionPipeline`
- Schedulers use `SchedulerMixin` with `ConfigMixin`
- Use `@torch.no_grad()` on pipeline `__call__`
- Support `output_type="latent"` for skipping VAE decode
- Support `generator` parameter for reproducibility
- Use `self.progress_bar(timesteps)` for progress tracking
- Don't subclass an existing pipeline for a variant — DO NOT use an existing pipeline class (e.g., `FluxPipeline`) to override another pipeline (e.g., `FluxImg2ImgPipeline`) which will be a part of the core codebase (`src`)

## Skills

Task-specific guides live in `.ai/skills/` and are loaded on demand by AI agents. Available skills include:

- [model-integration](./skills/model-integration/SKILL.md) (adding/converting pipelines)
- [parity-testing](./skills/parity-testing/SKILL.md) (debugging numerical parity).
