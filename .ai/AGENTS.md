# Diffusers — Agent Guide

## Coding style

Strive to write code as simple and explicit as possible.

- Minimize small helper/utility functions — inline the logic instead. A reader should be able to follow the full flow without jumping between functions.
- No defensive code or unused code paths — do not add fallback paths, safety checks, or configuration options "just in case". When porting from a research repo, delete training-time code paths, experimental flags, and ablation branches entirely — only keep the inference path you are actually integrating.
- Do not guess user intent and silently correct behavior. Make the expected inputs clear in the docstring, and raise a concise error for unsupported cases rather than adding complex fallback logic.

---

### Dependencies
- No new mandatory dependency without discussion (e.g. `einops`)
- Optional deps guarded with `is_X_available()` and a dummy in `utils/dummy_*.py`

## Code formatting
- `make style` and `make fix-copies` should be run as the final step before opening a PR

### Copied Code
- Many classes are kept in sync with a source via a `# Copied from ...` header comment
- Do not edit a `# Copied from` block directly — run `make fix-copies` to propagate changes from the source
- Remove the header to intentionally break the link

### Models
- All layer calls should be visible directly in `forward` — avoid helper functions that hide `nn.Module` calls.
- Avoid graph breaks for `torch.compile` compatibility — do not insert NumPy operations in forward implementations and any other patterns that can break `torch.compile` compatibility with `fullgraph=True`.
- See the **model-integration** skill for the attention pattern, pipeline rules, test setup instructions, and other important details.

## Skills

Task-specific guides live in `.ai/skills/` and are loaded on demand by AI agents.
Available skills: **model-integration** (adding/converting pipelines), **parity-testing** (debugging numerical parity).
