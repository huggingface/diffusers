# PR Review Rules

Rules for Claude to check during PR reviews. Focus on correctness — style is handled by ruff.

## Code style
- Inline logic — minimize small helper/utility functions. A reader should follow the full flow without jumping between functions.
- No defensive code or unused code paths — no fallback paths, safety checks, or config options "just in case".
- No silent fallbacks — raise a concise error for unsupported cases rather than guessing user intent.

## Dependencies
- No new mandatory dependencies without prior discussion.
- Optional deps must be guarded with `is_X_available()` and have a dummy in `utils/dummy_*.py`.
- Never use `einops` — rewrite with native PyTorch (`reshape`, `permute`, `unflatten`).

## Models
- All layer calls must be visible directly in `forward()` — no helper functions hiding `nn.Module` calls.
- No NumPy operations in `forward()` — breaks `torch.compile` with `fullgraph=True`.
- No hardcoded dtypes (e.g. `torch.float32`, `torch.bfloat16`) in forward — use input tensor dtype or `self.dtype`.
- Attention must use `dispatch_attention_fn`, not `F.scaled_dot_product_attention` directly.
- Every `__init__` parameter in a `ModelMixin` subclass must be captured by `register_to_config`.
- New classes must be registered in `__init__.py` with lazy imports (both `_import_structure` and `_lazy_modules`).

## Pipelines
- Must inherit from `DiffusionPipeline`.
- `@torch.no_grad()` on pipeline `__call__` — forgetting this causes OOM from gradient accumulation.
- Do NOT subclass an existing pipeline for a variant (e.g. don't subclass `FluxPipeline` for `FluxImg2ImgPipeline`).
- Support `output_type="latent"` for skipping VAE decode.
- Support `generator` parameter for reproducibility.

## Copied code
- Never edit a `# Copied from` block directly — run `make fix-copies` to propagate changes from the source.
- Remove the `# Copied from` header to intentionally break the sync link.

## Common mistakes (add new rules below this line)
