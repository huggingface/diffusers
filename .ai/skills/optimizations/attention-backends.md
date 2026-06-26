# Attention Backends

## Documentation

Read the full guide for the available backends, correct backend name strings, and usage examples:

- **Local:** `docs/source/en/optimization/attention_backends.md`
- **Online:** https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends

The docs contain the authoritative list of backend name strings. Backend names differ from package names (e.g. `flash_hub` not `flash_attention_2`) — always look them up in the table there rather than guessing from the package name.

## Implementation note

New models must use `dispatch_attention_fn` (not `F.scaled_dot_product_attention` directly) so that backend switching works automatically across all backends.
