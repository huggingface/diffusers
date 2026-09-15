# torch.compile

## Documentation

Read the full guide for usage examples, `compile_repeated_blocks()`, modes, and benchmarks:

- **Local:** `docs/source/en/optimization/fp16.md` (search for "torch.compile")
- **Online:** https://huggingface.co/docs/diffusers/main/en/optimization/fp16#torchcompile

For combining torch.compile with quantization and offloading:

- **Local:** `docs/source/en/optimization/speed-memory-optims.md`
- **Online:** https://huggingface.co/docs/diffusers/main/en/optimization/speed-memory-optims

## Compile modes quick reference

| Mode | Speed gain | Compile time | Notes |
|---|---|---|---|
| `"default"` | Moderate | Fast | Safe starting point |
| `"reduce-overhead"` | Good | Moderate | CUDA graphs to reduce Python overhead |
| `"max-autotune"` | Best | Very slow | Tries many kernel configs; best for repeated inference |

## Gotchas

- **Windows:** `reduce-overhead` and `max-autotune` may fail. Use `"default"` mode.
- **`torch._dynamo.config.capture_dynamic_output_shape_ops = True`** is required when compiling bitsandbytes-quantized models, otherwise it errors on dynamic output shapes.
- **`torch._dynamo.config.cache_size_limit = 1000`** is needed when combining compile with offloading to avoid excessive recompilation warnings.
