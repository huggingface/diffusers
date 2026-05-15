# Bringing `nvidia/Cosmos3-Nano` fully into diffusers format

## What needs to change on the Hub

Only one directory on `nvidia/Cosmos3-Nano` needs to be re-uploaded: **`sound_tokenizer/`**.

Everything else is already correct:

| Component | Status |
|---|---|
| `transformer/` | Correct (sharded `diffusion_pytorch_model-*.safetensors`, keys match `state_dict()`) |
| `vae/` | Correct (canonical Wan VAE layout) |
| `vision_encoder/` | Correct (transformers default `model.safetensors` for `Qwen3VLVisionModel`) |
| `scheduler/` | Config only |
| `text_tokenizer/` | Tokenizer files |
| `sound_tokenizer/` | **Broken — see below** |

### Why `sound_tokenizer/` is broken

The current revision (`bc5688f1fb6c2a289155761df0d4e0973100620b`) ships the raw upstream NVIDIA AVAE checkpoint. Four things are wrong:

| What's wrong | What it should be |
|---|---|
| Filename `model.safetensors` | `diffusion_pytorch_model.safetensors` |
| Keys: raw upstream `decoder.layers.N.layers.M.*` (flat `nn.Sequential`) | `OobleckDecoder` named layout: `decoder.conv1.*`, `decoder.block.0.snake1.*`, … |
| Snake1d `alpha`/`beta` shape `[C]` | `[1, C, 1]` |
| 67 encoder/bottleneck keys present | None (model is decoder-only) |

All four are masked at load time by runtime overrides in `src/diffusers/models/autoencoders/autoencoder_cosmos3_avae.py` (`from_pretrained` L164–205, `_remap_checkpoint_keys` L188–244, `_load_pretrained_model` L247–297). yiyixuxu's PR comment on L297 ("you should convert the checkpoint to diffusers format so that it can be loaded with our `from_pretrained` method") is exactly about this.

### Note on the transformer

The transformer state-dict is already correct on disk. The only "wrong" thing is that `rotary_emb.inv_freq` isn't saved — but that's correct behavior for a non-persistent buffer. The fix is purely code-side (compute `inv_freq` in `__init__` as `register_buffer(..., persistent=False)`, then delete the `from_pretrained` override at L578-582). No Hub re-upload needed for `transformer/`.

## Reusing `scripts/convert_cosmos3_to_diffusers.py`

The script already does all four AVAE fixes:

| Fix | Where in the script |
|---|---|
| Write `diffusion_pytorch_model.safetensors` | `_save_sound_tokenizer` L240 |
| Remap `decoder.layers.N.*` → `decoder.conv1.*` / `decoder.block.N.*` | `_remap_avae_state_dict` L137–195 |
| Reshape Snake1d `alpha`/`beta` to `[1, C, 1]` | `_remap_avae_state_dict` L191–193 |
| Drop encoder/bottleneck keys | `_remap_avae_state_dict` L150–151 (`return None` for non-`decoder.` keys) |

It also writes `sound_tokenizer/config.json` and inserts the `sound_tokenizer` entry into `model_index.json` (`_add_sound_tokenizer_to_model_index` L243).

### The script is a first-class artifact

The converter is NVIDIA-internal (uses `cosmos3.*` / `projects.cosmos3.*` packages) and is the **only** path from raw DCP training checkpoints to the diffusers-format Hub artifact. It must remain runnable through every cluster — letting it break means the Hub checkpoint becomes un-reproducible.

**Per-cluster converter delta:**

| Cluster | Converter change | Type |
|---|---|---|
| A | Drop `freeze_und=vlm_cfg.freeze_und,` at L404 | Deletion |
| B | Add a 4-key remap for `time_embedder`: `mlp.0.{w,b}` → `linear_1.{w,b}`, `mlp.2.{w,b}` → `linear_2.{w,b}` | Addition (small) |
| C | None — `inv_freq` was already non-persistent; activation/rotary changes are runtime-only | None |
| D | For each top-level config arg trimmed from the model, drop the matching line at L399-437 | Deletion |
| E | None **if** we preserve `q_proj`/`k_proj`/`v_proj`/`o_proj`/`q_norm`/`k_norm` and their `_moe_gen` siblings as attribute names | None |
| F | None — converter doesn't touch pipeline internals | None |
| G | None — pure model-file deletion | None |

The framing is "stop extracting/passing what the model no longer needs" — not "add elaborate translation logic". The converter ends up *simpler*, not more complex.

## Concrete steps

These are the steps to run **once at the end of the cluster series** (one converter run, one Hub push, one revision bump). Each cluster's commits should include the matching converter patch from the table above, so the converter is always lockstep with the model — but the actual `python scripts/convert_...` execution and Hub upload happen only once.

1. **Patch the script** lockstep with each cluster (per the table above). For Cluster A, that means removing `freeze_und=vlm_cfg.freeze_und,` at L404. Cluster B adds the 4-key `time_embedder` remap; Cluster D removes lines for each trimmed config arg. Other clusters need no converter change.

2. **Run the converter** (needs NVIDIA-internal `cosmos3` package and an upstream AVAE checkpoint):
   ```bash
   python scripts/convert_cosmos3_to_diffusers.py \
       --checkpoint-path Cosmos3-Nano \
       --output converted/cosmos3-nano-pipeline \
       --save-pipeline \
       --sound-tokenizer-path <path/to/upstream/avae.safetensors>
   ```

3. **Spot-check the output**:
   ```python
   from safetensors import safe_open
   p = "converted/cosmos3-nano-pipeline/sound_tokenizer/diffusion_pytorch_model.safetensors"
   with safe_open(p, framework="pt") as f:
       keys = list(f.keys())
   assert all(k.startswith("decoder.") for k in keys)            # no encoder/bottleneck keys
   assert any(k.startswith("decoder.conv1.") for k in keys)      # remapped names present
   assert all(not k.startswith("decoder.layers.") for k in keys) # raw layout gone
   with safe_open(p, framework="pt") as f:
       alpha = next(f.get_tensor(k) for k in keys if k.endswith(".alpha"))
   assert alpha.ndim == 3 and alpha.shape[0] == 1 and alpha.shape[-1] == 1  # [1, C, 1]
   ```

4. **Upload to the Hub**. Just the `sound_tokenizer/` directory (and `model_index.json` if its entry needs updating). With `huggingface_hub`:
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path="converted/cosmos3-nano-pipeline/sound_tokenizer",
       path_in_repo="sound_tokenizer",
       repo_id="nvidia/Cosmos3-Nano",
       commit_message="Convert sound_tokenizer to diffusers format",
   )
   # Plus model_index.json if its sound_tokenizer entry isn't already correct.
   ```
   The upload returns a `CommitInfo` with the new SHA — grab it for step 5.

5. **Bump the revision pin** in `examples/cosmos3/inference_cosmos3.py:33` (`HF_REVISION`) to the new commit SHA. Confirm the inference example still runs.

6. **Delete the runtime overrides** in `src/diffusers/models/autoencoders/autoencoder_cosmos3_avae.py`:
   - `from_pretrained` (L164–205) — no longer needed; standard filename
   - `_remap_checkpoint_keys` (L188–244) — no longer needed; keys already in target layout
   - `_load_pretrained_model` (L247–297) — no longer needed; nothing to fix at load time

   After deletion the file should be ~160 lines — just the model class definition. A `from_pretrained("nvidia/Cosmos3-Nano/sound_tokenizer")` will now use `ModelMixin.from_pretrained` with zero special-casing.

## Cluster G is independent

Dropping `Cosmos3OmniTransformer.from_pretrained` is purely code-side (Cluster C makes the override unnecessary by computing `inv_freq` in `__init__`). It doesn't require any Hub change and shouldn't be conflated with the steps above.

## Sequencing recommendation

The Hub re-upload only has to happen once. If Cluster D will drop more config args, batch the script-patch + run + upload after Cluster D rather than doing it twice. Until then, the runtime adapter in `autoencoder_cosmos3_avae.py` keeps the inference example working from the current `bc5688f1` revision.
