# FLUX.3 image integration notes — TEMPORARY, REMOVE BEFORE MERGE

> **⚠️ This folder must not be merged.** It holds the parity check against the confidential BFL
> reference implementation (`f3` package), the doc-style examples, and the scripts that build the
> test and release repos. Delete it before the PR is merged.

## Setup (workspace `/raid/yiyi`)

- `diffusers-new-model-addition-f3/` — this repo, branch `flux3-image`, with its `.venv` (transformers 5.x).
- `bfl-f3-image/` — the BFL reference on its `f3-image` branch (`f3/image_small/ImageSmallPipeline`).
- `f3-image-small/` — the BFL model package (`hf download herzogenhorn/f3-image-small`).
- `flux3-image-modular/` — the converted diffusers repo (see Conversion). The same content is on the Hub as
  `diffusers-internal-dev/f3-small-img` (private), with BFL's knight example image under `examples/`.
- `venv-tf457/` — a venv with transformers 4.57.6 (BFL's training pin) for the reference side.
- `tiny-flux3-image-modular/` — the tiny random repo the tests use (`make_tiny_flux3_repo.py`). Upload as
  `hf-internal-testing/tiny-flux3-image-modular` and update `FLUX3_TINY_REPO` in the modular tests.

Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` so `CUDA_VISIBLE_DEVICES` matches `nvidia-smi` on this machine.

## Conversion

```bash
python scripts/convert_flux3_to_diffusers.py \
    --checkpoint_path f3-image-small/model.safetensors --config_path f3-image-small/config.json \
    --output_path flux3-image-diffusers                 # -> transformer/
python scripts/convert_flux2_to_diffusers.py --vae \
    --checkpoint_path f3-image-small/ae.safetensors --output_path flux3-image-diffusers   # -> vae/
```

`make_flux3_modular_repo.py` does both, adds the text encoder and tokenizer from the package, writes the
`modular_model_index.json`, and can push the result to the Hub (that is how `diffusers-internal-dev/f3-small-img`
was built). `flux3_examples.py` runs the four parity cases as doc examples against that repo, without any
reference code.

`--variant sd` builds the step-distilled repo from `variants/sd/` of the package (`flow_map: true` in its config;
the converter maps `ref_time_in.*` to `target_timestep_embedder.*`), with `Flux3DistilledAutoBlocks` in the index:

```bash
.venv/bin/python integration-notes/make_flux3_modular_repo.py \
    --package /raid/yiyi/f3-image-small --variant sd --output /raid/yiyi/flux3-image-distilled-modular
```

## Parity (`parity_bf16.py`)

bf16, 50 steps, guidance 3.5, shift 6.93, seed 0, same prompts and the same initial noise on both sides.
The reference runs as shipped (`ImageSmallPipeline.from_package` + `generate`) in the transformers 4.57.6 venv,
diffusers in this repo's venv; the two sides hand over through PNG files:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /raid/yiyi/venv-tf457/bin/python integration-notes/parity_bf16.py reference
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python integration-notes/parity_bf16.py diffusers
```

| case | size | mean pixel diff | max |
|---|---|---|---|
| t2i (the fox) | 1024 x 1024 | 0.44/255 | 47/255 |
| i2i (the fox at night, the t2i output as reference) | 1024 x 1024 | 0.31/255 | 9/255 |
| bbox_t2i (BFL's layout-control example: regions in the caption) | 1024 x 1024 | 0.39/255 | 123/255 |
| bbox_i2i (BFL's knight edit with `<ref_image_0>`; size derived from the 1360 x 768 reference) | 1360 x 768 | 0.26/255 | 26/255 |

All four pairs are visually identical (`diffusers-scratch/flux3-image/parity-bf16/*_side_by_side.png`,
reference left, diffusers right). The differences are bf16 rounding over 50 steps; in fp32 the two
implementations agree to within 2/255 (checked during development: the text encoder is bit-identical and
the transformer agrees to 4e-5 in fp32).

### Step-distilled variant (`F3_VARIANT=sd`)

Same four cases with the `sd` package variant: 16 steps, no guidance, BFL's flow-map branch
(`/raid/yiyi/bfl-f3-flowmap`, their PR #5) against `flux3-image-distilled-modular`:

```bash
F3_VARIANT=sd CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 /raid/yiyi/venv-tf457/bin/python integration-notes/parity_bf16.py reference
F3_VARIANT=sd CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 .venv/bin/python integration-notes/parity_bf16.py diffusers
```

| case | size | mean pixel diff | max | pixels off by > 30 |
|---|---|---|---|---|
| t2i | 1024 x 1024 | 1.74/255 | 221/255 | 1.8 % |
| i2i | 1024 x 1024 | 0.21/255 | 12/255 | 0 % |
| bbox_t2i | 1024 x 1024 | 0.46/255 | 247/255 | 0.2 % |
| bbox_i2i | 1360 x 768 | 0.31/255 | 89/255 | 0.03 % |

Visually identical (`parity-bf16-sd/*_side_by_side.png`). The t2i case drifts more than with the base model
(1.8 % of pixels differ by more than 30, fine fur texture only): 16 large Euler steps amplify bf16 rounding more
than 50 small ones, and the attention kernels differ (cuDNN in the reference, PyTorch's default here).

## Why the reference runs under transformers 4.57.6

Qwen3-VL computes the position id of the padding tokens differently across transformers versions (1 in
4.57, BFL's training pin; 0 in 5.x), and the FLUX.3 transformer attends to the padding tokens. The real
tokens are bit-identical across versions; only the padding rows differ, which moves the final image by about
4/255 mean. `Flux3TextEncoderStep` passes the position ids explicitly, so on transformers 5.x it produces
the 4.57.6 embeddings bit for bit; running the reference in an actual 4.57.6 venv is what makes the table
above a comparison against the training-time behavior. This diffusers cannot be installed next to
transformers 4.57.6 (huggingface_hub pin), hence the two venvs.

## Speed and memory

Single A100 80GB, bf16 transformer and text encoder, fp32 VAE, 50 steps, no offloading:

| workflow | time | peak GPU memory |
|---|---|---|
| t2i 1024² | 30.5 s | 22.7 GiB |
| i2i 1024² (one 1024² reference) | 63.6 s | 22.7 GiB |
| i2i 1360 x 768 (two references) | 105 s | 22.7 GiB |

## Design notes

- `Flux3Transformer2DModel` (image + reference modes, no pooled-vector embedder, output head for the image
  tokens; forward: `hidden_states`, `encoder_hidden_states`, `timestep (B,)`, `img_ids`, `txt_ids`,
  `image_cond_hidden_states`, `image_cond_ids`). The video model, if it comes, is a separate class.
- One prompt per call (as the reference): the text window is the prompt's length rounded up to 80 tokens,
  and the transformer attends to the padding, so prompts of different lengths cannot share a batch without
  changing their context. `num_images_per_prompt` gives several images.
- References keep their own resolution (`Flux3ImageProcessor`, Flux2's helpers: downscale above 4096² pixels,
  center-crop to the 16-pixel grid), ids on the `l` axis `-(1 + index)`, H/W ids scaled by 0.5. The i2i size
  defaults to 1 MP at the first reference's aspect ratio; a given `height`/`width` overrides its side (BFL CLI).
- Sampling defaults on the from_config components: guidance 3.5, shift 6.93, 50 steps, for both workflows.

## Open items

- Batched prompts: not supported for now. Before adding: measure the quality effect of a shared text window on
  shorter prompts, whether batching helps throughput at all at this sequence length, and the semantics of
  several prompts together with several reference images.
- Seed layout: BFL draws the noise token-major `(h*w, C)` on the CPU; diffusers draws `(B, C, h, w)`, so the
  same seed gives a different image than BFL's CLI (the parity passes the noise explicitly). Two lines in
  `Flux3PrepareLatentsStep` would align them.
- VAE dtype: BFL runs the VAE in fp32; `load_components(dtype=torch.bfloat16)` casts it too. Pass
  `dtype={"vae": torch.float32, "default": torch.bfloat16}` (as the examples do) or decide the documented default.
- Hub repo layout with BFL (diffusers subfolders in the release repo vs a `-diffusers` repo); the repo ids in
  the docs are placeholders. `from_single_file` for the bf16 transformer is a small follow-up.
- fp8 / nvfp4 variants (`variants/fp8r`, `variants/nvfp4`, ModelOpt export layout with `hf_quant_config.json`):
  not loadable as-is; follow-up through the ModelOpt or torchao backend with BFL's weights repackaged.
- Merge upstream `main`; `test_compile_works_with_aot` fails for Flux2 too in this venv (torch 2.13).
