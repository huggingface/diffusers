# SDPA Mask Regression Analysis (QwenImage)

## What caused the regression

The regression is caused by always passing an attention mask to SDPA, even when the mask is effectively all-ones.
In PyTorch, providing `attn_mask` disables the Flash SDPA fast path for most shapes, forcing the kernel selection
into the slower math/mem-efficient path. That path introduces extra elementwise work (masking + reductions) and
reduces effective throughput.

In QwenImage, the joint attention mask is constructed unconditionally when `encoder_hidden_states_mask` is not
`None`. When the prompt lengths are uniform, the mask carries no information but still forces the slow path.
This is the source of the speed regression.

## Nsight-python plot

Plot: `nsight_sdpa_mask_regression.png`

We benchmarked SDPA on H100 with batch=1, heads=16, seq_len=1024, head_dim=64. The mask zeros the
second half of tokens (representative of padding). The mask path is **~3.4x slower** than no-mask:

- `no_mask` avg: **43,066 ns**
- `mask` avg: **145,888 ns**

This data is saved in `nsight_sdpa_mask_regression.csv` and plotted via nsight-python visualization.

Regeneration:
```
/mnt/data/data/kashif/miniconda3/envs/py312/bin/python3.12 scripts/nsight_sdpa_mask_regression.py \
  --seq-len 1024 --num-heads 16 --head-dim 64 --batch-size 1 --runs 5 \
  --plot-path nsight_sdpa_mask_regression.png
```

## Output comparison (before vs after)

Comparison image: `compare_default_vs_flash_varlen.png`

The \"before\" (`compare_default.png`) and \"after\" (`compare_flash_varlen.png`) outputs are pixel-identical.
Diff stats:

- Max diff: 0
- Mean diff: 0.0
- Non-zero pixels: 0

This confirms the PR only changes performance behavior, not output fidelity.

Regeneration:
```
/mnt/data/data/kashif/miniconda3/envs/py312/bin/python3.12 scripts/compare_output_images.py \
  --before compare_default.png --after compare_flash_varlen.png \
  --output compare_default_vs_flash_varlen.png
```
