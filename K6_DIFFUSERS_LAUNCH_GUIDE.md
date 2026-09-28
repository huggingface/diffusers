# Launching Kandinsky 6 through Diffusers

How to run every Kandinsky 6 (K6) model with the Diffusers code in
`diffusers-new-model-addition-kandinskyX` (branch `Kandinsky6`). The launches follow the notebooks in
`diffusers-new-model-addition-kandinskyX/notebooks/`. Code and Hub configs were checked as described in
[Verification status](#verification-status).

## 1. What you can launch

| # | Launch | Hub repo | Pipeline | Steps / guidance | Notes |
|---|--------|----------|----------|------------------|-------|
| 1 | Text → video + audio (T2VA), quality | `kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers` | `Kandinsky6TI2VAPipeline` | 50 steps, `guidance_scale=5.0` | `FlowMatchEulerDiscreteScheduler`; MagCache available |
| 2 | Text → video + audio, fast | `kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers` | `Kandinsky6TI2VAPipeline` | 16 steps, `guidance_scale=1.0` (mandatory) | `PiflowScheduler` rejects any other guidance; no MagCache |
| 3 | Image → video + audio (I2VA) | either repo above | `Kandinsky6TI2VAPipeline` | as 1 or 2 | pass `image=` |
| 4 | Video super-resolution, flow matching | `kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers` | `Kandinsky6SRPipeline` | `num_inference_steps=5` (= 4 Euler steps per tile) | x2, x2.25 (default), x4 |
| 5 | Video super-resolution, distilled | `kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers` | `Kandinsky6SRPipeline` | 2 model evaluations per tile, `num_inference_steps` ignored | `PiflowScheduler` |
| 6 | T2VA → SR chain | one of 1/2 + one of 4/5 | both | as above | video tensor handed over directly; audio muxed in afterward |

Component layout (from each repo's `model_index.json`):

- **T2VA repos:** `transformer` (`Kandinsky6Transformer3DModel`), `vae` (`AutoencoderKLHunyuanVideo`), `scheduler`,
  `text_encoder` (Qwen2.5-VL), `tokenizer`, `text_encoder_2` (CLIP), `tokenizer_2`, `audio_vae` (`Kandinsky6AudioVAE`).
  Audio is 44.1 kHz, video is 24 fps.
- **VSR repos:** `transformer` (`Kandinsky6SRTransformer3DModel`, 1.41B), `vae` (`Kandinsky6SRVAE`, 1.74B),
  `latent_upscaler` (`Kandinsky6SRLatentUpscalerBank`, x2 + x4, 3.65B), `scheduler`.

All K6 repos are private: the Hub answers 401 without a login (section 2.3). `Kandinsky-6.0-Lite-Diffusers` appears
only in the FastVideo and SGLang docs, not in the Diffusers code or notebooks, so it is not covered here.

## 2. Setup

### 2.1 Install this branch

Always use the uv `.venv` of the checkout. Do not use conda and do not use `uv run`/`uv sync`, which can swap your
torch build.

```bash
BASE_PATH="something"   # folder that contains the K6 repos
cd "$BASE_PATH/diffusers-new-model-addition-kandinskyX"
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install av ipykernel
python -c "from diffusers import Kandinsky6TI2VAPipeline, Kandinsky6SRPipeline; print('K6 OK')"
```

- Stock Diffusers from PyPI has no `Kandinsky6*` classes, so it fails with an `ImportError`. Use this branch.
- Install a torch build that matches the GPU driver first (torch >= 2.6). Then run `uv pip install -e .`.
- `av` (PyAV) is required by `diffusers.utils.encode_video` to write mp4 files (with or without audio). Both
  pipelines take/return tensors only — neither reads nor writes video files itself.
- `natten` and `flash_attn` are not required. The default `native` backend needs neither.
- On a remote GPU box, run the same block from the checkout on that box after `git pull` there.

### 2.2 GPU memory

The pipelines are meant to run with `enable_model_cpu_offload()`. That keeps one component on the GPU at a time. The
T2VA order is `text_encoder -> text_encoder_2 -> transformer -> vae -> audio_vae`. The SR order is
`source_vae -> latent_upscaler -> transformer -> vae`. Without offload, the full T2VA pipeline (Qwen2.5-VL plus a
60-block, 4096-wide transformer) needs a large GPU.

### 2.3 Hugging Face access

Log in once, interactively, and never paste a token into a script or notebook:

```bash
hf auth login
hf auth whoami
```

For non-interactive shells use `export HF_TOKEN=...` in the shell, never in a file. After the first download the
snapshot is cached in `~/.cache/huggingface/hub`. Set `HF_HUB_OFFLINE=1` to run without network.

Note: the notebooks contain `os.environ["HF_TOKEN"] = "TOKEN"`. Delete that line rather than putting a real token there.

## 3. Text/image → video + audio (`Kandinsky6TI2VAPipeline`)

Defaults of `__call__`: `height=512, width=768, num_frames=121, sample_fps=24.0, num_inference_steps=50,
guidance_scale=5.0, max_sequence_length=1024, sample_audio=True, expand_prompts=False, output_type="pt"`.
`height` and `width` must be divisible by 8, and `max_sequence_length` must be between 1 and 1024. The notebooks use
`480 x 864`, which is the size the `Pro` checkpoints were configured for (`k6_pro_125_480_864_*` in `k6_video`).

`output_type="pt"` returns `frames` as a **uint8 tensor `[B, 3, T, H, W]` in 0..255**. `audio` is a list with one 1-D
mono waveform per video. Prompts can carry speech as `<S>spoken words<E>`.

### 3.1 Launch 1: Pro-sft, quality (50 steps, CFG 5.0, MagCache)

Save as `run_t2va_sft.py` and run it from the venv:

```python
import torch
from diffusers import Kandinsky6TI2VAPipeline
from diffusers.hooks import MagCacheConfig
from diffusers.utils import encode_video

MODEL = "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers"
GPU_ID = 0
STEPS, GUIDANCE, FPS, SEED = 50, 5.0, 24.0, 1137
USE_MAGCACHE = True
MODE = "t2va"  # "t2va" or "i2va": selects the MagCache coefficients

pipe = Kandinsky6TI2VAPipeline.from_pretrained(MODEL, dtype=torch.bfloat16)
pipe.transformer.set_attention_backend("native")  # "_flash_3" needs FlashAttention 3 (Hopper)

cfg = pipe.transformer.config.get("magcache")
if USE_MAGCACHE and cfg is not None:
    ratios = cfg["mag_ratios"]
    ratios = ratios[MODE] if isinstance(ratios, dict) else ratios
    pipe.transformer.enable_cache(
        MagCacheConfig(
            threshold=cfg["threshold"],
            max_skip_steps=cfg["max_skip_steps"],
            retention_ratio=cfg["retention_ratio"],
            num_inference_steps=2 * STEPS,  # coefficients interleave cond and uncond, so 2 forwards per step
            mag_ratios=list(ratios),
        )
    )

pipe.enable_model_cpu_offload(gpu_id=GPU_ID)

result = pipe(
    prompt="A news anchor delivers the evening news in a clear, neutral tone, saying: "
           "<S>Good evening, ladies and gentlemen, and welcome to the evening news.<E>",
    negative_prompt="blurry, distorted, low quality, noisy audio",
    height=480,
    width=864,
    num_frames=121,
    sample_fps=FPS,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    sample_audio=True,
    generator=torch.Generator().manual_seed(SEED),
    output_type="pt",
)

video = result.frames[0].permute(1, 2, 3, 0)                        # [T,H,W,3] uint8
audio = torch.as_tensor(result.audio[0])[:, None].repeat(1, 2)      # mono -> stereo [samples, 2]
encode_video(video, fps=int(FPS), output_path="t2va_sft.mp4",
             audio=audio, audio_sample_rate=int(pipe.audio_sample_rate))
print("saved t2va_sft.mp4")
```

Notes:

- The MagCache coefficients live in the transformer config under `magcache`, with separate lists for `t2va` and `i2va`.
  Set `MODE` to match the generation you run.
- To turn MagCache off, set `USE_MAGCACHE = False`. You can also call `pipe.transformer.disable_cache()`.
- MagCache is used only with the sft checkpoint. The authors' notebook disables it for the distilled one.

### 3.2 Launch 2: Pro-distill, fast (PiFlow, 16 steps, guidance 1.0)

Same script with these changes:

```python
MODEL = "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers"
STEPS, GUIDANCE = 16, 1.0
USE_MAGCACHE = False
```

The output path can become `t2va_distill.mp4`. `guidance_scale` must stay `1.0` here: the denoise loop raises
`ValueError("PiflowScheduler requires guidance_weight=1.0")` for any other value. With guidance 1.0 there is no
unconditional pass, so no `negative_prompt` is needed.

You can choose steps and guidance automatically, as the notebook does:

```python
is_piflow = pipe.scheduler.__class__.__name__ == "PiflowScheduler"
STEPS = 16 if is_piflow else 50
GUIDANCE = 1.0 if is_piflow else 5.0
```

### 3.3 Launch 3: image → video + audio (I2VA)

Pass a reference image path (or a PIL image) as `image=`. Set `MODE = "i2va"` for MagCache. For a batch, pass one image
per prompt:

```python
BASE_PATH = "something"  # folder that contains the K6 repos

result = pipe(
    prompt=["A woman turns to the camera and says <S>Hello there!<E>"],
    image=[f"{BASE_PATH}/k6_video/assets/i2va_input.png"],
    negative_prompt=["blurry, distorted, low quality, noisy audio"],
    height=480, width=864, num_frames=121, sample_fps=24.0,
    num_inference_steps=STEPS, guidance_scale=GUIDANCE,
    sample_audio=True, generator=torch.Generator().manual_seed(1137), output_type="pt",
)
```

The repo has a sample image at `k6_video/assets/i2va_input.png`. It is not in the diffusers checkout.

### 3.4 Optional switches

| Switch | How | Comment |
|--------|-----|---------|
| Video only | `sample_audio=False` | `result.audio` is then `None`; skip the `audio=` arguments |
| Batching | lists for `prompt`, `negative_prompt`, `image` | the notebook uses batch size 2 |
| Prompt expansion | `expand_prompts=True` | rewrites the prompt with the Qwen2.5-VL model first; slower. Default is `False`. The notebook hard-codes `True` in the call |
| NumPy output | `output_type="np"` | `[B, T, H, W, 3]` |
| Faster attention | `pipe.transformer.set_attention_backend("_flash_3")` | needs FlashAttention 3 (Hopper). Other valid names: `flash`, `native`, `sage`, `xformers`, ... |
| `torch.compile` | see the block below | the notebook uses `native` attention when compiling |

Notebook-style block compile (only for repeated runs; call it after loading and before `enable_model_cpu_offload`):

```python
from diffusers.models.transformers.transformer_kandinsky6 import (
    Kandinsky6FusedTransformerDecoderBlock,
    Kandinsky6TransformerDecoderBlock,
    Kandinsky6TransformerEncoderBlock,
)

torch.set_float32_matmul_precision("high")
pipe.transformer.set_attention_backend("native")
for block in pipe.transformer.modules():
    if isinstance(block, Kandinsky6FusedTransformerDecoderBlock):
        block.forward = torch.compile(block.forward)
    elif isinstance(block, (Kandinsky6TransformerEncoderBlock, Kandinsky6TransformerDecoderBlock)):
        block.forward = torch.compile(block.forward, mode="max-autotune-no-cudagraphs", dynamic=True)
```

## 4. Video super-resolution (`Kandinsky6SRPipeline`)

Like `Kandinsky6TI2VAPipeline`, this pipeline only takes and returns tensors — no file I/O. Use
`diffusers.utils.load_video` / `encode_video` (or your own decoder) to bridge to mp4 files.

Inputs and constants:

- `video`: `uint8` pixels in `[0, 255]`, shape `(3, T, H, W)` or a batch `(B, 3, T, H, W)` — the same layout
  `Kandinsky6TI2VAPipeline` returns with `output_type="pt"`. `T` must be `1 + 8k` (the SR model's temporal chunk
  size; align/truncate yourself — see `notebooks/kandinsky6_sr.py` for a `load_video`-based helper). A batch requires
  every clip to already share resolution and frame count.
- `latents`: a pre-encoded SR-VAE latent, shape `(C, T', H', W')` or a batch `(B, C, T', H', W')`. Mutually exclusive
  with `video`. Pass `kvae_bridge=True` when the latents come from a *different* (base T2VA) VAE — the pipeline
  decodes them through `source_vae` and re-encodes in its own latent space before tiling.
- `resolution_scale` is `2`, `2.25` (default), or `4`. The `2.25` route is a x1.125 pixel pre-upscale followed by the
  x2 model. It is the route used after K6 generation.
- Other knobs are `min_overlap` (minimum tile overlap fraction), `tiles_batch_size`, and `output_type` (`"pt"` or
  `"np"`).

### 4.1 Launch 4: VSR flow-matching

```python
import numpy as np
import torch
from diffusers import Kandinsky6SRPipeline
from diffusers.utils import encode_video, load_video

pipe = Kandinsky6SRPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload(gpu_id=0)   # or: pipe.to("cuda")

frames, fps = load_video("input.mp4", return_fps=True)
video = torch.stack([torch.as_tensor(np.array(f)) for f in frames]).permute(3, 0, 1, 2)  # [3,T,H,W] uint8
video = video[:, : 1 + 8 * ((video.shape[1] - 1) // 8)]                                  # align to 1+8k frames

result = pipe(
    video=video,
    resolution_scale=2.25,                # 2, 2.25 or 4
    num_inference_steps=5,                # 4 Euler steps per tile (default)
    generator=torch.Generator("cuda:0").manual_seed(1137),
)
print(result.frames.shape)                # [1, 3, T, H*scale, W*scale] uint8
encode_video(result.frames[0].permute(1, 2, 3, 0), fps=round(fps), output_path="output_sr.mp4")
```

Batch call (equal-size clips, stacked into one tensor):

```python
result = pipe(
    video=torch.stack([video_a, video_b]),  # both [3,T,H,W], same shape
    resolution_scale=2.25,
    output_type="pt",
    generator=torch.Generator("cuda:0").manual_seed(1137),
)
```

### 4.2 Launch 5: VSR distilled (2 evaluations per tile)

The same code with the other repo. Do not pass `num_inference_steps`; it has no effect here.

```python
pipe = Kandinsky6SRPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers", dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload(gpu_id=0)
result = pipe(video=video, resolution_scale=2, generator=torch.Generator("cuda:0").manual_seed(1137))
```

## 5. Launch 6: T2VA → SR chain

Both pipelines stay independent. Stage 2 takes the stage-1 video tensor directly, with no intermediate mp4; the
source audio is kept aside and muxed in only at the very end. This version uses the Hub repos instead of the local
bundles in the stacked notebook.

```python
import torch
from diffusers import Kandinsky6SRPipeline, Kandinsky6TI2VAPipeline
from diffusers.utils import encode_video

GPU_ID, FPS = 0, 24.0

t2va = Kandinsky6TI2VAPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-Pro-sft-5s-Diffusers", dtype=torch.bfloat16)
t2va.enable_model_cpu_offload(gpu_id=GPU_ID)

sr = Kandinsky6SRPipeline.from_pretrained(
    "kandinskylab/Kandinsky-6.0-VSR-5s-Diffusers", dtype=torch.bfloat16)
sr.enable_model_cpu_offload(gpu_id=GPU_ID)

gen = t2va(
    prompt="A cinematic shot of a small red sailboat crossing a calm blue lake at sunset, "
           "with gentle waves and natural ambient sound.",
    negative_prompt="blurry, distorted, low quality, noisy audio",
    height=480, width=864,
    num_frames=121,                       # not 125: SR keeps at most 121 frames
    sample_fps=FPS, num_inference_steps=50, guidance_scale=5.0,
    sample_audio=True, output_type="pt",
)

out = sr(video=gen.frames, resolution_scale=2.25, output_type="pt")  # uint8 [B,3,T,H,W] in, out
print(tuple(out.frames.shape))

audio = torch.as_tensor(gen.audio[0])[:, None].repeat(1, 2)
encode_video(
    out.frames[0].permute(1, 2, 3, 0),
    fps=int(FPS),
    output_path="t2va_sr.mp4",
    audio=audio,
    audio_sample_rate=int(t2va.audio_sample_rate),
)
```

The output is 2.25x the generation size: `480 x 864` becomes about `1080 x 1944`. For the fast variant swap in the two
distilled repos, `STEPS=16`, `guidance_scale=1.0`, and drop `negative_prompt`.

## 6. Making the notebooks run

The three notebooks in `diffusers-new-model-addition-kandinskyX/notebooks/` were copied from a `k6_video` layout.
Three edits are needed:

1. **Repo detection.** Each notebook walks up until `dev/exports/diffusers` exists. That folder is in `k6_video`, not in
   the diffusers checkout, so the first cell raises `RuntimeError`. Replace that block with
   `from pathlib import Path; REPO = Path.cwd()`.
2. **Inputs.** `assets/vsr_input.mp4` does not exist anywhere in the workspace. Point `INPUT_VIDEO` at a real mp4.
   `assets/i2va_input.png` exists only in `k6_video/assets/`.
3. **Stacked notebook.** It loads `outputs/diffusers_bundle/kandinsky6_diffusers` and `.../kandinsky6_sr_diffusers`,
   which are local bundles from `k6_video`. Replace them with the Hub ids from section 1. It also generates
   `num_frames=125`; `Kandinsky6SRPipeline` requires `1 + 8k` frames and raises `ValueError` on anything else
   (it no longer silently truncates), so use 121.

Also remove the `HF_TOKEN = "TOKEN"` line in `kandinsky6_ti2va.ipynb`, and register the kernel with
`python -m ipykernel install --user --name k6-diffusers` from the activated venv.

## 7. Building a local Diffusers bundle from a native checkpoint (only if needed)

Official Hub repos exist for every launch above, so this is not needed for normal use. To make a bundle from a native
checkpoint, the `k6_video` repo (its own `.venv` with `just setup`) provides:

```bash
BASE_PATH="something"   # folder that contains the K6 repos
cd "$BASE_PATH/k6_video"
just convert-diffusers    --config src/kandinsky/configs/k6_pro_125_480_864_mCache_mOffload.yaml --output-dir outputs/kandinsky6_diffusers
just convert-diffusers-sr --config src/kandinsky/configs/k6_pro_125_480_864_mCache_mOffload.yaml --output-dir outputs/kandinsky6_sr_diffusers
```

Bundles created this way are loaded with the same `from_pretrained("<local dir>")` call. The SR recipe above needs
private checkpoints and extra flags (`--checkpoint-path`, `--vae-path`, `--latent-upscaler-config`); see the `JUSTFILE`
comments. `just integrate-diffusers <checkout>` overlays the generated K6 code onto a Diffusers checkout. It does not
commit or push.

## 8. Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `ImportError: cannot import name 'Kandinsky6TI2VAPipeline'` | stock Diffusers is installed; activate the venv where this branch is installed with `-e` |
| `401` / `RepositoryNotFoundError` on a K6 repo | not logged in, or the account has no access; `hf auth login` |
| `ValueError: PiflowScheduler requires guidance_weight=1.0` | distilled repo with `guidance_scale != 1.0` |
| `ValueError: height and width must be divisible by 8` | choose sizes divisible by 8 (480x864 works) |
| `ValueError: Expected samples with 2 channels` in `encode_video` | audio must be stereo: `torch.as_tensor(a)[:, None].repeat(1, 2)` |
| `No module named 'av'` | `uv pip install av` (needed by `encode_video` for mp4 output, with or without audio) |
| Error setting `_flash_3` | FlashAttention 3 not available; use `native` |
| CUDA OOM | make sure `enable_model_cpu_offload()` is on; lower `height`/`width`/`num_frames`; use the distilled model |
| `ValueError: \`video\` must have 1 + 8k frames` / `... at most 121 frames` | align/split the input yourself: 121 frames at 24 fps is about 5 s |

## Verification status

Checked on the Mac (CPU only, no GPU):

- The branch imports; every keyword used above exists in the current signatures of `Kandinsky6TI2VAPipeline.__call__`,
  `Kandinsky6SRPipeline.__call__`, `encode_video`, `load_video`, `MagCacheConfig` and
  `transformer.set_attention_backend`/`enable_cache`. `dtype=` and `torch_dtype=` are both accepted by `from_pretrained`.
- `_flash_3` and `native` are valid attention-backend names, and `PiflowScheduler` guidance is enforced as stated.
- Defaults, output layouts (`uint8 [B,3,T,H,W]`), the SR tensor layouts, and the 121-frame clip come from the code.
- The three Hub repos `Pro-sft-5s`, `VSR-5s` and `VSR-distilled2steps-5s` (all `-Diffusers`) were read from the local HF
  cache. Their `model_index.json`, scheduler and component configs match what is written here.

**Not verified:**

- No end-to-end run happened: no model weights were loaded and nothing was generated. Runtime, peak VRAM and output
  quality are untested.
- `Kandinsky-6.0-Pro-distill-5s-Diffusers` is not in the local cache and cannot be checked without access. Its id and
  the 16-step / guidance-1.0 settings come from your notebook and the class docstring.
- The MagCache path was checked against `MagCacheConfig`'s signature, not run.
