<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MiniMax-H3

MiniMax-H3 generates video and its soundtrack **jointly**. One transformer denoises a single packed sequence that holds the text conditioning, the conditioning image, video and audio rows, the target audio rows and the target video rows at once, with full self-attention over all of it. There is no separate vocoder and no audio post-hoc pass: video and audio come out of the same denoising loop.

You can find the original MiniMax-H3 checkpoints under the [MiniMaxAI](https://huggingface.co/MiniMaxAI) organization.

MiniMax-H3 is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only, the way [Anima](./anima) is: the blocks and their [`MiniMaxH3ModularPipeline`] are the whole integration, and there is no `DiffusionPipeline` half.

## Checkpoint layout

MiniMax-H3 was released as two checkpoint partitions that share every component except the transformer, so the diffusers conversion puts both in **one repository**:

| Subfolder | Blocks | Tasks |
|---|---|---|
| `transformer/` | [`MiniMaxH3Blocks`] | `t2va` (text only) and `fl2va` (first and/or last keyframe) |
| `transformer_ref/` | [`MiniMaxH3Ref2VABlocks`] | `ref2va` (an ordered mix of image, video and audio references) |

Everything but the transformer, i.e. the video VAE, the audio VAE, the Qwen3-VL conditioner, its tokenizer and processor, and the two schedulers, is shared and stored once.

The repository carries one `modular_model_index.json`, which names every component of both halves with its own loading spec. Each blockset declares only the components it runs, and `load_components` fetches exactly those subfolders: loading the `t2va` / `fl2va` half never touches `transformer_ref/`, and loading the `ref2va` half never touches `transformer/`. Nothing else in the repository is fetched either, which is what lets one repository carry the two partitions, and the original checkpoint folders next to the converted ones.

`modular_model_index.json` names the `t2va` / `fl2va` half as its own class, so [`~ModularPipeline.from_pretrained`] resolves that half. The `ref2va` half reads the very same file through its own blocks:

```py
import torch
from diffusers import ModularPipeline
from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks

# `t2va` / `fl2va`: loads `transformer/`, and never `transformer_ref/`.
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")

# `ref2va`: loads `transformer_ref/`, and never `transformer/`, out of the same repository.
pipe = MiniMaxH3Ref2VABlocks().init_pipeline("MiniMaxAI/MiniMax-H3")

pipe.load_components(dtype=torch.bfloat16)
```

Each blockset also carries the workflows it serves — `t2va`, `fl2va` and `fl2va_last_frame` for [`MiniMaxH3Blocks`], `ref2va` for [`MiniMaxH3Ref2VABlocks`] — which name the inputs each task requires.

The conditioner is a `Qwen3VLForConditionalGeneration`, and MiniMax-H3 reads the *unnormalized* hidden state after its 50th decoder layer rather than the last one, so the full released checkpoint is used with its language-model head unused.

## Two schedulers

Video and audio latents step down two different schedules inside a single transformer call per step, which is why both blocksets expect two [`MiniMaxH3Scheduler`] instances: `scheduler` for the video latents (`shift=12.0` in the released checkpoints) and `audio_scheduler` for the audio latents (`shift=3.0`).

The checkpoint is guidance-distilled: guidance is baked into the weights, so there is no guider, no `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass.

## Generation constraints

- **24 fps, 5 to 15 seconds.** `num_frames` is snapped up to the next `17 * n + 5` the video VAE can decode, and the resulting duration has to stay in that window.
- **A 768 pixel short edge by default.** `height` and `width` default to MiniMax-H3's own canvas for the aspect ratio of the first keyframe (or 16:9 without one) and must be multiples of 32. 768 is what it was trained at rather than what the code allows, and smaller canvases are the main lever on a small card: see [the canvas ladder](#the-canvas-ladder).
- **One generator, three draws.** A request draws the keyframe or reference conditioning noise first, then the video noise, then the audio noise, all from the `generator` it is passed, so two runs from the same generator state return the same video and soundtrack. Passing `latents` or `audio_latents` replaces the corresponding draw.
- **`num_inference_steps` counts sigma grid points**, the terminal `0` included, so it drives one model evaluation less.

## Performance and hardware recipes

MiniMax-H3 denoises one packed sequence that carries the video rows and the audio rows together, so a request is a large activation problem sitting on top of very large weights. Nothing here fits on one card by default: the transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB.

Two resources decide what a machine can run, and they are not the same resource.

**Video memory decides the canvas.** Under block level [group offloading](../../optimization/memory#group-offloading) the card never holds a whole component: it holds the block that is running, the block being prefetched, and the activations. Two transformer blocks are 1.3 GB at int8, so from 12 GB upward the weights are not what fills the card. The activations are, and they scale with the canvas.

**System memory decides the width.** Every byte the card is not holding sits in host RAM, and there is no disk fallback: diffusers refuses to offload torchao tensors to disk, because safetensors cannot serialize the subclass. A `t2va` request keeps the transformer, the conditioner and both autoencoders resident at once, 75 GB of them at int8. That, not the GPU, is what rules out a 32 GB desktop, and it is what [splitting the conditioner from the denoiser](#splitting-the-conditioner-from-the-denoiser) is for.

### Weight footprints

Both transformer partitions are the same size, and the two autoencoders are pinned to float32. Every figure is the size of the published checkpoint on disk, which is also its residency in host RAM.

| Component | bfloat16 | float8 | int8 | int4 | NVFP4 |
|---|---|---|---|---|---|
| `transformer` (and `transformer_ref`, each) | 61.73 GB | 30.93 GB | 31.69 GB | 19.46 GB | 17.42 GB |
| `text_encoder` (Qwen3-VL) | 62.13 GB | not quantized | 33.10 GB | 19.40 GB | not quantized |
| `vae` (video) | 9.70 GB, float32 | | | | |
| `audio_vae` | 0.56 GB, float32 | | | | |

int4 is not a clean quarter of bfloat16, and the reason matters before you budget for it. The tinygemm packing pads `in_features` up to a multiple of 1024, and MiniMax-H3's projections are 5376 and 2688 wide, so 5376 becomes 6144 and 2688 becomes 3072. With the group scales at one bfloat16 pair per 128 weights the transformer lands at 19.5 GB rather than 15.4 GB.

Only `transformer_blocks` is quantized, 32.28 B of the transformer's 33.12 B parameters. The input and output projections, the timestep embedder, the two token refiner blocks and the output AdaLN stay bfloat16, 1.7 GB that buys back the numerically sensitive parts. The conditioner quantizes its 64 language model decoder layers and leaves the vision tower, the token embeddings and the unused `lm_head` alone.

Both autoencoders carry `_keep_in_fp32_modules` over every module, so a `dtype` passed to `load_components` is refused for them and they stay float32. That is deliberate and automatic: a bfloat16 audio VAE decodes the soundtrack roughly 20 dB too quiet.

Do not read residency off the model. A torchao serialized checkpoint keeps reporting the bfloat16 numbers through `model.dtype` and `get_memory_footprint()`, because the tensor subclasses report the dtype they present rather than the dtype they store. The figures above are file sizes, which are honest.

### Which width your card can run

| Architecture | Cards | bfloat16 | int8 | int4 tinygemm | float8 | NVFP4 |
|---|---|---|---|---|---|---|
| sm86 Ampere | RTX 3060, 3090 | yes | yes | yes | no | no |
| sm89 Ada | RTX 4070, 4080, 4090 | yes | yes | yes | yes | no |
| sm90 Hopper | H100 | yes | yes | yes | yes | no |
| sm120 Blackwell | RTX 5090, RTX PRO 6000 | yes | yes | yes | yes | yes |

int8 exists for the two rows that have no float8. Ampere has int8 tensor cores and no float8 unit at all, and NVFP4's dynamic activation transform asserts sm100 or newer. **int8 is the width to use on every consumer card.** It is the closest of all the quantized widths to bfloat16, it costs nothing in step time, and one checkpoint covers every architecture in the table.

int4 is measured here and is not recommended. It saves 12 GB of host RAM over int8 and charges 8.4x the step time for it, because `aten._weight_int4pack_mm` is a batch-one decode kernel being asked to multiply a 37726 row sequence, and its soundtrack does not survive the width at all.

### The canvas ladder

The canvas is the largest lever on this model, larger than any quantization choice, because the packed sequence grows with it and attention grows with its square. MiniMax-H3 was trained with a 768 pixel short edge, which is a fact about the training data rather than a constraint in the code: `height` and `width` only have to be multiples of 32. Smaller canvases are out of distribution and they hold up, validated visually down to 960x544 including the prompt classes that break first, anatomy and dense detail.

| Canvas | Video rows | Packed rows | Relative cost | |
|---|---|---|---|---|
| 1344x768 | 37296 | 37726 | 1.00 | the trained canvas, 16:9 |
| 1280x704 | 32560 | 32990 | 0.87 | |
| 1152x640 | 26640 | 27070 | 0.71 | |
| 1024x576 | 21312 | 21742 | 0.58 | |
| 960x544 | 18870 | 19300 | 0.51 | the recommended local default, 16:9 |
| 544x960 | 18870 | 19300 | 0.51 | the same budget, portrait |
| 544x544 | 10693 | 11123 | 0.29 | square |

On local hardware start at 960x544, 544x960 or 544x544, and climb only if the card and the patience are there. 960x544 is 2.26x cheaper per step than 1344x768 on the same silicon, a larger factor than any width in this document buys.

### Configuration by VRAM class

| Class | Repository | Canvas | Video VAE | Peak reserved | s/it, H100 under cap | Host RAM, weights |
|---|---|---|---|---|---|---|
| 12 GB | `MiniMax-H3`, int8 at load | 960x544 | streamed | 9.78 GB | 5.18 | 75 GB, or 42 GB split |
| 16 GB | `MiniMax-H3`, int8 at load | 1152x640 | streamed | 11.71 GB | 7.01 | 75 GB, or 42 GB split |
| 24 GB | `MiniMax-H3`, int8 at load | 1152x640 | resident | 21.43 GB | 7.50 | 75 GB, or 42 GB split |
| 32 GB | `MiniMax-H3`, int8 at load | 1344x768 | resident | 24.00 GB | 11.81 | 75 GB, or 42 GB split |
| 80 GB | `MiniMax-H3`, optionally float8 at load | 1344x768 | resident | 62.2 to 69.7 GB | 9.21 | 134 GB |
| 96 GB | `MiniMax-H3`, split | 1344x768 | resident | 72 GB | 5.5 at 960x544 | 72 GB per process |

Every consumer row is the same recipe with two lines changed. Read `s/it` as an H100 number under an allocator cap, comparable across rows and to nothing else; [what the consumer cards themselves will do](#what-actually-binds-on-a-consumer-card) is estimated separately.

### The shape every consumer recipe has

```py
import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
from diffusers import TorchAoConfig as DiffusersTorchAoConfig
from diffusers.hooks import apply_group_offloading
from diffusers.utils.export_utils import encode_video
from torchao.quantization import Int8WeightOnlyConfig, PerRow
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")

# Both large components are quantized at load, shard by shard, straight from the bfloat16 checkpoint: peak host
# cost during loading is the quantized size plus one shard, not the bfloat16 size. Only `transformer_blocks` is
# quantized; the projections, the timestep embedder, the token refiner and the output heads are the numerically
# sensitive 1.7 GB and stay bfloat16. Registering both components before `load_components` makes it skip them.
pipe.update_components(
    transformer=MiniMaxH3Transformer3DModel.from_pretrained(
        "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=DiffusersTorchAoConfig(
            "int8wo",
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                "token_refiner", "norm_out", "proj_out", "audio_proj_out",
            ],
        ),
    ),
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        "MiniMaxAI/MiniMax-H3",
        subfolder="text_encoder",
        dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2, granularity=PerRow()),
            modules_to_not_convert=[
                "model.visual",
                "model.language_model.embed_tokens",
                "model.language_model.norm",
                "lm_head",
            ],
        ),
    ),
)
pipe.load_components(dtype=torch.bfloat16)

# Inference only, and load bearing rather than hygiene: onloading a streamed group swaps parameters through
# `torch.utils.swap_tensors`, whose use count check takes an autograd path an int8 tensor cannot serve while it
# still requires gradients.
for model in (pipe.transformer, pipe.text_encoder, pipe.vae):
    model.requires_grad_(False)

offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True)

# Block level for the transformer, whose `transformer_blocks` is a direct child ModuleList. One block per group is
# not a tuning choice: diffusers resets anything else when `use_stream=True`.
pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)

# Leaf level for the conditioner, and on `text_encoder.model`. MiniMax-H3 reads `hidden_states[50]` and never uses
# the language model head, so the block calls the submodule directly and a hook on `text_encoder` would never
# fire. Block level would not work either: Qwen3-VL's direct children are `visual` and `language_model`, neither a
# ModuleList, so the whole conditioner would become one group and onloading it would fill the card.
apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)

# Leaf level for the video VAE too, unstreamed, up to 16 GB. `decode()` is not `forward()`, so block level never
# onloads the group holding `post_quant_conv` and the tiled decode dies under the float16 autocast. From 24 GB up
# replace this with `pipe.vae.to("cuda")`, which is worth 15 s of decode against 7 minutes.
pipe.vae.enable_group_offload(
    offload_type="leaf_level", onload_device=torch.device("cuda"), offload_device=torch.device("cpu")
)
pipe.audio_vae.to("cuda")  # 0.56 GB, float32

# SDPA. The audio VAE is pinned first because `set_attention_backend` also sets the process wide default, and
# float32 attention cannot run an int8 or float8 kernel.
pipe.audio_vae.set_attention_backend("native")
pipe.transformer.set_attention_backend("native")

state = pipe(
    prompt="A red fox trotting through a snowy pine forest, snow crunching underfoot",
    height=544,
    width=960,
    num_frames=124,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(42),
)
encode_video(
    state.get("videos")[0],
    fps=24,
    output_path="fox.mp4",
    audio=state.get("audio")[0],
    audio_sample_rate=state.get("sampling_rate"),
)
```

#### 12 GB, RTX 3060 and RTX 4070

The recipe above unchanged. Measured under an 11.1 GB allocator budget: 6.67 GB peak allocated, 9.78 GB peak reserved, 2.2 GB spare. The card is not the problem at this tier.

#### 16 GB, RTX 4080 and RTX 4060 Ti 16GB

One canvas step up, `height=640, width=1152`. 7.98 GB allocated, 11.71 GB reserved, 4.3 GB spare. 1344x768 with the VAE streamed reserved 14.30 GB when it was measured under the 32 GB cap, so it should also fit here; the ceiling at 16 GB is not the canvas but how long the streamed decode takes.

#### 24 GB, RTX 3090 and RTX 4090

The first tier with room to keep the float32 video VAE resident, which is a 9.84 s decode instead of a 420 s one. Replace the VAE offload call with `pipe.vae.to("cuda")` and use `height=640, width=1152`: 17.69 GB allocated, 21.43 GB reserved, 2.6 GB spare. 1344x768 reserved 24.00 GB under the 32 GB cap, which is over the line on a 24 GB card, so at that canvas the VAE goes back to streaming.

#### 32 GB, RTX 5090

`pipe.vae.to("cuda")` and `height=768, width=1344`: 19.49 GB allocated, 24.00 GB reserved, 8 GB spare. On a 5090 also set `pipe.transformer.set_attention_backend("_native_cudnn")`, which beats SDPA on sm120 where flash-attention 4 does not.

#### 80 GB datacenter, H100

The only class that can hold a whole component, so use [`ComponentsManager`] instead of group offloading: moving entire models around block boundaries is cheaper than streaming when the model fits.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")

pipe.audio_vae.set_attention_backend("native")
pipe.transformer.set_attention_backend("_flash_3_hub")  # Hopper only, roughly 3x over SDPA
```

The reserve margin has to be wider than the 3 GB default. [`~ComponentsManager.enable_auto_cpu_offload`] decides what to evict from the weight sizes alone and has no idea how much room the activations will want. On the keyframe and reference tasks the float32 video VAE runs before the denoiser, and with the default margin the manager leaves it resident next to the 61.7 GB transformer: 72 GB of weights on a 79.2 GB card, which runs out of memory inside the first block. A 12 GB margin evicts the VAE before the transformer's forward and leaves about 17 GB for activations.

Quantizing the transformer at load, float8 or int8 as in the recipe below, holds it resident at about 31 GB. It does not change the step time; it removes the offload traffic, and a cold `t2va` request finishes in 491 s float8 or 529 s int8 against 687 s bfloat16.

#### 96 GB Blackwell, RTX PRO 6000

Do not quantize. Split instead, and keep reference quality. See below.

### Splitting the conditioner from the denoiser

The largest structural win is not a quantization width. It is that the conditioner and the denoiser never run at the same time, so they do not have to be in the same process.

The text encoder runs exactly once per request and produces two small tensors: `prompt_embeds` of shape `(1, N, 5120)` and `text_token_tags` of shape `(N,)`, N being a few dozen rows. Everything after that belongs to the denoiser. So the requirement stops being the sum of the components and becomes the largest of them.

| Arrangement | Requirement | bfloat16 |
|---|---|---|
| one process | transformer + conditioner + autoencoders | 61.7 + 62.1 + 10.3 = 134.1 GB |
| split | max(transformer + autoencoders, conditioner) | max(72.0, 62.1) = 72.0 GB |

That is what makes full bfloat16, quantized nowhere and therefore at reference quality, fit a 95 GB budget. It is the deployed configuration on two ZeroGPU Spaces of an RTX PRO 6000 slice: one holds the conditioner and exposes `encode` over its Gradio API, the other holds the transformer and both autoencoders and runs 960x544 at 5.5 s/step eager, 4.4 s/step with AoTI compiled blocks. It also satisfies the 150 GB per Space storage quota, which one Space holding all of bfloat16 MiniMax-H3 does not.

The modular blocks make it natural, because the text encoder is one block and its output is one state entry.

```py
# Conditioner process, or service. Loads `text_encoder`, `tokenizer` and `processor`, and nothing else.
embeds = conditioner.encode(prompt)  # prompt_embeds (1, N, 5120), text_token_tags (N,)

# Denoiser process. Pop the text encoder block and hand it the embeds instead.
from diffusers.modular_pipelines import MiniMaxH3Blocks

blocks = MiniMaxH3Blocks()
blocks.sub_blocks.pop("text_encoder")
pipe = blocks.init_pipeline("MiniMaxAI/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)  # `text_encoder` is no longer a declared component

state = pipe(
    prompt_embeds=embeds["prompt_embeds"].to("cuda"),
    text_token_tags=embeds["text_token_tags"].to("cuda"),
    height=544,
    width=960,
    num_frames=124,
    num_inference_steps=30,
)
```

Three places it pays, and one where it does not.

**A 95 GB slice, or any single card between 80 and 96 GB.** Split bfloat16 beats anything quantized in one process, because its quality is the reference quality. Reach for this first.

**Two cards.** The conditioner takes the second card, quantized to int8 if that card is small, and the transformer keeps the first to itself. Neither has to hold both.

**One small card, run sequentially.** The halves become two invocations of the same script: encode, write the embeds, exit, denoise. Nothing changes on the GPU, but host RAM stops having to hold both halves at once, and on a consumer box host RAM binds long before VRAM does. At int8 that is 42 GB of resident weights instead of 75 GB.

**One small card, one process.** Nothing is gained. The components were already streaming one block at a time, and splitting only adds the cost of moving the embeds.

### Measured under an emulated cap

Each consumer row was run on an H100 80GB PCIe with `torch.cuda.set_per_process_memory_fraction` set to the class size less 0.9 GB for the CUDA context, so an allocation past the budget raises the same out of memory error a smaller card would. That emulates the budget, not the silicon: **the timings are H100 timings and are comparable to each other and to nothing else.** What transfers is the fit. Every row generated 124 frames at 24 fps and decoded them; the step count is 8 rather than 30, which does not move the allocator high water mark, because that is set by the first forward and by the decode tail and both ran in full.

| Row | Width | Canvas | Video VAE | Peak alloc | Peak reserved | Spare | s/it | Decode | Host RSS | Fits |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 GB | int8 | 960x544 | streamed | 6.67 GB | 9.78 GB | 2.22 GB | 5.18 | 453 s | 133.8 GB | yes |
| 12 GB | int4 | 960x544 | streamed | 6.19 GB | 10.65 GB | 1.35 GB | 40.60 | 511 s | 91.2 GB | yes |
| 12 GB, unstreamed | int4 | 960x544 | streamed | 5.82 GB | 7.59 GB | 4.41 GB | 49.48 | 425 s | 126.8 GB | yes |
| 16 GB | int8 | 1152x640 | streamed | 7.98 GB | 11.71 GB | 4.29 GB | 7.01 | 420 s | 133.8 GB | yes |
| 24 GB | int8 | 1152x640 | resident | 17.69 GB | 21.43 GB | 2.57 GB | 7.50 | 9.8 s | 135.8 GB | yes |
| 32 GB | int8 | 1344x768 | resident | 19.49 GB | 24.00 GB | 8.00 GB | 11.81 | 15.2 s | 130.9 GB | yes |
| 32 GB, VAE streamed | int8 | 1344x768 | streamed | 9.78 GB | 14.30 GB | 17.70 GB | 11.79 | 902 s | 134.4 GB | yes |
| 32 GB, regional compile | int8 | 1344x768 | resident | | | | | | | no, see below |

Host RSS is the peak resident set size of the whole process, which is what a machine has to have in system RAM: the
weights plus the pinned host copies streamed offloading keeps. It is set by the widths, not by the card, which is
why four of those rows agree on it and why [splitting the conditioner out](#splitting-the-conditioner-from-the-denoiser)
matters more than any of them.

Four things fall out of that table.



The video VAE is the whole story below 24 GB. Streaming it holds the decode to under a gigabyte, which is what lets 1344x768 fit in 14.30 GB reserved, but it re-reads 9.7 GB of float32 weights for every tile and the decode goes from 15 s to 902 s. Resident, it costs 9.7 GB and decodes in seconds. There is no middle setting, and the crossover is at 24 GB.

The transformer is not what fills the card. At 1344x768 with everything streaming, peak allocated is 9.78 GB against 31.69 GB of weights, because only two blocks are ever resident.

int4 costs more video memory than int8, not less. 10.65 GB reserved against 9.78 GB, despite weights that are 12 GB smaller, because the tinygemm path pads every activation up to a multiple of 1024 before the matmul and that padded copy is 464 MB per projection at this canvas. What int4 buys is host RAM, 91.2 GB against 133.8 GB, and it charges 7.8x the step time for it.

Streaming is worth about 18 percent. On the int4 pair, `use_stream=True` runs at 40.60 s/it against 49.48 s/it, for 3.06 GB more reserved and 58 s of setup. It does not reduce host RAM; the unstreamed row measured higher, because offloading without a stream allocates fresh host tensors instead of restoring pinned ones.

Full precision reference rows, on the same card with no cap and [`ComponentsManager`] auto offload at a 12 GB margin, 1344x768, 124 frames, 30 steps, seed 42, flash-attention 3:

| Recipe | Task | Load | Peak alloc | s/it | Decode | Wall |
|---|---|---|---|---|---|---|
| bfloat16 | `t2va` | 181 s | 68.1 GB | 9.21 | 94.0 s | 687 s |
| bfloat16 | `fl2va`, first and last frame | 220 s | 68.8 GB | 10.83 | 50.1 s | 622 s |
| bfloat16 | `ref2va`, one video and one image reference | 167 s | 74.3 GB | 26.23 | 48.1 s | 1184 s |
| bfloat16, regional compile | `t2va` | 232 s | 66.4 GB | 8.21 | 55.4 s | 680 s |
| float8 transformer | `t2va` | 201 s | 62.2 GB | 9.25 | 15.3 s | 491 s |
| int8 transformer | `t2va` | 235 s | 62.2 GB | 9.67 | 60.0 s | 529 s |
| int4 transformer | `t2va` | 229 s | 62.2 GB | 77.04 | 48.0 s | 2333 s |

`ref2va` is 2.8x the per step cost of `t2va` at the same output size, because the reference rows lengthen the sequence every layer attends over. int8 costs 5 percent over bfloat16 per step and int4 costs 8.4x, which is the single strongest argument for int8 in this document.

Note the peak in those rows: 62.2 GB for every quantized one, and that is the *conditioner*, not the transformer. Quantizing the transformer alone moves the high water mark onto the bfloat16 conditioner and the ceiling stops falling. Quantizing or offloading both is what makes a tier real.

### Quality per width

One `t2va` request, seed 42, 1344x768, 124 frames, 30 steps, each width against the bfloat16 sample of the same request.

| Width | Video cosine | Video PSNR | Latents cosine | Audio cosine | Verdict |
|---|---|---|---|---|---|
| int8 | 0.984 | 19.36 dB | 0.953 | 0.927 | closest to bfloat16 of any width here |
| float8 | 0.943 | 13.90 dB | 0.860 | 0.781 | a different sample of the same prompt |
| NVFP4 | 0.916 | 12.07 dB | | | plus a visible brightness shift |
| int4 | 0.891 | 11.03 dB | 0.688 | 0.217 | soundtrack does not survive |

The ordering is not the one the bit widths suggest, and the reason is which tensors each width touches. int8 here is weight only, so the activations stay bfloat16; float8 and NVFP4 quantize the activations too, and that costs more than the extra bits buy. int4 is weight only as well, but at four bits with a 128 wide group the audio rows, which are 414 of 37726 and carry a much smaller dynamic range than the video rows, come out five times too loud at 0.217 cosine. Look at the audio column before choosing a width for a model that generates sound.

Quantization ends bitwise reproducibility either way. Two runs at the same seed and the same width agree; two widths do not. Pin the width along with the seed.

### Attention backend by architecture

| Architecture | Backend | Why |
|---|---|---|
| sm86 Ampere | `"native"`, or `"sage"` with `sageattention` installed | no flash-attention 3 build exists; SageAttention dispatches sm86 to its triton int8 QK kernel |
| sm89 Ada | `"native"`, or `"sage"` | SageAttention dispatches sm89 to `qk_int8_pv_fp8_cuda` |
| sm90 Hopper | `"_flash_3_hub"` | prebuilt from the Hub, no local build, roughly 3x over SDPA |
| sm120 Blackwell | `"_native_cudnn"` | beats SDPA; flash-attention 4 loses to it |

`_flash_3_hub` kernels are compiled for Hopper only, so an Ada, Blackwell or consumer card has no matching image and the call fails at dispatch. `"sage"` is the name to use rather than one of the `_sage_qk_*` variants, because SageAttention picks the kernel from the compute capability itself and the variant that is right for one architecture will not run on another.

Whatever the backend, pin the audio VAE to `"native"` first. [`~ModelMixin.set_attention_backend`] sets the process wide default as well as the model's own, the float32 audio VAE runs its own encoder attention whenever a reference carries audio, and none of the int8 or float8 attention kernels accept float32.

### Regional compilation

[`~ModelMixin.compile_repeated_blocks`] compiles the repeated block classes rather than the whole model, so compile time is one block instead of fifty. On the 80 GB bfloat16 `t2va` row it adds about 51 s to the first step and takes the steady step from 9.21 s to 8.21 s.

It does not compose with streamed group offloading, and the attempt to measure that is the evidence. Onloading a block replaces its parameters, so dynamo's guards fail on the new tensors and the block recompiles; the 32 GB row hit the recompile cache limit and was still compiling twenty minutes later with the card pegged, against 101 s for the identical row eager. Use it where the weights are resident, which means the 80 GB and 96 GB classes, and leave it alone on a consumer card.

Call it without `fullgraph=True`. The point is a small compiled region repeated fifty times, and the surrounding pipeline keeps its Python control flow, the offload hooks and the two schedulers. Regional compilation is also not bitwise identical to eager.

### What actually binds on a consumer card

Streaming weights over PCIe is not what makes MiniMax-H3 slow on a small card. Arithmetic is.

One transformer forward over the packed sequence is `2 * rows * 32.28 B` parameters of matmul plus `4 * rows^2 * 7168 * 50` of attention. At 1344x768 and 124 frames that is 37726 rows and 4.48 PFLOP per step, and the measured 9.21 s/step on an H100 PCIe puts the achieved rate at 486 TFLOP/s, 64 percent of that card's 756 TFLOP/s dense bfloat16 peak. Scaling that efficiency onto other cards at their **float32 accumulate** tensor rate, which is what a torch matmul asks for and half the number NVIDIA puts on a GeForce box:

| Canvas | 3060 | 4070 | 3090 | 4080 | 4090 | 5090 |
|---|---|---|---|---|---|---|
| 960x544 | 109 s | 48 s | 39 s | 28 s | 17 s | 13 s |
| 1152x640 | 171 s | 75 s | 61 s | 45 s | 26 s | 21 s |
| 1344x768 | 274 s | 120 s | 99 s | 72 s | 42 s | 33 s |

Against that, the transfer term is nothing. Block level offloading walks the whole transformer onto the card once per step, 31.69 GB at int8, which over PCIe 4.0 x16 at a usable 25 GB/s is 1.36 s, and with `use_stream=True` it overlaps with the compute it feeds. On the slowest card in the table the compute term is eighty times the transfer term. PCIe does not bind at any tier, any width, any canvas.

So, plainly: **an RTX 3060 cannot generate 124 frames at 960x544 in under ten minutes at 30 steps.** The bound is about 109 s/step of arithmetic, 54 minutes for 30 steps, and no amount of streaming, quantization width or offload tuning moves it, because none of them change the FLOP count. Ten minutes buys about five steps at that canvas, or a 544x544 canvas at nine.

Three levers would move it, and only the first two exist today. Fewer steps. A smaller canvas. And a width that quantizes the *activations*: int8 weight only and int4 tinygemm both dequantize into bfloat16 and issue a bfloat16 matmul, so neither touches the compute term, while int8 dynamic activation quantization would run the matmul on Ampere's int8 tensor cores at twice the bfloat16 rate. That is worth about a third off the step time at 960x544, where the matmul is 71 percent of the work, and it is the obvious next thing to build.

### Known gaps

torchao 0.17.0 is the current release and is what these recipes are written against. Three things in the stack do not yet work, and the recipes above are written so that none of them is reached.

| Gap | What triggers it | What the recipes do instead |
|---|---|---|
| `aten.view` unregistered on `Int8Tensor` | streamed group offloading, whose `swap_tensors` use count check calls `param.view_as(param)`, but only while the parameter requires gradients | `requires_grad_(False)` first, which is correct for inference anyway |
| `aten.is_pinned` and `aten._pin_memory` unregistered on `Int4TilePackedTo4dTensor` | streamed group offloading of an int4 checkpoint | int4 is not a recommended width |
| transformers reruns `_init_weights` after loading | `from_pretrained` on a pre-quantized `text_encoder/`, which calls `Tensor.normal_` on an `Int8Tensor` and dies before reading a weight | quantize the conditioner at load, a supported public API producing the same weights |

The last one is a round trip bug rather than a missing feature: transformers writes a torchao conditioner as `...q_proj._weight_qdata` and on reload cannot match those names back to `q_proj.weight`, so the parameter is never marked as loaded and the initialization pass runs over it. The pre-quantized `text_encoder/` in each repository is correct and will load unchanged once that is fixed.

One more, in diffusers rather than torchao: do not pass `low_cpu_mem_usage=True` to group offloading with a torchao model. The host copy it keeps is `tensor.cpu()`, which for a tensor already on the CPU returns the same object, so the entry in `cpu_param_dict` *is* the parameter; onloading swaps in place through `swap_tensors` and destroys the copy the group would later restore from. Nothing raises. The weights simply never go back to the CPU and allocated memory climbs until it hits whatever the cap is.

## Text and keyframes

[`MiniMaxH3Blocks`] covers text-to-video-and-audio and keyframe conditioning. A keyframe can be the frame the video starts from (`image`), the frame it ends on (`last_image`), or both.

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"

# Text to video + audio.
state = pipe(prompt=prompt, generator=torch.Generator().manual_seed(42))

# First frame (and optionally last frame) to video + audio. The canvas follows the first keyframe.
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
state = pipe(prompt=prompt, image=image, generator=torch.Generator().manual_seed(42))

encode_video(
    state.get("videos")[0],
    fps=24,
    output_path="minimax_h3_fl2va.mp4",
    audio=state.get("audio")[0],
    audio_sample_rate=state.get("sampling_rate"),
)
```

Video and audio are generated jointly and come out of the call as separate outputs, `videos` and `audio`, next to the `sampling_rate` the soundtrack carries; muxing them into one file is left to the caller, e.g. with [`~utils.export_utils.encode_video`].

## Omni-references

[`MiniMaxH3Ref2VABlocks`] conditions on an ordered list of references: up to 9 images, 3 videos and 3 audio clips, 12 in total. The order is semantic. It labels the references in the prompt presentation (`"<Picture 1>"`, `"<Audio 1>"`, `"<Video 1>"`) and it advances the shared audio/video rotary clock, so reordering the same references is a different request.

Unlike the keyframes above, references do not bind the generated geometry: they are encoded at their own resolution and the target canvas defaults to MiniMax-H3's 16:9.

Every reference is a [`~modular_pipelines.minimax_h3.MiniMaxH3Reference`], and it takes a path or a URL as well as in-memory media: a path is decoded when the reference is built, with [`~utils.load_image`] for an image and [PyAV](https://github.com/PyAV-Org/PyAV) for a video or an audio file, so the blocks themselves never open a media file. Decoding brings the rates along, which is what the model needs: a video reference reads its frame rate off the container and adopts the container's soundtrack when it has one, and an audio reference its sample rate.

In-memory media declares the rates it carries instead, defaulting to MiniMax-H3's own: `fps=24.0` and the audio VAE's sample rate, so only self-generated data at another rate has to say so. An explicit `fps` or `sample_rate` also wins over a container whose metadata is wrong. Frames are resampled onto MiniMax-H3's own 24 fps by dropping and duplicating whole frames, and a waveform onto the audio VAE's sample rate. A video reference conditions on its motion **and**, when it carries an `audio` waveform, on that soundtrack, which is then packed as this reference's own.

```py
import torch
from diffusers.modular_pipelines import MiniMaxH3Ref2VABlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.utils import load_image, load_video
from diffusers.utils.export_utils import encode_video

pipe = MiniMaxH3Ref2VABlocks().init_pipeline("MiniMaxAI/MiniMax-H3")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

subject = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

# A reference decodes a path or a URL as it is built, rates included: the video brings its own frame rate and its
# soundtrack, the audio clip its sample rate.
state = pipe(
    prompt="The character speaks in time with the reference recording, natural lip movement",
    references=[
        MiniMaxH3Reference(image=subject),
        MiniMaxH3Reference(
            video="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        ),
        MiniMaxH3Reference(audio="voice.wav"),
    ],
    num_frames=124,
)

# In-memory media says which rates it carries, MiniMax-H3's own by default.
motion = load_video("motion_ref.mp4")
state = pipe(
    prompt="The subject walks toward camera, matching the reference video's shot rhythm",
    references=[MiniMaxH3Reference(image=subject), MiniMaxH3Reference(video=motion, fps=30.0)],
    num_frames=124,
)

encode_video(
    state.get("videos")[0],
    fps=24,
    output_path="minimax_h3_ref2va.mp4",
    audio=state.get("audio")[0],
    audio_sample_rate=state.get("sampling_rate"),
)
```

`num_frames` may be left out, but only when exactly one reference carries audio: the duration is then that soundtrack's, snapped up to the next `17 * n + 5` the video VAE can decode. A soundtrack whose *aligned* duration falls outside the 5 to 15 seconds MiniMax-H3 generates is rejected rather than silently stretched, so a clip just under 15 seconds — which rounds up to 362 frames, i.e. 15.083 seconds — has to name a shorter `num_frames` explicitly.

## MiniMaxH3ModularPipeline

[[autodoc]] MiniMaxH3ModularPipeline

## MiniMaxH3Blocks

[[autodoc]] MiniMaxH3Blocks

## MiniMaxH3Ref2VAModularPipeline

[[autodoc]] MiniMaxH3Ref2VAModularPipeline

## MiniMaxH3Ref2VABlocks

[[autodoc]] MiniMaxH3Ref2VABlocks

## MiniMaxH3Reference

[[autodoc]] modular_pipelines.minimax_h3.MiniMaxH3Reference
