<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MiniMax-H3

MiniMax-H3 generates video and its soundtrack together. A single transformer denoises one packed sequence containing the text conditioning, conditioning media, and target video and audio latents. There is no separate vocoder and no audio post-hoc pass: video and audio come out of the same denoising loop.

You can find the original MiniMax-H3 checkpoints under the [MiniMaxAI](https://huggingface.co/MiniMaxAI) organization.

MiniMax-H3 is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only, the way [Anima](./anima) is: the blocks and their [`MiniMaxH3ModularPipeline`] are the whole integration, and there is no `DiffusionPipeline` half.

## Checkpoint layout

MiniMax-H3 was released as two checkpoint partitions that share every component except the transformer, so the diffusers conversion puts both in **one repository**:

| Subfolder | Workflows |
|---|---|
| `transformer/` | `t2va` (text only), `fl2va` (first and/or last keyframe) |
| `transformer_ref/` | `ref2va` (an ordered mix of image, video and audio references) |

Everything but the transformer, i.e. the video VAE, the audio VAE, the Qwen3-VL conditioner, its tokenizer and processor, and the two schedulers, is shared and stored once.

The conditioner is a `Qwen3VLForConditionalGeneration`, and MiniMax-H3 reads the *unnormalized* hidden state after its 50th decoder layer rather than the last one, so the full released checkpoint is used with its language-model head unused.

All three tasks are workflows of the one [`MiniMaxH3Blocks`], and the repository carries one `modular_model_index.json` naming every component with its own loading spec. To serve a single task, pass the workflow to `from_pretrained`: it keeps only that workflow's blocks, so the pipeline's signature (`pipe.doc`) documents exactly that task's inputs, only that task's components are declared, and `load_components` fetches exactly their subfolders — a `t2va` / `fl2va` pipeline never touches `transformer_ref/`, a `ref2va` one never touches `transformer/`.

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="ref2va")
pipe.load_components(dtype=torch.bfloat16)
```

> [!TIP]
> `pipe.doc` prints what the pipeline in front of you takes and returns — every input with its default, the components it expects and the outputs it produces. Pruned to one workflow it describes exactly that task, which is the quickest way to see what a request needs before making one.

To keep every workflow available on one pipeline instead, leave the `workflow` argument out: the pipeline then picks the workflow per call from the inputs it is passed.

```py
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
```

The *loading* can still go one workflow at a time. This one call fetches `transformer/` and every shared component, which serves both `t2va` and `fl2va`:

```py
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
```

A plain `load_components()` with no `workflow=` pulls **both** 61.7GB transformer partitions, which is what lets one pipeline serve all three workflows without another loading call. Pair it with a [`ComponentsManager`] and auto offloading: the weights live in host RAM and the manager moves onto the accelerator just what each step needs, so when the `ref2va` denoiser wants the device the strategy offloads whatever frees enough room. See [Memory](#memory) for the recipes.

## Two schedulers

Video and audio latents step down two different schedules inside a single transformer call per step, which is why the blocks expect two [`MiniMaxH3Scheduler`] instances: `scheduler` for the video latents (`shift=12.0` in the released checkpoints) and `audio_scheduler` for the audio latents (`shift=3.0`).

Both transformer partitions are guidance-distilled, so this holds for every workflow: guidance is baked into the weights, there is no guider, no `negative_prompt` and no `guidance_scale`, and every step runs exactly one forward pass.

## Generation constraints

- **24 fps, 5 to 15 seconds.** `num_frames` is snapped up to the next `17 * n + 5` the video VAE can decode, and the resulting duration has to stay in that window.
- **A 768 pixel short edge.** `height` and `width` default to MiniMax-H3's own canvas for the aspect ratio of the first keyframe (or 16:9 without one) and must be multiples of 32.
- **One generator, three draws.** A request draws the keyframe or reference conditioning noise first, then the video noise, then the audio noise, all from the `generator` it is passed, so two runs from the same generator state return the same video and soundtrack. Passing `latents` or `audio_latents` replaces the corresponding draw.
- **`num_inference_steps` counts sigma grid points**, the terminal `0` included, so it drives one model evaluation less.

## Memory

The transformer alone is 61.7 GB in bfloat16 and the Qwen3-VL conditioner is another 62.1 GB, so the loading recipe depends on the hardware. Smaller canvases are the biggest speed lever on every setup: `height` and `width` only have to be multiples of 32, and 960x544 runs about 2.3x faster per step than the trained 1344x768.

On one 80 GB card, register the components in a [`ComponentsManager`] and let it move them on and off the accelerator:

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="12GB")
pipe.transformer.set_attention_backend("_flash_3_hub")  # Hopper, roughly 3x faster; kernels fetched from the Hub
```

On a consumer card (24 to 32 GB), quantize the two large components to int8 as they load and stream the transformer's blocks from CPU RAM. Everything below uses supported loaders only, no patches, and works straight from the bfloat16 checkpoint:

```py
import torch
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
from diffusers.hooks import apply_group_offloading
from transformers import Qwen3VLForConditionalGeneration
from transformers import TorchAoConfig as TransformersTorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
pipe.update_components(
    transformer=MiniMaxH3Transformer3DModel.from_pretrained(
        "MiniMaxAI/MiniMax-H3", subfolder="transformer", dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                "token_refiner", "norm_out", "proj_out", "audio_proj_out",
            ],
        ),
        low_cpu_mem_usage=False,
    ),
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        "MiniMaxAI/MiniMax-H3", subfolder="text_encoder", dtype=torch.bfloat16,
        quantization_config=TransformersTorchAoConfig(
            Int8WeightOnlyConfig(version=2),
            modules_to_not_convert=["model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"],
        ),
    ),
)
pipe.load_components(workflow="t2va", dtype=torch.bfloat16)

# version=2 int8 tensors are pinnable, which streamed offload needs, and freezing removes the one autograd
# path the quantized tensors cannot serve.
pipe.transformer.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True)
pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
pipe.vae.to("cuda")
pipe.audio_vae.to("cuda")
```

On 12 to 16 GB the same recipe works with the video VAE group offloaded too (`offload_type="leaf_level"`, no stream) and a small canvas such as 960x544. Expect the weights to live in host RAM: around 75 GB of it at int8.

With two cards nothing has to be offloaded: split the pipeline in two and put each half on its own device.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

workflow = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3").blocks.get_workflow("t2va")

text_manager = ComponentsManager()
text_manager.enable_auto_cpu_offload(device="cuda:1")
conditioner = workflow.sub_blocks.pop("text_encoder").init_pipeline(
    "MiniMaxAI/MiniMax-H3", components_manager=text_manager
)
conditioner.load_components(dtype=torch.bfloat16)

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda:0")
rest = workflow.init_pipeline("MiniMaxAI/MiniMax-H3", components_manager=manager)
rest.load_components(dtype=torch.bfloat16)

prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
state = conditioner(prompt=prompt)
results = rest(
    state=state,
    num_frames=124,
    generator=torch.Generator().manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)
```

Two 80 GB cards run full bfloat16 this way: each half fits on its own card, so nothing is evicted back to host memory once it is resident. Two 48 GB cards do the same with the int8 loading above on both components.

## Text and keyframes

[`MiniMaxH3Blocks`] covers text-to-video-and-audio and keyframe conditioning. A keyframe can be the frame the video starts from (`image`), the frame it ends on (`last_image`), or both.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

# 61.7GB of transformer and 62.1GB of conditioner do not sit on one accelerator, so the components are
# registered in a manager that moves each one on and off as the blocks reach it. See [Memory](#memory).
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)

prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"
# `output=` returns exactly the named outputs instead of the whole pipeline state.
outputs = ["videos", "audio", "sampling_rate"]

# Text to video + audio.
results = pipe(prompt=prompt, num_frames=124, generator=torch.Generator().manual_seed(42), output=outputs)

# First frame (and optionally last frame) to video + audio. The canvas follows the first keyframe.
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
results = pipe(prompt=prompt, image=image, num_frames=124, generator=torch.Generator().manual_seed(42), output=outputs)

encode_video(
    results["videos"][0],
    fps=24,
    output_path="minimax_h3_fl2va.mp4",
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
```

Video and audio are generated jointly and come out of the call as separate outputs, `videos` and `audio`, next to the `sampling_rate` the soundtrack carries; muxing them into one file is left to the caller, e.g. with [`~utils.export_utils.encode_video`].

## Omni-references

The `ref2va` workflow conditions on an ordered list of references: up to 9 images, 3 videos and 3 audio clips, 12 in total. The order is semantic. It labels the references in the prompt presentation (`"<Picture 1>"`, `"<Audio 1>"`, `"<Video 1>"`) and it advances the shared audio/video rotary clock, so reordering the same references is a different request.

Unlike the keyframes above, references do not bind the generated geometry: they are encoded at their own resolution and the target canvas defaults to MiniMax-H3's 16:9.

There is one reference class per modality, each holding in-memory media and the rate that media carries:

| reference | media | rate it declares |
| --- | --- | --- |
| [`~modular_pipelines.minimax_h3.MiniMaxH3ImageReference`] | `image` | — |
| [`~modular_pipelines.minimax_h3.MiniMaxH3VideoReference`] | `frames`, and `audio` for its own soundtrack | `fps`, defaulting to MiniMax-H3's 24.0, and `sample_rate` |
| [`~modular_pipelines.minimax_h3.MiniMaxH3AudioReference`] | `audio` | `sample_rate`, defaulting to the audio VAE's own |

The rates are what everything is resampled from: frames onto MiniMax-H3's own 24 fps by dropping and duplicating whole frames, a waveform onto the audio VAE's sample rate. Media at MiniMax-H3's own rates flows through untouched, so only data produced at another rate has to say so.

A reference is built one of two ways: decoded from a media file with the class's `from_file` classmethod, or constructed directly from media the request already holds in memory — which is how a previous generation feeds back in (see [A generation as a reference](#a-generation-as-a-reference)).

The blocks never open a media file — decoding a path is the caller's job, as it is everywhere else in the library. Each reference class does it through its `from_file` classmethod, which takes a path or a URL (video and audio through [PyAV](https://github.com/PyAV-Org/PyAV)) and returns a reference carrying the rates the container reports — a video brings its frame rate and its soundtrack along. Prefer `from_file` over [`~utils.load_video`], which drops the frame rate: a reference built from frames whose real rate was lost is conditioned on at the wrong speed, and nothing raises.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from diffusers.utils import load_video
from diffusers.utils.export_utils import encode_video

# `ref2va` is a workflow of the one MiniMax-H3 pipeline; selecting it loads only the `transformer_ref/`
# checkpoint partition, and the manager moves each component on and off the accelerator in turn.
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", workflow="ref2va", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

subject = MiniMaxH3ImageReference.from_file(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

# Decoding a file brings its rates along: the video its own frame rate and its soundtrack, the clip its sample rate.
results = pipe(
    prompt="The character speaks in time with the reference recording, natural lip movement",
    references=[
        subject,
        MiniMaxH3VideoReference.from_file(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        ),
        MiniMaxH3AudioReference.from_file("voice.wav"),
    ],
    num_frames=124,
    output=["videos", "audio", "sampling_rate"],
)

# Frames a request already holds declare the rate they carry — `load_video` does not preserve it.
motion = load_video("motion_ref.mp4")
results = pipe(
    prompt="The subject walks toward camera, matching the reference video's shot rhythm",
    references=[subject, MiniMaxH3VideoReference(frames=motion, fps=30.0)],
    num_frames=124,
    output=["videos", "audio", "sampling_rate"],
)

encode_video(
    results["videos"][0],
    fps=24,
    output_path="minimax_h3_ref2va.mp4",
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
```

`num_frames` is required for `ref2va`. To generate a video exactly as long as a reference soundtrack, compute it from the clip — `round(samples / sample_rate * 24)` — and it is snapped up to the next `17 * n + 5` the video VAE can decode; the resulting duration must stay between the 5 and 15 seconds MiniMax-H3 generates.

### A generation as a reference

Because the workflows share one pipeline, a `t2va` generation can be fed straight back as a `ref2va` reference — and the in-memory constructor is built for exactly this hand-off. The generated media is already at MiniMax-H3's own rates (frames at 24 fps, the soundtrack at the audio VAE's sample rate), so the reference needs no rate arguments and nothing is re-encoded through a lossy container on the way:

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3VideoReference

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

# The full pipeline holds every workflow and picks one per call from the inputs. Loading without a
# `workflow=` brings both transformer partitions in one call, so the `ref2va` request that follows the
# `t2va` generation needs no further loading.
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)

results = pipe(
    prompt="An astronaut hiking through the mountains, humming a tune",
    num_frames=124,
    output=["videos", "audio", "sampling_rate"],
)

reference = MiniMaxH3VideoReference(
    frames=results["videos"][0],
    audio=results["audio"][0],
    sample_rate=results["sampling_rate"],
)
results = pipe(
    prompt="The same astronaut now walks along a beach at sunset, humming the same tune",
    references=[reference],
    num_frames=124,
    output=["videos", "audio", "sampling_rate"],
)
```

## MiniMaxH3ModularPipeline

[[autodoc]] MiniMaxH3ModularPipeline

## MiniMaxH3Blocks

[[autodoc]] MiniMaxH3Blocks

## MiniMaxH3ImageReference

[[autodoc]] modular_pipelines.minimax_h3.MiniMaxH3ImageReference
    - from_file

## MiniMaxH3VideoReference

[[autodoc]] modular_pipelines.minimax_h3.MiniMaxH3VideoReference
    - from_file

## MiniMaxH3AudioReference

[[autodoc]] modular_pipelines.minimax_h3.MiniMaxH3AudioReference
    - from_file
