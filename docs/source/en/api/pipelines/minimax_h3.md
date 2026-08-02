<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# MiniMax-H3

<!-- Remove this note once the pull request is merged. -->
> [!TIP]
> MiniMax-H3 is not part of a diffusers release yet. Install diffusers from the pull request to use it:
> `pip install git+https://github.com/huggingface/diffusers.git@refs/pull/14355/head`


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
pipe.load_components(dtype=torch.bfloat16)
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
pipe.load_components(dtype=torch.bfloat16)

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

With two cards nothing has to be offloaded: the conditioner takes the second card through a `device_map` and the denoiser keeps the first.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from transformers import Qwen3VLForConditionalGeneration

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.update_components(
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        "MiniMaxAI/MiniMax-H3", subfolder="text_encoder", dtype=torch.bfloat16, device_map={"": "cuda:1"}
    ),
)
pipe.load_components(dtype=torch.bfloat16)
pipe.transformer.to("cuda:0")
pipe.vae.to("cuda:0")
pipe.audio_vae.to("cuda:0")
```

Two 80 GB cards run full bfloat16 this way with nothing streaming; two 48 GB cards do the same with the int8 loading above on both components.

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
