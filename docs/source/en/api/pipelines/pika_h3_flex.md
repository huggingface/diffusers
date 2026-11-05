<!-- Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# Pika H3 Flex

<div class="flex flex-wrap space-x-1">
  <a href="https://huggingface.co/docs/diffusers/main/en/tutorials/using_peft_for_inference" target="_blank" rel="noopener">
    <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  </a>
</div>

[Pika H3 Flex](https://huggingface.co/Pika-Labs/Pika-H3-Flex) is Pika's post-training of [MiniMax-H3](./minimax_h3) into a six-step video-and-audio generator. It keeps MiniMax-H3's packed sequence, its Qwen3-VL conditioner and its two VAEs, and changes two things: the sampler, a fixed six-step schedule that re-noises the clean prediction at every step in place of the Euler update, and the weights, a pair of rank-128 LoRA adapters over the two MiniMax-H3 transformer partitions. Video and its synchronized stereo soundtrack come out of the same six forward passes.

The release is the adapters: the FL2VA adapter (`fl2va/adapter.safetensors`) drives the `transformer/` partition for text and endpoint-frame conditioning, the Ref2VA adapter (`ref2va/adapter.safetensors`) drives `transformer_ref/` for multimodal references. The base components are [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)'s, unchanged, and the repository carries a copy of them so that it loads on its own.

Pika H3 Flex is integrated as [Modular Diffusers](../../modular_diffusers/overview) blocks only, like MiniMax-H3: the blocks and their [`PikaH3FlexModularPipeline`] are the whole integration, and there is no `DiffusionPipeline` half.

## Loading

A pipeline is assembled in two steps: the components are loaded through a `modular_model_index.json` that names the MiniMax-H3 base components and the two scheduler configs, then the adapter of the selected workflow is loaded on top with [`~loaders.PikaH3FlexLoraLoaderMixin.load_lora_weights`]. The adapters are plain PEFT dumps of the diffusers module names and load at their trained scale (`alpha == rank`, 1.0).

| Workflow | Inputs | Transformer partition | Adapter |
|---|---|---|---|
| `t2va` | `prompt` | `transformer/` | `fl2va/adapter.safetensors` |
| `fl2va` | `prompt` and `image` and/or `last_image` | `transformer/` | `fl2va/adapter.safetensors` |
| `ref2va` | `prompt` and `references` | `transformer_ref/` | `ref2va/adapter.safetensors` |

```py
import torch
from diffusers import ModularPipeline

pipe = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", workflow="fl2va")
pipe.load_components(dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")
```

Passing the workflow prunes the blocks to that task, so only its own transformer partition is fetched, the adapter routes to the one partition present, and the pipeline accepts that task's requests alone — a `fl2va` pipeline needs a keyframe, a `t2va` one takes none. A pipeline that holds both partitions (loaded without a `workflow=`) needs to be told where the Ref2VA adapter goes: `pipe.load_lora_weights(..., weight_name="ref2va/adapter.safetensors", load_into_transformer_ref=True)`. The two VAEs are released in float32 and decode in float32, as the released package does; everything else runs in bfloat16.

> [!TIP]
> `pipe.doc` prints what the pipeline in front of you takes and returns — every input with its default, the components it expects and the outputs it produces.

### Reusing a loaded MiniMax-H3

The base components are MiniMax-H3's, so a MiniMax-H3 pipeline that is already loaded holds everything except the two schedulers and the adapters. This section assembles a Pika H3 Flex pipeline from those loaded components and keeps both pipelines ready side by side: one [`ComponentsManager`] holds each component once and offloads it for both, the Pika H3 Flex pipeline adds only its two schedulers, and the adapters decide which of the two models the shared transformers run.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline

dtype = {"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16}
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")

# The MiniMax-H3 pipeline already in use, with every workflow loaded. `collection` registers its components in the
# manager under that name, so they can be fetched by it later.
minimax = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", components_manager=manager, collection="minimax-h3")
minimax.load_components(dtype=dtype)

# The Pika H3 Flex pipeline, in its own collection. Only its own components are loaded: the two schedulers.
pika = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", components_manager=manager, collection="pika-h3-flex")
pika.load_components(names=["scheduler", "audio_scheduler"])

# The rest comes from the MiniMax-H3 collection, and `update_components` puts it in the Pika H3 Flex pipeline.
shared = ["text_encoder", "tokenizer", "processor", "vae", "audio_vae", "transformer", "transformer_ref"]
pika.update_components(**{name: manager.get_one(name=name, collection="minimax-h3") for name in shared})

# The adapters go into the shared transformers.
pika.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")
pika.load_lora_weights(
    "Pika-Labs/Pika-H3-Flex", weight_name="ref2va/adapter.safetensors", load_into_transformer_ref=True
)
```

`print(manager)` shows the result: every shared component is listed once and belongs to both collections, the Pika H3 Flex collection adds its two schedulers, and the two transformers carry an adapter.

```text
Models:
Name_ID               | Class                           | Size (GB) | Collection
transformer_1397...   | MiniMaxH3Transformer3DModel     | 63.02     | minimax-h3
                      |                                 |           | pika-h3-flex
transformer_ref_13... | MiniMaxH3Transformer3DModel     | 63.02     | minimax-h3
                      |                                 |           | pika-h3-flex
text_encoder_1397...  | Qwen3VLForConditionalGeneration | 62.13     | minimax-h3
                      |                                 |           | pika-h3-flex
vae_1397...           | AutoencoderKLMiniMaxH3          | 9.70      | minimax-h3
                      |                                 |           | pika-h3-flex
audio_vae_1397...     | AutoencoderKLMiniMaxH3Audio     | 0.56      | minimax-h3
                      |                                 |           | pika-h3-flex

Other Components:
ID                        | Class              | Collection
scheduler_1397...         | MiniMaxH3Scheduler | minimax-h3
audio_scheduler_1397...   | MiniMaxH3Scheduler | minimax-h3
scheduler_1397...         | MiniMaxH3Scheduler | pika-h3-flex
audio_scheduler_1397...   | MiniMaxH3Scheduler | pika-h3-flex
tokenizer_1397...         | Qwen2Tokenizer     | minimax-h3
                          |                    | pika-h3-flex
processor_1397...         | Qwen3VLProcessor   | minimax-h3
                          |                    | pika-h3-flex

Additional Component Info:
transformer_1397...:
  Adapters: ['default_0']
transformer_ref_13...:
  Adapters: ['default_0']
```

The adapters decide which model the shared transformers are. To go back to MiniMax-H3, call `pika.disable_lora()`, and that is all: the next `minimax(...)` request runs the base model. `pika.enable_lora()` turns Pika H3 Flex back on. Nothing is reloaded in between; the manager moves each component on and off the accelerator as either pipeline reaches it.

## What differs from MiniMax-H3

- **Six steps, no `num_inference_steps`.** Both [`MiniMaxH3Scheduler`] instances (`shift=12.0` for video, `shift=3.0` for audio) run with `stochastic_sampling` over the pipeline's `denoising_step_list` config, the released schedule `[1000, 750, 500, 250, 100, 40, 0]`: every step predicts the clean sample, and all but the last re-noise it with fresh noise to the next sigma. The step count is the adapter's contract and is not an input.
- **Fresh noise at every step.** A request draws from its `generator` in a fixed order: one draw per keyframe or visual reference for the conditioning noise, the video noise, the audio noise, and then the video and audio re-noising draws of each of the first five steps. Passing `latents` or `audio_latents` replaces the two initial draws and leaves the rest in place.
- **Keyframes are cropped, not stretched.** A `fl2va` keyframe is center-cropped to the canvas' aspect ratio at its own resolution and resized onto the canvas.
- **References follow the canvas' resolution tier.** Pika H3 Flex generates on a bucket table — a 480p tier and a 768p tier, eleven aspect ratios each — and a `ref2va` request conditions by the tier of its canvas: an image reference is encoded at a 1024 pixel short edge below the 768 tier and 2048 from it upwards, and a video reference is put on the tier's own canvas rule. Every video or standalone audio reference runs 2 to 15 seconds, and the video durations and the standalone audio durations each sum to at most 15 seconds.

## Generation

The released buckets are the canvases Pika H3 Flex was validated on. Pass one as `height` and `width`:

| | 21:9 | 16:9 | 3:2 | 4:3 | 5:4 | 1:1 | 4:5 | 3:4 | 2:3 | 9:16 | 9:21 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 480p | 960x416 | 832x480 | 704x480 | 640x480 | 608x480 | 480x480 | 480x608 | 480x640 | 480x704 | 480x832 | 416x960 |
| 768p | 1536x672 | 1344x768 | 1152x768 | 1024x768 | 960x768 | 768x768 | 768x960 | 768x1024 | 768x1152 | 768x1344 | 672x1536 |

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

# 61.7GB of transformer and 62.1GB of conditioner do not sit on one accelerator, so the components are
# registered in a manager that moves each one on and off as the blocks reach it.
manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")  # or "mps", "xpu", "cpu"

# Create one pipeline for all supported workflows, then load the `fl2va` workflow's components, which is
# enough to run both the fl2va and t2va requests below.
pipe = ModularPipeline.from_pretrained("Pika-Labs/Pika-H3-Flex", components_manager=manager)
pipe.load_components(
    workflow="fl2va", dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16}
)
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="fl2va/adapter.safetensors")

prompt = (
    "integrated_multimodal_description: A cinematic close shot of a red ceramic cup on a wooden table beside an "
    "open window. A soft breeze moves a linen curtain. The camera slowly pushes toward the cup as warm morning "
    "light crosses the table.\n"
    "overall_soundscape: Gentle wind through the window and faint birdsong outside. No speech.\n"
    "non_diegetic_music: None.\n"
)
outputs = ["videos", "audio", "sampling_rate"]

# Text to video + audio on the 480p 16:9 bucket, 5 seconds.
results = pipe(
    prompt=prompt, 
    height=480, 
    width=832, 
    num_frames=124, 
    generator=torch.Generator("cuda").manual_seed(42), 
    output=outputs
)

# First frame (and optionally `last_image`) to video + audio. The keyframe is center-cropped onto the canvas and
# held as the first frame, so the prompt describes the keyframe and what happens from it.
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
i2va_prompt = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
    "integrated_multimodal_description: [Shot 1] Begin with the supplied image of a small toy astronaut figurine "
    "standing on a grey rocky moon surface, holding a white egg-shaped object. Preserve the figurine, its orange "
    "visor and the rocks. The astronaut slowly turns its helmet toward the camera and lifts the egg a little; fine "
    "dust drifts up around its boots. The camera slowly pushes in in one continuous shot. By the end the astronaut "
    "stands still, egg held up.\n\n"
    "overall_soundscape: Soft low wind and faint crunch of dust under the boots. No speech.\n\n"
    "non_diegetic_music: N/A\n"
)
results = pipe(
    prompt=i2va_prompt, 
    image=image, 
    height=480, 
    width=832, 
    num_frames=124,
    generator=torch.Generator("cuda").manual_seed(42), 
    output=outputs,
)

encode_video(
    results["videos"][0],
    fps=24,
    output_path="pika_h3_flex_fl2va.mp4",
    audio=results["audio"][0],
    audio_sample_rate=results["sampling_rate"],
)
```

The `ref2va` workflow takes an ordered list of references — up to 9 images, 3 videos and 3 audio clips, 12 in total — the way MiniMax-H3 does, with one reference class per modality. It is also the request that needs the most memory: the references add their own rows to the sequence, and at 768p the transformer alone fills an 80 GB card. The example below therefore streams the transformer's blocks from CPU memory one at a time, while the manager moves the other components on and off the device as before. It runs a 768p reference request on one 80 GB card; without the streaming, keep to 480p.

```py
import torch
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines.pika_h3_flex import (
    PikaH3FlexAudioReference,
    PikaH3FlexImageReference,
    PikaH3FlexVideoReference,
)

manager = ComponentsManager()
manager.enable_auto_cpu_offload(device="cuda")  # or "mps", "xpu", "cpu"

pipe = ModularPipeline.from_pretrained(
    "Pika-Labs/Pika-H3-Flex", workflow="ref2va", components_manager=manager
)
pipe.load_components(dtype={"vae": torch.float32, "audio_vae": torch.float32, "default": torch.bfloat16})
pipe.load_lora_weights("Pika-Labs/Pika-H3-Flex", weight_name="ref2va/adapter.safetensors")

# Keep the transformer in CPU memory and bring in one block at a time while it runs.
pipe.transformer_ref.enable_group_offload(
    onload_device=torch.device("cuda"),
    offload_device=torch.device("cpu"),
    offload_type="block_level",
    num_blocks_per_group=1,
    use_stream=True,
)

results = pipe(
    prompt="subject_definitions: <Subject 1> is the woman in <Picture 1>. summary: [reference generation] <Subject 1> waves at the camera.",
    references=[
        PikaH3FlexImageReference.from_file("subject.png"),
        PikaH3FlexVideoReference.from_file("motion_ref.mp4"),  # frames, their `fps` and the soundtrack
        PikaH3FlexAudioReference.from_file("voice.wav"),  # waveform and its `sample_rate`
    ],
    height=768,
    width=1344,
    num_frames=124,
    generator=torch.Generator("cuda").manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)
```

## PikaH3FlexModularPipeline

[[autodoc]] PikaH3FlexModularPipeline

## PikaH3FlexBlocks

[[autodoc]] PikaH3FlexBlocks

## PikaH3FlexImageReference

[[autodoc]] modular_pipelines.pika_h3_flex.PikaH3FlexImageReference
    - from_file

## PikaH3FlexVideoReference

[[autodoc]] modular_pipelines.pika_h3_flex.PikaH3FlexVideoReference
    - from_file

## PikaH3FlexAudioReference

[[autodoc]] modular_pipelines.pika_h3_flex.PikaH3FlexAudioReference
    - from_file
