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

## Checkpoint layout

MiniMax-H3 was released as two checkpoint partitions that share every component except the transformer, so the diffusers conversion puts both in **one repository**:

| Subfolder | Pipeline | Tasks |
|---|---|---|
| `transformer/` | [`MiniMaxH3Pipeline`] | `t2va` (text only) and `fl2va` (first and/or last keyframe) |
| `transformer_ref/` | [`MiniMaxH3Ref2VAPipeline`] | `ref2va` (an ordered mix of image, video and audio references) |

Each pipeline declares only its own transformer, so `from_pretrained` on the same repository id downloads and loads only the partition that pipeline needs. Everything else, the video VAE, the audio VAE, the Qwen3-VL conditioner, its tokenizer and processor, and the two schedulers, is shared.

The conditioner is a `Qwen3VLForConditionalGeneration`, and MiniMax-H3 reads the *unnormalized* hidden state after its 50th decoder layer rather than the last one, so the full released checkpoint is used with its language-model head unused.

## Two schedulers

Video and audio latents step down two different schedules inside a single transformer call per step, which is why both pipelines register two [`MiniMaxH3Scheduler`] instances: `scheduler` for the video latents (`shift=12.0` in the released checkpoints) and `audio_scheduler` for the audio latents (`shift=3.0`).

The checkpoint is guidance-distilled: there is no `guidance_scale` and no negative prompt, and every step runs exactly one forward pass.

## Generation constraints

- **24 fps, 5 to 15 seconds.** `num_frames` is snapped up to the next `17 * n + 5` the video VAE can decode, and the resulting duration has to stay in that window.
- **A 768 pixel short edge.** `height` and `width` default to MiniMax-H3's own canvas for the aspect ratio of the first keyframe (or 16:9 without one) and must be multiples of 32.
- **One generator, three draws.** A request draws the keyframe or reference conditioning noise first, then the video noise, then the audio noise, all from the `generator` it is passed, so two runs from the same generator state return the same video and soundtrack. Passing `latents` or `audio_latents` replaces the corresponding draw.
- **`num_inference_steps` counts sigma grid points**, the terminal `0` included, so it drives one model evaluation less.

## Text and keyframes

[`MiniMaxH3Pipeline`] covers text-to-video-and-audio and keyframe conditioning. A keyframe can be the frame the video starts from (`image`), the frame it ends on (`last_image`), or both.

```py
import torch
from diffusers import MiniMaxH3Pipeline
from diffusers.utils import load_image
from diffusers.utils.export_utils import encode_video

pipe = MiniMaxH3Pipeline.from_pretrained("MiniMaxAI/MiniMax-H3", dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = "A red fox trotting through a snowy pine forest, snow crunching underfoot"

# Text to video + audio.
output = pipe(prompt=prompt, generator=torch.Generator().manual_seed(42))

# First frame (and optionally last frame) to video + audio. The canvas follows the first keyframe.
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)
output = pipe(prompt=prompt, image=image, generator=torch.Generator().manual_seed(42))

encode_video(
    output.frames[0],
    fps=24,
    output_path="minimax_h3_fl2va.mp4",
    audio=output.audio[0],
    audio_sample_rate=pipe.audio_sampling_rate,
)
```

## Omni-references

[`MiniMaxH3Ref2VAPipeline`] conditions on an ordered list of references: up to 9 images, 3 videos and 3 audio clips, 12 in total. The order is semantic. It labels the references in the prompt presentation (`"<Picture 1>"`, `"<Audio 1>"`, `"<Video 1>"`) and it advances the shared audio/video rotary clock, so reordering the same references is a different request.

Unlike the keyframes above, references do not bind the generated geometry: they are encoded at their own resolution and the target canvas defaults to MiniMax-H3's 16:9. The duration may be left to a reference soundtrack, but only when exactly one reference carries audio.

Every reference is a [`~pipelines.minimax_h3.MiniMaxH3Reference`], and it takes a path or a URL as well as in-memory media: a path is decoded when the reference is built, with [`~utils.load_image`] for an image and [PyAV](https://github.com/PyAV-Org/PyAV) for a video or an audio file, so the pipeline itself never opens a media file. Decoding brings the rates along, which is what the model needs: a video reference reads its frame rate off the container and adopts the container's soundtrack when it has one, and an audio reference its sample rate.

In-memory media declares the rates it carries instead, defaulting to MiniMax-H3's own: `fps=24.0` and the audio VAE's sample rate, so only self-generated data at another rate has to say so. An explicit `fps` or `sample_rate` also wins over a container whose metadata is wrong. Frames are resampled onto MiniMax-H3's own 24 fps by dropping and duplicating whole frames, and a waveform onto the audio VAE's sample rate. A video reference conditions on its motion **and**, when it carries an `audio` waveform, on that soundtrack, which is then packed as this reference's own.

A request that carries a single reference may name its medium on the call itself, as `image=`, `video=` or `audio=`, which is the same request as a one-item `references` list. The two ways are mutually exclusive.

```py
import torch
from diffusers import MiniMaxH3Ref2VAPipeline
from diffusers.pipelines.minimax_h3 import MiniMaxH3Reference
from diffusers.utils import load_image, load_video
from diffusers.utils.export_utils import encode_video

pipe = MiniMaxH3Ref2VAPipeline.from_pretrained("MiniMaxAI/MiniMax-H3", dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

subject = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
)

# A single reference may name its medium on the call itself.
output = pipe(
    prompt="The subject walks toward camera, turning slightly, natural arm swing",
    image=subject,
    generator=torch.Generator().manual_seed(42),
)

# A reference decodes a path or a URL as it is built, rates included: the video brings its own frame rate and its
# soundtrack, the audio clip its sample rate. An audio reference sets the duration on its own.
output = pipe(
    prompt="The character speaks in time with the reference recording, natural lip movement",
    references=[
        MiniMaxH3Reference(image=subject),
        MiniMaxH3Reference(
            video="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        ),
        MiniMaxH3Reference(audio="voice.wav"),
    ],
)

# In-memory media says which rates it carries, MiniMax-H3's own by default.
motion = load_video("motion_ref.mp4")
output = pipe(
    prompt="The subject walks toward camera, matching the reference video's shot rhythm",
    references=[MiniMaxH3Reference(image=subject), MiniMaxH3Reference(video=motion, fps=30.0)],
)

encode_video(
    output.frames[0],
    fps=24,
    output_path="minimax_h3_ref2va.mp4",
    audio=output.audio[0],
    audio_sample_rate=pipe.audio_sampling_rate,
)
```

## MiniMaxH3Pipeline

[[autodoc]] MiniMaxH3Pipeline
  - all
  - __call__

## MiniMaxH3Ref2VAPipeline

[[autodoc]] MiniMaxH3Ref2VAPipeline
  - all
  - __call__

## MiniMaxH3Reference

[[autodoc]] pipelines.minimax_h3.MiniMaxH3Reference

## MiniMaxH3PipelineOutput

[[autodoc]] pipelines.minimax_h3.pipeline_output.MiniMaxH3PipelineOutput
