<!--Copyright 2026 The Kandinsky Team and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Kandinsky S2V 1.0

Kandinsky S2V 1.0 generates an avatar video from a reference frame and a speech
track. The person in the frame is animated to match the speech — the whole figure,
not only the head. The speech is a condition, not an output: the model does not
generate audio, and the soundtrack of the result is the track you pass in.

Both conditions go through [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B):
its hidden states encode the prompt and its audio tower encodes the speech, which
enters the transformer through cross-attention adapters aligned frame by frame, so
each latent frame attends only to the speech it covers. A clip is generated in
chunks of 5 s (125 frames at 25 fps), each carrying FramePack memory of the
previous ones, so longer tracks stay consistent across chunk boundaries.

Kandinsky S2V 1.0 is integrated as a [modular pipeline](../../modular_diffusers/overview)
only; there is no standard `DiffusionPipeline` class for it.

## Checkpoints

| Checkpoint | Transformer | Blocks | Scheduler | Steps | Guidance |
|---|---|---|---|---|---|
| [Lite][lite] | 2B | [`KandinskyS2VAutoBlocks`] | [`FlowMatchEulerDiscreteScheduler`] | 50 | 3.0 |
| [Pro][pro] | 21B | [`KandinskyS2VAutoBlocks`] | [`FlowMatchEulerDiscreteScheduler`] | 50 | 3.0 |
| [Lite distilled][lite-distill] | 2B | [`KandinskyS2VDistilledAutoBlocks`] | [`PiflowScheduler`] | 16 | off |
| [Pro distilled][pro-distill] | 21B | [`KandinskyS2VDistilledAutoBlocks`] | [`PiflowScheduler`] | 16 | off |

The distilled checkpoints configure `PiflowScheduler` with `shift=10.0` and `n_grid=5`, not its defaults.

[lite]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-Diffusers
[pro]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers
[lite-distill]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers
[pro-distill]: https://huggingface.co/kandinskylab/Kandinsky-S2V-1.0-Pro-distill-Diffusers

`ModularPipeline.from_pretrained` picks the blocks from the checkpoint. The steps
default to the values above. Classifier-free guidance is the `guider` component,
not a call argument: the base checkpoints ship it at 3.0, and the distilled ones
at 1.0, because guidance is already part of the student — they reject any other.

## Usage

```py
import torch
from diffusers import ModularPipeline
from diffusers.utils import export_to_video

pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Lite-distill-Diffusers")
pipe.load_components(dtype=torch.bfloat16)
pipe.to("cuda")

video = pipe(image="reference.png", audio="speech.wav", output="videos")[0]
export_to_video(video, "avatar.mp4", fps=25)
```

`export_to_video` writes the frames only; add the speech track with any muxer, for
example `ffmpeg -i avatar.mp4 -i speech.wav -c:v copy -c:a aac -shortest out.mp4`.

The Pro checkpoints do not fit an 80 GB GPU next to the 7B speech and text encoder
when every component is on the device. Keep one model on the GPU at a time, which
brings the peak to about 50 GiB:

```py
from diffusers import ComponentsManager

manager = ComponentsManager()
pipe = ModularPipeline.from_pretrained("kandinskylab/Kandinsky-S2V-1.0-Pro-Diffusers", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
manager.enable_auto_cpu_offload(device="cuda", offload_strategy=lambda hooks, **_: hooks)
```

The default strategy is not enough here: it keeps the transformer beside the speech
encoder because both fit, and the VAE decode that follows needs the room they hold.
Enable the offload after `load_components`, so the strategy applies to every component.

## Notes

- The number of chunks follows the length of the speech track unless `num_chunks`
  is passed. The video is trimmed to the track either way, so chunks past its end
  are computed and dropped, with a warning.
- With guidance, the unconditional pass swaps in the negative prompt *and* silences
  the speech, so the guidance is joint over text and speech.

## KandinskyS2VModularPipeline

[[autodoc]] KandinskyS2VModularPipeline

## KandinskyS2VAutoBlocks

[[autodoc]] KandinskyS2VAutoBlocks

## KandinskyS2VDistilledModularPipeline

[[autodoc]] KandinskyS2VDistilledModularPipeline

## KandinskyS2VDistilledAutoBlocks

[[autodoc]] KandinskyS2VDistilledAutoBlocks
