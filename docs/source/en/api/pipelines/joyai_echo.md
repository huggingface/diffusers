# JoyAI-Echo

JoyAI-Echo is a text-to-audio-video generation pipeline for multi-shot video stories. It builds on the LTX-2 component
layout and adds the JoyAI-Echo few-step DMD denoising schedule plus a paired audio-video memory bank for cross-shot
consistency.

The pipeline accepts one prompt per shot. When a list of prompts is passed, generated video and audio latents from
earlier shots are kept as memory tokens for later shots.

```py
import torch
from diffusers import JoyAIEchoPipeline
from diffusers.utils import encode_video

pipe = JoyAIEchoPipeline.from_pretrained("path/to/converted-joyai-echo", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

output = pipe(
    [
        "A cinematic opening shot of the protagonist entering a quiet train station.",
        "The same protagonist speaks softly while the camera follows through the platform.",
    ],
    height=736,
    width=1280,
    num_frames=241,
    frame_rate=25.0,
)

for i, (frames, audio) in enumerate(zip(output.frames, output.audio)):
    encode_video(frames[0], fps=25, audio=audio[0].float().cpu(), output_path=f"shot_{i:03d}.mp4")
```

## JoyAIEchoPipeline

[[autodoc]] JoyAIEchoPipeline

## JoyAIEchoPipelineOutput

[[autodoc]] pipelines.joyai_echo.JoyAIEchoPipelineOutput

## JoyAIEchoShotOutput

[[autodoc]] pipelines.joyai_echo.JoyAIEchoShotOutput
