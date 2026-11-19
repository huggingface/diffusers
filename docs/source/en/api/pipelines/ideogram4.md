<!--Copyright 2026 Ideogram AI and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Ideogram 4

Ideogram 4 is a flow-matching text-to-image model that uses a multimodal text encoder and an asymmetric
classifier-free guidance scheme: a dedicated `unconditional_transformer` produces the negative branch with zeroed text
features, while the main `transformer` consumes the full packed text + image sequence.

The pipeline defaults are the recommended settings for best quality, so a plain `pipe(prompt)` call produces
best-quality results out of the box: 48 flow-matching steps on a logit-normal schedule (`mu=0.0`, `std=1.5`) with
classifier-free guidance held at 7.0 for the main steps and dropped to 3.0 for the final 3 "polish" steps.

Key inference-time knobs are exposed via the pipeline call:

- `num_inference_steps`, `mu`, and `std` control the resolution-aware logit-normal flow-matching schedule.
- `guidance_scale` (or a full per-step `guidance_schedule`) blends the conditional and unconditional velocities.

## Text-to-image

```python
import torch
from diffusers import Ideogram4Pipeline

pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-4-nf4-diffusers", dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

prompt = "A photo of a cat holding a sign that says hello world"
# The defaults are the recommended settings for best quality.
image = pipe(prompt, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("ideogram4.png")
```

## Prompt upsampling

Ideogram 4 is trained on a structured JSON caption rather than a free-form prompt, so a short prompt is best
expanded into that native schema before generation. There are two ways to produce the caption.

### Remote (Ideogram API)

For the best results, expand the prompt with Ideogram's hosted magic-prompt API and pass the returned caption
straight to the pipeline (get a key at [developer.ideogram.ai](https://developer.ideogram.ai/)):

```python
import json
import requests
import torch
from diffusers import Ideogram4Pipeline

pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-4-nf4-diffusers", dtype=torch.bfloat16)
pipe.to("cuda")  # or "mps", "xpu", "cpu"

# Expand the prompt into a structured JSON caption with Ideogram's hosted magic-prompt API.
response = requests.post(
    "https://api.ideogram.ai/v1/ideogram-v4/magic-prompt",
    headers={"Api-Key": "your_ideogram_api_key"},
    json={"text_prompt": "A photo of a cat holding a sign that says hello world", "aspect_ratio": "1x1"},
).json()
caption = json.dumps(response["json_prompt"])

# The caption is already upsampled, so pass it directly (no prompt_upsampling).
image = pipe(caption, height=1024, width=1024, generator=torch.Generator("cuda").manual_seed(0)).images[0]
image.save("ideogram4_upsampled.png")
```

### Local (on-device)

Use the [modular prompt enhancer](https://huggingface.co/diffusers/ideogram4-prompt-enhancer) to expand prompts on your
device. It reuses the image pipeline's `text_encoder` and `tokenizer`, so only the LM head is loaded separately.
Install `outlines` for schema-constrained captions; the nf4 image checkpoint also needs `bitsandbytes`.

<Tip warning={true}>

The built-in `Ideogram4PromptEnhancerHead`, `upsample_prompt()` method, and `prompt_upsampling` argument are deprecated.
Use the hosted modular block and pass its output as `prompt` to the image pipeline. Omit `prompt_upsampling`; built-in
upsampling is disabled by default.

</Tip>

The block and head load code from the Hub and execute it locally. Both loading calls require `trust_remote_code=True`.

```python
import torch
from diffusers import Ideogram4Pipeline, ModularPipeline

pipe = Ideogram4Pipeline.from_pretrained("ideogram-ai/ideogram-4-nf4-diffusers", dtype=torch.bfloat16)
pipe.to("cuda") # or "mps", "xpu", "cpu"

enhancer = ModularPipeline.from_pretrained(
    "diffusers/ideogram4-prompt-enhancer", trust_remote_code=True
)
enhancer.update_components(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
enhancer.load_components(
    names="prompt_enhancer_head", dtype=torch.bfloat16, trust_remote_code=True
)
enhancer.to("cuda") # or "mps", "xpu", "cpu"

prompt = "A photo of a cat holding a sign that says hello world"
prompt = enhancer(
    prompt=prompt,
    height=1024,
    width=1024,
    generator=torch.Generator("cuda").manual_seed(0),
    output="prompt",
)[0]
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]
image.save("ideogram4_upsampled.png")
```

The enhancer returns a list of structured JSON prompt strings, one per input prompt. The example selects the first
result with `[0]` and passes it through the image pipeline's existing `prompt` argument. For a batch, pass the whole
list as `prompt`. Use the same `height` and `width` for enhancement and image generation so the expanded prompt describes
the intended aspect ratio. Keep the shared encoder and the head on the same device with compatible dtypes.

You can also share the encoder and tokenizer from an `Ideogram4ModularPipeline` and pass the expanded prompts to it as
`prompt`, omitting `prompt_upsampling`.

## Ideogram4Pipeline

[[autodoc]] Ideogram4Pipeline
	- all
	- __call__

## Ideogram4PromptEnhancerHead

Deprecated. Use the [modular prompt enhancer](https://huggingface.co/diffusers/ideogram4-prompt-enhancer) instead.

[[autodoc]] Ideogram4PromptEnhancerHead

## Ideogram4PipelineOutput

[[autodoc]] pipelines.ideogram4.pipeline_output.Ideogram4PipelineOutput
