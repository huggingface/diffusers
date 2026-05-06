<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# UniLLaDA

[UniLLaDA](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) is a unified discrete diffusion language model that supports
text-to-image generation, image understanding, and image editing through block-wise iterative refinement. It extends
the [LLaDA2](./llada2) framework with multimodal capabilities.

## Usage

UniLLaDA supports three modes:
- **Text-to-Image**: Generate images from text prompts.
- **Image Understanding**: Answer questions about images.
- **Image Editing**: Edit images based on text instructions.

### Text-to-Image

```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import BlockRefinementScheduler, UniLLaDaPipeline

model_id = "inclusionAI/LLaDA2.0-Uni"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
scheduler = BlockRefinementScheduler()

pipe = UniLLaDaPipeline(transformer=model, tokenizer=tokenizer, scheduler=scheduler)

result = pipe(prompt="A cat sitting on a windowsill at sunset")
result.images[0].save("output.png")
```

### Image Understanding

```py
from PIL import Image

img = Image.open("photo.jpg")
result = pipe(image=img, question="Describe this image in detail.")
print(result.text)
```

### Image Editing

```py
result = pipe(image=img, instruction="Change the background to a beach.")
result.images[0].save("edited.png")
```

## UniLLaDaPipeline

[[autodoc]] UniLLaDaPipeline
    - all
    - __call__

## UniLLaDaPipelineOutput

[[autodoc]] pipelines.UniLLaDaPipelineOutput
