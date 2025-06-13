<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Chroma

<div class="flex flex-wrap space-x-1">
  <img alt="LoRA" src="https://img.shields.io/badge/LoRA-d8b4fe?style=flat"/>
  <img alt="MPS" src="https://img.shields.io/badge/MPS-000000?style=flat&logo=apple&logoColor=white%22">
</div>

Chroma is a text to image generation model based on Flux.

Original model checkpoints for Chroma can be found [here](https://huggingface.co/lodestones/Chroma).

<Tip>

Chroma can use all the same optimizations as Flux.

</Tip>

## Inference (Single File)

The `ChromaTransformer2DModel` supports loading checkpoints in the original format. This is also useful when trying to load finetunes or quantized versions of the models that have been published by the community.

The following example demonstrates how to run Chroma from a single file.

Then run the following example

```python
import torch
from diffusers import ChromaTransformer2DModel, ChromaPipeline
from transformers import T5EncoderModel

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16

transformer = ChromaTransformer2DModel.from_single_file("https://huggingface.co/lodestones/Chroma/blob/main/chroma-unlocked-v35.safetensors", torch_dtype=dtype)

text_encoder = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = T5Tokenizer.from_pretrained(bfl_repo, subfolder="tokenizer_2", torch_dtype=dtype)

pipe = ChromaPipeline.from_pretrained(bfl_repo, transformer=transformer, text_encoder=text_encoder, tokenizer=tokenizer, torch_dtype=dtype)

pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=4.0,
    output_type="pil",
    num_inference_steps=26,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

image.save("image.png")
```

## ChromaPipeline

[[autodoc]] ChromaPipeline
	- all
	- __call__
