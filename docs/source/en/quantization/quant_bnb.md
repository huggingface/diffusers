<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Quantize Your Diffusion Models with `diffusers`

Quantization reduces the number of bits used to represent numbers in a model's parameters,
saving memory without significantly sacrificing accuracy. This can be a game-changer for diffusion
models, which are popular for generating high-quality images but are notoriously resource-hungry.
While these models run well on high-end setups or in the cloud, lighter versions are essential for
consumer-grade devices—and that's where quantization comes into play.

In this guide, we'll explore different types of quantization, from 8-bit to NF4—using
[`bitsandbytes`](https://github.com/bitsandbytes-foundation/bitsandbytes),
a powerful library that simplifies the quantization process. We'll work with the
[FLUX.1-dev model]((https://huggingface.co/black-forest-labs/FLUX.1-dev)),
demonstrating how quantization can help you run it on less than 16GB of VRAM—even on a free Google Colab instance.

To follow along, first install the latest versions of `diffusers` and `bitsandbytes`:

```shell
pip install -Uq diffusers bitsandbytes
```

## Using the Unquantized Model

We'll start by running the model with full precision using the `bf16` (bfloat16) data type.
This serves as our baseline to compare against when we apply quantization techniques later.
If you want to follow along and run this test, you'll need a GPU with more than 32 GB of RAM.

To compare memory usage and output quality, we'll create two global dictionaries: `memory_dict` and `image_dict`.
These will store the memory allocation of the pipeline and the images generated from the same prompt, respectively.

```python
memory_dict = {}
image_dict = {}
```

Import the necessary libraries:

```python
import torch
from diffusers import FluxPipeline
```

Load the model with `bf16` precision:

```python
checkpoint_id = "black-forest-labs/FLUX.1-dev"

pipe = FluxPipeline.from_pretrained(
    checkpoint_id,
    torch_dtype=torch.bfloat16,
).to("cuda")
```

Set up the pipeline arguments:

```python
pipe_kwargs = {
    "prompt": "A cat holding a sign that says hello world",
    "height": 1024,
    "width": 1024,
    "guidance_scale": 3.5,
    "num_inference_steps": 50,
    "max_sequence_length": 512,
}

image = pipe(
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_kwargs,
).images[0]

image.resize((224, 224))
```

![bf16 image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quant-bnb/bf16.png)

Measure the GPU memory allocated:

```python
memory_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")
```

Store the memory usage and generated image:

```python
memory_dict["bf16"] = memory_allocated
image_dict["bf16"] = image
```

Define a function to free up GPU memory:

```python
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
```

Clean up before the next experiment:

```python
del pipe
flush()
```

## 8-Bit Quantization

Next, we'll apply 8-bit quantization to the model. Theoretically, this should reduce the memory
requirements by half, since `bf16` uses 16 bits per parameter compared to the 8 bits used in
8-bit quantization.

Import the necessary classes:

```python
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel
```

Set the data type to `fp16` (float16):

Quantize the language model:

```python
quant_config = TransformersBitsAndBytesConfig(
    load_in_8bit=True,
)

text_encoder_2_8bit = T5EncoderModel.from_pretrained(
    checkpoint_id,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
```

Quantize the denoiser model:

```python
quant_config = DiffusersBitsAndBytesConfig(
    load_in_8bit=True,
)

transformer_8bit = FluxTransformer2DModel.from_pretrained(
    checkpoint_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
```

Load the pipeline with the quantized models:

```python
pipe = FluxPipeline.from_pretrained(
    checkpoint_id,
    transformer=transformer_8bit,
    text_encoder_2=text_encoder_2_8bit,
    torch_dtype=torch.float16,
    device_map="balanced"
)
```

Generate the image:

```python
image = pipe(
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_kwargs,
).images[0]

image.resize((224, 224))
```

![8 bit image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quant-bnb/8bit.png)

Measure and store the GPU memory allocated:

```python
memory_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")

memory_dict["8bit"] = memory_allocated
image_dict["8bit"] = image
```

Clean up:

```python
del text_encoder_2_8bit
del transformer_8bit
del pipe
flush()
```

## 4-Bit Quantization

Why settle for reducing memory usage by half when we can do even better? Let's take it a step further
and apply 4-bit quantization.

Quantize the language model:

```python
quant_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    checkpoint_id,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
```

Quantize the denoiser model:

```python
quant_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
)

transformer_4bit = FluxTransformer2DModel.from_pretrained(
    checkpoint_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
```

Load the pipeline with the 4-bit quantized models:

```python
pipe = FluxPipeline.from_pretrained(
    checkpoint_id,
    transformer=transformer_4bit,
    text_encoder_2=text_encoder_2_4bit,
    torch_dtype=torch.float16,
    device_map="balanced"
)
```

Generate the image:

```python
image = pipe(
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_kwargs,
).images[0]

image.resize((224, 224))
```

![4 bit image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quant-bnb/4bit.png)

Measure and store the GPU memory allocated:

```python
memory_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")

memory_dict["4bit"] = memory_allocated
image_dict["4bit"] = image
```

Clean up:

```python
del text_encoder_2_4bit
del transformer_4bit
del pipe
flush()
```

## NF4 Quantization

Finally, we'll explore NF4 quantization, which uses a special 4-bit format that improves the
representation of numbers through normalization.

Quantize the language model:

```python
quant_config = TransformersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

text_encoder_2_nf4 = T5EncoderModel.from_pretrained(
    checkpoint_id,
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
```

Quantize the denoiser model:

```python
quant_config = DiffusersBitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

transformer_nf4 = FluxTransformer2DModel.from_pretrained(
    checkpoint_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
```

Load the pipeline with the NF4 quantized models:

```python
pipe = FluxPipeline.from_pretrained(
    checkpoint_id,
    transformer=transformer_nf4,
    text_encoder_2=text_encoder_2_nf4,
    torch_dtype=torch.float16,
    device_map="balanced"
)
```

Generate the image:

```python
image = pipe(
    generator=torch.Generator("cpu").manual_seed(0),
    **pipe_kwargs,
).images[0]

image.resize((224, 224))
```

![nf4 image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quant-bnb/nf4.png)

Measure and store the GPU memory allocated:

```python
memory_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
print(f"GPU Memory Allocated: {memory_allocated:.2f} GB")

memory_dict["nf4"] = memory_allocated
image_dict["nf4"] = image
```

Clean up:

```python
del text_encoder_2_nf4
del transformer_nf4
del pipe
flush()
```

## Comparison of Quantization Methods

Let's compare the memory usage and output images for each quantization method.

```python
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
```

Plot the images and memory usage:

```python
# Sort the keys based on memory usage in descending order
keys = sorted(memory_dict, key=memory_dict.get, reverse=True)
memory_values = [memory_dict[key] for key in keys]

# Create a figure
fig = plt.figure(figsize=(16, 8))

# Grid specification
gs = fig.add_gridspec(2, len(keys), height_ratios=[3, 1])

# Plot images in order of decreasing memory usage
for i, key in enumerate(keys):
    ax = fig.add_subplot(gs[0, i])
    img = image_dict[key]
    ax.imshow(img)
    ax.axis('off')

    # Add a semi-transparent rectangle at the bottom
    rect = Rectangle(
        (0, 0), 1, 0.15, transform=ax.transAxes,
        color='black', alpha=0.5
    )
    ax.add_patch(rect)

    # Add text over the rectangle
    mem_usage = memory_dict[key]
    ax.text(
        0.5, 0.01,
        f"{key}\nMemory: {mem_usage:.2f} GB",
        transform=ax.transAxes, color='white', fontsize=12,
        ha='center', va='bottom'
    )

# Plot the memory usage trend line
ax_line = fig.add_subplot(gs[1, :])
ax_line.plot(keys, memory_values, marker='o', linestyle='-', color='orange')
ax_line.set_xlabel('Quantization Type')
ax_line.set_ylabel('Memory Usage (GB)')
ax_line.set_title('Memory Usage with Different Quantization Methods')
ax_line.grid(True)

# Annotate each point with memory usage
for i, mem in enumerate(memory_values):
    ax_line.annotate(
        f"{mem:.2f} GB", (keys[i], mem), textcoords="offset points",
        xytext=(0,10), ha='center'
    )

# Adjust x-axis labels to match the images
ax_line.set_xticks(keys)
ax_line.set_xticklabels(keys)

# Adjust layout
plt.tight_layout()
plt.show()
```

![nf4 image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/quant-bnb/comparison.png)

This visualization helps us see how different quantization methods reduce memory usage and how they
affect the generated images. Generally, lower-bit quantization reduces memory consumption but may
introduce artifacts or degrade image quality. However, as you can see, the images generated with
4-bit and NF4 quantization are still quite acceptable, especially considering the significant memory
savings.

## Training

Diffusers also support training [LoRAs](./training/lora) with quantized weights. Refer to the
[flux_lora_quantization.py](https://github.com/huggingface/diffusers/tree/main/examples/research_projects/flux_lora_quantization)
script for an example of how to fine-tune FLUX.1-dev with LoRA and bitsandbytes.