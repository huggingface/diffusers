<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# bitsandbytes

[bitsandbytes](https://huggingface.co/docs/bitsandbytes/index) is a k-bit quantization library with two quantization algorithms.

- [LLM.int8](https://huggingface.co/papers/2208.07339) reduces memory use by half by quantizing most features to 8-bits, while handling outliers with 16-bit operations, all without performance loss.
- [QLoRA](https://huggingface.co/papers/2305.14314) compresses weights to 4-bits, and adds a small set of trainable low-rank adapters, reducing memory use without hurting performance.

This guide demonstrates how quantization enables inference with large diffusion models on less than 16GB of memory.

Make sure the bitsandbytes library is installed.

```bash
pip -U install diffusers transformers accelerate bitsandbytes
```

Pass a [`BitsAndBytesConfig`] to [`~ModelMixin.from_pretrained`] to quantize a model. The [`BitsAndBytesConfig`] contains your quantization configuration. The [`CLIPTextModel`] and [`AutoencoderKL`] aren't quantized because they're already small and because [`AutoencoderKL`] only has a few `torch.nn.Linear` layers. bitsandbytes is supported in Transformers and Diffusers, so you can quantize [`FluxTransformer2DModel`] and [`transformers.T5EncoderModel`].

By default, all the other modules such as `torch.nn.LayerNorm` are converted to `torch.float16`. You can change the data type of these modules with the `torch_dtype` parameter.

This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

> [!TIP]
> For Ada and higher-series GPUs, change `torch_dtype` to `torch.bfloat16`.

<hfoptions id="bnb">
<hfoption id="8-bit">

Quantizing a model to 8-bits reduces memory usage by 2x.

```py
import torch
from diffusers import AutoModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

quant_config = TransformersBitsAndBytesConfig(load_in_8bit=True,)
text_encoder_2_8bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True,)
transformer_8bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
```

Set `device_map="cuda"` to place the pipeline on an accelerator like a GPU.

```py
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer_8bit,
    text_encoder_2=text_encoder_2_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt="""
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

image = pipeline(prompt).images[0]
```

</hfoption>
<hfoption id="4-bit">

Quantizing a model to 4-bit reduces your memory usage by 4x.

```py
import torch
from diffusers import AutoModel
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import T5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

quant_config = TransformersBitsAndBytesConfig(load_in_4bit=True,)
text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="text_encoder_2",
    quantization_config=quant_config,
    dtype=torch.float16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_4bit=True,)
transformer_4bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
```

Set `device_map="cuda"` to place the pipeline on an accelerator like a GPU.

```py
from diffusers import FluxPipeline

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer_4bit,
    text_encoder_2=text_encoder_2_4bit,
    torch_dtype=torch.float16,
    device_map="cuda",
)

prompt="""
cinematic film still of a cat sipping a margarita in a pool in Palm Springs, California
highly detailed, high budget hollywood movie, cinemascope, moody, epic, gorgeous, film grain
"""

image = pipeline(prompt).images[0]
```

</hfoption>
</hfoptions>

Use [`~ModelMixin.get_memory_footprint`] to estimate the memory footprint of the model parameters. It does not estimate the inference memory requirements.

```py
print(model.get_memory_footprint())
```

## LLM.int8

This section goes over outlier thresholds and skipping module conversion, features specific to the LLM.int8 algorithm.

### Outlier threshold

An "outlier" is a hidden state value greater than a certain threshold and they're computed in fp16. While the values are usually normally distributed ([-3.5, 3.5]), this distribution can be very different for large models ([-60, 6] or [6, 60]). 8-bit quantization works well for values ~5, but beyond that, there is a significant performance penalty. A good default threshold value is 6, but a lower threshold may be needed for more unstable models (small models or fine-tuning).

Experiment with the `llm_int8_threshold` argument to find the best threshold for your model.

```py
from diffusers import AutoModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, llm_int8_threshold=10,
)

model_8bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quantization_config,
)
```

### Skip module conversion

Some models don't require every module to be quantized to 8-bits. This can actually cause instability. For example, in [Stable Diffusion 3](../api/pipelines/stable_diffusion/stable_diffusion_3), skip the `proj_out` module using the `llm_int8_skip_modules` argument.

```py
from diffusers import SD3Transformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, llm_int8_skip_modules=["proj_out"],
)

model_8bit = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    quantization_config=quantization_config,
)
```

## QLoRA

This section goes over compute data type, Normal Float 4 (NF4) data type, and nested quantization, features specific to QLoRA.

### Compute data type

Change the data type from float32 (the default value) to bf16 using the `bnb_4bit_compute_dtype` argument to speed up computation. Use the same `bnb_4bit_compute_dtype` and `torch_dtype` values to remain consistent.

```py
import torch
from diffusers import BitsAndBytesConfig, AutoModel

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
```

### Normal Float 4 (NF4)

NF4 is a 4-bit data type adapted for weights initialized from a normal distribution. Use NF4 for training 4-bit base models. For inference, NF4 does not have a significant impact on performance.

Configure the `bnb_4bit_quant_type` argument to `"nf4"`.

```py
from diffusers import AutoModel, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

transformer_4bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
```

### Nested quantization

Nested quantization quantizes the already quantized weights to save an additional 0.4 bits/parameter. Set `bnb_4bit_use_double_quant=True` to enable nested quantization.

```py
from diffusers import AutoModel, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

transformer_4bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
```

## Dequantize a model

Dequantizing recovers the model weights original precision but you may experience a small loss in quality. Make sure you have enough GPU memory to fit the dequantized model.

Call [`~ModelMixin.dequantize`] to dequantize a model.

```python
from diffusers import AutoModel, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

transformer_4bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

transformer_4bit.dequantize()
```

## torch.compile

Speed up inference with [torch.compile](../optimization/fp16#torchcompile). Make sure you have the latest bitsandbytes and [PyTorch nightly](https://pytorch.org/get-started/locally/) installed.

```py
import torch
from diffusers import BitsAndBytesConfig, AutoModel

torch._dynamo.config.capture_dynamic_output_shape_ops = True

quant_config = BitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = AutoModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)
transformer_8bit.compile(fullgraph=True)
```

On an RTX 4090 with compilation, 4-bit Flux generation completed in 25.809 seconds versus 32.570 seconds without. Check out the [benchmarking script](https://gist.github.com/sayakpaul/0db9d8eeeb3d2a0e5ed7cf0d9ca19b7d) for more details.

## Resources

* Read [A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using Hugging Face Transformers, Accelerate and bitsandbytes](https://huggingface.co/blog/hf-bitsandbytes-integration) to learn more about 8-bit quantization.
* Read [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) to learn more about 4-bit quantization.
* Check out this [notebook](https://gist.github.com/sayakpaul/c76bd845b48759e11687ac550b99d8b4) for an example of FLUX.1-dev inference on a free-tier instance of Colab.
* Take a look at this [training script](https://github.com/huggingface/diffusers/blob/8c661ea586bf11cb2440da740dd3c4cf84679b85/examples/dreambooth/README_hidream.md#using-quantization) which quantizes the base model with bitsandbytes.