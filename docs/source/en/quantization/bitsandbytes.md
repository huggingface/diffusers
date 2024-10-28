<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# bitsandbytes

[bitsandbytes](https://huggingface.co/docs/bitsandbytes/index) is the easiest option for quantizing a model to 8 and 4-bit. 8-bit quantization multiplies outliers in fp16 with non-outliers in int8, converts the non-outlier values back to fp16, and then adds them together to return the weights in fp16. This reduces the degradative effect outlier values have on a model's performance.

4-bit quantization compresses a model even further, and it is commonly used with [QLoRA](https://hf.co/papers/2305.14314) to finetune quantized LLMs.


To use bitsandbytes, make sure you have the following libraries installed:

```bash
pip install diffusers transformers accelerate bitsandbytes -U
```

Now you can quantize a model by passing a [`BitsAndBytesConfig`] to [`~ModelMixin.from_pretrained`]. This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

<hfoptions id="bnb">
<hfoption id="8-bit">

Quantizing a model in 8-bit halves the memory-usage:

```py
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    quantization_config=quantization_config
)
```

By default, all the other modules such as `torch.nn.LayerNorm` are converted to `torch.float16`. You can change the data type of these modules with the `torch_dtype` parameter if you want:

```py
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.float32
)
model_8bit.transformer_blocks.layers[-1].norm2.weight.dtype
```

Once a model is quantized, you can push the model to the Hub with the [`~ModelMixin.push_to_hub`] method. The quantization `config.json` file is pushed first, followed by the quantized model weights. You can also save the serialized 4-bit models locally with [`~ModelMixin.save_pretrained`].

</hfoption>
<hfoption id="4-bit">

Quantizing a model in 4-bit reduces your memory-usage by 4x:

```py
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    quantization_config=quantization_config
)
```

By default, all the other modules such as `torch.nn.LayerNorm` are converted to `torch.float16`. You can change the data type of these modules with the `torch_dtype` parameter if you want:

```py
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.float32
)
model_4bit.transformer_blocks.layers[-1].norm2.weight.dtype
```

Call [`~ModelMixin.push_to_hub`] after loading it in 4-bit precision. You can also save the serialized 4-bit models locally with [`~ModelMixin.save_pretrained`].  

</hfoption>
</hfoptions>

<Tip warning={true}>

Training with 8-bit and 4-bit weights are only supported for training *extra* parameters.

</Tip>

Check your memory footprint with the `get_memory_footprint` method:

```py
print(model.get_memory_footprint())
```

Quantized models can be loaded from the [`~ModelMixin.from_pretrained`] method without needing to specify the `quantization_config` parameters:

```py
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = FluxTransformer2DModel.from_pretrained(
    "hf-internal-testing/flux.1-dev-nf4-pkg", subfolder="transformer"
)
```

## 8-bit (LLM.int8() algorithm)

<Tip>

Learn more about the details of 8-bit quantization in this [blog post](https://huggingface.co/blog/hf-bitsandbytes-integration)!

</Tip>

This section explores some of the specific features of 8-bit models, such as outlier thresholds and skipping module conversion.

### Outlier threshold

An "outlier" is a hidden state value greater than a certain threshold, and these values are computed in fp16. While the values are usually normally distributed ([-3.5, 3.5]), this distribution can be very different for large models ([-60, 6] or [6, 60]). 8-bit quantization works well for values ~5, but beyond that, there is a significant performance penalty. A good default threshold value is 6, but a lower threshold may be needed for more unstable models (small models or finetuning).

To find the best threshold for your model, we recommend experimenting with the `llm_int8_threshold` parameter in [`BitsAndBytesConfig`]:

```py
from diffusers import FluxTransformer2DModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True, llm_int8_threshold=10,
)

model_8bit = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    quantization_config=quantization_config,
)
```

### Skip module conversion

For some models, you don't need to quantize every module to 8-bit which can actually cause instability. For example, for diffusion models like [Stable Diffusion 3](../api/pipelines/stable_diffusion/stable_diffusion_3), the `proj_out` module can be skipped using the `llm_int8_skip_modules` parameter in [`BitsAndBytesConfig`]:

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


## 4-bit (QLoRA algorithm)

<Tip>

Learn more about its details in this [blog post](https://huggingface.co/blog/4bit-transformers-bitsandbytes).

</Tip>

This section explores some of the specific features of 4-bit models, such as changing the compute data type, using the Normal Float 4 (NF4) data type, and using nested quantization.


### Compute data type

To speedup computation, you can change the data type from float32 (the default value) to bf16 using the `bnb_4bit_compute_dtype` parameter in [`BitsAndBytesConfig`]:

```py
import torch
from diffusers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
```

### Normal Float 4 (NF4)

NF4 is a 4-bit data type from the [QLoRA](https://hf.co/papers/2305.14314) paper, adapted for weights initialized from a normal distribution. You should use NF4 for training 4-bit base models. This can be configured with the `bnb_4bit_quant_type` parameter in the [`BitsAndBytesConfig`]:

```py
from diffusers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4 = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    quantization_config=nf4_config,
)
```

For inference, the `bnb_4bit_quant_type` does not have a huge impact on performance. However, to remain consistent with the model weights, you should use the `bnb_4bit_compute_dtype` and `torch_dtype` values.

### Nested quantization

Nested quantization is a technique that can save additional memory at no additional performance cost. This feature performs a second quantization of the already quantized weights to save an additional 0.4 bits/parameter. 

```py
from diffusers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

double_quant_model = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    quantization_config=double_quant_config,
)
```

## Dequantizing `bitsandbytes` models

Once quantized, you can dequantize the model to the original precision but this might result in a small quality loss of the model. Make sure you have enough GPU RAM to fit the dequantized model. 

```python
from diffusers import BitsAndBytesConfig

double_quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
)

double_quant_model = SD3Transformer2DModel.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    subfolder="transformer",
    quantization_config=double_quant_config,
)
model.dequantize()
```

## Resources

* [End-to-end notebook showing Flux.1 Dev inference in a free-tier Colab](https://gist.github.com/sayakpaul/c76bd845b48759e11687ac550b99d8b4)
* [Training](https://gist.github.com/sayakpaul/05afd428bc089b47af7c016e42004527)