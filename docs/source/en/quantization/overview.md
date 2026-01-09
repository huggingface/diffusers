<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Getting started

Quantization focuses on representing data with fewer bits while also trying to preserve the precision of the original data. This often means converting a data type to represent the same information with fewer bits. For example, if your model weights are stored as 32-bit floating points and they're quantized to 16-bit floating points, this halves the model size which makes it easier to store and reduces memory usage. Lower precision can also speedup inference because it takes less time to perform calculations with fewer bits.

Diffusers supports multiple quantization backends to make large diffusion models like [Flux](../api/pipelines/flux) more accessible. This guide shows how to use the [`~quantizers.PipelineQuantizationConfig`] class to quantize a pipeline during its initialization from a pretrained or non-quantized checkpoint.

## Pipeline-level quantization

There are two ways to use [`~quantizers.PipelineQuantizationConfig`] depending on how much customization you want to apply to the quantization configuration. 

- for basic use cases, define the `quant_backend`, `quant_kwargs`, and `components_to_quantize` arguments
- for granular quantization control, define a `quant_mapping` that provides the quantization configuration for individual model components

### Basic quantization

Initialize [`~quantizers.PipelineQuantizationConfig`] with the following parameters.

- `quant_backend` specifies which quantization backend to use. Currently supported backends include: `bitsandbytes_4bit`, `bitsandbytes_8bit`, `gguf`, `quanto`, and `torchao`.
- `quant_kwargs` specifies the quantization arguments to use.

> [!TIP]
> These `quant_kwargs` arguments are different for each backend. Refer to the [Quantization API](../api/quantization) docs to view the arguments for each backend.

- `components_to_quantize` specifies which component(s) of the pipeline to quantize. Typically, you should quantize the most compute intensive components like the transformer. The text encoder is another component to consider quantizing if a pipeline has more than one such as [`FluxPipeline`]. The example below quantizes the T5 text encoder in [`FluxPipeline`] while keeping the CLIP model intact.

   `components_to_quantize` accepts either a list for multiple models or a string for a single model.

The example below loads the bitsandbytes backend with the following arguments from [`~quantizers.quantization_config.BitsAndBytesConfig`], `load_in_4bit`, `bnb_4bit_quant_type`, and `bnb_4bit_compute_dtype`.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
```

Pass the `pipeline_quant_config` to [`~DiffusionPipeline.from_pretrained`] to quantize the pipeline.

```py
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("photo of a cute dog").images[0]
```


### Advanced quantization

The `quant_mapping` argument provides more options for how to quantize each individual component in a pipeline, like combining different quantization backends.

Initialize [`~quantizers.PipelineQuantizationConfig`] and pass a `quant_mapping` to it. The `quant_mapping` allows you to specify the quantization options for each component in the pipeline such as the transformer and text encoder.

The example below uses two quantization backends, [`~quantizers.quantization_config.QuantoConfig`] and [`transformers.BitsAndBytesConfig`], for the transformer and text encoder.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers.quantization_config import QuantoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": QuantoConfig(weights_dtype="int8"),
        "text_encoder_2": TransformersBitsAndBytesConfig(
            load_in_4bit=True, compute_dtype=torch.bfloat16
        ),
    }
)
```

There is a separate bitsandbytes backend in [Transformers](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig). You need to import and use [`transformers.BitsAndBytesConfig`] for components that come from Transformers. For example, `text_encoder_2` in [`FluxPipeline`] is a [`~transformers.T5EncoderModel`] from Transformers so you need to use [`transformers.BitsAndBytesConfig`] instead of [`diffusers.BitsAndBytesConfig`].

> [!TIP]
> Use the [basic quantization](#basic-quantization) method above if you don't want to manage these distinct imports or aren't sure where each pipeline component comes from.

```py
import torch
from diffusers import DiffusionPipeline
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": DiffusersBitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        "text_encoder_2": TransformersBitsAndBytesConfig(
            load_in_4bit=True, compute_dtype=torch.bfloat16
        ),
    }
)
```

Pass the `pipeline_quant_config` to [`~DiffusionPipeline.from_pretrained`] to quantize the pipeline.

```py
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("photo of a cute dog").images[0]
```

## Resources

Check out the resources below to learn more about quantization.

- If you are new to quantization, we recommend checking out the following beginner-friendly courses in collaboration with DeepLearning.AI.

    - [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
    - [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

- Refer to the [Contribute new quantization method guide](https://huggingface.co/docs/transformers/main/en/quantization/contribute) if you're interested in adding a new quantization method.

- The Transformers quantization [Overview](https://huggingface.co/docs/transformers/quantization/overview#when-to-use-what) provides an overview of the pros and cons of different quantization backends.

- Read the [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization) blog post for a brief introduction to each quantization backend, how to choose a backend, and combining quantization with other memory optimizations.
