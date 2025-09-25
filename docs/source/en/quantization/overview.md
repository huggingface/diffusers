<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Overview

Quantization represents data in lower precision to save memory. For example, quantizing model weights from fp32 to fp16 halves the model size. Lower precision also speeds up inference because calculations take less time with fewer bits.

Diffusers supports multiple quantization backends. This makes large diffusion models accessible on all hardware types. This guide shows how to use [`~quantizers.PipelineQuantizationConfig`] to quantize a pipeline from a pretrained checkpoint.

## Pipeline-level quantization

You can use [`~quantizers.PipelineQuantizationConfig`] in two ways.

- For a single backend, define the `quant_backend`, `quant_kwargs`, and `components_to_quantize` arguments.
- For multiple backends, define a `quant_mapping` that provides the quantization configuration for individual model components.

### Single quantization backend

Initialize [`~quantizers.PipelineQuantizationConfig`] with these parameters.

- `quant_backend` specifies which quantization backend to use. Supported backends include: `bitsandbytes_4bit`, `bitsandbytes_8bit`, `gguf`, `quanto`, and `torchao`.
- `quant_kwargs` specifies the quantization arguments to use.

> [!TIP]
> The `quant_kwargs` arguments differ for each backend. Refer to the [Quantization API](../api/quantization) docs to view the specific arguments for each backend.

- `components_to_quantize` specifies which component(s) of the pipeline to quantize. Quantize the most compute intensive components like the transformer. The text encoder is another component to consider quantizing if a pipeline has more than one such as [`FluxPipeline`]. The example below quantizes the T5 text encoder in [`FluxPipeline`] while keeping the CLIP model intact.

   `components_to_quantize` accepts either a list for multiple models or a string for a single model.

The example below configures the bitsandbytes backend with the `load_in_4bit`, `bnb_4bit_quant_type`, and `bnb_4bit_compute_dtype` arguments from [`~quantizers.quantization_config.BitsAndBytesConfig`].

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
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)
```

### Multi-quantization backend

The `quant_mapping` argument provides more options for quantizing each individual component in a pipeline to combine different quantization backends.

Initialize [`~quantizers.PipelineQuantizationConfig`] and pass a `quant_mapping` to it. The `quant_mapping` lets you specify the quantization options for each component in the pipeline such as the transformer and text encoder.

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

[Transformers](https://huggingface.co/docs/transformers/main_classes/quantization#transformers.BitsAndBytesConfig) has a separate bitsandbytes backend. Import and use [`transformers.BitsAndBytesConfig`] for components that come from Transformers. For example, `text_encoder_2` in [`FluxPipeline`] is from Transformers, so use [`transformers.BitsAndBytesConfig`] instead of [`diffusers.BitsAndBytesConfig`].

> [!TIP]
> Use the [single quantization backend](#single-quantization-backend) method above if you don't want to manage these distinct imports or aren't sure where each pipeline component comes from.

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
pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

image = pipeline("photo of a cute dog").images[0]
```

## Saving a quantized pipeline

Use the [`~PushToHubMixin.push_to_hub`] method to push the quantized pipeline to the Hub. This saves a quantization `config.json` file and the quantized model weights.

```py
pipeline.push_to_hub("my-repo")
```

You can also save the model locally with [`~ModelMixin.save_pretrained`].

```py
pipeline.save_pretrained("path/to/save/")
```

Reload the quantized model with [`~ModelMixin.from_pretrained`] without defining a [`~quantizers.PipelineQuantizationConfig`].

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "nunchaku-tech/nunchaku-flux.1-dev"
)
```

## Resources

Check out these resources to learn more about quantization.

- If you're new to quantization, check out these beginner-friendly courses in collaboration with DeepLearning.AI.

    - [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
    - [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

- Refer to the [Contribute new quantization method guide](https://huggingface.co/docs/transformers/main/en/quantization/contribute) if you want to add a new quantization method.

- The Transformers quantization [Overview](https://huggingface.co/docs/transformers/quantization/overview#when-to-use-what) shows the pros and cons of different quantization backends.

- Read the [Exploring Quantization Backends in Diffusers](https://huggingface.co/blog/diffusers-quantization) blog post for an introduction to each quantization backend, how to choose a backend, and combining quantization with other memory optimizations.
