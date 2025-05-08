<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

-->

# Quantization

Quantization techniques focus on representing data with less information while also trying to not lose too much accuracy. This often means converting a data type to represent the same information with fewer bits. For example, if your model weights are stored as 32-bit floating points and they're quantized to 16-bit floating points, this halves the model size which makes it easier to store and reduces memory-usage. Lower precision can also speedup inference because it takes less time to perform calculations with fewer bits.

<Tip>

Interested in adding a new quantization method to Diffusers? Refer to the [Contribute new quantization method guide](https://huggingface.co/docs/transformers/main/en/quantization/contribute) to learn more about adding a new quantization method.

</Tip>

<Tip>

If you are new to the quantization field, we recommend you to check out these beginner-friendly courses about quantization in collaboration with DeepLearning.AI:

* [Quantization Fundamentals with Hugging Face](https://www.deeplearning.ai/short-courses/quantization-fundamentals-with-hugging-face/)
* [Quantization in Depth](https://www.deeplearning.ai/short-courses/quantization-in-depth/)

</Tip>

## When to use what?

Diffusers currently supports the following quantization methods.
- [BitsandBytes](./bitsandbytes)
- [TorchAO](./torchao)
- [GGUF](./gguf)
- [Quanto](./quanto.md)

[This resource](https://huggingface.co/docs/transformers/main/en/quantization/overview#when-to-use-what) provides a good overview of the pros and cons of different quantization techniques.

## Pipeline-level quantization

Diffusers allows users to directly initialize pipelines from checkpoints that may contain quantized models ([example](https://huggingface.co/hf-internal-testing/flux.1-dev-nf4-pkg)). However, users may want to apply
quantization on-the-fly when initializing a pipeline from a pre-trained and non-quantized checkpoint. You can
do this with [`~quantizers.PipelineQuantizationConfig`].

Start by defining a `PipelineQuantizationConfig`:

```py
import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers.quantization_config import QuantoConfig
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={
        "transformer": QuantoConfig(weights_dtype="int8"),
        "text_encoder_2": BitsAndBytesConfig(
            load_in_4bit=True, compute_dtype=torch.bfloat16
        ),
    }
)
```

Then pass it to [`~DiffusionPipeline.from_pretrained`] and run inference:

```py
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe("photo of a cute dog").images[0]
```

This method allows for more granular control over the quantization specifications of individual 
model-level components of a pipeline. It also allows for different quantization backends for
different components. In the above example, you used a combination of Quanto and BitsandBytes. However,
one caveat of this method is that users need to know which components come from `transformers` to be able
to import the right quantization config class.

The other method is simpler in terms of experience but is
less-flexible. Start by defining a `PipelineQuantizationConfig` but in a different way:

```py
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer", "text_encoder_2"],
)
```

This `pipeline_quant_config` can now be passed to [`~DiffusionPipeline.from_pretrained`] similar to the above example.

In this case, `quant_kwargs` will be used to initialize the quantization specifications
of the respective quantization configuration class of `quant_backend`. `components_to_quantize`
is used to denote the components that will be quantized. For most pipelines, you would want to
keep `transformer` in the list as that is often the most compute and memory intensive.

The config below will work for most diffusion pipelines that have a `transformer` component present.
In most case, you will want to quantize the `transformer` component as that is often the most compute-
intensive part of a diffusion pipeline.

```py
pipeline_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": torch.bfloat16},
    components_to_quantize=["transformer"],
)
```

Below is a list of the supported quantization backends available in both `diffusers` and `transformers`:

* `bitsandbytes_4bit` 
* `bitsandbytes_8bit`
* `gguf`
* `quanto`
* `torchao`


Diffusion pipelines can have multiple text encoders. [`FluxPipeline`] has two, for example. It's
recommended to quantize the text encoders that are memory-intensive. Some examples include T5,
Llama, Gemma, etc. In the above example, you quantized the T5 model of [`FluxPipeline`] through
`text_encoder_2` while keeping the CLIP model intact (accessible through `text_encoder`). 