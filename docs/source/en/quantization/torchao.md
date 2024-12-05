<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# torchao

[TorchAO](https://github.com/pytorch/ao) is an architecture optimization library for PyTorch, it provides high performance dtypes, optimization techniques and kernels for inference and training, featuring composability with native PyTorch features like `torch.compile`, FSDP etc. Some benchmark numbers can be found [here](https://github.com/pytorch/ao/tree/main/torchao/quantization#benchmarks).

Before you begin, make sure you have Pytorch version 2.5, or above, and TorchAO installed:

```bash
pip install -U torch torchao
```

## Usage

Now you can quantize a model by passing a [`TorchAoConfig`] to [`~ModelMixin.from_pretrained`]. Loading pre-quantized models is supported as well! This works for any model in any modality, as long as it supports loading with [Accelerate](https://hf.co/docs/accelerate/index) and contains `torch.nn.Linear` layers.

```python
from diffusers import FluxPipeline, FluxTransformer2DModel, TorchAoConfig

model_id = "black-forest-labs/Flux.1-Dev"
dtype = torch.bfloat16

quantization_config = TorchAoConfig("int8wo")
transformer = FluxTransformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=dtype,
)
pipe = FluxPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=dtype,
)
pipe.to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
image.save("output.png")
```

Additionally, TorchAO supports an automatic quantization API exposed with [`autoquant`](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md#autoquantization). Autoquantization determines the best quantization strategy applicable to a model by comparing the performance of each technique on chosen input types and shapes. This can directly be used with the underlying modeling components at the moment, but Diffusers will also expose an autoquant configuration option in the future.

## Resources

- [TorchAO Quantization API](https://github.com/pytorch/ao/blob/main/torchao/quantization/README.md)
- [Diffusers-TorchAO examples](https://github.com/sayakpaul/diffusers-torchao)
