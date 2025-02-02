<!--Copyright 2024 The HuggingFace Team and Tencent Hunyuan Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Hunyuan-DiT
![chinese elements understanding](https://github.com/gnobitab/diffusers-hunyuan/assets/1157982/39b99036-c3cb-4f16-bb1a-40ec25eda573)

[Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](https://arxiv.org/abs/2405.08748) from Tencent Hunyuan.

The abstract from the paper is:

*We present Hunyuan-DiT, a text-to-image diffusion transformer with fine-grained understanding of both English and Chinese. To construct Hunyuan-DiT, we carefully design the transformer structure, text encoder, and positional encoding. We also build from scratch a whole data pipeline to update and evaluate data for iterative model optimization. For fine-grained language understanding, we train a Multimodal Large Language Model to refine the captions of the images. Finally, Hunyuan-DiT can perform multi-turn multimodal dialogue with users, generating and refining images according to the context. Through our holistic human evaluation protocol with more than 50 professional human evaluators, Hunyuan-DiT sets a new state-of-the-art in Chinese-to-image generation compared with other open-source models.*


You can find the original codebase at [Tencent/HunyuanDiT](https://github.com/Tencent/HunyuanDiT) and all the available checkpoints at [Tencent-Hunyuan](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

**Highlights**: HunyuanDiT supports Chinese/English-to-image, multi-resolution generation.

HunyuanDiT has the following components:
* It uses a diffusion transformer as the backbone
* It combines two text encoders, a bilingual CLIP and a multilingual T5 encoder

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

<Tip>

You can further improve generation quality by passing the generated image from [`HungyuanDiTPipeline`] to the [SDXL refiner](../../using-diffusers/sdxl#base-to-refiner-model) model.

</Tip>

## Optimization

You can optimize the pipeline's runtime and memory consumption with torch.compile and feed-forward chunking. To learn about other optimization methods, check out the [Speed up inference](../../optimization/fp16) and [Reduce memory usage](../../optimization/memory) guides.

### Inference

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

```python
from diffusers import HunyuanDiTPipeline
import torch

pipeline = HunyuanDiTPipeline.from_pretrained(
	"Tencent-Hunyuan/HunyuanDiT-Diffusers", torch_dtype=torch.float16
).to("cuda")
```

Then change the memory layout of the pipelines `transformer` and `vae` components to `torch.channels-last`:

```python
pipeline.transformer.to(memory_format=torch.channels_last)
pipeline.vae.to(memory_format=torch.channels_last)
```

Finally, compile the components and run inference:

```python
pipeline.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)
pipeline.vae.decode = torch.compile(pipeline.vae.decode, mode="max-autotune", fullgraph=True)

image = pipeline(prompt="一个宇航员在骑马").images[0]
```

The [benchmark](https://gist.github.com/sayakpaul/29d3a14905cfcbf611fe71ebd22e9b23) results on a 80GB A100 machine are:

```bash
With torch.compile(): Average inference time: 12.470 seconds.
Without torch.compile(): Average inference time: 20.570 seconds.
```

### Memory optimization

By loading the T5 text encoder in 8 bits, you can run the pipeline in just under 6 GBs of GPU VRAM. Refer to [this script](https://gist.github.com/sayakpaul/3154605f6af05b98a41081aaba5ca43e) for details.

Furthermore, you can use the [`~HunyuanDiT2DModel.enable_forward_chunking`] method to reduce memory usage. Feed-forward chunking runs the feed-forward layers in a transformer block in a loop instead of all at once. This gives you a trade-off between memory consumption and inference runtime.

```diff
+ pipeline.transformer.enable_forward_chunking(chunk_size=1, dim=1)
```


## HunyuanDiTPipeline

[[autodoc]] HunyuanDiTPipeline
	- all
	- __call__

