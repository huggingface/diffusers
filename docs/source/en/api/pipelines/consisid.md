<!--Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
-->

# ConsisID

[Identity-Preserving Text-to-Video Generation by Frequency Decomposition](https://arxiv.org/abs/2411.17440) from Peking University & University of Rochester & etc, by Shenghai Yuan, Jinfa Huang, Xianyi He, Yunyang Ge, Yujun Shi, Liuhan Chen, Jiebo Luo, Li Yuan.

The abstract from the paper is:

*Identity-preserving text-to-video (IPT2V) generation aims to create high-fidelity videos with consistent human identity. It is an important task in video generation but remains an open problem for generative models. This paper pushes the technical frontier of IPT2V in two directions that have not been resolved in the literature: (1) A tuning-free pipeline without tedious case-by-case finetuning, and (2) A frequency-aware heuristic identity-preserving Diffusion Transformer (DiT)-based control scheme. To achieve these goals, we propose **ConsisID**, a tuning-free DiT-based controllable IPT2V model to keep human-**id**entity **consis**tent in the generated video. Inspired by prior findings in frequency analysis of vision/diffusion transformers, it employs identity-control signals in the frequency domain, where facial features can be decomposed into low-frequency global features (e.g., profile, proportions) and high-frequency intrinsic features (e.g., identity markers that remain unaffected by pose changes). First, from a low-frequency perspective, we introduce a global facial extractor, which encodes the reference image and facial key points into a latent space, generating features enriched with low-frequency information. These features are then integrated into the shallow layers of the network to alleviate training challenges associated with DiT. Second, from a high-frequency perspective, we design a local facial extractor to capture high-frequency details and inject them into the transformer blocks, enhancing the model's ability to preserve fine-grained features. To leverage the frequency information for identity preservation, we propose a hierarchical training strategy, transforming a vanilla pre-trained video generation model into an IPT2V model. Extensive experiments demonstrate that our frequency-aware heuristic scheme provides an optimal control solution for DiT-based models. Thanks to this scheme, our **ConsisID** achieves excellent results in generating high-quality, identity-preserving videos, making strides towards more effective IPT2V. The model weight of ConsID is publicly available at https://github.com/PKU-YuanGroup/ConsisID.*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers.md) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading.md#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

This pipeline was contributed by [SHYuanBest](https://github.com/SHYuanBest). The original codebase can be found [here](https://github.com/PKU-YuanGroup/ConsisID). The original weights can be found under [hf.co/BestWishYsh](https://huggingface.co/BestWishYsh).

There are two official ConsisID checkpoints for identity-preserving text-to-video.

| checkpoints | recommended inference dtype |
|:---:|:---:|
| [`BestWishYsh/ConsisID-preview`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |
| [`BestWishYsh/ConsisID-1.5`](https://huggingface.co/BestWishYsh/ConsisID-preview) | torch.bfloat16 |

## Inference

Use [`torch.compile`](https://huggingface.co/docs/diffusers/main/en/tutorials/fast_diffusion#torchcompile) to reduce the inference latency.

First, load the pipeline:

```python
import torch
from diffusers import ConsisIDPipeline
from diffusers.pipelines.consisid.util_consisid import prepare_face_models, process_face_embeddings_infer
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

snapshot_download(repo_id="BestWishYsh/ConsisID-preview", local_dir="BestWishYsh/ConsisID-preview")
face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std = prepare_face_models("BestWishYsh/ConsisID-preview", device="cuda", dtype=torch.bfloat16)
pipe = ConsisIDPipeline.from_pretrained("BestWishYsh/ConsisID-preview", torch_dtype=torch.bfloat16)
```

Then change the memory layout of the pipelines `transformer` component to `torch.channels_last`:

```python
pipe.transformer.to(memory_format=torch.channels_last)
```

Compile the components and run inference:

```python
pipe.transformer = torch.compile(pipeline.transformer, mode="max-autotune", fullgraph=True)

# ConsisID works well with long and well-described prompts and image contain clear face (e.g., preferably half-body or full-body).
prompt = "A woman adorned with a delicate flower crown, is standing amidst a field of gently swaying wildflowers. Her eyes sparkle with a serene gaze, and a faint smile graces her lips, suggesting a moment of peaceful contentment. The shot is framed from the waist up, highlighting the gentle breeze lightly tousling her hair. The background reveals an expansive meadow under a bright blue sky, capturing the tranquility of a sunny afternoon."
image = "https://github.com/PKU-YuanGroup/ConsisID/blob/main/asserts/example_images/1.png?raw=true"

id_cond, id_vit_hidden, image, face_kps = process_face_embeddings_infer(face_helper_1, face_clip_model, face_helper_2, eva_transform_mean, eva_transform_std, face_main_model, "cuda", torch.bfloat16, image, is_align_face=True)
is_kps = getattr(pipe.transformer.config, 'is_kps', False)
kps_cond = face_kps if is_kps else None

video = pipe(image=image, prompt=prompt, use_dynamic_cfg=False, id_vit_hidden=id_vit_hidden, id_cond=id_cond, kps_cond=kps_cond, generator=torch.Generator("cuda").manual_seed(42))
export_to_video(video.frames[0], "output.mp4", fps=8)
```

### Memory optimization

ConsisID requires about 37 GB of GPU memory to decode 49 frames (6 seconds of video at 8 FPS) with output resolution 720x480 (W x H), which makes it not possible to run on consumer GPUs or free-tier T4 Colab. The following memory optimizations could be used to reduce the memory footprint. For replication, you can refer to [this](https://gist.github.com/a-r-r-o-w/3959a03f15be5c9bd1fe545b09dfcc93) script.

- `pipe.enable_model_cpu_offload()`:
  - Without enabling cpu offloading, memory usage is `33 GB`
  - With enabling cpu offloading, memory usage is `19 GB`
- `pipe.enable_sequential_cpu_offload()`:
  - Similar to `enable_model_cpu_offload` but can significantly reduce memory usage at the cost of slow inference
  - When enabled, memory usage is under `4 GB`
- `pipe.vae.enable_tiling()`:
  - With enabling cpu offloading and tiling, memory usage is `11 GB`
- `pipe.vae.enable_slicing()`

### Quantized inference

[torchao](https://github.com/pytorch/ao) and [optimum-quanto](https://github.com/huggingface/optimum-quanto/) can be used to quantize the text encoder, transformer and VAE modules to lower the memory requirements. This makes it possible to run the model on a free-tier T4 Colab or lower VRAM GPUs!

It is also worth noting that torchao quantization is fully compatible with [torch.compile](/optimization/torch2.0#torchcompile), which allows for much faster inference speed. Additionally, models can be serialized and stored in a quantized datatype to save disk space with torchao. Find examples and benchmarks in the gists below.
- [torchao](https://gist.github.com/a-r-r-o-w/4d9732d17412888c885480c6521a9897)
- [quanto](https://gist.github.com/a-r-r-o-w/31be62828b00a9292821b85c1017effa)

## ConsisIDPipeline

[[autodoc]] ConsisIDPipeline

  - all
  - __call__

## ConsisIDPipelineOutput

[[autodoc]] pipelines.consisid.pipeline_output.ConsisIDPipelineOutput