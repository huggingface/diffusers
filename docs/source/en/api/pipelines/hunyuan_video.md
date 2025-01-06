<!-- Copyright 2024 The HuggingFace Team. All rights reserved.
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
# limitations under the License. -->

# HunyuanVideo

[HunyuanVideo](https://www.arxiv.org/abs/2412.03603) by Tencent.

*Recent advancements in video generation have significantly impacted daily life for both individuals and industries. However, the leading video generation models remain closed-source, resulting in a notable performance gap between industry capabilities and those available to the public. In this report, we introduce HunyuanVideo, an innovative open-source video foundation model that demonstrates performance in video generation comparable to, or even surpassing, that of leading closed-source models. HunyuanVideo encompasses a comprehensive framework that integrates several key elements, including data curation, advanced architectural design, progressive model scaling and training, and an efficient infrastructure tailored for large-scale model training and inference. As a result, we successfully trained a video generative model with over 13 billion parameters, making it the largest among all open-source models. We conducted extensive experiments and implemented a series of targeted designs to ensure high visual quality, motion dynamics, text-video alignment, and advanced filming techniques. According to evaluations by professionals, HunyuanVideo outperforms previous state-of-the-art models, including Runway Gen-3, Luma 1.6, and three top-performing Chinese video generative models. By releasing the code for the foundation model and its applications, we aim to bridge the gap between closed-source and open-source communities. This initiative will empower individuals within the community to experiment with their ideas, fostering a more dynamic and vibrant video generation ecosystem. The code is publicly available at [this https URL](https://github.com/Tencent/HunyuanVideo).*

<Tip>

Make sure to check out the Schedulers [guide](../../using-diffusers/schedulers) to learn how to explore the tradeoff between scheduler speed and quality, and see the [reuse components across pipelines](../../using-diffusers/loading#reuse-a-pipeline) section to learn how to efficiently load the same components into multiple pipelines.

</Tip>

Recommendations for inference:
- Both text encoders should be in `torch.float16`.
- Transformer should be in `torch.bfloat16`.
- VAE should be in `torch.float16`.
- `num_frames` should be of the form `4 * k + 1`, for example `49` or `129`.
- For smaller resolution videos, try lower values of `shift` (between `2.0` to `5.0`) in the [Scheduler](https://huggingface.co/docs/diffusers/main/en/api/schedulers/flow_match_euler_discrete#diffusers.FlowMatchEulerDiscreteScheduler.shift). For larger resolution images, try higher values (between `7.0` and `12.0`). The default value is `7.0` for HunyuanVideo.
- For more information about supported resolutions and other details, please refer to the original repository [here](https://github.com/Tencent/HunyuanVideo/).

## Quantization

Quantization helps reduce the memory requirements of very large models by storing model weights in a lower precision data type. However, quantization may have varying impact on video quality depending on the video model.

Refer to the [Quantization](../../quantization/overview) overview to learn more about supported quantization backends and selecting a quantization backend that supports your use case. The example below demonstrates how to load a quantized [`HunyuanVideoPipeline`] for inference with bitsandbytes.

```py
import torch
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig, HunyuanVideoTransformer3DModel, HunyuanVideoPipeline
from diffusers.utils import export_to_video

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = HunyuanVideoTransformer3DModel.from_pretrained(
    "tencent/HunyuanVideo",
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.float16,
)

pipeline = HunyuanVideoPipeline.from_pretrained(
    "tencent/HunyuanVideo",
    transformer=transformer_8bit,
    torch_dtype=torch.float16,
    device_map="balanced",
)

prompt = "A cat walks on the grass, realistic style."
video = pipeline(prompt=prompt, num_frames=61, num_inference_steps=30).frames[0]
export_to_video(video, "cat.mp4", fps=15)
```

## HunyuanVideoPipeline

[[autodoc]] HunyuanVideoPipeline
  - all
  - __call__

## HunyuanVideoPipelineOutput

[[autodoc]] pipelines.hunyuan_video.pipeline_output.HunyuanVideoPipelineOutput
