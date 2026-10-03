<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 커스텀 파이프라인 불러오기

[[open-in-colab]]

커뮤니티 파이프라인은 논문에 명시된 원래의 구현체와 다른 형태로 구현된 모든 [`DiffusionPipeline`] 클래스를 의미합니다. (예를 들어, [`StableDiffusionControlNetPipeline`]는 ["Text-to-Image Generation with ControlNet Conditioning"](https://huggingface.co/papers/2302.05543) 해당) 이들은 추가 기능을 제공하거나 파이프라인의 원래 구현을 확장합니다.

[Speech to Image](https://github.com/huggingface/diffusers/tree/main/examples/community#speech-to-image) 또는 [Composable Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#composable-stable-diffusion) 과 같은 멋진 커뮤니티 파이프라인이 많이 있으며 [여기에서](https://github.com/huggingface/diffusers/tree/main/examples/community) 모든 공식 커뮤니티 파이프라인을 찾을 수 있습니다.

허브에서 커뮤니티 파이프라인을 로드하려면, 커뮤니티 파이프라인의 리포지토리 ID와 (파이프라인 가중치 및 구성 요소를 로드하려는) 모델의 리포지토리 ID를 인자로 전달해야 합니다. 예를 들어, 아래 예시에서는 `hf-internal-testing/diffusers-dummy-pipeline`에서 더미 파이프라인을 불러오고, `google/ddpm-cifar10-32`에서 파이프라인의 가중치와 컴포넌트들을 로드합니다.

> [!WARNING]
> 🔒 허깅 페이스 허브에서 커뮤니티 파이프라인을 불러오는 것은 곧 해당 코드가 안전하다고 신뢰하는 것입니다. 코드를 자동으로 불러오고 실행하기 앞서 반드시 온라인으로 해당 코드의 신뢰성을 검사하세요!

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline"
)
```

공식 커뮤니티 파이프라인을 불러오는 것은 비슷하지만, 공식 리포지토리 ID에서 가중치를 불러오는 것과 더불어 해당 파이프라인 내의 컴포넌트를 직접 지정하는 것 역시 가능합니다. 아래 예제를 보면 커뮤니티 [CLIP Guided Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion) 파이프라인을 로드할 때, 해당 파이프라인에서 사용할 `clip_model` 컴포넌트와 `feature_extractor` 컴포넌트를 직접 설정하는 것을 확인할 수 있습니다.

```py
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
)
```

커뮤니티 파이프라인에 대한 자세한 내용은 [커뮤니티 파이프라인](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/custom_pipeline_examples) 가이드를 살펴보세요. 커뮤니티 파이프라인 등록에 관심이 있는 경우 [커뮤니티 파이프라인에 기여하는 방법](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/contribute_pipeline)에 대한 가이드를 확인하세요 !