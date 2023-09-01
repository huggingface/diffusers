<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 커뮤니티 파이프라인에 기여하는 방법

<Tip>

💡 모든 사람이 속도 저하 없이 쉽게 작업을 공유할 수 있도록 커뮤니티 파이프라인을 추가하는 이유에 대한 자세한 내용은 GitHub 이슈 [#841](https://github.com/huggingface/diffusers/issues/841)를 참조하세요. 

</Tip>

커뮤니티 파이프라인을 사용하면 [`DiffusionPipeline`] 위에 원하는 추가 기능을 추가할 수 있습니다. `DiffusionPipeline` 위에 구축할 때의 가장 큰 장점은 누구나 인수를 하나만 추가하면 파이프라인을 로드하고 사용할 수 있어 커뮤니티가 매우 쉽게 접근할 수 있다는 것입니다.

이번 가이드에서는 커뮤니티 파이프라인을 생성하는 방법과 작동 원리를 설명합니다.
간단하게 설명하기 위해 `UNet`이 단일 forward pass를 수행하고 스케줄러를 한 번 호출하는 "one-step" 파이프라인을 만들겠습니다.

## 파이프라인 초기화

커뮤니티 파이프라인을 위한 `one_step_unet.py` 파일을 생성하는 것으로 시작합니다. 이 파일에서, Hub에서 모델 가중치와 스케줄러 구성을 로드할 수 있도록 [`DiffusionPipeline`]을 상속하는 파이프라인 클래스를 생성합니다. one-step 파이프라인에는 `UNet`과 스케줄러가 필요하므로 이를 `__init__` 함수에 인수로 추가해야합니다:

```python
from diffusers import DiffusionPipeline
import torch


class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
```

파이프라인과 그 구성요소(`unet` and `scheduler`)를 [`~DiffusionPipeline.save_pretrained`]으로 저장할 수 있도록 하려면 `register_modules` 함수에 추가하세요:

```diff
  from diffusers import DiffusionPipeline
  import torch

  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

+         self.register_modules(unet=unet, scheduler=scheduler)
```

이제 '초기화' 단계가 완료되었으니 forward pass로 이동할 수 있습니다! 🔥 

## Forward pass 정의

Forward pass 에서는(`__call__`로 정의하는 것이 좋습니다) 원하는 기능을 추가할 수 있는 완전한 창작 자유가 있습니다. 우리의 놀라운 one-step 파이프라인의 경우, 임의의 이미지를 생성하고 `timestep=1`을 설정하여 `unet`과 `scheduler`를 한 번만 호출합니다:

```diff
  from diffusers import DiffusionPipeline
  import torch


  class UnetSchedulerOneForwardPipeline(DiffusionPipeline):
      def __init__(self, unet, scheduler):
          super().__init__()

          self.register_modules(unet=unet, scheduler=scheduler)

+     def __call__(self):
+         image = torch.randn(
+             (1, self.unet.config.in_channels, self.unet.config.sample_size, self.unet.config.sample_size),
+         )
+         timestep = 1

+         model_output = self.unet(image, timestep).sample
+         scheduler_output = self.scheduler.step(model_output, timestep, image).prev_sample

+         return scheduler_output
```

끝났습니다! 🚀 이제 이 파이프라인에 `unet`과 `scheduler`를 전달하여 실행할 수 있습니다:

```python
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler()
unet = UNet2DModel()

pipeline = UnetSchedulerOneForwardPipeline(unet=unet, scheduler=scheduler)

output = pipeline()
```

하지만 파이프라인 구조가 동일한 경우 기존 가중치를 파이프라인에 로드할 수 있다는 장점이 있습니다. 예를 들어 one-step 파이프라인에 [`google/ddpm-cifar10-32`](https://huggingface.co/google/ddpm-cifar10-32) 가중치를 로드할 수 있습니다:

```python
pipeline = UnetSchedulerOneForwardPipeline.from_pretrained("google/ddpm-cifar10-32")

output = pipeline()
```

## 파이프라인 공유

🧨Diffusers [리포지토리](https://github.com/huggingface/diffusers)에서 Pull Request를 열어 [examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) 하위 폴더에 `one_step_unet.py`의 멋진 파이프라인을 추가하세요.

병합이 되면, `diffusers >= 0.4.0`이 설치된 사용자라면 누구나 `custom_pipeline` 인수에 지정하여 이 파이프라인을 마술처럼 🪄 사용할 수 있습니다:

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="one_step_unet")
pipe()
```

커뮤니티 파이프라인을 공유하는 또 다른 방법은 Hub 에서 선호하는 [모델 리포지토리](https://huggingface.co/docs/hub/models-uploading)에 직접  `one_step_unet.py` 파일을 업로드하는 것입니다. `one_step_unet.py` 파일을 지정하는 대신 모델 저장소 id를 `custom_pipeline` 인수에 전달하세요:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="stevhliu/one_step_unet")
```

다음 표에서 두 가지 공유 워크플로우를 비교하여 자신에게 가장 적합한 옵션을 결정하는 데 도움이 되는 정보를 확인하세요:

|                | GitHub 커뮤니티 파이프라인                                                                                        | HF Hub 커뮤니티 파이프라인                                                                 |
|----------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 사용법          | 동일                                                                                                             | 동일                                                                                      |
| 리뷰 과정 | 병합하기 전에 GitHub에서 Pull Request를 열고 Diffusers 팀의 검토 과정을 거칩니다. 속도가 느릴 수 있습니다. | 검토 없이 Hub 저장소에 바로 업로드합니다. 가장 빠른 워크플로우 입니다. |
| 가시성     | 공식 Diffusers 저장소 및 문서에 포함되어 있습니다.                                                  | HF 허브 프로필에 포함되며 가시성을 확보하기 위해 자신의 사용량/프로모션에 의존합니다. |

<Tip>

💡 커뮤니티 파이프라인 파일에 원하는 패키지를 사용할 수 있습니다. 사용자가 패키지를 설치하기만 하면 모든 것이 정상적으로 작동합니다. 파이프라인이 자동으로 감지되므로 `DiffusionPipeline`에서 상속하는 파이프라인 클래스가 하나만 있는지 확인하세요.

</Tip>

## 커뮤니티 파이프라인은 어떻게 작동하나요?

커뮤니티 파이프라인은 [`DiffusionPipeline`]을 상속하는 클래스입니다:

- [`custom_pipeline`] 인수로 로드할 수 있습니다.
- 모델 가중치 및 스케줄러 구성은 [`pretrained_model_name_or_path`]에서 로드됩니다.
- 커뮤니티 파이프라인에서 기능을 구현하는 코드는 `pipeline.py` 파일에 정의되어 있습니다.

공식 저장소에서 모든 파이프라인 구성 요소 가중치를 로드할 수 없는 경우가 있습니다. 이 경우 다른 구성 요소는 파이프라인에 직접 전달해야 합니다:

```python
from diffusers import DiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPModel

model_id = "CompVis/stable-diffusion-v1-4"
clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPFeatureExtractor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float16)

pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    scheduler=scheduler,
    torch_dtype=torch.float16,
)
```

커뮤니티 파이프라인의 마법은 다음 코드에 담겨 있습니다. 이 코드를 통해 커뮤니티 파이프라인을 GitHub 또는 Hub에서 로드할 수 있으며, 모든 🧨 Diffusers 패키지에서 사용할 수 있습니다.

```python
# 2. 파이프라인 클래스를 로드합니다. 사용자 지정 모듈을 사용하는 경우 Hub에서 로드합니다
# 명시적 클래스에서 로드하는 경우, 이를 사용해 보겠습니다.
if custom_pipeline is not None:
    pipeline_class = get_class_from_dynamic_module(
        custom_pipeline, module_file=CUSTOM_PIPELINE_FILE_NAME, cache_dir=custom_pipeline
    )
elif cls != DiffusionPipeline:
    pipeline_class = cls
else:
    diffusers_module = importlib.import_module(cls.__module__.split(".")[0])
    pipeline_class = getattr(diffusers_module, config_dict["_class_name"])
```
