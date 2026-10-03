<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->



# 파이프라인, 모델, 스케줄러 불러오기

기본적으로 diffusion 모델은 다양한 컴포넌트들(모델, 토크나이저, 스케줄러) 간의 복잡한 상호작용을 기반으로 동작합니다. 디퓨저스(Diffusers)는 이러한 diffusion 모델을 보다 쉽고 간편한 API로 제공하는 것을 목표로 설계되었습니다. [`DiffusionPipeline`]은 diffusion 모델이 갖는 복잡성을 하나의 파이프라인 API로 통합하고, 동시에 이를 구성하는 각각의 컴포넌트들을 태스크에 맞춰 유연하게 커스터마이징할 수 있도록 지원하고 있습니다.

diffusion 모델의 훈련과 추론에 필요한 모든 것은 [`DiffusionPipeline.from_pretrained`] 메서드를 통해 접근할 수 있습니다. (이 말의 의미는 다음 단락에서 보다 자세하게 다뤄보도록 하겠습니다.)

이 문서에서는 설명할 내용은 다음과 같습니다.

* 허브를 통해 혹은 로컬로 파이프라인을 불러오는 법

* 파이프라인에 다른 컴포넌트들을 적용하는 법
* 오리지널 체크포인트가 아닌 variant를 불러오는 법  (variant란 기본으로 설정된 `fp32`가 아닌 다른  부동 소수점 타입(예: `fp16`)을 사용하거나 Non-EMA 가중치를 사용하는 체크포인트들을 의미합니다.)
* 모델과 스케줄러를 불러오는 법



## Diffusion 파이프라인

> [!TIP]
> 💡 [`DiffusionPipeline`] 클래스가 동작하는 방식에 보다 자세한 내용이 궁금하다면,  [DiffusionPipeline explained](#diffusionpipeline에-대해-알아보기) 섹션을 확인해보세요.

[`DiffusionPipeline`] 클래스는 diffusion 모델을 [허브](https://huggingface.co/models?library=diffusers)로부터 불러오는 가장 심플하면서 보편적인 방식입니다. [`DiffusionPipeline.from_pretrained`] 메서드는 적합한 파이프라인 클래스를 자동으로 탐지하고, 필요한 구성요소(configuration)와 가중치(weight) 파일들을 다운로드하고 캐싱한 다음, 해당 파이프라인 인스턴스를 반환합니다.

```python
from diffusers import DiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = DiffusionPipeline.from_pretrained(repo_id)
```

물론 [`DiffusionPipeline`] 클래스를 사용하지 않고, 명시적으로 직접 해당 파이프라인 클래스를 불러오는 것도 가능합니다. 아래 예시 코드는 위 예시와 동일한 인스턴스를 반환합니다.

```python
from diffusers import StableDiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(repo_id)
```

[CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)이나 [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 같은 체크포인트들의 경우, 하나 이상의 다양한 태스크에 활용될 수 있습니다. (예를 들어 위의 두 체크포인트의 경우, text-to-image와 image-to-image에 모두 활용될 수 있습니다.)  만약 이러한 체크포인트들을 기본 설정 태스크가 아닌 다른 태스크에 활용하고자 한다면, 해당 태스크에 대응되는 파이프라인(task-specific pipeline)을 사용해야 합니다.

```python
from diffusers import StableDiffusionImg2ImgPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(repo_id)
```



### 로컬 파이프라인

파이프라인을 로컬로 불러오고자 한다면, `git-lfs`를 사용하여 직접 체크포인트를 로컬 디스크에 다운로드 받아야 합니다. 아래의 명령어를 실행하면 `./stable-diffusion-v1-5`란 이름으로 폴더가 로컬디스크에 생성됩니다.

```bash
git lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
```

그런 다음 해당 로컬 경로를 [`~DiffusionPipeline.from_pretrained`] 메서드에 전달합니다.

```python
from diffusers import DiffusionPipeline

repo_id = "./stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id)
```

위의 예시코드처럼 만약 `repo_id`가 로컬 패스(local path)라면, [`~DiffusionPipeline.from_pretrained`] 메서드는 이를 자동으로 감지하여 허브에서 파일을 다운로드하지 않습니다. 만약 로컬 디스크에 저장된 파이프라인 체크포인트가 최신 버전이 아닐 경우에도, 최신 버전을 다운로드하지 않고 기존 로컬 디스크에 저장된 체크포인트를 사용한다는 것을 의미합니다.



### 파이프라인 내부의 컴포넌트 교체하기

파이프라인 내부의 컴포넌트들은 호환 가능한 다른 컴포넌트로 교체될 수 있습니다. 이와 같은 컴포넌트 교체가 중요한 이유는 다음과 같습니다.

- 어떤 스케줄러를 사용할 것인가는 생성속도와 생성품질 간의 트레이드오프를 정의하는 중요한 요소입니다.
- diffusion 모델 내부의 컴포넌트들은 일반적으로 각각 독립적으로 훈련되기 때문에, 더 좋은 성능을 보여주는 컴포넌트가 있다면 그걸로 교체하는 식으로 성능을 향상시킬 수 있습니다.
- 파인 튜닝 단계에서는 일반적으로 UNet 혹은 텍스트 인코더와 같은 일부 컴포넌트들만 훈련하게 됩니다.

어떤 스케줄러들이 호환가능한지는 `compatibles` 속성을 통해 확인할 수 있습니다.

```python
from diffusers import DiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id)
stable_diffusion.scheduler.compatibles
```

이번에는 [`SchedulerMixin.from_pretrained`] 메서드를 사용해서, 기존 기본 스케줄러였던 [`PNDMScheduler`]를 보다 우수한 성능의 [`EulerDiscreteScheduler`]로 바꿔봅시다. 스케줄러를 로드할 때는 `subfolder` 인자를 통해, 해당 파이프라인의 리포지토리에서 [스케줄러에 관한 하위폴더](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/scheduler)를  명시해주어야 합니다.

그 다음 새롭게 생성한 [`EulerDiscreteScheduler`] 인스턴스를 [`DiffusionPipeline`]의 `scheduler` 인자에 전달합니다.

```python
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, DPMSolverMultistepScheduler

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")

stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, scheduler=scheduler)
```

### 세이프티 체커

스테이블 diffusion과 같은 diffusion 모델들은 유해한 이미지를 생성할 수도 있습니다. 이를 예방하기 위해 디퓨저스는 생성된 이미지의 유해성을 판단하는 [세이프티 체커(safety checker)](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py) 기능을 지원하고 있습니다. 만약 세이프티 체커의 사용을 원하지 않는다면, `safety_checker` 인자에 `None`을 전달해주시면 됩니다.

```python
from diffusers import DiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
stable_diffusion = DiffusionPipeline.from_pretrained(repo_id, safety_checker=None)
```

### 컴포넌트 재사용

복수의 파이프라인에 동일한 모델이 반복적으로 사용한다면, 굳이 해당 모델의 동일한 가중치를 중복으로 RAM에 불러올 필요는 없을 것입니다.  [`~DiffusionPipeline.components`] 속성을 통해 파이프라인 내부의 컴포넌트들을 참조할 수 있는데, 이번 단락에서는 이를 통해 동일한 모델 가중치를 RAM에 중복으로 불러오는 것을 방지하는 법에 대해 알아보겠습니다.

```python
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id)

components = stable_diffusion_txt2img.components
```

그 다음 위 예시 코드에서 선언한 `components` 변수를 다른 파이프라인에 전달함으로써, 모델의 가중치를 중복으로 RAM에 로딩하지 않고, 동일한 컴포넌트를 재사용할 수 있습니다.

```python
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)
```

물론 각각의 컴포넌트들을 따로 따로 파이프라인에 전달할 수도 있습니다.  예를 들어 `stable_diffusion_txt2img` 파이프라인 안의 컴포넌트들 가운데서 세이프티 체커(`safety_checker`)와 피쳐 익스트랙터(`feature_extractor`)를 제외한 컴포넌트들만 `stable_diffusion_img2img` 파이프라인에서 재사용하는 방식 역시 가능합니다.

```python
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id)
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
    vae=stable_diffusion_txt2img.vae,
    text_encoder=stable_diffusion_txt2img.text_encoder,
    tokenizer=stable_diffusion_txt2img.tokenizer,
    unet=stable_diffusion_txt2img.unet,
    scheduler=stable_diffusion_txt2img.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
```

## Checkpoint variants

Variant란 일반적으로 다음과 같은 체크포인트들을 의미합니다.

-  `torch.float16`과 같이 정밀도는 더 낮지만, 용량 역시 더 작은 부동소수점 타입의 가중치를 사용하는 체크포인트. *(다만 이와 같은 variant의 경우, 추가적인 훈련과 CPU환경에서의 구동이 불가능합니다.)*
- Non-EMA 가중치를 사용하는 체크포인트. *(Non-EMA 가중치의 경우, 파인 튜닝 단계에서 사용하는 것이 권장되는데, 추론 단계에선 사용하지 않는 것이 권장됩니다.)*

> [!TIP]
> 💡 모델 구조는 동일하지만 서로 다른 학습 환경에서 서로 다른 데이터셋으로 학습된 체크포인트들이 있을 경우, 해당 체크포인트들은 variant 단계가 아닌 리포지토리 단계에서 분리되어 관리되어야 합니다. (즉, 해당 체크포인트들은 서로 다른 리포지토리에서 따로 관리되어야 합니다. 예시: [`stable-diffusion-v1-4`], [`stable-diffusion-v1-5`]).

| **checkpoint type** | **weight name**                     | **argument for loading weights** |
| ------------------- | ----------------------------------- | -------------------------------- |
| original            | diffusion_pytorch_model.bin         |                                  |
| floating point      | diffusion_pytorch_model.fp16.bin    | `variant`, `dtype`         |
| non-EMA             | diffusion_pytorch_model.non_ema.bin | `variant`                        |

variant를 로드할 때 2개의 중요한 argument가 있습니다.

* `dtype`은 불러올 체크포인트의 부동소수점을 정의합니다. 예를 들어 `dtype=torch.float16`을 명시함으로써 가중치의 부동소수점 타입을 `fl16`으로 변환할 수 있습니다. (만약 따로 설정하지 않을 경우, 기본값으로 `fp32` 타입의 가중치가 로딩됩니다.) 또한 `variant` 인자를 명시하지 않은 채로 체크포인트를 불러온 다음, 해당 체크포인트를 `dtype=torch.float16` 인자를 통해 `fp16` 타입으로 변환하는 것 역시 가능합니다. 이 경우 기본으로 설정된 `fp32` 가중치가 먼저 다운로드되고, 해당 가중치들을 불러온 다음 `fp16` 타입으로 변환하게 됩니다.
* `variant` 인자는 리포지토리에서 어떤 variant를 불러올 것인가를 정의합니다. 가령  [`diffusers/stable-diffusion-variants`](https://huggingface.co/diffusers/stable-diffusion-variants/tree/main/unet) 리포지토리로부터 `non_ema` 체크포인트를 불러오고자 한다면, `variant="non_ema"` 인자를 전달해야 합니다.

```python
from diffusers import DiffusionPipeline

# load fp16 variant
stable_diffusion = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16", dtype=torch.float16
)
# load non_ema variant
stable_diffusion = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="non_ema")
```

다른 부동소수점 타입의 가중치 혹은 non-EMA 가중치를 사용하는 체크포인트를 저장하기 위해서는, [`DiffusionPipeline.save_pretrained`] 메서드를 사용해야 하며, 이 때 `variant` 인자를 명시해줘야 합니다. 원래의 체크포인트와 동일한 폴더에 variant를 저장해야 하며, 이렇게 하면 동일한 폴더에서 오리지널 체크포인트과 variant를 모두 불러올 수 있습니다.

```python
from diffusers import DiffusionPipeline

# save as fp16 variant
stable_diffusion.save_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="fp16")
# save as non-ema variant
stable_diffusion.save_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", variant="non_ema")
```

만약 variant를 기존 폴더에 저장하지 않을 경우, `variant` 인자를 반드시 명시해야 합니다. 그렇게 하지 않을 경우 원래의 오리지널 체크포인트를 찾을 수 없게 되기 때문에 에러가 발생합니다.

```python
# 👎 this won't work
stable_diffusion = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5", dtype=torch.float16)
# 👍 this works
stable_diffusion = DiffusionPipeline.from_pretrained(
    "./stable-diffusion-v1-5", variant="fp16", dtype=torch.float16
)
```

### 모델 불러오기

모델들은 [`ModelMixin.from_pretrained`] 메서드를 통해 불러올 수 있습니다. 해당 메서드는 최신 버전의 모델 가중치 파일과 설정 파일(configurations)을 다운로드하고 캐싱합니다. 만약 이러한 파일들이 최신 버전으로 로컬 캐시에 저장되어 있다면, [`ModelMixin.from_pretrained`]는 굳이 해당 파일들을 다시 다운로드하지 않으며, 그저 캐시에 있는 최신 파일들을 재사용합니다.

모델은 `subfolder` 인자에 명시된 하위 폴더로부터 로드됩니다. 예를 들어 `stable-diffusion-v1-5/stable-diffusion-v1-5`의 UNet 모델의 가중치는 [`unet`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main/unet) 폴더에 저장되어 있습니다.

```python
from diffusers import UNet2DConditionModel

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
model = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet")
```

혹은 [해당 모델의 리포지토리](https://huggingface.co/google/ddpm-cifar10-32/tree/main)로부터 다이렉트로 가져오는 것 역시 가능합니다.

```python
from diffusers import UNet2DModel

repo_id = "google/ddpm-cifar10-32"
model = UNet2DModel.from_pretrained(repo_id)
```

또한 앞서 봤던 `variant` 인자를 명시함으로써, Non-EMA나 `fp16`의 가중치를 가져오는 것 역시 가능합니다.

```python
from diffusers import UNet2DConditionModel

model = UNet2DConditionModel.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="unet", variant="non-ema")
model.save_pretrained("./local-unet", variant="non-ema")
```

### 스케줄러

스케줄러들은 [`SchedulerMixin.from_pretrained`] 메서드를 통해 불러올 수 있습니다. 모델과 달리 스케줄러는 별도의 가중치를 갖지 않으며, 따라서 당연히 별도의 학습과정을 요구하지 않습니다. 이러한 스케줄러들은 (해당 스케줄러 하위폴더의) configration 파일을 통해 정의됩니다.

여러개의 스케줄러를 불러온다고 해서 많은 메모리를 소모하는 것은 아니며, 다양한 스케줄러들에 동일한 스케줄러 configration을  적용하는 것 역시 가능합니다. 다음 예시 코드에서 불러오는 스케줄러들은 모두 [`StableDiffusionPipeline`]과 호환되는데, 이는 곧 해당 스케줄러들에 동일한 스케줄러 configration 파일을 적용할 수 있음을 의미합니다.

```python
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
dpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

# replace `dpm` with any of `ddpm`, `ddim`, `pndm`, `lms`, `euler_anc`, `euler`
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=dpm)
```

### DiffusionPipeline에 대해 알아보기

클래스 메서드로서  [`DiffusionPipeline.from_pretrained`]은 2가지를 담당합니다.

- 첫째로, `from_pretrained` 메서드는 최신 버전의 파이프라인을 다운로드하고, 캐시에 저장합니다. 이미 로컬 캐시에 최신 버전의 파이프라인이 저장되어 있다면, [`DiffusionPipeline.from_pretrained`]은 해당 파일들을 다시 다운로드하지 않고, 로컬 캐시에 저장되어 있는 파이프라인을 불러옵니다.
-  `model_index.json` 파일을 통해 체크포인트에 대응되는 적합한 파이프라인 클래스로 불러옵니다.

파이프라인의 폴더 구조는 해당 파이프라인 클래스의 구조와 직접적으로 일치합니다. 예를 들어 [`StableDiffusionPipeline`] 클래스는 [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) 리포지토리와 대응되는 구조를 갖습니다.

```python
from diffusers import DiffusionPipeline

repo_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id)
print(pipeline)
```

위의 코드 출력 결과를 확인해보면, `pipeline`은 [`StableDiffusionPipeline`]의 인스턴스이며, 다음과 같이 총 7개의 컴포넌트로 구성된다는 것을 알 수 있습니다.

- `"feature_extractor"`: [`~transformers.CLIPImageProcessor`]의 인스턴스
- `"safety_checker"`: 유해한 컨텐츠를 스크리닝하기 위한 [컴포넌트](https://github.com/huggingface/diffusers/blob/e55687e1e15407f60f32242027b7bb8170e58266/src/diffusers/pipelines/stable_diffusion/safety_checker.py#L32)
- `"scheduler"`: [`PNDMScheduler`]의 인스턴스
- `"text_encoder"`: [`~transformers.CLIPTextModel`]의 인스턴스
- `"tokenizer"`: a [`~transformers.CLIPTokenizer`]의 인스턴스
- `"unet"`: [`UNet2DConditionModel`]의 인스턴스
- `"vae"` [`AutoencoderKL`]의 인스턴스

```json
StableDiffusionPipeline {
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

파이프라인 인스턴스의 컴포넌트들을  [`stable-diffusion-v1-5/stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)의 폴더 구조와 비교해볼 경우, 각각의 컴포넌트마다 별도의 폴더가 있음을 확인할 수 있습니다.

```
.
├── feature_extractor
│   └── preprocessor_config.json
├── model_index.json
├── safety_checker
│   ├── config.json
│   └── pytorch_model.bin
├── scheduler
│   └── scheduler_config.json
├── text_encoder
│   ├── config.json
│   └── pytorch_model.bin
├── tokenizer
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── unet
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
└── vae
    ├── config.json
    ├── diffusion_pytorch_model.bin
```

또한 각각의 컴포넌트들을 파이프라인 인스턴스의 속성으로써 참조할 수 있습니다.

```py
pipeline.tokenizer
```

```python
CLIPTokenizer(
    name_or_path="/root/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819/tokenizer",
    vocab_size=49408,
    model_max_length=77,
    is_fast=False,
    padding_side="right",
    truncation_side="right",
    special_tokens={
        "bos_token": AddedToken("<|startoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "eos_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "unk_token": AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=True),
        "pad_token": "<|endoftext|>",
    },
)
```

모든 파이프라인은 `model_index.json` 파일을 통해 [`DiffusionPipeline`]에 다음과 같은 정보를 전달합니다.

- `_class_name` 는 어떤 파이프라인 클래스를 사용해야 하는지에 대해 알려줍니다.
- `_diffusers_version`는 어떤 버전의 디퓨저스로 파이프라인 안의 모델들이 만들어졌는지를 알려줍니다.
- 그 다음은 각각의 컴포넌트들이 어떤 라이브러리의 어떤 클래스로 만들어졌는지에 대해 알려줍니다. (아래 예시에서 `"feature_extractor" : ["transformers", "CLIPImageProcessor"]`의 경우, `feature_extractor` 컴포넌트는 `transformers` 라이브러리의 `CLIPImageProcessor` 클래스를 통해 만들어졌다는 것을 의미합니다.)

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "feature_extractor": [
    "transformers",
    "CLIPImageProcessor"
  ],
  "safety_checker": [
    "stable_diffusion",
    "StableDiffusionSafetyChecker"
  ],
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  "text_encoder": [
    "transformers",
    "CLIPTextModel"
  ],
  "tokenizer": [
    "transformers",
    "CLIPTokenizer"
  ],
  "unet": [
    "diffusers",
    "UNet2DConditionModel"
  ],
  "vae": [
    "diffusers",
    "AutoencoderKL"
  ]
}
```

