<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->
[[open-in-colab]]

# 훑어보기

Diffusion 모델은 이미지나 오디오와 같은 관심 샘플들을 생성하기 위해 랜덤 가우시안 노이즈를 단계별로 제거하도록 학습됩니다. 이로 인해 생성 AI에 대한 관심이 매우 높아졌으며, 인터넷에서 diffusion 생성 이미지의 예를 본 적이 있을 것입니다. 🧨 Diffusers는 누구나 diffusion 모델들을 널리 이용할 수 있도록 하기 위한 라이브러리입니다.

개발자든 일반 사용자든 이 훑어보기를 통해 🧨 diffusers를 소개하고 빠르게 생성할 수 있도록 도와드립니다! 알아야 할 라이브러리의 주요 구성 요소는 크게 세 가지입니다:

* [`DiffusionPipeline`]은 추론을 위해 사전 학습된 diffusion 모델에서 샘플을 빠르게 생성하도록 설계된 높은 수준의 엔드투엔드 클래스입니다.
* Diffusion 시스템 생성을 위한 빌딩 블록으로 사용할 수 있는 널리 사용되는 사전 학습된 [model](./api/models) 아키텍처 및 모듈.
* 다양한 [schedulers](./api/schedulers/overview) - 학습을 위해 노이즈를 추가하는 방법과 추론 중에 노이즈 제거된 이미지를 생성하는 방법을 제어하는 알고리즘입니다.

훑어보기에서는 추론을 위해 [`DiffusionPipeline`]을 사용하는 방법을 보여준 다음, 모델과 스케줄러를 결합하여 [`DiffusionPipeline`] 내부에서 일어나는 일을 복제하는 방법을 안내합니다.

<Tip>

훑어보기는 간결한 버전의 🧨 Diffusers 소개로서 [노트북](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/diffusers_intro.ipynb) 빠르게 시작할 수 있도록 도와드립니다. 디퓨저의 목표, 디자인 철학, 핵심 API에 대한 추가 세부 정보를 자세히 알아보려면 노트북을 확인하세요!

</Tip>

시작하기 전에 필요한 라이브러리가 모두 설치되어 있는지 확인하세요:

```py
# 주석 풀어서 Colab에 필요한 라이브러리 설치하기.
#!pip install --upgrade diffusers accelerate transformers
```

- [🤗 Accelerate](https://huggingface.co/docs/accelerate/index)는 추론 및 학습을 위한 모델 로딩 속도를 높여줍니다.
- [🤗 Transformers](https://huggingface.co/docs/transformers/index)는 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)과 같이 가장 많이 사용되는 diffusion 모델을 실행하는 데 필요합니다.

## DiffusionPipeline

[`DiffusionPipeline`] 은 추론을 위해 사전 학습된 diffusion 시스템을 사용하는 가장 쉬운 방법입니다. 모델과 스케줄러를 포함하는 엔드 투 엔드 시스템입니다. 다양한 작업에 [`DiffusionPipeline`]을 바로 사용할 수 있습니다. 아래 표에서 지원되는 몇 가지 작업을 살펴보고, 지원되는 작업의 전체 목록은 [🧨 Diffusers Summary](./api/pipelines/overview#diffusers-summary) 표에서 확인할 수 있습니다.

| **Task**                     | **Description**                                                                                              | **Pipeline**
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|
| Unconditional Image Generation          | generate an image from Gaussian noise | [unconditional_image_generation](./using-diffusers/unconditional_image_generation) |
| Text-Guided Image Generation | generate an image given a text prompt | [conditional_image_generation](./using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation     | adapt an image guided by a text prompt | [img2img](./using-diffusers/img2img) |
| Text-Guided Image-Inpainting          | fill the masked part of an image given the image, the mask and a text prompt | [inpaint](./using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | adapt parts of an image guided by a text prompt while preserving structure via depth estimation | [depth2img](./using-diffusers/depth2img) |

먼저 [`DiffusionPipeline`]의 인스턴스를 생성하고 다운로드할 파이프라인 체크포인트를 지정합니다.
허깅페이스 허브에 저장된 모든 [checkpoint](https://huggingface.co/models?library=diffusers&sort=downloads)에 대해 [`DiffusionPipeline`]을 사용할 수 있습니다.
이 훑어보기에서는 text-to-image 생성을 위한 [`stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) 체크포인트를 로드합니다.

<Tip warning={true}>

[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion) 모델의 경우, 모델을 실행하기 전에 [라이선스](https://huggingface.co/spaces/CompVis/stable-diffusion-license)를 먼저 주의 깊게 읽어주세요. 🧨 Diffusers는 불쾌하거나 유해한 콘텐츠를 방지하기 위해 [`safety_checker`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/safety_checker.py)를 구현하고 있지만, 모델의 향상된 이미지 생성 기능으로 인해 여전히 잠재적으로 유해한 콘텐츠가 생성될 수 있습니다.

</Tip>

[`~DiffusionPipeline.from_pretrained`] 방법으로 모델 로드하기:

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
```

The [`DiffusionPipeline`]은 모든 모델링, 토큰화, 스케줄링 컴포넌트를 다운로드하고 캐시합니다. Stable Diffusion Pipeline은 무엇보다도 [`UNet2DConditionModel`]과 [`PNDMScheduler`]로 구성되어 있음을 알 수 있습니다:

```py
>>> pipeline
StableDiffusionPipeline {
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.13.1",
  ...,
  "scheduler": [
    "diffusers",
    "PNDMScheduler"
  ],
  ...,
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

이 모델은 약 14억 개의 파라미터로 구성되어 있으므로 GPU에서 파이프라인을 실행할 것을 강력히 권장합니다.
PyTorch에서와 마찬가지로 제너레이터 객체를 GPU로 이동할 수 있습니다:

```python
>>> pipeline.to("cuda")
```

이제 `파이프라인`에 텍스트 프롬프트를 전달하여 이미지를 생성한 다음 노이즈가 제거된 이미지에 액세스할 수 있습니다. 기본적으로 이미지 출력은 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) 객체로 감싸집니다.

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
>>> image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/image_of_squirrel_painting.png"/>
</div>

`save`를 호출하여 이미지를 저장합니다:

```python
>>> image.save("image_of_squirrel_painting.png")
```

### 로컬 파이프라인

파이프라인을 로컬에서 사용할 수도 있습니다. 유일한 차이점은 가중치를 먼저 다운로드해야 한다는 점입니다:

```bash
!git lfs install
!git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

그런 다음 저장된 가중치를 파이프라인에 로드합니다:

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
```

이제 위 섹션에서와 같이 파이프라인을 실행할 수 있습니다.

### 스케줄러 교체

스케줄러마다 노이즈 제거 속도와 품질이 서로 다릅니다. 자신에게 가장 적합한 스케줄러를 찾는 가장 좋은 방법은 직접 사용해 보는 것입니다! 🧨 Diffusers의 주요 기능 중 하나는 스케줄러 간에 쉽게 전환이 가능하다는 것입니다. 예를 들어, 기본 스케줄러인 [`PNDMScheduler`]를 [`EulerDiscreteScheduler`]로 바꾸려면, [`~diffusers.ConfigMixin.from_config`] 메서드를 사용하여 로드하세요:

```py
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

새 스케줄러로 이미지를 생성해보고 어떤 차이가 있는지 확인해 보세요!

다음 섹션에서는 모델과 스케줄러라는 [`DiffusionPipeline`]을 구성하는 컴포넌트를 자세히 살펴보고 이러한 컴포넌트를 사용하여 고양이 이미지를 생성하는 방법을 배워보겠습니다.

## 모델

대부분의 모델은 노이즈가 있는 샘플을 가져와 각 시간 간격마다 노이즈가 적은 이미지와 입력 이미지 사이의 차이인 *노이즈 잔차*(다른 모델은 이전 샘플을 직접 예측하거나 속도 또는 [`v-prediction`](https://github.com/huggingface/diffusers/blob/5e5ce13e2f89ac45a0066cb3f369462a3cf1d9ef/src/diffusers/schedulers/scheduling_ddim.py#L110)을 예측하는 학습을 합니다)을 예측합니다. 모델을 믹스 앤 매치하여 다른 diffusion 시스템을 만들 수 있습니다.

모델은 [`~ModelMixin.from_pretrained`] 메서드로 시작되며, 이 메서드는 모델 가중치를 로컬에 캐시하여 다음에 모델을 로드할 때 더 빠르게 로드할 수 있습니다. 훑어보기에서는 고양이 이미지에 대해 학습된 체크포인트가 있는 기본적인 unconditional 이미지 생성 모델인 [`UNet2DModel`]을 로드합니다:

```py
>>> from diffusers import UNet2DModel

>>> repo_id = "google/ddpm-cat-256"
>>> model = UNet2DModel.from_pretrained(repo_id)
```

모델 매개변수에 액세스하려면 `model.config`를 호출합니다:

```py
>>> model.config
```

모델 구성은 🧊 고정된 🧊 딕셔너리로, 모델이 생성된 후에는 해당 매개 변수들을 변경할 수 없습니다. 이는 의도적인 것으로, 처음에 모델 아키텍처를 정의하는 데 사용된 매개변수는 동일하게 유지하면서 다른 매개변수는 추론 중에 조정할 수 있도록 하기 위한 것입니다.

가장 중요한 매개변수들은 다음과 같습니다:

* `sample_size`: 입력 샘플의 높이 및 너비 치수입니다.
* `in_channels`: 입력 샘플의 입력 채널 수입니다.
* `down_block_types` 및 `up_block_types`: UNet 아키텍처를 생성하는 데 사용되는 다운 및 업샘플링 블록의 유형.
* `block_out_channels`: 다운샘플링 블록의 출력 채널 수. 업샘플링 블록의 입력 채널 수에 역순으로 사용되기도 합니다.
* `layers_per_block`: 각 UNet 블록에 존재하는 ResNet 블록의 수입니다.

추론에 모델을 사용하려면 랜덤 가우시안 노이즈로 이미지 모양을 만듭니다. 모델이 여러 개의 무작위 노이즈를 수신할 수 있으므로 'batch' 축, 입력 채널 수에 해당하는 'channel' 축, 이미지의 높이와 너비를 나타내는 'sample_size' 축이 있어야 합니다:

```py
>>> import torch

>>> torch.manual_seed(0)

>>> noisy_sample = torch.randn(1, model.config.in_channels, model.config.sample_size, model.config.sample_size)
>>> noisy_sample.shape
torch.Size([1, 3, 256, 256])
```

추론을 위해 모델에 노이즈가 있는 이미지와 `timestep`을 전달합니다. 'timestep'은 입력 이미지의 노이즈 정도를 나타내며, 시작 부분에 더 많은 노이즈가 있고 끝 부분에 더 적은 노이즈가 있습니다. 이를 통해 모델이 diffusion 과정에서 시작 또는 끝에 더 가까운 위치를 결정할 수 있습니다. `sample` 메서드를 사용하여 모델 출력을 얻습니다:

```py
>>> with torch.no_grad():
...     noisy_residual = model(sample=noisy_sample, timestep=2).sample
```

하지만 실제 예를 생성하려면 노이즈 제거 프로세스를 안내할 스케줄러가 필요합니다. 다음 섹션에서는 모델을 스케줄러와 결합하는 방법에 대해 알아봅니다.

## 스케줄러

스케줄러는 모델 출력이 주어졌을 때 노이즈가 많은 샘플에서 노이즈가 적은 샘플로 전환하는 것을 관리합니다 - 이 경우 'noisy_residual'.

<Tip>

🧨 Diffusers는 Diffusion 시스템을 구축하기 위한 툴박스입니다. [`DiffusionPipeline`]을 사용하면 미리 만들어진 Diffusion 시스템을 편리하게 시작할 수 있지만, 모델과 스케줄러 구성 요소를 개별적으로 선택하여 사용자 지정 Diffusion 시스템을 구축할 수도 있습니다.

</Tip>

훑어보기의 경우, [`~diffusers.ConfigMixin.from_config`] 메서드를 사용하여 [`DDPMScheduler`]를 인스턴스화합니다:

```py
>>> from diffusers import DDPMScheduler

>>> scheduler = DDPMScheduler.from_config(repo_id)
>>> scheduler
DDPMScheduler {
  "_class_name": "DDPMScheduler",
  "_diffusers_version": "0.13.1",
  "beta_end": 0.02,
  "beta_schedule": "linear",
  "beta_start": 0.0001,
  "clip_sample": true,
  "clip_sample_range": 1.0,
  "num_train_timesteps": 1000,
  "prediction_type": "epsilon",
  "trained_betas": null,
  "variance_type": "fixed_small"
}
```

<Tip>

💡 스케줄러가 구성에서 어떻게 인스턴스화되는지 주목하세요. 모델과 달리 스케줄러에는 학습 가능한 가중치가 없으며 매개변수도 없습니다!

</Tip>

가장 중요한 매개변수는 다음과 같습니다:

* `num_train_timesteps`: 노이즈 제거 프로세스의 길이, 즉 랜덤 가우스 노이즈를 데이터 샘플로 처리하는 데 필요한 타임스텝 수입니다.
* `beta_schedule`: 추론 및 학습에 사용할 노이즈 스케줄 유형입니다.
* `beta_start` 및 `beta_end`: 노이즈 스케줄의 시작 및 종료 노이즈 값입니다.

노이즈가 약간 적은 이미지를 예측하려면 스케줄러의 [`~diffusers.DDPMScheduler.step`] 메서드에 모델 출력, `timestep`, 현재 `sample`을 전달하세요.

```py
>>> less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample
>>> less_noisy_sample.shape
```

`less_noisy_sample`을 다음 `timestep`으로 넘기면 노이즈가 더 줄어듭니다! 이제 이 모든 것을 한데 모아 전체 노이즈 제거 과정을 시각화해 보겠습니다.

먼저 노이즈 제거된 이미지를 후처리하여 `PIL.Image`로 표시하는 함수를 만듭니다:

```py
>>> import PIL.Image
>>> import numpy as np


>>> def display_sample(sample, i):
...     image_processed = sample.cpu().permute(0, 2, 3, 1)
...     image_processed = (image_processed + 1.0) * 127.5
...     image_processed = image_processed.numpy().astype(np.uint8)

...     image_pil = PIL.Image.fromarray(image_processed[0])
...     display(f"Image at step {i}")
...     display(image_pil)
```

노이즈 제거 프로세스의 속도를 높이려면 입력과 모델을 GPU로 옮기세요:

```py
>>> model.to("cuda")
>>> noisy_sample = noisy_sample.to("cuda")
```

이제 노이즈가 적은 샘플의 잔차를 예측하고 스케줄러로 노이즈가 적은 샘플을 계산하는 노이즈 제거 루프를 생성합니다:

```py
>>> import tqdm

>>> sample = noisy_sample

>>> for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
...     # 1. predict noise residual
...     with torch.no_grad():
...         residual = model(sample, t).sample

...     # 2. compute less noisy image and set x_t -> x_t-1
...     sample = scheduler.step(residual, t, sample).prev_sample

...     # 3. optionally look at image
...     if (i + 1) % 50 == 0:
...         display_sample(sample, i + 1)
```

가만히 앉아서 고양이가 소음으로만 생성되는 것을 지켜보세요!😻

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/diffusion-quicktour.png"/>
</div>

## 다음 단계

이번 훑어보기에서 🧨 Diffusers로 멋진 이미지를 만들어 보셨기를 바랍니다! 다음 단계로 넘어가세요:

* [training](./tutorials/basic_training) 튜토리얼에서 모델을 학습하거나 파인튜닝하여 나만의 이미지를 생성할 수 있습니다.
* 다양한 사용 사례는 공식 및 커뮤니티 [학습 또는 파인튜닝 스크립트](https://github.com/huggingface/diffusers/tree/main/examples#-diffusers-examples) 예시를 참조하세요.
* 스케줄러 로드, 액세스, 변경 및 비교에 대한 자세한 내용은 [다른 스케줄러 사용](./using-diffusers/schedulers) 가이드에서 확인하세요.
* [Stable Diffusion](./stable_diffusion) 가이드에서 프롬프트 엔지니어링, 속도 및 메모리 최적화, 고품질 이미지 생성을 위한 팁과 요령을 살펴보세요.
* [GPU에서 파이토치 최적화](./optimization/fp16) 가이드와 [애플 실리콘(M1/M2)에서의 Stable Diffusion](./optimization/mps) 및 [ONNX 런타임](./optimization/onnx) 실행에 대한 추론 가이드를 통해 🧨 Diffuser 속도를 높이는 방법을 더 자세히 알아보세요.