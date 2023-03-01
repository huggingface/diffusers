<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 훑어보기

🧨 Diffusers로 빠르게 시작하고 실행하세요!
이 훑어보기는 여러분이 개발자, 일반사용자 상관없이 시작하는 데 도움을 주며, 추론을 위해 [`DiffusionPipeline`] 사용하는 방법을 보여줍니다.

시작하기에 앞서서, 필요한 모든 라이브러리가 설치되어 있는지 확인하세요:

```bash
pip install --upgrade diffusers accelerate transformers
```

- [`accelerate`](https://huggingface.co/docs/accelerate/index)은 추론 및 학습을 위한 모델 불러오기 속도를 높입니다.
- [`transformers`](https://huggingface.co/docs/transformers/index)는 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview)과 같이 가장 널리 사용되는 확산 모델을 실행하기 위해 필요합니다.

## DiffusionPipeline

[`DiffusionPipeline`]은 추론을 위해 사전학습된 확산 시스템을 사용하는 가장 쉬운 방법입니다. 다양한 양식의 많은 작업에 [`DiffusionPipeline`]을 바로 사용할 수 있습니다. 지원되는 작업은 아래의 표를 참고하세요:

| **Task**                     | **Description**                                                                                              | **Pipeline**
|------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------|
| Unconditional Image Generation          | 가우시안 노이즈에서 이미지 생성 | [unconditional_image_generation](./using-diffusers/unconditional_image_generation`) |
| Text-Guided Image Generation | 텍스트 프롬프트로 이미지 생성 | [conditional_image_generation](./using-diffusers/conditional_image_generation) |
| Text-Guided Image-to-Image Translation     | 텍스트 프롬프트에 따라 이미지 조정 | [img2img](./using-diffusers/img2img) |
| Text-Guided Image-Inpainting          | 마스크 및 텍스트 프롬프트가 주어진 이미지의 마스킹된 부분을 채우기 | [inpaint](./using-diffusers/inpaint) |
| Text-Guided Depth-to-Image Translation | 깊이 추정을 통해 구조를 유지하면서 텍스트 프롬프트에 따라 이미지의 일부를 조정 | [depth2image](./using-diffusers/depth2image) |

확산 파이프라인이 다양한 작업에 대해 어떻게 작동하는지는 [**Using Diffusers**](./using-diffusers/overview)를 참고하세요.

예를들어, [`DiffusionPipeline`] 인스턴스를 생성하여 시작하고, 다운로드하려는 파이프라인 체크포인트를 지정합니다.
모든 [Diffusers' checkpoint](https://huggingface.co/models?library=diffusers&sort=downloads)에 대해 [`DiffusionPipeline`]을 사용할 수 있습니다.
하지만, 이 가이드에서는 [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion)을 사용하여 text-to-image를 하는데 [`DiffusionPipeline`]을 사용합니다.

[Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion) 기반 모델을 실행하기 전에 [license](https://huggingface.co/spaces/CompVis/stable-diffusion-license)를 주의 깊게 읽으세요.
이는 모델의 향상된 이미지 생성 기능과 이것으로 생성될 수 있는 유해한 콘텐츠 때문입니다. 선택한 Stable Diffusion 모델(*예*: [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5))로 이동하여 라이센스를 읽으세요.

다음과 같이 모델을 로드할 수 있습니다:

```python
>>> from diffusers import DiffusionPipeline

>>> pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
```

[`DiffusionPipeline`]은 모든 모델링, 토큰화 및 스케줄링 구성요소를 다운로드하고 캐시합니다.
모델은 약 14억개의 매개변수로 구성되어 있으므로 GPU에서 실행하는 것이 좋습니다.
PyTorch에서와 마찬가지로 생성기 객체를 GPU로 옮길 수 있습니다.

```python
>>> pipeline.to("cuda")
```

이제 `pipeline`을 사용할 수 있습니다:

```python
>>> image = pipeline("An image of a squirrel in Picasso style").images[0]
```

출력은 기본적으로 [PIL Image object](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class)로 래핑됩니다.

다음과 같이 함수를 호출하여 이미지를 저장할 수 있습니다:

```python
>>> image.save("image_of_squirrel_painting.png")
```

**참고**: 다음을 통해 가중치를 다운로드하여 로컬에서 파이프라인을 사용할 수도 있습니다:

```
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```

그리고 저장된 가중치를 파이프라인에 불러옵니다.

```python
>>> pipeline = DiffusionPipeline.from_pretrained("./stable-diffusion-v1-5")
```

파이프라인 실행은 동일한 모델 아키텍처이므로 위의 코드와 동일합니다.

```python
>>> generator.to("cuda")
>>> image = generator("An image of a squirrel in Picasso style").images[0]
>>> image.save("image_of_squirrel_painting.png")
```

확산 시스템은 각각 장점이 있는 여러 다른 [schedulers](./api/schedulers/overview)와 함께 사용할 수 있습니다. 기본적으로 Stable Diffusion은 `PNDMScheduler`로 실행되지만 다른 스케줄러를 사용하는 방법은 매우 간단합니다. *예* [`EulerDiscreteScheduler`] 스케줄러를 사용하려는 경우, 다음과 같이 사용할 수 있습니다:

```python
>>> from diffusers import EulerDiscreteScheduler

>>> pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

>>> # change scheduler to Euler
>>> pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
```

스케줄러 변경 방법에 대한 자세한 내용은 [Using Schedulers](./using-diffusers/schedulers) 가이드를 참고하세요.

[Stability AI's](https://stability.ai/)의 Stable Diffusion 모델은 인상적인 이미지 생성 모델이며 텍스트에서 이미지를 생성하는 것보다 훨씬 더 많은 작업을 수행할 수 있습니다. 우리는 Stable Diffusion만을 위한 전체 문서 페이지를 제공합니다 [link](./conceptual/stable_diffusion).

만약 더 적은 메모리, 더 높은 추론 속도, Mac과 같은 특정 하드웨어 또는 ONNX 런타임에서 실행되도록 Stable Diffusion을 최적화하는 방법을 알고 싶다면 최적화 페이지를 살펴보세요:

- [Optimized PyTorch on GPU](./optimization/fp16)
- [Mac OS with PyTorch](./optimization/mps)
- [ONNX](./optimization/onnx)
- [OpenVINO](./optimization/open_vino)

확산 모델을 미세조정하거나 학습시키려면, [**training section**](./training/overview)을 살펴보세요.

마지막으로, 생성된 이미지를 공개적으로 배포할 때 신중을 기해 주세요 🤗.