<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 조건부 이미지 생성

[[open-in-colab]]

조건부 이미지 생성을 사용하면 텍스트 프롬프트에서 이미지를 생성할 수 있습니다. 텍스트는 임베딩으로 변환되며, 임베딩은 노이즈에서 이미지를 생성하도록 모델을 조건화하는 데 사용됩니다.

[`DiffusionPipeline`]은 추론을 위해 사전 훈련된 diffusion 시스템을 사용하는 가장 쉬운 방법입니다.

먼저 [`DiffusionPipeline`]의 인스턴스를 생성하고 다운로드할 파이프라인 [체크포인트](https://huggingface.co/models?library=diffusers&sort=downloads)를 지정합니다.

이 가이드에서는 [잠재 Diffusion](https://huggingface.co/CompVis/ldm-text2im-large-256)과 함께 텍스트-이미지 생성에 [`DiffusionPipeline`]을 사용합니다:

```python
>>> from diffusers import DiffusionPipeline

>>> generator = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
```

[`DiffusionPipeline`]은 모든 모델링, 토큰화, 스케줄링 구성 요소를 다운로드하고 캐시합니다. 
이 모델은 약 14억 개의 파라미터로 구성되어 있기 때문에 GPU에서 실행할 것을 강력히 권장합니다.
PyTorch에서와 마찬가지로 생성기 객체를 GPU로 이동할 수 있습니다:

```python
>>> generator.to("cuda")
```

이제 텍스트 프롬프트에서 `생성기`를 사용할 수 있습니다:

```python
>>> image = generator("An image of a squirrel in Picasso style").images[0]
```

출력값은 기본적으로 [`PIL.Image`](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) 객체로 래핑됩니다.

호출하여 이미지를 저장할 수 있습니다:

```python
>>> image.save("image_of_squirrel_painting.png")
```

아래 스페이스를 사용해보고 안내 배율 매개변수를 자유롭게 조정하여 이미지 품질에 어떤 영향을 미치는지 확인해 보세요!

<iframe
	src="https://stabilityai-stable-diffusion.hf.space"
	frameborder="0"
	width="850"
	height="500"
></iframe>