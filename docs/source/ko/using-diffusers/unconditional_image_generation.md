<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Unconditional 이미지 생성

[[open-in-colab]]

Unconditional 이미지 생성은 비교적 간단한 작업입니다. 모델이 텍스트나 이미지와 같은 추가 조건 없이 이미 학습된 학습 데이터와 유사한 이미지만 생성합니다.

['DiffusionPipeline']은 추론을 위해 미리 학습된 diffusion 시스템을 사용하는 가장 쉬운 방법입니다.

먼저 ['DiffusionPipeline']의 인스턴스를 생성하고 다운로드할 파이프라인의 [체크포인트](https://huggingface.co/models?library=diffusers&sort=downloads)를 지정합니다. 허브의 🧨 diffusion 체크포인트 중 하나를 사용할 수 있습니다(사용할 체크포인트는 나비 이미지를 생성합니다).

> [!TIP]
> 💡 나만의 unconditional 이미지 생성 모델을 학습시키고 싶으신가요? 학습 가이드를 살펴보고 나만의 이미지를 생성하는 방법을 알아보세요.


이 가이드에서는 unconditional 이미지 생성에 ['DiffusionPipeline']과 [DDPM](https://huggingface.co/papers/2006.11239)을 사용합니다:

```python
 >>> from diffusers import DiffusionPipeline

 >>> generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128")
```

[diffusion 파이프라인]은 모든 모델링, 토큰화, 스케줄링 구성 요소를 다운로드하고 캐시합니다. 이 모델은 약 14억 개의 파라미터로 구성되어 있기 때문에 GPU에서 실행할 것을 강력히 권장합니다. PyTorch에서와 마찬가지로 제너레이터 객체를 GPU로 옮길 수 있습니다:

```python
 >>> generator.to("cuda")
```

이제 제너레이터를 사용하여 이미지를 생성할 수 있습니다:

```python
 >>> image = generator().images[0]
```

출력은 기본적으로 [PIL.Image](https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) 객체로 감싸집니다.

다음을 호출하여 이미지를 저장할 수 있습니다:

```python
 >>> image.save("generated_image.png")
```

아래 스페이스(데모 링크)를 이용해 보고, 추론 단계의 매개변수를 자유롭게 조절하여 이미지 품질에 어떤 영향을 미치는지 확인해 보세요!

<iframe src="https://stevhliu-ddpm-butterflies-128.hf.space" frameborder="0" width="850" height="500"></iframe>
