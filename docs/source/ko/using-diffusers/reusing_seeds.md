<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Deterministic(결정적) 생성을 통한 이미지 품질 개선

생성된 이미지의 품질을 개선하는 일반적인 방법은 *결정적 batch(배치) 생성*을 사용하는 것입니다. 이 방법은 이미지 batch(배치)를 생성하고 두 번째 추론 라운드에서 더 자세한 프롬프트와 함께 개선할 이미지 하나를 선택하는 것입니다. 핵심은 일괄 이미지 생성을 위해 파이프라인에 [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#generator) 목록을 전달하고, 각 `Generator`를 시드에 연결하여 이미지에 재사용할 수 있도록 하는 것입니다.

예를 들어 [`runwayml/stable-diffusion-v1-5`](runwayml/stable-diffusion-v1-5)를 사용하여 다음 프롬프트의 여러 버전을 생성해 봅시다.

```py
prompt = "Labrador in the style of Vermeer"
```

(가능하다면) 파이프라인을 [`DiffusionPipeline.from_pretrained`]로 인스턴스화하여 GPU에 배치합니다.

```python
>>> from diffusers import DiffusionPipeline

>>> pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
>>> pipe = pipe.to("cuda")
```

이제 네 개의 서로 다른 `Generator`를 정의하고 각 `Generator`에 시드(`0` ~ `3`)를 할당하여 나중에 특정 이미지에 대해 `Generator`를 재사용할 수 있도록 합니다.

```python
>>> import torch

>>> generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]
```

이미지를 생성하고 살펴봅니다.

```python
>>> images = pipe(prompt, generator=generator, num_images_per_prompt=4).images
>>> images
```

![img](https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds.jpg)

이 예제에서는 첫 번째 이미지를 개선했지만 실제로는 원하는 모든 이미지를 사용할 수 있습니다(심지어 두 개의 눈이 있는 이미지도!). 첫 번째 이미지에서는 시드가 '0'인 '생성기'를 사용했기 때문에 두 번째 추론 라운드에서는 이 '생성기'를 재사용할 것입니다. 이미지의 품질을 개선하려면 프롬프트에 몇 가지 텍스트를 추가합니다:

```python
prompt = [prompt + t for t in [", highly realistic", ", artsy", ", trending", ", colorful"]]
generator = [torch.Generator(device="cuda").manual_seed(0) for i in range(4)]
```

시드가 `0`인 제너레이터 4개를 생성하고, 이전 라운드의 첫 번째 이미지처럼 보이는 다른 이미지 batch(배치)를 생성합니다!

```python
>>> images = pipe(prompt, generator=generator).images
>>> images
```

![img](https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/reusabe_seeds_2.jpg)
