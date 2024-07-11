<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 프롬프트에 가중치 부여하기

[[open-in-colab]]

텍스트 가이드 기반의 diffusion 모델은 주어진 텍스트 프롬프트를 기반으로 이미지를 생성합니다.
텍스트 프롬프트에는 모델이 생성해야 하는 여러 개념이 포함될 수 있으며 프롬프트의 특정 부분에 가중치를 부여하는 것이 바람직한 경우가 많습니다.

Diffusion 모델은 문맥화된 텍스트 임베딩으로 diffusion 모델의 cross attention 레이어를 조절함으로써 작동합니다.
([더 많은 정보를 위한 Stable Diffusion Guide](https://huggingface.co/docs/optimum-neuron/main/en/package_reference/modeling#stable-diffusion)를 참고하세요).
따라서 프롬프트의 특정 부분을 강조하는(또는 강조하지 않는) 간단한 방법은 프롬프트의 관련 부분에 해당하는 텍스트 임베딩 벡터의 크기를 늘리거나 줄이는 것입니다.
이것은 "프롬프트 가중치 부여" 라고 하며, 커뮤니티에서 가장 요구하는 기능입니다.([이곳](https://github.com/huggingface/diffusers/issues/2431)의 issue를 보세요 ).

## Diffusers에서 프롬프트 가중치 부여하는 방법

우리는 `diffusers`의 역할이 다른 프로젝트를 가능하게 하는 필수적인 기능을 제공하는 toolbex라고 생각합니다.
[InvokeAI](https://github.com/invoke-ai/InvokeAI) 나 [diffuzers](https://github.com/abhishekkrthakur/diffuzers) 같은 강력한 UI를 구축할 수 있습니다.
프롬프트를 조작하는 방법을 지원하기 위해, `diffusers` 는
[StableDiffusionPipeline](https://huggingface.co/docs/diffusers/v0.18.2/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline)와 같은
많은 파이프라인에 [prompt_embeds](https://huggingface.co/docs/diffusers/v0.14.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline.__call__.prompt_embeds)
인수를 노출시켜, "prompt-weighted"/축척된 텍스트 임베딩을 파이프라인에 바로 전달할 수 있게 합니다.

[Compel 라이브러리](https://github.com/damian0815/compel)는 프롬프트의 일부를 강조하거나 강조하지 않을 수 있는 쉬운 방법을 제공합니다.
임베딩을 직접 준비하는 것 대신 이 방법을 사용하는 것을 강력히 추천합니다.

간단한 예제를 살펴보겠습니다.
다음과 같이 `"공을 갖고 노는 붉은색 고양이"` 이미지를 생성하고 싶습니다:

```py
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

prompt = "a red cat playing with a ball"

generator = torch.Generator(device="cpu").manual_seed(33)

image = pipe(prompt, generator=generator, num_inference_steps=20).images[0]
image
```

생성된 이미지:

![img](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_0.png)

사진에서 알 수 있듯이, "공"은 이미지에 없습니다. 이 부분을 강조해 볼까요!

먼저 `compel` 라이브러리를 설치해야합니다:

```sh
pip install compel
```

그런 다음에는 `Compel` 오브젝트를 생성합니다:

```py
from compel import Compel

compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
```

이제 `"++"` 를 사용해서 "공" 을 강조해 봅시다:

```py
prompt = "a red cat playing with a ball++"
```

그리고 이 프롬프트를 파이프라인에 바로 전달하지 않고, `compel_proc` 를 사용하여 처리해야합니다:

```py
prompt_embeds = compel_proc(prompt)
```

파이프라인에 `prompt_embeds` 를 바로 전달할 수 있습니다:

```py
generator = torch.Generator(device="cpu").manual_seed(33)

images = pipe(prompt_embeds=prompt_embeds, generator=generator, num_inference_steps=20).images[0]
image
```

이제 "공"이 있는 그림을 출력할 수 있습니다!

![img](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/compel/forest_1.png)

마찬가지로 `--` 접미사를 단어에 사용하여 문장의 일부를 강조하지 않을 수 있습니다. 한번 시도해 보세요!

즐겨찾는 파이프라인에 `prompt_embeds` 입력이 없는 경우 issue를 새로 만들어주세요.
Diffusers 팀은 최대한 대응하려고 노력합니다.

Compel 1.1.6 는 textual inversions을 사용하여 단순화하는 유티릴티 클래스를 추가합니다.
`DiffusersTextualInversionManager`를 인스턴스화 한 후 이를 Compel init에 전달합니다:

```
textual_inversion_manager = DiffusersTextualInversionManager(pipe)
compel = Compel(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder,
    textual_inversion_manager=textual_inversion_manager)
```

더 많은 정보를 얻고 싶다면 [compel](https://github.com/damian0815/compel) 라이브러리 문서를 참고하세요.
