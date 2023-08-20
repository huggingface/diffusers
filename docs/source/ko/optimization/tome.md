<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Token Merging (토큰 병합)

Token Merging (introduced in [Token Merging: Your ViT But Faster](https://arxiv.org/abs/2210.09461))은 트랜스포머 기반 네트워크의 forward pass에서 중복 토큰이나 패치를 점진적으로 병합하는 방식으로 작동합니다. 이를 통해 기반 네트워크의 추론 지연 시간을 단축할 수 있습니다.

Token Merging(ToMe)이 출시된 후, 저자들은 [Fast Stable Diffusion을 위한 토큰 병합](https://arxiv.org/abs/2303.17604)을 발표하여 Stable Diffusion과 더 잘 호환되는 ToMe 버전을 소개했습니다. ToMe를 사용하면 [`DiffusionPipeline`]의 추론 지연 시간을 부드럽게 단축할 수 있습니다. 이 문서에서는 ToMe를 [`StableDiffusionPipeline`]에 적용하는 방법, 예상되는 속도 향상, [`StableDiffusionPipeline`]에서 ToMe를 사용할 때의 질적 측면에 대해 설명합니다.

## ToMe 사용하기

ToMe의 저자들은 [`tomesd`](https://github.com/dbolya/tomesd)라는 편리한 Python 라이브러리를 공개했는데, 이 라이브러리를 이용하면 [`DiffusionPipeline`]에 ToMe를 다음과 같이 적용할 수 있습니다:

```diff
from diffusers import StableDiffusionPipeline
import tomesd

pipeline = StableDiffusionPipeline.from_pretrained(
      "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
+ tomesd.apply_patch(pipeline, ratio=0.5)

image = pipeline("a photo of an astronaut riding a horse on mars").images[0]
```

이것이 다입니다!

`tomesd.apply_patch()`는 파이프라인 추론 속도와 생성된 토큰의 품질 사이의 균형을 맞출 수 있도록 [여러 개의 인자](https://github.com/dbolya/tomesd#usage)를 노출합니다. 이러한 인수 중 가장 중요한 것은 `ratio(비율)`입니다. `ratio`은 forward pass 중에 병합될 토큰의 수를 제어합니다. `tomesd`에 대한 자세한 내용은 해당 리포지토리(https://github.com/dbolya/tomesd) 및 [논문](https://arxiv.org/abs/2303.17604)을 참고하시기 바랍니다.

## `StableDiffusionPipeline`으로 `tomesd` 벤치마킹하기

We benchmarked the impact of using `tomesd` on [`StableDiffusionPipeline`] along with [xformers](https://huggingface.co/docs/diffusers/optimization/xformers) across different image resolutions. We used A100 and V100 as our test GPU devices with the following development environment (with Python 3.8.5):
다양한 이미지 해상도에서 [xformers](https://huggingface.co/docs/diffusers/optimization/xformers)를 적용한 상태에서, [`StableDiffusionPipeline`]에 `tomesd`를 사용했을 때의 영향을 벤치마킹했습니다. 테스트 GPU 장치로 A100과 V100을 사용했으며 개발 환경은 다음과 같습니다(Python 3.8.5 사용):

```bash
- `diffusers` version: 0.15.1
- Python version: 3.8.16
- PyTorch version (GPU?): 1.13.1+cu116 (True)
- Huggingface_hub version: 0.13.2
- Transformers version: 4.27.2
- Accelerate version: 0.18.0
- xFormers version: 0.0.16
- tomesd version: 0.1.2
```

벤치마킹에는 다음 스크립트를 사용했습니다: [https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335](https://gist.github.com/sayakpaul/27aec6bca7eb7b0e0aa4112205850335). 결과는 다음과 같습니다:

### A100

| 해상도 | 배치 크기 | Vanilla | ToMe | ToMe + xFormers | ToMe 속도 향상 (%) | ToMe + xFormers 속도 향상 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 10 | 6.88 | 5.26 | 4.69 | 23.54651163 | 31.83139535 |
|  |  |  |  |  |  |  |
| 768 | 10 | OOM | 14.71 | 11 |  |  |
|  | 8 | OOM | 11.56 | 8.84 |  |  |
|  | 4 | OOM | 5.98 | 4.66 |  |  |
|  | 2 | 4.99 | 3.24 | 3.1 | 35.07014028 | 37.8757515 |
|  | 1 | 3.29 | 2.24 | 2.03 | 31.91489362 | 38.29787234 |
|  |  |  |  |  |  |  |
| 1024 | 10 | OOM | OOM | OOM |  |  |
|  | 8 | OOM | OOM | OOM |  |  |
|  | 4 | OOM | 12.51 | 9.09 |  |  |
|  | 2 | OOM | 6.52 | 4.96 |  |  |
|  | 1 | 6.4 | 3.61 | 2.81 | 43.59375 | 56.09375 |

***결과는 초 단위입니다. 속도 향상은 `Vanilla`과 비교해 계산됩니다.***

### V100

| 해상도 | 배치 크기 | Vanilla | ToMe | ToMe + xFormers | ToMe 속도 향상 (%) | ToMe + xFormers 속도 향상 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| 512 | 10 | OOM | 10.03 | 9.29 |  |  |
|  | 8 | OOM | 8.05 | 7.47 |  |  |
|  | 4 | 5.7 | 4.3 | 3.98 | 24.56140351 | 30.1754386 |
|  | 2 | 3.14 | 2.43 | 2.27 | 22.61146497 | 27.70700637 |
|  | 1 | 1.88 | 1.57 | 1.57 | 16.4893617 | 16.4893617 |
|  |  |  |  |  |  |  |
| 768 | 10 | OOM | OOM | 23.67 |  |  |
|  | 8 | OOM | OOM | 18.81 |  |  |
|  | 4 | OOM | 11.81 | 9.7 |  |  |
|  | 2 | OOM | 6.27 | 5.2 |  |  |
|  | 1 | 5.43 | 3.38 | 2.82 | 37.75322284 | 48.06629834 |
|  |  |  |  |  |  |  |
| 1024 | 10 | OOM | OOM | OOM |  |  |
|  | 8 | OOM | OOM | OOM |  |  |
|  | 4 | OOM | OOM | 19.35 |  |  |
|  | 2 | OOM | 13 | 10.78 |  |  |
|  | 1 | OOM | 6.66 | 5.54 |  |  |

위의 표에서 볼 수 있듯이, 이미지 해상도가 높을수록 `tomesd`를 사용한 속도 향상이 더욱 두드러집니다. 또한 `tomesd`를 사용하면 1024x1024와 같은 더 높은 해상도에서 파이프라인을 실행할 수 있다는 점도 흥미롭습니다. 

[`torch.compile()`](https://huggingface.co/docs/diffusers/optimization/torch2.0)을 사용하면 추론 속도를 더욱 높일 수 있습니다. 

## 품질

As reported in [the paper](https://arxiv.org/abs/2303.17604), ToMe can preserve the quality of the generated images to a great extent while speeding up inference. By increasing the `ratio`, it is possible to further speed up inference, but that might come at the cost of a deterioration in the image quality. 

To test the quality of the generated samples using our setup, we sampled a few prompts from the “Parti Prompts” (introduced in [Parti](https://parti.research.google/)) and performed inference with the [`StableDiffusionPipeline`] in the following settings:

[논문](https://arxiv.org/abs/2303.17604)에 보고된 바와 같이, ToMe는 생성된 이미지의 품질을 상당 부분 보존하면서 추론 속도를 높일 수 있습니다. `ratio`을 높이면 추론 속도를 더 높일 수 있지만, 이미지 품질이 저하될 수 있습니다. 

해당 설정을 사용하여 생성된 샘플의 품질을 테스트하기 위해, "Parti 프롬프트"([Parti](https://parti.research.google/)에서 소개)에서 몇 가지 프롬프트를 샘플링하고 다음 설정에서 [`StableDiffusionPipeline`]을 사용하여 추론을 수행했습니다:

- Vanilla [`StableDiffusionPipeline`]
- [`StableDiffusionPipeline`] + ToMe
- [`StableDiffusionPipeline`] + ToMe + xformers

생성된 샘플의 품질이 크게 저하되는 것을 발견하지 못했습니다. 다음은 샘플입니다: 

![tome-samples](https://huggingface.co/datasets/diffusers/docs-images/resolve/main/tome/tome_samples.png)

생성된 샘플은 [여기](https://wandb.ai/sayakpaul/tomesd-results/runs/23j4bj3i?workspace=)에서 확인할 수 있습니다. 이 실험을 수행하기 위해 [이 스크립트](https://gist.github.com/sayakpaul/8cac98d7f22399085a060992f411ecbd)를 사용했습니다.