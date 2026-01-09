<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 제어된 생성

Diffusion 모델에 의해 생성된 출력을 제어하는 것은 커뮤니티에서 오랫동안 추구해 왔으며 현재 활발한 연구 주제입니다. 널리 사용되는 많은 diffusion 모델에서는 이미지와 텍스트 프롬프트 등 입력의 미묘한 변화로 인해 출력이 크게 달라질 수 있습니다. 이상적인 세계에서는 의미가 유지되고 변경되는 방식을 제어할 수 있기를 원합니다.

의미 보존의 대부분의 예는 입력의 변화를 출력의 변화에 정확하게 매핑하는 것으로 축소됩니다. 즉, 프롬프트에서 피사체에 형용사를 추가하면 전체 이미지가 보존되고 변경된 피사체만 수정됩니다. 또는 특정 피사체의 이미지를 변형하면 피사체의 포즈가 유지됩니다.

추가적으로 생성된 이미지의 품질에는 의미 보존 외에도 영향을 미치고자 하는 품질이 있습니다. 즉, 일반적으로 결과물의 품질이 좋거나 특정 스타일을 고수하거나 사실적이기를 원합니다.

diffusion 모델 생성을 제어하기 위해 `diffusers`가 지원하는 몇 가지 기술을 문서화합니다. 많은 부분이 최첨단 연구이며 미묘한 차이가 있을 수 있습니다. 명확한 설명이 필요하거나 제안 사항이 있으면 주저하지 마시고 [포럼](https://discuss.huggingface.co/) 또는 [GitHub 이슈](https://github.com/huggingface/diffusers/issues)에서 토론을 시작하세요.

생성 제어 방법에 대한 개략적인 설명과 기술 개요를 제공합니다. 기술에 대한 자세한 설명은 파이프라인에서 링크된 원본 논문을 참조하는 것이 가장 좋습니다.

사용 사례에 따라 적절한 기술을 선택해야 합니다. 많은 경우 이러한 기법을 결합할 수 있습니다. 예를 들어, 텍스트 반전과 SEGA를 결합하여 텍스트 반전을 사용하여 생성된 출력에 더 많은 의미적 지침을 제공할 수 있습니다.

별도의 언급이 없는 한, 이러한 기법은 기존 모델과 함께 작동하며 자체 가중치가 필요하지 않은 기법입니다.

1. [Instruct Pix2Pix](#instruct-pix2pix)
2. [Pix2Pix Zero](#pix2pixzero)
3. [Attend and Excite](#attend-and-excite)
4. [Semantic Guidance](#semantic-guidance)
5. [Self-attention Guidance](#self-attention-guidance)
6. [Depth2Image](#depth2image)
7. [MultiDiffusion Panorama](#multidiffusion-panorama)
8. [DreamBooth](#dreambooth)
9. [Textual Inversion](#textual-inversion)
10. [ControlNet](#controlnet)
11. [Prompt Weighting](#prompt-weighting)
12. [Custom Diffusion](#custom-diffusion)
13. [Model Editing](#model-editing)
14. [DiffEdit](#diffedit)
15. [T2I-Adapter](#t2i-adapter)

편의를 위해, 추론만 하거나 파인튜닝/학습하는 방법에 대한 표를 제공합니다.

|                     **Method**                      | **Inference only** | **Requires training /<br> fine-tuning** |                                          **Comments**                                           |
| :-------------------------------------------------: | :----------------: | :-------------------------------------: | :---------------------------------------------------------------------------------------------: |
|        [Instruct Pix2Pix](#instruct-pix2pix)        |         ✅         |                   ❌                    | Can additionally be<br>fine-tuned for better <br>performance on specific <br>edit instructions. |
|            [Pix2Pix Zero](#pix2pixzero)             |         ✅         |                   ❌                    |                                                                                                 |
|       [Attend and Excite](#attend-and-excite)       |         ✅         |                   ❌                    |                                                                                                 |
|       [Semantic Guidance](#semantic-guidance)       |         ✅         |                   ❌                    |                                                                                                 |
| [Self-attention Guidance](#self-attention-guidance) |         ✅         |                   ❌                    |                                                                                                 |
|             [Depth2Image](#depth2image)             |         ✅         |                   ❌                    |                                                                                                 |
| [MultiDiffusion Panorama](#multidiffusion-panorama) |         ✅         |                   ❌                    |                                                                                                 |
|              [DreamBooth](#dreambooth)              |         ❌         |                   ✅                    |                                                                                                 |
|       [Textual Inversion](#textual-inversion)       |         ❌         |                   ✅                    |                                                                                                 |
|              [ControlNet](#controlnet)              |         ✅         |                   ❌                    |             A ControlNet can be <br>trained/fine-tuned on<br>a custom conditioning.             |
|        [Prompt Weighting](#prompt-weighting)        |         ✅         |                   ❌                    |                                                                                                 |
|        [Custom Diffusion](#custom-diffusion)        |         ❌         |                   ✅                    |                                                                                                 |
|           [Model Editing](#model-editing)           |         ✅         |                   ❌                    |                                                                                                 |
|                [DiffEdit](#diffedit)                |         ✅         |                   ❌                    |                                                                                                 |
|             [T2I-Adapter](#t2i-adapter)             |         ✅         |                   ❌                    |                                                                                                 |

## Pix2Pix Instruct

[Paper](https://huggingface.co/papers/2211.09800)

[Instruct Pix2Pix](../api/pipelines/stable_diffusion/pix2pix) 는 입력 이미지 편집을 지원하기 위해 stable diffusion에서 미세-조정되었습니다. 이미지와 편집을 설명하는 프롬프트를 입력으로 받아 편집된 이미지를 출력합니다.
Instruct Pix2Pix는 [InstructGPT](https://openai.com/blog/instruction-following/)와 같은 프롬프트와 잘 작동하도록 명시적으로 훈련되었습니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion/pix2pix)를 참조하세요.

## Pix2Pix Zero

[Paper](https://huggingface.co/papers/2302.03027)

[Pix2Pix Zero](../api/pipelines/stable_diffusion/pix2pix_zero)를 사용하면 일반적인 이미지 의미를 유지하면서 한 개념이나 피사체가 다른 개념이나 피사체로 변환되도록 이미지를 수정할 수 있습니다.

노이즈 제거 프로세스는 한 개념적 임베딩에서 다른 개념적 임베딩으로 안내됩니다. 중간 잠복(intermediate latents)은 디노이징(denoising?) 프로세스 중에 최적화되어 참조 주의 지도(reference attention maps)를 향해 나아갑니다. 참조 주의 지도(reference attention maps)는 입력 이미지의 노이즈 제거(?) 프로세스에서 나온 것으로 의미 보존을 장려하는 데 사용됩니다.

Pix2Pix Zero는 합성 이미지와 실제 이미지를 편집하는 데 모두 사용할 수 있습니다.

- 합성 이미지를 편집하려면 먼저 캡션이 지정된 이미지를 생성합니다.
  다음으로 편집할 컨셉과 새로운 타겟 컨셉에 대한 이미지 캡션을 생성합니다. 이를 위해 [Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5)와 같은 모델을 사용할 수 있습니다. 그런 다음 텍스트 인코더를 통해 소스 개념과 대상 개념 모두에 대한 "평균" 프롬프트 임베딩을 생성합니다. 마지막으로, 합성 이미지를 편집하기 위해 pix2pix-zero 알고리즘을 사용합니다.
- 실제 이미지를 편집하려면 먼저 [BLIP](https://huggingface.co/docs/transformers/model_doc/blip)과 같은 모델을 사용하여 이미지 캡션을 생성합니다. 그런 다음 프롬프트와 이미지에 ddim 반전을 적용하여 "역(inverse)" latents을 생성합니다. 이전과 마찬가지로 소스 및 대상 개념 모두에 대한 "평균(mean)" 프롬프트 임베딩이 생성되고 마지막으로 "역(inverse)" latents와 결합된 pix2pix-zero 알고리즘이 이미지를 편집하는 데 사용됩니다.

> [!TIP]
> Pix2Pix Zero는 '제로 샷(zero-shot)' 이미지 편집이 가능한 최초의 모델입니다.
> 즉, 이 모델은 다음과 같이 일반 소비자용 GPU에서 1분 이내에 이미지를 편집할 수 있습니다(../api/pipelines/stable_diffusion/pix2pix_zero#usage-example).

위에서 언급했듯이 Pix2Pix Zero에는 특정 개념으로 세대를 유도하기 위해 (UNet, VAE 또는 텍스트 인코더가 아닌) latents을 최적화하는 기능이 포함되어 있습니다.즉, 전체 파이프라인에 표준 [StableDiffusionPipeline](../api/pipelines/stable_diffusion/text2img)보다 더 많은 메모리가 필요할 수 있습니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion/pix2pix_zero)를 참조하세요.

## Attend and Excite

[Paper](https://huggingface.co/papers/2301.13826)

[Attend and Excite](../api/pipelines/stable_diffusion/attend_and_excite)를 사용하면 프롬프트의 피사체가 최종 이미지에 충실하게 표현되도록 할 수 있습니다.

이미지에 존재해야 하는 프롬프트의 피사체에 해당하는 일련의 토큰 인덱스가 입력으로 제공됩니다. 노이즈 제거 중에 각 토큰 인덱스는 이미지의 최소 한 패치 이상에 대해 최소 주의 임계값을 갖도록 보장됩니다. 모든 피사체 토큰에 대해 주의 임계값이 통과될 때까지 노이즈 제거 프로세스 중에 중간 잠복기가 반복적으로 최적화되어 가장 소홀히 취급되는 피사체 토큰의 주의력을 강화합니다.

Pix2Pix Zero와 마찬가지로 Attend and Excite 역시 파이프라인에 미니 최적화 루프(사전 학습된 가중치를 그대로 둔 채)가 포함되며, 일반적인 'StableDiffusionPipeline'보다 더 많은 메모리가 필요할 수 있습니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion/attend_and_excite)를 참조하세요.

## Semantic Guidance (SEGA)

[Paper](https://huggingface.co/papers/2301.12247)

의미유도(SEGA)를 사용하면 이미지에서 하나 이상의 컨셉을 적용하거나 제거할 수 있습니다. 컨셉의 강도도 조절할 수 있습니다. 즉, 스마일 컨셉을 사용하여 인물 사진의 스마일을 점진적으로 늘리거나 줄일 수 있습니다.

분류기 무료 안내(classifier free guidance)가 빈 프롬프트 입력을 통해 안내를 제공하는 방식과 유사하게, SEGA는 개념 프롬프트에 대한 안내를 제공합니다. 이러한 개념 프롬프트는 여러 개를 동시에 적용할 수 있습니다. 각 개념 프롬프트는 안내가 긍정적으로 적용되는지 또는 부정적으로 적용되는지에 따라 해당 개념을 추가하거나 제거할 수 있습니다.

Pix2Pix Zero 또는 Attend and Excite와 달리 SEGA는 명시적인 그라데이션 기반 최적화를 수행하는 대신 확산 프로세스와 직접 상호 작용합니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/semantic_stable_diffusion)를 참조하세요.

## Self-attention Guidance (SAG)

[Paper](https://huggingface.co/papers/2210.00939)

[자기 주의 안내](../api/pipelines/stable_diffusion/self_attention_guidance)는 이미지의 전반적인 품질을 개선합니다.

SAG는 고빈도 세부 정보를 기반으로 하지 않은 예측에서 완전히 조건화된 이미지에 이르기까지 가이드를 제공합니다. 고빈도 디테일은 UNet 자기 주의 맵에서 추출됩니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion/self_attention_guidance)를 참조하세요.

## Depth2Image

[Project](https://huggingface.co/stabilityai/stable-diffusion-2-depth)

[Depth2Image](../pipelines/stable_diffusion_2#depthtoimage)는 텍스트 안내 이미지 변화에 대한 시맨틱을 더 잘 보존하도록 안정적 확산에서 미세 조정되었습니다.

원본 이미지의 단안(monocular) 깊이 추정치를 조건으로 합니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion_2#depthtoimage)를 참조하세요.

> [!TIP]
> InstructPix2Pix와 Pix2Pix Zero와 같은 방법의 중요한 차이점은 전자의 경우
> 는 사전 학습된 가중치를 미세 조정하는 반면, 후자는 그렇지 않다는 것입니다. 즉, 다음을 수행할 수 있습니다.
> 사용 가능한 모든 안정적 확산 모델에 Pix2Pix Zero를 적용할 수 있습니다.

## MultiDiffusion Panorama

[Paper](https://huggingface.co/papers/2302.08113)

MultiDiffusion은 사전 학습된 diffusion model을 통해 새로운 생성 프로세스를 정의합니다. 이 프로세스는 고품질의 다양한 이미지를 생성하는 데 쉽게 적용할 수 있는 여러 diffusion 생성 방법을 하나로 묶습니다. 결과는 원하는 종횡비(예: 파노라마) 및 타이트한 분할 마스크에서 바운딩 박스에 이르는 공간 안내 신호와 같은 사용자가 제공한 제어를 준수합니다.
[MultiDiffusion 파노라마](../api/pipelines/stable_diffusion/panorama)를 사용하면 임의의 종횡비(예: 파노라마)로 고품질 이미지를 생성할 수 있습니다.

파노라마 이미지를 생성하는 데 사용하는 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion/panorama)를 참조하세요.

## 나만의 모델 파인튜닝

사전 학습된 모델 외에도 Diffusers는 사용자가 제공한 데이터에 대해 모델을 파인튜닝할 수 있는 학습 스크립트가 있습니다.

## DreamBooth

[DreamBooth](../training/dreambooth)는 모델을 파인튜닝하여 새로운 주제에 대해 가르칩니다. 즉, 한 사람의 사진 몇 장을 사용하여 다양한 스타일로 그 사람의 이미지를 생성할 수 있습니다.

사용 방법에 대한 자세한 내용은 [여기](../training/dreambooth)를 참조하세요.

## Textual Inversion

[Textual Inversion](../training/text_inversion)은 모델을 파인튜닝하여 새로운 개념에 대해 학습시킵니다. 즉, 특정 스타일의 아트웍 사진 몇 장을 사용하여 해당 스타일의 이미지를 생성할 수 있습니다.

사용 방법에 대한 자세한 내용은 [여기](../training/text_inversion)를 참조하세요.

## ControlNet

[Paper](https://huggingface.co/papers/2302.05543)

[ControlNet](../api/pipelines/stable_diffusion/controlnet)은 추가 조건을 추가하는 보조 네트워크입니다.
가장자리 감지, 낙서, 깊이 맵, 의미적 세그먼트와 같은 다양한 조건에 대해 훈련된 8개의 표준 사전 훈련된 ControlNet이 있습니다,
깊이 맵, 시맨틱 세그먼테이션과 같은 다양한 조건으로 훈련된 8개의 표준 제어망이 있습니다.

사용 방법에 대한 자세한 내용은 [여기](../api/pipelines/stable_diffusion/controlnet)를 참조하세요.

## Prompt Weighting

프롬프트 가중치는 텍스트의 특정 부분에 더 많은 관심 가중치를 부여하는 간단한 기법입니다.
입력에 가중치를 부여하는 간단한 기법입니다.

자세한 설명과 예시는 [여기](../using-diffusers/weighted_prompts)를 참조하세요.

## Custom Diffusion

[Custom Diffusion](../training/custom_diffusion)은 사전 학습된 text-to-image 간 확산 모델의 교차 관심도 맵만 미세 조정합니다.
또한 textual inversion을 추가로 수행할 수 있습니다. 설계상 다중 개념 훈련을 지원합니다.
DreamBooth 및 Textual Inversion 마찬가지로, 사용자 지정 확산은 사전학습된 text-to-image diffusion 모델에 새로운 개념을 학습시켜 관심 있는 개념과 관련된 출력을 생성하는 데에도 사용됩니다.

자세한 설명은 [공식 문서](../training/custom_diffusion)를 참조하세요.

## Model Editing

[Paper](https://huggingface.co/papers/2303.08084)

[텍스트-이미지 모델 편집 파이프라인](../api/pipelines/model_editing)을 사용하면 사전학습된 text-to-image diffusion 모델이 입력 프롬프트에 있는 피사체에 대해 내릴 수 있는 잘못된 암시적 가정을 완화하는 데 도움이 됩니다.
예를 들어, 안정적 확산에 "A pack of roses"에 대한 이미지를 생성하라는 메시지를 표시하면 생성된 이미지의 장미는 빨간색일 가능성이 높습니다. 이 파이프라인은 이러한 가정을 변경하는 데 도움이 됩니다.

자세한 설명은 [공식 문서](../api/pipelines/model_editing)를 참조하세요.

## DiffEdit

[Paper](https://huggingface.co/papers/2210.11427)

[DiffEdit](../api/pipelines/diffedit)를 사용하면 원본 입력 이미지를 최대한 보존하면서 입력 프롬프트와 함께 입력 이미지의 의미론적 편집이 가능합니다.


자세한 설명은 [공식 문서](../api/pipelines/diffedit)를 참조하세요.

## T2I-Adapter

[Paper](https://huggingface.co/papers/2302.08453)

[T2I-어댑터](../api/pipelines/stable_diffusion/adapter)는 추가적인 조건을 추가하는 auxiliary 네트워크입니다.
가장자리 감지, 스케치, depth maps, semantic segmentations와 같은 다양한 조건에 대해 훈련된 8개의 표준 사전훈련된 adapter가 있습니다,

[공식 문서](api/pipelines/stable_diffusion/adapter)에서 사용 방법에 대한 정보를 참조하세요.