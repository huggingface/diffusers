<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Low-Rank Adaptation of Large Language Models (LoRA)

[[open-in-colab]]

> [!WARNING]
> 현재 LoRA는 [`UNet2DConditionalModel`]의 어텐션 레이어에서만 지원됩니다.

[LoRA(Low-Rank Adaptation of Large Language Models)](https://huggingface.co/papers/2106.09685)는 메모리를 적게 사용하면서 대규모 모델의 학습을 가속화하는 학습 방법입니다. 이는 rank-decomposition weight 행렬 쌍(**업데이트 행렬**이라고 함)을 추가하고 새로 추가된 가중치**만** 학습합니다. 여기에는 몇 가지 장점이 있습니다.

- 이전에 미리 학습된 가중치는 고정된 상태로 유지되므로 모델이 [치명적인 망각](https://www.pnas.org/doi/10.1073/pnas.1611835114) 경향이 없습니다.
- Rank-decomposition 행렬은 원래 모델보다 파라메터 수가 훨씬 적으므로 학습된 LoRA 가중치를 쉽게 끼워넣을 수 있습니다.
- LoRA 매트릭스는 일반적으로 원본 모델의 어텐션 레이어에 추가됩니다. 🧨 Diffusers는 [`~diffusers.loaders.UNet2DConditionLoadersMixin.load_attn_procs`] 메서드를 제공하여 LoRA 가중치를 모델의 어텐션 레이어로 불러옵니다. `scale` 매개변수를 통해 모델이 새로운 학습 이미지에 맞게 조정되는 범위를 제어할 수 있습니다.
- 메모리 효율성이 향상되어 Tesla T4, RTX 3080 또는 RTX 2080 Ti와 같은 소비자용 GPU에서 파인튜닝을 실행할 수 있습니다! T4와 같은 GPU는 무료이며 Kaggle 또는 Google Colab 노트북에서 쉽게 액세스할 수 있습니다.


> [!TIP]
> 💡 LoRA는 어텐션 레이어에만 한정되지는 않습니다. 저자는 언어 모델의 어텐션 레이어를 수정하는 것이 매우 효율적으로 죻은 성능을 얻기에 충분하다는 것을 발견했습니다. 이것이 LoRA 가중치를 모델의 어텐션 레이어에 추가하는 것이 일반적인 이유입니다. LoRA 작동 방식에 대한 자세한 내용은 [Using LoRA for effective Stable Diffusion fine-tuning](https://huggingface.co/blog/lora) 블로그를 확인하세요!

[cloneofsimo](https://github.com/cloneofsimo)는 인기 있는 [lora](https://github.com/cloneofsimo/lora) GitHub 리포지토리에서 Stable Diffusion을 위한 LoRA 학습을 최초로 시도했습니다. 🧨 Diffusers는 [text-to-image 생성](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image#training-with-lora) 및 [DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth#training-with-low-rank-adaptation-of-large-language-models-lora)을 지원합니다. 이 가이드는 두 가지를 모두 수행하는 방법을 보여줍니다.

모델을 저장하거나 커뮤니티와 공유하려면 Hugging Face 계정에 로그인하세요(아직 계정이 없는 경우 [생성](https://huggingface.co/join)하세요):

```bash
hf auth login
```

## Text-to-image

수십억 개의 파라메터들이 있는 Stable Diffusion과 같은 모델을 파인튜닝하는 것은 느리고 어려울 수 있습니다. LoRA를 사용하면 diffusion 모델을 파인튜닝하는 것이 훨씬 쉽고 빠릅니다. 8비트 옵티마이저와 같은 트릭에 의존하지 않고도 11GB의 GPU RAM으로 하드웨어에서 실행할 수 있습니다.


### 학습[[dreambooth-training]]

[Naruto BLIP 캡션](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) 데이터셋으로 [`stable-diffusion-v1-5`](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)를 파인튜닝해 나만의 포켓몬을 생성해 보겠습니다.

시작하려면 `MODEL_NAME` 및 `DATASET_NAME` 환경 변수가 설정되어 있는지 확인하십시오. `OUTPUT_DIR` 및 `HUB_MODEL_ID` 변수는 선택 사항이며 허브에서 모델을 저장할 위치를 지정합니다.

```bash
export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export OUTPUT_DIR="/sddata/finetune/lora/naruto"
export HUB_MODEL_ID="naruto-lora"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
```

학습을 시작하기 전에 알아야 할 몇 가지 플래그가 있습니다.

* `--push_to_hub`를 명시하면 학습된 LoRA 임베딩을 허브에 저장합니다.
* `--report_to=wandb`는 학습 결과를 가중치 및 편향 대시보드에 보고하고 기록합니다(예를 들어, 이 [보고서](https://wandb.ai/pcuenq/text2image-fine-tune/run/b4k1w0tn?workspace=user-pcuenq)를 참조하세요).
* `--learning_rate=1e-04`, 일반적으로 LoRA에서 사용하는 것보다 더 높은 학습률을 사용할 수 있습니다.

이제 학습을 시작할 준비가 되었습니다 (전체 학습 스크립트는 [여기](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py)에서 찾을 수 있습니다).

```bash
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=50 \
  --seed="0" \
  --push_to_hub
```

### 추론[[dreambooth-inference]]

이제 [`StableDiffusionPipeline`]에서 기본 모델을 불러와 추론을 위해 모델을 사용할 수 있습니다:

```py
>>> import torch
>>> from diffusers import StableDiffusionPipeline

>>> model_base = "stable-diffusion-v1-5/stable-diffusion-v1-5"

>>> pipe = StableDiffusionPipeline.from_pretrained(model_base, dtype=torch.float16)
```

*기본 모델의 가중치 위에* 파인튜닝된 DreamBooth 모델에서 LoRA 가중치를 불러온 다음, 더 빠른 추론을 위해 파이프라인을 GPU로 이동합니다. LoRA 가중치를 프리징된 사전 훈련된 모델 가중치와 병합할 때, 선택적으로 'scale' 매개변수로 어느 정도의 가중치를 병합할 지 조절할 수 있습니다:

> [!TIP]
> 💡 `0`의 `scale` 값은 LoRA 가중치를 사용하지 않아 원래 모델의 가중치만 사용한 것과 같고, `1`의 `scale` 값은 파인튜닝된 LoRA 가중치만 사용함을 의미합니다. 0과 1 사이의 값들은 두 결과들 사이로 보간됩니다.

```py
>>> pipe.unet.load_attn_procs(model_path)
>>> pipe.to("cuda")
# LoRA 파인튜닝된 모델의 가중치 절반과 기본 모델의 가중치 절반 사용

>>> image = pipe(
...     "A picture of a sks dog in a bucket.",
...     num_inference_steps=25,
...     guidance_scale=7.5,
...     cross_attention_kwargs={"scale": 0.5},
... ).images[0]
# 완전히 파인튜닝된 LoRA 모델의 가중치 사용

>>> image = pipe("A picture of a sks dog in a bucket.", num_inference_steps=25, guidance_scale=7.5).images[0]
>>> image.save("bucket-dog.png")
```