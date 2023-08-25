<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# DreamBooth

[DreamBooth](https://arxiv.org/abs/2208.12242)는 한 주제에 대한 적은 이미지(3~5개)만으로도 stable diffusion과 같이 text-to-image 모델을 개인화할 수 있는 방법입니다. 이를 통해 모델은 다양한 장면, 포즈 및 장면(뷰)에서 피사체에 대해 맥락화(contextualized)된 이미지를 생성할 수 있습니다.

![프로젝트 블로그에서의 DreamBooth 예시](https://dreambooth.github.io/DreamBooth_files/teaser_static.jpg)
<small>에서의 Dreambooth 예시 <a href="https://dreambooth.github.io">project's blog.</a></small>


이 가이드는 다양한 GPU, Flax 사양에 대해 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) 모델로 DreamBooth를 파인튜닝하는 방법을 보여줍니다. 더 깊이 파고들어 작동 방식을 확인하는 데 관심이 있는 경우, 이 가이드에 사용된 DreamBooth의 모든 학습 스크립트를 [여기](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)에서 찾을 수 있습니다.

스크립트를 실행하기 전에 라이브러리의 학습에 필요한 dependencies를 설치해야 합니다. 또한 `main` GitHub 브랜치에서 🧨 Diffusers를 설치하는 것이 좋습니다.

```bash
pip install git+https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
```

xFormers는 학습에 필요한 요구 사항은 아니지만, 가능하면 [설치](../optimization/xformers)하는 것이 좋습니다. 학습 속도를 높이고 메모리 사용량을 줄일 수 있기 때문입니다.

모든 dependencies을 설정한 후 다음을 사용하여 [🤗 Accelerate](https://github.com/huggingface/accelerate/) 환경을 다음과 같이 초기화합니다:

```bash
accelerate config
```

별도 설정 없이 기본 🤗 Accelerate 환경을 설치하려면 다음을 실행합니다:

```bash
accelerate config default
```

또는 현재 환경이 노트북과 같은 대화형 셸을 지원하지 않는 경우 다음을 사용할 수 있습니다:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

## 파인튜닝

<Tip warning={true}>

DreamBooth 파인튜닝은 하이퍼파라미터에 매우 민감하고 과적합되기 쉽습니다. 적절한 하이퍼파라미터를 선택하는 데 도움이 되도록 다양한 권장 설정이 포함된 [심층 분석](https://huggingface.co/blog/dreambooth)을 살펴보는 것이 좋습니다.

</Tip>

<frameworkcontent>
<pt>
[몇 장의 강아지 이미지들](https://drive.google.com/drive/folders/1BO_dyz-p65qhBRRMRA4TbZ8qW4rB99JZ)로 DreamBooth를 시도해봅시다. 
이를 다운로드해 디렉터리에 저장한 다음 `INSTANCE_DIR` 환경 변수를 해당 경로로 설정합니다:


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export OUTPUT_DIR="path_to_saved_model"
```

그런 다음, 다음 명령을 사용하여 학습 스크립트를 실행할 수 있습니다 (전체 학습 스크립트는 [여기](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py)에서 찾을 수 있습니다):

```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400
```
</pt>
<jax>

TPU에 액세스할 수 있거나 더 빠르게 훈련하고 싶다면 [Flax 학습 스크립트](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_flax.py)를 사용해 볼 수 있습니다. Flax 학습 스크립트는 gradient checkpointing 또는 gradient accumulation을 지원하지 않으므로, 메모리가 30GB 이상인 GPU가 필요합니다.

스크립트를 실행하기 전에 요구 사항이 설치되어 있는지 확인하십시오.

```bash
pip install -U -r requirements.txt
```

그러면 다음 명령어로 학습 스크립트를 실행시킬 수 있습니다:

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="path-to-instance-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="a photo of sks dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --max_train_steps=400
```
</jax>
</frameworkcontent>

### Prior-preserving(사전 보존) loss를 사용한 파인튜닝

과적합과 language drift를 방지하기 위해 사전 보존이 사용됩니다(관심이 있는 경우 [논문](https://arxiv.org/abs/2208.12242)을 참조하세요).  사전 보존을 위해 동일한 클래스의 다른 이미지를 학습 프로세스의 일부로 사용합니다. 좋은 점은 Stable Diffusion 모델 자체를 사용하여 이러한 이미지를 생성할 수 있다는 것입니다! 학습 스크립트는 생성된 이미지를 우리가 지정한 로컬 경로에 저장합니다.

저자들에 따르면 사전 보존을 위해 `num_epochs * num_samples`개의 이미지를 생성하는 것이 좋습니다. 200-300개에서 대부분 잘 작동합니다.

<frameworkcontent>
<pt>
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```
</pt>
<jax>
```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=5e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```
</jax>
</frameworkcontent>

## 텍스트 인코더와 and UNet로 파인튜닝하기

해당 스크립트를 사용하면 `unet`과 함께 `text_encoder`를 파인튜닝할 수 있습니다. 실험에서(자세한 내용은 [🧨 Diffusers를 사용해 DreamBooth로 Stable Diffusion 학습하기](https://huggingface.co/blog/dreambooth) 게시물을 확인하세요), 특히 얼굴 이미지를 생성할 때 훨씬 더 나은 결과를 얻을 수 있습니다.

<Tip warning={true}>

텍스트 인코더를 학습시키려면 추가 메모리가 필요해 16GB GPU로는 동작하지 않습니다. 이 옵션을 사용하려면 최소 24GB VRAM이 필요합니다.

</Tip>

`--train_text_encoder` 인수를 학습 스크립트에 전달하여 `text_encoder` 및 `unet`을 파인튜닝할 수 있습니다:

<frameworkcontent>
<pt>
```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --use_8bit_adam
  --gradient_checkpointing \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```
</pt>
<jax>
```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

python train_dreambooth_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --train_text_encoder \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --learning_rate=2e-6 \
  --num_class_images=200 \
  --max_train_steps=800
```
</jax>
</frameworkcontent>

## LoRA로 파인튜닝하기

DreamBooth에서 대규모 모델의 학습을 가속화하기 위한 파인튜닝 기술인 LoRA(Low-Rank Adaptation of Large Language Models)를 사용할 수 있습니다. 자세한 내용은 [LoRA 학습](training/lora#dreambooth) 가이드를 참조하세요.

### 학습 중 체크포인트 저장하기

Dreambooth로 훈련하는 동안 과적합하기 쉬우므로, 때때로 학습 중에 정기적인 체크포인트를 저장하는 것이 유용합니다. 중간 체크포인트 중 하나가 최종 모델보다 더 잘 작동할 수 있습니다! 체크포인트 저장 기능을 활성화하려면 학습 스크립트에 다음 인수를 전달해야 합니다:

```bash
  --checkpointing_steps=500
```

이렇게 하면 `output_dir`의 하위 폴더에 전체 학습 상태가 저장됩니다. 하위 폴더 이름은 접두사 `checkpoint-`로 시작하고 지금까지 수행된 step 수입니다. 예시로 `checkpoint-1500`은 1500 학습 step 후에 저장된 체크포인트입니다.

#### 저장된 체크포인트에서 훈련 재개하기

저장된 체크포인트에서 훈련을 재개하려면, `--resume_from_checkpoint` 인수를 전달한 다음 사용할 체크포인트의 이름을 지정하면 됩니다. 특수 문자열 `"latest"`를 사용하여 저장된 마지막 체크포인트(즉, step 수가 가장 많은 체크포인트)에서 재개할 수도 있습니다. 예를 들어 다음은 1500 step 후에 저장된 체크포인트에서부터 학습을 재개합니다:

```bash
  --resume_from_checkpoint="checkpoint-1500"
```

원하는 경우 일부 하이퍼파라미터를 조정할 수 있습니다.

#### 저장된 체크포인트를 사용하여 추론 수행하기

저장된 체크포인트는 훈련 재개에 적합한 형식으로 저장됩니다. 여기에는 모델 가중치뿐만 아니라 옵티마이저, 데이터 로더 및 학습률의 상태도 포함됩니다.

**`"accelerate>=0.16.0"`**이 설치된 경우 다음 코드를 사용하여 중간 체크포인트에서 추론을 실행합니다.

```python
from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

# 학습에 사용된 것과 동일한 인수(model, revision)로 파이프라인을 불러옵니다.
model_id = "CompVis/stable-diffusion-v1-4"

unet = UNet2DConditionModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/unet")

# `args.train_text_encoder`로 학습한 경우면 텍스트 인코더를 꼭 불러오세요
text_encoder = CLIPTextModel.from_pretrained("/sddata/dreambooth/daruma-v2-1/checkpoint-100/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(model_id, unet=unet, text_encoder=text_encoder, dtype=torch.float16)
pipeline.to("cuda")

# 추론을 수행하거나 저장하거나, 허브에 푸시합니다.
pipeline.save_pretrained("dreambooth-pipeline")
```

If you have **`"accelerate<0.16.0"`** installed, you need to convert it to an inference pipeline first:

```python
from accelerate import Accelerator
from diffusers import DiffusionPipeline

# 학습에 사용된 것과 동일한 인수(model, revision)로 파이프라인을 불러옵니다.
model_id = "CompVis/stable-diffusion-v1-4"
pipeline = DiffusionPipeline.from_pretrained(model_id)

accelerator = Accelerator()

# 초기 학습에 `--train_text_encoder`가 사용된 경우 text_encoder를 사용합니다.
unet, text_encoder = accelerator.prepare(pipeline.unet, pipeline.text_encoder)

# 체크포인트 경로로부터 상태를 복원합니다. 여기서는 절대 경로를 사용해야 합니다.
accelerator.load_state("/sddata/dreambooth/daruma-v2-1/checkpoint-100")

# unwrapped 모델로 파이프라인을 다시 빌드합니다.(.unet and .text_encoder로의 할당도 작동해야 합니다)
pipeline = DiffusionPipeline.from_pretrained(
    model_id,
    unet=accelerator.unwrap_model(unet),
    text_encoder=accelerator.unwrap_model(text_encoder),
)

# 추론을 수행하거나 저장하거나, 허브에 푸시합니다.
pipeline.save_pretrained("dreambooth-pipeline")
```

## 각 GPU 용량에서의 최적화

하드웨어에 따라 16GB에서 8GB까지 GPU에서 DreamBooth를 최적화하는 몇 가지 방법이 있습니다!

### xFormers

[xFormers](https://github.com/facebookresearch/xformers)는 Transformers를 최적화하기 위한 toolbox이며, 🧨 Diffusers에서 사용되는[memory-efficient attention](https://facebookresearch.github.io/xformers/components/ops.html#module-xformers.ops)  메커니즘을 포함하고 있습니다. [xFormers를 설치](./optimization/xformers)한 다음 학습 스크립트에 다음 인수를 추가합니다:

```bash
  --enable_xformers_memory_efficient_attention
```

xFormers는 Flax에서 사용할 수 없습니다.

### 그래디언트 없음으로 설정

메모리 사용량을 줄일 수 있는 또 다른 방법은 [기울기 설정](https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html)을 0 대신 `None`으로 하는 것입니다. 그러나 이로 인해 특정 동작이 변경될 수 있으므로 문제가 발생하면 이 인수를 제거해 보십시오. 학습 스크립트에 다음 인수를 추가하여 그래디언트를 `None`으로 설정합니다.

```bash
  --set_grads_to_none
```

### 16GB GPU

Gradient checkpointing과 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)의 8비트 옵티마이저의 도움으로, 16GB GPU에서 dreambooth를 훈련할 수 있습니다. bitsandbytes가 설치되어 있는지 확인하세요:

```bash
pip install bitsandbytes
```

그 다음, 학습 스크립트에 `--use_8bit_adam` 옵션을 명시합니다:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 --gradient_checkpointing \
  --use_8bit_adam \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### 12GB GPU

12GB GPU에서 DreamBooth를 실행하려면 gradient checkpointing, 8비트 옵티마이저, xFormers를 활성화하고 그래디언트를 `None`으로 설정해야 합니다.

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path-to-instance-images"
export CLASS_DIR="path-to-class-images"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --set_grads_to_none \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800
```

### 8GB GPU에서 학습하기

8GB GPU에 대해서는 [DeepSpeed](https://www.deepspeed.ai/)를 사용해 일부 텐서를 VRAM에서 CPU 또는 NVME로 오프로드하여 더 적은 GPU 메모리로 학습할 수도 있습니다.

🤗 Accelerate 환경을 구성하려면 다음 명령을 실행하세요:

```bash
accelerate config
```

환경 구성 중에 DeepSpeed를 사용할 것을 확인하세요.
그러면 DeepSpeed stage 2, fp16 혼합 정밀도를 결합하고 모델 매개변수와 옵티마이저 상태를 모두 CPU로 오프로드하면 8GB VRAM 미만에서 학습할 수 있습니다. 
단점은 더 많은 시스템 RAM(약 25GB)이 필요하다는 것입니다. 추가 구성 옵션은 [DeepSpeed 문서](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)를 참조하세요.

또한 기본 Adam 옵티마이저를 DeepSpeed의 최적화된 Adam 버전으로 변경해야 합니다.
이는 상당한 속도 향상을 위한 Adam인 [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu)입니다. 
`DeepSpeedCPUAdam`을 활성화하려면 시스템의 CUDA toolchain 버전이 PyTorch와 함께 설치된 것과 동일해야 합니다.

8비트 옵티마이저는 현재 DeepSpeed와 호환되지 않는 것 같습니다.

다음 명령으로 학습을 시작합니다:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="path_to_saved_model"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --sample_batch_size=1 \
  --gradient_accumulation_steps=1 --gradient_checkpointing \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800 \
  --mixed_precision=fp16
```

## 추론

모델을 학습한 후에는, 모델이 저장된 경로를 지정해 [`StableDiffusionPipeline`]로 추론을 수행할 수 있습니다. 프롬프트에 학습에 사용된 특수 `식별자`(이전 예시의 `sks`)가 포함되어 있는지 확인하세요.

**`"accelerate>=0.16.0"`**이 설치되어 있는 경우 다음 코드를 사용하여 중간 체크포인트에서 추론을 실행할 수 있습니다:

```python
from diffusers import StableDiffusionPipeline
import torch

model_id = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A photo of sks dog in a bucket"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save("dog-bucket.png")
```

[저장된 학습 체크포인트](#inference-from-a-saved-checkpoint)에서도 추론을 실행할 수도 있습니다.
