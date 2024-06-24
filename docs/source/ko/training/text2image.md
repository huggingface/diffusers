<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->


# Text-to-image

<Tip warning={true}>

text-to-image 파인튜닝 스크립트는 experimental 상태입니다. 과적합하기 쉽고 치명적인 망각과 같은 문제에 부딪히기 쉽습니다. 자체 데이터셋에서 최상의 결과를 얻으려면 다양한 하이퍼파라미터를 탐색하는 것이 좋습니다.

</Tip>

Stable Diffusion과 같은 text-to-image 모델은 텍스트 프롬프트에서 이미지를 생성합니다. 이 가이드는 PyTorch 및 Flax를 사용하여 자체 데이터셋에서 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4) 모델로 파인튜닝하는 방법을 보여줍니다. 이 가이드에 사용된 text-to-image 파인튜닝을 위한 모든 학습 스크립트에 관심이 있는 경우 이 [리포지토리](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)에서 자세히 찾을 수 있습니다.

스크립트를 실행하기 전에, 라이브러리의 학습 dependency들을 설치해야 합니다:

```bash
pip install git+https://github.com/huggingface/diffusers.git
pip install -U -r requirements.txt
```

그리고 [🤗Accelerate](https://github.com/huggingface/accelerate/) 환경을 초기화합니다:

```bash
accelerate config
```

리포지토리를 이미 복제한 경우, 이 단계를 수행할 필요가 없습니다. 대신, 로컬 체크아웃 경로를 학습 스크립트에 명시할 수 있으며 거기에서 로드됩니다.

### 하드웨어 요구 사항

`gradient_checkpointing` 및 `mixed_precision`을 사용하면 단일 24GB GPU에서 모델을 파인튜닝할 수 있습니다. 더 높은 `batch_size`와 더 빠른 훈련을 위해서는 GPU 메모리가 30GB 이상인 GPU를 사용하는 것이 좋습니다. TPU 또는 GPU에서 파인튜닝을 위해 JAX나 Flax를 사용할 수도 있습니다. 자세한 내용은 [아래](#flax-jax-finetuning)를 참조하세요.

xFormers로 memory efficient attention을 활성화하여 메모리 사용량 훨씬 더 줄일 수 있습니다. [xFormers가 설치](./optimization/xformers)되어 있는지 확인하고 `--enable_xformers_memory_efficient_attention`를 학습 스크립트에 명시합니다.

xFormers는 Flax에 사용할 수 없습니다.

## Hub에 모델 업로드하기

학습 스크립트에 다음 인수를 추가하여 모델을 허브에 저장합니다:

```bash
  --push_to_hub
```


## 체크포인트 저장 및 불러오기

학습 중 발생할 수 있는 일에 대비하여 정기적으로 체크포인트를 저장해 두는 것이 좋습니다. 체크포인트를 저장하려면 학습 스크립트에 다음 인수를 명시합니다.

```bash
  --checkpointing_steps=500
```

500스텝마다 전체 학습 state가 'output_dir'의 하위 폴더에 저장됩니다. 체크포인트는 'checkpoint-'에 지금까지 학습된 step 수입니다. 예를 들어 'checkpoint-1500'은 1500 학습 step 후에 저장된 체크포인트입니다.

학습을 재개하기 위해 체크포인트를 불러오려면 '--resume_from_checkpoint' 인수를 학습 스크립트에 명시하고 재개할 체크포인트를 지정하십시오. 예를 들어 다음 인수는 1500개의 학습 step 후에 저장된 체크포인트에서부터 훈련을 재개합니다.

```bash
  --resume_from_checkpoint="checkpoint-1500"
```

## 파인튜닝

<frameworkcontent>
<pt>
다음과 같이 [Naruto BLIP 캡션](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) 데이터셋에서 파인튜닝 실행을 위해 [PyTorch 학습 스크립트](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)를 실행합니다:


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sd-naruto-model"
```

자체 데이터셋으로 파인튜닝하려면 🤗 [Datasets](https://huggingface.co/docs/datasets/index)에서 요구하는 형식에 따라 데이터셋을 준비하세요. [데이터셋을 허브에 업로드](https://huggingface.co/docs/datasets/image_dataset#upload-dataset-to-the-hub)하거나 [파일들이 있는 로컬 폴더를 준비](https ://huggingface.co/docs/datasets/image_dataset#imagefolder)할 수 있습니다.

사용자 커스텀 loading logic을 사용하려면 스크립트를 수정하십시오. 도움이 되도록 코드의 적절한 위치에 포인터를 남겼습니다. 🤗 아래 예제 스크립트는 `TRAIN_DIR`의 로컬 데이터셋으로를 파인튜닝하는 방법과 `OUTPUT_DIR`에서 모델을 저장할 위치를 보여줍니다:


```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export TRAIN_DIR="path_to_your_dataset"
export OUTPUT_DIR="path_to_save_model"

accelerate launch train_text_to_image.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=${OUTPUT_DIR}
```

</pt>
<jax>
[@duongna211](https://github.com/duongna21)의 기여로, Flax를 사용해 TPU 및 GPU에서 Stable Diffusion 모델을 더 빠르게 학습할 수 있습니다. 이는 TPU 하드웨어에서 매우 효율적이지만 GPU에서도 훌륭하게 작동합니다. Flax 학습 스크립트는 gradient checkpointing나 gradient accumulation과 같은 기능을 아직 지원하지 않으므로 메모리가 30GB 이상인 GPU 또는 TPU v3가 필요합니다.

스크립트를 실행하기 전에 요구 사항이 설치되어 있는지 확인하십시오:

```bash
pip install -U -r requirements_flax.txt
```

그러면 다음과 같이 [Flax 학습 스크립트](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_flax.py)를 실행할 수 있습니다.

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

python train_text_to_image_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$dataset_name \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-naruto-model"
```

자체 데이터셋으로 파인튜닝하려면 🤗 [Datasets](https://huggingface.co/docs/datasets/index)에서 요구하는 형식에 따라 데이터셋을 준비하세요. [데이터셋을 허브에 업로드](https://huggingface.co/docs/datasets/image_dataset#upload-dataset-to-the-hub)하거나 [파일들이 있는 로컬 폴더를 준비](https ://huggingface.co/docs/datasets/image_dataset#imagefolder)할 수 있습니다.

사용자 커스텀 loading logic을 사용하려면 스크립트를 수정하십시오. 도움이 되도록 코드의 적절한 위치에 포인터를 남겼습니다. 🤗 아래 예제 스크립트는 `TRAIN_DIR`의 로컬 데이터셋으로를 파인튜닝하는 방법을 보여줍니다:

```bash
export MODEL_NAME="duongna/stable-diffusion-v1-4-flax"
export TRAIN_DIR="path_to_your_dataset"

python train_text_to_image_flax.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DIR \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --mixed_precision="fp16" \
  --max_train_steps=15000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --output_dir="sd-naruto-model"
```
</jax>
</frameworkcontent>

## LoRA

Text-to-image 모델 파인튜닝을 위해, 대규모 모델 학습을 가속화하기 위한 파인튜닝 기술인 LoRA(Low-Rank Adaptation of Large Language Models)를 사용할 수 있습니다. 자세한 내용은 [LoRA 학습](lora#text-to-image) 가이드를 참조하세요.

## 추론

허브의 모델 경로 또는 모델 이름을 [`StableDiffusionPipeline`]에 전달하여 추론을 위해 파인 튜닝된 모델을 불러올 수 있습니다:

<frameworkcontent>
<pt>
```python
from diffusers import StableDiffusionPipeline

model_path = "path_to_saved_model"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("yoda-naruto.png")
```
</pt>
<jax>
```python
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

model_path = "path_to_saved_model"
pipe, params = FlaxStableDiffusionPipeline.from_pretrained(model_path, dtype=jax.numpy.bfloat16)

prompt = "yoda naruto"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("yoda-naruto.png")
```
</jax>
</frameworkcontent>