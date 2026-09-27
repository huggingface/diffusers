<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 어댑터 불러오기

[[open-in-colab]]

특정 물체의 이미지 또는 특정 스타일의 이미지를 생성하도록 diffusion 모델을 개인화하기 위한 몇 가지 [학습](../training/overview) 기법이 있습니다. 이러한 학습 방법은 각각 다른 유형의 어댑터를 생성합니다. 일부 어댑터는 완전히 새로운 모델을 생성하는 반면, 다른 어댑터는 임베딩 또는 가중치의 작은 부분만 수정합니다. 이는 각 어댑터의 로딩 프로세스도 다르다는 것을 의미합니다.

이 가이드에서는 DreamBooth, textual inversion 및 LoRA 가중치를 불러오는 방법을 설명합니다.

> [!TIP]
> 사용할 체크포인트와 임베딩은 [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer), [LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer), [Diffusers Models Gallery](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery)에서 찾아보시기 바랍니다.

## DreamBooth

[DreamBooth](https://dreambooth.github.io/)는 물체의 여러 이미지에 대한 *diffusion 모델 전체*를 미세 조정하여 새로운 스타일과 설정으로 해당 물체의 이미지를 생성합니다. 이 방법은 모델이 물체 이미지와 연관시키는 방법을 학습하는 프롬프트에 특수 단어를 사용하는 방식으로 작동합니다. 모든 학습 방법 중에서 드림부스는 전체 체크포인트 모델이기 때문에 파일 크기가 가장 큽니다(보통 몇 GB).

Hergé가 그린 단 10개의 이미지로 학습된 [herge_style](https://huggingface.co/sd-dreambooth-library/herge-style) 체크포인트를 불러와 해당 스타일의 이미지를 생성해 보겠습니다. 이 모델이 작동하려면 체크포인트를 트리거하는 프롬프트에 특수 단어 `herge_style`을 포함시켜야 합니다:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("sd-dreambooth-library/herge-style", dtype=torch.float16).to("cuda")
prompt = "A cute herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_dreambooth.png" />
</div>

## Textual inversion

[Textual inversion](https://textual-inversion.github.io/)은 DreamBooth와 매우 유사하며 몇 개의 이미지만으로 특정 개념(스타일, 개체)을 생성하는 diffusion 모델을 개인화할 수도 있습니다. 이 방법은 프롬프트에 특정 단어를 입력하면 해당 이미지를 나타내는 새로운 임베딩을 학습하고 찾아내는 방식으로 작동합니다. 결과적으로 diffusion 모델 가중치는 동일하게 유지되고 훈련 프로세스는 비교적 작은(수 KB) 파일을 생성합니다.

Textual inversion은 임베딩을 생성하기 때문에 DreamBooth처럼 단독으로 사용할 수 없으며 또 다른 모델이 필요합니다.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16).to("cuda")
```

이제 [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] 메서드를 사용하여 textual inversion 임베딩을 불러와 이미지를 생성할 수 있습니다. [sd-concepts-library/gta5-artwork](https://huggingface.co/sd-concepts-library/gta5-artwork) 임베딩을 불러와 보겠습니다. 이를 트리거하려면 프롬프트에 특수 단어 `<gta5-artwork>`를 포함시켜야 합니다:

```py
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, <gta5-artwork> style"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_txt_embed.png" />
</div>

Textual inversion은 또한 바람직하지 않은 사물에 대해 *네거티브 임베딩*을 생성하여 모델이 흐릿한 이미지나 손의 추가 손가락과 같은 바람직하지 않은 사물이 포함된 이미지를 생성하지 못하도록 학습할 수도 있습니다. 이는 프롬프트를 빠르게 개선하는 것이 쉬운 방법이 될 수 있습니다. 이는 이전과 같이 임베딩을 [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]으로 불러오지만 이번에는 두 개의 매개변수가 더 필요합니다:

- `weight_name`: 파일이 특정 이름의 🤗 Diffusers 형식으로 저장된 경우이거나 파일이 A1111 형식으로 저장된 경우, 불러올 가중치 파일을 지정합니다.
- `token`: 임베딩을 트리거하기 위해 프롬프트에서 사용할 특수 단어를 지정합니다.

[sayakpaul/EasyNegative-test](https://huggingface.co/sayakpaul/EasyNegative-test) 임베딩을 불러와 보겠습니다:

```py
pipeline.load_textual_inversion(
    "sayakpaul/EasyNegative-test", weight_name="EasyNegative.safetensors", token="EasyNegative"
)
```

이제 `token`을 사용해 네거티브 임베딩이 있는 이미지를 생성할 수 있습니다:

```py
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, EasyNegative"
negative_prompt = "EasyNegative"

image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png" />
</div>

## LoRA

[Low-Rank Adaptation (LoRA)](https://huggingface.co/papers/2106.09685)은 속도가 빠르고 파일 크기가 (수백 MB로) 작기 때문에 널리 사용되는 학습 기법입니다. 이 가이드의 다른 방법과 마찬가지로, LoRA는 몇 장의 이미지만으로 새로운 스타일을 학습하도록 모델을 학습시킬 수 있습니다. 이는 diffusion 모델에 새로운 가중치를 삽입한 다음 전체 모델 대신 새로운 가중치만 학습시키는 방식으로 작동합니다. 따라서 LoRA를 더 빠르게 학습시키고 더 쉽게 저장할 수 있습니다.

> [!TIP]
> LoRA는 다른 학습 방법과 함께 사용할 수 있는 매우 일반적인 학습 기법입니다. 예를 들어, DreamBooth와 LoRA로 모델을 학습하는 것이 일반적입니다. 또한 새롭고 고유한 이미지를 생성하기 위해 여러 개의 LoRA를 불러오고 병합하는 것이 점점 더 일반화되고 있습니다. 병합은 이 불러오기 가이드의 범위를 벗어나므로 자세한 내용은 심층적인 [LoRA 병합](merge_loras) 가이드에서 확인할 수 있습니다.

LoRA는 다른 모델과 함께 사용해야 합니다:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16).to("cuda")
```

그리고 [`~loaders.LoraLoaderMixin.load_lora_weights`] 메서드를 사용하여 [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora) 가중치를 불러오고 리포지토리에서 가중치 파일명을 지정합니다:

```py
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors")
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_lora.png" />
</div>

[`~loaders.LoraLoaderMixin.load_lora_weights`] 메서드는 LoRA 가중치를 UNet과 텍스트 인코더에 모두 불러옵니다. 이 메서드는 해당 케이스에서 LoRA를 불러오는 데 선호되는 방식입니다:

- LoRA 가중치에 UNet 및 텍스트 인코더에 대한 별도의 식별자가 없는 경우
- LoRA 가중치에 UNet과 텍스트 인코더에 대한 별도의 식별자가 있는 경우

하지만 LoRA 가중치만 UNet에 로드해야 하는 경우에는 [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] 메서드를 사용할 수 있습니다. [jbilcke-hf/sdxl-cinematic-1](https://huggingface.co/jbilcke-hf/sdxl-cinematic-1) LoRA를 불러와 보겠습니다:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")

# 프롬프트에서 cnmt를 사용하여 LoRA를 트리거합니다.
prompt = "A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_attn_proc.png" />
</div>

LoRA 가중치를 언로드하려면 [`~loaders.LoraLoaderMixin.unload_lora_weights`] 메서드를 사용하여 LoRA 가중치를 삭제하고 모델을 원래 가중치로 복원합니다:

```py
pipeline.unload_lora_weights()
```

### LoRA 가중치 스케일 조정하기

[`~loaders.LoraLoaderMixin.load_lora_weights`] 및 [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] 모두 `cross_attention_kwargs={"scale": 0.5}` 파라미터를 전달하여 얼마나 LoRA 가중치를 사용할지 조정할 수 있습니다. 값이 `0`이면 기본 모델 가중치만 사용하는 것과 같고, 값이 `1`이면 완전히 미세 조정된 LoRA를 사용하는 것과 같습니다.

레이어당 사용되는 LoRA 가중치의 양을 보다 세밀하게 제어하려면 [`~loaders.LoraLoaderMixin.set_adapters`]를 사용하여 각 레이어의 가중치를 얼마만큼 조정할지 지정하는 딕셔너리를 전달할 수 있습니다.
```python
pipe = ... # 파이프라인 생성
pipe.load_lora_weights(..., adapter_name="my_adapter")
scales = {
    "text_encoder": 0.5,
    "text_encoder_2": 0.5,  # 파이프에 두 번째 텍스트 인코더가 있는 경우에만 사용 가능
    "unet": {
        "down": 0.9,  # down 부분의 모든 트랜스포머는 스케일 0.9를 사용
        # "mid"  # 이 예제에서는 "mid"가 지정되지 않았으므로 중간 부분의 모든 트랜스포머는 기본 스케일 1.0을 사용
        "up": {
            "block_0": 0.6,  # # up의 0번째 블록에 있는 3개의 트랜스포머는 모두 스케일 0.6을 사용
            "block_1": [0.4, 0.8, 1.0],  # up의 첫 번째 블록에 있는 3개의 트랜스포머는 각각 스케일 0.4, 0.8, 1.0을 사용
        }
    }
}
pipe.set_adapters("my_adapter", scales)
```

이는 여러 어댑터에서도 작동합니다. 방법은 [이 가이드](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#customize-adapters-strength)를 참조하세요.

> [!WARNING]
> 현재 [`~loaders.LoraLoaderMixin.set_adapters`]는 어텐션 가중치의 스케일링만 지원합니다. LoRA에 다른 부분(예: resnets or down-/upsamplers)이 있는 경우 1.0의 스케일을 유지합니다.

### Kohya와 TheLastBen

커뮤니티에서 인기 있는 다른 LoRA trainer로는 [Kohya](https://github.com/kohya-ss/sd-scripts/)와 [TheLastBen](https://github.com/TheLastBen/fast-stable-diffusion)의 trainer가 있습니다. 이 trainer들은 🤗 Diffusers가 훈련한 것과는 다른 LoRA 체크포인트를 생성하지만, 같은 방식으로 불러올 수 있습니다.

<hfoptions id="other-trainers">
<hfoption id="Kohya">

Kohya LoRA를 불러오기 위해, 예시로 [Civitai](https://civitai.com/)에서 [Blueprintify SD XL 1.0](https://civitai.com/models/150986/blueprintify-sd-xl-10) 체크포인트를 다운로드합니다:

```sh
!wget https://civitai.com/api/download/models/168776 -O blueprintify-sd-xl-10.safetensors
```

LoRA 체크포인트를 [`~loaders.LoraLoaderMixin.load_lora_weights`] 메서드로 불러오고 `weight_name` 파라미터에 파일명을 지정합니다:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/weights", weight_name="blueprintify-sd-xl-10.safetensors")
```

이미지를 생성합니다:

```py
# LoRA를 트리거하기 위해 bl3uprint를 프롬프트에 사용
prompt = "bl3uprint, a highly detailed blueprint of the eiffel tower, explaining how to build all parts, many txt, blueprint grid backdrop"
image = pipeline(prompt).images[0]
image
```

> [!WARNING]
> Kohya LoRA를 🤗 Diffusers와 함께 사용할 때 몇 가지 제한 사항이 있습니다:
>
> - [여기](https://github.com/huggingface/diffusers/pull/4287/#issuecomment-1655110736)에 설명된 여러 가지 이유로 인해 이미지가 ComfyUI와 같은 UI에서 생성된 이미지와 다르게 보일 수 있습니다.
> - [LyCORIS 체크포인트](https://github.com/KohakuBlueleaf/LyCORIS)가 완전히 지원되지 않습니다. [`~loaders.LoraLoaderMixin.load_lora_weights`] 메서드는 LoRA 및 LoCon 모듈로 LyCORIS 체크포인트를 불러올 수 있지만, Hada 및 LoKR은 지원되지 않습니다.

</hfoption>
<hfoption id="TheLastBen">

TheLastBen에서 체크포인트를 불러오는 방법은 매우 유사합니다. 예를 들어, [TheLastBen/William_Eggleston_Style_SDXL](https://huggingface.co/TheLastBen/William_Eggleston_Style_SDXL) 체크포인트를 불러오려면:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("TheLastBen/William_Eggleston_Style_SDXL", weight_name="wegg.safetensors")

# LoRA를 트리거하기 위해 william eggleston를 프롬프트에 사용
prompt = "a house by william eggleston, sunrays, beautiful, sunlight, sunrays, beautiful"
image = pipeline(prompt=prompt).images[0]
image
```

</hfoption>
</hfoptions>

## IP-Adapter

[IP-Adapter](https://ip-adapter.github.io/)는 모든 diffusion 모델에 이미지 프롬프트를 사용할 수 있는 경량 어댑터입니다. 이 어댑터는 이미지와 텍스트 feature의 cross-attention 레이어를 분리하여 작동합니다. 다른 모든 모델 컴포넌트튼 freeze되고 UNet의 embedded 이미지 features만 학습됩니다. 따라서 IP-Adapter 파일은 일반적으로 최대 100MB에 불과합니다.

다양한 작업과 구체적인 사용 사례에 IP-Adapter를 사용하는 방법에 대한 자세한 내용은 [IP-Adapter](../using-diffusers/ip_adapter) 가이드에서 확인할 수 있습니다.

> [!TIP]
> Diffusers는 현재 가장 많이 사용되는 일부 파이프라인에 대해서만 IP-Adapter를 지원합니다. 멋진 사용 사례가 있는 지원되지 않는 파이프라인에 IP-Adapter를 통합하고 싶다면 언제든지 기능 요청을 여세요!
> 공식 IP-Adapter 체크포인트는 [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)에서 확인할 수 있습니다.

시작하려면 Stable Diffusion 체크포인트를 불러오세요.

```py
from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", dtype=torch.float16).to("cuda")
```

그런 다음 IP-Adapter 가중치를 불러와 [`~loaders.IPAdapterMixin.load_ip_adapter`] 메서드를 사용하여 파이프라인에 추가합니다.

```py
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
```

불러온 뒤, 이미지 및 텍스트 프롬프트가 있는 파이프라인을 사용하여 이미지 생성 프로세스를 가이드할 수 있습니다.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
    prompt='best quality, high quality, wearing sunglasses',
    ip_adapter_image=image,
    negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
    num_inference_steps=50,
    generator=generator,
).images[0]
images
```

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip-bear.png" />
</div>

### IP-Adapter Plus

IP-Adapter는 이미지 인코더를 사용하여 이미지 feature를 생성합니다. IP-Adapter 리포지토리에 `image_encoder` 하위 폴더가 있는 경우, 이미지 인코더가 자동으로 불러와 파이프라인에 등록됩니다. 그렇지 않은 경우, [`~transformers.CLIPVisionModelWithProjection`] 모델을 사용하여 이미지 인코더를 명시적으로 불러와 파이프라인에 전달해야 합니다.

이는 ViT-H 이미지 인코더를 사용하는 *IP-Adapter Plus* 체크포인트에 해당하는 케이스입니다.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter",
    subfolder="models/image_encoder",
    dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    image_encoder=image_encoder,
    dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
```

### IP-Adapter Face ID 모델

IP-Adapter FaceID 모델은 CLIP 이미지 임베딩 대신 `insightface`에서 생성한 이미지 임베딩을 사용하는 실험적인 IP Adapter입니다. 이러한 모델 중 일부는 LoRA를 사용하여 ID 일관성을 개선하기도 합니다.
이러한 모델을 사용하려면 `insightface`와 해당 요구 사항을 모두 설치해야 합니다.

> [!WARNING]
> InsightFace 사전학습된 모델은 비상업적 연구 목적으로만 사용할 수 있으므로, IP-Adapter-FaceID 모델은 연구 목적으로만 릴리즈되었으며 상업적 용도로는 사용할 수 없습니다.

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None)
```

두 가지 IP 어댑터 FaceID Plus 모델 중 하나를 사용하려는 경우, 이 모델들은 더 나은 사실감을 얻기 위해 `insightface`와 CLIP 이미지 임베딩을 모두 사용하므로, CLIP 이미지 인코더도 불러와야 합니다.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    image_encoder=image_encoder,
    dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plus_sd15.bin")
```
