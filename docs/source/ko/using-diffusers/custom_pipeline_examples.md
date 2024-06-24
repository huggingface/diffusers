<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# 커뮤니티 파이프라인

> **커뮤니티 파이프라인에 대한 자세한 내용은 [이 이슈](https://github.com/huggingface/diffusers/issues/841)를 참조하세요.

**커뮤니티** 예제는 커뮤니티에서 추가한 추론 및 훈련 예제로 구성되어 있습니다.
다음 표를 참조하여 모든 커뮤니티 예제에 대한 개요를 확인하시기 바랍니다. **코드 예제**를 클릭하면 복사하여 붙여넣기할 수 있는 코드 예제를 확인할 수 있습니다.
커뮤니티가 예상대로 작동하지 않는 경우 이슈를 개설하고 작성자에게 핑을 보내주세요.

| 예                                     | 설명                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 코드 예제                                                                         | 콜랩                                                                                                                                                                                                              |저자                                                         |
|:---------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------:|
| CLIP Guided Stable Diffusion           | CLIP 가이드 기반의 Stable Diffusion으로 텍스트에서 이미지로 생성하기                                                                                                                                                                                                                                                                                                                                                                                                                                | [CLIP Guided Stable Diffusion](#clip-guided-stable-diffusion)                    | [![콜랩에서 열기](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/CLIP_Guided_Stable_diffusion_with_diffusers.ipynb)  | [Suraj Patil](https://github.com/patil-suraj/)             |
| One Step U-Net (Dummy)                 | 커뮤니티 파이프라인을 어떻게 사용해야 하는지에 대한 예시(참고 https://github.com/huggingface/diffusers/issues/841)                                                                                                                                                                                                                                                                                                                                                                                   | [One Step U-Net](#one-step-unet)                                                 | -                                                                                                                                                                                                                  | [Patrick von Platen](https://github.com/patrickvonplaten/) |
| Stable Diffusion Interpolation         | 서로 다른 프롬프트/시드 간 Stable Diffusion의 latent space 보간                                                                                                                                                                                                                                                                                                                                                                                                                                    | [Stable Diffusion Interpolation](#stable-diffusion-interpolation)                | -                                                                                                                                                                                                                  | [Nate Raw](https://github.com/nateraw/)                    |
| Stable Diffusion Mega                  | 모든 기능을 갖춘 **하나의** Stable Diffusion 파이프라인 [Text2Image](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py), [Image2Image](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py) and [Inpainting](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py) | [Stable Diffusion Mega](#stable-diffusion-mega)                                  | -                                                                                                                                                                                                                  | [Patrick von Platen](https://github.com/patrickvonplaten/) |
| Long Prompt Weighting Stable Diffusion | 토큰 길이 제한이 없고 프롬프트에서 파싱 가중치 지원을 하는 **하나의** Stable Diffusion 파이프라인,                                                                                                                                                                                                                                                                                                                                                                                                    | [Long Prompt Weighting Stable Diffusion](#long-prompt-weighting-stable-diffusion) |-                                                                                                                                                                                                                  | [SkyTNT](https://github.com/SkyTNT)                        |
| Speech to Image                        | 자동 음성 인식을 사용하여 텍스트를 작성하고 Stable Diffusion을 사용하여 이미지를 생성합니다.                                                                                                                                                                                                                                                                                                                                                                                                          | [Speech to Image](#speech-to-image)                                               | -                                                                                                                                                                                                                 | [Mikail Duzenli](https://github.com/MikailINTech)          |

커스텀 파이프라인을 불러오려면 `diffusers/examples/community`에 있는 파일 중 하나로서 `custom_pipeline` 인수를 `DiffusionPipeline`에 전달하기만 하면 됩니다. 자신만의 파이프라인이 있는 PR을 보내주시면 빠르게 병합해드리겠습니다.
```py
pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", custom_pipeline="filename_in_the_community_folder"
)
```

## 사용 예시

### CLIP 가이드 기반의 Stable Diffusion

모든 노이즈 제거 단계에서 추가 CLIP 모델을 통해 Stable Diffusion을 가이드함으로써 CLIP 모델 기반의 Stable Diffusion은 보다 더 사실적인 이미지를 생성을 할 수 있습니다.

다음 코드는 약 12GB의 GPU RAM이 필요합니다.

```python
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel
import torch


feature_extractor = CLIPImageProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")
clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16)


guided_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="clip_guided_stable_diffusion",
    clip_model=clip_model,
    feature_extractor=feature_extractor,
    torch_dtype=torch.float16,
)
guided_pipeline.enable_attention_slicing()
guided_pipeline = guided_pipeline.to("cuda")

prompt = "fantasy book cover, full moon, fantasy forest landscape, golden vector elements, fantasy magic, dark light night, intricate, elegant, sharp focus, illustration, highly detailed, digital painting, concept art, matte, art by WLOP and Artgerm and Albert Bierstadt, masterpiece"

generator = torch.Generator(device="cuda").manual_seed(0)
images = []
for i in range(4):
    image = guided_pipeline(
        prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
        clip_guidance_scale=100,
        num_cutouts=4,
        use_cutouts=False,
        generator=generator,
    ).images[0]
    images.append(image)

# 이미지 로컬에 저장하기
for i, img in enumerate(images):
    img.save(f"./clip_guided_sd/image_{i}.png")
```

이미지` 목록에는 로컬에 저장하거나 구글 콜랩에 직접 표시할 수 있는 PIL 이미지 목록이 포함되어 있습니다. 생성된 이미지는 기본적으로 안정적인 확산을 사용하는 것보다 품질이 높은 경향이 있습니다. 예를 들어 위의 스크립트는 다음과 같은 이미지를 생성합니다:

![clip_guidance](https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/clip_guidance/merged_clip_guidance.jpg).

### One Step Unet

예시 "one-step-unet"는 다음과 같이 실행할 수 있습니다.

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("google/ddpm-cifar10-32", custom_pipeline="one_step_unet")
pipe()
```

**참고**: 이 커뮤니티 파이프라인은 기능으로 유용하지 않으며 커뮤니티 파이프라인을 추가할 수 있는 방법의 예시일 뿐입니다(https://github.com/huggingface/diffusers/issues/841 참조).

### Stable Diffusion Interpolation

다음 코드는 최소 8GB VRAM의 GPU에서 실행할 수 있으며 약 5분 정도 소요됩니다.

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    safety_checker=None,  # Very important for videos...lots of false positives while interpolating
    custom_pipeline="interpolate_stable_diffusion",
).to("cuda")
pipe.enable_attention_slicing()

frame_filepaths = pipe.walk(
    prompts=["a dog", "a cat", "a horse"],
    seeds=[42, 1337, 1234],
    num_interpolation_steps=16,
    output_dir="./dreams",
    batch_size=4,
    height=512,
    width=512,
    guidance_scale=8.5,
    num_inference_steps=50,
)
```

walk(...)` 함수의 출력은 `output_dir`에 정의된 대로 폴더에 저장된 이미지 목록을 반환합니다. 이 이미지를 사용하여 안정적으로 확산되는 동영상을 만들 수 있습니다.

> 안정된 확산을 이용한 동영상 제작 방법과 더 많은 기능에 대한 자세한 내용은 https://github.com/nateraw/stable-diffusion-videos 에서 확인하시기 바랍니다.

### Stable Diffusion Mega

The Stable Diffusion Mega 파이프라인을 사용하면 Stable Diffusion 파이프라인의 주요 사용 사례를 단일 클래스에서 사용할 수 있습니다.
```python
#!/usr/bin/env python3
from diffusers import DiffusionPipeline
import PIL
import requests
from io import BytesIO
import torch


def download_image(url):
    response = requests.get(url)
    return PIL.Image.open(BytesIO(response.content)).convert("RGB")


pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="stable_diffusion_mega",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
pipe.enable_attention_slicing()


### Text-to-Image

images = pipe.text2img("An astronaut riding a horse").images

### Image-to-Image

init_image = download_image(
    "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
)

prompt = "A fantasy landscape, trending on artstation"

images = pipe.img2img(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

### Inpainting

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"
init_image = download_image(img_url).resize((512, 512))
mask_image = download_image(mask_url).resize((512, 512))

prompt = "a cat sitting on a bench"
images = pipe.inpaint(prompt=prompt, image=init_image, mask_image=mask_image, strength=0.75).images
```

위에 표시된 것처럼 하나의 파이프라인에서 '텍스트-이미지 변환', '이미지-이미지 변환', '인페인팅'을 모두 실행할 수 있습니다.

### Long Prompt Weighting Stable Diffusion

파이프라인을 사용하면 77개의 토큰 길이 제한 없이 프롬프트를 입력할 수 있습니다. 또한 "()"를 사용하여 단어 가중치를 높이거나 "[]"를 사용하여 단어 가중치를 낮출 수 있습니다.
또한 파이프라인을 사용하면 단일 클래스에서 Stable Diffusion 파이프라인의 주요 사용 사례를 사용할 수 있습니다.

#### pytorch

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion", custom_pipeline="lpw_stable_diffusion", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "best_quality (1girl:1.3) bow bride brown_hair closed_mouth frilled_bow frilled_hair_tubes frills (full_body:1.3) fox_ear hair_bow hair_tubes happy hood japanese_clothes kimono long_sleeves red_bow smile solo tabi uchikake white_kimono wide_sleeves cherry_blossoms"
neg_prompt = "lowres, bad_anatomy, error_body, error_hair, error_arm, error_hands, bad_hands, error_fingers, bad_fingers, missing_fingers, error_legs, bad_legs, multiple_legs, missing_legs, error_lighting, error_shadow, error_reflection, text, error, extra_digit, fewer_digits, cropped, worst_quality, low_quality, normal_quality, jpeg_artifacts, signature, watermark, username, blurry"

pipe.text2img(prompt, negative_prompt=neg_prompt, width=512, height=512, max_embeddings_multiples=3).images[0]
```

#### onnxruntime

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="lpw_stable_diffusion_onnx",
    revision="onnx",
    provider="CUDAExecutionProvider",
)

prompt = "a photo of an astronaut riding a horse on mars, best quality"
neg_prompt = "lowres, bad anatomy, error body, error hair, error arm, error hands, bad hands, error fingers, bad fingers, missing fingers, error legs, bad legs, multiple legs, missing legs, error lighting, error shadow, error reflection, text, error, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

pipe.text2img(prompt, negative_prompt=neg_prompt, width=512, height=512, max_embeddings_multiples=3).images[0]
```

토큰 인덱스 시퀀스 길이가 이 모델에 지정된 최대 시퀀스 길이보다 길면(*** > 77). 이 시퀀스를 모델에서 실행하면 인덱싱 오류가 발생합니다`. 정상적인 현상이니 걱정하지 마세요.
### Speech to Image

다음 코드는 사전학습된 OpenAI whisper-small과 Stable Diffusion을 사용하여 오디오 샘플에서 이미지를 생성할 수 있습니다.
```Python
import torch

import matplotlib.pyplot as plt
from datasets import load_dataset
from diffusers import DiffusionPipeline
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


device = "cuda" if torch.cuda.is_available() else "cpu"

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

audio_sample = ds[3]

text = audio_sample["text"].lower()
speech_data = audio_sample["audio"]["array"]

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

diffuser_pipeline = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    custom_pipeline="speech_to_image_diffusion",
    speech_model=model,
    speech_processor=processor,

    torch_dtype=torch.float16,
)

diffuser_pipeline.enable_attention_slicing()
diffuser_pipeline = diffuser_pipeline.to(device)

output = diffuser_pipeline(speech_data)
plt.imshow(output.images[0])
```
위 예시는 다음의 결과 이미지를 보입니다.

![image](https://user-images.githubusercontent.com/45072645/196901736-77d9c6fc-63ee-4072-90b0-dc8b903d63e3.png)