# الانعكاس النصي

[[open-in-colab]]

تدعم [`StableDiffusionPipeline`] الانعكاس النصي، وهي تقنية تُمكّن نموذجًا مثل Stable Diffusion من تعلم مفهوم جديد من مجرد بضع صور عينة. يمنحك ذلك تحكمًا أكبر في الصور المولدة ويسمح لك بتكييف النموذج مع مفاهيم محددة. يمكنك البدء بسرعة باستخدام مجموعة من المفاهيم التي أنشأها المجتمع في [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer).

سيوضح هذا الدليل كيفية تشغيل الاستدلال باستخدام الانعكاس النصي باستخدام مفهوم مُتعلم مسبقًا من Stable Diffusion Conceptualizer. إذا كنت مهتمًا بتعليم النماذج مفاهيم جديدة باستخدام الانعكاس النصي، فراجع دليل التدريب [Textual Inversion](../training/text_inversion).

استورد المكتبات الضرورية:

```py
import torch
from diffusers import StableDiffusionPipeline
from diffusers.utils import make_image_grid
```

## Stable Diffusion 1 و 2

اختر نقطة تفتيش Stable Diffusion ومفهومًا مُتعلمًا مسبقًا من [Stable Diffusion Conceptualizer](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer):

```py
pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
repo_id_embeds = "sd-concepts-library/cat-toy"
```

الآن يمكنك تحميل خط أنابيب، ومرور المفهوم المُتعلم مسبقًا إليه:

```py
pipeline = StableDiffusionPipeline.from_pretrained(
pretrained_model_name_or_path, torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

pipeline.load_textual_inversion(repo_id_embeds)
```

قم بإنشاء موجه باستخدام المفهوم المُتعلم مسبقًا عن طريق استخدام رمز المحل الخاص `<cat-toy>`، واختر عدد العينات وصفوف الصور التي تريد توليدها:

```py
prompt = "a grafitti in a favela wall with a <cat-toy> on it"

num_samples_per_row = 2
num_rows = 2
```

بعد ذلك، قم بتشغيل خط الأنابيب (يمكنك ضبط المعلمات مثل `num_inference_steps` و`guidance_scale` لمعرفة تأثيرها على جودة الصورة)، وقم بحفظ الصور المولدة وعرضها باستخدام دالة المساعدة التي قمت بإنشائها في البداية:

```py
all_images = []
for _ in range(num_rows):
images = pipeline(prompt, num_images_per_prompt=num_samples_per_row, num_inference_steps=50, guidance_scale=7.5).images
all_images.extend(images)

grid = make_image_grid(all_images, num_rows, num_samples_per_row)
grid
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/textual_inversion_inference.png">
</div>

## Stable Diffusion XL

يمكن لـ Stable Diffusion XL (SDXL) أيضًا استخدام متجهات الانعكاس النصي للاستدلال. على عكس Stable Diffusion 1 و 2، يحتوي SDXL على مشفرين نصيين، لذلك ستحتاج إلى انعكاسين نصيين - واحد لكل مشفر نصي.

دعونا نقوم بتنزيل تضمين الانعكاس النصي SDXL وإلقاء نظرة فاحصة على بنيته:

```py
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

file = hf_hub_download("dn118/unaestheticXL", filename="unaestheticXLv31.safetensors")
state_dict = load_file(file)
state_dict
```

```
{'clip_g': tensor([[ 0.0077, -0.0112,  0.0065,  ...,  0.0195,  0.0159,  0.0275],
...,
[-0.0170,  0.0213,  0.0143,  ..., -0.0302, -0.0240, -0.0362]],
'clip_l': tensor([[ 0.0023,  0.0192,  0.0213,  ..., -0.0385,  0.0048, -0.0011],
...,
[ 0.0475, -0.0508, -0.0145,  ...,  0.0070, -0.0089, -0.0163]],
```

هناك موترين، `"clip_g"` و `"clip_l"`.
يرتبط `"clip_g"` بمشفر النص الأكبر في SDXL ويشير إلى
`pipe.text_encoder_2` ويشير `"clip_l"` إلى `pipe.text_encoder`.

الآن يمكنك تحميل كل موتر بشكل منفصل عن طريق تمريره مع مشفر النص ومحلل الصحيح
إلى [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

pipe.load_textual_inversion(state_dict["clip_g"], token="unaestheticXLv31", text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)
pipe.load_textual_inversion(state_dict["clip_l"], token="unaestheticXLv31", text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)

# يجب استخدام التضمين كإدخال سلبي، لذا نمرره كسلسلة سلبية
generator = torch.Generator().manual_seed(33)
image = pipe("a woman standing in front of a mountain", negative_prompt="unaestheticXLv31", generator=generator).images[0]
image
```