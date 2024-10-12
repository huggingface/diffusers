# Stable Diffusion 2

Stable Diffusion 2 هو نموذج تحويل النص إلى صورة _latent diffusion_ مبني على عمل [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) الأصلي، بقيادة Robin Rombach وKatherine Crowson من [Stability AI](https://stability.ai/) و[LAION](https://laion.ai/).

*يتضمن إصدار Stable Diffusion 2.0 نماذج قوية للتحويل من نص إلى صورة تم تدريبها باستخدام مشفر نصي جديد تمامًا (OpenCLIP)، طورته LAION بدعم من Stability AI، والذي يحسن بشكل كبير جودة الصور المولدة مقارنة بإصدارات V1 السابقة. يمكن لنماذج تحويل النص إلى صورة في هذا الإصدار توليد صور بدقة افتراضية تبلغ 512x512 بكسل و768x768 بكسل.*

تم تدريب هذه النماذج على مجموعة فرعية جمالية من [مجموعة بيانات LAION-5B](https://laion.ai/blog/laion-5b/) التي أنشأها فريق DeepFloyd في Stability AI، والتي تم تصفيتها بعد ذلك لإزالة المحتوى الخاص بالبالغين باستخدام [مرشح LAION NSFW](https://openreview.net/forum?id=M3Y74vmsMcY).

للحصول على مزيد من التفاصيل حول كيفية عمل Stable Diffusion 2 وكيف يختلف عن Stable Diffusion الأصلي، يرجى الرجوع إلى منشور الإعلان الرسمي [هنا](https://stability.ai/blog/stable-diffusion-v2-release).

تتشابه بنية Stable Diffusion 2 إلى حد كبير مع نموذج Stable Diffusion [الأصلي](./text2img) لذا يرجى الاطلاع على وثائق API الخاصة به لمعرفة كيفية استخدام Stable Diffusion 2. نوصي باستخدام [`DPMSolverMultistepScheduler`] حيث يوفر توازنًا معقولًا بين السرعة والجودة ويمكن تشغيله في خطوات قليلة تصل إلى 20 خطوة.

Stable Diffusion 2 متاح لمهام مثل تحويل النص إلى صورة، وملء الفراغات في الصورة، وزيادة دقة الصورة، وتحويل العمق إلى صورة:

| المهمة                    | المستودع                                                                                                    |
|-------------------------|---------------------------------------------------------------------------------------------------------------|
| تحويل النص إلى صورة (512x512) | [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base)             |
| تحويل النص إلى صورة (768x768) | [stabilityai/stable-diffusion-2](https://huggingface.co/stabilityai/stable-diffusion-2)                       |
| ملء الفراغات في الصورة              | [stabilityai/stable-diffusion-2-inpainting](https://huggingface.co/stabilityai/stable-diffusion-2-inpainting) |
| زيادة دقة الصورة        | [stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)               |
| تحويل العمق إلى صورة          | [stabilityai/stable-diffusion-2-depth](https://huggingface.co/stabilityai/stable-diffusion-2-depth)           |

فيما يلي بعض الأمثلة على كيفية استخدام Stable Diffusion 2 لكل مهمة:

<Tip>

تأكد من الاطلاع على قسم [نصائح Stable Diffusion](overview#tips) لمعرفة كيفية استكشاف التوازن بين سرعة المجدول والجودة، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!

إذا كنت مهتمًا باستخدام إحدى نقاط التفتيش الرسمية لمهمة ما، فاستكشف منظمات [CompVis](https://huggingface.co/CompVis) و[Runway](https://huggingface.co/runwayml) و[Stability AI](https://huggingface.co/stabilityai) Hub!

</Tip>

## تحويل النص إلى صورة

```py
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

repo_id = "stabilityai/stable-diffusion-2-base"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "High quality photo of an astronaut riding a horse in space"
image = pipe(prompt, num_inference_steps=25).images[0]
image
```

## ملء الفراغات في الصورة

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import load_image, make_image_grid

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

repo_id = "stabilityai/stable-diffusion-2-inpainting"
pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, revision="fp16")

pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image, num_inference_steps=25).images[0]
make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

## زيادة دقة الصورة

```py
from diffusers import StableDiffusionUpscalePipeline
from diffusers.utils import load_image, make_image_grid
import torch

# تحميل النموذج والمجدول
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")

# قم بتنزيل صورة
url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
low_res_img = load_image(url)
low_res_img = low_res_img.resize((128, 128))
prompt = "a white cat"
upscaled_image = pipeline(prompt=prompt, image=low_res_img).images[0]
make_image_grid([low_res_img.resize((512, 512)), upscaled_image.resize((512, 512))], rows=1, cols=2)
```

## تحويل العمق إلى صورة

```py
import torch
from diffusers import StableDiffusionDepth2ImgPipeline
from diffusers.utils import load_image, make_image_grid

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-depth",
    torch_dtype=torch.float16,
).to("cuda")


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
init_image = load_image(url)
prompt = "two tigers"
negative_prompt = "bad, deformed, ugly, bad anatomy"
image = pipe(prompt=prompt, image=init_image, negative_prompt=negative_prompt, strength=0.7).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```