# Outpainting

تتيح لك تقنية Outpainting توسيع الصورة خارج حدودها الأصلية، مما يتيح لك إضافة عناصر بصرية أو استبدالها أو تعديلها في الصورة مع الحفاظ على الصورة الأصلية. وكما هو الحال في [inpainting](../using-diffusers/inpaint)، تريد ملء المساحة البيضاء (في هذه الحالة، المساحة خارج الصورة الأصلية) بعناصر بصرية جديدة مع الحفاظ على الصورة الأصلية (التي يمثلها قناع من البكسلات السوداء). هناك عدة طرق للرسم خارج الإطار، مثل استخدام [ControlNet](https://hf.co/blog/OzzyGT/outpainting-controlnet) أو [Differential Diffusion](https://hf.co/blog/OzzyGT/outpainting-differential-diffusion).

سيوضح هذا الدليل كيفية استخدام تقنية الرسم خارج الإطار باستخدام نموذج inpainting، وControlNet، ومقدّر ZoeDepth.

قبل البدء، تأكد من تثبيت مكتبة [controlnet_aux](https://github.com/huggingface/controlnet_aux) حتى تتمكن من استخدام مقدّر ZoeDepth.

```py
!pip install -q controlnet_aux
```

## إعداد الصورة

ابدأ باختيار صورة للرسم خارج الإطار وإزالة الخلفية باستخدام مساحة مثل [BRIA-RMBG-1.4](https://hf.co/spaces/briaai/BRIA-RMBG-1.4).

<iframe
src="https://briaai-bria-rmbg-1-4.hf.space"
frameborder="0"
width="850"
height="450"
></iframe>

على سبيل المثال، قم بإزالة الخلفية من هذه الصورة لزوج من الأحذية.

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/original-jordan.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/no-background-jordan.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تمت إزالة الخلفية</figcaption>
</div>
</div>

تعمل نماذج [Stable Diffusion XL (SDXL)](../using-diffusers/sdxl) بشكل أفضل مع الصور بحجم 1024x1024، ولكن يمكنك تغيير حجم الصورة إلى أي حجم طالما أن عتادك لديه ذاكرة كافية لدعمه. يجب أيضًا استبدال الخلفية الشفافة في الصورة بخلفية بيضاء. قم بإنشاء دالة (مثل تلك الموجودة أدناه) تقوم بتصغير الصورة ولصقها على خلفية بيضاء.

```py
import random

import requests
import torch
from controlnet_aux import ZoeDetector
from PIL import Image, ImageOps

from diffusers import (
AutoencoderKL,
ControlNetModel,
StableDiffusionXLControlNetPipeline,
StableDiffusionXLInpaintPipeline,
)

def scale_and_paste(original_image):
aspect_ratio = original_image.width / original_image.height

if original_image.width > original_image.height:
new_width = 1024
new_height = round(new_width / aspect_ratio)
else:
new_height = 1024
new_width = round(new_height * aspect_ratio)

resized_original = original_image.resize((new_width, new_height), Image.LANCZOS)
white_background = Image.new("RGBA", (1024, 1024), "white")
x = (1024 - new_width) // 2
y = (1024 - new_height) // 2
white_background.paste(resized_original, (x, y), resized_original)

return resized_original, white_background

original_image = Image.open(
requests.get(
"https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/no-background-jordan.png",
stream=True,
).raw
).convert("RGBA")
resized_img, white_bg_image = scale_and_paste(original_image)
```

لمنع إضافة تفاصيل غير مرغوب فيها، استخدم مقدّر ZoeDepth لتوفير إرشادات إضافية أثناء التوليد ولضمان تناسق الأحذية مع الصورة الأصلية.

```py
zoe = ZoeDetector.from_pretrained("lllyasviel/Annotators")
image_zoe = zoe(white_bg_image, detect_resolution=512, image_resolution=1024)
image_zoe
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/zoedepth-jordan.png"/>
</div>

## الرسم خارج الإطار

بمجرد أن تصبح الصورة جاهزة، يمكنك إنشاء محتوى في المساحة البيضاء حول الأحذية باستخدام [controlnet-inpaint-dreamer-sdxl](https://hf.co/destitech/controlnet-inpaint-dreamer-sdxl)، وهو نموذج SDXL ControlNet مدرب على inpainting.

قم بتحميل نموذج inpainting ControlNet، ونموذج ZoeDepth، وVAE، ومررها إلى [`StableDiffusionXLControlNetPipeline`]. ثم يمكنك إنشاء دالة اختيارية `generate_image` (للراحة) للرسم خارج الإطار لصورة أولية.

```py
controlnets = [
ControlNetModel.from_pretrained(
"destitech/controlnet-inpaint-dreamer-sdxl", torch_dtype=torch.float16, variant="fp16"
),
ControlNetModel.from_pretrained(
"diffusers/controlnet-zoe-depth-sdxl-1.0", torch_dtype=torch.float16
),
]
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to("cuda")
pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
"SG161222/RealVisXL_V4.0", torch_dtype=torch.float16, variant="fp16", controlnet=controlnets, vae=vae
).to("cuda")

def generate_image(prompt, negative_prompt, inpaint_image, zoe_image, seed: int = None):
if seed is None:
seed = random.randint(0, 2**32 - 1)

generator = torch.Generator(device="cpu").manual_seed(seed)

image = pipeline(
prompt,
negative_prompt=negative_prompt,
image=[inpaint_image, zoe_image],
guidance_scale=6.5,
num_inference_steps=25,
generator=generator,
controlnet_conditioning_scale=[0.5, 0.8],
control_guidance_end=[0.9, 0.6],
).images[0]

return image

prompt = "nike air jordans on a basketball court"
negative_prompt = ""

temp_image = generate_image(prompt, negative_prompt, white_bg_image, image_zoe, 908097)
```

الصق الصورة الأصلية فوق الصورة الأولية المرسومة خارج الإطار. ستعمل على تحسين الخلفية المرسومة خارج الإطار في خطوة لاحقة.

```py
x = (1024 - resized_img.width) // 2
y = (1024 - resized_img.height) // 2
temp_image.paste(resized_img, (x, y), resized_img)
temp_image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/initial-outpaint.png"/>
</div>

> [!TIP]
> الآن هو الوقت المناسب لتحرير بعض الذاكرة إذا كنت تعاني من نقصها!
>
> ```py
> pipeline=None
> torch.cuda.empty_cache()
> ```

الآن بعد أن أصبحت لديك صورة أولية مرسومة خارج الإطار، قم بتحميل [`StableDiffusionXLInpaintPipeline`] مع نموذج [RealVisXL](https://hf.co/SG161222/RealVisXL_V4.0) لتوليد الصورة النهائية المرسومة خارج الإطار بجودة أفضل.

```py
pipeline = StableDiffusionXLInpaintPipeline.from_pretrained(
"OzzyGT/RealVisXL_V4.0_inpainting",
torch_dtype=torch.float16,
variant="fp16",
vae=vae,
).to("cuda")
```

قم بإعداد قناع للصورة النهائية المرسومة خارج الإطار. لإنشاء انتقال أكثر طبيعية بين الصورة الأصلية والخلفية المرسومة خارج الإطار، قم بتشويش القناع ليساعد على مزجه بشكل أفضل.

```py
mask = Image.new("L", temp_image.size)
mask.paste(resized_img.split()[3], (x, y))
mask = ImageOps.invert(mask)
final_mask = mask.point(lambda p: p > 128 and 255)
mask_blurred = pipeline.mask_processor.blur(final_mask, blur_factor=20)
mask_blurred
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/blurred-mask.png"/>
</div>

قم بإنشاء موجه أفضل ومرره إلى دالة `generate_outpaint` لتوليد الصورة النهائية المرسومة خارج الإطار. مرة أخرى، الصق الصورة الأصلية فوق الخلفية النهائية المرسومة خارج الإطار.

```py
def generate_outpaint(prompt, negative_prompt, image, mask, seed: int = None):
if seed is None:
seed = random.randint(0, 2**32 - 1)

generator = torch.Generator(device="cpu").manual_seed(seed)

image = pipeline(
prompt,
negative_prompt=negative_prompt,
image=image,
mask_image=mask,
guidance_scale=10.0,
strength=0.8,
num_inference_steps=30,
generator=generator,
).images[0]

return image

prompt = "high quality photo of nike air jordans on a basketball court, highly detailed"
negative_prompt = ""

final_image = generate_outpaint(prompt, negative_prompt, temp_image, mask_blurred, 7688778)
x = (1024 - resized_img.width) // 2
y = (1024 - resized_img.height) // 2
final_image.paste(resized_img, (x, y), resized_img)
final_image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/stevhliu/testing-images/resolve/main/final-outpaint.png"/>
</div>