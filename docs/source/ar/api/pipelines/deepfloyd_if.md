# DeepFloyd IF  

## نظرة عامة
DeepFloyd IF هو نموذج مفتوح المصدر ومبتكر لتحويل النص إلى صورة، يتميز بدرجة عالية من الواقعية وفهم اللغة. يتكون النموذج من وحدات نمطية، حيث يتكون من مشفر نص مجمد وثلاث وحدات انتشار بكسل متتالية:

- المرحلة 1: نموذج أساسي يقوم بتوليد صورة 64x64 بكسل بناءً على موجه النص

- المرحلة 2: نموذج فائق الدقة يحول الصورة من 64x64 بكسل إلى 256x256 بكسل

- المرحلة 3: نموذج فائق الدقة يحول الصورة من 256x256 بكسل إلى 1024x1024 بكسل

تستخدم المرحلتان 1 و2 مشفر نص مجمدًا يعتمد على محول T5 لاستخراج تضمينات النص، والتي يتم تغذيتها بعد ذلك في بنية UNet المحسنة باهتمام متقاطع وتجميع الاهتمام.

المرحلة 3 هي [نموذج Stability AI's x4 Upscaling](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler).

والنتيجة هي نموذج عالي الكفاءة يتفوق على النماذج الحالية الرائدة في المجال، حيث يحقق درجة FID تساوي 6.66 على مجموعة بيانات COCO في حالة عدم وجود بيانات تدريب.

يؤكد عملنا على الإمكانات الكبيرة لبنيات UNet الأكبر في المرحلة الأولى من نماذج الانتشار المتتالية، ويرسم مستقبلاً واعدًا لتركيب الصور من النص.

## الاستخدام

قبل أن تتمكن من استخدام IF، يجب عليك قبول شروط الاستخدام. للقيام بذلك:

1. تأكد من امتلاك حساب على [Hugging Face](https://huggingface.co/join) وأنك مسجل الدخول.

2. قبول الترخيص على بطاقة نموذج [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0). سيؤدي قبول الترخيص على بطاقة نموذج المرحلة الأولى إلى الموافقة التلقائية على النماذج الأخرى لـ IF.

3. تأكد من تسجيل الدخول محليًا. قم بتثبيت `huggingface_hub`:

```sh
pip install huggingface_hub --upgrade
```  

قم بتشغيل دالة تسجيل الدخول في غلاف Python:

```py
from huggingface_hub import login

login()
```  

وأدخل رمز الوصول إلى [Hugging Face Hub](https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens).

بعد ذلك، نقوم بتثبيت `diffusers` والاعتماديات:

```sh
pip install -q diffusers accelerate transformers
```  

تقدم الأقسام التالية أمثلة أكثر تفصيلاً حول كيفية استخدام IF. على وجه التحديد:

- [توليد نص إلى صورة](#توليد-نص-إلى-صورة)

- [توليد صورة إلى صورة موجهة بالنص](#توليد-صورة-إلى-صورة-موجهة-بالنص)

- [التلوين](#التلوين)

- [إعادة استخدام أوزان النموذج](#إعادة-استخدام-أوزان-النموذج)

- [تحسين السرعة](#تحسين-السرعة)

- [تحسين الذاكرة](#تحسين-الذاكرة)

**نقاط التحقق المتاحة**

- *المرحلة 1*

- [DeepFloyd/IF-I-XL-v1.0](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0)

- [DeepFloyd/IF-I-L-v1.0](https://huggingface.co/DeepFloyd/IF-I-L-v1.0)

- [DeepFloyd/IF-I-M-v1.0](https://huggingface.co/DeepFloyd/IF-I-M-v1.0)  

- *المرحلة 2*

- [DeepFloyd/IF-II-L-v1.0](https://huggingface.co/DeepFloyd/IF-II-L-v1.0)

- [DeepFloyd/IF-II-M-v1.0](https://huggingface.co/DeepFloyd/IF-II-M-v1.0)  

- *المرحلة 3*

- [stabilityai/stable-diffusion-x4-upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)  

**Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github.com/huggingface/notebooks/blob/main/diffusers/deepfloyd_if_free_tier_google_colab.ipynb)

### توليد نص إلى صورة

بشكل افتراضي، يستخدم برنامج Diffusers [إلغاء تحميل وحدة المعالجة المركزية للنموذج](../../optimization/memory#model-offloading) لتشغيل خط أنابيب IF بالكامل باستخدام 14 جيجابايت فقط من VRAM.

```python
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil, make_image_grid
import torch

# المرحلة 1
stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# المرحلة 2
stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# المرحلة 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = 'صورة لكンガرو يرتدي هودي برتقالي ونظارات شمسية زرقاء وهو يقف أمام برج إيفل ويحمل لافتة مكتوب عليها "تعلم عميق جدا"'
generator = torch.manual_seed(1)

# تضمينات النص
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# المرحلة 1
stage_1_output = stage_1(
    prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
).images
#pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# المرحلة 2
stage_2_output = stage_2(
    image=stage_1_output,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
#pt_toََُto_pil(stage_2_output)[0].save("./if_stage_II.png")

# المرحلة 3
stage_3_output = stage_3(prompt=prompt, image=stage_2_output, noise_level=100, generator=generator).images
#stage_3_output[0].save("./if_stage_III.png")
make_image_grid([pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], rows=1, rows=3)
```

### توليد صورة إلى صورة موجهة بالنص

يمكن استخدام أوزان نموذج IF نفسه للترجمة من صورة إلى صورة موجهة بالنص أو لتغيير الصورة.

في هذه الحالة، تأكد فقط من تحميل الأوزان باستخدام خطوط الأنابيب [`IFImg2ImgPipeline`] و [`IFImg2ImgSuperResolutionPipeline`].

**ملاحظة**: يمكنك أيضًا نقل أوزان خطوط أنابيب النص إلى الصورة مباشرةً

بدون تحميلها مرتين عن طريق استخدام وسيط [`~DiffusionPipeline.components`] كما هو موضح [هنا](#إعادة-استخدام-أوزان-النموذج).

```python
from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil, load_image, make_image_grid
import torch

# قم بتنزيل الصورة
url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
original_image = load_image(url)
original_image = original_image.resize((768, 512))

# المرحلة 1
stage_1 = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# المرحلة 2
stage_2 = IFImg2ImgSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# المرحلة 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = "منظر طبيعي خيالي على طريقة ماين كرافت"
generator = torch.manual_seed(1)

# تضمينات النص
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# المرحلة 1
stage_1_output = stage_1(
    image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
#pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# المرحلة 2
stage_2_output = stage_2(
    image=stage_1_output,
    original_image=original_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
#pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")

# المرحلة 3
stage_3_output = stage_3(prompt=prompt, image=stage_2_output, generator=generator, noise_level=100).images
#stage_3_output[0].save("./if_stage_III.png")
make_image_grid([original_image, pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], rows=1, rows=4)
```

### التلوين

يمكن استخدام أوزان نموذج IF نفسه لتلوين الصور الموجهة بالنص.

في هذه الحالة، تأكد فقط من تحميل الأوزان باستخدام خطوط الأنابيب [`IFInpaintingPipeline`] و [`IFInpaintingSuperResolutionPipeline`].

**ملاحظة**: يمكنك أيضًا نقل أوزان خطوط أنابيب النص إلى الصورة مباشرةً

بدون تحميلها مرتين عن طريق استخدام دالة [`~DiffusionPipeline.components()`] كما هو موضح [هنا](#إعادة-استخدام-أوزان-النموذج).

```python
from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline, DiffusionPipeline
from diffusers.utils import pt_to_pil, load_image, make_image_grid
import torch

# قم بتنزيل الصورة
url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/person.png"
original_image = load_image(url)

# قم بتنزيل القناع
url = "https://huggingface.co/datasets/diffusers/docs-images/resolve/main/if/glasses_mask.png"
mask_image = load_image(url)

# المرحلة 1
stage_1 = IFInpaintingPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
stage_1.enable_model_cpu_offload()

# المرحلة 2
stage_2 = IFInpaintingSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
stage_2.enable_model_cpu_offload()

# المرحلة 3
safety_modules = {
    "feature_extractor": stage_1.feature_extractor,
    "safety_checker": stage_1.safety_checker,
    "watermarker": stage_1.watermarker,
}
stage_3 = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", **safety_modules, torch_dtype=torch.float16
)
stage_3.enable_model_cpu_offload()

prompt = "نظارات شمسية زرقاء"
generator = torch.manual_seed(1)

# تضمينات النص
prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)

# المرحلة 1
stage_1_output = stage_1(
    image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
#pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# المرحلة 2
stage_2_output = stage_2(
    image=stage_1_output,
    original_image=original_image,
    mask_image=mask_image,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    generator=generator,
    output_type="pt",
).images
#pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")

# المرحلة 3
stage_3_output = stage_3(prompt=prompt, image=stage_2_output, generator=generator, noise_level=100).images
#stage_3_output[0].save("./if_stage_III.png")
make_image_grid([original_image, mask_image, pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0], stage_3_output[0]], rows=1, rows=5)
```

### التحويل بين خطوط الأنابيب المختلفة

بالإضافة إلى التحميل باستخدام `from_pretrained`، يمكن أيضًا تحميل خطوط الأنابيب مباشرة من بعضها البعض.

```python
from diffusers import IFPipeline, IFSuperResolutionPipeline

pipe_1 = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0")
pipe_2 = IFSuperResolutionPipeline.from_pretrained("DeepFloyd/IF-II-L-v1.0")


from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline

pipe_1 = IFImg2ImgPipeline(**pipe_1.components)
pipe_2 = IFImg2ImgSuperResolutionPipeline(**pipe_2.components)


from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline

pipe_1 = IFInpaintingPipeline(**pipe_1.components)
pipe_2 = IFInpaintingSuperResolutionPipeline(**pipe_2.components)

```
### التحسين من أجل السرعة

أبسط طريقة لتحسين تشغيل IF بشكل أسرع هي نقل جميع مكونات النموذج إلى وحدة معالجة الرسومات (GPU).

يمكنك أيضًا تشغيل عملية الانتشار لعدد أقل من الخطوات الزمنية. يمكن القيام بذلك إما باستخدام وسيط num_inference_steps:

```py
pipe("<prompt>", num_inference_steps=30)
```

أو باستخدام وسيط timesteps:

```py
from diffusers.pipelines.deepfloyd_if import fast27_timesteps

pipe("<prompt>", timesteps=fast27_timesteps)
```

عند إجراء التباين الصوري أو التلوين، يمكنك أيضًا تقليل عدد الخطوات الزمنية باستخدام وسيط القوة. وسيط القوة هو مقدار الضوضاء التي سيتم إضافتها إلى الصورة المدخلة، والتي تحدد أيضًا عدد الخطوات التي سيتم تشغيلها في عملية إزالة التشويش. يؤدي استخدام رقم أصغر إلى تقليل تباين الصورة ولكنه يعمل بشكل أسرع.

```py
pipe = IFImg2ImgPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(image=image, prompt="<prompt>", strength=0.3).images
```

يمكنك أيضًا استخدام [torch.compile](../../optimization/torch2.0). لاحظ أننا لم نقم باختبار torch.compile بشكل شامل مع IF وقد لا يعطي النتائج المتوقعة.

```py
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

pipe.text_encoder = torch.compile(pipe.text_encoder, mode="reduce-overhead", fullgraph=True)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
```

### التحسين من أجل الذاكرة

عند التحسين من أجل ذاكرة GPU، يمكننا استخدام واجهات برمجة التطبيقات القياسية لنقل الحوسبة من وحدة المعالجة المركزية إلى وحدة معالجة الرسومات (GPU) في diffusers.

إما النقل من وحدة المعالجة المركزية إلى وحدة معالجة الرسومات (GPU) القائم على النموذج،

```py
pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()
```

أو النقل من وحدة المعالجة المركزية إلى وحدة معالجة الرسومات (GPU) القائم على الطبقة الأكثر عدوانية.

```py
pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
pipe.enable_sequential_cpu_offload()
```

بالإضافة إلى ذلك، يمكن تحميل T5 بدقة 8 بت

```py
from transformers import T5EncoderModel

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0", subfolder="text_encoder", device_map="auto", load_in_8bit=True, variant="8bit"
)

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0",
    text_encoder=text_encoder,  # تمرير مشفر النص الذي تم تثبيته مسبقًا بدقة 8 بت
    unet=None,
    device_map="auto",
)

prompt_embeds, negative_embeds = pipe.encode_prompt("<prompt>")
```

بالنسبة لأجهزة الكمبيوتر المقيدة بذاكرة الوصول العشوائي (RAM) لوحدة المعالجة المركزية (CPU) مثل Google Colab من الطبقة المجانية، حيث لا يمكننا تحميل جميع مكونات النموذج إلى وحدة المعالجة المركزية (CPU) في نفس الوقت، يمكننا يدويًا تحميل خط الأنابيب فقط مع مشفر النص أو UNet عندما تكون مكونات النموذج المطلوبة.

```py
from diffusers import IFPipeline, IFSuperResolutionPipeline
import torch
import gc
from transformers import T5EncoderModel
from diffusers.utils import pt_to_pil، make_image_grid

text_encoder = T5EncoderModel.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0"، subfolder="text_encoder"، device_map="auto"، load_in_8bit=True، variant="8bit"
)

# نص إلى صورة
pipe = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0"،
    text_encoder=text_encoder،  # تمرير مشفر النص الذي تم تثبيته مسبقًا بدقة 8 بت
    unet=None،
    device_map="auto"،
)

prompt = 'صورة لحيوان الكنغر يرتدي هودي برتقالي ونظارات شمسية زرقاء يقف أمام برج إيفل ويحمل لافتة تقول "تعلم عميق للغاية"'
prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

# إزالة خط الأنابيب حتى نتمكن من إعادة تحميل خط الأنابيب باستخدام UNet
del text_encoder
del pipe
gc.collect()
torch.cuda.empty_cache()

pipe = IFPipeline.from_pretrained(
    "DeepFloyd/IF-I-XL-v1.0"، text_encoder=None، variant="fp16"، torch_dtype=torch.float16، device_map="auto"
)

generator = torch.Generator().manual_seed(0)
stage_1_output = pipe(
    prompt_embeds=prompt_embeds،
    negative_prompt_embeds=negative_embeds،
    output_type="pt"،
    generator=generator،
).images

# حفظ الإخراج في ملف
# pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")

# إزالة خط الأنابيب حتى نتمكن من تحميل خط أنابيب فائق الدقة
del pipe
gc.collect()
torch.cuda.empty_cache()

# فائقة الدقة الأولى

pipe = IFSuperResolutionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0"، text_encoder=None، variant="fp16"، torch_dtype=torch.float16، device_map="auto"
)

generator = torch.Generator().manual_seed(0)
stage_2_output = pipe(
    image=stage_1_outpSr،
).images

# حفظ الإخراج في ملف
# pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")
# إنشاء شبكة صور
make_image_grid([pt_to_pil(stage_1_output)[0], pt_to_pil(stage_2_output)[0]]، rows=1، rows=2)
```

## خطوط الأنابيب المتاحة:

| خط الأنابيب | المهام | Colab |
|---|---|:---:|
| [pipeline_if.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py) | *توليد الصور من النص* | - |
| [pipeline_if_superresolution.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_superresolution.py) | *توليد الصور من النص* | - |
| [pipeline_if_img2img.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img.py) | *توليد الصور من الصور* | - |
| [pipeline_if_img2img_superresolution.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_img2img_superresolution.py) | *توليد الصور من الصور* | - |
| [pipeline_if_inpainting.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting.py) | *توليد الصور من الصور* | - |
| [pipeline_if_inpainting_superresolution.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if_inpainting_superresolution.py) | *توليد الصور من الصور* | - |

## IFPipeline

[[autodoc]] IFPipeline

- all
- __call__

## IFSuperResolutionPipeline

[[autodoc]] IFSuperResolutionPipeline

- all
- __call__

## IFImg2ImgPipeline

[[autodoc]] IFImg2ImgPipeline

- all
- __call__

## IFImg2ImgSuperResolutionPipeline

[[autodoc]] IFImg2ImgSuperResolutionPipeline

- all
- __call__

## IFInpaintingPipeline

[[autodoc]] IFInpaintingPipeline

- all
- __call__

## IFInpaintingSuperResolutionPipeline

[[autodoc]] IFInpaintingSuperResolutionPipeline

- all
- __call__