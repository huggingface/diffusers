# PyTorch 2.0
يدعم 🤗 Diffusers أحدث التحسينات من [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/) والتي تشمل:

1. تنفيذ فعال للذاكرة، واهتمام المنتج النقطي، دون الحاجة إلى أي تبعيات إضافية مثل xFormers.
2. [`torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)، وهو مترجم JIT (Just-In-Time) لتوفير تعزيز الأداء الإضافي عند تجميع النماذج الفردية.

يتطلب كل من هذه التحسينات PyTorch 2.0 أو إصدار أحدث و 🤗 Diffusers > 0.13.0.

```bash
pip install --upgrade torch diffusers
```

## اهتمام المنتج النقطي المُصعّد

[`torch.nn.functional.scaled_dot_product_attention`](https://pytorch.org/docs/master/generated/torch.nn.functional.scaled_dot_product_attention) (SDPA) هو اهتمام محسّن وفعال من حيث الذاكرة (مشابه لـ xFormers) والذي يمكّن تلقائيًا العديد من التحسينات الأخرى اعتمادًا على إدخالات النموذج ونوع GPU. يتم تمكين SDPA بشكل افتراضي إذا كنت تستخدم PyTorch 2.0 والإصدار الأخير من 🤗 Diffusers، لذلك لا تحتاج إلى إضافة أي شيء إلى رمزك.

ومع ذلك، إذا كنت تريد تمكينه بشكل صريح، فيمكنك تعيين [`DiffusionPipeline`] لاستخدام [`~models.attention_processor.AttnProcessor2_0`]:

```diff
import torch
from diffusers import DiffusionPipeline
+ from diffusers.models.attention_processor import AttnProcessor2_0

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
+ pipe.unet.set_attn_processor(AttnProcessor2_0())

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
```

يجب أن يكون SDPA سريعًا وفعالًا من حيث الذاكرة مثل `xFormers`؛ راجع [المعيار المرجعي](#benchmark) لمزيد من التفاصيل.

في بعض الحالات - مثل جعل الأنبوب أكثر حتمية أو تحويله إلى تنسيقات أخرى - قد يكون من المفيد استخدام معالج الاهتمام الفانيليا، [`~models.attention_processor.AttnProcessor`]. للرجوع إلى [`~models.attention_processor.AttnProcessor`]، اتصل بوظيفة [`~UNet2DConditionModel.set_default_attn_processor`] على الأنبوب:

```diff
import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
+ pipe.unet.set_default_attn_processor()

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
```

## torch.compile

يمكن لوظيفة `torch.compile` غالبًا أن توفر تسريعًا إضافيًا لرمز PyTorch الخاص بك. في 🤗 Diffusers، من الأفضل عادةً لف UNet بـ `torch.compile` لأنه يقوم بمعظم العمل الشاق في الأنبوب.

```python
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
images = pipe(prompt, num_inference_steps=steps, num_images_per_prompt=batch_size).images[0]
```

حسب نوع GPU، يمكن لـ `torch.compile` توفير تسريع *إضافي* من **5-300x** بالإضافة إلى SDPA! إذا كنت تستخدم معماريات GPU الأحدث مثل Ampere (A100، 3090)، Ada (4090)، و Hopper (H100)، فإن `torch.compile` قادر على استخراج المزيد من الأداء من هذه GPUs.

يتطلب التجميع بعض الوقت للانتهاء، لذلك فهو الأنسب للحالات التي تقوم فيها بإعداد خط أنابيبك مرة واحدة ثم إجراء نفس نوع عمليات الاستدلال عدة مرات. على سبيل المثال، يؤدي استدعاء خط الأنابيب المجمع على حجم صورة مختلف إلى تشغيل التجميع مرة أخرى والذي قد يكون مكلفًا.

لمزيد من المعلومات والخيارات المختلفة حول `torch.compile`، راجع البرنامج التعليمي [`torch_compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

> [!TIP]
> تعرف على المزيد حول الطرق الأخرى التي يمكن أن تساعد بها PyTorch 2.0 في تحسين نموذجك في البرنامج التعليمي [تسريع الاستدلال لنموذج النص إلى الصورة النشر](../tutorials/fast_diffusion).

## المعيار

أجرينا معيارًا شاملاً مع تنفيذ PyTorch 2.0 للاهتمام الفعال و `torch.compile` عبر GPUs وحجم الدفعات المختلفة لخمسة من خطوط الأنابيب الأكثر استخدامًا لدينا. تم اختبار الكود على 🤗 Diffusers v0.17.0.dev0 لتحسين استخدام `torch.compile` (راجع [هنا] (https://github.com/huggingface/diffusers/pull/3313) لمزيد من التفاصيل).

قم بتوسيع القائمة المنسدلة أدناه للعثور على الكود المستخدم لمعيار كل خط أنابيب:

<details>

### Stable Diffusion text-to-image

```python
from diffusers import DiffusionPipeline
import torch

path = "runwayml/stable-diffusion-v1-5"

run_compile = True  # Set True / False

pipe = DiffusionPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

if run_compile:
print("Run torch compile")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
images = pipe(prompt=prompt).images
```

### Stable Diffusion image-to-image

```python
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import load_image
import torch

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

init_image = load_image(url)
init_image = init_image.resize((512, 512))

path = "runwayml/stable-diffusion-v1-5"

run_compile = True  # Set True / False

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

if run_compile:
print("Run torch compile")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
image = pipe(prompt=prompt, image=init_image).images[0]
```

### Stable Diffusion inpainting

```python
from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils import load_image
import torch

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((512, 512))
mask_image = load_image(mask_url).resize((512, 512))

path = "runwayml/stable-diffusion-inpainting"

run_compile = True  # Set True / False

pipe = StableDiffusionInpaintPipeline.from_pretrained(path, torch_dtype=torch.float16, use_safetensors=True)
pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)

if run_compile:
print("Run torch compile")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]
```

### ControlNet

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

init_image = load_image(url)
init_image = init_image.resize((512, 512))

path = "runwayml/stable-diffusion-v1-5"

run_compile = True  # Set True / False
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, use_safetensors=True)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
path, controlnet=controlnet, torch_dtype=torch.float16, use_safetensors=True
)

pipe = pipe.to("cuda")
pipe.unet.to(memory_format=torch.channels_last)
pipe.controlnet.to(memory_format=torch.channels_last)

if run_compile:
print("Run torch compile")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.controlnet = torch.compile(pipe.controlnet, mode="reduce-overhead", fullgraph=True)

prompt = "ghibli style, a fantasy landscape with castles"

for _ in range(3):
image = pipe(prompt=prompt, image=init_image).images[0]
```
بالتأكيد، سأتبع تعليماتك بدقة لترجمة النص الموجود في الفقرات والعناوين فقط، مع تجاهل الأكواد البرمجية والروابط ورموز HTML وCSS.

---

يسلط الرسم البياني أدناه الضوء على التحسينات النسبية لسرعة [`StableDiffusionPipeline`] عبر خمس عائلات من وحدات معالجة الرسوميات (GPU) مع PyTorch 2.0 و`torch.compile` مفعلة. تم قياس المعايير للرسوم البيانية التالية من حيث *عدد التكرارات في الثانية*.

توفر الجداول التالية نتائجنا من حيث *عدد التكرارات في الثانية*.

### A100 (حجم الدفعة: 1)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 21.66 | 23.13 | 44.03 | 49.74 |
| SD - img2img | 21.81 | 22.40 | 43.92 | 46.32 |
| SD - inpaint | 22.24 | 23.23 | 43.76 | 49.25 |
| SD - controlnet | 15.02 | 15.82 | 32.13 | 36.08 |
| IF | 20.21 / <br>13.84 / <br>24.00 | 20.12 / <br>13.70 / <br>24.03 | ❌ | 97.34 / <br>27.23 / <br>111.66 |
| SDXL - txt2img | 8.64 | 9.9 | - | - |

### A100 (حجم الدفعة: 4)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 11.6 | 13.12 | 14.62 | 17.27 |
| SD - img2img | 11.47 | 13.06 | 14.66 | 17.25 |
| SD - inpaint | 11.67 | 13.31 | 14.88 | 17.48 |
| SD - controlnet | 8.28 | 9.38 | 10.51 | 12.41 |
| IF | 25.02 | 18.04 | ❌ | 48.47 |
| SDXL - txt2img | 2.44 | 2.74 | - | - |

### A100 (حجم الدفعة: 16)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 3.04 | 3.6 | 3.83 | 4.68 |
| SD - img2img | 2.98 | 3.58 | 3.83 | 4.67 |
| SD - inpaint | 3.04 | 3.66 | 3.9 | 4.76 |
| SD - controlnet | 2.15 | 2.58 | 2.74 | 3.35 |
| IF | 8.78 | 9.82 | ❌ | 16.77 |
| SDXL - txt2img | 0.64 | 0.72 | - | - |

### V100 (حجم الدفعة: 1)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 18.99 | 19.14 | 20.95 | 22.17 |
| SD - img2img | 18.56 | 19.18 | 20.95 | 22.11 |
| SD - inpaint | 19.14 | 19.06 | 21.08 | 22.20 |
| SD - controlnet | 13.48 | 13.93 | 15.18 | 15.88 |
| IF | 20.01 / <br>9.08 / <br>23.34 | 19.79 / <br>8.98 / <br>24.10 | ❌ | 55.75 / <br>11.57 / <br>57.67 |

### V100 (حجم الدفعة: 4)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 5.96 | 5.89 | 6.83 | 6.86 |
| SD - img2img | 5.90 | 5.91 | 6.81 | 6.82 |
| SD - inpaint | 5.99 | 6.03 | 6.93 | 6.95 |
| SD - controlnet | 4.26 | 4.29 | 4.92 | 4.93 |
| IF | 15.41 | 14.76 | ❌ | 22.95 |

### V100 (حجم الدفعة: 16)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 1.66 | 1.66 | 1.92 | 1.90 |
| SD - img2img | 1.65 | 1.65 | 1.91 | 1.89 |
| SD - inpaint | 1.69 | 1.69 | 1.95 | 1.93 |
| SD - controlnet | 1.19 | 1.19 | OOM بعد الإحماء | 1.36 |
| IF | 5.43 | 5.29 | ❌ | 7.06 |

### T4 (حجم الدفعة: 1)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 6.9 | 6.95 | 7.3 | 7.56 |
| SD - img2img | 6.84 | 6.99 | 7.04 | 7.55 |
| SD - inpaint | 6.91 | 6.7 | 7.01 | 7.37 |
| SD - controlnet | 4.89 | 4.86 | 5.35 | 5.48 |
| IF | 17.42 / <br>2.47 / <br>18.52 | 16.96 / <br>2.45 / <br>18.69 | ❌ | 24.63 / <br>2.47 / <br>23.39 |
| SDXL - txt2img | 1.15 | 1.16 | - | - |

### T4 (حجم الدفعة: 4)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 1.79 | 1.79 | 2.03 | 1.99 |
| SD - img2img | 1.77 | 1.77 | 2.05 | 2.04 |
| SD - inpaint | 1.81 | 1.82 | 2.09 | 2.09 |
| SD - controlnet | 1.34 | 1.27 | 1.47 | 1.46 |
| IF | 5.79 | 5.61 | ❌ | 7.39 |
| SDXL - txt2img | 0.288 | 0.289 | - | - |

### T4 (حجم الدفعة: 16)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 2.34s | 2.30s | OOM بعد التكرار الثاني | 1.99s |
| SD - img2img | 2.35s | 2.31s | OOM بعد الإحماء | 2.00s |
| SD - inpaint | 2.30s | 2.26s | OOM بعد التكرار الثاني | 1.95s |
| SD - controlnet | OOM بعد التكرار الثاني | OOM بعد التكرار الثاني | OOM بعد الإحماء | OOM بعد الإحماء |
| IF * | 1.44 | 1.44 | ❌ | 1.94 |
| SDXL - txt2img | OOM | OOM | - | - |

### RTX 3090 (حجم الدفعة: 1)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 22.56 | 22.84 | 23.84 | 25.69 |
| SD - img2img | 22.25 | 22.61 | 24.1 | 25.83 |
| SD - inpaint | 22.22 | 22.54 | 24.26 | 26.02 |
| SD - controlnet | 16.03 | 16.33 | 17.38 | 18.56 |
| IF | 27.08 / <br>9.07 / <br>31.23 | 26.75 / <br>8.92 / <br>31.47 | ❌ | 68.08 / <br>11.16 / <br>65.29 |

### RTX 3090 (حجم الدفعة: 4)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 6.46 | 6.35 | 7.29 | 7.3 |
| SD - img2img | 6.33 | 6.27 | 7.31 | 7.26 |
| SD - inpaint | 6.47 | 6.4 | 7.44 | 7.39 |
| SD - controlnet | 4.59 | 4.54 | 5.27 | 5.26 |
| IF | 16.81 | 16.62 | ❌ | 21.57 |

### RTX 3090 (حجم الدفعة: 16)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 1.7 | 1.69 | 1.93 | 1.91 |
| SD - img2img | 1.68 | 1.67 | 1.93 | 1.9 |
| SD - inpaint | 1.72 | 1.71 | 1.97 | 1.94 |
| SD - controlnet | 1.23 | 1.22 | 1.4 | 1.38 |
| IF | 5.01 | 5.00 | ❌ | 6.33 |

### RTX 4090 (حجم الدفعة: 1)

| **الأنابيب** | **شعلة 2.0 - <br>لا تجميع** | **شعلة ليلية - <br>لا تجميع** | **شعلة 2.0 - <br>تجميع** | **شعلة ليلية - <br>تجميع** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 40.5 | 41.89 | 44.65 | 49.81 |
| SD - img2img | 40.39 | 41.95 | 44.46 | 49.8 |
| SD - inpaint | 40.51 | 41.88 | 44.58 | 49.72 |
| SD - controlnet | 29.27 | 30.29 | 32.26 | 36.03 |
| IF | 69.71 / <br>18.78 / <br>85.49 | 69.13 / <br>18.80 / <br>85.56 | ❌ | 124.60 / <br>26.37 / <br>138.79 |
| SDXL - txt2img | 6.8 | 8.18 | - | - |

#### RTX 4090 (batch size: 4)

| **Pipeline** | **torch 2.0 - <br>no compile** | **torch nightly - <br>no compile** | **torch 2.0 - <br>compile** | **torch nightly - <br>compile** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 12.62 | 12.84 | 15.32 | 15.59 |
| SD - img2img | 12.61 | 12,.79 | 15.35 | 15.66 |
| SD - inpaint | 12.65 | 12.81 | 15.3 | 15.58 |
| SD - controlnet | 9.1 | 9.25 | 11.03 | 11.22 |
| IF | 31.88 | 31.14 | ❌ | 43.92 |
| SDXL - txt2img | 2.19 | 2.35 | - | - |

### RTX 4090 (batch size: 16)

| **Pipeline** | **torch 2.0 - <br>no compile** | **torch nightly - <br>no compile** | **torch 2.0 - <br>compile** | **torch nightly - <br>compile** |
|:---:|:---:|:---:|:---:|:---:|
| SD - txt2img | 3.17 | 3.2 | 3.84 | 3.85 |
| SD - img2img | 3.16 | 3.2 | 3.84 | 3.85 |
| SD - inpaint | 3.17 | 3.2 | 3.85 | 3.85 |
| SD - controlnet | 2.23 | 2.3 | 2.7 | 2.75 |
| IF | 9.26 | 9.2 | ❌ | 13.31 |
| SDXL - txt2img | 0.52 | 0.53 | - | - |

## Notes

* Follow this [PR](https://github.com/huggingface/diffusers/pull/3313) for more details on the environment used for conducting the benchmarks.
* For the DeepFloyd IF pipeline where batch sizes > 1, we only used a batch size of > 1 in the first IF pipeline for text-to-image generation and NOT for upscaling. That means the two upscaling pipelines received a batch size of 1.

*Thanks to [Horace He](https://github.com/Chillee) from the PyTorch team for their support in improving our support of `torch.compile()` in Diffusers.*
