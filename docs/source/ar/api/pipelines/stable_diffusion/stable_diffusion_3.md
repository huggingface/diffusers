# Stable Diffusion 3

تم اقتراح Stable Diffusion 3 (SD3) في ورقة "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" بواسطة Patrick Esser و Sumith Kulal و Andreas Blattmann و Rahim Entezari و Jonas Muller و Harry Saini و Yam Levi و Dominik Lorenz و Axel Sauer و Frederic Boesel و Dustin Podell و Tim Dockhorn و Zion English و Kyle Lacey و Alex Goodwin و Yannik Marek و Robin Rombach.

الملخص من الورقة هو:

*تُنشئ نماذج الانتشار البيانات من الضوضاء عن طريق عكس المسارات الأمامية للبيانات نحو الضوضاء، وقد برزت كتقنية نمذجة مولدة قوية للبيانات عالية الأبعاد والملموسة مثل الصور ومقاطع الفيديو. تدفق المستطيل هو صيغة نموذج تنشئة حديثة تربط البيانات والضوضاء في خط مستقيم. وعلى الرغم من خصائصه النظرية الأفضل وبساطته المفاهيمية، إلا أنه لم يتم ترسيخه بشكل قاطع كممارسة قياسية. في هذا العمل، نقوم بتحسين تقنيات أخذ عينات الضوضاء الحالية لنماذج تدفق المستطيل عن طريق تحيزها نحو المقاييس المهمة من الناحية الإدراكية. ومن خلال دراسة واسعة النطاق، نثبت الأداء المتفوق لهذا النهج مقارنة بصيغ الانتشار المُنشأة لتوليف الصور النصية عالية الدقة. بالإضافة إلى ذلك، نقدم بنية جديدة تعتمد على المحول للنص إلى توليد الصور التي تستخدم أوزانًا منفصلة للنمطين وتمكن التدفق ثنائي الاتجاه للمعلومات بين رموز الصورة والنص، مما يحسن فهم النص وعلم الطباعة وتقييمات التفضيل البشري. نثبت أن هذه البنية تتبع اتجاهات قابلة للتنبؤ في النطاق وأن انخفاض خسارة التحقق يرتبط بتحسين توليف النص إلى الصورة كما تقيسه العديد من المقاييس والتقييمات البشرية.*

## مثال على الاستخدام

_نظرًا لأن النموذج محمي بكلمة مرور، قبل استخدامه مع أجهزة الانتشار، يجب أولاً الانتقال إلى [صفحة Stable Diffusion 3 Medium Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)، وملء النموذج، وقبول كلمة المرور. بمجرد دخولك، تحتاج إلى تسجيل الدخول حتى يعرف نظامك أنك قبلت كلمة المرور._

استخدم الأمر التالي لتسجيل الدخول:

```bash
huggingface-cli login
```

<Tip>

يستخدم خط أنابيب SD3 ثلاثة مشفرات نصية لتوليد صورة. تعد عملية تفريغ النموذج ضرورية لتشغيله على معظم أجهزة الأجهزة الشائعة. يرجى استخدام نوع البيانات `torch.float16` لتوفير الذاكرة الإضافية.

</Tip>

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world.png")
```

## تحسين الذاكرة لـ SD3

يستخدم SD3 ثلاثة مشفرات نصية، أحدها نموذج T5-XXL الكبير جدًا. وهذا يجعل من الصعب تشغيل النموذج على وحدات معالجة الرسوميات (GPUs) التي تحتوي على أقل من 24 جيجابايت من ذاكرة الوصول العشوائي VRAM، حتى عند استخدام دقة `fp16`. يصف القسم التالي بعض تحسينات الذاكرة في أجهزة الانتشار التي تجعل من السهل تشغيل SD3 على أجهزة الأجهزة منخفضة الموارد.

### تشغيل الاستدلال باستخدام تفريغ النموذج

تتيح لك تحسين الذاكرة الأساسي المتاح في أجهزة الانتشار إمكانية تفريغ مكونات النموذج إلى وحدة المعالجة المركزية (CPU) أثناء الاستدلال لتوفير الذاكرة، مع رؤية زيادة طفيفة في وقت الاستدلال. لن يقوم تفريغ النموذج بنقل مكون النموذج إلى وحدة معالجة الرسوميات (GPU) إلا عندما يحتاج إلى التنفيذ، مع الحفاظ على المكونات المتبقية على وحدة المعالجة المركزية (CPU).

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world.png")
```

### إسقاط مشفر النص T5 أثناء الاستدلال

يمكن أن يؤدي إزالة مشفر النص T5-XXL كثيف الذاكرة الذي يحتوي على 4.7 مليار معلمة أثناء الاستدلال إلى تقليل متطلبات الذاكرة لـ SD3 بشكل كبير مع فقدان بسيط في الأداء فقط.

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    text_encoder_3=None,
    tokenizer_3=None,
    torch_dtype=torch.float16
)
pipe.to("cuda")

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world-no-T5.png")
```

### استخدام نسخة كمية من مشفر النص T5

يمكننا الاستفادة من مكتبة `bitsandbytes` لتحميل مشفر النص T5-XXL وتكميمه إلى دقة 8 بت. يسمح لك هذا بالاستمرار في استخدام جميع مشفرات النص الثلاثة مع التأثير بشكل طفيف على الأداء فقط.

قم أولاً بتثبيت مكتبة `bitsandbytes`.

```shell
pip install bitsandbytes
```

ثم قم بتحميل نموذج T5-XXL باستخدام تكوين `BitsAndBytesConfig`.

```python
import torch
from diffusers import StableDiffusion3Pipeline
from transformers import T5EncoderModel, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder_3",
    quantization_config=quantization_config,
)
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    text_encoder_3=text_encoder,
    device_map="balanced",
    torch_dtype=torch.float16
)

image = pipe(
    prompt="a photo of a cat holding a sign that says hello world",
    negative_prompt="",
    num_inference_steps=28,
    height=1024,
    width=1024,
    guidance_scale=7.0,
).images[0]

image.save("sd3_hello_world-8bit-T5.png")
```

يمكنك العثور على البرنامج النصي من البداية إلى النهاية [هنا](https://gist.github.com/sayakpaul/82acb5976509851f2db1a83456e504f1).

## تحسينات الأداء لـ SD3

### استخدام Torch Compile لتسريع الاستدلال

يمكن أن يؤدي استخدام المكونات المجمعة في خط أنابيب SD3 إلى تسريع الاستدلال بمقدار 4 مرات. توضح مقتطفات الشفرة التالية كيفية تجميع مكونات المحول والشبكة العصبية التلافيفية المتنوعة ذات الطبقات (VAE) من خط أنابيب SD3.

```python
import torch
from diffusers import StableDiffusion3Pipeline

torch.set_float32_matmul_precision("high")

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
).to("cuda")
pipe.set_progress_bar_config(disable=True)

pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)

pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

# Warm Up
prompt = "a photo of a cat holding a sign that says hello world"
for _ in range(3):
    _ = pipe(prompt=prompt, generator=torch.manual_seed(1))

# Run Inference
image = pipe(prompt=prompt, generator=torch.manual_seed(1)).images[0]
image.save("sd3_hello_world.png")
```

اطلع على البرنامج النصي الكامل [هنا](https://gist.github.com/sayakpaul/508d89d7aad4f454900813da5d42ca97).

## تحميل نقاط المراقبة الأصلية عبر `from_single_file`

تدعم فئات `SD3Transformer2DModel` و`StableDiffusion3Pipeline` تحميل نقاط المراقبة الأصلية عبر طريقة `from_single_file`. تتيح لك هذه الطريقة تحميل ملفات نقاط المراقبة الأصلية التي تم استخدامها لتدريب النماذج.

## تحميل نقاط المراقبة الأصلية لـ `SD3Transformer2DModel`

```python
from diffusers import SD3Transformer2DModel

model = SD3Transformer2DModel.from_single_file("https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium.safetensors")
```

## تحميل نقطة المراقبة الفردية لـ `StableDiffusion3Pipeline`

### تحميل نقطة المراقبة للملف الفردي بدون T5

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips.safetensors",
    torch_dtype=torch.float16,
    text_encoder_3=None
)
pipe.enable_model_cpu_offload()

image = pipe("a picture of a cat holding a sign that says hello world").images[0]
image.save('sd3-single-file.png')
```

### تحميل نقطة المراقبة للملف الفردي مع T5

```python
import torch
from diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_single_file(
    "https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/sd3_medium_incl_clips_t5xxlfp8.safetensors",
    torch_dtype=torch.float16,
)
pipe.enable_model_cpu_offload()

image = pipe("a picture of a cat holding a sign that says hello world").images[0]
image.save('sd3-single-file-t5-fp8.png')
```

## StableDiffusion3Pipeline

[[autodoc]] StableDiffusion3Pipeline

- all
- __call__