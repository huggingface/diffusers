# PixArt-Σ

[PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://huggingface.co/papers/2403.04692) هو جونسنج شين، جينشنج يو، تشونججيان جي، ليوي يايو، إينزي زي، يو وو، زونجداو وانج، جيمس كووك، بينج لورو، هوشوان لو، وزينجوو لي.

الملخص من الورقة هو:

*في هذه الورقة، نقدم PixArt-Σ، وهو نموذج تحويل انتشار قادر على توليد الصور مباشرة بدقة 4K. يمثل PixArt-Σ تقدمًا كبيرًا عن سابقه، PixArt-α، حيث يقدم صورًا ذات دقة أعلى بشكل ملحوظ وتحسين الاتساق مع موجهات النص. إحدى الميزات الرئيسية لـ PixArt-Σ هي كفاءة التدريب. بالاعتماد على التدريب التمهيدي لـ PixArt-α، فإنه يتطور من خط الأساس "الأضعف" إلى نموذج "أقوى" من خلال دمج بيانات أعلى جودة، وهي عملية نطلق عليها اسم "التدريب الضعيف إلى القوي". وتتمثل أوجه التقدم في PixArt-Σ في: (1) بيانات التدريب عالية الجودة: يدمج PixArt-Σ بيانات صور ذات جودة فائقة، مقترنة بتعليقات توضيحية للصور أكثر دقة وتفصيلاً. (2) ضغط الرموز المميزة بكفاءة: نقترح وحدة اهتمام جديدة داخل إطار DiT تضغط كل من المفاتيح والقيم، مما يحسن الكفاءة بشكل كبير وييسر توليد الصور فائقة الدقة. وبفضل هذه التحسينات، يحقق PixArt-Σ جودة صورة فائقة وقدرات الالتزام بتوجيهات المستخدم بحجم نموذج أصغر بكثير (0.6 مليار معلمة) مقارنة بنماذج الانتشار من النص إلى الصورة الحالية، مثل SDXL (2.6 مليار معلمة) وSD Cascade (5.1 مليار معلمة). علاوة على ذلك، تدعم قدرة PixArt-Σ على إنشاء صور 4K إنشاء الملصقات والخلفيات عالية الدقة، مما يعزز بكفاءة إنتاج المحتوى المرئي عالي الجودة في صناعات مثل الأفلام والألعاب.*

يمكنك العثور على الكود الأصلي في [PixArt-alpha/PixArt-sigma](https://github.com/PixArt-alpha/PixArt-sigma) وجميع نقاط التفتيش المتاحة في [PixArt-alpha](https://huggingface.co/PixArt-alpha).

بعض الملاحظات حول هذا الخط الأنابيب:

* يستخدم العمود الفقري للمحول (بدلاً من UNet) لإزالة التشويش. وبهذه الصفة، فإن له بنية مماثلة لـ [DiT](https://hf.co/docs/transformers/model_doc/dit).
* تم تدريبه باستخدام شروط النص المحسوبة من T5. وهذا الجانب يجعل الأنبوب أفضل في اتباع موجهات النص المعقدة بتفاصيل دقيقة.
* إنه جيد في إنتاج صور عالية الدقة بنسب عرض مختلفة. للحصول على أفضل النتائج، يوصي المؤلفون ببعض أقواس الأحجام التي يمكن العثور عليها [هنا](https://github.com/PixArt-alpha/PixArt-sigma/blob/master/diffusion/data/datasets/utils.py).
* ينافس جودة أنظمة توليد الصور من النص إلى الصورة المتقدمة (اعتبارًا من وقت الكتابة) مثل PixArt-α وStable Diffusion XL وPlayground V2.0 وDALL-E 3، مع كونه أكثر كفاءة منها.
* يظهر القدرة على توليد صور فائقة الدقة للغاية، مثل 2048 بكسل أو حتى 4K.
* يُظهر أن النماذج من النص إلى الصورة يمكن أن تتطور من نموذج ضعيف إلى نموذج أقوى من خلال العديد من التحسينات (VAEs، ومجموعات البيانات، وما إلى ذلك).

## الاستنتاج مع VRAM GPU أقل من 8 جيجا

قم بتشغيل ["PixArtSigmaPipeline"] مع VRAM GPU أقل من 8 جيجا عن طريق تحميل مشفر النص بدقة 8 بت. دعونا نمر عبر مثال كامل الميزات.

أولاً، قم بتثبيت مكتبة [bitsandbytes](https://github.com/TimDettmers/bitsandbytes):

```bash
pip install -U bitsandbytes
```

ثم قم بتحميل مشفر النص بدقة 8 بت:

```python
from transformers import T5EncoderModel
from diffusers import PixArtSigmaPipeline
import torch

text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    subfolder="text_encoder",
    load_in_8bit=True,
    device_map="auto",
)
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    text_encoder=text_encoder,
    transformer=None,
    device_map="balanced"
)
```

الآن، استخدم `pipe` لتشفير موجه:

```python
with torch.no_grad():
    prompt = "cute cat"
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
```

نظرًا لحساب تضمين النص، فقم بإزالة `text_encoder` و`pipe` من الذاكرة، وحرر بعض VRAM GPU:

```python
import gc

def flush():
gc.collect()
torch.cuda.empty_cache()

del text_encoder
del pipe
flush()
```

ثم احسب المخفيات مع تضمين الموجه كمدخلات:

```python
pipe = PixArtSigmaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")

latents = pipe(
    negative_prompt=None,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_embeds,
    prompt_attention_mask=prompt_attention_mask,
    negative_prompt_attention_mask=negative_prompt_attention_mask,
    num_images_per_prompt=1,
    output_type="latent",
).images

del pipe.transformer
flush()
```

<Tip>

لاحظ أنه أثناء تهيئة `pipe`، تقوم بتعيين `text_encoder` إلى `None` حتى لا يتم تحميله.

</Tip>

بمجرد حساب المخفيات، مررها إلى VAE لفك تشفيرها إلى صورة حقيقية:

```python
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    image.save("cat.png")
```

من خلال حذف المكونات التي لا تستخدمها وتفريغ VRAM GPU، يجب أن تتمكن من تشغيل ["PixArtSigmaPipeline"] مع VRAM GPU أقل من 8 جيجا.

إذا كنت تريد تقريرًا عن استخدام الذاكرة، فقم بتشغيل هذا [script](https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e).

<Tip warning={true}>

يمكن أن تؤثر تضمين النص المحسوب بدقة 8 بت على جودة الصور المولدة بسبب فقدان المعلومات في مساحة التمثيل الناتج عن الدقة المخفضة. يُنصح بمقارنة المخرجات بدقة 8 بت وبدونها.

</Tip>

عند تحميل `text_encoder`، قمت بتعيين `load_in_8bit` إلى `True`. يمكنك أيضًا تحديد `load_in_4bit` لخفض متطلبات الذاكرة الخاصة بك إلى أقل من 7 جيجا بايت.

## PixArtSigmaPipeline

[[autodoc]] PixArtSigmaPipeline

- all
- __call__