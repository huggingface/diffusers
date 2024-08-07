# PixArt-α

[PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://huggingface.co/papers/2310.00426) هو عمل من تأليف جونسونغ تشن، وجينتشينغ يو، وتشونغجيان جي، وليوي يايو، وإينزي زي، ويوي وو، وتشونغداو وانغ، وجيمس كووك، وبينغ ليو، وهوتشوان لو، وزينغو لي.

ملخص الورقة البحثية هو:

*تتطلب أكثر نماذج تحويل النص إلى صورة (Text-to-Image) تقدماً تكاليف تدريب كبيرة (على سبيل المثال، ملايين الساعات من وحدة معالجة الرسوميات GPU)، مما يعيق بشكل خطير الابتكار الأساسي لمجتمع الذكاء الاصطناعي الإبداعي (AIGC) مع زيادة انبعاثات ثاني أكسيد الكربون. تقدم هذه الورقة نموذج PIXART-α، وهو نموذج انتشار قائم على محول (Transformer) لتحويل النص إلى صورة، تتنافس جودة توليد الصور فيه مع مولدات الصور المتقدمة (مثل Imagen وSDXL وحتى Midjourney)، حيث يصل إلى معايير التطبيقات التجارية القريبة. بالإضافة إلى ذلك، فإنه يدعم توليف الصور عالية الدقة بحد أقصى 1024 بكسل مع انخفاض تكلفة التدريب، كما هو موضح في الشكل 1 و2. ولتحقيق هذا الهدف، تم اقتراح ثلاثة تصاميم أساسية: (1) استراتيجية التدريب التفكيك: قمنا بتصميم ثلاث خطوات تدريب متميزة تحين بشكل منفصل اعتماد البكسل، والمحاذاة بين النص والصورة، وجودة الصورة الجمالية؛ (2) محول T2I الكفء: قمنا بدمج وحدات الاهتمام المتبادل في محول الانتشار (DiT) لحقن شروط النص وتبسيط فرع الشروط الحسابية المكثفة؛ (3) البيانات عالية المعلومات: نؤكد على أهمية كثافة المفاهيم في أزواج النص والصورة ونستفيد من نموذج اللغة والرؤية واسع النطاق لوضع علامات تلقائية على التعليقات التوضيحية الكثيفة الزائفة للمساعدة في تعلم محاذاة النص والصورة. ونتيجة لذلك، تتفوق سرعة تدريب PIXART-α بشكل ملحوظ على النماذج الكبيرة الحالية لتحويل النص إلى صورة، على سبيل المثال، يستغرق PIXART-α فقط 10.8% من وقت تدريب Stable Diffusion v1.5 (675 مقابل 6,250 يوم من أيام وحدة معالجة الرسوميات GPU من نوع A100)، مما يوفر حوالي 300,000 دولار (26,000 دولار مقابل 320,000 دولار) ويقلل انبعاثات ثاني أكسيد الكربون بنسبة 90%. علاوة على ذلك، مقارنة بنموذج SOTA أكبر، وهو RAPHAEL، تبلغ تكلفة تدريبنا مجرد 1%. وتظهر التجارب المستفيضة أن PIXART-α يتفوق في جودة الصورة والإبداع والتحكم الدلالي. نأمل أن يوفر PIXART-α رؤى جديدة لمجتمع الذكاء الاصطناعي الإبداعي والشركات الناشئة لتسريع بناء نماذجهم التوليدية عالية الجودة ومنخفضة التكلفة من الصفر.*

يمكنك العثور على الشفرة البرمجية الأصلية في [PixArt-alpha/PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha) وجميع نقاط التفتيش المتاحة في [PixArt-alpha](https://huggingface.co/PixArt-alpha).

بعض الملاحظات حول هذا الخط الأنابيب:

* يستخدم العمود الفقري لمحول (Transformer) (بدلاً من U-Net) لإزالة التشويش. وبهذه الطريقة، يكون له بنية مماثلة لـ [DiT](./dit).
* تم تدريبه باستخدام شروط نصية محسوبة من T5. ويجعل هذا الجانب الخط الأنابيب أفضل في اتباع موجهات النص المعقدة مع تفاصيل دقيقة.
* إنه جيد في إنتاج صور عالية الدقة بمعدلات ارتفاع مختلفة. وللحصول على أفضل النتائج، يوصي المؤلفون ببعض الأقواس الحجمية التي يمكن العثور عليها [هنا](https://github.com/PixArt-alpha/PixArt-alpha/blob/08fbbd281ec96866109bdd2cdb75f2f58fb17610/diffusion/data/datasets/utils.py).
* ينافس جودة أنظمة تحويل النص إلى صورة المتقدمة (اعتبارًا من وقت الكتابة) مثل Stable Diffusion XL وImagen وDALL-E 2، بينما يكون أكثر كفاءة منها.

## الاستنتاج باستخدام ذاكرة GPU VRAM أقل من 8 جيجابايت

قم بتشغيل [`PixArtAlphaPipeline`] باستخدام ذاكرة GPU VRAM أقل من 8 جيجابايت عن طريق تحميل مشفر النص بدقة 8 بت. دعونا نتعمق في مثال شامل.

أولاً، قم بتثبيت مكتبة [bitsandbytes](https://github.com/TimDettmers/bitsandbytes):

```bash
pip install -U bitsandbytes
```

ثم قم بتحميل مشفر النص بدقة 8 بت:

```python
from transformers import T5EncoderModel
from diffusers import PixArtAlphaPipeline
import torch

text_encoder = T5EncoderModel.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    subfolder="text_encoder",
    load_in_8bit=True,
    device_map="auto",

)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
    text_encoder=text_encoder,
    transformer=None,
    device_map="auto"
)
```

الآن، استخدم `pipe` لتشفير موجه النص:

```python
with torch.no_grad():
    prompt = "cute cat"
    prompt_embeds, prompt_attention_mask, negative_embeds, negative_prompt_attention_mask = pipe.encode_prompt(prompt)
```

نظرًا لحساب تضمين النص، قم بإزالة `text_encoder` و`pipe` من الذاكرة، وحرر بعض ذاكرة GPU VRAM:

```python
import gc

def flush():
gc.collect()
torch.cuda.empty_cache()

del text_encoder
del pipe
flush()
```

بعد ذلك، احسب المخفيات مع تضمينات موجه النص كمدخلات:

```python
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-1024-MS",
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

لاحظ أنه أثناء تهيئة `pipe`، تقوم بتعيين `text_encoder` إلى `None` بحيث لا يتم تحميله.

</Tip>

بمجرد حساب المخفيات، مررها إلى الشبكة العصبية التلافيفية (VAE) لفك تشفيرها إلى صورة حقيقية:

```python
with torch.no_grad():
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    image.save("cat.png")
```

من خلال حذف المكونات التي لا تستخدمها وتفريغ ذاكرة GPU VRAM، يجب أن تتمكن من تشغيل [`PixArtAlphaPipeline`] باستخدام ذاكرة GPU VRAM أقل من 8 جيجابايت.

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pixart/8bits_cat.png)

إذا كنت تريد تقريراً عن استخدام الذاكرة، فقم بتشغيل هذا [script](https://gist.github.com/sayakpaul/3ae0f847001d342af27018a96f467e4e).

<Tip warning={true}>

يمكن أن تؤثر تضمينات النص المحسوبة بدقة 8 بت على جودة الصور المولدة بسبب فقدان المعلومات في مساحة التمثيل الناتج عن الدقة المخفضة. يوصى بمقارنة المخرجات بدقة 8 بت وبدونها.

</Tip>

أثناء تحميل `text_encoder`، قمت بتعيين `load_in_8bit` إلى `True`. يمكنك أيضًا تحديد `load_in_4bit` لخفض متطلبات الذاكرة الخاصة بك إلى أقل من 7 جيجابايت.

## PixArtAlphaPipeline

[[autodoc]] PixArtAlphaPipeline
- all
- __call__