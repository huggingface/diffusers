# تكييف نموذج لمهمة جديدة

تشترك العديد من أنظمة الانتشار في نفس المكونات، مما يتيح لك تكييف نموذج مُدرب مسبقًا لمهمة واحدة لمهمة مختلفة تمامًا. سيُظهر لك هذا الدليل كيفية تكييف نموذج مُدرب مسبقًا للنص إلى الصورة للرسم التلقائي من خلال تهيئة وتعديل بنية نموذج [`UNet2DConditionModel`] مُدرب مسبقًا.

## تكوين معلمات UNet2DConditionModel

يقبل [`UNet2DConditionModel`] بشكل افتراضي 4 قنوات في [عينة الإدخال](https://huggingface.co/docs/diffusers/v0.16.0/en/api/models#diffusers.UNet2DConditionModel.in_channels). على سبيل المثال، قم بتحميل نموذج مُدرب مسبقًا للنص إلى الصورة مثل [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) وتحقق من عدد `in_channels`:

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_safetensors=True)
pipeline.unet.config["in_channels"]
4
```

يتطلب الرسم التلقائي 9 قنوات في عينة الإدخال. يمكنك التحقق من هذه القيمة في نموذج الرسم التلقائي المُدرب مسبقًا مثل [`runwayml/stable-diffusion-inpainting`](https://huggingface.co/runwayml/stable-diffusion-inpainting):

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", use_safetensors=True)
pipeline.unet.config["in_channels"]
9
```

لتكييف نموذج النص إلى الصورة للرسم التلقائي، ستحتاج إلى تغيير عدد `in_channels` من 4 إلى 9. قم بتهيئة [`UNet2DConditionModel`] باستخدام أوزان النموذج المُدرب مسبقًا للنص إلى الصورة، وقم بتغيير `in_channels` إلى 9. يعني تغيير عدد `in_channels` أنه يتعين عليك تعيين `ignore_mismatched_sizes=True` و`low_cpu_mem_usage=False` لتجنب حدوث خطأ عدم تطابق الحجم لأن الشكل مختلف الآن.

```py
from diffusers import UNet2DConditionModel

model_id = "runwayml/stable-diffusion-v1-5"
unet = UNet2DConditionModel.from_pretrained(
model_id,
subfolder="unet",
in_channels=9,
low_cpu_mem_usage=False,
ignore_mismatched_sizes=True,
use_safetensors=True,
)
```

يتم تهيئة الأوزان المُدربة مسبقًا للمكونات الأخرى من نموذج النص إلى الصورة من نقاط التفتيش الخاصة بها، ولكن يتم تهيئة أوزان قناة الإدخال (`conv_in.weight`) لـ `unet` بشكل عشوائي. من المهم ضبط دقة النموذج للرسم التلقائي؛ وإلا فإن النموذج سيعيد ضجيجًا.