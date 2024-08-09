# تحميل المحولات (الموائمات)

هناك العديد من تقنيات [التدريب](../training/overview) لتخصيص نماذج الانتشار لإنشاء صور لموضوع محدد أو صور بأساليب معينة. وينتج عن كل طريقة من طرق التدريب هذه نوع مختلف من المحولات. حيث يقوم بعضها بتوليد نموذج جديد بالكامل، بينما يقوم البعض الآخر بتعديل مجموعة فرعية فقط من المُعلمات أو الأوزان. وهذا يعني أن عملية التحميل لكل محول تختلف أيضًا.

سيوضح هذا الدليل كيفية تحميل أوزان DreamBooth وInversion النصي وLoRA.

<Tip>
يمكنك الاطلاع على [مستقر الانتشار المفاهيمي](https://huggingface.co/spaces/sd-concepts-library/stable-diffusion-conceptualizer) و[LoRA the Explorer](https://huggingface.co/spaces/multimodalart/LoraTheExplorer) و[معرض نماذج Diffusers](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery) للحصول على نقاط مرجعية ومعلمات للاستخدام.
</Tip>

## DreamBooth

يقوم [DreamBooth](https://dreambooth.github.io/) بتعديل دقيق لنموذج الانتشار *كامل* على مجرد عدة صور لموضوع ما لتوليد صور لهذا الموضوع بأساليب وإعدادات جديدة. تعمل هذه الطريقة من خلال استخدام كلمة خاصة في المطالبة التي يتعلمها النموذج لربطها بصورة الموضوع. ومن بين جميع طرق التدريب، ينتج DreamBooth أكبر حجم ملف (عادةً بضعة غيغابايت) لأنه نموذج نقطة مرجعية كامل.

دعونا نحمل نقطة المرجعية [herge_style](https://huggingface.co/sd-dreambooth-library/herge-style)، والتي تم تدريبها على 10 صور فقط رسمها Hergé، لتوليد الصور بهذا الأسلوب. لكي يعمل، تحتاج إلى تضمين الكلمة الخاصة `herge_style` في مطالبتك لتشغيل نقطة المرجعية:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("sd-dreambooth-library/herge-style", torch_dtype=torch.float16).to("cuda")
prompt = "A cute herge_style brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_dreambooth.png" />
</div>

## الانقلاب النصي

[الانقلاب النصي](https://textual-inversion.github.io/) مشابه جدًا لـ DreamBooth ويمكنه أيضًا تخصيص نموذج انتشار لتوليد مفاهيم معينة (الأساليب، الأشياء) من مجرد بضع صور. تعمل هذه الطريقة من خلال تدريب وإيجاد معلمات جديدة تمثل الصور التي تقدمها مع كلمة خاصة في المطالبة. ونتيجة لذلك، تظل أوزان نموذج الانتشار كما هي وتنتج عملية التدريب ملفًا صغيرًا جدًا (بضعة كيلوبايتات).

نظرًا لأن الانقلاب النصي يقوم بإنشاء معلمات، فإنه لا يمكن استخدامه بمفرده مثل DreamBooth ويتطلب نموذجًا آخر.

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

الآن يمكنك تحميل معلمات الانقلاب النصي باستخدام طريقة [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] وتوليد بعض الصور. دعونا نحمل معلمات [sd-concepts-library/gta5-artwork](https://huggingface.co/sd-concepts-library/gta5-artwork) وسيتعين عليك تضمين الكلمة الخاصة `<gta5-artwork>` في مطالبتك لتشغيلها:

```py
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, <gta5-artwork> style"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_txt_embed.png" />
</div>

يمكن أيضًا تدريب الانقلاب النصي على أشياء غير مرغوب فيها لإنشاء معلمات *سلبية* لمنع نموذج من إنشاء صور بتلك الأشياء غير المرغوب فيها مثل الصور الضبابية أو الأصابع الإضافية على اليد. يمكن أن يكون هذا طريقة سهلة لتحسين مطالبتك بسرعة. ستقوم أيضًا بتحميل المعلمات باستخدام [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`]، ولكن هذه المرة، ستحتاج إلى معلمتين أخريين:

- `weight_name`: يحدد ملف الأوزان لتحميله إذا تم حفظ الملف بتنسيق 🤗 Diffusers باسم محدد أو إذا تم تخزين الملف بتنسيق A1111
- `token`: يحدد الكلمة الخاصة التي سيتم استخدامها في المطالبة لتشغيل المعلمات

دعونا نحمل معلمات [sayakpaul/EasyNegative-test](https://huggingface.co/sayakpaul/EasyNegative-test):

```py
pipeline.load_textual_inversion(
"sayakpaul/EasyNegative-test", weight_name="EasyNegative.safetensors", token="EasyNegative"
)
```

الآن يمكنك استخدام `token` لتوليد صورة بمعلمات سلبية:

```py
prompt = "A cute brown bear eating a slice of pizza, stunning color scheme, masterpiece, illustration, EasyNegative"
negative_prompt = "EasyNegative"

image = pipeline(prompt, negative_prompt=negative_prompt, num_inference_steps=50).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png" />
</div>

## LoRA

[التكيف منخفض الرتبة (LoRA)](https://huggingface.co/papers/2106.09685) هي تقنية تدريب شائعة لأنها سريعة وتولد أحجام ملفات أصغر (بضع مئات الميغابايت). مثل الطرق الأخرى في هذا الدليل، يمكن لـ LoRA تدريب نموذج على تعلم أساليب جديدة من مجرد بضع صور. تعمل من خلال إدراج أوزان جديدة في نموذج الانتشار ثم تدريب الأوزان الجديدة فقط بدلاً من النموذج بالكامل. وهذا يجعل LoRAs أسرع في التدريب وأسهل في التخزين.

<Tip>
LoRA هي تقنية تدريب عامة جدًا يمكن استخدامها مع طرق تدريب أخرى. على سبيل المثال، من الشائع تدريب نموذج باستخدام DreamBooth وLoRA. كما أصبح من الشائع بشكل متزايد تحميل ودمج عدة LoRAs لإنشاء صور جديدة وفريدة. يمكنك معرفة المزيد عنها في دليل [دمج LoRAs](merge_loras) المتعمق نظرًا لأن الدمج خارج نطاق دليل التحميل هذا.
</Tip>

تحتاج LoRAs أيضًا إلى استخدامها مع نموذج آخر:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
```

ثم استخدم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] لتحميل أوزان [ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora) وحدد اسم ملف الأوزان من المستودع:

```py
pipeline.load_lora_weights("ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors")
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_lora.png" />
</div>

تحمّل طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] أوزان LoRA في كل من UNet ومشفر النص. إنها الطريقة المفضلة لتحميل LoRAs لأنها يمكن أن تتعامل مع الحالات التي:

- لا تحتوي أوزان LoRA على محددات منفصلة لـ UNet ومشفر النص
- تحتوي أوزان LoRA على محددات منفصلة لـ UNet ومشفر النص

ولكن إذا كنت بحاجة فقط إلى تحميل أوزان LoRA في UNet، فيمكنك استخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]. دعونا نحمل [jbilcke-hf/sdxl-cinematic-1](https://huggingface.co/jbilcke-hf/sdxl-cinematic-1) LoRA:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs("jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors")

# use cnmt in the prompt to trigger the LoRA
prompt = "A cute cnmt eating a slice of pizza, stunning color scheme, masterpiece, illustration"
image = pipeline(prompt).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_attn_proc.png" />
</div>

لإلغاء تحميل أوزان LoRA، استخدم طريقة [`~loaders.LoraLoaderMixin.unload_lora_weights`] للتخلص من أوزان LoRA واستعادة النموذج إلى أوزانه الأصلية:

```py
pipeline.unload_lora_weights()
```

### ضبط مقياس وزن LoRA

بالنسبة لكل من [`~loaders.LoraLoaderMixin.load_lora_weights`] و [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`]، يمكنك تمرير المعلمة `cross_attention_kwargs={"scale": 0.5}` لتعديل مقدار أوزان LoRA التي سيتم استخدامها. القيمة `0` هي نفسها مثل استخدام أوزان النموذج الأساسي فقط، والقيمة `1` تعادل استخدام LoRA المعدل الدقيق بالكامل.

لمزيد من التحكم الدقيق في مقدار أوزان LoRA المستخدمة لكل طبقة، يمكنك استخدام [`~loaders.LoraLoaderMixin.set_adapters`] وتمرير قاموس يحدد مقدار المقياس المستخدم في كل طبقة.

```python
pipe = ... # إنشاء خط أنابيب
pipe.load_lora_weights(..., adapter_name="my_adapter")
scales = {
"text_encoder": 0.5,
"text_encoder_2": 0.5، # قابل للاستخدام فقط إذا كان لدى pipe مشفر نص ثانٍ
"unet": {
"down": 0.9، # ستستخدم جميع المحولات في الجزء السفلي المقياس 0.9
# "mid" # في هذا المثال، لم يتم إعطاء "mid"، لذلك ستستخدم جميع المحولات في الجزء الأوسط المقياس الافتراضي 1.0
"up": {
"block_0": 0.6، # ستستخدم جميع المحولات الثلاثة في الكتلة 0 في الجزء العلوي المقياس 0.6
"block_1": [0.4، 0.8، 1.0]، # ستستخدم المحولات الثلاثة في الكتلة 1 في الجزء العلوي المقاييس 0.4 و 0.8 و 1.0 على التوالي
}
}
}
pipe.set_adapters("my_adapter"، scales)
```

يعمل هذا أيضًا مع محولات متعددة - راجع [هذا الدليل](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference#customize-adapters-strength) لمعرفة كيفية القيام بذلك.

<Tip warning={true}>
حاليًا، يدعم [`~loaders.LoraLoaderMixin.set_adapters`] فقط مقاييس أوزان الاهتمام. إذا كان لدى LoRA أجزاء أخرى (مثل شبكات ResNet أو down-/upsamplers)، فستظل مقاييسها 1.0.
</Tip>
### Kohya and TheLastBen

ومن مدربي LoRA الآخرين المعروفين في المجتمع ما أنشأه كل من [Kohya](https://github.com/kohya-ss/sd-scripts/) و [TheLastBen](https://github.com/TheLastBen/fast-stable-diffusion). وينشئ هذان المُدربان نقاط تفتيش مختلفة عن نقاط تفتيش LoRA التي تدربها 🤗 Diffusers، ولكن يمكن تحميلها بنفس الطريقة.

<hfoptions id="other-trainers">

<hfoption id="Kohya">

لتحميل نقطة تفتيش LoRA من Kohya، دعنا نقوم بتنزيل مثال على نقطة تفتيش [Blueprintify SD XL 1.0](https://civitai.com/models/150986/blueprintify-sd-xl-10) من [Civitai](https://civitai.com/):

```sh
!wget https://civitai.com/api/download/models/168776 -O blueprintify-sd-xl-10.safetensors
```

قم بتحميل نقطة تفتيش LoRA باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]، وحدد اسم الملف في معلمة `weight_name`:

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("path/to/weights", weight_name="blueprintify-sd-xl-10.safetensors")
```

قم بإنشاء صورة:

```py
# استخدم bl3uprint في المطالبة لتشغيل LoRA
prompt = "bl3uprint, a highly detailed blueprint of the eiffel tower, explaining how to build all parts, many txt, blueprint grid backdrop"
image = pipeline(prompt).images[0]
image
```

<Tip warning={true}>

تشمل بعض القيود على استخدام LoRAs من Kohya مع 🤗 Diffusers ما يلي:

- قد لا تبدو الصور مثل تلك التي تم إنشاؤها بواسطة واجهات المستخدم - مثل ComfyUI - لأسباب متعددة، والتي تم شرحها [هنا](https://github.com/huggingface/diffusers/pull/4287/#issuecomment-1655110736).
- لا يتم دعم [نقاط تفتيش LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) بشكل كامل. تقوم طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`] بتحميل نقاط تفتيش LyCORIS مع وحدات LoRA و LoCon، ولكن Hada و LoKR غير مدعومة.

</Tip>

</hfoption>

<hfoption id="TheLastBen">

تحميل نقطة تفتيش من TheLastBen مشابه جدا. على سبيل المثال، لتحميل نقطة تفتيش [TheLastBen/William_Eggleston_Style_SDXL](https://huggingface.co/TheLastBen/William_Eggleston_Style_SDXL):

```py
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("TheLastBen/William_Eggleston_Style_SDXL", weight_name="wegg.safetensors")

# استخدم by william eggleston في المطالبة لتشغيل LoRA
prompt = "a house by william eggleston, sunrays, beautiful, sunlight, sunrays, beautiful"
image = pipeline(prompt=prompt).images[0]
image
```

</hfoption>

</hfoptions>

## IP-Adapter

[IP-Adapter](https://ip-adapter.github.io/) عبارة عن محول خفيف الوزن يمكّن المطالبة بالصور لأي نموذج انتشار. تعمل هذه الأداة عن طريق فصل طبقات الاهتمام المتقاطع لميزات الصورة والنص. يتم تجميد جميع مكونات النموذج الأخرى، ويتم تدريب ميزات الصورة المضمنة في UNet فقط. ونتيجة لذلك، عادة ما تكون ملفات IP-Adapter بحجم ~100 ميجابايت فقط.

يمكنك معرفة المزيد حول كيفية استخدام IP-Adapter لمختلف المهام وحالات الاستخدام المحددة في دليل [IP-Adapter](../using-diffusers/ip_adapter).

> [!TIP]
> تدعم Diffusers حاليًا IP-Adapter لبعض الأنابيب الأكثر شهرة فقط. لا تتردد في فتح طلب ميزة إذا كان لديك حالة استخدام رائعة وتريد دمج IP-Adapter مع خط أنابيب غير مدعوم!
> تتوفر نقاط تفتيش IP-Adapter الرسمية من [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter).

للبدء، قم بتحميل نقطة تفتيش Stable Diffusion.

```py
from diffusers import AutoPipelineForText2Image
import torch
from diffusers.utils import load_image

pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
```

بعد ذلك، قم بتحميل أوزان IP-Adapter وإضافتها إلى الأنبوب باستخدام طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`].

```py
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
```

بمجرد التحميل، يمكنك استخدام الأنبوب بصورة وصورة مكتوبة لتوجيه عملية إنشاء الصورة.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png")
generator = torch.Generator(device="cpu").manual_seed(33)
images = pipeline(
prompt='best quality, high quality, wearing sunglasses',
ip_adapter_image=image,
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
num_inference_steps=50,
generator=generator,
).images[0]
images
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip-bear.png" />
</div>

### IP-Adapter Plus

يعتمد IP-Adapter على مشفر الصور لتوليد ميزات الصورة. إذا كان مستودع IP-Adapter يحتوي على مجلد فرعي `image_encoder`، يتم تحميل مشفر الصور تلقائيًا وتسجيله في الأنبوب. وإلا، سيتعين عليك تحميل مشفر الصور بشكل صريح باستخدام نموذج [`~transformers.CLIPVisionModelWithProjection`] وتمريره إلى الأنبوب.

هذا هو الحال بالنسبة لنقاط تفتيش *IP-Adapter Plus* التي تستخدم مشفر الصور ViT-H.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
"h94/IP-Adapter",
subfolder="models/image_encoder",
torch_dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
image_encoder=image_encoder,
torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
```

### نماذج IP-Adapter Face ID

نماذج IP-Adapter FaceID هي محولات IP تجريبية تستخدم تضمينات الصور التي تم إنشاؤها بواسطة `insightface` بدلاً من تضمينات الصور CLIP. يستخدم بعض هذه النماذج أيضًا LoRA لتحسين اتساق التعريف.

يجب تثبيت `insightface` وجميع متطلباتها لاستخدام هذه النماذج.

<Tip warning={true}>
نظرًا لأن النماذج المسبقة التدريب على InsightFace متاحة لأغراض البحث غير التجارية، يتم إصدار نماذج IP-Adapter-FaceID حصريًا لأغراض البحث ولا يُقصد بها الاستخدام التجاري.
</Tip>

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid_sdxl.bin", image_encoder_folder=None)
```

إذا كنت تريد استخدام أحد نموذجي IP-Adapter FaceID Plus، فيجب عليك أيضًا تحميل مشفر الصور CLIP، حيث تستخدم هذه النماذج كل من تضمينات الصور `insightface` و CLIP لتحقيق واقعية أفضل.

```py
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
"laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
torch_dtype=torch.float16,
)

pipeline = AutoPipelineForText2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5",
image_encoder=image_encoder,
torch_dtype=torch.float16
).to("cuda")

pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid-plus_sd15.bin")
```