
# تحميل LoRAs للاستنتاج

هناك العديد من أنواع المحولات (مع [LoRAs](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) الأكثر شعبية) المدربة بأساليب مختلفة لتحقيق تأثيرات مختلفة. يمكنك حتى الجمع بين عدة محولات لإنشاء صور جديدة وفريدة من نوعها.

في هذا البرنامج التعليمي، ستتعلم كيفية تحميل وإدارة المحولات بسهولة للاستنتاج مع تكامل 🤗 [PEFT](https://huggingface.co/docs/peft/index) في 🤗 Diffusers. ستستخدم تقنية LoRA كتقنية محول أساسية، لذا سترى مصطلحي LoRA والمحول المستخدمة بالتبادل.

دعونا أولاً نقوم بتثبيت جميع المكتبات المطلوبة.

```bash
!pip install -q transformers accelerate peft diffusers
```

الآن، قم بتحميل خط أنابيب باستخدام نقطة تفتيش [Stable Diffusion XL (SDXL)](../api/pipelines/stable_diffusion/stable_diffusion_xl):

```python
from diffusers import DiffusionPipeline
import torch

pipe_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(pipe_id, torch_dtype=torch.float16).to("cuda")
```

بعد ذلك، قم بتحميل محول [CiroN2022/toy-face](https://huggingface.co/CiroN2022/toy-face) باستخدام طريقة [`~diffusers.loaders.StableDiffusionXLLoraLoaderMixin.load_lora_weights`]. مع تكامل 🤗 PEFT، يمكنك تعيين اسم محدد `adapter_name` لنقطة التفتيش، مما يتيح لك التبديل بسهولة بين نقاط تفتيش LoRA المختلفة. دعونا نطلق على هذا المحول اسم "toy".

```python
pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")
```

تأكد من تضمين الرمز `toy_face` في المطالبة، ثم يمكنك إجراء الاستدلال:

```python
prompt = "toy_face of a hacker with a hoodie"

lora_scale = 0.9
image = pipe(
prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_8_1.png)

مع معلمة `adapter_name`، من السهل جدًا استخدام محول آخر للاستدلال! قم بتحميل محول [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl) الذي تم ضبط دقته لتوليد صور فن البكسل ودعوته "pixel".

يحدد خط الأنابيب تلقائيًا المحول المحمل أولاً (`"toy"`) كمحول نشط، ولكن يمكنك تنشيط محول `"pixel"` باستخدام طريقة [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`]:

```python
pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")
```

تأكد من تضمين الرمز `pixel art` في مطالبتك لتوليد صورة فن البكسل:

```python
prompt = "a hacker with a hoodie, pixel art"
image = pipe(
prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

![pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_12_1.png)

## دمج المحولات

يمكنك أيضًا دمج نقاط تفتيش محول مختلفة للاستدلال لمزج أساليبها معًا.

باستخدام طريقة [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] مرة أخرى، قم بتنشيط محولات `pixel` و`toy` وحدد الأوزان لكيفية دمجها.

```python
pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])
```

<Tip>

غالبًا ما يتم الحصول على نقاط تفتيش LoRA في مجتمع الانتشار باستخدام [DreamBooth](https://huggingface.co/docs/diffusers/main/en/training/dreambooth). غالبًا ما يعتمد التدريب على DreamBooth على كلمات "مُشغِّل" في مطالبات نص الإدخال حتى تبدو نتائج التوليد كما هو متوقع. عند الجمع بين عدة نقاط تفتيش LoRA، من المهم التأكد من وجود كلمات التشغيل المقابلة لنقاط تفتيش LoRA في مطالبات نص الإدخال.

</Tip>

تذكر استخدام كلمات التشغيل لـ [CiroN2022/toy-face](https://hf.co/CiroN2022/toy-face) و [nerijs/pixel-art-xl](https://hf.co/nerijs/pixel-art-xl) (الموجودة في مستودعاتها) في المطالبة لتوليد صورة.

```python
prompt = "toy_face of a hacker with a hoodie, pixel art"
image = pipe(
prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}, generator=torch.manual_seed(0)
).images[0]
image
```

![toy-face-pixel-art](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_16_1.png)

مثير للإعجاب! كما ترون، قام النموذج بتوليد صورة مزجت خصائص كلا المحولين.

> [!TIP]
> من خلال تكامل PEFT، تقدم Diffusers أيضًا طرق دمج أكثر كفاءة والتي يمكنك معرفتها في دليل [Merge LoRAs](../using-diffusers/merge_loras)!

للعودة إلى استخدام محول واحد فقط، استخدم طريقة [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`] لتنشيط محول `"toy"`:

```python
pipe.set_adapters("toy")

prompt = "toy_face of a hacker with a hoodie"
lora_scale = 0.9
image = pipe(
prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}, generator=torch.manual_seed(0)
).images[0]
image
```

أو لإيقاف جميع المحولات تمامًا، استخدم طريقة [`~diffusers.loaders.UNet2DConditionLoadersMixin.disable_lora`] للعودة إلى النموذج الأساسي.

```python
pipe.disable_lora()

prompt = "toy_face of a hacker with a hoodie"
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![no-lora](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_20_1.png)

### تخصيص قوة المحولات

لمزيد من التخصيص، يمكنك التحكم في مدى تأثير المحول على كل جزء من خط الأنابيب. للقيام بذلك، قم بتمرير قاموس بقيم القوة (تسمى "المقاييس") إلى [`~diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters`].

على سبيل المثال، إليك كيفية تشغيل المحول لأسفل الأجزاء، ولكن إيقاف تشغيله لأسفل `mid` و`up` الأجزاء:

```python
pipe.enable_lora() # enable lora again, after we disabled it above
prompt = "toy_face of a hacker with a hoodie, pixel art"
adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-text-and-down](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_down.png)

دعونا نرى كيف يتغير الصورة عند إيقاف تشغيل الجزء `down` وتشغيل الجزء `mid` و`up` على التوالي.

```python
adapter_weight_scales = { "unet": { "down": 0, "mid": 1, "up": 0} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-text-and-mid](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_mid.png)

```python
adapter_weight_scales = { "unet": { "down": 0, "mid": 0, "up": 1} }
pipe.set_adapters("pixel", adapter_weight_scales)
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-text-and-up](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_up.png)

تبدو رائعة!

هذه ميزة قوية حقًا. يمكنك استخدامه للتحكم في قوى المحول وصولاً إلى مستوى المحول الفردي. ويمكنك حتى استخدامه لمحولات متعددة.

```python
adapter_weight_scales_toy = 0.5
adapter_weight_scales_pixel = {
"unet": {
"down": 0.9, # جميع المحولات في الجزء السفلي ستستخدم المقياس 0.9
# "mid" # لأن "mid" غير محدد في هذا المثال، ستستخدم جميع المحولات في الجزء الأوسط مقياس 1.0 الافتراضي
"up": {
"block_0": 0.6، # جميع المحولات الثلاثة في الكتلة 0 في الجزء العلوي ستستخدم المقياس 0.6
"block_1": [0.4، 0.8، 1.0]، # ستستخدم المحولات الثلاثة في الكتلة 1 في الجزء العلوي المقاييس 0.4 و0.8 و1.0 على التوالي
}
}
}
pipe.set_adapters(["toy"، "pixel"]، [adapter_weight_scales_toy، adapter_weight_scales_pixel])
image = pipe(prompt, num_inference_steps=30, generator=torch.manual_seed(0)).images[0]
image
```

![block-lora-mixed](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/peft_integration/diffusers_peft_lora_inference_block_mixed.png)

## إدارة المحولات النشطة

لقد قمت بتعليق عدة محولات في هذا البرنامج التعليمي، وإذا شعرت بالحيرة قليلاً بشأن المحولات التي تم ربطها بمكونات خط الأنابيب، فاستخدم طريقة [`~diffusers.loaders.LoraLoaderMixin.get_active_adapters`] للتحقق من قائمة المحولات النشطة:

```py
active_adapters = pipe.get_active_adapters()
active_adapters
["toy"، "pixel"]
```

يمكنك أيضًا الحصول على المحولات النشطة لكل مكون من مكونات خط الأنابيب باستخدام [`~diffusers.loaders.LoraLoaderMixin.get_list_adapters`]:

```py
list_adapters_component_wise = pipe.get_list_adapters()
list_adapters_component_wise
{"text_encoder": ["toy"، "pixel"]، "unet": ["toy"، "pixel"]، "text_encoder_2": ["toy"، "pixel"]}
```