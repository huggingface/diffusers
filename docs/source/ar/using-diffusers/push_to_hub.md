# رفع الملفات إلى المنصة

[[open-in-colab]]

يوفر 🤗 Diffusers [`~diffusers.utils.PushToHubMixin`] لرفع نموذجك أو جدولك الزمني أو خط أنابيبك إلى المنصة. إنها طريقة سهلة لتخزين ملفاتك على المنصة، كما تتيح لك مشاركة عملك مع الآخرين. وفيما يلي، يقوم [`~diffusers.utils.PushToHubMixin`]:

1. إنشاء مستودع على المنصة
2. حفظ ملفات نموذجك أو جدولك الزمني أو خط أنابيبك بحيث يمكن إعادة تحميلها لاحقًا
3. تحميل المجلد الذي يحتوي على هذه الملفات إلى المنصة

سيوضح هذا الدليل كيفية استخدام [`~diffusers.utils.PushToHubMixin`] لتحميل ملفاتك إلى المنصة.

يجب تسجيل الدخول أولاً إلى حسابك على المنصة باستخدام رمز [الوصول](https://huggingface.co/settings/tokens):

```py
from huggingface_hub import notebook_login

notebook_login()
```

## النماذج

لرفع نموذج إلى المنصة، اتصل [`~diffusers.utils.PushToHubMixin.push_to_hub`] وحدد معرف مستودع النموذج المراد تخزينه على المنصة:

```py
from diffusers import ControlNetModel

controlnet = ControlNetModel(
block_out_channels=(32, 64),
layers_per_block=2,
in_channels=4,
down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
cross_attention_dim=32,
conditioning_embedding_out_channels=(16, 32),
)
controlnet.push_to_hub("my-controlnet-model")
```

بالنسبة للنماذج، يمكنك أيضًا تحديد [*variant*](loading#checkpoint-variants) من الأوزان لرفعها إلى المنصة. على سبيل المثال، لدفع أوزان `fp16`:

```py
controlnet.push_to_hub("my-controlnet-model", variant="fp16")
```

يقوم [`~diffusers.utils.PushToHubMixin.push_to_hub`] بحفظ ملف `config.json` للنموذج ويتم حفظ الأوزان تلقائيًا بتنسيق `safetensors`.

الآن يمكنك إعادة تحميل النموذج من مستودعك على المنصة:

```py
model = ControlNetModel.from_pretrained("your-namespace/my-controlnet-model")
```

## الجدول الزمني

لرفع جدول زمني إلى المنصة، اتصل [`~diffusers.utils.PushToHubMixin.push_to_hub`] وحدد معرف مستودع الجدول الزمني المراد تخزينه على المنصة:

```py
from diffusers import DDIMScheduler

scheduler = DDIMScheduler(
beta_start=0.00085,
beta_end=0.012,
beta_schedule="scaled_linear",
clip_sample=False,
set_alpha_to_one=False,
)
scheduler.push_to_hub("my-controlnet-scheduler")
```

يقوم [`~diffusers.utils.PushToHubMixin.push_to_hub`] بحفظ ملف `scheduler_config.json` للجدول الزمني إلى المستودع المحدد.

الآن يمكنك إعادة تحميل الجدول الزمني من مستودعك على المنصة:

```py
scheduler = DDIMScheduler.from_pretrained("your-namepsace/my-controlnet-scheduler")
```

## خط الأنابيب

يمكنك أيضًا رفع خط أنابيب كامل مع جميع مكوناته إلى المنصة. على سبيل المثال، قم بتهيئة مكونات [`StableDiffusionPipeline`] بالمعلمات التي تريدها:

```py
from diffusers import (
UNet2DConditionModel,
AutoencoderKL,
DDIMScheduler,
StableDiffusionPipeline,
)
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

unet = UNet2DConditionModel(
block_out_channels=(32, 64),
layers_per_block=2,
sample_size=32,
in_channels=4,
out_channels=4,
down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
cross_attention_dim=32,
)

scheduler = DDIMScheduler(
beta_start=0.00085,
beta_end=0.012,
beta_schedule="scaled_linear",
clip_sample=False,
set_alpha_to_one=False,
)

vae = AutoencoderKL(
block_out_channels=[32, 64],
in_channels=3,
out_channels=3,
down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
latent_channels=4,
)

text_encoder_config = CLIPTextConfig(
bos_token_id=0,
eos_token_Multiplier=2,
hidden_size=32,
intermediate_size=37,
layer_norm_eps=1e-05,
num_attention_heads=4,
num_hidden_layers=5,
pad_token_id=1,
vocab_size=1000,
)
text_encoder = CLIPTextModel(text_encoder_config)
tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
```

مرر جميع المكونات إلى [`StableDiffusionPipeline`] واتصل [`~diffusers.utils.PushToHubMixin.push_to_hub`] لدفع خط الأنابيب إلى المنصة:

```py
components = {
"unet": unet,
"scheduler": scheduler,
"vae": vae,
"text_encoder": text_encoder,
"tokenizer": tokenizer,
"safety_checker": None,
"feature_extractor": None,
}

pipeline = StableDiffusionPipeline(**components)
pipeline.push_to_hub("my-pipeline")
```

يقوم [`~diffusers.utils.PushToHubMixin.push_to_hub`] بحفظ كل مكون في مجلد فرعي داخل المستودع. الآن يمكنك إعادة تحميل خط الأنابيب من مستودعك على المنصة:

```py
pipeline = StableDiffusionPipeline.from_pretrained("your-namespace/my-pipeline")
```

## الخصوصية

قم بتعيين `private=True` في [`~diffusers.utils.PushToHubMixin.push_to_hub`] لوظيفة للحفاظ على ملفات نموذجك أو جدولك الزمني أو خط أنابيبك خاصة:

```py
controlnet.push_to_hub("my-controlnet-model-private", private=True)
```

المستودعات الخاصة مرئية لك فقط، ولن يتمكن المستخدمون الآخرون من استنساخ المستودع ولن يظهر مستودعك في نتائج البحث. حتى إذا كان لدى المستخدم عنوان URL لمستودعك الخاص، فسيحصل على `404 - عذرًا، لا يمكننا العثور على الصفحة التي تبحث عنها`. يجب أن يكون المستخدم [مسجلاً الدخول](https://huggingface.co/docs/huggingface_hub/quick-start#login) لتحميل نموذج من مستودع خاص.