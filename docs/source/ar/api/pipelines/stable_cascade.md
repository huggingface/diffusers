# Stable Cascade

هذا النموذج مبني على بنية [Würstchen](https://openreview.net/forum?id=gU58d5QeGv) ويتمثل الاختلاف الرئيسي بينه وبين النماذج الأخرى مثل Stable Diffusion في أنه يعمل في مساحة خفية أصغر بكثير. لماذا هذا مهم؟ كلما صغرت المساحة الخفية، كلما زادت **سرعة** تشغيل الاستنتاج و**رخصت** تكلفة التدريب. ما مدى صغر المساحة الخفية؟ يستخدم Stable Diffusion عامل ضغط قدره 8، مما يؤدي إلى ضغط صورة بحجم 1024x1024 إلى 128x128. ويحقق Stable Cascade عامل ضغط قدره 42، مما يعني أنه من الممكن ضغط صورة بحجم 1024x1024 إلى 24x24 مع الحفاظ على إعادة البناء الواضحة. يتم بعد ذلك تدريب النموذج المشروط بالنص في مساحة الخفاء عالية الضغط. حققت الإصدارات السابقة من هذه البنية خفضًا في التكلفة قدره 16x مقارنة بـ Stable Diffusion 1.5.

لذلك، هذا النوع من النماذج مناسب تمامًا للاستخدامات التي تكون فيها الكفاءة مهمة. علاوة على ذلك، فإن جميع الإضافات المعروفة مثل الضبط الدقيق، وLoRA، وControlNet، وIP-Adapter، وLCM، وما إلى ذلك، ممكنة أيضًا مع هذه الطريقة.

يمكن العثور على الكود الأصلي في [Stability-AI/StableCascade](https://github.com/Stability-AI/StableCascade).

## نظرة عامة على النموذج

يتكون Stable Cascade من ثلاثة نماذج: المرحلة A، والمرحلة B، والمرحلة C، والتي تمثل شلالًا لتوليد الصور، ومن هنا جاء اسم "Stable Cascade".

تُستخدم المرحلتان A وB لضغط الصور، على غرار مهمة VAE في Stable Diffusion. ومع ذلك، مع هذا الإعداد، يمكن تحقيق ضغط أعلى بكثير للصور. في حين أن نماذج Stable Diffusion تستخدم عامل ضغط مكاني قدره 8، لضغط صورة بدقة 1024x1024 إلى 128x128، يحقق Stable Cascade عامل ضغط قدره 42. وهذا يضغط صورة بحجم 1024x1024 إلى 24x24، مع القدرة على فك تشفير الصورة بدقة. يأتي هذا بميزة كبيرة تتمثل في رخص التدريب والاستدلال. علاوة على ذلك، فإن المرحلة C مسؤولة عن توليد الخفيات الصغيرة 24x24 بالنظر إلى موجه النص.

تعمل مرحلة النموذج C على الخفيات الصغيرة 24x24 وتقوم بإزالة الضوضاء من الخفيات المشروطة بموجهات النص. النموذج هو أيضًا أكبر مكون في خط أنابيب الشلال، ويُقصد استخدامه مع `StableCascadePriorPipeline`.

تُستخدم مرحلتا النموذج B وA مع `StableCascadeDecoderPipeline` وهما مسؤولتان عن توليد الصورة النهائية بالنظر إلى الخفيات الصغيرة 24x24.

<Tip warning={true}>
هناك بعض القيود على أنواع البيانات التي يمكن استخدامها مع نماذج Stable Cascade. لا تدعم نقاط التفتيش الرسمية لـ `StableCascadePriorPipeline` نوع بيانات `torch.float16`. يرجى استخدام `torch.bfloat16` بدلاً من ذلك.

لاستخدام نوع بيانات `torch.bfloat16` مع `StableCascadeDecoderPipeline`، يجب أن يكون لديك إصدار PyTorch 2.2.0 أو أعلى مثبتًا. وهذا يعني أيضًا أن استخدام `StableCascadeCombinedPipeline` مع `torch.bfloat16` يتطلب إصدار PyTorch 2.2.0 أو أعلى، لأنه يستدعي `StableCascadeDecoderPipeline` داخليًا.

إذا لم يكن من الممكن تثبيت إصدار PyTorch 2.2.0 أو أعلى في بيئتك، فيمكن استخدام `StableCascadeDecoderPipeline` بمفرده مع نوع بيانات `torch.float16`. يمكنك تنزيل أوزان الإصدار الكامل أو متغير `bf16` لخط الأنابيب وتحويل الأوزان إلى `torch.float16`.
</Tip>

## مثال على الاستخدام

```python
import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", variant="bf16", torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", variant="bf16", torch_dtype=torch.float16)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings.to(torch.float16),
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade.png")
```

## استخدام الإصدارات الخفيفة من نماذج المرحلتين B وC

```python
import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade-prior", subfolder="prior_lite")
decoder_unet = StableCascadeUNet.from_pretrained("stabilityai/stable-cascade", subfolder="decoder_lite")

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade.png")
```

## تحميل نقاط التفتيش الأصلية باستخدام `from_single_file`

يتم دعم تحميل نقاط تفتيش بتنسيق الأصلي عبر طريقة `from_single_file` في StableCascadeUNet.

```python
import torch
from diffusers import (
    StableCascadeDecoderPipeline,
    StableCascadePriorPipeline,
    StableCascadeUNet,
)

prompt = "an image of a shiba inu, donning a spacesuit and helmet"
negative_prompt = ""

prior_unet = StableCascadeUNet.from_single_file(
    "https://huggingface.co/stabilityai/stable-cascade/resolve/main/stage_c_bf16.safetensors",
torch_dtype=torch.bfloat16
)
decoder_unet = StableCascadeUNet.from_single_file(
    "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_bf16.safetensors",
torch_dtype=torch.bfloat16
)

prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", prior=prior_unet, torch_dtype=torch.bfloat16)
decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", decoder=decoder_unet, torch_dtype=torch.bfloat16)

prior.enable_model_cpu_offload()
prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=1,
    num_inference_steps=20
)

decoder.enable_model_cpu_offload()
decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
    num_inference_steps=10
).images[0]
decoder_output.save("cascade-single-file.png")
```

## الاستخدامات

### الاستخدام المباشر

يُقصد بالنموذج أن يكون لأغراض البحث في الوقت الحالي. تشمل مجالات البحث والمهام المحتملة ما يلي:

- البحث في النماذج التوليدية.
- النشر الآمن للنماذج التي لديها القدرة على توليد محتوى ضار.
- استكشاف وفهم قيود وتحيزات النماذج التوليدية.
- توليد الأعمال الفنية واستخدامها في التصميم والعمليات الفنية الأخرى.
- التطبيقات في الأدوات التعليمية أو الإبداعية.

يتم وصف الاستخدامات المستبعدة أدناه.

### الاستخدام خارج النطاق

لم يتم تدريب النموذج ليكون تمثيلًا دقيقًا أو حقيقيًا للأشخاص أو الأحداث، ولذلك فإن استخدام النموذج لتوليد مثل هذا المحتوى خارج نطاق قدرات هذا النموذج.

لا ينبغي استخدام النموذج بأي طريقة تنتهك سياسة الاستخدام المقبول لدى Stability AI [Acceptable Use Policy](https://stability.ai/use-policy).

## القيود والتحيز

### القيود

- قد لا يتم توليد الوجوه والأشخاص بشكل عام بشكل صحيح.
- الجزء الخاص بالترميز التلقائي من النموذج يفقد بعض البيانات.

## StableCascadeCombinedPipeline

[[autodoc]] StableCascadeCombinedPipeline
- all
- __call__

## StableCascadePriorPipeline

[[autodoc]] StableCascadePriorPipeline
- all
- __call__

## StableCascadePriorPipelineOutput

[[autodoc]] pipelines.stable_cascade.pipeline_stable_cascade_prior.StableCascadePriorPipelineOutput

## StableCascadeDecoderPipeline

[[autodoc]] StableCascadeDecoderPipeline
- all
- __call__