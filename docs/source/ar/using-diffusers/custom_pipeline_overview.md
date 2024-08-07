# تحميل خطوط الأنابيب والمكونات المجتمعية 

[[open-in-colab]]

## خطوط أنابيب المجتمع 

> [!TIP] راجع GitHub Issue [#841](https://github.com/huggingface/diffusers/issues/841) لمزيد من السياق حول سبب إضافة خطوط أنابيب المجتمع لمساعدة الجميع على مشاركة عملهم بسهولة دون إبطاء.

خطوط أنابيب المجتمع هي أي فئة [`DiffusionPipeline`] تختلف عن التنفيذ الورقي الأصلي (على سبيل المثال، يتوافق [`StableDiffusionControlNetPipeline`] مع ورقة [Text-to-Image Generation with ControlNet Conditioning](https://arxiv.org/abs/2302.05543)). إنها توفر وظائف إضافية أو تمدد التنفيذ الأصلي لخط الأنابيب.

هناك العديد من خطوط أنابيب المجتمع الرائعة مثل [Marigold Depth Estimation](https://github.com/huggingface/diffusers/tree/main/examples/community#marigold-depth-estimation) أو [InstantID](https://github.com/huggingface/diffusers/tree/main/examples/community#instantid-pipeline)، ويمكنك العثور على جميع خطوط الأنابيب المجتمعية الرسمية [هنا](https://github.com/huggingface/diffusers/tree/main/examples/community).

هناك نوعان من خطوط أنابيب المجتمع، تلك المخزنة في Hugging Face Hub وتلك المخزنة في مستودع GitHub Diffusers. يمكن تخصيص خطوط أنابيب Hub بشكل كامل (المخطط، النماذج، كود خط الأنابيب، إلخ.)، في حين أن خطوط أنابيب GitHub في Diffusers تقتصر فقط على كود خط الأنابيب المخصص.

|                | GitHub community pipeline                                                                                        | HF Hub community pipeline                                                                 |
|----------------|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| usage          | same                                                                                                             | same                                                                                      |
| review process | open a Pull Request on GitHub and undergo a review process from the Diffusers team before merging; may be slower | upload directly to a Hub repository without any review; this is the fastest workflow      |
| visibility     | included in the official Diffusers repository and documentation                                                  | included on your HF Hub profile and relies on your own usage/promotion to gain visibility |

<hfoptions id="community">

<hfoption id="Hub pipelines">

لتحميل خط أنابيب Hugging Face Hub المجتمعي، قم بتمرير معرف مستودع خط الأنابيب المجتمعي إلى وسيط `custom_pipeline` ومعرف مستودع النموذج الذي تريد تحميل أوزان خط الأنابيب والمكونات منه. على سبيل المثال، يحمّل المثال أدناه خط أنابيب وهميًا من [hf-internal-testing/diffusers-dummy-pipeline](https://huggingface.co/hf-internal-testing/diffusers-dummy-pipeline/blob/main/pipeline.py) وأوزان خط الأنابيب والمكونات من [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32):

> [!WARNING]
> من خلال تحميل خط أنابيب المجتمع من Hugging Face Hub، فأنت تثق بأن الكود الذي تقوم بتحميله آمن. تأكد من فحص الكود عبر الإنترنت قبل تحميله وتشغيله تلقائيًا!

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"google/ddpm-cifar10-32", custom_pipeline="hf-internal-testing/diffusers-dummy-pipeline", use_safetensors=True
)
```

</hfoption>

<hfoption id="GitHub pipelines">

لتحميل خط أنابيب المجتمع من GitHub، قم بتمرير معرف مستودع خط الأنابيب المجتمعي إلى وسيط `custom_pipeline` ومعرف مستودع النموذج الذي تريد تحميل أوزان خط الأنابيب والمكونات منه. يمكنك أيضًا تحميل مكونات النموذج مباشرة. يقوم المثال أدناه بتحميل خط أنابيب المجتمع [CLIP Guided Stable Diffusion](https://github.com/huggingface/diffusers/tree/main/examples/community#clip-guided-stable-diffusion) ومكونات نموذج CLIP.

```py
from diffusers import DiffusionPipeline
from transformers import CLIPImageProcessor, CLIPModel

clip_model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
custom_pipeline="clip_guided_stable_diffusion",
clip_model=clip_model,
feature_extractor=feature_extractor,
use_safetensors=True,
)
```

</hfoption>

</hfoptions>

### التحميل من ملف محلي

يمكن أيضًا تحميل خطوط أنابيب المجتمع من ملف محلي إذا قمت بتمرير مسار ملف بدلاً من ذلك. يجب أن يحتوي المسار إلى الدليل الذي تم تمريره على ملف pipeline.py الذي يحتوي على فئة خط الأنابيب.

```py
pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
custom_pipeline="./path/to/pipeline_directory/",
clip_model=clip_model,
feature_extractor=feature_extractor,
use_safetensors=True,
)
```

### التحميل من إصدار محدد

يتم تحميل خطوط أنابيب المجتمع بشكل افتراضي من أحدث إصدار مستقر من Diffusers. لتحميل خط أنابيب المجتمع من إصدار آخر، استخدم معلمة `custom_revision`.

<hfoptions id="version">

<hfoption id="main">

على سبيل المثال، لتحميل من الفرع الرئيسي:

```py
pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
custom_pipeline="clip_guided_stable_diffusion",
custom_revision="main",
clip_model=clip_model,
feature_extractor=feature_extractor,
use_safetensors=True,
)
```

</hfoption>

<hfoption id="older version">

على سبيل المثال، لتحميل من إصدار سابق من Diffusers مثل v0.25.0:

```py
pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
custom_pipeline="clip_guided_stable_diffusion",
custom_revision="v0.25.0",
clip_model=clip_model,
feature_extractor=feature_extractor,
use_safetensors=True,
)
```

</hfoption>

</hfoptions>

### التحميل باستخدام from_pipe

يمكن أيضًا تحميل خطوط أنابيب المجتمع باستخدام طريقة [`~DiffusionPipeline.from_pipe`] التي تتيح لك تحميل وإعادة استخدام خطوط أنابيب متعددة دون أي زيادة في استخدام الذاكرة (تعرف على المزيد في دليل [إعادة استخدام خط الأنابيب](./loading#reuse-a-pipeline]). يتم تحديد متطلبات الذاكرة بواسطة أكبر خط أنابيب فردي تم تحميله.

على سبيل المثال، دعنا نحمل خط أنابيب المجتمع الذي يدعم [المطالبات الطويلة مع الترجيح](https://github.com/huggingface/diffusers/tree/main/examples/community#long-prompt-weighting-stable-diffusion) من خط أنابيب Stable Diffusion.

```py
import torch
from diffusers import DiffusionPipeline

pipe_sd = DiffusionPipeline.from_pretrained("emilianJR/CyberRealistic_V3", torch_dtype=torch.float16)
pipe_sd.to("cuda")
# تحميل خط أنابيب الترجيح المطول
pipe_lpw = DiffusionPipeline.from_pipe(
pipe_sd,
custom_pipeline="lpw_stable_diffusion",
).to("cuda")

prompt = "cat, hiding in the leaves, ((rain)), zazie rainyday, beautiful eyes, macro shot, colorful details, natural lighting, amazing composition, subsurface scattering, amazing textures, filmic, soft light, ultra-detailed eyes, intricate details, detailed texture, light source contrast, dramatic shadows, cinematic light, depth of field, film grain, noise, dark background, hyperrealistic dslr film still, dim volumetric cinematic lighting"
neg_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"
generator = torch.Generator(device="cpu").manual_seed(20)
out_lpw = pipe_lpw(
prompt,
negative_prompt=neg_prompt,
width=512,
height=512,
max_embeddings_multiples=3,
num_inference_steps=50,
generator=generator,
).images[0]
out_lpw
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/from_pipe_lpw.png" />
<figcaption class="mt-2 text-center text-sm text-gray-500">Stable Diffusion with long prompt weighting</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/from_pipe_non_lpw.png" />
<figcaption class="mt-Multiplier text-center text-sm text-gray-500">Stable Diffusion</figcaption>
</div>
</div>
## أمثلة على خطوط أنابيب المجتمع

تعد خطوط أنابيب المجتمع طريقة ممتعة ومبتكرة لتوسيع قدرات خط الأنابيب الأصلي بميزات جديدة وفريدة. يمكنك العثور على جميع خطوط أنابيب المجتمع في مجلد [diffusers/examples/community](https://github.com/huggingface/diffusers/tree/main/examples/community) مع أمثلة على الاستدلال والتدريب حول كيفية استخدامها.

يُظهر هذا القسم بعضًا من خطوط أنابيب المجتمع، ويُؤمل أن يلهمك ذلك لإنشاء خط أنابيب خاص بك (لا تتردد في فتح PR لخط أنابيب المجتمع الخاص بك والتواصل معنا لإجراء مراجعة)!

> [!TIP]
> طريقة [`~DiffusionPipeline.from_pipe`] مفيدة بشكل خاص لتحميل خطوط أنابيب المجتمع لأن العديد منها لا يحتوي على أوزان مُدربة مسبقًا ويضيف ميزة أعلى خط أنابيب موجود مثل Stable Diffusion أو Stable Diffusion XL. يمكنك معرفة المزيد عن طريقة [`~DiffusionPipeline.from_pipe`] في قسم [التحميل باستخدام from_pipe](custom_pipeline_overview#load-with-from_pipe).

<hfoptions id="community">
<hfoption id="Marigold">

[Marigold](https://marigoldmonodepth.github.io/) هو خط أنابيب انتشار تقدير العمق الذي يستخدم المعرفة المرئية الموجودة والمتأصلة في نماذج الانتشار. فهو يأخذ صورة دخل ويقوم بإزالة الضوضاء وفك تشفيرها إلى خريطة عمق. يعمل Marigold بشكل جيد حتى على الصور التي لم يرها من قبل.

```py
import torch
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
"prs-eth/marigold-lcm-v1-0",
custom_pipeline="marigold_depth_estimation",
torch_dtype=torch.float16,
variant="fp16",
)

pipeline.to("cuda")
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/community-marigold.png")
output = pipeline(
image,
denoising_steps=4,
ensemble_size=5,
processing_res=768,
match_input_res=True,
batch_size=0,
seed=33,
color_map="Spectral",
show_progress_bar=True,
)
depth_colored: Image.Image = output.depth_colored
depth_colored.save("./depth_colored.png")
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/community-marigold.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/marigold-depth.png"/>
<figcaption class="mt-ătălia-center text-sm text-gray-500">صورة العمق الملونة</figcaption>
</div>
</div>

</hfoption>

<hfoption id="HD-Painter">

[HD-Painter](https://hf.co/papers/2312.14091) هو خط أنابيب إكمال عالي الدقة. فهو يقدم طبقة *Prompt-Aware Introverted Attention (PAIntA)* لتحسين محاذاة الفلتر مع المنطقة التي سيتم إكمالها، و*Reweighting Attention Score Guidance (RASG)* لإبقاء المحفزات أكثر اتساقًا مع الفلتر وداخل نطاقها المدرب لتوليد صور واقعية.

```py
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image

pipeline = DiffusionPipeline.from_pretrained(
"Lykon/dreamshaper-8-inpainting",
custom_pipeline="hd_painter"
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter.jpg")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter-mask.png")
prompt = "football"
image = pipeline(prompt, init_image, mask_image, use_rasg=True, use_painta=True, generator=torch.manual_seed(0)).images[0]
image
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter.jpg"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hd-painter-output.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>

</hfoptions>

## مكونات المجتمع

تسمح مكونات المجتمع للمستخدمين ببناء خطوط أنابيب قد تحتوي على مكونات مخصصة غير موجودة في برنامج Diffusers. إذا كان خط الأنابيب الخاص بك يحتوي على مكونات مخصصة لا يدعمها Diffusers بالفعل، فيجب عليك توفير تطبيقاتها كنماذج Python. يمكن أن تكون هذه المكونات المخصصة عبارة عن VAE أو UNet أو جدول زمني. في معظم الحالات، يتم استيراد مشفر النص من مكتبة المحولات. يمكن أيضًا تخصيص كود خط الأنابيب نفسه.

يوضح هذا القسم كيفية استخدام المستخدمين لمكونات المجتمع لبناء خط أنابيب المجتمع.

ستستخدم نقطة تفتيش خط أنابيب [showlab/show-1-base](https://huggingface.co/showlab/show-1-base) كمثال.

1. استيراد وتحميل مشفر النص من المحولات:

```python
from transformers import T5Tokenizer, T5EncoderModel

pipe_id = "showlab/show-1-base"
tokenizer = T5Tokenizer.from_pretrained(pipe_id, subfolder="tokenizer")
text_encoder = T5EncoderModel.from_pretrained(pipe_id, subfolder="text_encoder")
```

2. تحميل جدول زمني:

```python
from diffusers import DPMSolverMultistepScheduler

scheduler = DPMSolverMultistepScheduler.from_pretrained(pipe_id, subfolder="scheduler")
```

3. تحميل معالج الصور:

```python
from transformers import CLIPFeatureExtractor

feature_extractor = CLIPFeatureExtractor.from_pretrained(pipe_id, subfolder="feature_extractor")
```

<Tip warning={true}>

في الخطوتين 4 و 5، يجب أن يتطابق التنفيذ المخصص لـ [UNet](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py) و [pipeline](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) مع التنسيق الموضح في ملفاتهم لكي يعمل هذا المثال.

</Tip>

4. الآن، ستقوم بتحميل [UNet مخصص](https://github.com/showlab/Show-1/blob/main/showone/models/unet_3d_condition.py)، والذي في هذا المثال، تم تنفيذه بالفعل في [showone_unet_3d_condition.py](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py) لراحتك. ستلاحظ أن اسم فئة `UNet3DConditionModel` قد تغير إلى `ShowOneUNet3DConditionModel` لأن [`UNet3DConditionModel`] موجود بالفعل في Diffusers. يجب وضع أي مكونات مطلوبة لفئة `ShowOneUNet3DConditionModel` في showone_unet_3d_condition.py.

بمجرد الانتهاء من ذلك، يمكنك تهيئة UNet:

```python
from showone_unet_3d_condition import ShowOneUNet3DConditionModel

unet = ShowOneUNet3DConditionModel.from_pretrained(pipe_id, subfolder="unet")
```

5. أخيرًا، ستقوم بتحميل كود خط الأنابيب المخصص. بالنسبة لهذا المثال، تم إنشاؤه بالفعل من أجلك في [pipeline_t2v_base_pixel.py](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/pipeline_t2v_base_pixel.py). يحتوي هذا البرنامج النصي على فئة `TextToVideoIFPipeline` مخصصة لتوليد مقاطع فيديو من النص. تمامًا مثل UNet المخصص، يجب وضع أي رمز مطلوب لخط الأنابيب المخصص للعمل في pipeline_t2v_base_pixel.py.

بمجرد وضع كل شيء في مكانه، يمكنك تهيئة `TextToVideoIFPipeline` باستخدام `ShowOneUNet3DConditionModel`:

```python
from pipeline_t2v_base_pixel import TextToVideoIFPipeline
import torch

pipeline = TextToVideoIFPipeline(
unet=unet,
text_encoder=text_encoder,
tokenizer=tokenizer,
scheduler=scheduler,
feature_extractor=feature_extractor
)
pipeline = pipeline.to(device="cuda")
pipeline.torch_dtype = torch.float16
```

قم بإرسال خط الأنابيب إلى Hub لمشاركته مع المجتمع!

```python
pipeline.push_to_hub("custom-t2v-pipeline")
```

بعد النشر الناجح لخط الأنابيب، يلزم إجراء بعض التغييرات:

1. تغيير سمة `_class_name` في [model_index.json](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/model_index.json#L2) إلى `"pipeline_t2v_base_pixel"` و `"TextToVideoIFPipeline"`.

2. تحميل `showone_unet_3d_condition.py` إلى المجلد الفرعي [unet](https://huggingface.co/sayakpaul/show-1-base-with-code/blob/main/unet/showone_unet_3d_condition.py).

3. تحميل `pipeline_t2v_base_pixel.py` إلى [مستودع](https://huggingface.co/sayakpaul/show-1-base-with-code/tree/main) خط الأنابيب.

لتشغيل الاستدلال، أضف وسيط `trust_remote_code` أثناء تهيئة خط الأنابيب للتعامل مع كل "السحر" وراء الكواليس.

> [!WARNING]
> كإجراء احترازي إضافي مع `trust_remote_code=True`، نشجعك بشدة على تمرير هاش الالتزام إلى معلمة `revision` في [`~DiffusionPipeline.from_pretrained`] للتأكد من أن الكود لم يتم تحديثه بأسطر جديدة من التعليمات البرمجية الضارة (ما لم تثق تمامًا في مالكي النموذج).

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
"<change-username>/<change-id>", trust_remote_code=True, torch_dtype=torch.float16
).to("cuda")

prompt = "hello"

# ترميز النص
prompt_embeds, negative_embeds = pipeline.encode_prompt(prompt)

# إنشاء مفتاح الإطارات (8x64x40، 2fps)
video_frames = pipeline(
prompt_embeds=prompt_embeds،
negative_prompt_embeds=negative_embeds،
num_frames=8،
height=40،
width=64،
num_inference_steps=2،
guidance_scale=9.0،
output_type="pt"
).frames
```

كمرجع إضافي، راجع بنية مستودع [stabilityai/japanese-stable-diffusion-xl](https://huggingface.co/stabilityai/japanese-stable-diffusion-xl/) الذي يستخدم أيضًا ميزة `trust_remote_code`.

```python
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/japanese-stable-diffusion-xl"، trust_remote_code=True
)
pipeline.to("cuda")
```