# ملفات النموذج وتخطيطاته

تُخزن نماذج الانتشار في أنواع ملفات مختلفة وتنظم في تخطيطات مختلفة. ويخزن برنامج "Diffusers" أوزان النموذج كملفات "safetensors" في تخطيط "Diffusers-multifolder"، كما يدعم تحميل الملفات (مثل ملفات "safetensors" و "ckpt") من تخطيط "single-file" الذي يستخدم عادة في نظام الانتشار.

يتمتع كل تخطيط بمزايا واستخدامات خاصة به، وسيوضح هذا الدليل كيفية تحميل الملفات والتخطيطات المختلفة، وكيفية تحويلها.

## الملفات

عادة ما يتم حفظ أوزان نموذج "PyTorch" باستخدام أداة "pickle" من "Python" كملفات "ckpt" أو "bin". ومع ذلك، فإن "pickle" غير آمن وقد تحتوي الملفات المؤقتة على تعليمات برمجية ضارة يمكن تنفيذها. وهذا الضعف يمثل مصدر قلق خطير بالنظر إلى شعبية مشاركة النماذج. ولمعالجة هذه المشكلة الأمنية، تم تطوير مكتبة "Safetensors" كبديل آمن لـ "pickle"، والذي يحفظ النماذج كملفات "safetensors".

### "safetensors"

> [!TIP]
> تعرف أكثر على قرارات التصميم ولماذا يفضل استخدام ملفات "safetensor" لحفظ وتحميل أوزان النموذج في منشور المدونة "Safetensors audited as really safe and becoming the default".

"Safetensors" هو تنسيق ملف آمن وسريع لتخزين وتحميل المصفوفات بشكل آمن. ويقيد "Safetensors" حجم الرأس للحد من أنواع معينة من الهجمات، ويدعم التحميل البطيء (مفيد للإعدادات الموزعة)، ويتميز بسرعة تحميل عامة أسرع.

تأكد من تثبيت مكتبة "Safetensors".

```py
!pip install safetensors
```

ويخزن "Safetensors" الأوزان في ملف "safetensors". ويحمل "Diffusers" ملفات "safetensors" بشكل افتراضي إذا كانت متاحة وكان قد تم تثبيت مكتبة "Safetensors". وهناك طريقتان يمكن من خلالهما تنظيم ملفات "safetensors":

1. تخطيط "Diffusers-multifolder": قد يكون هناك العديد من ملفات "safetensors" المنفصلة، واحد لكل مكون من مكونات الأنابيب (مشفّر النص، UNet، VAE)، منظمة في مجلدات فرعية (تفقد مستودع "runwayml/stable-diffusion-v1-5" كمثال)
2. تخطيط "single-file": قد يتم حفظ جميع أوزان النموذج في ملف واحد (تفقد مستودع "WarriorMama777/OrangeMixs" كمثال)

<hfoptions id="safetensors">
<hfoption id="multifolder">

استخدم طريقة `~DiffusionPipeline.from_pretrained` لتحميل نموذج بملفات "safetensors" مخزنة في مجلدات متعددة.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
use_safetensors=True
)
```

</hfoption>
<hfoption id="single file">

استخدم طريقة `~loaders.FromSingleFileMixin.from_single_file` لتحميل نموذج بكل الأوزان المخزنة في ملف "safetensors" واحد.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
"https://huggingface.co/WarriorMama777/OrangeMixs/blob/main/Models/AbyssOrangeMix/AbyssOrangeMix.safetensors"
)
```

</hfoption>
</hfoptions>

#### ملفات "LoRA"

"LoRA" هو محول خفيف الوزن سريع وسهل التدريب، مما يجعله شائعًا بشكل خاص لتوليد الصور بطريقة أو نمط معين. وعادة ما يتم تخزين هذه المحولات في ملف "safetensors"، وهي شائعة على نطاق واسع على منصات مشاركة النماذج مثل "civitai".

ويتم تحميل "LoRAs" في نموذج أساسي باستخدام طريقة `~loaders.LoraLoaderMixin.load_lora_weights`.

```py
from diffusers import StableDiffusionXLPipeline
import torch

# نموذج أساسي
pipeline = StableDiffusionXLPipeline.from_pretrained(
"Lykon/dreamshaper-xl-1-0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

# تحميل أوزان "LoRA"
!wget https://civitai.com/api/download/models/168776 -O blueprintify.safetensors

# تحميل أوزان "LoRA"
pipeline.load_lora_weights(".", weight_name="blueprintify.safetensors")
prompt = "bl3uprint, a highly detailed blueprint of the empire state building, explaining how to build all parts, many txt, blueprint grid backdrop"
negative_prompt = "lowres, cropped, worst quality, low quality, normal quality, artifacts, signature, watermark, username, blurry, more than one bridge, bad architecture"

image = pipeline(
prompt=prompt,
negative_prompt=negative_prompt,
generator=torch.manual_seed(0),
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/blueprint-lora.png"/>
</div>

### "ckpt"

> [!WARNING]
> قد تكون الملفات المؤقتة غير آمنة لأنها يمكن أن تتعرض للاستغلال لتنفيذ تعليمات برمجية ضارة. ويوصى باستخدام ملفات "safetensors" بدلاً من ذلك حيثما أمكن، أو تحويل الأوزان إلى ملفات "safetensors".

تستخدم وظيفة "torch.save" من "PyTorch" أداة "pickle" من "Python" لتهيئة النماذج وحفظها. ويتم حفظ هذه الملفات كملف "ckpt" وتحتوي على أوزان النموذج بالكامل.

استخدم طريقة `~loaders.FromSingleFileMixin.from_single_file` لتحميل ملف "ckpt" مباشرة.

```py
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_single_file(
"https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
```

## تخطيط التخزين

هناك طريقتان لتنظيم ملفات النموذج، إما في تخطيط "Diffusers-multifolder" أو في تخطيط "single-file". ويكون تخطيط "Diffusers-multifolder" هو التخطيط الافتراضي، ويتم تخزين كل ملف مكون (مشفّر النص، UNet، VAE) في مجلد فرعي منفصل. ويدعم "Diffusers" أيضًا تحميل النماذج من تخطيط "single-file" حيث يتم تجميع جميع المكونات معًا.

### "Diffusers-multifolder"

تخطيط "Diffusers-multifolder" هو تخطيط التخزين الافتراضي لـ "Diffusers". ويتم تخزين أوزان كل مكون (مشفّر النص، UNet، VAE) في مجلد فرعي منفصل. ويمكن تخزين الأوزان كملفات "safetensors" أو "ckpt".

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multifolder-layout.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">تخطيط "multifolder"</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/multifolder-unet.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">مجلد فرعي "UNet"</figcaption>
</div>
</div>

لتحميل التخطيط "Diffusers-multifolder"، استخدم طريقة `~DiffusionPipeline.from_pretrained`.

```py
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True,
).to("cuda")
```

تشمل فوائد استخدام تخطيط "Diffusers-multifolder" ما يلي:

1. أسرع لتحميل كل ملف مكون بشكل فردي أو بالتوازي.
2. تقليل استخدام الذاكرة لأنك لا تحمل سوى المكونات التي تحتاجها. على سبيل المثال، تمتلك النماذج مثل "SDXL Turbo" و "SDXL Lightning"، و "Hyper-SD" نفس المكونات باستثناء "UNet". ويمكنك إعادة استخدام مكوناتها المشتركة باستخدام طريقة `~DiffusionPipeline.from_pipe` دون استهلاك أي ذاكرة إضافية (الق نظرة على دليل "إعادة استخدام الأنابيب"). وكل ما عليك فعله هو تحميل "UNet". بهذه الطريقة، لن تحتاج إلى تنزيل المكونات الزائدة واستخدام المزيد من الذاكرة دون داع.

```py
import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler

# تحميل نموذج واحد
sdxl_pipeline = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True,
).to("cuda")

# استبدل "UNet" بنموذج آخر
unet = UNet2DConditionModel.from_pretrained(
"stabilityai/sdxl-turbo",
subfolder="unet",
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True
)
# إعادة استخدام جميع المكونات نفسها في نموذج جديد باستثناء "UNet"
turbo_pipeline = StableDiffusionXLPipeline.from_pipe(
sdxl_pipeline, unet=unet,
).to("cuda")
turbo_pipeline.scheduler = EulerDiscreteScheduler.from_config(
turbo_pipeline.scheduler.config,
timestep+spacing="trailing"
)
image = turbo_pipeline(
"an astronaut riding a unicorn on mars",
num_inference_steps=1,
guidance_scale=0.0,
).images[0]
image
```

3. تقليل متطلبات التخزين لأنه إذا كان أحد المكونات، مثل "VAE" من "SDXL"، مشتركًا عبر عدة نماذج، فلن تحتاج إلى تنزيله وتخزينه عدة مرات. وبالنسبة لعشرة نماذج من "SDXL"، يمكن أن يوفر ذلك حوالي 3.5 جيجابايت من مساحة التخزين. وتكون وفورات التخزين أكبر بالنسبة للنماذج الأحدث مثل "PixArt Sigma"، حيث يبلغ حجم "مشفّر النص" وحده حوالي 19 جيجابايت!
4. المرونة لاستبدال أحد مكونات النموذج بإصدار أحدث أو أفضل.

```py
from diffusers import DiffusionPipeline, AutoencoderKL

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
vae=vae,
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True,
).to("cuda")
```

5. المزيد من الرؤية والمعلومات حول مكونات النموذج، والتي يتم تخزينها في ملف "config.json" في كل مجلد فرعي للمكونات.

### "Single-file"

يتم في تخطيط "single-file" تخزين جميع أوزان النموذج في ملف واحد. ويتم الاحتفاظ بأوزان جميع مكونات النموذج (مشفّر النص، UNet، VAE) معًا بدلاً من تخزينها بشكل منفصل في مجلدات فرعية. ويمكن أن يكون هذا الملف إما "safetensors" أو "ckpt".

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/single-file-layout.png"/>
</div>

لتحميل تخطيط "single-file"، استخدم طريقة `~loaders.FromSingleFileMixin.from_single_file`.

```py
import torch
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True,
).to("cuda")
```

تشمل فوائد استخدام تخطيط "single-file" ما يلي:

1. التوافق السهل مع واجهات الانتشار مثل "ComfyUI" أو "Automatic1111" التي تستخدم عادة تخطيط "single-file".
2. من الأسهل إدارة (تنزيل ومشاركة) ملف واحد.
## تحويل التخطيط والملفات

يوفر Diffusers العديد من النصوص البرمجية والطرق لتحويل تخطيطات التخزين وتنسيقات الملفات لتمكين دعم أوسع عبر نظام بيئي diffusion.

الق نظرة على مجموعة [diffusers/scripts](https://github.com/huggingface/diffusers/tree/main/scripts) للعثور على نص برمجي يناسب احتياجاتك في التحويل.

> [!TIP]
> النصوص البرمجية التي تحتوي على "to_diffusers" ملحقة في النهاية تعني أنها تحول نموذجًا إلى تخطيط Diffusers-multifolder. لكل نص برمجي مجموعة محددة خاصة به من الحجج لتكوين التحويل، لذا تأكد من التحقق من الحجج المتاحة!

على سبيل المثال، لتحويل نموذج Stable Diffusion XL المخزن في تخطيط Diffusers-multifolder إلى تخطيط ملف واحد، قم بتشغيل نص [convert_diffusers_to_original_sdxl.py](https://github.com/huggingface/diffusers/blob/main/scripts/convert_diffusers_to_original_sdxl.py) البرمجي. قم بتوفير المسار إلى النموذج الذي تريد تحويله، ومسار لحفظ النموذج المحول إليه. يمكنك أيضًا تحديد ما إذا كنت تريد حفظ النموذج كملف safetensors وما إذا كنت تريد حفظ النموذج بنصف الدقة.

```bash
python convert_diffusers_to_original_sdxl.py --model_path path/to/model/to/convert --checkpoint_path path/to/save/model/to --use_safetensors
```

يمكنك أيضًا حفظ نموذج إلى تخطيط Diffusers-multifolder باستخدام طريقة [`~DiffusionPipeline.save_pretrained`]. يقوم هذا بإنشاء دليل لك إذا لم يكن موجودًا بالفعل، كما يقوم أيضًا بحفظ الملفات كملف safetensors بشكل افتراضي.

```py
from diffusers import StableDiffusionXLPipeline

pipeline = StableDiffusionXLPipeline.from_single_file(
"https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors",
)
pipeline.save_pretrained()
```

أخيرًا، هناك أيضًا مساحات، مثل [SD To Diffusers](https://hf.co/spaces/diffusers/sd-to-diffusers) و [SD-XL To Diffusers](https://hf.co/spaces/diffusers/sdxl-to-diffusers)، والتي توفر واجهة أكثر ملاءمة للمستخدم لتحويل النماذج إلى تخطيط Diffusers-multifolder. هذا هو أسهل وأكثر الخيارات ملاءمة لتحويل التخطيطات، وسيفتح PR في مستودع نموذجك بالملفات المحولة. ومع ذلك، هذا الخيار ليس موثوقًا مثل تشغيل نص برمجي، وقد تفشل المساحة بالنسبة للنماذج الأكثر تعقيدًا.

## استخدام تخطيط ملف واحد

الآن بعد أن تعرفت على الاختلافات بين تخطيط Diffusers-multifolder وتخطيط ملف واحد، يُظهر لك هذا القسم كيفية تحميل مكونات النموذج وخط الأنابيب، وتخصيص خيارات التكوين للتحميل، وتحميل الملفات المحلية باستخدام طريقة [`~loaders.FromSingleFileMixin.from_single_file`].

### تحميل خط أنابيب أو نموذج

مرر مسار ملف خط الأنابيب أو النموذج إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`] لتحميله.

<hfoptions id="pipeline-model">
<hfoption id="pipeline">

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path)
```

</hfoption>
<hfoption id="model">

```py
from diffusers import StableCascadeUNet

ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
model = StableCascadeUNet.from_single_file(ckpt_path)
```

</hfoption>
</hfoptions>

قم بتخصيص المكونات في خط الأنابيب عن طريق تمريرها مباشرةً إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`]. على سبيل المثال، يمكنك استخدام جدول زمني مختلف في خط الأنابيب.

```py
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
scheduler = DDIMScheduler()
pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler)
```

أو يمكنك استخدام نموذج ControlNet في خط الأنابيب.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipeline = StableDiffusionControlNetPipeline.from_single_file(ckpt_path, controlnet=controlnet)
```

### تخصيص خيارات التكوين

تحتوي النماذج على ملف تكوين يحدد سماتها مثل عدد الإدخالات في UNet. خيارات تكوين خط الأنابيب متاحة في فئة خط الأنابيب. على سبيل المثال، إذا نظرت إلى فئة [`StableDiffusionXLInstructPix2PixPipeline`]`]، فهناك خيار لقياس الصورة latents مع معلمة `is_cosxl_edit`.

يمكن العثور على ملفات التكوين هذه في مستودع نموذج Hub أو موقع آخر الذي نشأ منه ملف التكوين (على سبيل المثال، مستودع GitHub أو محليًا على جهازك).

<hfoptions id="config-file">
<hfoption id="Hub configuration file">

> [!TIP]
> تقوم طريقة [`~loaders.FromSingleFileMixin.from_single_file`] تلقائيًا بتعيين نقطة التفتيش إلى مستودع نموذج مناسب، ولكن هناك حالات يكون من المفيد فيها استخدام معلمة "config". على سبيل المثال، إذا كانت مكونات النموذج في نقطة التفتيش مختلفة عن نقطة التفتيش الأصلية أو إذا لم يكن لنقطة التفتيش البيانات الوصفية اللازمة لتحديد التكوين الذي سيتم استخدامه لخط الأنابيب بشكل صحيح.

تقوم طريقة [`~loaders.FromSingleFileMixin.from_single_file`] تلقائيًا بتحديد التكوين الذي سيتم استخدامه من ملف التكوين في مستودع النموذج. يمكنك أيضًا تحديد التكوين الذي سيتم استخدامه بشكل صريح من خلال توفير معرف المستودع إلى معلمة "config".

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/segmind/SSD-1B/blob/main/SSD-1B.safetensors"
repo_id = "segmind/SSD-1B"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, config=repo_id)
```

يحمّل النموذج ملف تكوين [UNet](https://huggingface.co/segmind/SSD-1B/blob/main/unet/config.json) و [VAE](https://huggingface.co/segmind/SSD-1B/blob/main/vae/config.json) و [encoder النصي](https://huggingface.co/segmind/SSD-1B/blob/main/text_encoder/config.json) من مجلدات فرعية خاصة بهم في المستودع.

</hfoption>
<hfoption id="original configuration file">

يمكن لطريقة [`~loaders.FromSingleFileMixin.from_single_file`] أيضًا تحميل ملف تكوين الأصلي لخط أنابيب مخزن في مكان آخر. قم بتمرير مسار محلي أو عنوان URL لملف تكوين الأصلي إلى معلمة "original_config".

```py
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
original_config = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"

pipeline = StableDiffusionXLPipeline.from_single_file(ckpt_path, original_config=original_config)
```

> [!TIP]
> تحاول Diffusers استنتاج مكونات خط الأنابيب بناءً على تواقيع الأنواع لفئة خط الأنابيب عندما تستخدم `original_config` مع `local_files_only=True`، بدلاً من استرداد ملفات التكوين من مستودع النموذج على Hub. يمنع هذا التغييرات التراجعية في التعليمات البرمجية التي لا يمكنها الاتصال بالإنترنت لاسترداد ملفات التكوين اللازمة.
>
> هذا ليس موثوقًا مثل توفير مسار إلى مستودع نموذج محلي مع معلمة "config"، وقد يؤدي إلى أخطاء أثناء تكوين خط الأنابيب. لتجنب الأخطاء، قم بتشغيل خط الأنابيب باستخدام `local_files_only=False` مرة واحدة لتحميل ملفات تكوين خط الأنابيب المناسبة إلى ذاكرة التخزين المؤقت المحلية.

</hfoption>
</hfoptions>

في حين أن ملفات التكوين تحدد الافتراضيات الافتراضية لخط الأنابيب أو النماذج، يمكنك تجاوزها من خلال توفير المعلمات مباشرةً إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`]. يمكن تكوين أي معلمة مدعومة بواسطة فئة النموذج أو خط الأنابيب بهذه الطريقة.

<hfoptions id="override">
<hfoption id="pipeline">

على سبيل المثال، لقياس الصورة latents في [`StableDiffusionXLInstructPix2PixPipeline`]، قم بتمرير معلمة `is_cosxl_edit`.

```python
from diffusers import StableDiffusionXLInstructPix2PixPipeline

ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
pipeline = StableDiffusionXLInstructPix2PixPipeline.from_single_file(ckpt_path, config="diffusers/sdxl-instructpix2pix-768"، is_cosxl_edit=True)
```

</hfoption>
<hfoption id="model">

على سبيل المثال، لتصعيد أبعاد الاهتمام في [`UNet2DConditionModel`]، قم بتمرير معلمة `upcast_attention`.

```python
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, upcast_attention=True)
```

</hfoption>
</hfoptions>

### الملفات المحلية

في Diffusers>=v0.28.0، تحاول طريقة [`~loaders.FromSingleFileMixin.from_single_file`] تكوين خط أنابيب أو نموذج من خلال استنتاج نوع النموذج من المفاتيح في ملف نقطة التفتيش. يتم استخدام نوع النموذج المستنتج لتحديد مستودع النموذج المناسب على Hugging Face Hub لتكوين النموذج أو خط الأنابيب.

على سبيل المثال، سيستخدم أي ملف نقطة تفتيش أحادي يعتمد على نموذج Stable Diffusion XL الأساسي مستودع النموذج [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) لتكوين خط الأنابيب.

ولكن إذا كنت تعمل في بيئة ذات إمكانية وصول مقيدة إلى الإنترنت، فيجب عليك تنزيل ملفات التكوين باستخدام وظيفة [`~huggingface_hub.snapshot_download`]`]، ونقطة تفتيش النموذج باستخدام وظيفة [`~huggingface_hub.hf_hub_download`]. بشكل افتراضي، يتم تنزيل هذه الملفات إلى دليل ذاكرة التخزين المؤقت لـ Hugging Face Hub [cache directory](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache)، ولكن يمكنك تحديد دليل مفضل لتنزيل الملفات إليه باستخدام معلمة "local_dir".

قم بتمرير مسارات التكوين ونقطة التفتيش إلى طريقة [`~loaders.FromSingleFileMixin.from_single_file`] لتحميلها محليًا.

<hfoptions id="local">
<hfoption id="Hub cache directory">

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
repo_id="segmind/SSD-1B"،
filename="SSD-1B.safetensors"
)

my_local_config_path = snapshot_download(
repo_id="segmind/SSD-1B"،
allowed_patterns=["*.json"، "**/*.json"، "*.txt"، "**/*.txt"]
)

pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```

</hfoption>
<hfoption id="specific local directory">

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
repo_id="segmind/SSD-1B"،
filename="SSD-1B.safetensors"
local_dir="my_local_checkpoints"
)

my_local_config_path = snapshot_download(
repo_id="segmind/SSD-1B"،
allowed_patterns=["*.json"، "**/*.json"، "*.txt"، "**/*.txt"]
local_dir="my_local_config"
)

pipeline = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)
```

</hfoption>
</hfoptions>

#### الملفات المحلية بدون symlink

> [!TIP]
> في huggingface_hub>=v0.23.0، لا يلزم وجود وسيط "local_dir_use_symlinks" لوظائف [`~huggingface_hub.hf_hub_download`] و [`~huggingface_hub.snapshot_download`].

تعتمد طريقة [`~loaders.FromSingleFileMixin.from_single_file`] على آلية التخزين المؤقت لـ [huggingface_hub](https://hf.co/docs/huggingface_hub/index) لاسترداد وتخزين نقاط التفتيش وملفات التكوين للنماذج وخطوط الأنابيب. إذا كنت تعمل بنظام ملفات لا يدعم إنشاء الارتباطات الرمزية، فيجب عليك أولاً تنزيل ملف نقطة التفتيش إلى دليل محلي وتعطيل الارتباطات الرمزية باستخدام معلمة "local_dir_use_symlink=False" في وظائف [`~huggingface_hub.hf_hub_download`] و [`~huggingface_hub.snapshot_download`].

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
repo_id="segmind/SSD-1B"،
filename="SSD-1B.safetensors"
local_dir="my_local_checkpoints"،
local_dir_use_symlinks=False
)
print("My local checkpoint: "، my_local_checkpoint_path)

my_local_config_path = snapshot_download(
repo_id="segmind/SSD-1B"،
allowed_patterns=["*.json"، "**/*.json"، "*.txt"، "**/*.txt"]
local_dir_use_symlinks=False،
)
print("My local config: "، my_local_config_path)

```