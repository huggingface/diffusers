# AutoPipeline

يوفر Diffusers العديد من الأنابيب للمهام الأساسية مثل إنشاء الصور ومقاطع الفيديو والصوت، وinpainting. بالإضافة إلى ذلك، هناك خطوط أنابيب متخصصة للوحدات الإضافية والميزات مثل تحسين الدقة والوضوح الفائق، والمزيد. يمكن لفئات خطوط الأنابيب المختلفة حتى استخدام نفس نقطة التفتيش لأنها تشترك في نفس النموذج مسبق التدريب! مع وجود العديد من خطوط الأنابيب المختلفة، قد يكون من المحبط معرفة فئة خط الأنابيب التي يجب استخدامها.

تم تصميم فئة [AutoPipeline](../api/pipelines/auto_pipeline) لتبسيط مجموعة متنوعة من خطوط الأنابيب في Diffusers. إنه خط أنابيب عام *يركز على المهمة أولاً* يتيح لك التركيز على مهمة ([`AutoPipelineForText2Image`]، [`AutoPipelineForImage2Image`]، و [`AutoPipelineForInpainting`]) دون الحاجة إلى معرفة فئة خط الأنابيب المحددة. يكتشف [AutoPipeline](../api/pipelines/auto_pipeline) تلقائيًا فئة خط الأنابيب الصحيحة التي يجب استخدامها.

على سبيل المثال، دعنا نستخدم نقطة تفتيش [dreamlike-art/dreamlike-photoreal-2.0](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0).

تحت الغطاء، [AutoPipeline](../api/pipelines/auto_pipeline):

1. يكتشف فئة `"stable-diffusion"` من ملف [model_index.json](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0/blob/main/model_index.json).
2. اعتمادًا على المهمة التي تهتم بها، فإنه يحمّل [`StableDiffusionPipeline`] أو [`StableDiffusionImg2ImgPipeline`] أو [`StableDiffusionInpaintPipeline`]. يمكن تمرير أي معلمة (`strength`، `num_inference_steps`، وما إلى ذلك) التي ستمررها إلى خطوط الأنابيب المحددة هذه أيضًا إلى [AutoPipeline](../api/pipelines/auto_pipeline).

<hfoptions id="autopipeline">
<hfoption id="text-to-image">

```py
from diffusers import AutoPipelineForText2Image
import torch

pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
"dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

prompt = "cinematic photo of Godzilla eating sushi with a cat in a izakaya, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(37)
image = pipe_txt2img(prompt, generator=generator).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png"/>
</div>

</hfoption>
<hfoption id="image-to-image">

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipe_img2img = AutoPipelineForImage2Image.from_pretrained(
"dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-text2img.png")

prompt = "cinematic photo of Godzilla eating burgers with a cat in a fast food restaurant, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(53)
image = pipe_img2img(prompt, image=init_image, generator=generator).images[0]
image
```

لاحظ كيف يتم استخدام نقطة تفتيش [dreamlike-art/dreamlike-photoreal-2.0](https://hf.co/dreamlike-art/dreamlike-photoreal-2.0) لكل من مهام النص إلى الصورة والصورة إلى الصورة؟ لتوفير الذاكرة وتجنب تحميل نقطة التفتيش مرتين، استخدم طريقة [`~DiffusionPipeline.from_pipe`].

```py
pipe_img2img = AutoPipelineForImage2Image.from_pipe(pipe_txt2img).to("cuda")
image = pipeline(prompt, image=init_image, generator=generator).images[0]
image
```

يمكنك معرفة المزيد عن طريقة [`~DiffusionPipeline.from_pipe`] في دليل [إعادة استخدام خط الأنابيب](../using-diffusers/loading#reuse-a-pipeline).

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png"/>
</div>

</hfoption>
<hfoption id="inpainting">

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")

init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-img2img.png")
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-mask.png")

prompt = "cinematic photo of a owl, 35mm photograph, film, professional, 4k, highly detailed"
generator = torch.Generator(device="cpu").manual_seed(38)
image = pipeline(prompt, image=init_image, mask_image=mask_image, generator=generator, strength=0.4).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/autopipeline-inpaint.png"/>
</div>

</hfoption>
</hfoptions>

## نقاط التفتيش غير المدعومة

يدعم [AutoPipeline](../api/pipelines/auto_pipeline) نقاط تفتيش [Stable Diffusion](../api/pipelines/stable_diffusion/overview) و [Stable Diffusion XL](../api/pipelines/stable_diffusion/stable_diffusion_xl) و [ControlNet](../api/pipelines/controlnet) و [Kandinsky 2.1](../api/pipelines/kandinsky.md) و [Kandinsky 2.2](../api/pipelines/kandinsky_v22) و [DeepFloyd IF](../api/pipelines/deepfloyd_if).

إذا حاولت تحميل نقطة تفتيش غير مدعومة، فستحصل على خطأ.

```py
from diffusers import AutoPipelineForImage2Image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained(
"openai/shap-e-img2img", torch_dtype=torch.float16, use_safetensors=True
)
"ValueError: AutoPipeline can't find a pipeline linked to ShapEImg2ImgPipeline for None"
```