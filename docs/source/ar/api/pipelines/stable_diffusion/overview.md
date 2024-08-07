# Stable Diffusion pipelines

Stable Diffusion هو نموذج توليد الصور من النص باستخدام النشر في الفضاء الكامن، أنشأه باحثون ومهندسون من [CompVis](https://github.com/CompVis) و [Stability AI](https://stability.ai/) و [LAION](https://laion.ai/). يطبق النشر الكامن عملية النشر على فضاء كامن أقل أبعاد لخفض تعقيد الذاكرة والحوسبة. اقترح هذا النوع المحدد من نموذج النشر في [High-Resolution Image Synthesis with Latent Diffusion Models](https://huggingface.co/papers/2112.10752) بواسطة Robin Rombach و Andreas Blattmann و Dominik Lorenz و Patrick Esser و Björn Ommer.

تم تدريب Stable Diffusion على صور 512x512 من مجموعة فرعية من مجموعة بيانات LAION-5B. يستخدم هذا النموذج مشفر نص CLIP ViT-L/14 المجمد لتهيئة النموذج على موجهات النص. وباستخدام شبكة UNet بحجم 860 مليون ومشفر نص بحجم 123 مليون، يعد النموذج خفيفًا نسبيًا ويمكن تشغيله على وحدات معالجة الرسومات (GPU) الاستهلاكية.

للحصول على مزيد من التفاصيل حول كيفية عمل Stable Diffusion وكيف يختلف عن نموذج النشر الكامن الأساسي، اطلع على إعلان [Stability AI](https://stability.ai/blog/stable-diffusion-announcement) وتدوينتنا [الخاصة](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work) لمزيد من التفاصيل التقنية.

يمكنك العثور على كود المصدر الأصلي لـ Stable Diffusion v1.0 في [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) و Stable Diffusion v2.0 في [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion) بالإضافة إلى النصوص البرمجية الأصلية الخاصة بهم لمختلف المهام. يمكن العثور على نقاط تفتيش رسمية إضافية لمختلف إصدارات Stable Diffusion ومهامها على منظمات [CompVis](https://huggingface.co/CompVis) و [Runway](https://huggingface.co/runwayml) و [Stability AI](https://huggingface.co/stabilityai) Hub. استكشف هذه المنظمات للعثور على أفضل نقطة تفتيش لحالتك الاستخدام!

يوضح الجدول أدناه ملخصًا لأنابيب Stable Diffusion المتاحة والمهام التي تدعمها وبيانًا توضيحيًا تفاعليًا:

<div class="flex justify-center">
<div class="rounded-xl border border-gray-200">
<table class="min-w-full divide-y-2 divide-gray-200 bg-white text-sm">
<thead>
<tr>
<th class="px-4 py-2 font-medium text-gray-900 text-left">
Pipeline
</th>
<th class="px-4 py-2 font-medium text-gray-900 text-left">
Supported tasks
</th>
<th class="px-4 py-2 font-medium text-gray-900 text-left">
🤗 Space
</th>
</tr>
</thead>
<tbody class="divide-y divide-gray-200">
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./text2img">StableDiffusion</a>
</td>
<td class="px-4 py-2 text-gray-700">text-to-image</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./img2img">StableDiffusionImg2Img</a>
</td>
<td class="px-4 py-2 text-gray-700">image-to-image</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/huggingface/diffuse-the-rest"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./inpaint">StableDiffusionInpaint</a>
</td>
<td class="px-4 py-2 text-gray-700">inpainting</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/runwayml/stable-diffusion-inpainting"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./depth2img">StableDiffusionDepth2Img</a>
</td>
<td class="px-4 py-2 text-gray-700">depth-to-image</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/radames/stable-diffusion-depth2img"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./image_variation">StableDiffusionImageVariation</a>
</td>
<td class="px-4 py-2 text-gray-700">image variation</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/lambdalabs/stable-diffusion-image-variations"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./stable_diffusion_safe">StableDiffusionPipelineSafe</a>
</td>
<td class="px-4 py-2 text-gray-700">filtered text-to-image</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/AIML-TUDA/unsafe-vs-safe-stable-diffusion"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./stable_diffusion_2">StableDiffusion2</a>
</td>
<td class="px-4 py-2 text-gray-700">text-to-image, inpainting, depth-to-image, super-resolution</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./stable_diffusion_xl">StableDiffusionXL</a>
</td>
<td class="px-4 py-2 text-gray-700">text-to-image, image-to-image</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/RamAnanth1/stable-diffusion-xl"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./latent_upscale">StableDiffusionLatentUpscale</a>
</td>
<td class="px-4 py-2 text-gray-700">super-resolution</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/huggingface-projects/stable-diffusion-latent-upscaler"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./upscale">StableDiffusionUpscale</a>
</td>
<td class="px-4 py-2 text-gray-700">super-resolution</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./ldm3d_diffusion">StableDiffusionLDM3D</a>
</td>
<td class="px-4 py-2 text-gray-700">text-to-rgb, text-to-depth, text-to-pano</td>
<td class="px-4 py-2"><a href="https://huggingface.co/spaces/r23/ldm3d-space"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"/></a>
</td>
</tr>
<tr>
<td class="px-4 py-2 text-gray-700">
<a href="./ldm3d_diffusion">StableDiffusionUpscaleLDM3D</a>
</td>
<td class="px-4 py-2 text-gray-700">ldm3d super-resolution</td>
</tr>
</tbody>
</table>
</div>
</div>

## نصائح

للمساعدة في الاستفادة القصوى من أنابيب Stable Diffusion، إليك بعض النصائح لتحسين الأداء وسهولة الاستخدام. تنطبق هذه النصائح على جميع أنابيب Stable Diffusion.

### استكشف المقايضة بين السرعة والجودة

يستخدم [`StableDiffusionPipeline`] بشكل افتراضي [`PNDMScheduler`]، ولكن يوفر 🤗 Diffusers العديد من الجداول الزمنية الأخرى (بعضها أسرع أو ينتج مخرجات بجودة أفضل) والتي تتوافق معها. على سبيل المثال، إذا كنت تريد استخدام [`EulerDiscreteScheduler`] بدلاً من الافتراضي:

```py
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)

# or
euler_scheduler = EulerDiscreteScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=euler_scheduler)
```

### إعادة استخدام مكونات الأنبوب لتوفير الذاكرة

لتوفير الذاكرة واستخدام نفس المكونات عبر أنابيب متعددة، استخدم طريقة `.components` لتجنب تحميل الأوزان في ذاكرة الوصول العشوائي (RAM) أكثر من مرة.

```py
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)

text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
img2img = StableDiffusionImg2ImgPipeline(**text2img.components)
inpaint = StableDiffusionInpaintPipeline(**text2img.components)

# الآن يمكنك استخدام text2img(...)، img2img(...)، inpaint(...) تمامًا مثل طرق الاستدعاء الخاصة بكل خط أنابيب على حدة
```

### إنشاء عروض توضيحية ويب باستخدام `gradio`

يتم دعم أنابيب Stable Diffusion تلقائيًا في [Gradio](https://github.com/gradio-app/gradio/)، وهي مكتبة تجعل إنشاء تطبيقات التعلم الآلي الجميلة وسهلة الاستخدام على الويب أمرًا سهلاً. أولاً، تأكد من تثبيت Gradio:

```sh
pip install -U gradio
```

ثم قم بإنشاء عرض توضيحي ويب حول أي خط أنابيب يعتمد على Stable Diffusion. على سبيل المثال، يمكنك إنشاء خط أنابيب لتوليد الصور في سطر واحد من التعليمات البرمجية باستخدام وظيفة [`Interface.from_pipeline`](https://www.gradio.app/docs/interface#interface-from-pipeline) من Gradio:

```py
from diffusers import StableDiffusionPipeline
import gradio as gr

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

gr.Interface.from_pipeline(pipe).launch()
```

والذي يفتح واجهة بديهية للسحب والإفلات في متصفحك:

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gradio-panda.png)

وبالمثل، يمكنك إنشاء عرض توضيحي لخط أنابيب الصورة إلى الصورة باستخدام:

```py
from diffusers import StableDiffusionImg2ImgPipeline
import gradio as gr


pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

gr.Interface.from_pipeline(pipe).launch()
```

افتراضيًا، يعمل العرض التوضيحي على الويب على خادم محلي. إذا كنت تريد مشاركتها مع الآخرين، فيمكنك إنشاء رابط عام مؤقت عن طريق تعيين `share=True` في `launch()`. أو، يمكنك استضافة العرض التوضيحي الخاص بك على [Hugging Face Spaces](https://huggingface.co/spaces) للحصول على رابط دائم.