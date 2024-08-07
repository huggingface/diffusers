# Stable Diffusion XL

اقترح Dustin Podell و Zion English و Kyle Lacey و Andreas Blattmann و Tim Dockhorn و Jonas Müller و Joe Penna و Robin Rombach في [SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://huggingface.co/papers/2307.01952) نظام Stable Diffusion XL (SDXL).

مقدمة الورقة البحثية هي:

*نحن نقدم SDXL، وهو نموذج انتشار خفي لتوليد الصور من النص. مقارنة بالإصدارات السابقة من Stable Diffusion، يستفيد SDXL من شبكة UNet أكبر بثلاث مرات: ترجع زيادة معلمات النموذج بشكل أساسي إلى كتل اهتمام إضافية وسياق اهتمام متقاطع أكبر حيث يستخدم SDXL مشفر نص ثانوي. نقوم بتصميم مخططات تكييف متعددة وتدريب SDXL على نسب متعددة الجوانب. كما نقدم نموذج تحسين يتم استخدامه لتحسين الدقة البصرية للعينات التي تم إنشاؤها بواسطة SDXL باستخدام تقنية صورة إلى صورة لاحقة للمعالجة. نثبت أن SDXL يحقق أداءً محسنًا بشكل كبير مقارنة بالإصدارات السابقة من Stable Diffusion ويحقق نتائج تنافسية مع مولدات الصور المتقدمة الأخرى.*

## نصائح

- من المعروف أن استخدام SDXL مع جدول DPM++ لأقل من 50 خطوة ينتج عنه [التشوهات المرئية](https://github.com/huggingface/diffusers/issues/5433) لأن المحلل يصبح غير مستقر عدديًا. لإصلاح هذه المشكلة، راجع هذا [PR](https://github.com/huggingface/diffusers/pull/5541) الذي يوصي لمحللات ODE/SDE بما يلي:
- قم بتعيين `use_karras_sigmas=True` أو `lu_lambdas=True` لتحسين جودة الصورة
- قم بتعيين `euler_at_final=True` إذا كنت تستخدم محلاًل بخطوات ثابتة (DPM++2M أو DPM++2M SDE)
- تعمل معظم نقاط تفتيش SDXL بشكل أفضل مع حجم صورة 1024x1024. يتم أيضًا دعم أحجام الصور 768x768 و 512x512، ولكن النتائج ليست جيدة. لا يوصى بأي شيء أقل من 512x512 ولن يكون كذلك بالنسبة لنقاط التفتيش الافتراضية مثل [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0).
- يمكن لـ SDXL تمرير موجه نص مختلف لكل من مشفرات النص التي تم تدريبه عليها. يمكننا حتى تمرير أجزاء مختلفة من نفس الموجه إلى مشفرات النص.
- يمكن تحسين صور إخراج SDXL من خلال استخدام نموذج تحسين في إعداد الصورة إلى الصورة.
- يقدم SDXL `negative_original_size` و `negative_crops_coords_top_left` و `negative_target_size` للتكييف السلبي للنموذج على دقة الصورة ومعلمات القص.

<Tip>

لمعرفة كيفية استخدام SDXL لمختلف المهام، وكيفية تحسين الأداء، وأمثلة الاستخدام الأخرى، راجع دليل [Stable Diffusion XL](../../../using-diffusers/sdxl).

تحقق من [Stability AI](https://huggingface.co/stabilityai) منظمة Hub للحصول على نقاط تفتيش النماذج الأساسية والرسمية!

</Tip>

## StableDiffusionXLPipeline

[[autodoc]] StableDiffusionXLPipeline

- all
- __call__

## StableDiffusionXLImg2ImgPipeline

[[autodoc]] StableDiffusionXLImg2ImgPipeline

- all
- __call__

## StableDiffusionXLInpaintPipeline

[[autodoc]] StableDiffusionXLInpaintPipeline

- all
- __call__