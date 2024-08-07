# Text-to-image

قام باحثون ومهندسون من [CompVis] (https://github.com/CompVis) و [Stability AI] (https://stability.ai/) و [Runway] (https://github.com/runwayml) و [LAION] (https://laion.ai/) بإنشاء نموذج Stable Diffusion. [Pipeline 'StableDiffusionPipeline'] قادر على توليد صور واقعية تبدو كصور حقيقية بناءً على أي إدخال نصي. تم تدريبه على صور بحجم 512x512 من مجموعة فرعية من مجموعة بيانات LAION-5B. يستخدم هذا النموذج مشفر نص CLIP ViT-L/14 مجمد لتهيئة النموذج بناءً على مطالبات النص. وباستخدام شبكة UNet بحجم 860 ميجابايت ومشفر نص بحجم 123 ميجابايت، يعد النموذج خفيفًا نسبيًا ويمكن تشغيله على وحدات معالجة الرسومات (GPUs) الخاصة بالمستهلك. يعد Latent diffusion البحث الذي تم بناء Stable Diffusion عليه. اقترحه روبن رومباخ، وأندرياس بلاتمان، ودومينيك لورنز، وباتريك إسر، وبيورن أومير في [High-Resolution Image Synthesis with Latent Diffusion Models] (https://huggingface.co/papers/2112.10752).

ملخص الورقة هو:

*من خلال تحليل عملية تكوين الصورة إلى تطبيق متسلسل لتشفير فك تشفير الضوضاء، تحقق نماذج الانتشار (DMs) نتائج تركيب رائدة في مجال الصور وما بعدها. بالإضافة إلى ذلك، تسمح صياغتها بآلية توجيه للتحكم في عملية إنشاء الصور دون إعادة التدريب. ومع ذلك، نظرًا لأن هذه النماذج تعمل عادةً مباشرة في مساحة البكسل، فإن تحسين نماذج DM القوية غالبًا ما يستهلك مئات أيام وحدة معالجة الرسومات (GPU)، كما أن الاستدلال مكلف بسبب التقييمات التسلسلية. لتمكين تدريب DM على موارد حوسبة محدودة مع الحفاظ على جودتها ومرونتها، فإننا نطبقها في مساحة الكمون لتشفير فك تشفير قوي مسبق التدريب. على عكس العمل السابق، يسمح تدريب نماذج الانتشار على مثل هذا التمثيل، للمرة الأولى، بالوصول إلى نقطة مثالية تقريبًا بين تقليل التعقيد والحفاظ على التفاصيل، مما يعزز بشكل كبير الدقة المرئية. من خلال تقديم طبقات الاهتمام المتبادل في بنية نموذج DM، نحول نماذج الانتشار إلى مولدات قوية ومرنة لإدخالات التهيئة العامة مثل النص أو صناديق الحدود، ويصبح التوليف عالي الدقة ممكنًا بطريقة التلافيف. تحقق نماذج الانتشار الكامنة (LDMs) الخاصة بنا حالة جديدة من الفن لإكمال الصور وأداءً تنافسيًا للغاية في مهام مختلفة، بما في ذلك توليد الصور غير المشروطة، وتوليف المشاهد الدلالية، والتحسين الفائق، مع تقليل كبير في المتطلبات الحسابية مقارنة بنماذج DM المستندة إلى البكسل. الكود متاح على https://github.com/CompVis/latent-diffusion.*

<Tip>
تأكد من الاطلاع على قسم [نصائح Stable Diffusion] (overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!

إذا كنت مهتمًا باستخدام إحدى نقاط التفتيش الرسمية لمهمة ما، فاستكشف منظمات [CompVis] (https://huggingface.co/CompVis) و [Runway] (https://huggingface.co/runwayml) و [Stability AI] (https://huggingface.co/stabilityai) Hub!
</Tip>

## StableDiffusionPipeline

[[autodoc]] StableDiffusionPipeline
- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention
- enable_vae_tiling
- disable_vae_tiling
- load_textual_inversion
- from_single_file
- load_lora_weights
- save_lora_weights

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput

## FlaxStableDiffusionPipeline

[[autodoc]] FlaxStableDiffusionPipeline
- all
- __call__

## FlaxStableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput