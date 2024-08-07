# GLIGEN (Grounded Language-to-Image Generation) 

قام باحثون ومهندسون من [جامعة ويسكونسن-ماديسون، وجامعة كولومبيا، ومايكروسوفت](https://github.com/gligen/GLIGEN) بإنشاء نموذج GLIGEN. ويمكن لأنابيب [StableDiffusionGLIGENPipeline] و [StableDiffusionGLIGENTextImagePipeline] إنشاء صور واقعية مشروطة على إدخالات التوطئة. بالإضافة إلى النص وصناديق الحدود مع [StableDiffusionGLIGENPipeline]، إذا تم توفير صور الإدخال، فيمكن لـ [StableDiffusionGLIGENTextImagePipeline] إدراج الكائنات التي يصفها النص في المنطقة التي تحددها صناديق الحدود. وإلا، فسيقوم بتوليد صورة موصوفة بواسطة التعليق/المطالبة وإدراج الكائنات التي يصفها النص في المنطقة التي تحددها صناديق الحدود. تم تدريبه على مجموعات بيانات COCO2014D و COCO2014CD، ويستخدم النموذج مشفر نص CLIP ViT-L/14 مجمد لتشفير نفسه على إدخالات التوطئة.

المستخلص من [الورقة](https://huggingface.co/papers/2301.07093) هو:

*حققت نماذج النص إلى الصورة واسعة النطاق المستندة إلى الانتشار تقدمًا مذهلاً. ومع ذلك، فإن الوضع الراهن هو استخدام إدخال النص وحده، والذي يمكن أن يعوق إمكانية التحكم. في هذا العمل، نقترح GLIGEN، وهو نهج جديد يعتمد على النص إلى الصورة المستندة إلى النص، والذي يعتمد على وظائف نماذج النص إلى الصورة المستندة إلى النصوص الموجودة مسبقًا ويمتد بها من خلال تمكينها أيضًا من أن تكون مشروطة بإدخالات التوطئة. للحفاظ على المعرفة المفاهيمية الواسعة للنموذج المُدرب مسبقًا، نقوم بتجميد جميع أوزانه وإدخال معلومات التوطئة في طبقات قابلة للتدريب جديدة عبر آلية بوابية. يحقق نموذجنا إنشاء نص موصوف ومفتوح المجال مع إدخالات شروط التعليق وصندوق الحدود، وتعميم قدرة التوطئة جيدًا على التكوينات والمفاهيم المكانية الجديدة. ويتفوق أداء GLIGEN على الصفر على مجموعات بيانات COCO و LVIS على الخط القاعدي المُشرف على تخطيط الصورة بهامش كبير.*

<Tip>

تأكد من الاطلاع على قسم Stable Diffusion [Tips](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/overview#tips) لمعرفة كيفية استكشاف التوازن بين سرعة المخطط والنوعية وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!

إذا كنت تريد استخدام إحدى نقاط التفتيش الرسمية لمهمة، فاستكشف [gligen](https://huggingface.co/gligen) منظمات Hub!

</Tip>

تمت المساهمة بـ [StableDiffusionGLIGENPipeline] بواسطة [Nikhil Gajendrakumar](https://github.com/nikhil-masterful) وتمت المساهمة بـ [StableDiffusionGLIGENTextImagePipeline] بواسطة [Nguyễn Công Tú Anh](https://github.com/tuanh123789).

## StableDiffusionGLIGENPipeline

[[autodoc]] StableDiffusionGLIGENPipeline

- all
- __call__
- enable_vae_slicing
- disable_vae_slicing
- enable_vae_tiling
- disable_vae_tiling
- enable_model_cpu_offload
- prepare_latents
- enable_fuser

## StableDiffusionGLIGENTextImagePipeline

[[autodoc]] StableDiffusionGLIGENTextImagePipeline

- all
- __call__
- enable_vae_slicing
- disable_vae_slicing
- enable_vae_tiling
- disable_vae_tiling
- enable_model_cpu_offload
- prepare_latents
- enable_fuser

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput