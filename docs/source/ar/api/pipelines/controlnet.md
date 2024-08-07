# ControlNet

تم تقديم ControlNet في ورقة [Adding Conditional Control to Text-to-Image Diffusion Models](https://huggingface.co/papers/2302.05543) بواسطة Lvmin Zhang وAnyi Rao وManeesh Agrawala.

مع نموذج ControlNet، يمكنك توفير صورة تحكم إضافية لتشكيل وتحكم عملية التوليد في Stable Diffusion. على سبيل المثال، إذا قدمت خريطة عمق، فإن نموذج ControlNet يقوم بتوليد صورة تحافظ على المعلومات المكانية من خريطة العمق. إنها طريقة أكثر مرونة ودقة للتحكم في عملية توليد الصور.

الملخص من الورقة هو:

*نقدم ControlNet، وهو تصميم شبكة عصبية لإضافة عناصر تحكم في التشكيل المكاني إلى النماذج الكبيرة المُدربة مسبقًا للنشر النصي-للصور. تقوم ControlNet بتأمين نماذج الانتشار الكبيرة الجاهزة للإنتاج، وتعيد استخدام طبقات الترميز المتعمقة والمتينة المُدربة مسبقًا باستخدام مليارات الصور كعمود فقري قوي لتعلم مجموعة متنوعة من عناصر التحكم الشرطية. يتم توصيل البنية العصبية بـ "التقنيات الصفرية" (طبقات التقنية الصفرية) التي تنمو المعلمات تدريجيًا من الصفر وتضمن عدم وجود ضوضاء ضارة يمكن أن تؤثر على الضبط الدقيق. نختبر عناصر تحكم شرطية مختلفة، مثل الحواف والعمق والتجزئة ووضع الإنسان، وما إلى ذلك، مع Stable Diffusion، باستخدام شرط واحد أو أكثر، مع أو بدون موجهات. نُظهر أن تدريب ControlNets قوي مع مجموعات بيانات صغيرة (<50 ألف) وكبيرة (>1 مليون). تُظهر النتائج المستفيضة أن ControlNet قد تسهل تطبيقات أوسع للتحكم في نماذج انتشار الصور.*

تمت المساهمة بهذا النموذج من قبل [takuma104](https://huggingface.co/takuma104). ❤️

يمكن العثور على كود المصدر الأصلي في [lllyasviel/ControlNet](https://github.com/lllyasviel/ControlNet)، ويمكنك العثور على نقاط تفتيش ControlNet الرسمية على ملف تعريف [lllyasviel](https://huggingface.co/lllyasviel) Hub.

<Tip>

تأكد من الاطلاع على دليل Schedulers [guide](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة المخطط والنوعية، وقسم [reuse components across pipelines](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.

</Tip>

## StableDiffusionControlNetPipeline

[[autodoc]] StableDiffusionControlNetPipeline

- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention
- load_textual_inversion

## StableDiffusionControlNetImg2ImgPipeline

[[autodoc]] StableDiffusionControlNetImg2ImgPipeline

- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention
- load_textual_inversion

## StableDiffusionControlNetInpaintPipeline

[[autodoc]] StableDiffusionControlNetInpaintPipeline

- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_vae_slicing
- disable_vae_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention
- load_textual_inversion

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput

## FlaxStableDiffusionControlNetPipeline

[[autodoc]] FlaxStableDiffusionControlNetPipeline

- all
- __call__

## FlaxStableDiffusionControlNetPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput