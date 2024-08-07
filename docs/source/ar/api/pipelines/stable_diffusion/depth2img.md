# من العمق إلى الصورة
يمكن لنموذج Stable Diffusion أيضًا استنتاج العمق بناءً على صورة باستخدام [MiDaS](https://github.com/isl-org/MiDaS). يسمح لك ذلك بتمرير موجه نص وصورة أولية لاشتقاق صور جديدة بالإضافة إلى `depth_map` للحفاظ على بنية الصورة.

<Tip>
تأكد من الاطلاع على قسم "النصائح" في Stable Diffusion [Tips](overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة المخطط وجودته، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!
إذا كنت مهتمًا باستخدام إحدى نقاط التفتيش الرسمية لمهمة ما، فاستكشف منظمات [CompVis](https://huggingface.co/CompVis) و [Runway](https://huggingface.co/runwayml) و [Stability AI](https://huggingface.co/stabilityai) Hub!
</Tip>

## StableDiffusionDepth2ImgPipeline
[[autodoc]] StableDiffusionDepth2ImgPipeline
- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention
- load_textual_inversion
- load_lora_weights
- save_lora_weights

## StableDiffusionPipelineOutput
[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput
