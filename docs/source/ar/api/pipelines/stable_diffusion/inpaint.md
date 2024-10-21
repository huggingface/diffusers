# Inpainting

يمكن أيضًا تطبيق نموذج Stable Diffusion على المعالجة الفنية، مما يتيح لك تحرير أجزاء محددة من صورة ما من خلال توفير قناع وملء النص باستخدام Stable Diffusion.

## نصائح

من المستحسن استخدام هذا الأنبوب مع نقاط تفتيش تمت معايرتها خصيصًا للمعالجة الفنية، مثل [runwayml/stable-diffusion-inpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting). نقاط تفتيش Stable Diffusion الافتراضية من النص إلى الصورة، مثل [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) متوافقة أيضًا ولكن قد تكون أقل أداءً.

<Tip>

تأكد من الاطلاع على قسم "النصائح" في Stable Diffusion [Tips](overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والنوعية، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!

إذا كنت مهتمًا باستخدام إحدى نقاط التفتيش الرسمية لمهمة ما، فاستكشف منظمات [CompVis](https://huggingface.co/CompVis) و [Runway](https://huggingface.co/runwayml) و [Stability AI](https://huggingface.co/stabilityai) Hub!

</Tip>

## StableDiffusionInpaintPipeline

[[autodoc]] StableDiffusionInpaintPipeline

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

## FlaxStableDiffusionInpaintPipeline

[[autodoc]] FlaxStableDiffusionInpaintPipeline

- all
- __call__

## FlaxStableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.FlaxStableDiffusionPipelineOutput