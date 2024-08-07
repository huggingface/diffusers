# محسن الصورة الكامن 

تم إنشاء نموذج Stable Diffusion latent upscaler بواسطة [Katherine Crowson](https://github.com/crowsonkb/k-diffusion) بالتعاون مع [Stability AI](https://stability.ai/). ويستخدم لتعزيز دقة الصورة الناتجة بعامل 2 (راجع هذا الدفتر [notebook](https://colab.research.google.com/drive/1o1qYJcFeywzCIdkfKJy7cTpgZTCM2EI4) لمشاهدة عرض توضيحي للتنفيذ الأصلي).

<Tip>
تأكد من الاطلاع على قسم Stable Diffusion [Tips](overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة المخطط والنوعية، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!
إذا كنت مهتمًا باستخدام إحدى نقاط التفتيش الرسمية لمهمة ما، فاستكشف منظمات [CompVis](https://huggingface.co/CompVis) و [Runway](https://huggingface.co/runwayml) و [Stability AI](https://huggingface.co/stabilityai) Hub!
</Tip>

## StableDiffusionLatentUpscalePipeline

[[autodoc]] StableDiffusionLatentUpscalePipeline

- all
- __call__
- enable_sequential_cpu_offload
- enable_attention_slicing
- disable_attention_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput