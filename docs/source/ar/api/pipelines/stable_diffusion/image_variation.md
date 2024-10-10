# اختلاف الصورة

يمكن لنموذج Stable Diffusion أيضًا إنشاء اختلافات من صورة المدخلات. يستخدم نسخة معدلة من نموذج Stable Diffusion بواسطة [Justin Pinkney](https://www.justinpinkney.com/) من [Lambda](https://lambdalabs.com/).

يمكن العثور على كود المصدر الأصلي في [LambdaLabsML/lambda-diffusers](https://github.com/LambdaLabsML/lambda-diffusers#stable-diffusion-image-variations) ويمكن العثور على نقاط تفتيش رسمية إضافية لاختلاف الصور في [lambdalabs/sd-image-variations-diffusers](https://huggingface.co/lambdalabs/sd-image-variations-diffusers).

<Tip>
تأكد من الاطلاع على قسم "نصائح Stable Diffusion" [tips](./overview#tips) لمعرفة كيفية استكشاف المقايضة بين سرعة الجدولة والجودة، وكيفية إعادة استخدام مكونات الأنابيب بكفاءة!
</Tip>

## StableDiffusionImageVariationPipeline

[[autodoc]] StableDiffusionImageVariationPipeline

- all
- __call__
- enable_attention_slicing
- disable_attention_slicing
- enable_xformers_memory_efficient_attention
- disable_xformers_memory_efficient_attention

## StableDiffusionPipelineOutput

[[autodoc]] pipelines.stable_diffusion.StableDiffusionPipelineOutput