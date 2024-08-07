# ملفات فردية

تتيح طريقة [`~loaders.FromSingleFileMixin.from_single_file`] لك تحميل ما يلي:

* نموذج مخزن في ملف واحد، وهو مفيد إذا كنت تعمل مع نماذج من نظام الانتشار، مثل Automatic1111، وتعتمد عادةً على تخطيط ملف واحد لتخزين ومشاركة النماذج.

* نموذج مخزن في تخطيط التوزيع الأصلي الخاص به، وهو مفيد إذا كنت تعمل مع نماذج تمت تهيئتها باستخدام خدمات أخرى، وتريد تحميلها مباشرةً في كائنات ونماذج أنابيب Diffusers.

> [!TIP]
> اقرأ دليل [ملفات النماذج والتخطيطات](../../using-diffusers/other-formats) لمعرفة المزيد حول تخطيط Diffusers-multifolder مقابل تخطيط الملف الفردي، وكيفية تحميل النماذج المخزنة في هذه التخطيطات المختلفة.

## خطوط الأنابيب المدعومة

- [`StableDiffusionPipeline`]

- [`StableDiffusionImg2ImgPipeline`]

- [`StableDiffusionInpaintPipeline`]

- [`StableDiffusionControlNetPipeline`]

- [`StableDiffusionControlNetImg2ImgPipeline`]

- [`StableDiffusionControlNetInpaintPipeline`]

- [`StableDiffusionUpscalePipeline`]

- [`StableDiffusionXLPipeline`]

- [`StableDiffusionXLImg2ImgPipeline`]

- [`StableDiffusionXLInpaintPipeline`]

- [`StableDiffusionXLInstructPix2PixPipeline`]

- [`StableDiffusionXLControlNetPipeline`]

- [`StableDiffusionXLKDiffusionPipeline`]

- [`StableDiffusion3Pipeline`]

- [`LatentConsistencyModelPipeline`]

- [`LatentConsistencyModelImg2ImgPipeline`]

- [`StableDiffusionControlNetXSPipeline`]

- [`StableDiffusionXLControlNetXSPipeline`]

- [`LEditsPPPipelineStableDiffusion`]

- [`LEditsPPPipelineStableDiffusionXL`]

- [`PIAPipeline`]

## النماذج المدعومة

- [`UNet2DConditionModel`]

- [`StableCascadeUNet`]

- [`AutoencoderKL`]

- [`ControlNetModel`]

- [`SD3Transformer2DModel`]

## FromSingleFileMixin

[[autodoc]] loaders.single_file.FromSingleFileMixin

## FromOriginalModelMixin

[[autodoc]] loaders.single_file_model.FromOriginalModelMixin