# LoRA

LoRA هي طريقة تدريب سريعة وخفيفة الوزن تقوم بإدراج وتدريب عدد أقل بكثير من المعلمات بدلاً من جميع معلمات النموذج. ينتج عن ذلك ملف أصغر (~100 ميجابايت) ويجعل من السهل تدريب النموذج بسرعة لتعلم مفهوم جديد. عادةً ما يتم تحميل أوزان LoRA في UNet أو encoder النصي أو كليهما. هناك فئتان لتحميل أوزان LoRA:

- [`LoraLoaderMixin`] توفر وظائف لتحميل وإلغاء تحميل، دمج وفصل، تمكين وتعطيل، والمزيد من الوظائف لإدارة أوزان LoRA. يمكن استخدام هذه الفئة مع أي نموذج.

- [`StableDiffusionXLLoraLoaderMixin`] هي نسخة [Stable Diffusion (SDXL)](../../api/pipelines/stable_diffusion/stable_diffusion_xl) من فئة [`LoraLoaderMixin`] لتحميل وحفظ أوزان LoRA. لا يمكن استخدامها إلا مع نموذج SDXL.

<Tip>

لمعرفة المزيد حول كيفية تحميل أوزان LoRA، راجع دليل التحميل [LoRA](../../using-diffusers/loading_adapters#lora).

</Tip>

## LoraLoaderMixin

[[autodoc]] loaders.lora.LoraLoaderMixin

## StableDiffusionXLLoraLoaderMixin

[[autodoc]] loaders.lora.StableDiffusionXLLoraLoaderMixin