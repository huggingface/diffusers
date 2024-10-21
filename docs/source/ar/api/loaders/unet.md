# UNet

تركز بعض طرق التدريب - مثل LoRA و Custom Diffusion - عادةً على طبقات الانتباه في UNet، ولكن يمكن أيضًا أن تستهدف هذه الطرق التدريبية طبقات غير الانتباه. بدلاً من تدريب جميع معلمات النموذج، يتم تدريب مجموعة فرعية فقط من المعلمات، وهو ما يكون أسرع وأكثر كفاءة. هذه الفئة مفيدة إذا كنت *تقوم فقط* بتحميل الأوزان في UNet. إذا كنت بحاجة إلى تحميل الأوزان في encoder النصي أو encoder النصي و UNet، فحاول استخدام دالة [`~loaders.LoraLoaderMixin.load_lora_weights`] بدلاً من ذلك.

توفر فئة [`UNet2DConditionLoadersMixin`] دالات لتحميل الأوزان وحفظها، ودمج وفصل LoRAs، وتعطيل وتمكين LoRAs، وتعيين وحذف المحولات.

<Tip>

لمعرفة المزيد حول كيفية تحميل أوزان LoRA، راجع دليل تحميل [LoRA] (../../using-diffusers/loading_adapters#lora).

</Tip>

## UNet2DConditionLoadersMixin

[[autodoc]] loaders.unet.UNet2DConditionLoadersMixin