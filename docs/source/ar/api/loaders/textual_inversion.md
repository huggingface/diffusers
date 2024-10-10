# الانعكاس النصي

الانعكاس النصي هو طريقة تدريب لشخصنة النماذج من خلال تعلم ترميزات نصية جديدة من بعض الصور النموذجية. الملف الناتج عن التدريب صغير للغاية (بضعة كيلوبايتات) ويمكن تحميل الترميزات الجديدة في المشفر النصي.

يوفر [`TextualInversionLoaderMixin`] وظيفة لتحميل ترميزات الانعكاس النصي من Diffusers و Automatic1111 إلى المشفر النصي وتحميل رمز خاص لتنشيط الترميزات.

<Tip>

لمعرفة المزيد حول كيفية تحميل ترميزات الانعكاس النصي، راجع دليل تحميل [الانعكاس النصي] (../../using-diffusers/loading_adapters#textual-inversion).

</Tip>

## TextualInversionLoaderMixin

[[autodoc]] loaders.textual_inversion.TextualInversionLoaderMixin