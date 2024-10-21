لمحة عامة عن طبقات التطبيع

طبقات تطبيع مخصصة لدعم نماذج مختلفة في 🤗 Diffusers.

## AdaLayerNorm

[[autodoc]] models.normalization.AdaLayerNorm

طبقة تطبيع قابلة للتكيف مع معامل تطبيع قابل للتعلم لكل قناة.

## AdaLayerNormZero

[[autodoc]] models.normalization.AdaLayerNormZero

تعديل AdaLayerNorm الذي يستخدم معاملات تطبيع صفرية افتراضية.

## AdaLayerNormSingle

[[autodoc]] models.normalization.AdaLayerNormSingle

طبقة تطبيع AdaLayerNorm مع معامل تطبيع واحد لجميع القنوات.

## AdaGroupNorm

[[autodoc]] models.normalization.AdaGroupNorm

طبقة تطبيع تجمع بين التطبيع على مستوى القناة والتطبيع على مستوى المجموعة.