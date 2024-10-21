# Transformer2DModel

نموذج Transformer للبيانات الشبيهة بالصور من [CompVis](https://huggingface.co/CompVis) والمبني على [Vision Transformer](https://huggingface.co/papers/2010.11929) الذي قدمه Dosovitskiy et al. يقبل [`Transformer2DModel`] المدخلات المنفصلة (فئات من embeddings الناقلات) أو المستمرة (embeddings الفعلية).

عندما يكون الإدخال **مستمرًا**:

1. قم بمشروع الإدخال وإعادة تشكيله إلى `(batch_size, sequence_length, feature_dimension)`.
2. تطبيق كتل المحول بالطريقة القياسية.
3. إعادة تشكيل إلى صورة.

عندما يكون الإدخال **منفصل**:

<Tip>

يفترض أن تكون إحدى فئات الإدخال هي البكسل المخفي. لا تحتوي الفئات المتوقعة للصورة غير المضطربة على تنبؤ للبكسل المخفي لأن الصورة غير المضطربة لا يمكن أن تكون مقنعة.

</Tip>

1. تحويل الإدخال (فئات البكسل الكامنة) إلى embeddings وتطبيق embeddings الموضعية.
2. تطبيق كتل المحول بالطريقة القياسية.
3. التنبؤ بفئات الصورة غير المضطربة.

## Transformer2DModel

[[autodoc]] Transformer2DModel

## Transformer2DModelOutput

[[autodoc]] models.transformers.transformer_2d.Transformer2DModelOutput