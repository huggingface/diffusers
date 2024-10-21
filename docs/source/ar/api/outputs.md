# Outputs

جميع مخرجات النموذج هي فئات فرعية من [`~utils.BaseOutput`]، وهياكل بيانات تحتوي على جميع المعلومات التي يرجعها النموذج. يمكن أيضًا استخدام المخرجات على أنها مجموعات أو قواميس.

على سبيل المثال:

```python
from diffusers import DDIMPipeline

pipeline = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
outputs = pipeline()
```

كائن `outputs` هو [`~pipelines.ImagePipelineOutput`]، مما يعني أنه يحتوي على سمة صورة.

يمكنك الوصول إلى كل سمة كما تفعل عادةً أو باستخدام بحث الكلمات الرئيسية، وإذا لم يرجع النموذج هذه السمة، فستحصل على `None`:

```python
outputs.images
outputs["images"]
```

عند اعتبار كائن `outputs` على أنه مجموعة، فإنه يأخذ في الاعتبار فقط السمات التي لا تحتوي على قيم `None`.

على سبيل المثال، عند استرداد صورة عن طريق الفهرسة، فإنه يعيد المجموعة `(outputs.images)`:

```python
outputs[:1]
```

<Tip>
للتحقق من مخرجات خط أنابيب أو نموذج محدد، راجع وثائق API المقابلة.
</Tip>

## BaseOutput

[[autodoc]] utils.BaseOutput

- to_tuple

## ImagePipelineOutput

[[autodoc]] pipelines.ImagePipelineOutput

## FlaxImagePipelineOutput

[[autodoc]] pipelines.pipeline_flax_utils.FlaxImagePipelineOutput

## AudioPipelineOutput

[[autodoc]] pipelines.AudioPipelineOutput

## ImageTextPipelineOutput

[[autodoc]] ImageTextPipelineOutput
