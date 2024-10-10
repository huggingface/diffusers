# الإعداد

يتم حفظ جميع المعلمات التي يتم تمريرها إلى أساليب `__init__` الخاصة بها في ملف تكوين JSON.

<Tip>

لاستخدام النماذج الخاصة أو [المحمية](https://huggingface.co/docs/hub/models-gated#gated-models)، قم بتسجيل الدخول باستخدام `huggingface-cli login`.

</Tip>

## ConfigMixin

[[autodoc]] ConfigMixin

- load_config
- from_config
- save_config
- to_json_file
- to_json_string