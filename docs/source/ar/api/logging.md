# Logging

🤗 Diffusers لديه نظام تسجيل مركزي لإدارة سهولة مكتبة اللغة. يتم تعيين مستوى التفصيل الافتراضي على "تحذير".

لتغيير مستوى التفصيل، استخدم أحد برامج الإعداد المباشر. على سبيل المثال، لتغيير مستوى التفصيل إلى مستوى "معلومات".

```python
import diffusers

diffusers.logging.set_verbosity_info()
```

يمكنك أيضًا استخدام متغير البيئة `DIFFUSERS_VERBOSITY` لتجاوز مستوى التفصيل الافتراضي. يمكنك تعيينه

إلى واحد مما يلي: `debug`، `info`، `warning`، `error`، `critical`. على سبيل المثال:

```bash
DIFFUSERS_VERBOSITY=error ./myprogram.py
```

بالإضافة إلى ذلك، يمكن تعطيل بعض `التحذيرات` عن طريق تعيين متغير البيئة

`DIFFUSERS_NO_ADVISORY_WARNINGS` إلى قيمة صحيحة، مثل `1`. يؤدي هذا إلى تعطيل أي تحذير يسجله

[`logger.warning_advice`]. على سبيل المثال:

```bash
DIFFUSERS_NO_ADVISORY_WARNINGS=1 ./myprogram.py
```

فيما يلي مثال على كيفية استخدام نفس مسجل البيانات مثل المكتبة في الوحدة النمطية أو البرنامج النصي الخاص بك:

```python
from diffusers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("diffusers")
logger.info("INFO")
logger.warning("WARN")
```

جميع طرق وحدة التسجيل موثقة أدناه. الطرق الرئيسية هي

[`logging.get_verbosity`] للحصول على مستوى التفصيل الحالي في مسجل البيانات و

[`logging.set_verbosity`] لتعيين مستوى التفاصيل إلى مستوى اختيارك.

بترتيب من الأقل تفصيلاً إلى الأكثر تفصيلاً:

| الأسلوب | القيمة الصحيحة | الوصف |
|------------:|------------:|------------:|
| `diffusers.logging.CRITICAL` أو `diffusers.logging.FATAL` | 50 | قم بالإبلاغ عن الأخطاء الأكثر خطورة فقط |
| `diffusers.logging.ERROR` | 40 | قم بالإبلاغ عن الأخطاء فقط |
| `diffusers.logging.WARNING` أو `diffusers.logging.WARN` | 30 | قم بالإبلاغ عن الأخطاء والتحذيرات (افتراضي) |
| `diffusers.logging.INFO` | 20 | قم بالإبلاغ عن الأخطاء والتحذيرات والمعلومات الأساسية فقط |
| `diffusers.logging.DEBUG` | 10 | الإبلاغ عن جميع المعلومات |

بشكل افتراضي، يتم عرض مؤشرات تقدم `tqdm` أثناء تنزيل النموذج. يتم استخدام [`logging.disable_progress_bar`] و [`logging.enable_progress_bar`] لتمكين أو تعطيل هذا السلوك.

## برامج الإعداد الأساسية

[[autodoc]] utils.logging.set_verbosity_error

[[autodoc]] utils.logging.set_verbosity_warning

[[autodoc]] utils.logging.set_verbosity_info

[[autodoc]] utils.logging.set_verbosity_debug

## الوظائف الأخرى

[[autodoc]] utils.logging.get_verbosity

[[autodoc]] utils.logging.set_verbosity

[[autodoc]] utils.logging.get_logger

[[autodoc]] utils.logging.enable_default_handler

[[autodoc]] utils.logging.disable_default_handler

[[autodoc]] utils.logging.enable_explicit_format

[[autodoc]] utils.logging.reset_format

[[autodoc]] utils.logging.enable_progress_bar

[[autodoc]] utils.logging.disable_progress_bar