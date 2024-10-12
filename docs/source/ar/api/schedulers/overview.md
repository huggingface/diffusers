# Schedulers

يوفر 🤗 Diffusers العديد من دالات الجدولة لعملية الانتشار. تتلقى دالة الجدولة ناتج نموذج (العينة التي تعمل عملية الانتشار على تكرارها) وخطوة زمنية لإرجاع عينة منخفضة التشويش. وتعد الخطوة الزمنية مهمة لأنها تحدد مكان عملية الانتشار؛ حيث يتم إنشاء البيانات عن طريق التكرار للأمام *n* من الخطوات الزمنية، ويحدث الاستنتاج عن طريق الانتشار للخلف عبر الخطوات الزمنية. بناءً على الخطوة الزمنية، قد تكون الجدولة *منفصلة*، وفي هذه الحالة تكون الخطوة الزمنية `int` أو *مستمرة*، وفي هذه الحالة تكون الخطوة الزمنية `float`.

اعتمادًا على السياق، يحدد الجدول كيفية إضافة التشويش إلى صورة بشكل تكراري أو كيفية تحديث عينة بناءً على ناتج النموذج:

- أثناء *التدريب*، يضيف الجدول التشويش (هناك خوارزميات مختلفة لكيفية إضافة التشويش) إلى عينة لتدريب نموذج الانتشار.
- أثناء *الاستنتاج*، يحدد الجدول كيفية تحديث عينة بناءً على ناتج نموذج مُدرب مسبقًا.

تم تنفيذ العديد من الجداول من مكتبة [k-diffusion](https://github.com/crowsonkb/k-diffusion) بواسطة [Katherine Crowson](https://github.com/crowsonkb/)، وهي مستخدمة أيضًا على نطاق واسع في A1111. ولمساعدتك في مطابقة الجداول من k-diffusion وA1111 إلى الجداول في 🤗 Diffusers، راجع الجدول أدناه:

| A1111/k-diffusion    | 🤗 Diffusers                         | Usage                                                                                                         |
|---------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------|
| DPM++ 2M            | [`DPMSolverMultistepScheduler`]     |                                                                                                               |
| DPM++ 2M Karras     | [`DPMSolverMultistepScheduler`]     | init with `use_karras_sigmas=True`                                                                            |
| DPM++ 2M SDE        | [`DPMSolverMultistepScheduler`]     | init with `algorithm_type="sde-dpmsolver++"`                                                                  |
| DPM++ 2M SDE Karras | [`DPMSolverMultistepScheduler`]     | init with `use_karras_sigmas=True` and `algorithm_type="sde-dpmsolver++"`                                     |
| DPM++ 2S a          | N/A                                 | very similar to  `DPMSolverSinglestepScheduler`                         |
| DPM++ 2S a Karras   | N/A                                 | very similar to  `DPMSolverSinglestepScheduler(use_karras_sigmas=True, ...)` |
| DPM++ SDE           | [`DPMSolverSinglestepScheduler`]    |                                                                                                               |
| DPM++ SDE Karras    | [`DPMSolverSinglestMultiplier]]`    | init with `use_karras_sigmas=True`                                                                            |
| DPM2                | [`KDPM2DiscreteScheduler`]          |                                                                                                               |
| DPM2 Karras         | [`KDPM2DiscreteScheduler`]          | init with `use_karras_sigmas=True`                                                                            |
| DPM2 a              | [`KDPM2AncestralDiscreteScheduler`] |                                                                                                               |
| DPM2 a Karras       | [`KDPM2AncestralDiscreteScheduler`] | init with `use_karras_sigmas=True`                                                                            |
| DPM adaptive        | N/A                                 |                                                                                                               |
| DPM fast            | N/A                                 |                                                                                                               |
| Euler               | [`EulerDiscreteScheduler`]          |                                                                                                               |
| Euler a             | [`EulerAncestralDiscreteScheduler`] |                                                                                                               |
| Heun                | [`HeunDiscreteScheduler`]           |                                                                                                               |
| LMS                 | [`LMSDiscreteScheduler`]            |                                                                                                               |
| LMS Karras          | [`LMSDiscreteScheduler`]            | init with `use_karras_sigmas=True`                                                                            |
| N/A                 | [`DEISMultistepScheduler`]          |                                                                                                               |
| N/A                 | [`UniPCMultistepScheduler`]         |                                                                                                               |

تم بناء جميع الجداول من فئة الأساس [`SchedulerMixin`] التي تنفذ المرافق منخفضة المستوى المشتركة بين جميع الجداول.

## SchedulerMixin

[[autodoc]] SchedulerMixin

## SchedulerOutput

[[autodoc]] schedulers.scheduling_utils.SchedulerOutput

## KarrasDiffusionSchedulers

[`KarrasDiffusionSchedulers`] هي تعميم واسع لجداول 🤗 Diffusers. وتتميز الجداول في هذه الفئة، على مستوى عالٍ، باستراتيجية أخذ عينات التشويش، ونوع الشبكة والمقياس، واستراتيجية التدريب، وكيفية وزن الخسارة.

تندرج الجداول المختلفة في هذه الفئة، اعتمادًا على نوع معادلة تفاضلية عادية (ODE)، في التصنيف الموضح أعلاه وتوفر تجريدًا جيدًا لتصميم الجداول الرئيسية المنفذة في 🤗 Diffusers. يمكن العثور على الجداول في هذه الفئة [هنا](https://github.com/huggingface/diffusers/blob/a69754bb879ed55b9b6dc9dd0b3cf4fa4124c765/src/diffusers/schedulers/scheduling_utils.py#L32).

## PushToHubMixin

[[autodoc]] utils.PushToHubMixin