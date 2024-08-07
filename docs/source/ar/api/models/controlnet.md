# ControlNetModel

تم تقديم نموذج ControlNet في [إضافة تحكم شرطي إلى نماذج النص إلى الصورة](https://huggingface.co/papers/2302.05543) بواسطة ليفمين جانج، وأني راو، ومينيش أجروالا. يوفر درجة أكبر من التحكم في عملية توليد الصور من النص عن طريق شرط النموذج على مدخلات إضافية مثل خرائط الحواف، وخرائط العمق، وخرائط التجزئة، ونقاط رئيسية لكشف الوضع.

الملخص من الورقة هو:

*نحن نقدم ControlNet، وهو تصميم شبكة عصبية لإضافة عناصر تحكم شرطية مكانية إلى نماذج النشر النصي إلى الصورة كبيرة الحجم والمدربة مسبقًا. يقوم ControlNet بتأمين نماذج النشر الكبيرة الجاهزة للإنتاج، ويعيد استخدام طبقات الترميز المتعمقة والمتينة الخاصة بها والمدربة مسبقًا باستخدام مليارات الصور كعمود فقري قوي لتعلم مجموعة متنوعة من عناصر التحكم الشرطية. يتم توصيل الهندسة العصبية بـ "التقنيات الصفرية" (طبقات التقليب ذات القيمة الأولية الصفرية) التي تنمو تدريجيًا المعلمات من الصفر وتضمن عدم وجود ضوضاء ضارة يمكن أن تؤثر على الضبط الدقيق. نختبر عناصر تحكم شرطية مختلفة، مثل الحواف والعمق والتجزئة ووضع الإنسان، وما إلى ذلك، مع Stable Diffusion، باستخدام شرط واحد أو عدة شروط، مع أو بدون مطالبات. نحن نثبت أن تدريب ControlNets قوي مع مجموعات بيانات صغيرة (<50 ألف) وكبيرة (> 1 مليون). تُظهر النتائج المستفيضة أن ControlNet قد يسهل تطبيقات أوسع نطاقًا للتحكم في نماذج انتشار الصور.*

## التحميل من التنسيق الأصلي

بشكل افتراضي، يجب تحميل [`ControlNetModel`] باستخدام [`~ModelMixin.from_pretrained`]، ولكنه أيضًا يمكن تحميله
من التنسيق الأصلي باستخدام [`FromOriginalControlnetMixin.from_single_file`] كما يلي:

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

url = "https://huggingface.co/lllyasviel/ControlNet-v1-1/blob/main/control_v11p_sd15_canny.pth"  # يمكن أن يكون مسارًا محليًا أيضًا
controlnet = ControlNetModel.from_single_file(url)

url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned.safetensors"  # يمكن أن يكون مسارًا محليًا أيضًا
pipe = StableDiffusionControlNetPipeline.from_single_file(url, controlnet=controlnet)
```

## ControlNetModel

[[autodoc]] ControlNetModel

## ControlNetOutput

[[autodoc]] models.controlnet.ControlNetOutput

## FlaxControlNetModel

[[autodoc]] FlaxControlNetModel

## FlaxControlNetOutput

[[autodoc]] models.controlnet_flax.FlaxControlNetOutput