# إنشاء الفيديو من الصور باستخدام PIA (محرك الرسوم المتحركة للصور الشخصية)
# Image-to-Video Generation with PIA (Personalized Image Animator)
## نظرة عامة

[PIA: محرك الرسوم المتحركة للصور الشخصية عبر الوحدات الإضافية القابلة للتوصيل والتشغيل في نماذج تحويل النص إلى صورة](https://arxiv.org/abs/2312.13964) بقلم ييمينغ تشانغ، زهينينغ شينغ، يانهونغ زينغ، يوكينغ فونغ، كاي تشين

حققت التطورات الحديثة في النماذج الشخصية لتحويل النص إلى صورة (T2I) ثورة في إنشاء المحتوى، مما مكن غير الخبراء من إنشاء صور مذهلة بأنماط فريدة. وعلى الرغم من الوعد الذي أظهرته، إلا أن إضافة حركات واقعية إلى هذه الصور الشخصية من خلال النص يطرح تحديات كبيرة في الحفاظ على الأساليب المميزة وتفاصيل عالية الدقة، وتحقيق إمكانية التحكم في الحركة من خلال النص. في هذه الورقة، نقدم PIA، وهو محرك رسوم متحركة للصور الشخصية يتفوق في المحاذاة مع صور الشرط، وتحقيق إمكانية التحكم في الحركة من خلال النص، والتوافق مع نماذج T2I الشخصية المختلفة دون ضبط محدد. ولتحقيق هذه الأهداف، يعتمد PIA على نموذج T2I الأساسي مع طبقات محاذاة زمنية مدربة تدريبًا جيدًا، مما يسمح بالتحويل السلس لأي نموذج T2I شخصي إلى نموذج رسوم متحركة للصور. يتمثل أحد المكونات الرئيسية لـ PIA في تقديم وحدة الشرط، والتي تستخدم إطار الشرط والانجذاب بين الإطارات كإدخال لنقل معلومات المظهر التي يوجهها تلميح الانجذاب لتركيب الإطار الفردي في الفراغ. يخفف هذا التصميم من تحديات محاذاة الصور المتعلقة بالمظهر ويسمح بالتركيز بشكل أكبر على المحاذاة مع التوجيهات المتعلقة بالحركة.

[صفحة المشروع](https://pi-animator.github.io/)

## خطوط الأنابيب المتاحة

| الأنبوب | المهام | عرض توضيحي |
|---|---|:---:|
| [خط أنابيب PIAPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pia/pipeline_pia.py) | *إنشاء الفيديو من الصور باستخدام PIA* |

## نقاط التفتيش المتاحة

يمكن العثور على نقاط تفتيش Motion Adapter لـ PIA ضمن [منظمة OpenMMLab](https://huggingface.co/openmmlab/PIA-condition-adapter). يُقصد بنقاط التفتيش هذه العمل مع أي نموذج يعتمد على Stable Diffusion 1.5

## مثال على الاستخدام

يعمل PIA مع نقطة تفتيش MotionAdapter ونقطة تفتيش نموذج Stable Diffusion 1.5. MotionAdapter عبارة عن مجموعة من وحدات الحركة المسؤولة عن إضافة حركة متماسكة عبر إطارات الصور. يتم تطبيق هذه الوحدات بعد كتل Resnet و Attention في شبكة UNet الخاصة بنموذج Stable Diffusion. بالإضافة إلى وحدات الحركة، يستبدل PIA أيضًا طبقة التجزئة المدخلة لشبكة UNet الخاصة بنموذج SD 1.5 بطبقة تجزئة مدخلة ذات 9 قنوات.

يوضح المثال التالي كيفية استخدام PIA لإنشاء فيديو من صورة واحدة.

```python
import torch
from diffusers import (
    EulerDiscreteScheduler,
    MotionAdapter,
    PIAPipeline,
)
from diffusers.utils import export_to_gif, load_image

adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
pipe = PIAPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter, torch_dtype=torch.float16)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
)
image = image.resize((512, 512))
prompt = "cat in a field"
negative_prompt = "wrong white balance, dark, sketches,worst quality,low quality"

generator = torch.Generator("cpu").manual_seed(0)
output = pipe(image=image, prompt=prompt, generator=generator)
frames = output.frames[0]
export_to_gif(frames, "pia-animation.gif")
```

فيما يلي بعض النماذج الإخراجية:

<table>
<tr>
<td><center>
cat in a field.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pia-default-output.gif"
alt="cat in a field"
style="width: 300px;" />
</center></td>
</tr>
</table>

<Tip>

إذا كنت تخطط لاستخدام مخطط يمكنه قص العينات، فتأكد من تعطيله عن طريق تعيين `clip_sample=False` في المخطط حيث يمكن أن يكون له أيضًا تأثير سلبي على العينات المولدة. بالإضافة إلى ذلك، يمكن أن تكون نقاط تفتيش PIA حساسة لجدول بيتا للمخطط. نوصي بتعيين هذا إلى `linear`.

</Tip>

## استخدام FreeInit

[FreeInit: سد فجوة التهيئة في نماذج انتشار الفيديو](https://arxiv.org/abs/2312.07537) بقلم تيانكسينغ وو، شينيانغ سي، يومينغ جيانغ، زيقي هوانغ، زيوي ليو.

FreeInit هي طريقة فعالة لتحسين الاتساق الزمني والجودة الشاملة للفيديوهات التي تم إنشاؤها باستخدام نماذج انتشار الفيديو دون أي تدريب إضافي. يمكن تطبيقه على PIA و AnimateDiff و ModelScope و VideoCrafter ومختلف نماذج إنشاء الفيديو الأخرى بسلاسة في وقت الاستدلال، ويعمل عن طريق تنقية ضوضاء التهيئة الأولية بشكل تكراري. يمكن العثور على مزيد من التفاصيل في الورقة.

يوضح المثال التالي استخدام FreeInit.

```python
import torch
from diffusers import (
    DDIMScheduler,
    MotionAdapter,
    PIAPipeline,
)
from diffusers.utils import export_to_gif, load_image

adapter = MotionAdapter.from_pretrained("openmmlab/PIA-condition-adapter")
pipe = PIAPipeline.from_pretrained("SG161222/Realistic_Vision_V6.0_B1_noVAE", motion_adapter=adapter)

# enable FreeInit
# راجع وثائق enable_free_init للحصول على قائمة كاملة من المعلمات القابلة للتكوين
pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

# خيارات توفير الذاكرة
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()

pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
image = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png?download=true"
)
image = image.resize((512, 512))
prompt = "cat in a field"
negative_prompt = "wrong white balance, dark, sketches,worst quality,low quality"

generator = torch.Generator("cpu").manual_seed(0)

output = pipe(image=image, prompt=prompt, generator=generator)
frames = output.frames[0]
export_to_gif(frames, "pia-freeinit-animation.gif")
```

<table>
<tr>
<td><center>
cat in a field.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/pia-freeinit-output-cat.gif"
alt="cat in a field"
style="width: 300px;" />
</center></td>
</tr>
</table>

<Tip warning={true}>

FreeInit ليس مجانيًا بالفعل - تأتي الجودة المحسنة بتكلفة حسابات إضافية. فهو يتطلب أخذ عينات إضافية عدة مرات اعتمادًا على معلمة `num_iters` التي يتم تعيينها عند تمكينها. يمكن أن يؤدي تعيين معلمة `use_fast_sampling` إلى `True` إلى تحسين الأداء بشكل عام (بتكلفة جودة أقل مقارنة عندما `use_fast_sampling=False` ولكن لا تزال نتائج أفضل من نماذج إنشاء الفيديو الأساسية).

</Tip>

## خط أنابيب PIAPipeline

[[autodoc]] خط أنابيب PIAPipeline

- all
- __call__
- enable_freeu
- disable_freeu
- enable_free_init
- disable_free_init
- enable_vae_slicing
- disable_vae_slicing
- enable_vae_tiling
- disable_vae_tiling

## إخراج خط أنابيب PIAPipeline

[[autodoc]] pipelines.pia.PIAPipelineOutput