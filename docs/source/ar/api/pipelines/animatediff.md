# إنشاء فيديو من نص باستخدام AnimateDiff

## نظرة عامة
[AnimateDiff: قم بتحريك نماذج النص إلى الصورة المخصصة الخاصة بك دون ضبط محدد](https://arxiv.org/abs/2307.04725) بواسطة Yuwei Guo, Ceyuan Yang, Anyi Rao, Yaohui Wang, Yu Qiao, Dahua Lin, Bo Dai.

ملخص الورقة هو ما يلي:

*مع تقدم نماذج النص إلى الصورة (مثل Stable Diffusion) وتقنيات التخصيص المقابلة مثل DreamBooth وLoRA، يمكن لأي شخص تجسيد خياله في صور عالية الجودة بتكلفة معقولة. بعد ذلك، هناك طلب كبير على تقنيات تحريك الصور لمواصلة دمج الصور الثابتة المولدة مع ديناميكيات الحركة. في هذا التقرير، نقترح إطار عمل عملي لتحريك معظم نماذج النص إلى الصورة المخصصة الموجودة مرة واحدة وإلى الأبد، مما يوفر الجهد في الضبط المحدد للنموذج. يكمن جوهر إطار العمل المقترح في إدراج وحدة نمذجة حركة تم تهيئتها حديثًا في نموذج النص إلى الصورة المجمد وتدريبه على مقاطع الفيديو لاستخلاص أولويات الحركة المعقولة. بمجرد تدريبه، من خلال حقن وحدة نمذجة الحركة هذه ببساطة، تصبح جميع الإصدارات المخصصة المستمدة من نفس النص الأساسي إلى نموذج النص إلى الصورة جاهزة نماذج مدفوعة بالنص تنتج صورًا متحركة متنوعة وشخصية. نجري تقييمنا على عدة نماذج نصية عامة مخصصة عبر صور الأنمي والصور الفوتوغرافية الواقعية، ونثبت أن إطار العمل المقترح يساعد هذه النماذج على إنشاء مقاطع متحركة سلسة زمنيًا مع الحفاظ على مجال تنوع مخرجاتها ومجالها.*

## خطوط الأنابيب المتاحة

| خط الأنابيب | المهام | عرض توضيحي |
|---|---|:---:|
| [خط أنابيب AnimateDiff](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff.py) | *إنشاء فيديو من نص باستخدام AnimateDiff* |
| [خط أنابيب AnimateDiffVideoToVideo](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/animatediff/pipeline_animatediff_video2video.py) | *إنشاء فيديو من فيديو باستخدام AnimateDiff* |

## نقاط التفتيش المتاحة

يمكن العثور على نقاط تفتيش Motion Adapter في [guoyww](https://huggingface.co/guoyww/). يُقصد بنقاط التفتيش هذه العمل مع أي نموذج يعتمد على Stable Diffusion 1.4/1.5.

## مثال على الاستخدام

### خط أنابيب AnimateDiff

يعمل AnimateDiff مع نقطة تفتيش MotionAdapter ونقطة تفتيش نموذج Stable Diffusion. يتكون MotionAdapter من مجموعة من وحدات الحركة المسؤولة عن إضافة حركة متسقة عبر إطارات الصور. يتم تطبيق هذه الوحدات بعد كتل Resnet وAttention في شبكة UNet الخاصة بنموذج Stable Diffusion.

يوضح المثال التالي كيفية استخدام نقطة تفتيش *MotionAdapter* مع أجهزة نشر الحركة للاستدلال القائم على StableDiffusion-1.4/1.5.

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# تحميل محول الحركة
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# تحميل نموذج SD 1.5 المُدرب بشكل دقيق
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# تمكين توفير الذاكرة
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")

```

فيما يلي بعض النماذج:

<table>
<tr>
<td><center>
masterpiece, bestquality, sunset.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-realistic-doc.gif"
alt="masterpiece, bestquality, sunset"
style="width: 300px;" />
</center></td>
</tr>
</table>

<Tip>

يعمل AnimateDiff بشكل أفضل مع نماذج Stable Diffusion المدربة بشكل دقيق. إذا كنت تخطط لاستخدام جدول زمني يمكنه قص العينات، فتأكد من تعطيله عن طريق تعيين `clip_sample=False` في الجدول الزمني حيث يمكن أن يكون لذلك أيضًا تأثير سلبي على العينات المولدة. بالإضافة إلى ذلك، قد تكون نقاط تفتيش AnimateDiff حساسة لجدول بيتا للجدول الزمني. نوصي بتعيين هذا إلى `linear`.

</Tip>

### خط أنابيب AnimateDiffSDXL

يمكن أيضًا استخدام AnimateDiff مع نماذج SDXL. هذه ميزة تجريبية حاليًا حيث تتوفر فقط نسخة تجريبية من نقطة تفتيش محول الحركة.

```python
import torch
from diffusers.models import MotionAdapter
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-sdxl-beta", torch_dtype=torch.float16)

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe = AnimateDiffSDXLPipeline.from_pretrained(
    model_id,
    motion_adapter=adapter,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# تمكين توفير الذاكرة
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

output = pipe(
    prompt="a panda surfing in the ocean, realistic, high quality",
    negative_prompt="low quality, worst quality",
    num_inference_steps=20,
    guidance_scale=8,
    width=1024,
    height=1024,
    num_frames=16,
)

frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

### خط أنابيب AnimateDiffVideoToVideo

يمكن أيضًا استخدام AnimateDiff لإنشاء مقاطع فيديو بصرية مماثلة أو تمكين التعديلات على الأسلوب/الشخصية/الخلفية أو غيرها من التعديلات بدءًا من الفيديو الأولي، مما يتيح لك استكشاف الإمكانات الإبداعية بسلاسة.

```python
import imageio
import requests
import torch
from diffusers import AnimateDiffVideoToVideoPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from io import BytesIO
from PIL import Image

# تحميل محول الحركة
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# تحميل نموذج SD 1.5 المُدرب بشكل دقيق
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# تمكين توفير الذاكرة
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

# دالة مساعدة لتحميل مقاطع الفيديو
def load_video(file_path: str):
    images = []

    if file_path.startswith(('http://', 'https://')):
        # إذا كان file_path عنوان URL
        response = requests.get(file_path)
        response.raise_for_status()
        content = BytesIO(response.content)
        vid = imageio.get_reader(content)
    else:
        # يفترض أنه مسار ملف محلي
        vid = imageio.get_reader(file_path)

    for frame in vid:
        pil_image = Image.fromarray(frame)
        images.append(pil_image)

    return images

video = load_video("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif")

output = pipe(
    video = video,
    prompt="panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    guidance_scale=7.5,
    num_inference_steps=25,
    strength=0.5,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

فيما يلي بعض النماذج:

<table>
<tr>
<th align=center>فيديو المصدر</th>
<th align=center>فيديو الإخراج</th>
</tr>
<tr>
<td align=center>
raccoon playing a guitar
<br />
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-1.gif"
alt="racoon playing a guitar"
style="width: 300px;" />
</td>
<td align=center>
panda playing a guitar
<br/>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-output-1.gif"
alt="panda playing a guitar"
style="width: 300px;" />
</td>
</tr>
<tr>
<td align=center>
closeup of margot robbie, fireworks in the background, high quality
<br />
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-input-2.gif"
alt="closeup of margot robbie, fireworks in the background, high quality"
style="width: 300px;" />
</td>
<td align=center>
closeup of tony stark, robert downey jr, fireworks
<br/>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-vid2vid-output-2.gif"
alt="closeup of tony stark, robert downey jr, fireworks"
style="width: 300px;" />
</td>
</tr>
</table>

## استخدام Motion LoRAs

Motion LoRAs عبارة عن مجموعة من LoRAs التي تعمل مع نقطة تفتيش `guoyww/animatediff-motion-adapter-v1-5-2`. هذه LoRAs مسؤولة عن إضافة أنواع محددة من الحركة إلى الرسوم المتحركة.

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# تحميل محول الحركة
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# تحميل نموذج SD 1.5 المُدرب بشكل دقيق
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)
pipe.load_lora_weights(
    "guoyww/animatediff-motion-lora-zoom-out", adapter_name="zoom-out"
)

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    beta_schedule="linear",
    timestep_spacing="linspace",
    steps_offset=1,
)
pipe.scheduler = scheduler

# تمكين توفير الذاكرة
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")

```

<table>
<tr>
<td><center>
masterpiece, bestquality, sunset.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-zoom-out-lora.gif"
alt="masterpiece, bestquality, sunset"
style="width: 300px;" />
</center></td>
</tr>
</table>
## استخدام Motion LoRAs مع PEFT

يمكنك أيضًا الاستفادة من واجهة برمجة التطبيقات الخلفية لـ [PEFT](https://github.com/huggingface/peft) لدمج Motion LoRA وإنشاء رسوم متحركة أكثر تعقيدًا.

قم أولاً بتثبيت PEFT باستخدام:

```shell
pip install peft
```

بعد ذلك، يمكنك استخدام الكود التالي لدمج Motion LoRAs.

```python
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

# تحميل محول الحركة
adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
# تحميل النموذج الدقيق المستند إلى SD 1.5
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)

pipe.load_lora_weights(
    "diffusers/animatediff-motion-lora-zoom-out", adapter_name="zoom-out",
)
pipe.load_lora_weights(
    "diffusers/animatediff-motion-lora-pan-left", adapter_name="pan-left",
)
pipe.set_adapters(["zoom-out", "pan-left"], adapter_weights=[1.0, 1.0])

scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    clip_sample=False,
    timestep_spacing="linspace",
    beta_schedule="linear",
    steps_offset=1,
)
pipe.scheduler = scheduler

# تمكين توفير الذاكرة
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt=(
        "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
        "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
        "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
        "golden hour, coastal landscape, seaside scenery"
    ),
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(42),
)
frames = output.frames[0]
export_to_gif(frames, "animation.gif")

```

<table>
<tr>
<td><center>
masterpiece, bestquality, sunset.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-zoom-out-pan-left-lora.gif"
alt="masterpiece, bestquality, sunset"
style="width: 300px;" />
</center></td>
</tr>
</table>

## استخدام FreeInit

[FreeInit: Bridging Initialization Gap in Video Diffusion Models](https://arxiv.org/abs/2312.07537) بواسطة Tianxing Wu، وChenyang Si، وYuming Jiang، وZiqi Huang، وZiwei Liu.

FreeInit هي طريقة فعالة لتحسين الاتساق الزمني والجودة الشاملة للفيديوهات التي تم إنشاؤها باستخدام نماذج انتشار الفيديو دون أي تدريب إضافي. يمكن تطبيقه بسلاسة على AnimateDiff، وModelScope، وVideoCrafter، ومختلف نماذج إنشاء الفيديو الأخرى في وقت الاستدلال، وهو يعمل عن طريق تنقية ضوضاء التهيئة الأولية بشكل تكراري. يمكن العثور على مزيد من التفاصيل في الورقة.

يوضح المثال التالي استخدام FreeInit.

```python
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_pretrained(
    model_id,
    subfolder="scheduler",
    beta_schedule="linear",
    clip_sample=False,
    timestep_spacing="linspace",
    steps_offset=1
)

# تمكين توفير الذاكرة
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# تمكين FreeInit
# راجع وثائق enable_free_init للحصول على قائمة كاملة من المعلمات القابلة للتكوين
pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

# تشغيل الاستدلال
output = pipe(
    prompt="a panda playing a guitar, on a boat, in the ocean, high quality",
    negative_prompt="bad quality, worse quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(666),
)

# تعطيل FreeInit
pipe.disable_free_init()

frames = output.frames[0]
export_to_gif(frames, "animation.gif")
```

<Tip warning={true}>
FreeInit ليست مجانية حقًا - تأتي الجودة المحسنة بتكلفة حسابات إضافية. فهو يتطلب أخذ عينات إضافية عدة مرات اعتمادًا على معلمة `num_iters` التي يتم تعيينها عند التمكين. يمكن أن يؤدي تعيين معلمة `use_fast_sampling` إلى `True` إلى تحسين الأداء العام (بتكلفة جودة أقل مقارنة بـ `use_fast_sampling=False` ولكن لا تزال النتائج أفضل من نماذج إنشاء الفيديو الأساسية).
</Tip>

<Tip>
تأكد من مراجعة دليل [Schedulers](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة المجدول والجودة، وقسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.
</Tip>

<table>
<tr>
<th align=center>بدون تمكين FreeInit</th>
<th align=center>مع تمكين FreeInit</th>
</tr>
<tr>
<td align=center>
panda playing a guitar
<br />
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-no-freeinit.gif"
alt="panda playing a guitar"
style="width: 300px;" />
</td>
<td align=center>
panda playing a guitar
<br/>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatediff-freeinit.gif"
alt="panda playing a guitar"
style="width: 300px;" />
</td>
</tr>
</table>

## استخدام AnimateLCM

[AnimateLCM](https://animatelcm.github.io/) هو نقطة تفتيش لوحدة الحركة و [LCM LoRA](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm_lora) تم إنشاؤها باستخدام استراتيجية تعلم الاتساق التي تفصل بين تقطير سوابق إنشاء الصور وسوابق إنشاء الحركة.

```python
import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")

pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animatelcm.gif")
```

<table>
<tr>
<td><center>
A space rocket, 4K.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatelcm-output.gif"
alt="A space rocket, 4K"
style="width: 300px;" />
</center></td>
</tr>
</table>

AnimateLCM متوافق أيضًا مع [Motion LoRAs](https://huggingface.co/collections/dn6/animatediff-motion-loras-654cb8ad732b9e3cf4d3c17e) الموجودة.

```python
import torch
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")
pipe = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors", adapter_name="lcm-lora")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-tilt-up", adapter_name="tilt-up")

pipe.set_adapters(["lcm-lora", "tilt-up"], [1.0, 0.8])
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="A space rocket with trails of smoke behind it launching into space from the desert, 4k, high resolution",
    negative_prompt="bad quality, worse quality, low resolution",
    num_frames=16,
    guidance_scale=1.5,
    num_inference_steps=6,
    generator=torch.Generator("cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "animatelcm-motion-lora.gif")
```

<table>
<tr>
<td><center>
A space rocket, 4K.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/animatelcm-motion-lora.gif"
alt="A space rocket, 4K"
style="width: 300px;" />
</center></td>
</tr>
</table>

## AnimateDiffPipeline

[[autodoc]] AnimateDiffPipeline
- all
- __call__

## AnimateDiffSDXLPipeline

[[autodoc]] AnimateDiffSDXLPipeline
- all
- __call__

## AnimateDiffVideoToVideoPipeline

[[autodoc]] AnimateDiffVideoToVideoPipeline
- all
- __call__

## AnimateDiffPipelineOutput

[[autodoc]] pipelines.animatediff.AnimateDiffPipelineOutput