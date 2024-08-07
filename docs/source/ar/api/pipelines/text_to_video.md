<Tip warning={true}>
🧪 هذا الأنبوب للأغراض البحثية فقط.
</Tip>

# النص إلى الفيديو

[تقرير ModelScope Text-to-Video Technical](https://arxiv.org/abs/2308.06571) من تأليف جييونيو وانغ، هانغجيه يوان، دايو تشن، يينغيا تشانغ، شيانغ وانغ، وشييوي تشانغ.

الملخص من الورقة هو:

*يقدم هذا البحث ModelScopeT2V، وهو نموذج توليف نص إلى فيديو يتطور من نموذج توليف نص إلى صورة (أي Stable Diffusion). يدمج ModelScopeT2V كتلًا مكانية زمانية لضمان اتساق إنشاء الإطارات وانتقالات الحركة السلسة. يمكن للنموذج التكيف مع أعداد الإطارات المختلفة أثناء التدريب والاستدلال، مما يجعله مناسبًا لكل من مجموعات البيانات النصية والمرئية النصية. يجمع ModelScopeT2V بين ثلاثة مكونات (أي VQGAN، ومشفر نصي، و Denoising UNet)، ويضم إجمالي 1.7 مليار معلمة، يتم تخصيص 0.5 مليار منها للقدرات الزمنية. يظهر النموذج أداء متفوقًا على الطرق المتقدمة عبر ثلاثة مقاييس تقييم. الكود ونسخة تجريبية عبر الإنترنت متوفرة على https://modelscope.cn/models/damo/text-to-video-synthesis/summary.*

يمكنك العثور على معلومات إضافية حول Text-to-Video على [صفحة المشروع](https://modelscope.cn/models/damo/text-to-video-synthesis/summary)، [قاعدة الكود الأصلية](https://github.com/modelscope/modelscope/)، وجربها في [نسخة تجريبية](https://huggingface.co/spaces/damo-vilab/modelscope-text-to-video-synthesis). يمكن العثور على نقاط التفتيش الرسمية في [damo-vilab](https://huggingface.co/damo-vilab) و [cerspense](https://huggingface.co/cerspense).

## مثال على الاستخدام

### `text-to-video-ms-1.7b`

لنبدأ من خلال إنشاء فيديو قصير بطول افتراضي يبلغ 16 إطارًا (2 ثانية بسرعة 8 إطارات في الثانية):

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to("cuda")

prompt = "Spiderman is surfing"
video_frames = pipe(prompt).frames[0]
video_path = export_to_video(video_frames)
video_path
```

تدعم Diffusers تقنيات تحسين مختلفة لتحسين الكمون وبصمة الذاكرة للأنبوب. نظرًا لأن مقاطع الفيديو غالبًا ما تكون أكثر كثافة في الذاكرة من الصور، يمكننا تمكين الإخراج إلى CPU وتقسيم VAE لإبقاء بصمة الذاكرة تحت السيطرة.

لننشئ فيديو مدته 8 ثوانٍ (64 إطارًا) على نفس وحدة معالجة الرسومات باستخدام الإخراج إلى CPU وتقسيم VAE:

```python
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.enable_model_cpu_offload()

# memory optimization
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=64).frames[0]
video_path = export_to_video(video_frames)
video_path
```

يستهلك الأمر **7 جيجابايت فقط من ذاكرة GPU** لتوليد 64 إطار فيديو باستخدام PyTorch 2.0 ودقة "fp16" والتقنيات المذكورة أعلاه.

يمكننا أيضًا استخدام مخطط مختلف بسهولة، باستخدام نفس الطريقة التي نستخدمها لـ Stable Diffusion:

```python
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

prompt = "Spiderman is surfing"
video_frames = pipe(prompt, num_inference_steps=25).frames[0]
video_path = export_to_video(video_frames)
video_path
```

فيما يلي بعض النماذج الإخراجية:

<table>
<tr>
<td><center>
رائد فضاء يركب حصانًا.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astr.gif"
alt="رائد فضاء يركب حصانًا."
style="width: 300px;" />
</center></td>
<td ><center>
دارث فيدر يركب الأمواج.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/vader.gif"
alt="دارث فيدر يركب الأمواج."
style="width: 300px;" />
</center></td>
</tr>
</table>

### `cerspense/zeroscope_v2_576w` و`cerspense/zeroscope_v2_XL`

Zeroscope هي نموذج خالٍ من العلامات المائية وتم تدريبه على أحجام محددة مثل `576x320` و`1024x576`.

يجب أولاً إنشاء فيديو باستخدام نقطة تفتيش الدقة المنخفضة [`cerspense/zeroscope_v2_576w`](https://huggingface.co/cerspense/zeroscope_v2_576w) مع [`TextToVideoSDPipeline`]،
والذي يمكن بعد ذلك توسيعه باستخدام [`VideoToVideoSDPipeline`] و [`cerspense/zeroscope_v2_XL`](https://huggingface.co/cerspense/zeroscope_v2_XL).

```py
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

prompt = "Darth Vader surfing a wave"
video_frames = pipe(prompt, num_frames=24).frames[0]
video_path = export_to_video(video_frames)
video_path
```

الآن يمكن توسيع نطاق الفيديو:

```py
pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_XL", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# memory optimization
pipe.unet.enable_forward_chunking(chunk_size=1, dim=1)
pipe.enable_vae_slicing()

video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]

video_frames = pipe(prompt, video=video, strength=0.6).frames[0]
video_path = export_to_video(video_frames)
video_path
```

فيما يلي بعض النماذج الإخراجية:

<table>
<tr>
<td ><center>
دارث فيدر يركب الأمواج.
<br>
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/darthvader_cerpense.gif"
alt="دارث فيدر يركب الأمواج."
style="width: 576px;" />
</center></td>
</tr>
</table>

## نصائح

إن إنشاء الفيديو كثيف الاستخدام للذاكرة، وأحد طرق تقليل استخدام الذاكرة هو تعيين `enable_forward_chunking` على UNet الخاص بالأنبوب بحيث لا تقوم بتشغيل طبقة التغذية الأمامية بالكامل مرة واحدة. من الأكثر كفاءة تقسيمها إلى مجموعات في حلقة.

راجع الدليل [من النص أو الصورة إلى الفيديو](text-img2vid) لمزيد من التفاصيل حول كيفية تأثير بعض المعلمات على إنشاء الفيديو وكيفية تحسين الاستدلال عن طريق تقليل استخدام الذاكرة.

<Tip>
تأكد من الاطلاع على دليل [المخططات](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف المقايضة بين سرعة المخطط والنوعية، وانظر قسم [إعادة استخدام المكونات عبر الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في أنابيب متعددة.
</Tip>

## TextToVideoSDPipeline
[[autodoc]] TextToVideoSDPipeline
- all
- __call__

## VideoToVideoSDPipeline
[[autodoc]] VideoToVideoSDPipeline
- all
- __call__

## TextToVideoSDPipelineOutput
[[autodoc]] pipelines.text_to_video_synthesis.TextToVideoSDPipelineOutput