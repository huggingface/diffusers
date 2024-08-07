# Text2Video-Zero

[Text2Video-Zero: Text-to-Image Diffusion Models are Zero-Shot Video Generators](https://huggingface.co/papers/2303.13439) هو من تأليف Levon Khachatryan و Andranik Movsisyan و Vahram Tadevosyan و Roberto Henschel و [Zhangyang Wang](https://www.ece.utexas.edu/people/faculty/atlas-wang) و Shant Navasardyan و [Humphrey Shi](https://www.humphreyshi.com).

يتيح Text2Video-Zero إمكانية توليد الفيديو بدون تدريب باستخدام إما:

1. موجه نصي
2. موجه مدمج مع إرشادات من الوضعيات أو الحواف
3. فيديو Instruct-Pix2Pix (تحرير الفيديو الموجه بالتعليمات)

تكون النتائج متسقة زمنيا وتتبع عن كثب الإرشادات والموجهات النصية.

![teaser-img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/t2v_zero_teaser.png)

الملخص من الورقة هو:

* تعتمد طرق توليد الفيديو من النص الحديثة على التدريب المكثف حسابيا وتتطلب مجموعات بيانات فيديو واسعة النطاق. وفي هذه الورقة، نقدم مهمة جديدة لتوليد الفيديو من النص بدون تدريب ونقترح نهجًا منخفض التكلفة (بدون أي تدريب أو تحسين) من خلال الاستفادة من قوة طرق توليد الصور من النص الموجودة (مثل Stable Diffusion)، مما يجعلها مناسبة لمجال الفيديو. وتتمثل تعديلاتنا الرئيسية في (1) إثراء الرموز المخفية للأطر المولدة بديناميكيات الحركة للحفاظ على المشهد العالمي واتساق الخلفية الزمني؛ و (2) إعادة برمجة الاهتمام الذاتي على مستوى الإطار باستخدام اهتمام متقاطع جديد لكل إطار على الإطار الأول، للحفاظ على سياق الكائن في المقدمة ومظهره وهويته. وتظهر التجارب أن هذا يؤدي إلى توليد فيديو عالي الجودة ومتسق بشكل ملحوظ، مع زيادة طفيفة في التحميل. علاوة على ذلك، لا تقتصر طريقة عملنا على توليد الفيديو من النص، بل تنطبق أيضًا على مهام أخرى مثل توليد الفيديو المشروط والمتخصص في المحتوى، و Video Instruct-Pix2Pix، أي تحرير الفيديو الموجه بالتعليمات. وكما تظهر التجارب، فإن طريقة عملنا تؤدي أداءً مماثلاً أو أفضل في بعض الأحيان من الطرق الحديثة، على الرغم من عدم تدريبها على بيانات فيديو إضافية.*

يمكنك العثور على معلومات إضافية حول Text2Video-Zero على [صفحة المشروع](https://text2video-zero.github.io/) و[الورقة](https://arxiv.org/abs/2303.13439) و[رمز المصدر الأصلي](https://github.com/Picsart-AI-Research/Text2Video-Zero).

## مثال على الاستخدام

### نص إلى فيديو

لإنشاء فيديو من موجه، قم بتشغيل رمز Python التالي:

```python
import torch
from diffusers import TextToVideoZeroPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

prompt = "A panda is playing guitar on times square"
result = pipe(prompt=prompt).images
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video.mp4", result, fps=4)
```

يمكنك تغيير هذه المعلمات في مكالمة الأنبوب:

* قوة حقل الحركة (راجع [الورقة](https://arxiv.org/abs/2303.13439)، القسم 3.3.1):
* `motion_field_strength_x` و`motion_field_strength_y`. الافتراضي: `motion_field_strength_x=12`، `motion_field_strength_y=12`
* `T` و`T'` (راجع [الورقة](https://arxiv.org/abs/2303.13439)، القسم 3.3.1)
* `t0` و`t1` في النطاق `{0, ..., num_inference_steps}`. الافتراضي: `t0=45`، `t1=48`
* طول الفيديو:
* `video_length`، عدد الأطر التي سيتم إنشاؤها. الافتراضي: `video_length=8`

يمكننا أيضًا إنشاء مقاطع فيديو أطول عن طريق المعالجة بطريقة المجزأة:

```python
import torch
from diffusers import TextToVideoZeroPipeline
import numpy as np

model_id = "runwayml/stable-diffusion-v1-5"
pipe = TextToVideoZeroPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
seed = 0
video_length = 24  #24 ÷ 4fps = 6 seconds
chunk_size = 8
prompt = "A panda is playing guitar on times square"

# Generate the video chunk-by-chunk
result = []
chunk_ids = np.arange(0, video_length, chunk_size - 1)
generator = torch.Generator(device="cuda")
for i in range(len(chunk_ids)):
    print(f"Processing chunk {i + 1} / {len(chunk_ids)}")
    ch_start = chunk_ids[i]
    ch_end = video_length if i == len(chunk_ids) - 1 else chunk_ids[i + 1]
    # Attach the first frame for Cross Frame Attention
    frame_ids = [0] + list(range(ch_start, ch_end))
    # Fix the seed for the temporal consistency
    generator.manual_seed(seed)
    output = pipe(prompt=prompt, video_length=len(frame_ids), generator=generator, frame_ids=frame_ids)
    result.append(output.images[1:])

# Concatenate chunks and save
result = np.concatenate(result)
result = [(r * 255).astype("uint8") for r in result]
imageio.mimsave("video.mp4", result, fps=4)
```

- #### دعم SDXL

لاستخدام نموذج SDXL عند إنشاء فيديو من موجه، استخدم خط أنابيب `TextToVideoZeroSDXLPipeline`:

```python
import torch
from diffusers import TextToVideoZeroSDXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = TextToVideoZeroSDXLPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")
```

### نص إلى فيديو مع التحكم في الوضعية

لإنشاء فيديو من موجه مع التحكم الإضافي في الوضعية:

1. قم بتنزيل فيديو توضيحي:

```python
from huggingface_hub import hf_hub_download

filename = "__assets__/poses_skeleton_gifs/dance1_corr.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
```

2. اقرأ الفيديو الذي يحتوي على صور الوضعية المستخرجة:

```python
from PIL import Image
import imageio

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
pose_images = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

لاستخراج الوضعية من الفيديو الفعلي، اقرأ [وثائق ControlNet](controlnet).

3. قم بتشغيل `StableDiffusionControlNetPipeline` مع معالج الاهتمام المخصص لدينا:

```python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "runwayml/stable-diffusion-v1-5"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

- #### دعم SDXL

نظرًا لأن معالج الاهتمام الخاص بنا يعمل أيضًا مع SDXL، فيمكن استخدامه لإنشاء فيديو من موجه باستخدام نماذج ControlNet المدعومة من SDXL:

```python
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

controlnet_model_id = 'thibaud/controlnet-openpose-sdxl-1.0'
model_id = 'stabilityai/stable-diffusion-xl-base-1.0'

controlnet = ControlNetModel.from_pretrained(controlnet_model_id, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to('cuda')

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 128, 128), device="cuda", dtype=torch.float16).repeat(len(pose_images), 1, 1, 1)

prompt = "Darth Vader dancing in a desert"
result = pipe(prompt=[prompt] * len(pose_images), image=pose_images, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

### نص إلى فيديو مع التحكم في الحافة

لإنشاء فيديو من موجه مع التحكم الإضافي في حافة Canny، اتبع الخطوات نفسها الموضحة أعلاه لتوليد الوضعية الموجهة باستخدام [نموذج ControlNet للحافة Canny](https://huggingface.co/lllyasviel/sd-controlnet-canny).

### فيديو Instruct-Pix2Pix

للقيام بتحرير الفيديو الموجه بالنص (مع [InstructPix2Pix](pix2pix)):

1. قم بتنزيل فيديو توضيحي:

```python
from huggingface_hub import hf_hub_download

filename = "__assets__/pix2pix video/camel.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
```

2. اقرأ الفيديو من المسار:

```python
from PIL import Image
import imageio

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
video = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

3. قم بتشغيل `StableDiffusionInstructPix2PixPipeline` مع معالج الاهتمام المخصص لدينا:

```python
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=3))

prompt = "make it Van Gogh Starry Night style"
result = pipe(prompt=[prompt] * len(video), image=video).images
imageio.mimsave("edited_video.mp4", result, fps=4)
```

### تخصص DreamBooth

يمكن لطرق **Text-To-Video** و**Text-To-Video with Pose Control** و**Text-To-Video with Edge Control**
أن تعمل مع نماذج [DreamBooth](../../training/dreambooth) المخصصة، كما هو موضح أدناه لنموذج
[Canny edge ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-canny) و
[Avatar style DreamBooth](https://huggingface.co/PAIR/text2video-zero-controlnet-canny-avatar) model:

1. قم بتنزيل فيديو توضيحي:

```python
from huggingface_hub import hf_hub_download

filename = "__assets__/canny_videos_mp4/girl_turning.mp4"
repo_id = "PAIR/Text2Video-Zero"
video_path = hf_hub_download(repo_type="space", repo_id=repo_id, filename=filename)
```

2. اقرأ الفيديو من المسار:

```python
from PIL import Image
import imageio

reader = imageio.get_reader(video_path, "ffmpeg")
frame_count = 8
canny_edges = [Image.fromarray(reader.get_data(i)) for i in range(frame_count)]
```

3. قم بتشغيل `StableDiffusionControlNetPipeline` مع نموذج DreamBooth المدرب المخصص:

```python
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor

# set model id to custom model
model_id = "PAIR/text2video-zero-controlnet-canny-avatar"
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    model_id, controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

# Set the attention processor
pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
pipe.controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))

# fix latents for all frames
latents = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float16).repeat(len(canny_edges), 1, 1, 1)

prompt = "oil painting of a beautiful girl avatar style"
result = pipe(prompt=[prompt] * len(canny_edges), image=canny_edges, latents=latents).images
imageio.mimsave("video.mp4", result, fps=4)
```

يمكنك تصفية بعض نماذج DreamBooth المتاحة من خلال [هذا الرابط](https://huggingface.co/models?search=dreambooth).

<Tip>

تأكد من مراجعة دليل [الموقتات](../../using-diffusers/schedulers) لمعرفة كيفية استكشاف التوازن بين سرعة الموقت والجودة، وقسم [إعادة استخدام المكونات عبر خطوط الأنابيب](../../using-diffusers/loading#reuse-components-across-pipelines) لمعرفة كيفية تحميل المكونات نفسها بكفاءة في خطوط أنابيب متعددة.

</Tip>

## TextToVideoZeroPipeline

[[autodoc]] TextToVideoZero