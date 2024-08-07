[[open-in-colab]]
 # التقطير باستمرارية المسار-LoRA

تمكّن تقطير استمرارية المسار (TCD) النموذج من إنشاء صور ذات جودة أعلى وتفاصيل أكثر في عدد أقل من الخطوات. علاوة على ذلك، وبفضل التخفيف الفعال من الأخطاء أثناء عملية التقطير، يظهر TCD أداءً متفوقًا حتى في ظل ظروف خطوات الاستنتاج الكبيرة.

المزايا الرئيسية لـ TCD هي:

- أفضل من المعلم: يظهر TCD جودة توليدية متفوقة في كل من خطوات الاستنتاج الصغيرة والكبيرة ويتجاوز أداء [DPM-Solver++(2S)](../../api/schedulers/multistep_dpm_solver) مع Stable Diffusion XL (SDXL). لا توجد أي ميزات تمييزية أو إشراف LPIPS مدرجة أثناء تدريب TCD.

- خطوات الاستنتاج المرنة: يمكن ضبط خطوات الاستنتاج لعينات TCD بحرية دون التأثير سلبًا على جودة الصورة.

- تغيير مستوى التفاصيل بحرية: أثناء الاستنتاج، يمكن ضبط مستوى التفاصيل في الصورة باستخدام معامل واحد، "غاما".

> [!TIP]
> للحصول على مزيد من التفاصيل التقنية حول TCD، يرجى الرجوع إلى [الورقة البحثية](https://arxiv.org/abs/2402.19159) أو صفحة [المشروع الرسمية](https://mhh0318.github.io/tcd/).

بالنسبة للنماذج الكبيرة مثل SDXL، يتم تدريب TCD باستخدام [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora) لتقليل استخدام الذاكرة. هذا مفيد أيضًا لأنه يمكنك إعادة استخدام LoRAs بين النماذج الدقيقة المختلفة، طالما أنها تشترك في نفس النموذج الأساسي، دون الحاجة إلى تدريب إضافي.

سيوضح هذا الدليل كيفية إجراء الاستدلال باستخدام TCD-LoRAs لمجموعة متنوعة من المهام مثل النص إلى الصورة والطلاء، وكذلك كيفية الجمع بين TCD-LoRAs بسهولة مع المحولات الأخرى. اختر أحد النماذج الأساسية المدعومة ونقطة تفتيش TCD-LoRA المقابلة من الجدول أدناه للبدء.

| النموذج الأساسي | نقطة تفتيش TCD-LoRA |
|-------------------------------------------------------------------------------------------------|----------------------------------------------------------------|
| [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | [TCD-SD15](https://huggingface.co/h1t/TCD-SD15-LoRA) |
| [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) | [TCD-SD21-base](https://huggingface.co/h1t/TCD-SD21-base-LoRA) |
| [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | [TCD-SDXL](https://huggingface.co/h1t/TCD-SDXL-LoRA) |

تأكد من تثبيت [PEFT](https://github.com/huggingface/peft) للحصول على دعم LoRA أفضل.

```bash
pip install -U peft
```

## المهام العامة

في هذا الدليل، دعنا نستخدم [`StableDiffusionXLPipeline`] و [`TCDScheduler`]. استخدم طريقة [`~StableDiffusionPipeline.load_lora_weights`] لتحميل أوزان TCD-LoRA المتوافقة مع SDXL.

فيما يلي بعض النصائح التي يجب مراعاتها عند الاستدلال باستخدام TCD-LoRA:

- حافظ على `num_inference_steps` بين 4 و50

- قم بتعيين `eta` (يستخدم للتحكم في العشوائية في كل خطوة) بين 0 و1. يجب استخدام "إيتا" أعلى عند زيادة عدد خطوات الاستنتاج، ولكن الجانب السلبي هو أن "إيتا" أكبر في [`TCDScheduler`] يؤدي إلى صور أكثر ضبابية. القيمة الموصى بها هي 0.3 لإنتاج نتائج جيدة.

<hfoptions id="tasks">
<hfoption id="text-to-image">

```python
import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "رسم للقطة البرتقالية أوتو فون غارفيلد، كونت بيسمارك-شونهاوزن، دوق لاوينبورغ، رئيس وزراء بروسيا. ويصور وهو يرتدي خوذة بروسية ويأكل وجبته المفضلة - اللازانيا."

image = pipe(
prompt=prompt,
num_inference_steps=4,
guidance_scale=0,
eta=0.3,
generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/demo_image.png)

</hfoption>

<hfoption id="inpainting">

```python
import torch
from diffusers import AutoPipelineForInpainting, TCDScheduler
from diffusers.utils import load_image, make_image_grid

device = "cuda"
base_model_id = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = AutoPipelineForInpainting.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).resize((1024, 1024))
mask_image = load_image(mask_url).resize((1024, 1024))

prompt = "نمر يجلس على مقعد في الحديقة"

image = pipe(
prompt=prompt,
image=init_image,
mask_image=mask_image,
num_inference_steps=8,
guidance_scale=0,
eta=0.3,
strength=0.99, # تأكد من استخدام "strength" أقل من 1.0
generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([init_image, mask_image, image], rows=1, cols=3)
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/inpainting_tcd.png)

</hfoption>
</hfoptions>

## النماذج المجتمعية

يعمل TCD-LoRA أيضًا مع العديد من النماذج الدقيقة المُدربة من قبل المجتمع والمكونات الإضافية. على سبيل المثال، قم بتحميل نقطة تفتيش [animagine-xl-3.0](https://huggingface.co/cagliostrolab/animagine-xl-3.0) والتي تعد نسخة مُدربة من قبل المجتمع لـ SDXL لتوليد صور أنيمي.

```python
import torch
from diffusers import StableDiffusionXLPipeline, TCDScheduler

device = "cuda"
base_model_id = "cagliostrolab/animagine-xl-3.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "رجل، يرتدي زيًا عسكريًا مصممًا بعناية، يقف بتصميم لا يتزعزع. ويتباهى الزي بتفاصيل معقدة، وتتوهج عيناه بالعزم. وتتدلى خصلات شعر نابضة بالحياة من تحت حافة قبعته بفعل الرياح."

image = pipe(
prompt=prompt,
num_inference_steps=8,
guidance_scale=0,
eta=0.3,
generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/animagine_xl.png)

يدعم TCD-LoRA أيضًا LoRAs أخرى مدربة على أنماط مختلفة. على سبيل المثال، دعنا نحمل [TheLastBen/Papercut_SDXL](https://huggingface.co/TheLastBen/Papercut_SDXL) LoRA وندمجها مع TCD-LoRA باستخدام طريقة [`~loaders.UNet2DConditionLoadersMixin.set_adapters`].

> [!TIP]
> اطلع على دليل [دمج LoRAs](merge_loras) لمعرفة المزيد حول طرق الدمج الفعالة.

```python
import torch
from diffusers import StableDiffusionXLPipeline
from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"
styled_lora_id = "TheLastBen/Papercut_SDXL"

pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16").to(device)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id, adapter_name="tcd")
pipe.load_lora_weights(styled_lora_id, adapter_name="style")
pipe.set_adapters(["tcd", "style"], adapter_weights=[1.0, 1.0])

prompt = "قصاصة ورقية لجبل شتوي، ثلجي"

image = pipe(
prompt=prompt,
num_inference_steps=4,
guidance_scale=0,
eta=0.3,
generator=torch.Generator(device=device).manual_seed(0),
).images[0]
```

![](https://github.com/jabir-zheng/TCD/raw/main/assets/styled_lora.png)

## المحولات

TCD-LoRA متعدد الاستخدامات، ويمكن دمجه مع أنواع أخرى من المحولات مثل ControlNets وIP-Adapter وAnimateDiff.

<hfoptions id="adapters">
<hfoption id="ControlNet">
### Depth ControlNet

في هذا المثال، نستخدم شبكة التحكم التي تم تدريبها على خرائط العمق لتوجيه عملية توليد الصور. أولاً، نقوم بتحميل نموذج تقدير العمق المُدرب مسبقًا واستخراج خريطة العمق من الصورة المدخلة. ثم نقوم بتمرير خريطة العمق هذه إلى أنبوب StableDiffusion XL ControlNet لتوجيه عملية التوليد.

```python
import torch
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image, make_image_grid
from scheduling_tcd import TCDScheduler

device = "cuda"
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")

def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad(), torch.autocast(device):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch0.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-depth-sdxl-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

controlnet = ControlNetModel.from_pretrained(
    controlnet_id,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "stormtrooper lecture, photorealistic"

image = load_image("https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/images/stormtrooper.png")
depth_image = get_depth_map(image)

controlnet_conditioning_scale = 0.5  # الموصى به للتعميم الجيد

image = pipe(
    prompt,
    image=depth_image,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([depth_image, image], rows=1, cols=2)
```

![صورة توضيحية](https://github.com/jabir-zheng/TCD/raw/main/assets/controlnet_depth_tcd.png)

### Canny ControlNet

في هذا المثال، نستخدم شبكة تحكم تم تدريبها على خرائط Canny لتوجيه عملية التوليد. خرائط Canny هي تمثيل للصورة يبرز حواف الكائنات، مما يساعد نموذج Stable Diffusion على التركيز على محتوى الصورة.

```python
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from diffusers.utils import load_image, make_image_grid
from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
controlnet_id = "diffusers/controlnet-canny-sdxl-1.0"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

controlnet = ControlNetModel.from_pretrained(
    controlnet_id,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.enable_model_cpu_offload()

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

prompt = "ultrarealistic shot of a furry blue bird"

canny_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/bird_canny.png")

controlnet_conditioning_scale = 0.5  # الموصى به للتعميم الجيد

image = pipe(
    prompt,
    image=canny_image,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    controlnet_conditioning_scale=controlnet_conditioning_scale,
    generator=torch.Generator(device=device).manual_seed(0),
).images[0]

grid_image = make_image_grid([canny_image, image], rows=1, cols=2)
```

![صورة توضيحية](https://github.com/jabir-zheng/TCD/raw/main/assets/controlnet_canny_tcd.png)

<Tip>
قد لا تعمل معلمات الاستنتاج في هذا المثال مع جميع الصور، لذلك نوصي بتجربة قيم مختلفة لمعلمات num_inference_steps و guidance_scale و controlnet_conditioning_scale و cross_attention_kwargs واختيار الأنسب.
</Tip>

<hfoption id="IP-Adapter">
يُظهر هذا المثال كيفية استخدام TCD-LoRA مع [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter/tree/main) و SDXL. يستخدم IP-Adapter ترميز الصور المُدرب مسبقًا لتعديل الصور المرجعية أثناء عملية التوليد.

```python
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image, make_image_grid

from ip_adapter import IPAdapterXL
from scheduling_tcd import TCDScheduler

device = "cuda"
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
tcd_lora_id = "h1t/TCD-SDXL-LoRA"

pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

pipe.load_lora_weights(tcd_lora_id)
pipe.fuse_lora()

ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

ref_image = load_image("https://raw.githubusercontent.com/tencent-ailab/IP-Adapter/main/assets/images/woman.png").resize((512, 512))

prompt = "best quality, high quality, wearing sunglasses"

image = ip_model.generate(
    pil_image=ref_image,
    prompt=prompt,
    scale=0.5,
    num_samples=1,
    num_inference_steps=4,
    guidance_scale=0,
    eta=0.3,
    seed=0,
)[0]

grid_image = make_image_grid([ref_image, image], rows=1, cols=2)
```

![صورة توضيحية](https://github.com/jabir-zheng/TCD/raw/main/assets/ip_adapter.png)

</hfoption>

<hfoption id="AnimateDiff">
يسمح [AnimateDiff] بتحريك الصور باستخدام نماذج Stable Diffusion. يمكن لـ TCD-LoRA تسريع العملية بشكل كبير دون المساس بجودة الصورة. تنتج جودة الرسوم المتحركة مع TCD-LoRA و AnimateDiff نتائج أكثر وضوحًا.

```python
import torch
from diffusers import MotionAdapter, AnimateDiffPipeline, DDIMScheduler
from scheduling_tcd import TCDScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5")
pipe = AnimateDiffPipeline.from_pretrained(
    "frankjoshua/toonyou_beta6",
    motion_adapter=adapter,
).to("cuda")

# قم بضبط TCDScheduler
pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

# تحميل TCD LoRA
pipe.load_lora_weights("h1t/TCD-SD15-LoRA", adapter_name="tcd")
pipe.load_lora_weights("guoyww/animatediff-motion-lora-zoom-in", weight_name="diffusion_pytorch_model.safetensors", adapter_name="motion-lora")

pipe.set_adapters(["tcd", "motion-lora"], adapter_weights=[1.0, 1.2])

prompt = "best quality, masterpiece, 1girl, looking at viewer, blurry background, upper body, contemporary, dress"
generator = torch.manual_seed(0)
frames = pipe(
    prompt=prompt,
    num_inference_steps=5,
    guidance_scale=0,
    cross_attention_kwargs={"scale": 1},
    num_frames=24,
    eta=0.3,
    generator=generator
).frames[0]
export_to_gif(frames, "animation.gif")
```

![صورة متحركة توضيحية](https://github.com/jabir-zheng/TCD/raw/main/assets/animation_example.gif)

</hfoption>

</hfoptions>