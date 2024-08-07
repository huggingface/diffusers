# T-GATE  

يعمل T-GATE على تسريع الاستنتاج لأنابيب [Stable Diffusion](../api/pipelines/stable_diffusion/overview) و [PixArt](../api/pipelines/pixart) و [Latency Consistency Model](../api/pipelines/latent_consistency_models.md) عن طريق تخطي حساب cross-attention بمجرد تقاربه. لا تتطلب هذه الطريقة أي تدريب إضافي ويمكنها تسريع الاستنتاج من 10-50%. T-GATE متوافق أيضًا مع طرق التحسين الأخرى مثل [DeepCache](./deepcache).  

قبل البدء، تأكد من تثبيت T-GATE.  

```bash
pip install tgate
pip install -U torch diffusers transformers accelerate DeepCache
```  

لاستخدام T-GATE مع خط أنابيب، يجب استخدام محملها المقابل.  

| خط الأنابيب | محمل T-GATE |  
| --- | --- |  
| PixArt | TgatePixArtLoader |  
| Stable Diffusion XL | TgateSDXLLoader |  
| Stable Diffusion XL + DeepCache | TgateSDXLDeepCacheLoader |  
| Stable Diffusion | TgateSDLoader |  
| Stable Diffusion + DeepCache | TgateSDDeepCacheLoader |  

بعد ذلك، قم بإنشاء `TgateLoader` باستخدام خط الأنابيب، وخطوة البوابة (خطوة الوقت لإيقاف حساب الانتباه المتقاطع)، وعدد خطوات الاستنتاج. ثم قم باستدعاء طريقة `tgate` على خط الأنابيب باستخدام موجه، وخطوة البوابة، وعدد خطوات الاستنتاج.  

دعونا نرى كيفية تمكين هذا لعدة خطوط أنابيب مختلفة.  

<hfoptions id="pipelines">
<hfoption id="PixArt">  

تسريع `PixArtAlphaPipeline` باستخدام T-GATE:  

```py
import torch
from diffusers import PixArtAlphaPipeline
from tgate import TgatePixArtLoader

pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)

gate_step = 8
inference_step = 25
pipe = TgatePixArtLoader(
pipe,
gate_step=gate_step,
num_inference_steps=inference_step,
).to("cuda")

image = pipe.tgate(
"An alpaca made of colorful building blocks, cyberpunk.",
gate_step=gate_step,
num_inference_steps=inference_step,
).images[0]
```

</hfoption>
<hfoption id="Stable Diffusion XL">  

تسريع `StableDiffusionXLPipeline` باستخدام T-GATE:  

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLLoader

pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

gate_step = 10
inference_step = 25
pipe = TgateSDXLLoader(
pipe,
gate_step=gate_step,
num_inference_steps=inference_step,
).to("cuda")

image = pipe.tgate(
"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
gate_step=gate_step,
num_inference_steps=inference_step
).images[0]
```

</hfoption>
<hfoption id="StableDiffusionXL with DeepCache">  

تسريع `StableDiffusionXLPipeline` باستخدام [DeepCache](https://github.com/horseee/DeepCache) و T-GATE:  

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLDeepCacheLoader

pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
variant="fp16",
use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

gate_step = 10
inference_step = 25
pipe = TgateSDXLDeepCacheLoader(
pipe,
cache_interval=3,
cache_branch_id=0,
).to("cuda")

image = pipe.tgate(
"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
gate_step=gate_step,
num_inference_steps=inference_step
).images[0]
```

</hfoption>
<hfoption id="Latent Consistency Model">  

تسريع `latent-consistency/lcm-sdxl` باستخدام T-GATE:  

```py
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import UNet2DConditionModel, LCMScheduler
from diffusers import DPMSolverMultistepScheduler
from tgate import TgateSDXLLoader

unet = UNet2DConditionModel.from_pretrained(
"latent-consistency/lcm-sdxl",
torch_dtype=torch.float16,
variant="fp16",
)
pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
unet=unet,
torch_dtype=torch.float16,
variant="fp16",
)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

gate_step = 1
inference_step = 4
pipe = TgateSDXLLoader(
pipe,
gate_step=gate_step,
num_inference_steps=inference_step,
lcm=True
).to("cuda")

image = pipe.tgate(
"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k.",
gate_step=gate_step,
num_inference_steps=inference_step
).images[0]
```

</hfoption>
</hfoptions>  

يدعم T-GATE أيضًا [`StableDiffusionPipeline`] و [PixArt-alpha/PixArt-LCM-XL-2-1024-MS](https://hf.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS).  
 ## المعايير  

| النموذج | MACs | المعلمات | الكمون | Zero-shot 10K-FID على MS-COCO |  
| --- | --- | --- | --- | --- |  
| SD-1.5 | 16.938T | 859.520M | 7.032s | 23.927 |  
| SD-1.5 w/ T-GATE | 9.875T | 815.557M | 4.313s | 20.789 |  
| SD-2.1 | 38.041T | 865.785M | 16.121s | 22.609 |  
| SD-2.1 w/ T-GATE | 22.208T | 815.433 M | 9.878s | 19.940 |  
| SD-XL | 149.438T | 2.570B | 53.187s | 24.628 |  
| SD-XL w/ T-GATE | 84.438T | 2.024B | 27.932s | 22.738 |  
| Pixart-Alpha | 107.031T | 611.350M | 61.502s | 38.669 |  
| Pixart-Alpha w/ T-GATE | 65.318T | 462.585M | 37.867s | 35.825 |  
| DeepCache (SD-XL) | 57.888T | - | 19.931s | 23.755 |  
| DeepCache w/ T-GATE | 43.868T | - | 14.666s | 23.999 |  
| LCM (SD-XL) | 11.955T | 2.570B | 3.805s | 25.044 |  
| LCM w/ T-GATE | 11.171T | 2.024B | 3.533s | 25.028 |  
| LCM (Pixart-Alpha) | 8.563T | 611.350M | 4.733s | 36.086 |  
| LCM w/ T-GATE | 7.623T | 462.585M | 4.543s | 37.048 |  

تم اختبار الكمون على NVIDIA 1080TI، وتم حساب MACs و Params باستخدام [calflops](https://github.com/MrYxJ/calculate-flops.pytorch)، وتم حساب FID باستخدام [PytorchFID](https://github.com/mseitzer/pytorch-fid).