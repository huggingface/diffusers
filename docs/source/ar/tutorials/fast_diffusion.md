
# تسريع الاستنتاج لنماذج الانتشار من النص إلى الصورة

تعد نماذج الانتشار أبطأ من نظيراتها من GAN بسبب عملية الانتشار العكسي المتكررة والتسلسلية. هناك العديد من التقنيات التي يمكن أن تعالج هذا القيد مثل تقطير خطوة الوقت التدريجي (LCM LoRA)، وضغط النموذج (SSD-1B)، وإعادة استخدام الميزات المجاورة للمحو (DeepCache).

ومع ذلك، فأنت لست بحاجة إلى استخدام هذه التقنيات بالضرورة لتسريع الاستنتاج. باستخدام PyTorch 2 وحده، يمكنك تسريع وقت استجابة خط أنابيب الانتشار من النص إلى الصورة بمقدار 3 مرات. سيوضح هذا البرنامج التعليمي كيفية التطبيق التدريجي للتحسينات الموجودة في PyTorch 2 لتقليل وقت الاستنتاج. ستستخدم خط أنابيب Stable Diffusion XL (SDXL) في هذا البرنامج التعليمي، ولكن هذه التقنيات تنطبق على خطوط أنابيب الانتشار الأخرى من النص إلى الصورة أيضًا.

تأكد من استخدام أحدث إصدار من Diffusers:

```bash
pip install -U diffusers
```

ثم قم بترقية المكتبات المطلوبة الأخرى أيضًا:

```bash
pip install -U transformers accelerate peft
```

قم بتثبيت PyTorch الليلي للاستفادة من أحدث وأسرع النواة:

```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```


<Tip>

النتائج المبلغ عنها أدناه هي من 80 جيجا بايت 400 واط A100 مع معدل الساعة الخاص بها تعيين الحد الأقصى. <br>
إذا كنت مهتمًا برمز المعيار المرجعي الكامل، فقم بإلقاء نظرة على huggingface/diffusion-fast.

</Tip>

## خط الأساس

دعونا نبدأ بخط الأساس. تعطيل الدقة المخفضة ووظيفة [`scaled_dot_product_attention` (SDPA)](../optimization/torch2.0#scaled-dot-product-attention) التي يستخدمها Diffusers تلقائيًا:

```python
from diffusers import StableDiffusionXLPipeline

# Load the pipeline in full-precision and place its model components on CUDA.
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0"
).to("cuda")

# Run the attention ops without SDPA.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```
هذا الإعداد الافتراضي يستغرق 7.36 ثانية.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_0.png" width=500>
</div>

## bfloat16

تمكين أول تحسين، الدقة المخفضة أو بشكل أكثر تحديدًا bfloat16. هناك العديد من فوائد استخدام الدقة المخفضة:

* استخدام دقة رقمية مخفضة (مثل float16 أو bfloat16) للاستنتاج لا يؤثر على جودة التوليد ولكنه يحسن بشكل كبير وقت الاستجابة.
* تعتمد فوائد استخدام bfloat16 مقارنة بـ float16 على الأجهزة، ولكن تميل وحدات معالجة الرسومات (GPU) الحديثة إلى تفضيل bfloat16.
* bfloat16 أكثر مرونة عند استخدامه مع التكميم مقارنة بـ float16، ولكن الإصدارات الأحدث من مكتبة التكميم (torchao) التي استخدمناها لا تحتوي على مشكلات رقمية مع float16.

```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# Run the attention ops without SDPA.
pipe.unet.set_default_attn_processor()
pipe.vae.set_default_attn_processor()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

يقلل bfloat16 من وقت الاستجابة من 7.36 ثانية إلى 4.63 ثانية.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_1.png" width=500>
</div>

<Tip>

في تجاربنا اللاحقة باستخدام float16، لا تسبب الإصدارات الأحدث من torchao مشكلات رقمية من float16.

</Tip>

الق نظرة على دليل تسريع الاستنتاج لمعرفة المزيد حول تشغيل الاستنتاج بدقة مخفضة.

## SDPA

تعد كتل الاهتمام كثيفة الاستخدام لتشغيلها. ولكن مع وظيفة [`scaled_dot_product_attention`](../optimization/torch2.0#scaled-dot-product-attention) في PyTorch، فهي أكثر كفاءة بكثير. يتم استخدام هذه الوظيفة بشكل افتراضي في Diffusers لذلك لا تحتاج إلى إجراء أي تغييرات على التعليمات البرمجية.


```python
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```


يحسن المنتج النقطي المُمَيز وقت الاستجابة من 4.63 ثانية إلى 3.31 ثانية.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_2.png" width=500>
</div>

## torch.compile

تتضمن PyTorch 2 وظيفة "torch.compile" التي تستخدم نواة سريعة ومحسنة. في Diffusers، يتم عادةً تجميع UNet وVAE لأنهما أكثر الوحدات النمطية كثيفة الحساب. أولاً، قم بتكوين بعض أعلام المترجم (راجع القائمة الكاملة لمزيد من الخيارات):

```python
from diffusers import StableDiffusionXLPipeline
import torch

torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
```

من المهم أيضًا تغيير تخطيط الذاكرة لـ UNet وVAE إلى "channels_last" عند تجميعها لضمان السرعة القصوى.

```python
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
```

الآن قم بتجميع وإجراء الاستدلال:

```python
# Compile the UNet and VAE.
pipe.unet = torch.compile(pipe.unet, mode="max-autotune", fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# First call to `pipe` is slow, subsequent ones are faster.
image = pipe(prompt, num_inference_steps=30).images[0]
```

يقدم "torch.compile" أوضاعًا وخلفيات مختلفة. للحصول على أقصى سرعة للاستدلال، استخدم "max-autotune" لخلفية المُحث. يستخدم "max-autotune" رسومات CUDA ويحسن رسم التجميع خصيصًا للاتصال. تقلل رسومات CUDA بشكل كبير من النفقات العامة لتشغيل عمليات GPU من خلال استخدام آلية لتشغيل عمليات GPU متعددة من خلال عملية CPU واحدة.

يقلل استخدام الانتباه SDPA وتجميع كل من UNet وVAE وقت الاستجابة من 3.31 ثانية إلى 2.54 ثانية.
<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_3.png" width=500>
</div>

### منع كسور الرسم

يضمن تحديد "fullgraph=True" عدم وجود كسور في الرسم في النموذج الأساسي للاستفادة الكاملة من "torch.compile" دون أي تدهور في الأداء. بالنسبة لـ UNet وVAE، يعني ذلك تغيير طريقة الوصول إلى متغيرات الإرجاع.

```diff
- latents = unet(
-   latents, timestep=timestep, encoder_hidden_states=prompt_embeds
-).sample

+ latents = unet(
+   latents, timestep=timestep, encoder_hidden_states=prompt_embeds, return_dict=False
+)[0]
```

### إزالة مزامنة GPU بعد التجميع

خلال عملية الانتشار العكسي المتكررة، يتم استدعاء وظيفة "step()" على المخطط كل مرة بعد أن يتنبأ المحو بالتشفيرات المخفية الأقل ضوضاءً. داخل "step()"، يتم فهرسة متغير "sigmas" الذي عندما يتم وضعه على GPU، يتسبب في مزامنة الاتصال بين CPU وGPU. وهذا يؤدي إلى حدوث تأخير ويصبح أكثر وضوحًا عندما يكون المحو قد تم تجميعه بالفعل.

ولكن إذا كانت مصفوفة "sigmas" دائمًا على وحدة المعالجة المركزية (CPU) [stays on the CPU](https://github.com/huggingface/diffusers/blob/35a969d297cba69110d175ee79c59312b9f49e1e/src/diffusers/schedulers/scheduling_euler_discrete.py#L240)، فلا يحدث مزامنة CPU وGPU ولا تحصل على أي تأخير. بشكل عام، يجب أن يكون أي مزامنة للاتصال بين CPU وGPU غير موجود أو في حده الأدنى لأنه يمكن أن يؤثر على وقت الاستنتاج.

## دمج مصفوفات الإسقاط في كتلة الاهتمام

يستخدم UNet و VAE في SDXL كتلًا شبيهة بـ Transformer والتي تتكون من كتل اهتمام وكتل للأمام.

في كتلة اهتمام، يتم إسقاط الإدخال في ثلاثة مساحات فرعية باستخدام ثلاث مصفوفات إسقاط مختلفة - Q و K و V. يتم تنفيذ هذه الإسقاطات بشكل منفصل على الإدخال. ولكن يمكننا دمج مصفوفات الإسقاط أفقيًا في مصفوفة واحدة وإجراء الإسقاط في خطوة واحدة. وهذا يزيد من حجم ضربات المصفوفة لإسقاطات الإدخال ويحسن تأثير التكميم.

يمكنك دمج مصفوفات الإسقاط باستخدام سطر واحد فقط من التعليمات البرمجية:

```python
pipe.fuse_qkv_projections()
```

هذا يوفر تحسنًا طفيفًا من 2.54 ثانية إلى 2.52 ثانية.

<div class="flex justify-center">
    <img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_4.png" width=500>
</div>

<Tip warning={true}>

الدعم [`~StableDiffusionXLPipeline.fuse_qkv_projections`] محدود وتجريبي. فهو غير متاح للعديد من خطوط أنابيب التشتت غير المستقرة مثل [Kandinsky] (../using-diffusers/kandinsky). يمكنك الرجوع إلى هذا [PR] (https://github.com/huggingface/diffusers/pull/6179) للحصول على فكرة حول كيفية تمكين هذا الأمر لخطوط الأنابيب الأخرى.

</Tip>

## التكميم الديناميكي

يمكنك أيضًا استخدام مكتبة PyTorch الكمية فائقة الخفة، [torchao](https://github.com/pytorch-labs/ao) (رمز SHA للالتزام `54bcd5a10d0abbe7b0c045052029257099f83fd9`)، لتطبيق [التكميم الديناميكي int8](https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html) على UNet و VAE. يضيف التكميم تكاليف تحويل إضافية إلى النموذج الذي نأمل أن يعوضه ضربات المصفوفة الأسرع (التكميم الديناميكي). إذا كانت ضربات المصفوفة صغيرة جدًا، فقد تؤدي هذه التقنيات إلى تدهور الأداء.

أولاً، قم بتكوين جميع علامات المترجم:

```python
from diffusers import StableDiffusionXLPipeline
import torch

# لاحظ العلمين الجديدين في النهاية.
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True
```

لا تستفيد بعض الطبقات الخطية في UNet و VAE من التكميم الديناميكي int8. يمكنك تصفية تلك الطبقات باستخدام [`dynamic_quant_filter_fn`](https://github.com/huggingface/diffusion-fast/blob/0f169640b1db106fe6a479f78c1ed3bfaeba3386/utils/pipeline_utils.py#L16) الموضح أدناه.

```python
def dynamic_quant_filter_fn(mod, *args):
return (
isinstance(mod, torch.nn.Linear)
and mod.in_features > 16
and (mod.in_features, mod.out_features)
not in [
(1280, 640),
(1920, 1280),
(1920, 640),
(2048, 1280),
(2048, 2560),
(2560, 1280),
(256, 128),
(2816, 1280),
(320, 640),
(512, 1536),
(512, 256),
(512, 512),
(640, 1280),
(640, 1920),
(640, 320),
(640, 5120),
(640, 640),
(960, 320),
(960, 640),
]
)


def conv_filter_fn(mod, *args):
return (
isinstance(mod, torch.nn.Conv2d) and mod.kernel_size == (1, 1) and 128 in [mod.in_channels, mod.out_channels]
)
```

أخيرًا، قم بتطبيق جميع التحسينات التي تمت مناقشتها حتى الآن:

```python
# SDPA + bfloat16.
pipe = StableDiffusionXLPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.bfloat16
).to("cuda")

# دمج مصفوفات إسقاط الاهتمام.
pipe.fuse_qkv_projections()

# تغيير تخطيط الذاكرة.
pipe.unet.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
```

نظرًا لأن التكميم الديناميكي يقتصر فقط على الطبقات الخطية، فقم بتحويل طبقات التحويل النقطي المناسبة إلى طبقات خطية لتعظيم الفائدة.

```python
from torchao import swap_conv2d_1x1_to_linear

swap_conv2d_1x1_to_linear(pipe.unet، conv_filter_fn)
swap_conv2d_1x1_to_linear(pipe.vae، conv_filter_fn)
```

تطبيق التكميم الديناميكي:

```python
from torchao import apply_dynamic_quant

apply_dynamic_quant(pipe.unet, dynamic_quant_filter_fn)
apply_dynamic_quant(pipe.vae, dynamic_quant_filter_fn)
```

أخيرًا، قم بالترجمة وإجراء الاستدلال:

```python
pipe.unet = torch.compile(pipe.unet، mode="max-autotune"، fullgraph=True)
pipe.vae.decode = torch.compile(pipe.vae.decode، mode="max-autotune"، fullgraph=True)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt, num_inference_steps=30).images[0]
```

يحسن تطبيق التكميم الديناميكي الكمون من 2.52 ثانية إلى 2.43 ثانية.

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/progressive-acceleration-sdxl/SDXL%2C_Batch_Size%3A_1%2C_Steps%3A_30_5.png" width=500>
</div>