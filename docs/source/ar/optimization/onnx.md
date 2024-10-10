# ONNX Runtime

يوفر 🤗 [Optimum](https://github.com/huggingface/optimum) خط أنابيب Stable Diffusion متوافق مع ONNX Runtime. ستحتاج إلى تثبيت 🤗 Optimum باستخدام الأمر التالي لدعم ONNX Runtime:

هذا الدليل سيوضح لك كيفية استخدام أنابيب Stable Diffusion و Stable Diffusion XL (SDXL) مع ONNX Runtime.

## Stable Diffusion

لتحميل وتشغيل الاستنتاج، استخدم ["~optimum.onnxruntime.ORTStableDiffusionPipeline"]. إذا كنت تريد تحميل نموذج PyTorch وتحويله إلى تنسيق ONNX أثناء التنقل، قم بتعيين "export=True":

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
pipeline.save_pretrained("./onnx-stable-diffusion-v1-5")
```

<Tip warning={true}>

يبدو أن إنشاء دفعات متعددة من المطالبات يستهلك الكثير من الذاكرة. بينما نبحث في الأمر، قد تحتاج إلى التكرار بدلاً من الدفعات.

</Tip>

لتصدير خط الأنابيب بتنسيق ONNX دون اتصال واستخدامه لاحقًا للاستنتاج، استخدم أمر ["optimum-cli export"](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli):

```bash
optimum-cli export onnx --model runwayml/stable-diffusion-v1-5 sd_v15_onnx/
```

ثم لتنفيذ الاستنتاج (لا يلزم تحديد "export=True" مرة أخرى):

```python
from optimum.onnxruntime import ORTStableDiffusionPipeline

model_id = "sd_v15_onnx"
pipeline = ORTStableDiffusionPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/optimum/documentation-images/resolve/main/onnxruntime/stable_diffusion_v1_5_ort_sail_boat.png">
</div>

يمكنك العثور على المزيد من الأمثلة في وثائق 🤗 Optimum [documentation](https://huggingface.co/docs/optimum/)، ويتم دعم Stable Diffusion للصور النصية والصور والصور.

## Stable Diffusion XL

لتحميل وتشغيل الاستنتاج مع SDXL، استخدم ["~optimum.onnxruntime.ORTStableDiffusionXLPipeline"]:

```python
from optimum.onnxruntime import ORTStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = ORTStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Leonardo da Vinci"
image = pipeline(prompt).images[0]
```

لتصدير خط الأنابيب بتنسيق ONNX واستخدامه لاحقًا للاستنتاج، استخدم أمر ["optimum-cli export"](https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model#exporting-a-model-to-onnx-using-the-cli):

```bash
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task stable-diffusion-xl sd_xl_onnx/
```

يدعم SDXL بتنسيق ONNX للصور النصية والصور.