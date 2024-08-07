# OpenVINO  

يوفر [Optimum](https://github.com/huggingface/optimum-intel) أنابيب Stable Diffusion المتوافقة مع OpenVINO لإجراء الاستدلال على مجموعة متنوعة من معالجات Intel (راجع [القائمة الكاملة](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html) للأجهزة المدعومة).  

يجب تثبيت حزمة 🤗 Optimum Intel مع خيار `--upgrade-strategy eager` للتأكد من استخدام [`optimum-intel`](https://github.com/huggingface/optimum-intel) لأحدث إصدار:  

```bash
pip install --upgrade-strategy eager optimum["openvino"]
```  

سيوضح هذا الدليل كيفية استخدام أنابيب Stable Diffusion و Stable Diffusion XL (SDXL) مع OpenVINO.

## Stable Diffusion  

لتحميل وتشغيل الاستدلال، استخدم [`~optimum.intel.OVStableDiffusionPipeline`]. إذا كنت تريد تحميل نموذج PyTorch وتحويله إلى تنسيق OpenVINO أثناء التنقل، قم بتعيين `export=True`:  

```python
from optimum.intel import OVStableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"
pipeline = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]

# Don't forget to save the exported model
pipeline.save_pretrained("openvino-sd-v1-5")
```  

لزيادة تسريع الاستدلال، قم بإعادة تشكيل النموذج بشكل ثابت. إذا قمت بتغيير أي معلمات مثل ارتفاع الإخراج أو عرضه، فستحتاج إلى إعادة تشكيل نموذجك بشكل ثابت مرة أخرى.  

```python
# Define the shapes related to the inputs and desired outputs
batch_size, num_images, height, width = 1, 1, 512, 512

# Statically reshape the model
pipeline.reshape(batch_size, height, width, num_images)
# Compile the model before inference
pipeline.compile()

image = pipeline(
prompt,
height=height,
width=width,
num_images_per_prompt=num_images,
).images[0]
```

يمكنك العثور على المزيد من الأمثلة في وثائق 🤗 Optimum [documentation](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion)، ويدعم Stable Diffusion للتحويل من نص إلى صورة، ومن صورة إلى صورة، وللتلوين.

## Stable Diffusion XL  

لتحميل وتشغيل الاستدلال مع SDXL، استخدم [`~optimum.intel.OVStableDiffusionXLPipeline`]:  

```python
from optimum.intel import OVStableDiffusionXLPipeline

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline = OVStableDiffusionXLPipeline.from_pretrained(model_id)
prompt = "sailing ship in storm by Rembrandt"
image = pipeline(prompt).images[0]
```  

لزيادة تسريع الاستدلال، [قم بإعادة تشكيل](#stable-diffusion) النموذج بشكل ثابت كما هو موضح في قسم Stable Diffusion.  

يمكنك العثور على المزيد من الأمثلة في وثائق 🤗 Optimum [documentation](https://huggingface.co/docs/optimum/intel/inference#stable-diffusion-xl)، ويدعم تشغيل SDXL في OpenVINO للتحويل من نص إلى صورة ومن صورة إلى صورة.