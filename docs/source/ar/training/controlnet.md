# ControlNet

تعد نماذج ControlNet محولات يتم تدريبها أعلى نموذج آخر مُدرب مسبقًا. يتيح ذلك درجة أكبر من التحكم في إنشاء الصور عن طريق ضبط النموذج باستخدام صورة إدخال إضافية. يمكن أن تكون صورة الإدخال صورة Canny Edge أو خريطة عمق أو وضع إنسان، والكثير غير ذلك.

إذا كنت تتدرب على وحدة معالجة رسومات (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين معلمات "gradient_checkpointing" و "gradient_accumulation_steps" و "mixed_precision" في أمر التدريب. يمكنك أيضًا تقليل البصمة الخاصة بك باستخدام انتباه فعال للذاكرة مع [xFormers](../optimization/xformers). يتم أيضًا دعم التدريب JAX/Flax للتدريب الفعال على وحدات معالجة الرسومات (TPUs) ووحدات معالجة الرسومات (GPUs)، ولكنه لا يدعم نقاط تفتيش التدرج أو xFormers. يجب أن يكون لديك وحدة معالجة رسومات (GPU) بها ذاكرة وصول عشوائي (RAM) أكبر من 30 جيجابايت إذا كنت تريد التدريب بشكل أسرع باستخدام Flax.

سيتناول هذا الدليل برنامج النص البرمجي [train_controlnet.py](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) التدريبي لمساعدتك على التعرف عليه، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

قبل تشغيل البرنامج النصي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

بعد ذلك، انتقل إلى مجلد المثال الذي يحتوي على البرنامج النصي للتدريب وقم بتثبيت التبعيات المطلوبة للبرنامج النصي الذي تستخدمه:

<hfoptions id="installation">
<hfoption id="PyTorch">

```bash
cd examples/controlnet
pip install -r requirements.txt
```

</hfoption>
<hfoption id="Flax">

إذا كان لديك حق الوصول إلى وحدة معالجة الرسومات (TPU)، فسينفذ برنامج النص التدريبي Flax بشكل أسرع! دعنا نقوم بتشغيل البرنامج النصي التدريبي على [Google Cloud TPU VM](https://cloud.google.com/tpu/docs/run-calculation-jax). قم بإنشاء وحدة معالجة رسومات (TPU) افتراضية واحدة من نوع v4-8 وقم بالاتصال بها:

```bash
ZONE=us-central2-b
TPU_TYPE=v4-8
VM_NAME=hg_flax

gcloud alpha compute tpus tpu-vm create $VM_NAME \
--zone $ZONE \
--accelerator-type $TPU_TYPE \
--version  tpu-vm-v4-base

gcloud alpha compute tpus tpu-vm ssh $VM_NAME --zone $ZONE -- \
```

قم بتثبيت JAX 0.4.5:

```bash
pip install "jax[tpu]==0.4.5" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

بعد ذلك، قم بتثبيت التبعيات المطلوبة لبرنامج النص التدريبي Flax:

```bash
cd examples/controlnet
pip install -r requirements_flax.txt
```

</hfoption>
</hfoptions>

<Tip>

🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات معالجة الرسومات (GPU) / وحدات معالجة الرسومات (TPU) متعددة أو مع الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على الأجهزة والبيئة الخاصة بك. الق نظرة على جولة 🤗 Accelerate [السريعة](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع البرنامج النصي التدريبي.

<Tip>

تسلط الأقسام التالية الضوء على أجزاء من البرنامج النصي التدريبي المهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي [النصي](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet.py) ودعنا نعرف إذا كان لديك أي أسئلة أو مخاوف.

</Tip>

## معلمات البرنامج النصي

يوفر البرنامج النصي التدريبي العديد من المعلمات لمساعدتك في تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L231). توفر هذه الدالة قيمًا افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت تريد ذلك.

على سبيل المثال، لزيادة سرعة التدريب باستخدام الدقة المختلطة بتنسيق fp16، أضف معلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_controlnet.py \
--mixed_precision="fp16"
```

تم وصف العديد من المعلمات الأساسية والمهمة في دليل التدريب [Text-to-image](text2image#script-parameters)، لذلك يركز هذا الدليل فقط على المعلمات ذات الصلة بـ ControlNet:

- `--max_train_samples`: عدد عينات التدريب؛ يمكن تقليل هذا للتسريع التدريب، ولكن إذا كنت تريد بث مجموعات بيانات كبيرة جدًا، فستحتاج إلى تضمين هذه المعلمة ومعلمة `--streaming` في أمر التدريب

- `--gradient_accumulation_steps`: عدد خطوات التحديث لتراكمها قبل التمرير الخلفي؛ يسمح لك ذلك بالتدريب باستخدام حجم دفعة أكبر مما يمكن لذاكرة وحدة معالجة الرسومات (GPU) التعامل معه عادةً

### وزن الحد الأدنى لـ SNR

يمكن أن تساعد استراتيجية وزن الحد الأدنى لـ [SNR](https://huggingface.co/papers/2303.09556) في التدريب عن طريق إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم البرنامج النصي التدريبي التنبؤ بـ `epsilon` (الضوضاء) أو `v_prediction`، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. استراتيجية الترجيح هذه مدعومة فقط بواسطة PyTorch وغير متوفرة في البرنامج النصي التدريبي Flax.

أضف معلمة `--snr_gamma` وقم بتعيينها على القيمة الموصى بها 5.0:

```bash
accelerate launch train_controlnet.py \
--snr_gamma=5.0
```

## البرنامج النصي للتدريب

كما هو الحال مع معلمات البرنامج النصي، يتم توفير نظرة عامة عامة على البرنامج النصي التدريبي في دليل التدريب [Text-to-image](text2image#training-script). بدلاً من ذلك، يلقي هذا الدليل نظرة على الأجزاء ذات الصلة من البرنامج النصي ControlNet.

يحتوي البرنامج النصي التدريبي على دالة [`make_train_dataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L582) لمعالجة مجموعة البيانات مسبقًا باستخدام تحويلات الصور ونمذجة التعليقات التوضيحية. سترى أنه بالإضافة إلى نمذجة التعليقات التوضيحية للصور وتحويلاتها المعتادة، يتضمن البرنامج النصي أيضًا تحويلات لصورة الضبط.

<Tip>

إذا كنت تقوم ببث مجموعة بيانات على وحدة معالجة الرسومات (TPU)، فقد يتم الحد من الأداء بواسطة مكتبة مجموعات البيانات 🤗 التي لم يتم تحسينها للصور. لضمان أقصى قدر من الإنتاجية، يُنصح باستكشاف تنسيقات مجموعات البيانات الأخرى مثل [WebDataset](https://webdataset.github.io/webdataset/) و [TorchData](https://github.com/pytorch/data) و [TensorFlow Datasets](https://www.tensorflow.org/datasets/tfless_tfds).

</Tip>

```py
conditioning_image_transforms = transforms.Compose(
[
transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
transforms.CenterCrop(args.resolution),
transforms.ToTensor(),
]
)
```

ضمن دالة [`main()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L713)، ستجد التعليمات البرمجية لتحميل برنامج الترميز والنص ومخطط التدرج والنماذج. هذا هو المكان الذي يتم فيه تحميل نموذج ControlNet إما من الأوزان الموجودة أو يتم تهيئته بشكل عشوائي من UNet:

```py
if args.controlnet_model_name_or_path:
logger.info("Loading existing controlnet weights")
controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
else:
logger.info("Initializing controlnet weights from unet")
controlnet = ControlNetModel.from_unet(unet)
```

يتم إعداد [المحسن](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L871) لتحديث معلمات ControlNet:

```py
params_to_optimize = controlnet.parameters()
optimizer = optimizer_class(
params_to_optimize,
lr=args.learning_rate,
betas=(args.adam_beta1, args.adam_beta2),
weight_decay=args.adam_weight_decay,
eps=args.adam_epsilon,
)
```

أخيرًا، في [حلقة التدريب](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/controlnet/train_controlnet.py#L943)، يتم تمرير تضمين النص والصورة الشرطية إلى الكتل السفلية والمتوسطة لنموذج ControlNet:

```py
encoder_hidden_states = text_encoder(batch["input_ids"])[0]
controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

down_block_res_samples, mid_block_res_sample = controlnet(
noisy_latents,
timesteps,
encoder_hidden_states=encoder_hidden_states,
controlnet_cond=controlnet_image,
return_dict=False,
)
```

إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [فهم الأنابيب والنماذج والمخططات](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة التشويش.
## تشغيل السكربت

الآن أنت مستعد لتشغيل سكربت التدريب! 🚀
يستخدم هذا الدليل مجموعة البيانات [fusing/fill50k](https://huggingface.co/datasets/fusing/fill50k)، ولكن تذكر أنه يمكنك إنشاء واستخدام مجموعة البيانات الخاصة بك إذا أردت (راجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

قم بتعيين متغير البيئة `MODEL_NAME` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه.

قم بتنزيل الصور التالية لتكييف التدريب الخاص بك:

```bash
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
```

هناك شيء واحد قبل إطلاق السكربت! اعتمادًا على وحدة معالجة الرسوميات (GPU) التي لديك، قد تحتاج إلى تمكين بعض التحسينات لتدريب ControlNet. يتطلب التكوين الافتراضي في هذا السكربت حوالي 38 جيجابايت من ذاكرة الوصول العشوائي للرسومات (VRAM). إذا كنت تقوم بالتدريب على أكثر من وحدة معالجة رسومات واحدة، فقم بإضافة المعلمة `--multi_gpu` إلى أمر `accelerate launch`.

<hfoptions id="gpu-select">

<hfoption id="16GB">

على وحدة معالجة رسومات بسعة 16 جيجابايت، يمكنك استخدام محسن bitsandbytes 8-bit ونقاط تفتيش التدرج لتسريع عملية التدريب. قم بتثبيت bitsandbytes:

```py
pip install bitsandbytes
```

بعد ذلك، أضف المعلمة التالية إلى أمر التدريب الخاص بك:

```bash
accelerate launch train_controlnet.py \
--gradient_checkpointing \
--use_8bit_adam \
```

</hfoption>

<hfoption id="12GB">

على وحدة معالجة رسومات بسعة 12 جيجابايت، ستحتاج إلى محسن bitsandbytes 8-bit، ونقاط تفتيش التدرج، وxFormers، وتعيين التدرجات إلى `None` بدلاً من الصفر لتقليل استخدام الذاكرة.

```bash
accelerate launch train_controlnet.py \
--use_8bit_adam \
--gradient_checkpointing \
--enable_xformers_memory_efficient_attention \
--set_grads_to_none \
```

</hfoption>

<hfoption id="8GB">

على وحدة معالجة رسومات بسعة 8 جيجابايت، ستحتاج إلى استخدام [DeepSpeed](https://www.deepspeed.ai/) لنقل بعض المصفوفات من ذاكرة الوصول العشوائي للرسومات (VRAM) إلى وحدة المعالجة المركزية (CPU) أو NVME للسماح بالتدريب باستخدام ذاكرة GPU أقل.

قم بتشغيل الأمر التالي لتكوين بيئة 🤗 Accelerate الخاصة بك:

```bash
accelerate config
```

أثناء التكوين، تأكد من أنك تريد استخدام DeepSpeed stage 2. الآن يجب أن يكون من الممكن التدريب على أقل من 8 جيجابايت من ذاكرة الوصول العشوائي للرسومات من خلال الجمع بين DeepSpeed stage 2، والدقة المختلطة fp16، ونقل معلمات النموذج وحالة المحسن إلى وحدة المعالجة المركزية. تتمثل السلبية في أن هذا يتطلب المزيد من ذاكرة الوصول العشوائي للنظام (حوالي 25 جيجابايت). راجع وثائق DeepSpeed للحصول على خيارات تكوين إضافية. يجب أن يبدو ملف التكوين الخاص بك كما يلي:

```bash
compute_environment: LOCAL_MACHINE
deepspeed_config:
gradient_accumulation_steps: 4
offload_optimizer_device: cpu
offload_param_device: cpu
zero3_init_flag: false
zero_stage: 2
distributed_type: DEEPSPEED
```

يجب عليك أيضًا تغيير محسن Adam الافتراضي إلى إصدار DeepSpeed المحسن من Adam [`deepspeed.ops.adam.DeepSpeedCPUAdam`](https://deepspeed.readthedocs.io/en/latest/optimizers.html#adam-cpu) للحصول على تسريع كبير. يتطلب تمكين `DeepSpeedCPUAdam` أن يكون إصدار CUDA في نظامك مطابقًا للإصدار المثبت مع PyTorch.

لا يبدو أن محسنات 8-bit bitsandbytes متوافقة مع DeepSpeed في الوقت الحالي.

هذا كل شيء! لا تحتاج إلى إضافة أي معلمات إضافية إلى أمر التدريب الخاص بك.

</hfoption>

</hfoptions>

<hfoptions id="training-inference">

<hfoption id="PyTorch">

```bash
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/save/model"

accelerate launch train_controlnet.py \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=fusing/fill50k \
--resolution=512 \
--learning_rate=1e-5 \
--validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--push_to_hub
```

</hfoption>

<hfoption id="Flax">

مع Flax، يمكنك [تحليل التعليمات البرمجية](https://jax.readthedocs.io/en/latest/profiling.html) الخاصة بك عن طريق إضافة المعلمة `--profile_steps==5` إلى أمر التدريب الخاص بك. قم بتثبيت برنامج Tensorboard profile plugin:

```bash
pip install tensorflow tensorboard-plugin-profile
tensorboard --logdir runs/fill-circle-100steps-20230411_165612/
```

بعد ذلك، يمكنك فحص الملف الشخصي في [http://localhost:6006/#profile](http://localhost:6006/#profile).

<Tip warning={true}>

إذا واجهتك صراعات الإصدار مع المكون الإضافي، فحاول إلغاء تثبيت جميع إصدارات TensorFlow وTensorboard وإعادة تثبيتها. لا تزال وظيفة التصحيح في المكون الإضافي للملف الشخصي تجريبية، وليست جميع وجهات النظر تعمل بشكل كامل. يقوم `trace_viewer` بقطع الأحداث بعد 1M، مما قد يؤدي إلى فقدان جميع آثار الجهاز الخاصة بك إذا قمت، على سبيل المثال، بتصحيح خطوة التجميع عن طريق الخطأ.

</Tip>

```bash
python3 train_controlnet_flax.py \
--pretrained_model_name_or_path=$MODEL_DIR \
--output_dir=$OUTPUT_DIR \
--dataset_name=fusing/fill50k \
--resolution=512 \
--learning_rate=1e-5 \
--validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
--validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
--validation_steps=1000 \
--train_batch_size=2 \
--revision="non-ema" \
--from_pt \
--report_to="wandb" \
--tracker_project_name=$HUB_MODEL_ID \
--num_train_epochs=11 \
--push_to_hub \
--hub_model_id=$HUB_MODEL_ID
```

</hfoption>

</hfoptions>

بمجرد اكتمال التدريب، يمكنك استخدام النموذج الذي تم تدريبه حديثًا للاستنتاج!

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained("path/to/controlnet", torch_dtype=torch.float16)
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
"path/to/base/model", controlnet=controlnet, torch_dtype=torch.float16
).to("cuda")

control_image = load_image("./conditioning_image_1.png")
prompt = "pale golden rod circle with old lace background"

generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=20, generator=generator, image=control_image).images[0]
image.save("./output.png")
```

## Stable Diffusion XL

Stable Diffusion XL (SDXL) هو نموذج قوي للصور النصية ينشئ صورًا عالية الدقة، ويضيف مشفر نص ثانٍ إلى تصميمه. استخدم سكربت [`train_controlnet_sdxl.py`](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/train_controlnet_sdxl.py) لتدريب محول ControlNet لنموذج SDXL.

يناقش دليل [SDXL training](sdxl) تفاصيل سكربت التدريب SDXL.

## الخطوات التالية

تهانينا على تدريب ControlNet الخاص بك! لمعرفة المزيد عن كيفية استخدام النموذج الجديد، قد تكون الأدلة التالية مفيدة:

- تعرف على كيفية [استخدام ControlNet](../using-diffusers/controlnet) للاستدلال على مجموعة متنوعة من المهام.