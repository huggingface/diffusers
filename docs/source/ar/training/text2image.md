# النص إلى الصورة

<Tip warning={true}>

يعد نص النص إلى الصورة تجريبيًا، ومن السهل أن يصبح مفرطًا في التكيف وأن يواجه مشكلات مثل النسيان الكارثي. جرّب استكشاف معلمات مختلفة للحصول على أفضل النتائج في مجموعة بياناتك.

</Tip>

تُستخدم نماذج النص إلى الصورة مثل Stable Diffusion في توليد الصور بناءً على موجه نصي. يمكن أن يكون تدريب النموذج مرهقًا لأجهزتك، ولكن إذا قمت بتمكين "gradient_checkpointing" و"mixed_precision"، فمن الممكن تدريب نموذج على وحدة GPU واحدة بسعة 24 جيجابايت. إذا كنت تتدرب باستخدام أحجام دفعات أكبر أو تريد التدريب بشكل أسرع، فمن الأفضل استخدام وحدات GPU التي تحتوي على أكثر من 30 جيجابايت من الذاكرة. يمكنك تقليل البصمة الخاصة بك عن طريق تمكين الاهتمام بكفاءة الذاكرة مع [xFormers](../optimization/xformers). يتم أيضًا دعم التدريب JAX/Flax للتدريب الفعال على وحدات TPUs وGPUs، ولكنه لا يدعم نقطة تفتيش التدرج أو تراكم التدرج أو xFormers. يوصى باستخدام وحدة GPU بسعة 30 جيجابايت على الأقل أو وحدة TPU v3 للتدريب باستخدام Flax.

سيتناول هذا الدليل برنامج النص البرمجي [train_text_to_image.py](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py) لمساعدتك على التعرف عليه، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

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
cd examples/text_to_image
pip install -r requirements.txt
```

</hfoption>

<hfoption id="Flax">

```bash
cd examples/text_to_image
pip install -r requirements_flax.txt
```

</hfoption>

</hfoptions>

<Tip>

🤗 Accelerate هي مكتبة للمساعدة في التدريب على وحدات GPU/TPUs متعددة أو باستخدام الدقة المختلطة. سيقوم تلقائيًا بتكوين إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. الق نظرة على 🤗 تسريع [جولة سريعة](https://huggingface.co/docs/accelerate/quicktour) لمعرفة المزيد.

</Tip>

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر الملاحظات، فيمكنك استخدام ما يلي:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة البيانات الخاصة بك، فراجع دليل [إنشاء مجموعة بيانات للتدريب](create_dataset) لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع برنامج النص البرمجي للتدريب.

## معلمات البرنامج النصي

<Tip>

تسلط الأقسام التالية الضوء على أجزاء من برنامج النص البرمجي للتدريب والتي تُعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب البرنامج النصي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فلا تتردد في قراءة البرنامج النصي [النصي](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py) وأخبرنا إذا كان لديك أي أسئلة أو مخاوف.

</Tip>

يوفر برنامج النص البرمجي للتدريب العديد من المعلمات لمساعدتك في تخصيص عملية تشغيل التدريب. يمكن العثور على جميع المعلمات ووصفاتها في دالة [`parse_args()`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L193). توفر هذه الدالة قيم افتراضية لكل معلمة، مثل حجم دفعة التدريب ومعدل التعلم، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا كنت تريد ذلك.

على سبيل المثال، لزيادة سرعة التدريب باستخدام الدقة المختلطة بتنسيق fp16، أضف معلمة `--mixed_precision` إلى أمر التدريب:

```bash
accelerate launch train_text_to_image.py \
--mixed_precision="fp16"
```

تشمل بعض المعلمات الأساسية ما يلي:

- `--pretrained_model_name_or_path`: اسم النموذج على Hub أو مسار محلي للنموذج المدرب مسبقًا
- `--dataset_name`: اسم مجموعة البيانات على Hub أو مسار محلي لمجموعة البيانات التي سيتم التدريب عليها
- `--image_column`: اسم عمود الصورة في مجموعة البيانات التي سيتم التدريب عليها
- `--caption_column`: اسم عمود النص في مجموعة البيانات التي سيتم التدريب عليها
- `--output_dir`: المكان الذي سيتم فيه حفظ النموذج المدرب
- `--push_to_hub`: ما إذا كان سيتم دفع النموذج المدرب إلى Hub
- `--checkpointing_steps`: تكرار حفظ نقطة تفتيش أثناء تدريب النموذج؛ هذا مفيد إذا تم مقاطعة التدريب لسبب ما، فيمكنك الاستمرار في التدريب من تلك النقطة عن طريق إضافة `--resume_from_checkpoint` إلى أمر التدريب الخاص بك

### وزن SNR الأدنى

يمكن أن تساعد استراتيجية وزن [Min-SNR](https://huggingface.co/papers/2303.09556) في التدريب عن طريق إعادة توازن الخسارة لتحقيق تقارب أسرع. يدعم برنامج النص البرمجي للتدريب التنبؤ بـ `epsilon` (noise) أو `v_prediction`، ولكن Min-SNR متوافق مع كلا نوعي التنبؤ. استراتيجية الترجيح هذه مدعومة فقط بواسطة PyTorch وغير متوفرة في برنامج النص البرمجي للتدريب Flax.

أضف معلمة `--snr_gamma` وقم بتعيينها على القيمة الموصى بها 5.0:

```bash
accelerate launch train_text_to_image.py \
--snr_gamma=5.0
```

يمكنك مقارنة أسطح الخسارة لقيم مختلفة من `snr_gamma` في هذا التقرير [Weights and Biases](https://wandb.ai/sayakpaul/text2image-finetune-minsnr). بالنسبة لمجموعات البيانات الأصغر، قد لا تكون تأثيرات Min-SNR واضحة مثل مجموعات البيانات الأكبر.

## برنامج النص البرمجي للتدريب

يمكن العثور على رمز معالجة البيانات السابق للتدريب وحلقة التدريب في دالة [`main()`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L490). إذا كنت بحاجة إلى تكييف برنامج النص البرمجي للتدريب، فهذا هو المكان الذي ستحتاج إلى إجراء تغييراتك فيه.

يبدأ برنامج النص البرمجي `train_text_to_image` عن طريق [تحميل جدول](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L543) ومميز. يمكنك اختيار استخدام جدول زمني مختلف هنا إذا أردت:

```py
noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(
args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
)
```

بعد ذلك، يقوم البرنامج النصي [بتحميل نموذج UNet](https://github.com/huggingface/diffusers/blob/8959c5bsubNav>

```py
load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
model.register_to_config(**load_model.config)

model.load_state_dict(load_model.state_dict())
```

بعد ذلك، تحتاج أعمدة النص والصورة في مجموعة البيانات إلى معالجتها مسبقًا. تتولى دالة [`tokenize_captions`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L724) التعامل مع توكينز المدخلات، وتحدد دالة [`train_transforms`](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L742) نوع التحويلات التي يجب تطبيقها على الصورة. يتم تجميع كلتا الدالتين في `preprocess_train`:

```py
def preprocess_train(examples):
images = [image.convert("RGB") for image in examples[image_column]]
examples["pixel_values"] = [train_transforms(image) for image in images]
examples["input_ids"] = tokenize_captions(examples)
return examples
```

أخيرًا، تتولى [حلقة التدريب](https://github.com/huggingface/diffusers/blob/8959c5b9dec1c94d6ba482c94a58d2215c5fd026/examples/text_to_image/train_text_to_image.py#L878) كل شيء آخر. يقوم بترميز الصور في مساحة خفية، وإضافة ضوضاء إلى المخفيين، وحساب تضمين النص للشرط، وتحديث معلمات النموذج، وحفظ النموذج ودفعه إلى Hub. إذا كنت تريد معرفة المزيد حول كيفية عمل حلقة التدريب، فراجع البرنامج التعليمي [Understanding pipelines, models and schedulers](../using-diffusers/write_own_pipeline) الذي يكسر النمط الأساسي لعملية إزالة الضوضاء.

## إطلاق البرنامج النصي

بمجرد إجراء جميع التغييرات أو كنت راضيًا عن التكوين الافتراضي، فأنت مستعد لإطلاق برنامج النص البرمجي للتدريب! 🚀

<hfoptions id="training-inference">

<hfoption id="PyTorch">

دعونا نتدرب على مجموعة بيانات [Naruto BLIP captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) لإنشاء شخصيات ناروتو الخاصة بك. قم بتعيين متغيرات البيئة `MODEL_NAME` و`dataset_name` إلى النموذج ومجموعة البيانات (إما من Hub أو مسار محلي). إذا كنت تتدرب على أكثر من وحدة GPU واحدة، فأضف معلمة `--multi_gpu` إلى أمر `accelerate launch`.

<Tip>

لتدريب مجموعة بيانات محلية، قم بتعيين متغيرات البيئة `TRAIN_DIR` و`OUTPUT_DIR` إلى مسار مجموعة البيانات والمكان الذي سيتم فيه حفظ النموذج.

</Tip>

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

accelerate launch --mixed_precision="fp16"  train_text_to_image.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$dataset_name \
--use_ema \
--resolution=512 --center_crop --random_flip \
--train_batch_size=1 \
--gradient_accumulation_steps=4 \
--gradient_checkpointing \
--max_train_steps=15000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--enable_xformers_memory_efficient_attention
--lr_scheduler="constant" --lr_warmup_steps=0 \
--output_dir="sd-naruto-model" \
--push_to_hub
```

</hfoption>

<hfoption id="Flax">

يمكن أن يكون التدريب باستخدام Flax أسرع على وحدات TPUs وGPUs بفضل [@duongna211](https://github.com/duongna21). Flax أكثر كفاءة على وحدة TPU، ولكن أداء وحدة GPU رائع أيضًا.

قم بتعيين متغيرات البيئة `MODEL_NAME` و`dataset_name` إلى النموذج ومجموعة البيانات (إما من Hub أو مسار محلي).

<Tip>

لتدريب مجموعة بيانات محلية، قم بتعيين متغيرات البيئة `TRAIN_DIR` و`OUTPUT_DIR` إلى مسار مجموعة البيانات والمكان الذي سيتم فيه حفظ النموذج.

</Tip>

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export dataset_name="lambdalabs/naruto-blip-captions"

python train_text_to_image_flax.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$dataset_name \
--resolution=512 --center_crop --random_flip \
--train_batch_size=1 \
--max_train_steps=15000 \
--learning_rate=1e-05 \
--max_grad_norm=1 \
--output_dir="sd-naruto-model" \
--push_to_hub
```

</hfoption>

</hfoptions>

بمجرد اكتمال التدريب، يمكنك استخدام نموذجك المدرب للتنبؤ:

<hfoptions id="training-inference">

<hfoption id="PyTorch">

```py
from diffusers import StableDiffusionPipeline
import torch

pipeline = StableDiffusionPipeline.from_pretrained("path/to/saved_model", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

image = pipeline(prompt="yoda").images[0]
image.save("yoda-naruto.png")
```

</hfoption>

<hfoption id="Flax">

```py
import jax
import numpy as np
from flax.jax_utils import replicate
from flax.training.common_utils import shard
from diffusers import FlaxStableDiffusionPipeline

pipeline, params = FlaxStableDiffusionPipeline.from_pretrained("path/to/saved_model", dtype=jax.numpy.bfloat16)

prompt = "yoda naruto"
prng_seed = jax.random.PRNGKey(0)
num_inference_steps = 50

num_samples = jax.device_count()
prompt = num_samples * [prompt]
prompt_ids = pipeline.prepare_inputs(prompt)

# shard inputs and rng
params = replicate(params)
prng_seed = jax.random.split(prng_seed, jax.device_count())
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, params, prng_seed, num_inference_steps, jit=True).images
images = pipeline.numpy_to_pil(np.asarray(images.reshape((num_samples,) + images.shape[-3:])))
image.save("yoda-naruto.png")
```

</hfoption>
</hfoptions>

## Next steps

تهانينا على تدريب نموذج تحويل النص إلى صورة الخاص بك! لمعرفة المزيد حول كيفية استخدام النموذج الجديد، قد تكون الأدلة التالية مفيدة:


- Learn how to [load LoRA weights](../using-diffusers/loading_adapters#LoRA) for inference if you trained your model with LoRA.
- Learn more about how certain parameters like guidance scale or techniques such as prompt weighting can help you control inference in the [Text-to-image](../using-diffusers/conditional_image_generation) task guide.
