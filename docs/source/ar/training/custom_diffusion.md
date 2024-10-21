# Custom Diffusion

تقنية Custom Diffusion هي تقنية تدريب لشخصنة نماذج توليد الصور. وعلى غرار Textual Inversion وDreamBooth وLoRA، لا تتطلب تقنية Custom Diffusion سوى عدد قليل من الصور (4-5 صور تقريبًا) كأمثلة. تعمل هذه التقنية من خلال تدريب الأوزان في طبقات الاهتمام المتقاطع فقط، وتستخدم كلمة خاصة لتمثيل المفهوم الذي تم تعلمه حديثًا. وتتميز تقنية Custom Diffusion بكونها فريدة من نوعها لأنها يمكن أن تتعلم أيضًا مفاهيم متعددة في نفس الوقت.

إذا كنت تقوم بالتدريب على وحدة معالجة رسومات (GPU) ذات ذاكرة وصول عشوائي (VRAM) محدودة، فيجب عليك تجربة تمكين xFormers باستخدام --enable_xformers_memory_efficient_attention لتسريع التدريب مع تقليل متطلبات ذاكرة VRAM (16 جيجابايت). ولتوفير المزيد من الذاكرة، أضف --set_grads_to_none في حجة التدريب لتعيين التدرجات إلى None بدلاً من الصفر (قد يسبب هذا الخيار بعض المشكلات، لذا إذا واجهت أي مشكلات، حاول إزالة هذا المعامل).

سيتناول هذا الدليل بالتفصيل النص البرمجي train_custom_diffusion.py لمساعدتك على التعرف عليه بشكل أفضل، وكيف يمكنك تكييفه مع حالتك الاستخدام الخاصة.

قبل تشغيل النص البرمجي، تأكد من تثبيت المكتبة من المصدر:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

انتقل إلى مجلد الأمثلة الذي يحتوي على النص البرمجي للتدريب وقم بتثبيت التبعيات المطلوبة:

```bash
cd examples/custom_diffusion
pip install -r requirements.txt
pip install clip-retrieval
```

🤗 Accelerate هي مكتبة تساعدك على التدريب على وحدات معالجة رسومات (GPUs) أو وحدات معالجة الدقة الفائقة (TPUs) متعددة أو باستخدام الدقة المختلطة. وستقوم تلقائيًا بتهيئة إعداد التدريب الخاص بك بناءً على أجهزتك وبيئتك. اطلع على الجولة السريعة في 🤗 Accelerate لمعرفة المزيد.

قم بتهيئة بيئة 🤗 Accelerate:

```bash
accelerate config
```

لإعداد بيئة 🤗 Accelerate الافتراضية دون اختيار أي تكوينات:

```bash
accelerate config default
```

أو إذا لم يدعم بيئتك غلافًا تفاعليًا، مثل دفتر ملاحظات، فيمكنك استخدام:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

أخيرًا، إذا كنت تريد تدريب نموذج على مجموعة بياناتك الخاصة، فراجع دليل إنشاء مجموعة بيانات للتدريب لمعرفة كيفية إنشاء مجموعة بيانات تعمل مع النص البرمجي للتدريب.

تسلط الأقسام التالية الضوء على أجزاء من النص البرمجي للتدريب والتي تُعد مهمة لفهم كيفية تعديلها، ولكنها لا تغطي كل جانب من جوانب النص البرمجي بالتفصيل. إذا كنت مهتمًا بمعرفة المزيد، فيُرجى الاطلاع على النص البرمجي وإخبارنا إذا كان لديك أي أسئلة أو مخاوف.

## معلمات النص البرمجي

يحتوي النص البرمجي للتدريب على جميع المعلمات لمساعدتك على تخصيص عملية تشغيل التدريب الخاصة بك. ويمكن العثور عليها في دالة parse_args(). تأتي الدالة مع القيم الافتراضية، ولكن يمكنك أيضًا تعيين قيمك الخاصة في أمر التدريب إذا أردت ذلك.

على سبيل المثال، لتغيير دقة صورة الإدخال:

```bash
accelerate launch train_custom_diffusion.py \
--resolution=256
```

تم وصف العديد من المعلمات الأساسية في دليل تدريب DreamBooth، لذلك يركز هذا الدليل على المعلمات الفريدة لتقنية Custom Diffusion:

- --freeze_model: يقوم بتجميد المعلمات الأساسية والقيم في طبقة الاهتمام المتقاطع؛ والقيمة الافتراضية هي crossattn_kv، ولكن يمكنك تعيينها على crossattn لتدريب جميع المعلمات في طبقة الاهتمام المتقاطع
- --concepts_list: لتعلم مفاهيم متعددة، قم بتوفير مسار إلى ملف JSON يحتوي على المفاهيم
- --modifier_token: كلمة خاصة تُستخدم لتمثيل المفهوم الذي تم تعلمه
- --initializer_token: كلمة خاصة تُستخدم لتهيئة تضمينات modifier_token

### خسارة الحفاظ على الأولوية

خسارة الحفاظ على الأولوية هي طريقة تستخدم عينات مولدة من النموذج نفسه لمساعدته على تعلم كيفية إنشاء صور أكثر تنوعًا. وبما أن صور العينات المولدة هذه تنتمي إلى نفس الفئة التي تنتمي إليها الصور التي قدمتها، فإنها تساعد النموذج على الاحتفاظ بما تعلمه عن الفئة وكيف يمكنه استخدام ما يعرفه بالفعل عن الفئة لإنشاء تركيبات جديدة.

تم وصف العديد من معلمات خسارة الحفاظ على الأولوية في دليل تدريب DreamBooth.

### الضبط

تتضمن تقنية Custom Diffusion تدريب الصور المستهدفة باستخدام مجموعة صغيرة من الصور الحقيقية لمنع الإفراط في التكيّف. وكما يمكنك أن تتخيل، يمكن أن يكون من السهل القيام بذلك عند التدريب على عدد قليل من الصور فقط! قم بتنزيل 200 صورة حقيقية باستخدام clip_retrieval. يجب أن يكون class_prompt من نفس فئة الصور المستهدفة. يتم تخزين هذه الصور في class_data_dir.

```bash
python retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
```

لتمكين الضبط، أضف المعلمات التالية:

- --with_prior_preservation: ما إذا كان سيتم استخدام خسارة الحفاظ على الأولوية
- --prior_loss_weight: يتحكم في تأثير خسارة الحفاظ على الأولوية على النموذج
- --real_prior: ما إذا كان سيتم استخدام مجموعة صغيرة من الصور الحقيقية لمنع الإفراط في التكيّف

```bash
accelerate launch train_custom_diffusion.py \
--with_prior_preservation \
--prior_loss_weight=1.0 \
--class_data_dir="./real_reg/samples_cat" \
--class_prompt="cat" \
--real_prior=True \
```

## النص البرمجي للتدريب

يوجد الكثير من الرموز في نص Custom Diffusion البرمجي للتدريب مماثلة لتلك الموجودة في نص DreamBooth البرمجي للتدريب. يركز هذا الدليل بدلاً من ذلك على الرموز ذات الصلة بتقنية Custom Diffusion.

يحتوي نص Custom Diffusion البرمجي للتدريب على فئتين من مجموعات البيانات:

- CustomDiffusionDataset: معالجة الصور المسبقة، وصور الفئة، والفئات التدريبية
- PromptDataset: إعداد الفئات لتوليد صور الفئة

بعد ذلك، يتم [إضافة modifier_token إلى المحلل اللغوي](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L811)، وتحويلها إلى معرفات الرموز، وإعادة تحجيم تضمينات الرموز لحساب modifier_token الجديد. ثم يتم تهيئة تضمينات modifier_token باستخدام تضمينات initializer_token. يتم تجميد جميع المعلمات في المشفر النصي، باستثناء تضمينات الرموز لأن هذا هو ما يحاول النموذج تعلمه لربطه بالمفاهيم.

```py
params_to_freeze = itertools.chain(
text_encoder.text_model.encoder.parameters(),
text_encoder.text_model.final_layer_norm.parameters(),
text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)
```

الآن، ستحتاج إلى إضافة [أوزان Custom Diffusion](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L911C3-L911C3) إلى طبقات الاهتمام. هذه خطوة مهمة للغاية للحصول على الشكل والحجم الصحيح لأوزان الاهتمام، ولتعيين العدد المناسب من معالجات الاهتمام في كل كتلة UNet.

```py
st = unet.state_dict()
for name, _ in unet.attn_processors.items():
cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
if name.startswith("mid_block"):
hidden_size = unet.config.block_out_channels[-1]
elif name.startswith("up_blocks"):
block_id = int(name[len("up_blocks.")])
hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
elif name.startswith("down_blocks"):
block_id = int(name[len("down_blocks.")])
hidden_size = unet.config.block_out_channels[block_id]
layer_name = name.split(".processor")[0]
weights = {
"to_k_custom_diffusion.weight": st[layer_name + ".to_k.weight"],
"to_v_custom_diffusion.weight": st[layer_name + ".to_v.weight"],
}
if train_q_out:
weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
if cross_attention_dim is not None:
custom_diffusion_attn_procs[name] = attention_class(
train_kv=train_kv,
train_q_out=train_q_out,
hidden_size=hidden_size,
cross_attention_dim=cross_attention_dim,
).to(unet.device)
custom_diffusion_attn_procs[name].load_state_dict(weights)
else:
custom_diffusion_attn_procs[name] = attention_class(
train_kv=False,
train_q_out=False,
hidden_size=hidden_size,
cross_attention_dim=cross_attention_dim,
)
del st
unet.set_attn_processor(custom_diffusion_attn_procs)
custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)
```

يتم تهيئة [المحسن](https://github.com/huggingface/diffusers/blob/84cd9e8d01adb47f046b1ee449fc76a0c32dc4e2/examples/custom_diffusion/train_custom_diffusion.py#L982) لتحديث معلمات طبقة الاهتمام المتقاطع:

```py
optimizer = optimizer_class(
itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters())
if args.modifier_token is not None
else custom_diffusion_layers.parameters(),
lr=args.learning_rate,
betas=(args.adam_beta1, args.adam_beta2),
weight_decay=args.adam_weight_decay,
eps=args.adam_epsilon,
)
```

في [حلقة التدريب](https://github.com/huggingface/diffusers/blob/84cd9e8d01adb47f046b1ee449fc76a0c32dc4e2/examples/custom_diffusion/train_custom_diffusion.py#L1048)، من المهم تحديث التضمينات للمفهوم الذي تحاول تعلمه فقط. وهذا يعني تعيين التدرجات لجميع التضمينات الرمزية الأخرى إلى الصفر:

```py
if args.modifier_token is not None:
if accelerator.num_processes > 1:
grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
else:
grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
for i in range(len(modifier_token_id[1:])):
index_grads_to_zero = index_grads_to_zero & (
torch.arange(len(tokenizer)) != modifier_token_id[i]
)
grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
index_grads_to_zero, :
].fill_(0)
```
## تشغيل السكربت

عندما تنتهي من إجراء جميع التغييرات أو تكون راضيًا عن التكوين الافتراضي، ستكون جاهزًا لتشغيل سكربت التدريب! 🚀

في هذا الدليل، ستقوم بتنزيل واستخدام هذه الصور مثال [صور القطط](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip). يمكنك أيضًا إنشاء واستخدام مجموعة البيانات الخاصة بك إذا كنت تريد (راجع الدليل [إنشاء مجموعة بيانات للتدريب](create_dataset)).

قم بتعيين متغير البيئة `MODEL_NAME` إلى معرف نموذج على Hub أو مسار إلى نموذج محلي، و`INSTANCE_DIR` إلى المسار الذي قمت بتنزيل صور القط إليه للتو، و`OUTPUT_DIR` إلى المكان الذي تريد حفظ النموذج فيه. ستستخدم `<new1>` ككلمة خاصة لربط التضمينات التي تم تعلمها حديثًا بها. يقوم السكربت بإنشاء وحفظ نقاط تفتيش النموذج وملف pytorch_custom_diffusion_weights.bin إلى مستودعك.

لمراقبة تقدم التدريب باستخدام Weights and Biases، أضف المعلمة `--report_to=wandb` إلى أمر التدريب وحدد موجه تحقق من الصحة باستخدام `--validation_prompt`. هذا مفيد للتصحيح وحفظ النتائج الوسيطة.

<Tip>

إذا كنت تتدرب على وجوه بشرية، فقد وجد فريق Custom Diffusion أن المعلمات التالية تعمل بشكل جيد:

- `--learning_rate=5e-6`
- يمكن أن يكون `--max_train_steps` أي رقم بين 1000 و 2000
- `--freeze_model=crossattn`
- استخدم ما لا يقل عن 15-20 صورة للتدريب

</Tip>

<hfoptions id="training-inference">
<hfoption id="single concept">

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
--pretrained_model_name_or_path=$MODEL_NAME \
--instance_data_dir=$INSTANCE_DIR \
--output_dir=$OUTPUT_DIR \
--class_data_dir=./real_reg/samples_cat/ \
--with_prior_preservation \
--real_prior \
--prior_loss_weight=1.0 \
--class_prompt="cat" \
--num_class_images=200 \
--instance_prompt="photo of a <new1> cat" \
--resolution=512 \
--train_batch_size=2 \
--learning_rate=1e-5 \
--lr_warmup_steps=0 \
--max_train_steps=250 \
--scale_lr \
--hflip \
--modifier_token "<new1>" \
--validation_prompt="<new1> cat sitting in a bucket" \
--report_to="wandb" \
--push_to_hub
```

</hfoption>
<hfoption id="multiple concepts">

يمكن لـ Custom Diffusion أيضًا تعلم مفاهيم متعددة إذا قدمت ملف [JSON](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json) مع بعض التفاصيل حول كل مفهوم يجب أن يتعلمه.

قم بتشغيل clip-retrieval لجمع بعض الصور الحقيقية لاستخدامها في التنظيم:

```bash
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```

بعد ذلك، يمكنك تشغيل السكربت:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_custom_diffusion.py \
--pretrained_model_Multiplier_path=$MODEL_NAME \
--output_dir=$OUTPUT_DIR \
--concepts_list=./concept_list.json \
--with_prior_preservation \
--real_prior \
--prior_loss_weight=1.0 \
--resolution=512 \
--train_batch_size=2 \
--learning_rate=1e-5 \
--lr_warmup_steps=0 \
--max_train_steps=500 \
--num_class_images=200 \
--scale_lr \
--hflip \
--modifier_token "<new1>+<new2>" \
--push_to_hub
```

</hfoption>
</hfoptions>

بمجرد الانتهاء من التدريب، يمكنك استخدام نموذج Custom Diffusion الجديد للاستنتاج.

<hfoptions id="training-inference">
<hfoption id="single concept">

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16,
).to("cuda")
pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion("path-to-save-model", weight_name="<new1>.bin")

image = pipeline(
"<new1> cat sitting in a bucket",
num_inference_steps=100,
guidance_scale=6.0,
eta=1.0,
).images[0]
image.save("cat.png")
```

</hfoption>
<hfoption id="multiple concepts">

```py
import torch
from huggingface_hub.repocard import RepoCard
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("sayakpaul/custom-diffusion-cat-wooden-pot", torch_dtype=torch.float16).to("cuda")
pipeline.unet.load_attn_procs(model_id, weight_name="pytorch_custom_diffusion_weights.bin")
pipeline.load_textual_inversion(model_id, weight_name="<new1>.bin")
pipeline.load_textual_inversion(model_id, weight_name="<new2>.bin")

image = pipeline(
"the <new1> cat sculpture in the style of a <new2> wooden pot",
num_inference_steps=100,
guidance_scale=6.0,
eta=1.0,
).images[0]
image.save("multi-subject.png")
```

</hfoption>
</hfoptions>

## الخطوات التالية

تهانينا على تدريب نموذج باستخدام Custom Diffusion! 🎉 لمزيد من المعلومات:

- اقرأ منشور المدونة [تخصيص المفهوم المتعدد لنشر النص إلى الصورة](https://www.cs.cmu.edu/~custom-diffusion/) لمعرفة المزيد من التفاصيل حول النتائج التجريبية من فريق Custom Diffusion.