# الاستنتاج الموزع مع وحدات معالجة الرسوميات (GPUs) متعددة

في الإعدادات الموزعة، يمكنك تشغيل الاستنتاج عبر وحدات معالجة الرسوميات (GPU) متعددة باستخدام [Accelerate](https://huggingface.co/docs/accelerate/index) أو [PyTorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html) من 🤗، وهو مفيد لإنشاء موجهات متعددة بالتوازي.

سيوضح هذا الدليل كيفية استخدام 🤗 Accelerate و PyTorch Distributed للاستنتاج الموزع.

## 🤗 Accelerate

🤗 [Accelerate](https://huggingface.co/docs/accelerate/index) هي مكتبة مصممة لتسهيل التدريب أو تشغيل الاستدلال عبر الإعدادات الموزعة. فهو يبسط عملية إعداد البيئة الموزعة، مما يتيح لك التركيز على رمز PyTorch الخاص بك.

للبدء، قم بإنشاء ملف Python وقم بتهيئة [`accelerate.PartialState`] لإنشاء بيئة موزعة؛ يتم اكتشاف إعدادك تلقائيًا، لذلك لا تحتاج إلى تحديد `rank` أو `world_size` بشكل صريح. قم بنقل [`DiffusionPipeline`] إلى `distributed_state.device` لتعيين وحدة معالجة الرسوميات (GPU) لكل عملية.

الآن استخدم أداة [`~accelerate.PartialState.split_between_processes`] المساعدة كمدير سياق لتوزيع الموجهات تلقائيًا بين عدد العمليات.

```py
import torch
from accelerate import PartialState
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, use_safetensors=True
)
distributed_state = PartialState()
pipeline.to(distributed_state.device)

with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipeline(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

استخدم الحجة `--num_processes` لتحديد عدد وحدات معالجة الرسوميات (GPU) التي سيتم استخدامها، واستدعي `accelerate launch` لتشغيل البرنامج النصي:

```bash
accelerate launch run_distributed.py --num_processes=2
```

<Tip>

لمعرفة المزيد، راجع دليل [الاستدلال الموزع باستخدام 🤗 Accelerate](https://huggingface.co/docs/accelerate/en/usage_guides/distributed_inference#distributed-inference-with-accelerate).

</Tip>

### وضع الجهاز

> [!WARNING]
> هذه الميزة تجريبية وقد تتغير واجهات برمجة التطبيقات الخاصة بها في المستقبل.

مع Accelerate، يمكنك استخدام `device_map` لتحديد كيفية توزيع نماذج خط الأنابيب عبر أجهزة متعددة. هذا مفيد في الحالات التي تحتوي على أكثر من وحدة معالجة الرسوميات (GPU) واحدة.

على سبيل المثال، إذا كان لديك وحدتي معالجة رسوميات (GPU) بسعة 8 جيجابايت، فقد لا يعمل استخدام [`~DiffusionPipeline.enable_model_cpu_offload`] بشكل جيد لأنه:

- يعمل فقط على وحدة معالجة الرسوميات (GPU) واحدة
- قد لا يناسب نموذج واحد وحدة معالجة الرسوميات (GPU) واحدة (قد يعمل [`~DiffusionPipeline.enable_sequential_cpu_offload`] ولكنه سيكون بطيئًا للغاية وهو أيضًا محدود بوحدة معالجة الرسوميات (GPU) واحدة)

لاستخدام وحدتي معالجة الرسوميات (GPU)، يمكنك استخدام إستراتيجية وضع الجهاز "المتوازنة" التي تقوم بتقسيم النماذج عبر جميع وحدات معالجة الرسوميات (GPU) المتاحة.

> [!WARNING]
> يتم دعم إستراتيجية "المتوازنة" فقط في الوقت الحالي، ونخطط لدعم خرائط إضافية في المستقبل.

```diff
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
-    "runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True،
+    "runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True، device_map="balanced"
)
image = pipeline ("a dog").images [0]
image
```

يمكنك أيضًا تمرير قاموس لفرض الحد الأقصى لذاكرة وحدة معالجة الرسوميات (GPU) التي يمكن استخدامها على كل جهاز:

```diff
from diffusers import DiffusionPipeline
import torch

max_memory = {0:"1GB"، 1:"1GB"}
pipeline = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"،
torch_dtype=torch.float16،
use_safetensors=True،
device_map="balanced"،
+   max_memory=max_memory
)
image = pipeline ("a dog").images [0]
image
```

إذا لم يكن الجهاز موجودًا في `max_memory`، فسيتم تجاهله تمامًا ولن يشارك في وضع الجهاز.

افتراضيًا، يستخدم Diffusers الحد الأقصى لذاكرة جميع الأجهزة. إذا لم تناسب النماذج وحدات معالجة الرسوميات (GPU)، فسيتم نقلها إلى وحدة المعالجة المركزية (CPU). إذا لم يكن لدى وحدة المعالجة المركزية (CPU) ذاكرة كافية، فقد ترى خطأً. في هذه الحالة، يمكنك اللجوء إلى استخدام [`~DiffusionPipeline.enable_sequential_cpu_offload`] و [`~DiffusionPipeline.enable_model_cpu_offload`].

اتصل [`~DiffusionPipeline.reset_device_map`] لإعادة تعيين `device_map` لخط الأنابيب. هذا ضروري أيضًا إذا كنت تريد استخدام طرق مثل `to()`، [`~DiffusionPipeline.enable_sequential_cpu_offload`]، و [`~DiffusionPipeline.enable_model_cpu_offload`] على خط أنابيب تم تعيين جهازه.

```py
pipeline.reset_device_map()
```

بمجرد تعيين جهاز خط الأنابيب، يمكنك أيضًا الوصول إلى خريطة الجهاز الخاصة به عبر `hf_device_map`:

```py
print(pipeline.hf_device_map)
```

قد تبدو خريطة الجهاز على النحو التالي:

```bash
{"unet": 1، "vae": 1، "safety_checker": 0، "text_encoder": 0}
```

## PyTorch الموزع

تدعم PyTorch [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) الذي يمكّن الموازاة للبيانات.

للبدء، قم بإنشاء ملف Python واستورد `torch.distributed` و `torch.multiprocessing` لإعداد مجموعة العمليات الموزعة ولإنشاء العمليات للاستدلال على كل وحدة معالجة الرسوميات (GPU). يجب عليك أيضًا تهيئة [`DiffusionPipeline`]:

```py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from diffusers import DiffusionPipeline

sd = DiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5"، torch_dtype=torch.float16، use_safetensors=True
)
```

ستحتاج إلى إنشاء دالة لتشغيل الاستدلال؛ [`init_process_group`](https://pytorch.org/docs/stable/distributed.html?highlight=init_process_group#torch.distributed.init_process_group) يتعامل مع إنشاء بيئة موزعة مع نوع backend الذي سيتم استخدامه، `rank` للعملية الحالية، و`world_size` أو عدد العمليات المشاركة. إذا كنت تقوم بتشغيل الاستدلال بالتوازي على وحدتي معالجة الرسوميات (GPU)، فإن `world_size` هو 2.

قم بنقل [`DiffusionPipeline`] إلى `rank` واستخدم `get_rank` لتعيين وحدة معالجة الرسوميات (GPU) لكل عملية، حيث تتعامل كل عملية مع موجه مختلف:

```py
def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    sd.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    image = sd(prompt).images[0]
    image.save(f"./{'_'.join(prompt)}.png")
```

لتشغيل الاستدلال الموزع، اتصل [`mp.spawn`](https://pytorch.org/docs/stable/multiprocessing.html#torch.multiprocessing.spawn) لتشغيل دالة `run_inference` على عدد وحدات معالجة الرسوميات (GPU) المحددة في `world_size`:

```py
def main():
world_size = 2
mp.spawn(run_inference، args=(world_size،)، nprocs=world_size، join=True)


if __name__ == "__main__":
main()
```

بمجرد الانتهاء من كتابة نص الاستدلال، استخدم الحجة `--nproc_per_node` لتحديد عدد وحدات معالجة الرسوميات (GPU) التي سيتم استخدامها واستدعاء `torchrun` لتشغيل البرنامج النصي:

```bash
torchrun run_distributed.py --nproc_per_node=2
```