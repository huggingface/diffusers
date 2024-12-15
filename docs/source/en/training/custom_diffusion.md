<!--Copyright 2024 Custom Diffusion authors The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Custom Diffusion

[Custom Diffusion](https://huggingface.co/papers/2212.04488) is a training technique for personalizing image generation models. Like Textual Inversion, DreamBooth, and LoRA, Custom Diffusion only requires a few (~4-5) example images. This technique works by only training weights in the cross-attention layers, and it uses a special word to represent the newly learned concept. Custom Diffusion is unique because it can also learn multiple concepts at the same time.

If you're training on a GPU with limited vRAM, you should try enabling xFormers with `--enable_xformers_memory_efficient_attention` for faster training with lower vRAM requirements (16GB). To save even more memory, add `--set_grads_to_none` in the training argument to set the gradients to `None` instead of zero (this option can cause some issues, so if you experience any, try removing this parameter).

This guide will explore the [train_custom_diffusion.py](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py) script to help you become more familiar with it, and how you can adapt it for your own use-case.

Before running the script, make sure you install the library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

Navigate to the example folder with the training script and install the required dependencies:

```bash
cd examples/custom_diffusion
pip install -r requirements.txt
pip install clip-retrieval
```

<Tip>

🤗 Accelerate is a library for helping you train on multiple GPUs/TPUs or with mixed-precision. It'll automatically configure your training setup based on your hardware and environment. Take a look at the 🤗 Accelerate [Quick tour](https://huggingface.co/docs/accelerate/quicktour) to learn more.

</Tip>

Initialize an 🤗 Accelerate environment:

```bash
accelerate config
```

To setup a default 🤗 Accelerate environment without choosing any configurations:

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell, like a notebook, you can use:

```py
from accelerate.utils import write_basic_config

write_basic_config()
```

Lastly, if you want to train a model on your own dataset, take a look at the [Create a dataset for training](create_dataset) guide to learn how to create a dataset that works with the training script.

<Tip>

The following sections highlight parts of the training script that are important for understanding how to modify it, but it doesn't cover every aspect of the script in detail. If you're interested in learning more, feel free to read through the [script](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion/train_custom_diffusion.py) and let us know if you have any questions or concerns.

</Tip>

## Script parameters

The training script contains all the parameters to help you customize your training run. These are found in the [`parse_args()`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L319) function. The function comes with default values, but you can also set your own values in the training command if you'd like.

For example, to change the resolution of the input image:

```bash
accelerate launch train_custom_diffusion.py \
  --resolution=256
```

Many of the basic parameters are described in the [DreamBooth](dreambooth#script-parameters) training guide, so this guide focuses on the parameters unique to Custom Diffusion:

- `--freeze_model`: freezes the key and value parameters in the cross-attention layer; the default is `crossattn_kv`, but you can set it to `crossattn` to train all the parameters in the cross-attention layer
- `--concepts_list`: to learn multiple concepts, provide a path to a JSON file containing the concepts
- `--modifier_token`: a special word used to represent the learned concept
- `--initializer_token`: a special word used to initialize the embeddings of the `modifier_token`

### Prior preservation loss

Prior preservation loss is a method that uses a model's own generated samples to help it learn how to generate more diverse images. Because these generated sample images belong to the same class as the images you provided, they help the model retain what it has learned about the class and how it can use what it already knows about the class to make new compositions.

Many of the parameters for prior preservation loss are described in the [DreamBooth](dreambooth#prior-preservation-loss) training guide.

### Regularization

Custom Diffusion includes training the target images with a small set of real images to prevent overfitting. As you can imagine, this can be easy to do when you're only training on a few images! Download 200 real images with `clip_retrieval`. The `class_prompt` should be the same category as the target images. These images are stored in `class_data_dir`.

```bash
python retrieve.py --class_prompt cat --class_data_dir real_reg/samples_cat --num_class_images 200
```

To enable regularization, add the following parameters:

- `--with_prior_preservation`: whether to use prior preservation loss
- `--prior_loss_weight`: controls the influence of the prior preservation loss on the model
- `--real_prior`: whether to use a small set of real images to prevent overfitting

```bash
accelerate launch train_custom_diffusion.py \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --class_data_dir="./real_reg/samples_cat" \
  --class_prompt="cat" \
  --real_prior=True \
```

## Training script

<Tip>

A lot of the code in the Custom Diffusion training script is similar to the [DreamBooth](dreambooth#training-script) script. This guide instead focuses on the code that is relevant to Custom Diffusion.

</Tip>

The Custom Diffusion training script has two dataset classes:

- [`CustomDiffusionDataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L165): preprocesses the images, class images, and prompts for training
- [`PromptDataset`](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L148): prepares the prompts for generating class images

Next, the `modifier_token` is [added to the tokenizer](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L811), converted to token ids, and the token embeddings are resized to account for the new `modifier_token`. Then the `modifier_token` embeddings are initialized with the embeddings of the `initializer_token`. All parameters in the text encoder are frozen, except for the token embeddings since this is what the model is trying to learn to associate with the concepts.

```py
params_to_freeze = itertools.chain(
    text_encoder.text_model.encoder.parameters(),
    text_encoder.text_model.final_layer_norm.parameters(),
    text_encoder.text_model.embeddings.position_embedding.parameters(),
)
freeze_params(params_to_freeze)
```

Now you'll need to add the [Custom Diffusion weights](https://github.com/huggingface/diffusers/blob/64603389da01082055a901f2883c4810d1144edb/examples/custom_diffusion/train_custom_diffusion.py#L911C3-L911C3) to the attention layers. This is a really important step for getting the shape and size of the attention weights correct, and for setting the appropriate number of attention processors in each UNet block.

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

The [optimizer](https://github.com/huggingface/diffusers/blob/84cd9e8d01adb47f046b1ee449fc76a0c32dc4e2/examples/custom_diffusion/train_custom_diffusion.py#L982) is initialized to update the cross-attention layer parameters:

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

In the [training loop](https://github.com/huggingface/diffusers/blob/84cd9e8d01adb47f046b1ee449fc76a0c32dc4e2/examples/custom_diffusion/train_custom_diffusion.py#L1048), it is important to only update the embeddings for the concept you're trying to learn. This means setting the gradients of all the other token embeddings to zero:

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

## Launch the script

Once you’ve made all your changes or you’re okay with the default configuration, you’re ready to launch the training script! 🚀

In this guide, you'll download and use these example [cat images](https://www.cs.cmu.edu/~custom-diffusion/assets/data.zip). You can also create and use your own dataset if you want (see the [Create a dataset for training](create_dataset) guide).

Set the environment variable `MODEL_NAME` to a model id on the Hub or a path to a local model, `INSTANCE_DIR`  to the path where you just downloaded the cat images to, and `OUTPUT_DIR` to where you want to save the model. You'll use `<new1>` as the special word to tie the newly learned embeddings to. The script creates and saves model checkpoints and a pytorch_custom_diffusion_weights.bin file to your repository.

To monitor training progress with Weights and Biases, add the `--report_to=wandb` parameter to the training command and specify a validation prompt with `--validation_prompt`. This is useful for debugging and saving intermediate results.

<Tip>

If you're training on human faces, the Custom Diffusion team has found the following parameters to work well:

- `--learning_rate=5e-6`
- `--max_train_steps` can be anywhere between 1000 and 2000
- `--freeze_model=crossattn`
- use at least 15-20 images to train with

</Tip>

<hfoptions id="training-inference">
<hfoption id="single concept">

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"
export INSTANCE_DIR="./data/cat"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --class_data_dir=./real_reg/samples_cat/ \
  --with_prior_preservation \
  --real_prior \
  --prior_loss_weight=1.0 \
  --class_prompt="cat" \
  --num_class_images=200 \
  --instance_prompt="photo of a <new1> cat"  \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=250 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>" \
  --validation_prompt="<new1> cat sitting in a bucket" \
  --report_to="wandb" \
  --push_to_hub
```

</hfoption>
<hfoption id="multiple concepts">

Custom Diffusion can also learn multiple concepts if you provide a [JSON](https://github.com/adobe-research/custom-diffusion/blob/main/assets/concept_list.json) file with some details about each concept it should learn.

Run clip-retrieval to collect some real images to use for regularization:

```bash
pip install clip-retrieval
python retrieve.py --class_prompt {} --class_data_dir {} --num_class_images 200
```

Then you can launch the script:

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="path-to-save-model"

accelerate launch train_custom_diffusion.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --output_dir=$OUTPUT_DIR \
  --concepts_list=./concept_list.json \
  --with_prior_preservation \
  --real_prior \
  --prior_loss_weight=1.0 \
  --resolution=512  \
  --train_batch_size=2  \
  --learning_rate=1e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --num_class_images=200 \
  --scale_lr \
  --hflip  \
  --modifier_token "<new1>+<new2>" \
  --push_to_hub
```

</hfoption>
</hfoptions>

Once training is finished, you can use your new Custom Diffusion model for inference.

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

## Next steps

Congratulations on training a model with Custom Diffusion! 🎉 To learn more:

- Read the [Multi-Concept Customization of Text-to-Image Diffusion](https://www.cs.cmu.edu/~custom-diffusion/) blog post to learn more details about the experimental results from the Custom Diffusion team.