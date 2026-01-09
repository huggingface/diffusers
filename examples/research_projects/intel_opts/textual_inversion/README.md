## Textual Inversion fine-tuning example

[Textual inversion](https://huggingface.co/papers/2208.01618) is a method to personalize text2image models like stable diffusion on your own images using just 3-5 examples.
The `textual_inversion.py` script shows how to implement the training procedure and adapt it for stable diffusion.

## Training with Intel Extension for PyTorch

Intel Extension for PyTorch provides the optimizations for faster training and inference on CPUs. You can leverage the training example "textual_inversion.py". Follow the [instructions](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) to get the model and [dataset](https://huggingface.co/sd-concepts-library/dicoo2) before running the script.

The example supports both single node and multi-node distributed training:

### Single node training

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="path-to-dir-containing-dicoo-images"

python textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<dicoo>" --initializer_token="toy" \
  --seed=7 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=3000 \
  --learning_rate=2.5e-03 --scale_lr \
  --output_dir="textual_inversion_dicoo"
```

Note: Bfloat16 is available on Intel Xeon Scalable Processors Cooper Lake or Sapphire Rapids. You may not get performance speedup without Bfloat16 support.

### Multi-node distributed training

Before running the scripts, make sure to install the library's training dependencies successfully:

```bash
python -m pip install oneccl_bind_pt==1.13 -f https://developer.intel.com/ipex-whl-stable-cpu
```

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="path-to-dir-containing-dicoo-images"

oneccl_bindings_for_pytorch_path=$(python -c "from oneccl_bindings_for_pytorch import cwd; print(cwd)")
source $oneccl_bindings_for_pytorch_path/env/setvars.sh

python -m intel_extension_for_pytorch.cpu.launch --distributed \
  --hostfile hostfile --nnodes 2 --nproc_per_node 2 textual_inversion.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --learnable_property="object" \
  --placeholder_token="<dicoo>" --initializer_token="toy" \
  --seed=7 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --max_train_steps=750 \
  --learning_rate=2.5e-03 --scale_lr \
  --output_dir="textual_inversion_dicoo"
```
The above is a simple distributed training usage on 2 nodes with 2 processes on each node. Add the right hostname or ip address in the "hostfile" and make sure these 2 nodes are reachable from each other. For more details, please refer to the [user guide](https://github.com/intel/torch-ccl).


### Reference

We publish a [Medium blog](https://medium.com/intel-analytics-software/personalized-stable-diffusion-with-few-shot-fine-tuning-on-a-single-cpu-f01a3316b13) on how to create your own Stable Diffusion model on CPUs using textual inversion. Try it out now, if you have interests.
