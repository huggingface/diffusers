# InstructPix2Pix text-to-edit-image fine-tuning
This extended LoRA training script was authored by [Aiden-Frost](https://github.com/Aiden-Frost).
This is an experimental LoRA extension of [this example](https://github.com/huggingface/diffusers/blob/main/examples/instruct_pix2pix/train_instruct_pix2pix.py). This script provides further support add LoRA layers for unet model.

## Training script example

```bash
export MODEL_ID="timbrooks/instruct-pix2pix"
export DATASET_ID="instruction-tuning-sd/cartoonization"
export OUTPUT_DIR="instructPix2Pix-cartoonization"

accelerate launch finetune_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_ID \
  --dataset_name=$DATASET_ID \
  --enable_xformers_memory_efficient_attention \
  --resolution=256 --random_flip \
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing \
  --max_train_steps=15000 \
  --checkpointing_steps=5000 --checkpoints_total_limit=1 \
  --learning_rate=5e-05 --lr_warmup_steps=0 \
  --val_image_url="https://hf.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png" \
  --validation_prompt="Generate a cartoonized version of the natural image" \
  --seed=42 \
  --rank=4 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --push_to_hub
```

## Inference
After training the model and the lora weight of the model is stored in the ```$OUTPUT_DIR```.

```py
# load the base model pipeline
pipe_lora = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix")

# Load LoRA weights from the provided path
output_dir = "path/to/lora_weight_directory"
pipe_lora.unet.load_attn_procs(output_dir)

input_image_path = "/path/to/input_image"
input_image = Image.open(input_image_path)
edited_images = pipe_lora(num_images_per_prompt=1, prompt=args.edit_prompt, image=input_image, num_inference_steps=1000).images
edited_images[0].show()
```

## Results

Here is an example of using the script to train a instructpix2pix model.
Trained on google colab T4 GPU

```bash
MODEL_ID="timbrooks/instruct-pix2pix"
DATASET_ID="instruction-tuning-sd/cartoonization"
TRAIN_EPOCHS=100
```

Below are few examples for given the input image, edit_prompt and the edited_image (output of the model)

<p align="center">
    <img src="https://github.com/Aiden-Frost/Efficiently-teaching-counting-and-cartoonization-to-InstructPix2Pix.-/blob/main/diffusers_result_assets/edited_image_results.png?raw=true" alt="instructpix2pix-inputs" width=600/>
</p>


Here are some rough statistics about the training model using this script

<p align="center">
    <img src="https://github.com/Aiden-Frost/Efficiently-teaching-counting-and-cartoonization-to-InstructPix2Pix.-/blob/main/diffusers_result_assets/results.png?raw=true" alt="instructpix2pix-inputs" width=600/>
</p>

## References

* InstructPix2Pix - https://github.com/timothybrooks/instruct-pix2pix
* Dataset and example training script - https://huggingface.co/blog/instruction-tuning-sd
* For more information about the project - https://github.com/Aiden-Frost/Efficiently-teaching-counting-and-cartoonization-to-InstructPix2Pix.-