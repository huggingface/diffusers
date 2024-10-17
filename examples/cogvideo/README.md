# LoRA finetuning example for CogVideoX

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

At the moment, LoRA finetuning has only been tested for [CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b).

> [!NOTE]
> The scripts for CogVideoX come with limited support and may not be fully compatible with different training techniques. They are not feature-rich either and simply serve as minimal examples of finetuning to take inspiration from and improve.
>
> A repository containing memory-optimized finetuning scripts with support for multiple resolutions, dataset preparation, captioning, etc. is available [here](https://github.com/a-r-r-o-w/cogvideox-factory), which will be maintained jointly by the CogVideoX and Diffusers team.

## Data Preparation

The training scripts accepts data in two formats.

**First data format**

Two files where one file contains line-separated prompts and another file contains line-separated paths to video data (the path to video files must be relative to the path you pass when specifying `--instance_data_root`). Let's take a look at an example to understand this better!

Assume you've specified `--instance_data_root` as `/dataset`, and that this directory contains the files: `prompts.txt` and `videos.txt`.

The `prompts.txt` file should contain line-separated prompts:

```
A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.
A black and white animated sequence on a ship's deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language. The character progresses from confident to focused, then to strained and distressed, displaying a range of emotions as it navigates challenges. The ship's interior remains static in the background, with minimalistic details such as a bell and open door. The character's dynamic movements and changing expressions drive the narrative, with no camera movement to distract from its evolving reactions and physical gestures.
...
```

The `videos.txt` file should contain line-separate paths to video files. Note that the path should be _relative_ to the `--instance_data_root` directory.

```
videos/00000.mp4
videos/00001.mp4
...
```

Overall, this is how your dataset would look like if you ran the `tree` command on the dataset root directory:

```
/dataset
├── prompts.txt
├── videos.txt
├── videos
    ├── videos/00000.mp4
    ├── videos/00001.mp4
    ├── ...
```

When using this format, the `--caption_column` must be `prompts.txt` and `--video_column` must be `videos.txt`.

**Second data format**

You could use a single CSV file. For the sake of this example, assume you have a `metadata.csv` file. The expected format is:

```
<CAPTION_COLUMN>,<PATH_TO_VIDEO_COLUMN>
"""A black and white animated sequence featuring a rabbit, named Rabbity Ribfried, and an anthropomorphic goat in a musical, playful environment, showcasing their evolving interaction.""","""00000.mp4"""
"""A black and white animated sequence on a ship's deck features a bulldog character, named Bully Bulldoger, showcasing exaggerated facial expressions and body language. The character progresses from confident to focused, then to strained and distressed, displaying a range of emotions as it navigates challenges. The ship's interior remains static in the background, with minimalistic details such as a bell and open door. The character's dynamic movements and changing expressions drive the narrative, with no camera movement to distract from its evolving reactions and physical gestures.""","""00001.mp4"""
...
```

In this case, the `--instance_data_root` should be the location where the videos are stored and `--dataset_name` should be either a path to local folder or `load_dataset` compatible hosted HF Dataset Repository or URL. Assuming you have videos of your Minecraft gameplay at `https://huggingface.co/datasets/my-awesome-username/minecraft-videos`, you would have to specify `my-awesome-username/minecraft-videos`.

When using this format, the `--caption_column` must be `<CAPTION_COLUMN>` and `--video_column` must be `<PATH_TO_VIDEO_COLUMN>`.

You are not strictly restricted to the CSV format. As long as the `load_dataset` method supports the file format to load a basic `<PATH_TO_VIDEO_COLUMN>` and `<CAPTION_COLUMN>`, you should be good to go. The reason for going through these dataset organization gymnastics for loading video data is because we found `load_dataset` from the datasets library to not fully support all kinds of video formats. This will undoubtedly be improved in the future.

>![NOTE]
> CogVideoX works best with long and descriptive LLM-augmented prompts for video generation. We recommend pre-processing your videos by first generating a summary using a VLM and then augmenting the prompts with an LLM. To generate the above captions, we use [MiniCPM-V-26](https://huggingface.co/openbmb/MiniCPM-V-2_6) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct). A very barebones and no-frills example for this is available [here](https://gist.github.com/a-r-r-o-w/4dee20250e82f4e44690a02351324a4a). The official recommendation for augmenting prompts is [ChatGLM](https://huggingface.co/THUDM?search_models=chatglm) and a length of 50-100 words is considered good.

>![NOTE]
> It is expected that your dataset is already pre-processed. If not, some basic pre-processing can be done by playing with the following parameters:
> `--height`, `--width`, `--fps`, `--max_num_frames`, `--skip_frames_start` and `--skip_frames_end`.
> Presently, all videos in your dataset should contain the same number of video frames when using a training batch size > 1.

<!-- TODO: Implement frame packing in future to address above issue. -->

## Training

You need to setup your development environment by installing the necessary requirements. The following packages are required:
- Torch 2.0 or above based on the training features you are utilizing (might require latest or nightly versions for quantized/deepspeed training)
- `pip install diffusers transformers accelerate peft huggingface_hub` for all things modeling and training related
- `pip install datasets decord` for loading video training data
- `pip install bitsandbytes` for using 8-bit Adam or AdamW optimizers for memory-optimized training
- `pip install wandb` optionally for monitoring training logs
- `pip install deepspeed` optionally for [DeepSpeed](https://github.com/microsoft/DeepSpeed) training
- `pip install prodigyopt` optionally if you would like to use the Prodigy optimizer for training

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

And initialize an [🤗 Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

Or for a default accelerate configuration without answering questions about your environment

```bash
accelerate config default
```

Or if your environment doesn't support an interactive shell (e.g., a notebook)

```python
from accelerate.utils import write_basic_config
write_basic_config()
```

When running `accelerate config`, if we specify torch compile mode to True there can be dramatic speedups. Note also that we use PEFT library as backend for LoRA training, make sure to have `peft>=0.6.0` installed in your environment.

If you would like to push your model to the HF Hub after training is completed with a neat model card, make sure you're logged in:

```
huggingface-cli login

# Alternatively, you could upload your model manually using:
# huggingface-cli upload my-cool-account-name/my-cool-lora-name /path/to/awesome/lora
```

Make sure your data is prepared as described in [Data Preparation](#data-preparation). When ready, you can begin training!

Assuming you are training on 50 videos of a similar concept, we have found 1500-2000 steps to work well. The official recommendation, however, is 100 videos with a total of 4000 steps. Assuming you are training on a single GPU with a `--train_batch_size` of `1`:
- 1500 steps on 50 videos would correspond to `30` training epochs
- 4000 steps on 100 videos would correspond to `40` training epochs

The following bash script launches training for text-to-video lora.

```bash
#!/bin/bash

GPU_IDS="0"

accelerate launch --gpu_ids $GPU_IDS examples/cogvideo/train_cogvideox_lora.py \
  --pretrained_model_name_or_path THUDM/CogVideoX-2b \
  --cache_dir <CACHE_DIR> \
  --instance_data_root <PATH_TO_WHERE_VIDEO_FILES_ARE_STORED> \
  --dataset_name my-awesome-name/my-awesome-dataset \
  --caption_column <CAPTION_COLUMN> \
  --video_column <PATH_TO_VIDEO_COLUMN> \
  --id_token <ID_TOKEN> \
  --validation_prompt "<ID_TOKEN> Spiderman swinging over buildings:::A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance" \
  --validation_prompt_separator ::: \
  --num_validation_videos 1 \
  --validation_epochs 10 \
  --seed 42 \
  --rank 64 \
  --lora_alpha 64 \
  --mixed_precision fp16 \
  --output_dir /raid/aryan/cogvideox-lora \
  --height 480 --width 720 --fps 8 --max_num_frames 49 --skip_frames_start 0 --skip_frames_end 0 \
  --train_batch_size 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-3 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 200 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --optimizer Adam \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --report_to wandb
```

For launching image-to-video finetuning instead, run the `train_cogvideox_image_to_video_lora.py` file instead. Additionally, you will have to pass `--validation_images` as paths to initial images corresponding to `--validation_prompts` for I2V validation to work.

To better track our training experiments, we're using the following flags in the command above:
* `--report_to wandb` will ensure the training runs are tracked on Weights and Biases. To use it, be sure to install `wandb` with `pip install wandb`.
* `validation_prompt` and `validation_epochs` to allow the script to do a few validation inference runs. This allows us to qualitatively check if the training is progressing as expected.

Note that setting the `<ID_TOKEN>` is not necessary. From some limited experimentation, we found it to work better (as it resembles [Dreambooth](https://huggingface.co/docs/diffusers/en/training/dreambooth) like training) than without. When provided, the ID_TOKEN is appended to the beginning of each prompt. So, if your ID_TOKEN was `"DISNEY"` and your prompt was `"Spiderman swinging over buildings"`, the effective prompt used in training would be `"DISNEY Spiderman swinging over buildings"`. When not provided, you would either be training without any such additional token or could augment your dataset to apply the token where you wish before starting the training.

> [!TIP]
> You can pass `--use_8bit_adam` to reduce the memory requirements of training.
> You can pass `--video_reshape_mode` video cropping functionality, supporting options: ['center', 'random', 'none']. See [this](https://gist.github.com/glide-the/7658dbfd5f555be0a1a687a4139dba40) notebook for examples.

> [!IMPORTANT]
> The following settings have been tested at the time of adding CogVideoX LoRA training support:
> - Our testing was primarily done on CogVideoX-2b. We will work on CogVideoX-5b and CogVideoX-5b-I2V soon
> - One dataset comprised of 70 training videos of resolutions `200 x 480 x 720` (F x H x W). From this, by using frame skipping in data preprocessing, we created two smaller 49-frame and 16-frame datasets for faster experimentation and because the maximum limit recommended by the CogVideoX team is 49 frames. Out of the 70 videos, we created three groups of 10, 25 and 50 videos. All videos were similar in nature of the concept being trained.
> - 25+ videos worked best for training new concepts and styles.
> - We found that it is better to train with an identifier token that can be specified as `--id_token`. This is similar to Dreambooth-like training but normal finetuning without such a token works too.
> - Trained concept seemed to work decently well when combined with completely unrelated prompts. We expect even better results if CogVideoX-5B is finetuned.
> - The original repository uses a `lora_alpha` of `1`. We found this not suitable in many runs, possibly due to difference in modeling backends and training settings. Our recommendation is to set to the `lora_alpha` to either `rank` or `rank // 2`.
> - If you're training on data whose captions generate bad results with the original model, a `rank` of 64 and above is good and also the recommendation by the team behind CogVideoX. If the generations are already moderately good on your training captions, a `rank` of 16/32 should work. We found that setting the rank too low, say `4`, is not ideal and doesn't produce promising results.
> - The authors of CogVideoX recommend 4000 training steps and 100 training videos overall to achieve the best result. While that might yield the best results, we found from our limited experimentation that 2000 steps and 25 videos could also be sufficient.
> - When using the Prodigy opitimizer for training, one can follow the recommendations from [this](https://huggingface.co/blog/sdxl_lora_advanced_script) blog. Prodigy tends to overfit quickly. From my very limited testing, I found a learning rate of `0.5` to be suitable in addition to `--prodigy_use_bias_correction`, `prodigy_safeguard_warmup` and `--prodigy_decouple`.
> - The recommended learning rate by the CogVideoX authors and from our experimentation with Adam/AdamW is between `1e-3` and `1e-4` for a dataset of 25+ videos.
>
> Note that our testing is not exhaustive due to limited time for exploration. Our recommendation would be to play around with the different knobs and dials to find the best settings for your data.

## Inference

Once you have trained a lora model, the inference can be done simply loading the lora weights into the `CogVideoXPipeline`.

```python
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)
# pipe.load_lora_weights("/path/to/lora/weights", adapter_name="cogvideox-lora") # Or,
pipe.load_lora_weights("my-awesome-hf-username/my-awesome-lora-name", adapter_name="cogvideox-lora") # If loading from the HF Hub
pipe.to("cuda")

# Assuming lora_alpha=32 and rank=64 for training. If different, set accordingly
pipe.set_adapters(["cogvideox-lora"], [32 / 64])

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The "
    "panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
    "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
    "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
    "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
    "atmosphere of this unique musical performance"
)
frames = pipe(prompt, guidance_scale=6, use_dynamic_cfg=True).frames[0]
export_to_video(frames, "output.mp4", fps=8)
```

If you've trained a LoRA for `CogVideoXImageToVideoPipeline` instead, everything in the above example remains the same except you must also pass an image as initial condition for generation.
