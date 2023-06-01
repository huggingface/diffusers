# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_lora/checkpoint-3500/pytorch_model.bin" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_lora" --create_grid --lora_edit_alpha -0.9

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_lora/checkpoint-3500/pytorch_model.bin" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_lora_relu" --create_grid --lora_edit_alpha -0.9

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_lora_relu/checkpoint-3500/pytorch_model.bin" \
#         --prompts "Post-apocalyptic landscape by Kilian Eng" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_lora_relu_rank=4" --create_grid --lora_edit_alpha -1.2

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_lora_relu/checkpoint-3500/pytorch_model.bin" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_lora_relu_rank=4" --create_grid --lora_edit_alpha -1.2

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_multi_lora/checkpoint-3400/pytorch_model.bin" \
#         --prompts "Post-apocalyptic landscape by Kilian Eng" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_multi_lora" --create_grid --lora_edit_alpha -1.1

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_multi_lora/checkpoint-3400/pytorch_model.bin" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_multi_lora" --create_grid --lora_edit_alpha -1.1

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/elon_musk/checkpoint-3400/pytorch_model.bin" \
#         --prompts "a portrait of Bill Gates" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/elon_musk" --create_grid --lora_edit_alpha -1.5

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/elon_musk/checkpoint-3400/pytorch_model.bin" \
#         --prompts "a portrait of Elon Musk" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/elon_musk" --create_grid --lora_edit_alpha -1.5

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/thomas_kinkade_gpt4_pretrained_clip_filtering_full_lr=1e-05" \
#         --prompts "Thomas Kinkade inspired depiction of a peaceful park" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/thomas_kinkade_gpt4_pretrained_clip_filtering_full_lr=1e-05" --create_grid --tv_edit_alpha 0.6

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/thomas_kinkade_gpt4_pretrained_clip_filtering_full_lr=1e-05" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/thomas_kinkade_gpt4_pretrained_clip_filtering_full_lr=1e-05" --create_grid --tv_edit_alpha 0.6

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_pretrained_full_lr=1e-05" \
#         --prompts "Post-apocalyptic landscape by Kilian Eng" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_pretrained_full_lr=1e-05" --create_grid --tv_edit_alpha 0.3

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_pretrained_full_lr=1e-05" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_pretrained_full_lr=1e-05" --create_grid --tv_edit_alpha 0.3

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/elon_musk_full_lr=1e-05" \
#         --prompts "Elon Musk as a doctor" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/elon_musk_full_lr=1e-05" --create_grid --tv_edit_alpha 0.4

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/elon_musk_full_lr=1e-05" \
#         --prompts "Bill Gates as a doctor" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/elon_musk_full_lr=1e-05" --create_grid --tv_edit_alpha 0.4

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/thomas_kinkade_gpt4_pretrained_clip_filtering_full_lr=1e-05" \
#         --prompts "The Starry Night by Van Gogh" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/thomas_kinkade_gpt4_pretrained_clip_filtering_full_lr=1e-05" --create_grid --tv_edit_alpha 0.6

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kilian_eng_full_lr=1e-5" \
#         --prompts "Post-apocalyptic landscape by Kilian Eng" "a portrait of Bill Gates" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_full_lr=1e-5_0.6" --create_grid --tv_edit_alpha 0.6

#unedited model
# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kelly_mckernan_5000_attention_lr=1e-5" \
#         --prompts "Whimsical creatures with floral elements by Kelly McKernan" "New York city in the style of Van Gogh" "Otherworldly landscape by Kilian Eng" "Thomas Kinkade inspired depiction of a peaceful park" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/Meeting 05.23.2023/Stable Diffusion" --create_grid --tv_edit_alpha 0.0

#!/bin/bash

 python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/van_gogh_5000_attention_lr=1e-5_sd_v1.4" \
                --prompts "Whimsical creatures with floral elements by Kelly McKernan" "New York city in the style of Van Gogh" "Otherworldly landscape by Kilian Eng" "Thomas Kinkade inspired depiction of a peaceful park" \
                --num_images 4 --output_dir "./van_gogh_attention_lr=1e-5_sd_v1.4_alpha=1.8_masking=0.0" --create_grid --tv_edit_alpha 1.8 \
                --model_pretrained "CompVis/stable-diffusion-v1-4"


# alphas=(0.2 0.4)
# ft_layers=("full")

# for alpha in "${alphas[@]}"
# do
#         for ft_layer in "${ft_layers[@]}"
#         do      

#         #edited model all attention
#         python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kelly_mckernan_5000_${ft_layer}_lr=1e-5" \
#                 --prompts "Whimsical creatures with floral elements by Kelly McKernan" "New York city in the style of Van Gogh" "Otherworldly landscape by Kilian Eng" "Thomas Kinkade inspired depiction of a peaceful park" \
#                 --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/Meeting 05.23.2023/kelly_mckernan_5000_${ft_layer}_lr=1e-5_$alpha" --create_grid --tv_edit_alpha $alpha

#         python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kilian_eng_5000_${ft_layer}_lr=1e-5" \
#                 --prompts "Whimsical creatures with floral elements by Kelly McKernan" "New York city in the style of Van Gogh" "Otherworldly landscape by Kilian Eng" "Thomas Kinkade inspired depiction of a peaceful park" \
#                 --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/Meeting 05.23.2023/kilian_eng_5000_${ft_layer}_lr=1e-5_$alpha" --create_grid --tv_edit_alpha $alpha

#         python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/van_gogh_5000_${ft_layer}_lr=1e-5" \
#                 --prompts "Whimsical creatures with floral elements by Kelly McKernan" "New York city in the style of Van Gogh" "Otherworldly landscape by Kilian Eng" "Thomas Kinkade inspired depiction of a peaceful park" \
#                 --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/Meeting 05.23.2023/van_gogh_5000_${ft_layer}_lr=1e-5_$alpha" --create_grid --tv_edit_alpha $alpha

#         python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/thomas_kinkade_5000_${ft_layer}_lr=1e-5" \
#                 --prompts "Whimsical creatures with floral elements by Kelly McKernan" "New York city in the style of Van Gogh" "Otherworldly landscape by Kilian Eng" "Thomas Kinkade inspired depiction of a peaceful park" \
#                 --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/Meeting 05.23.2023/thomas_kinkade_5000_${ft_layer}_lr=1e-5_$alpha" --create_grid --tv_edit_alpha $alpha
#         done
# done