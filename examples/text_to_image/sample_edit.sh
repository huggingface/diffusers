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

python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_pretrained_full" \
        --prompts "Post-apocalyptic landscape by Kilian Eng" --num_images 4 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_pretrained_full" --create_grid --tv_edit_alpha 0.6

# python3 sample_edit.py --model_finetuned "/scratch/mp5847/diffusers_ckpt/elon_musk_full" \
#         --prompts "portrait of Elon Musk" --num_images 4 --output_dir "/scratch/mp5847/diffusers_testelon_musk_full" --create_grid --tv_edit_alpha 0.6