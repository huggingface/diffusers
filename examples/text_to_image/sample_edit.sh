python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_gpt4_lora_relu/checkpoint-3500/pytorch_model.bin" \
        --prompts "Kilian Eng inspired depiction of apocalypse" --num_images 36 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_gpt4_lora_relu" --create_grid --lora_edit_alpha -0.9

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/kilian_eng_multi_lora_clip_filtering/checkpoint-500/pytorch_model.bin" \
#         --prompts "Kilian Eng inspired depiction of apocalypse" --num_images 36 --output_dir "/scratch/mp5847/diffusers_test/kilian_eng_multi_lora_clip_filtering" --create_grid --lora_edit_alpha -0.9