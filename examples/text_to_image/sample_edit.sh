python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/van_gogh_multi_lora/checkpoint-3400/pytorch_model.bin" \
        --prompts "The Starry Night by Van Gogh" --num_images 36 --output_dir "/scratch/mp5847/diffusers_test/van_gogh_multi_lora" --create_grid --lora_edit_alpha -1.4

# python3 sample_edit.py --model_finetuned_lora "/scratch/mp5847/diffusers_ckpt/van_gogh_multi_lora/checkpoint-3400/pytorch_model.bin" \
#         --prompts "Bedroom in Arles by Van Gogh" --num_images 36 --output_dir "/scratch/mp5847/diffusers_test/van_gogh_multi_lora" --create_grid --lora_edit_alpha -1.2