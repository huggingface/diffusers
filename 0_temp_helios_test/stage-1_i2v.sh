CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Base" \
    --transformer_path "BestWishYsh/Helios-Base" \
    --sample_type "i2v" \
    --image_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/wave.jpg" \
    --prompt "A towering emerald wave surges forward, its crest curling with raw power and energy. Sunlight glints off the translucent water, illuminating the intricate textures and deep green hues within the wave’s body. A thick spray erupts from the breaking crest, casting a misty veil that dances above the churning surface. As the perspective widens, the immense scale of the wave becomes apparent, revealing the restless expanse of the ocean stretching beyond. The scene captures the ocean’s untamed beauty and relentless force, with every droplet and ripple shimmering in the light. The dynamic motion and vivid colors evoke both awe and respect for nature’s might." \
    --output_folder "./output_helios/stage-1"


    # --use_default_loader \
    # --enable_compile \
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \