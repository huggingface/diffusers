CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Base" \
    --transformer_path "BestWishYsh/Helios-Base" \
    --sample_type "v2v" \
    --video_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4" \
    --prompt "A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery." \
    --guidance_scale 5.0 \
    --output_folder "./output_helios/stage-1"


    # --enable_compile \
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \