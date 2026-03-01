CUDA_VISIBLE_DEVICES=1 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Distilled" \
    --transformer_path "BestWishYsh/Helios-Distilled" \
    --sample_type "v2v" \
    --video_path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/helios/car.mp4" \
    --prompt "A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery." \
    --num_frames 240 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_enable_stage3 \
    --is_amplify_first_chunk \
    --output_folder "./output_helios/stage-3"


    # --pyramid_num_inference_steps_list 1 1 1 \
    # --enable_compile \