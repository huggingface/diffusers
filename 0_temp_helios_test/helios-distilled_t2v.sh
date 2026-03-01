CUDA_VISIBLE_DEVICES=1 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Distilled" \
    --transformer_path "BestWishYsh/Helios-Distilled" \
    --sample_type "t2v" \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --num_frames 240 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_enable_stage3 \
    --is_amplify_first_chunk \
    --output_folder "./output_helios/helios-distilled"


    # --pyramid_num_inference_steps_list 1 1 1 \
    # --enable_compile \