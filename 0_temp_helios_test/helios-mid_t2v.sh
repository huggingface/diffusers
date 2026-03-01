CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Mid" \
    --transformer_path "BestWishYsh/Helios-Mid" \
    --sample_type "t2v" \
    --prompt "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond." \
    --guidance_scale 5.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 20 20 20 \
    --use_zero_init \
    --zero_steps 1 \
    --output_folder "./output_helios/stage-2"


    # --pyramid_num_inference_steps_list 17 17 17 \
    # --enable_compile \