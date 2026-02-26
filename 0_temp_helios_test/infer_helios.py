import os


os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import time

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from diffusers import HeliosTransformer3DModel
from diffusers import HeliosPipeline
from diffusers.schedulers.scheduling_helios import HeliosScheduler

from diffusers import ContextParallelConfig
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="BestWishYsh/Helios-Base",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_folder", type=str, default="./output_helios")
    parser.add_argument("--use_default_loader", action="store_true")
    parser.add_argument("--enable_compile", action="store_true")
    parser.add_argument("--low_vram_mode", action="store_true")
    parser.add_argument("--enable_parallelism", action="store_true")

    # === Generation parameters ===
    # environment
    parser.add_argument("--debug_mode", action="store_true")
    parser.add_argument(
        "--sample_type",
        type=str,
        default="t2v",
        choices=["t2v", "i2v", "v2v"],
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    # base
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--use_dynamic_shifting", action="store_true")
    # cfg zero
    parser.add_argument("--use_cfg_zero_star", action="store_true")
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)
    # stage 1
    parser.add_argument("--latent_window_size", type=int, default=9)
    # stage 2
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--stage2_num_stages", type=int, default=3)
    parser.add_argument("--stage2_timestep_shift", type=float, default=1.0)
    parser.add_argument("--stage2_scheduler_gamma", type=float, default=1 / 3)
    parser.add_argument("--stage2_stage_range", type=int, nargs="+", default=[0, 1 / 3, 2 / 3, 1])
    parser.add_argument("--stage2_num_inference_steps_list", type=int, nargs="+", default=[20, 20, 20])
    # stage 3
    parser.add_argument("--is_enable_stage3", action="store_true")
    parser.add_argument("--is_skip_first_section", action="store_true")
    parser.add_argument("--is_amplify_first_chunk", action="store_true")

    # === Prompts ===
    parser.add_argument("--use_interpolate_prompt", action="store_true")
    parser.add_argument("--interpolation_steps", type=int, default=3)
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. Medium shot focusing on the train window and the rushing scenery beyond.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    parser.add_argument(
        "--prompt_txt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--interactive_prompt_csv_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base_image_prompt_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--image_prompt_csv_path",
        type=str,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assert not (args.low_vram_mode and args.enable_compile), (
        "low_vram_mode and enable_compile cannot be used together."
    )

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_available() and "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)
        assert world_size == 1 or not args.low_vram_mode, "low_vram_mode is only for single GPU."
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    prompt = None
    image_path = None
    video_path = None
    interpolate_time_list = None
    if args.sample_type == "t2v" and args.prompt is None:
        prompt = "An extreme close-up of an gray-haired man with a beard in his 60s, he is deep in thought pondering the history of the universe as he sits at a cafe in Paris, his eyes focus on people offscreen as they walk as he sits mostly motionless, he is dressed in a wool coat suit coat with a button-down shirt , he wears a brown beret and glasses and has a very professorial appearance, and the end he offers a subtle closed-mouth smile as if he found the answer to the mystery of life, the lighting is very cinematic with the golden light and the Parisian streets and city in the background, depth of field, cinematic 35mm film."
    elif args.sample_type == "i2v" and (args.image_path is None and args.prompt is None):
        image_path = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        )
        prompt = "An astronaut hatching from an egg, on the surface of the moon, the darkness and depth of space realised in the background. High quality, ultrarealistic detail and breath-taking movie-like camera shot."
    elif args.sample_type == "v2v" and (args.video_path is None and args.prompt is None):
        video_path = (
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4"
        )
        prompt = "A robot standing on a mountain top. The sun is setting in the background."
    else:
        image_path = args.image_path
        video_path = args.video_path
        if args.interactive_prompt_csv_path is not None and args.use_interpolate_prompt:
            with open(args.prompt, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            interpolate_time_list = []
            prompt = []
            for line in lines:
                parts = line.split(",", 1)
                if len(parts) == 2:
                    time_value = int(parts[0].strip())
                    prompt_text = parts[1].strip().strip('"')

                    interpolate_time_list.append(time_value)
                    prompt.append(prompt_text)
        else:
            prompt = args.prompt

    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=args.weight_dtype,
        use_default_loader=args.use_default_loader,
    )
    transformer.set_attention_backend("_flash_3_hub")

    vae = AutoencoderKLWan.from_pretrained(
        args.base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    if args.is_enable_stage2:
        scheduler = HeliosScheduler(
            shift=args.stage2_timestep_shift,
            stages=args.stage2_num_stages,
            stage_range=args.stage2_stage_range,
            gamma=args.stage2_scheduler_gamma,
        )
        pipe = HeliosPipeline.from_pretrained(
            args.base_model_path,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            torch_dtype=args.weight_dtype,
        )
    else:
        pipe = HeliosPipeline.from_pretrained(
            args.base_model_path,
            transformer=transformer,
            vae=vae,
            torch_dtype=args.weight_dtype,
        )

    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])

    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

    if args.low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            # offload_type="leaf_level",
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    if world_size > 1 and args.enable_parallelism:
        # transformer.set_attention_backend("flash")
        pipe.transformer.enable_parallelism(config=ContextParallelConfig(ulysses_degree=world_size))

    if args.debug_mode:

        def parse_list_input(input_string):
            input_string = input_string.strip("[]").strip()
            if "," in input_string:
                return [int(x.strip()) for x in input_string.split(",") if x.strip()]
            else:
                return [int(x.strip()) for x in input_string.split() if x.strip()]

        while True:
            user_input = input("Please enter stage2_num_inference_steps_list (e.g., 10 20 30): ").strip()

            if user_input.lower() in ["q", "quit", "exit"]:
                break

            try:
                pyramid_steps = parse_list_input(user_input)
                print(f"✅ Parsing successful: {pyramid_steps}")
            except ValueError as e:
                print(f"❌ Input format error: {e}")
                print("Please re-enter...\n")
                continue

            args.stage2_num_inference_steps_list = pyramid_steps

            with torch.no_grad():
                output = pipe(
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,  # 73 109 145 181 215
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                    # stage 1
                    history_sizes=[16, 2, 1],
                    latent_window_size=args.latent_window_size,
                    is_keep_x0=True,
                    use_dynamic_shifting=args.use_dynamic_shifting,
                    # stage 2
                    is_enable_stage2=args.is_enable_stage2,
                    stage2_num_stages=args.stage2_num_stages,
                    stage2_num_inference_steps_list=args.stage2_num_inference_steps_list,
                    # stage 3
                    use_dmd=args.is_enable_stage3,
                    is_skip_first_section=args.is_skip_first_section,
                    is_amplify_first_chunk=args.is_amplify_first_chunk,
                    # cfg zero
                    use_cfg_zero_star=args.use_cfg_zero_star,
                    use_zero_init=args.use_zero_init,
                    zero_steps=args.zero_steps,
                    # i2v
                    image=load_image(image_path).resize((args.width, args.height)) if image_path is not None else None,
                    # t2v
                    video=load_video(video_path) if video_path is not None else None,
                    # interpolate_prompt
                    use_interpolate_prompt=args.use_interpolate_prompt,
                    interpolation_steps=args.interpolation_steps,
                    interpolate_time_list=interpolate_time_list,
                ).frames[0]

            if not args.enable_parallelism or rank == 0:
                file_count = len(
                    [f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))]
                )
                output_path = os.path.join(
                    args.output_folder, f"{file_count:04d}_{args.sample_type}_{int(time.time())}.mp4"
                )
                export_to_video(output, output_path, fps=24)
    elif args.prompt_txt_path is not None:
        with open(args.prompt_txt_path, "r") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        if not args.enable_parallelism:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]
            prompt_list_with_idx = prompt_list_with_idx[rank::world_size]
        else:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]

        for idx, prompt in tqdm(prompt_list_with_idx, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{idx}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,  # 73 109 145 181 215
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        latent_window_size=args.latent_window_size,
                        is_keep_x0=True,
                        use_dynamic_shifting=args.use_dynamic_shifting,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        stage2_num_stages=args.stage2_num_stages,
                        stage2_num_inference_steps_list=args.stage2_num_inference_steps_list,
                        # stage 3
                        use_dmd=args.is_enable_stage3,
                        is_skip_first_section=args.is_skip_first_section,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_cfg_zero_star=args.use_cfg_zero_star,
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        # t2v
                        video=load_video(video_path) if video_path is not None else None,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=24)
    elif args.interactive_prompt_csv_path is not None:
        df = pd.read_csv(args.interactive_prompt_csv_path)

        df = df.sort_values(by=["id", "prompt_index"])
        all_video_ids = df["id"].unique()

        if not args.enable_parallelism:
            my_video_ids = all_video_ids[rank::world_size]
        else:
            my_video_ids = all_video_ids

        for video_id in tqdm(my_video_ids, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{video_id}.mp4")

            if os.path.exists(output_path):
                print(f"skipping {output_path}!")
                continue

            group_df = df[df["id"] == video_id]

            if "refined_prompt" in df.columns:
                prompt_list = group_df["refined_prompt"].fillna(group_df["prompt"]).tolist()
            else:
                prompt_list = group_df["prompt"].tolist()
            interpolate_time_list = [7] * len(prompt_list)

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt_list,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,  # 73 109 145 181 215
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        latent_window_size=args.latent_window_size,
                        is_keep_x0=True,
                        use_dynamic_shifting=args.use_dynamic_shifting,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        stage2_num_stages=args.stage2_num_stages,
                        stage2_num_inference_steps_list=args.stage2_num_inference_steps_list,
                        # stage 3
                        use_dmd=args.is_enable_stage3,
                        is_skip_first_section=args.is_skip_first_section,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_cfg_zero_star=args.use_cfg_zero_star,
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        # t2v
                        video=load_video(video_path) if video_path is not None else None,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=24)
    elif args.image_prompt_csv_path is not None:
        df = pd.read_csv(args.image_prompt_csv_path)
        if not args.enable_parallelism:
            df = df.iloc[rank::world_size]

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing prompts"):
            # output_path = os.path.join(args.output_folder, f"{idx}.mp4")
            output_path = os.path.join(args.output_folder, f"{row['id']}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            prompt = row.get("refined_prompt") or row["prompt"]
            image_path = os.path.join(args.base_image_prompt_path, row["image_name"])

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        height=args.height,
                        width=args.width,
                        num_frames=args.num_frames,  # 73 109 145 181 215
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        latent_window_size=args.latent_window_size,
                        is_keep_x0=True,
                        use_dynamic_shifting=args.use_dynamic_shifting,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        stage2_num_stages=args.stage2_num_stages,
                        stage2_num_inference_steps_list=args.stage2_num_inference_steps_list,
                        # stage 3
                        use_dmd=args.is_enable_stage3,
                        is_skip_first_section=args.is_skip_first_section,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_cfg_zero_star=args.use_cfg_zero_star,
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # i2v
                        image=load_image(image_path).resize((args.width, args.height))
                        if image_path is not None
                        else None,
                        # t2v
                        video=load_video(video_path) if video_path is not None else None,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                    ).frames[0]
                except Exception:
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=24)
    else:
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,  # 73 109 145 181 215
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                # stage 1
                history_sizes=[16, 2, 1],
                latent_window_size=args.latent_window_size,
                is_keep_x0=True,
                use_dynamic_shifting=args.use_dynamic_shifting,
                # stage 2
                is_enable_stage2=args.is_enable_stage2,
                stage2_num_stages=args.stage2_num_stages,
                stage2_num_inference_steps_list=args.stage2_num_inference_steps_list,
                # stage 3
                use_dmd=args.is_enable_stage3,
                is_skip_first_section=args.is_skip_first_section,
                is_amplify_first_chunk=args.is_amplify_first_chunk,
                # cfg zero
                use_cfg_zero_star=args.use_cfg_zero_star,
                use_zero_init=args.use_zero_init,
                zero_steps=args.zero_steps,
                # i2v
                image=load_image(image_path).resize((args.width, args.height)) if image_path is not None else None,
                # t2v
                video=load_video(video_path) if video_path is not None else None,
                # interpolate_prompt
                use_interpolate_prompt=args.use_interpolate_prompt,
                interpolation_steps=args.interpolation_steps,
                interpolate_time_list=interpolate_time_list,
            ).frames[0]

        if not args.enable_parallelism or rank == 0:
            file_count = len(
                [f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))]
            )
            output_path = os.path.join(
                args.output_folder, f"{file_count:04d}_{args.sample_type}_{int(time.time())}.mp4"
            )
            export_to_video(output, output_path, fps=24)

    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


if __name__ == "__main__":
    main()
