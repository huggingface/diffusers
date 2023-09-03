import argparse
import sys

import tensorrt as trt


def convert_models(onnx_path: str, num_controlnet: int, output_path: str, fp16: bool = False, sd_xl: bool = False):
    """
    Function to convert models in stable diffusion controlnet pipeline into TensorRT format

    Example:
    python convert_stable_diffusion_controlnet_to_tensorrt.py
    --onnx_path path-to-models-stable_diffusion/RevAnimated-v1-2-2/unet/model.onnx
    --output_path path-to-models-stable_diffusion/RevAnimated-v1-2-2/unet/model.engine
    --fp16
    --num_controlnet 2

    Example for SD XL:
    python convert_stable_diffusion_controlnet_to_tensorrt.py
    --onnx_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0/unet/model.onnx
    --output_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0/unet/model.engine
    --fp16
    --num_controlnet 1
    --sd_xl

    Returns:
        unet/model.engine

        run test script in diffusers/examples/community
        python test_onnx_controlnet.py
        --sd_model danbrown/RevAnimated-v1-2-2
        --onnx_model_dir path-to-models-stable_diffusion/RevAnimated-v1-2-2
        --unet_engine_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0/unet/model.engine
        --qr_img_path path-to-qr-code-image
    """
    # UNET
    if sd_xl:
        batch_size = 1
        unet_in_channels = 4
        unet_sample_size = 64
        num_tokens = 77
        text_hidden_size = 2048
        img_size = 512

        text_embeds_shape = (2 * batch_size, 1280)
        time_ids_shape = (2 * batch_size, 6)
    else:
        batch_size = 1
        unet_in_channels = 4
        unet_sample_size = 64
        num_tokens = 77
        text_hidden_size = 768
        img_size = 512
        batch_size = 1

    latents_shape = (2 * batch_size, unet_in_channels, unet_sample_size, unet_sample_size)
    embed_shape = (2 * batch_size, num_tokens, text_hidden_size)
    controlnet_conds_shape = (num_controlnet, 2 * batch_size, 3, img_size, img_size)

    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
    TRT_BUILDER = trt.Builder(TRT_LOGGER)
    TRT_RUNTIME = trt.Runtime(TRT_LOGGER)

    network = TRT_BUILDER.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    onnx_parser = trt.OnnxParser(network, TRT_LOGGER)

    parse_success = onnx_parser.parse_from_file(onnx_path)
    for idx in range(onnx_parser.num_errors):
        print(onnx_parser.get_error(idx))
    if not parse_success:
        sys.exit("ONNX model parsing failed")
    print("Load Onnx model done")

    profile = TRT_BUILDER.create_optimization_profile()

    profile.set_shape("sample", latents_shape, latents_shape, latents_shape)
    profile.set_shape("encoder_hidden_states", embed_shape, embed_shape, embed_shape)
    profile.set_shape("controlnet_conds", controlnet_conds_shape, controlnet_conds_shape, controlnet_conds_shape)
    if sd_xl:
        profile.set_shape("text_embeds", text_embeds_shape, text_embeds_shape, text_embeds_shape)
        profile.set_shape("time_ids", time_ids_shape, time_ids_shape, time_ids_shape)

    config = TRT_BUILDER.create_builder_config()
    config.add_optimization_profile(profile)
    config.set_preview_feature(trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805, True)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    plan = TRT_BUILDER.build_serialized_network(network, config)
    if plan is None:
        sys.exit("Failed building engine")
    print("Succeeded building engine")

    engine = TRT_RUNTIME.deserialize_cuda_engine(plan)

    ## save TRT engine
    with open(output_path, "wb") as f:
        f.write(engine.serialize())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")

    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the onnx checkpoint to convert",
    )

    parser.add_argument("--num_controlnet", type=int)

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    convert_models(args.onnx_path, args.num_controlnet, args.output_path, args.fp16, args.sd_xl)
