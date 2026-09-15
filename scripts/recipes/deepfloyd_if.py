import argparse
import os

import numpy as np
import torch
import yaml
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, T5Tokenizer

from diffusers import DDPMScheduler, IFPipeline, IFSuperResolutionPipeline, UNet2DConditionModel
from diffusers.loaders.conversion import get_conversion
from diffusers.loaders.conversion.configs.deepfloyd_if import (
    create_unet_diffusers_config,
    superres_create_unet_diffusers_config,
)
from diffusers.pipelines.deepfloyd_if.safety_checker import IFSafetyChecker


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dump_path", required=False, default=None, type=str)

    parser.add_argument("--dump_path_stage_2", required=False, default=None, type=str)

    parser.add_argument("--dump_path_stage_3", required=False, default=None, type=str)

    parser.add_argument("--unet_config", required=False, default=None, type=str, help="Path to unet config file")

    parser.add_argument(
        "--unet_checkpoint_path", required=False, default=None, type=str, help="Path to unet checkpoint file"
    )

    parser.add_argument(
        "--unet_checkpoint_path_stage_2",
        required=False,
        default=None,
        type=str,
        help="Path to stage 2 unet checkpoint file",
    )

    parser.add_argument(
        "--unet_checkpoint_path_stage_3",
        required=False,
        default=None,
        type=str,
        help="Path to stage 3 unet checkpoint file",
    )

    parser.add_argument("--p_head_path", type=str, required=True)

    parser.add_argument("--w_head_path", type=str, required=True)

    args = parser.parse_args()

    return args


def main(args):
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    text_encoder = T5EncoderModel.from_pretrained("google/t5-v1_1-xxl")

    feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    safety_checker = convert_safety_checker(p_head_path=args.p_head_path, w_head_path=args.w_head_path)

    if args.unet_config is not None and args.unet_checkpoint_path is not None and args.dump_path is not None:
        convert_stage_1_pipeline(tokenizer, text_encoder, feature_extractor, safety_checker, args)

    if args.unet_checkpoint_path_stage_2 is not None and args.dump_path_stage_2 is not None:
        convert_super_res_pipeline(tokenizer, text_encoder, feature_extractor, safety_checker, args, stage=2)

    if args.unet_checkpoint_path_stage_3 is not None and args.dump_path_stage_3 is not None:
        convert_super_res_pipeline(tokenizer, text_encoder, feature_extractor, safety_checker, args, stage=3)


def convert_stage_1_pipeline(tokenizer, text_encoder, feature_extractor, safety_checker, args):
    unet = get_stage_1_unet(args.unet_config, args.unet_checkpoint_path)

    scheduler = DDPMScheduler(
        variance_type="learned_range",
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        thresholding=True,
        dynamic_thresholding_ratio=0.95,
        sample_max_value=1.5,
    )

    pipe = IFPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=True,
    )

    pipe.save_pretrained(args.dump_path)


def convert_super_res_pipeline(tokenizer, text_encoder, feature_extractor, safety_checker, args, stage):
    if stage == 2:
        unet_checkpoint_path = args.unet_checkpoint_path_stage_2
        sample_size = None
        dump_path = args.dump_path_stage_2
    elif stage == 3:
        unet_checkpoint_path = args.unet_checkpoint_path_stage_3
        sample_size = 1024
        dump_path = args.dump_path_stage_3
    else:
        assert False

    unet = get_super_res_unet(unet_checkpoint_path, verify_param_count=False, sample_size=sample_size)

    image_noising_scheduler = DDPMScheduler(
        beta_schedule="squaredcos_cap_v2",
    )

    scheduler = DDPMScheduler(
        variance_type="learned_range",
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        thresholding=True,
        dynamic_thresholding_ratio=0.95,
        sample_max_value=1.0,
    )

    pipe = IFSuperResolutionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=unet,
        scheduler=scheduler,
        image_noising_scheduler=image_noising_scheduler,
        safety_checker=safety_checker,
        feature_extractor=feature_extractor,
        requires_safety_checker=True,
    )

    pipe.save_pretrained(dump_path)


def get_stage_1_unet(unet_config, unet_checkpoint_path):
    original_unet_config = yaml.safe_load(unet_config)
    original_unet_config = original_unet_config["params"]

    unet_diffusers_config = create_unet_diffusers_config(original_unet_config)

    unet = UNet2DConditionModel(**unet_diffusers_config)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    unet_checkpoint = torch.load(unet_checkpoint_path, map_location=device)

    converted_unet_checkpoint = convert_ldm_unet_checkpoint(
        unet_checkpoint, unet_diffusers_config, path=unet_checkpoint_path
    )

    unet.load_state_dict(converted_unet_checkpoint)

    return unet


def convert_safety_checker(p_head_path, w_head_path):
    state = {}
    for name, path in (("p_head", p_head_path), ("w_head", w_head_path)):
        with np.load(path) as archive:
            state.update({f"{name}.{key}": torch.from_numpy(archive[key]) for key in ("weights", "biases")})
    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    state.update({"vision_model." + key: value for key, value in vision_model.state_dict().items()})
    config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
    model = IFSafetyChecker(config)
    model.load_state_dict(get_conversion("IFSafetyChecker", config.to_dict()).to_diffusers(state), strict=True)
    return model


def convert_ldm_unet_checkpoint(unet_state_dict, config, path=None, **kwargs):
    return get_conversion("UNet2DConditionModel", config).to_diffusers(unet_state_dict)


# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)


# below is copy and pasted from original convert_if_stage_2.py script


def get_super_res_unet(unet_checkpoint_path, verify_param_count=True, sample_size=None):
    orig_path = unet_checkpoint_path

    with open(os.path.join(orig_path, "config.yml")) as handle:
        original_unet_config = yaml.safe_load(handle)
    original_unet_config = original_unet_config["params"]

    unet_diffusers_config = superres_create_unet_diffusers_config(original_unet_config)
    unet_diffusers_config["time_embedding_dim"] = original_unet_config["model_channels"] * int(
        original_unet_config["channel_mult"].split(",")[-1]
    )
    if original_unet_config["encoder_dim"] != original_unet_config["encoder_channels"]:
        unet_diffusers_config["encoder_hid_dim"] = original_unet_config["encoder_dim"]
        unet_diffusers_config["class_embed_type"] = "timestep"
        unet_diffusers_config["addition_embed_type"] = "text"

    unet_diffusers_config["time_embedding_act_fn"] = "gelu"
    unet_diffusers_config["resnet_skip_time_act"] = True
    unet_diffusers_config["resnet_out_scale_factor"] = 1 / 0.7071
    unet_diffusers_config["mid_block_scale_factor"] = 1 / 0.7071
    unet_diffusers_config["only_cross_attention"] = (
        bool(original_unet_config["disable_self_attentions"])
        if (
            "disable_self_attentions" in original_unet_config
            and isinstance(original_unet_config["disable_self_attentions"], int)
        )
        else True
    )

    if sample_size is None:
        unet_diffusers_config["sample_size"] = original_unet_config["image_size"]
    else:
        # The second upscaler unet's sample size is incorrectly specified
        # in the config and is instead hardcoded in source
        unet_diffusers_config["sample_size"] = sample_size

    unet_checkpoint = torch.load(os.path.join(unet_checkpoint_path, "pytorch_model.bin"), map_location="cpu")

    if verify_param_count:
        # check that architecture matches - is a bit slow
        verify_parameter_count(orig_path, unet_diffusers_config)

    converted_unet_checkpoint = superres_convert_ldm_unet_checkpoint(
        unet_checkpoint, unet_diffusers_config, path=unet_checkpoint_path
    )
    converted_keys = converted_unet_checkpoint.keys()

    model = UNet2DConditionModel(**unet_diffusers_config)
    expected_weights = model.state_dict().keys()

    diff_c_e = set(converted_keys) - set(expected_weights)
    diff_e_c = set(expected_weights) - set(converted_keys)

    assert len(diff_e_c) == 0, f"Expected, but not converted: {diff_e_c}"
    assert len(diff_c_e) == 0, f"Converted, but not expected: {diff_c_e}"

    model.load_state_dict(converted_unet_checkpoint)

    return model


def superres_convert_ldm_unet_checkpoint(unet_state_dict, config, path=None, **kwargs):
    return get_conversion("UNet2DConditionModel", config).to_diffusers(unet_state_dict)


def verify_parameter_count(orig_path, unet_diffusers_config):
    if "-II-" in orig_path:
        from deepfloyd_if.modules import IFStageII

        if_II = IFStageII(device="cpu", dir_or_name=orig_path)
    elif "-III-" in orig_path:
        from deepfloyd_if.modules import IFStageIII

        if_II = IFStageIII(device="cpu", dir_or_name=orig_path)
    else:
        assert f"Weird name. Should have -II- or -III- in path: {orig_path}"

    unet = UNet2DConditionModel(**unet_diffusers_config)

    # in params
    assert_param_count(unet.time_embedding, if_II.model.time_embed)
    assert_param_count(unet.conv_in, if_II.model.input_blocks[:1])

    # downblocks
    assert_param_count(unet.down_blocks[0], if_II.model.input_blocks[1:4])
    assert_param_count(unet.down_blocks[1], if_II.model.input_blocks[4:7])
    assert_param_count(unet.down_blocks[2], if_II.model.input_blocks[7:11])

    if "-II-" in orig_path:
        assert_param_count(unet.down_blocks[3], if_II.model.input_blocks[11:17])
        assert_param_count(unet.down_blocks[4], if_II.model.input_blocks[17:])
    if "-III-" in orig_path:
        assert_param_count(unet.down_blocks[3], if_II.model.input_blocks[11:15])
        assert_param_count(unet.down_blocks[4], if_II.model.input_blocks[15:20])
        assert_param_count(unet.down_blocks[5], if_II.model.input_blocks[20:])

    # mid block
    assert_param_count(unet.mid_block, if_II.model.middle_block)

    # up block
    if "-II-" in orig_path:
        assert_param_count(unet.up_blocks[0], if_II.model.output_blocks[:6])
        assert_param_count(unet.up_blocks[1], if_II.model.output_blocks[6:12])
        assert_param_count(unet.up_blocks[2], if_II.model.output_blocks[12:16])
        assert_param_count(unet.up_blocks[3], if_II.model.output_blocks[16:19])
        assert_param_count(unet.up_blocks[4], if_II.model.output_blocks[19:])
    if "-III-" in orig_path:
        assert_param_count(unet.up_blocks[0], if_II.model.output_blocks[:5])
        assert_param_count(unet.up_blocks[1], if_II.model.output_blocks[5:10])
        assert_param_count(unet.up_blocks[2], if_II.model.output_blocks[10:14])
        assert_param_count(unet.up_blocks[3], if_II.model.output_blocks[14:18])
        assert_param_count(unet.up_blocks[4], if_II.model.output_blocks[18:21])
        assert_param_count(unet.up_blocks[5], if_II.model.output_blocks[21:24])

    # out params
    assert_param_count(unet.conv_norm_out, if_II.model.out[0])
    assert_param_count(unet.conv_out, if_II.model.out[2])

    # make sure all model architecture has same param count
    assert_param_count(unet, if_II.model)


def assert_param_count(model_1, model_2):
    count_1 = sum(p.numel() for p in model_1.parameters())
    count_2 = sum(p.numel() for p in model_2.parameters())
    assert count_1 == count_2, f"{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}"


if __name__ == "__main__":
    main(parse_args())
