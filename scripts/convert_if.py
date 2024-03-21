import argparse
import inspect
import os

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, T5Tokenizer

from diffusers import DDPMScheduler, IFPipeline, IFSuperResolutionPipeline, UNet2DConditionModel
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
    state_dict = {}

    # p head

    p_head = np.load(p_head_path)

    p_head_weights = p_head["weights"]
    p_head_weights = torch.from_numpy(p_head_weights)
    p_head_weights = p_head_weights.unsqueeze(0)

    p_head_biases = p_head["biases"]
    p_head_biases = torch.from_numpy(p_head_biases)
    p_head_biases = p_head_biases.unsqueeze(0)

    state_dict["p_head.weight"] = p_head_weights
    state_dict["p_head.bias"] = p_head_biases

    # w head

    w_head = np.load(w_head_path)

    w_head_weights = w_head["weights"]
    w_head_weights = torch.from_numpy(w_head_weights)
    w_head_weights = w_head_weights.unsqueeze(0)

    w_head_biases = w_head["biases"]
    w_head_biases = torch.from_numpy(w_head_biases)
    w_head_biases = w_head_biases.unsqueeze(0)

    state_dict["w_head.weight"] = w_head_weights
    state_dict["w_head.bias"] = w_head_biases

    # vision model

    vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    vision_model_state_dict = vision_model.state_dict()

    for key, value in vision_model_state_dict.items():
        key = f"vision_model.{key}"
        state_dict[key] = value

    # full model

    config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
    safety_checker = IFSafetyChecker(config)

    safety_checker.load_state_dict(state_dict)

    return safety_checker


def create_unet_diffusers_config(original_unet_config, class_embed_type=None):
    attention_resolutions = parse_list(original_unet_config["attention_resolutions"])
    attention_resolutions = [original_unet_config["image_size"] // int(res) for res in attention_resolutions]

    channel_mult = parse_list(original_unet_config["channel_mult"])
    block_out_channels = [original_unet_config["model_channels"] * mult for mult in channel_mult]

    down_block_types = []
    resolution = 1

    for i in range(len(block_out_channels)):
        if resolution in attention_resolutions:
            block_type = "SimpleCrossAttnDownBlock2D"
        elif original_unet_config["resblock_updown"]:
            block_type = "ResnetDownsampleBlock2D"
        else:
            block_type = "DownBlock2D"

        down_block_types.append(block_type)

        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        if resolution in attention_resolutions:
            block_type = "SimpleCrossAttnUpBlock2D"
        elif original_unet_config["resblock_updown"]:
            block_type = "ResnetUpsampleBlock2D"
        else:
            block_type = "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    head_dim = original_unet_config["num_head_channels"]

    use_linear_projection = (
        original_unet_config["use_linear_in_transformer"]
        if "use_linear_in_transformer" in original_unet_config
        else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim = [5, 10, 20, 20]

    projection_class_embeddings_input_dim = None

    if class_embed_type is None:
        if "num_classes" in original_unet_config:
            if original_unet_config["num_classes"] == "sequential":
                class_embed_type = "projection"
                assert "adm_in_channels" in original_unet_config
                projection_class_embeddings_input_dim = original_unet_config["adm_in_channels"]
            else:
                raise NotImplementedError(
                    f"Unknown conditional unet num_classes config: {original_unet_config['num_classes']}"
                )

    config = {
        "sample_size": original_unet_config["image_size"],
        "in_channels": original_unet_config["in_channels"],
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": original_unet_config["num_res_blocks"],
        "cross_attention_dim": original_unet_config["encoder_channels"],
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "out_channels": original_unet_config["out_channels"],
        "up_block_types": tuple(up_block_types),
        "upcast_attention": False,  # TODO: guessing
        "cross_attention_norm": "group_norm",
        "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
        "addition_embed_type": "text",
        "act_fn": "gelu",
    }

    if original_unet_config["use_scale_shift_norm"]:
        config["resnet_time_scale_shift"] = "scale_shift"

    if "encoder_dim" in original_unet_config:
        config["encoder_hid_dim"] = original_unet_config["encoder_dim"]

    return config


def convert_ldm_unet_checkpoint(unet_state_dict, config, path=None):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    if config["class_embed_type"] in [None, "identity"]:
        # No parameters to port
        ...
    elif config["class_embed_type"] == "timestep" or config["class_embed_type"] == "projection":
        new_checkpoint["class_embedding.linear_1.weight"] = unet_state_dict["label_emb.0.0.weight"]
        new_checkpoint["class_embedding.linear_1.bias"] = unet_state_dict["label_emb.0.0.bias"]
        new_checkpoint["class_embedding.linear_2.weight"] = unet_state_dict["label_emb.0.2.weight"]
        new_checkpoint["class_embedding.linear_2.bias"] = unet_state_dict["label_emb.0.2.bias"]
    else:
        raise NotImplementedError(f"Not implemented `class_embed_type`: {config['class_embed_type']}")

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key]
        for layer_id in range(num_output_blocks)
    }

    for i in range(1, num_input_blocks):
        block_id = (i - 1) // (config["layers_per_block"] + 1)
        layer_in_block_id = (i - 1) % (config["layers_per_block"] + 1)

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)

        # TODO need better check than i in [4, 8, 12, 16]
        block_type = config["down_block_types"][block_id]
        if (block_type == "ResnetDownsampleBlock2D" or block_type == "SimpleCrossAttnDownBlock2D") and i in [
            4,
            8,
            12,
            16,
        ]:
            meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.downsamplers.0"}
        else:
            meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}

        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            old_path = f"input_blocks.{i}.1"
            new_path = f"down_blocks.{block_id}.attentions.{layer_in_block_id}"

            assign_attention_to_checkpoint(
                new_checkpoint=new_checkpoint,
                unet_state_dict=unet_state_dict,
                old_path=old_path,
                new_path=new_path,
                config=config,
            )

            paths = renew_attention_paths(attentions)
            meta_path = {"old": old_path, "new": new_path}
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    old_path = "middle_block.1"
    new_path = "mid_block.attentions.0"

    assign_attention_to_checkpoint(
        new_checkpoint=new_checkpoint,
        unet_state_dict=unet_state_dict,
        old_path=old_path,
        new_path=new_path,
        config=config,
    )

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )

    for i in range(num_output_blocks):
        block_id = i // (config["layers_per_block"] + 1)
        layer_in_block_id = i % (config["layers_per_block"] + 1)
        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        # len(output_block_list) == 1 -> resnet
        # len(output_block_list) == 2 -> resnet, attention
        # len(output_block_list) == 3 -> resnet, attention, upscale resnet

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]
            attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}

            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # Clear attentions as they have been attributed above.
                if len(attentions) == 2:
                    attentions = []

            if len(attentions):
                old_path = f"output_blocks.{i}.1"
                new_path = f"up_blocks.{block_id}.attentions.{layer_in_block_id}"

                assign_attention_to_checkpoint(
                    new_checkpoint=new_checkpoint,
                    unet_state_dict=unet_state_dict,
                    old_path=old_path,
                    new_path=new_path,
                    config=config,
                )

                paths = renew_attention_paths(attentions)
                meta_path = {
                    "old": old_path,
                    "new": new_path,
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )

            if len(output_block_list) == 3:
                resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.2" in key]
                paths = renew_resnet_paths(resnets)
                meta_path = {"old": f"output_blocks.{i}.2", "new": f"up_blocks.{block_id}.upsamplers.0"}
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    if "encoder_proj.weight" in unet_state_dict:
        new_checkpoint["encoder_hid_proj.weight"] = unet_state_dict.pop("encoder_proj.weight")
        new_checkpoint["encoder_hid_proj.bias"] = unet_state_dict.pop("encoder_proj.bias")

    if "encoder_pooling.0.weight" in unet_state_dict:
        new_checkpoint["add_embedding.norm1.weight"] = unet_state_dict.pop("encoder_pooling.0.weight")
        new_checkpoint["add_embedding.norm1.bias"] = unet_state_dict.pop("encoder_pooling.0.bias")

        new_checkpoint["add_embedding.pool.positional_embedding"] = unet_state_dict.pop(
            "encoder_pooling.1.positional_embedding"
        )
        new_checkpoint["add_embedding.pool.k_proj.weight"] = unet_state_dict.pop("encoder_pooling.1.k_proj.weight")
        new_checkpoint["add_embedding.pool.k_proj.bias"] = unet_state_dict.pop("encoder_pooling.1.k_proj.bias")
        new_checkpoint["add_embedding.pool.q_proj.weight"] = unet_state_dict.pop("encoder_pooling.1.q_proj.weight")
        new_checkpoint["add_embedding.pool.q_proj.bias"] = unet_state_dict.pop("encoder_pooling.1.q_proj.bias")
        new_checkpoint["add_embedding.pool.v_proj.weight"] = unet_state_dict.pop("encoder_pooling.1.v_proj.weight")
        new_checkpoint["add_embedding.pool.v_proj.bias"] = unet_state_dict.pop("encoder_pooling.1.v_proj.bias")

        new_checkpoint["add_embedding.proj.weight"] = unet_state_dict.pop("encoder_pooling.2.weight")
        new_checkpoint["add_embedding.proj.bias"] = unet_state_dict.pop("encoder_pooling.2.bias")

        new_checkpoint["add_embedding.norm2.weight"] = unet_state_dict.pop("encoder_pooling.3.weight")
        new_checkpoint["add_embedding.norm2.bias"] = unet_state_dict.pop("encoder_pooling.3.bias")

    return new_checkpoint


def shave_segments(path, n_shave_prefix_segments=1):
    """
    Removes segments. Positive values shave the first segments, negative shave the last segments.
    """
    if n_shave_prefix_segments >= 0:
        return ".".join(path.split(".")[n_shave_prefix_segments:])
    else:
        return ".".join(path.split(".")[:n_shave_prefix_segments])


def renew_resnet_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside resnets to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item.replace("in_layers.0", "norm1")
        new_item = new_item.replace("in_layers.2", "conv1")

        new_item = new_item.replace("out_layers.0", "norm2")
        new_item = new_item.replace("out_layers.3", "conv2")

        new_item = new_item.replace("emb_layers.1", "time_emb_proj")
        new_item = new_item.replace("skip_connection", "conv_shortcut")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def renew_attention_paths(old_list, n_shave_prefix_segments=0):
    """
    Updates paths inside attentions to the new naming scheme (local renaming)
    """
    mapping = []
    for old_item in old_list:
        new_item = old_item

        if "qkv" in new_item:
            continue

        if "encoder_kv" in new_item:
            continue

        new_item = new_item.replace("norm.weight", "group_norm.weight")
        new_item = new_item.replace("norm.bias", "group_norm.bias")

        new_item = new_item.replace("proj_out.weight", "to_out.0.weight")
        new_item = new_item.replace("proj_out.bias", "to_out.0.bias")

        new_item = new_item.replace("norm_encoder.weight", "norm_cross.weight")
        new_item = new_item.replace("norm_encoder.bias", "norm_cross.bias")

        new_item = shave_segments(new_item, n_shave_prefix_segments=n_shave_prefix_segments)

        mapping.append({"old": old_item, "new": new_item})

    return mapping


def assign_attention_to_checkpoint(new_checkpoint, unet_state_dict, old_path, new_path, config):
    qkv_weight = unet_state_dict.pop(f"{old_path}.qkv.weight")
    qkv_weight = qkv_weight[:, :, 0]

    qkv_bias = unet_state_dict.pop(f"{old_path}.qkv.bias")

    is_cross_attn_only = "only_cross_attention" in config and config["only_cross_attention"]

    split = 1 if is_cross_attn_only else 3

    weights, bias = split_attentions(
        weight=qkv_weight,
        bias=qkv_bias,
        split=split,
        chunk_size=config["attention_head_dim"],
    )

    if is_cross_attn_only:
        query_weight, q_bias = weights, bias
        new_checkpoint[f"{new_path}.to_q.weight"] = query_weight[0]
        new_checkpoint[f"{new_path}.to_q.bias"] = q_bias[0]
    else:
        [query_weight, key_weight, value_weight], [q_bias, k_bias, v_bias] = weights, bias
        new_checkpoint[f"{new_path}.to_q.weight"] = query_weight
        new_checkpoint[f"{new_path}.to_q.bias"] = q_bias
        new_checkpoint[f"{new_path}.to_k.weight"] = key_weight
        new_checkpoint[f"{new_path}.to_k.bias"] = k_bias
        new_checkpoint[f"{new_path}.to_v.weight"] = value_weight
        new_checkpoint[f"{new_path}.to_v.bias"] = v_bias

    encoder_kv_weight = unet_state_dict.pop(f"{old_path}.encoder_kv.weight")
    encoder_kv_weight = encoder_kv_weight[:, :, 0]

    encoder_kv_bias = unet_state_dict.pop(f"{old_path}.encoder_kv.bias")

    [encoder_k_weight, encoder_v_weight], [encoder_k_bias, encoder_v_bias] = split_attentions(
        weight=encoder_kv_weight,
        bias=encoder_kv_bias,
        split=2,
        chunk_size=config["attention_head_dim"],
    )

    new_checkpoint[f"{new_path}.add_k_proj.weight"] = encoder_k_weight
    new_checkpoint[f"{new_path}.add_k_proj.bias"] = encoder_k_bias
    new_checkpoint[f"{new_path}.add_v_proj.weight"] = encoder_v_weight
    new_checkpoint[f"{new_path}.add_v_proj.bias"] = encoder_v_bias


def assign_to_checkpoint(paths, checkpoint, old_checkpoint, additional_replacements=None, config=None):
    """
    This does the final conversion step: take locally converted weights and apply a global renaming to them. It splits
    attention layers, and takes into account additional replacements that may arise.

    Assigns the weights to the new checkpoint.
    """
    assert isinstance(paths, list), "Paths should be a list of dicts containing 'old' and 'new' keys."

    for path in paths:
        new_path = path["new"]

        # Global renaming happens here
        new_path = new_path.replace("middle_block.0", "mid_block.resnets.0")
        new_path = new_path.replace("middle_block.1", "mid_block.attentions.0")
        new_path = new_path.replace("middle_block.2", "mid_block.resnets.1")

        if additional_replacements is not None:
            for replacement in additional_replacements:
                new_path = new_path.replace(replacement["old"], replacement["new"])

        # proj_attn.weight has to be converted from conv 1D to linear
        if "proj_attn.weight" in new_path or "to_out.0.weight" in new_path:
            checkpoint[new_path] = old_checkpoint[path["old"]][:, :, 0]
        else:
            checkpoint[new_path] = old_checkpoint[path["old"]]


# TODO maybe document and/or can do more efficiently (build indices in for loop and extract once for each split?)
def split_attentions(*, weight, bias, split, chunk_size):
    weights = [None] * split
    biases = [None] * split

    weights_biases_idx = 0

    for starting_row_index in range(0, weight.shape[0], chunk_size):
        row_indices = torch.arange(starting_row_index, starting_row_index + chunk_size)

        weight_rows = weight[row_indices, :]
        bias_rows = bias[row_indices]

        if weights[weights_biases_idx] is None:
            weights[weights_biases_idx] = weight_rows
            biases[weights_biases_idx] = bias_rows
        else:
            assert weights[weights_biases_idx] is not None
            weights[weights_biases_idx] = torch.concat([weights[weights_biases_idx], weight_rows])
            biases[weights_biases_idx] = torch.concat([biases[weights_biases_idx], bias_rows])

        weights_biases_idx = (weights_biases_idx + 1) % split

    return weights, biases


def parse_list(value):
    if isinstance(value, str):
        value = value.split(",")
        value = [int(v) for v in value]
    elif isinstance(value, list):
        pass
    else:
        raise ValueError(f"Can't parse list for type: {type(value)}")

    return value


# below is copy and pasted from original convert_if_stage_2.py script


def get_super_res_unet(unet_checkpoint_path, verify_param_count=True, sample_size=None):
    orig_path = unet_checkpoint_path

    original_unet_config = yaml.safe_load(os.path.join(orig_path, "config.yml"))
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
        verify_param_count(orig_path, unet_diffusers_config)

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


def superres_create_unet_diffusers_config(original_unet_config):
    attention_resolutions = parse_list(original_unet_config["attention_resolutions"])
    attention_resolutions = [original_unet_config["image_size"] // int(res) for res in attention_resolutions]

    channel_mult = parse_list(original_unet_config["channel_mult"])
    block_out_channels = [original_unet_config["model_channels"] * mult for mult in channel_mult]

    down_block_types = []
    resolution = 1

    for i in range(len(block_out_channels)):
        if resolution in attention_resolutions:
            block_type = "SimpleCrossAttnDownBlock2D"
        elif original_unet_config["resblock_updown"]:
            block_type = "ResnetDownsampleBlock2D"
        else:
            block_type = "DownBlock2D"

        down_block_types.append(block_type)

        if i != len(block_out_channels) - 1:
            resolution *= 2

    up_block_types = []
    for i in range(len(block_out_channels)):
        if resolution in attention_resolutions:
            block_type = "SimpleCrossAttnUpBlock2D"
        elif original_unet_config["resblock_updown"]:
            block_type = "ResnetUpsampleBlock2D"
        else:
            block_type = "UpBlock2D"
        up_block_types.append(block_type)
        resolution //= 2

    head_dim = original_unet_config["num_head_channels"]
    use_linear_projection = (
        original_unet_config["use_linear_in_transformer"]
        if "use_linear_in_transformer" in original_unet_config
        else False
    )
    if use_linear_projection:
        # stable diffusion 2-base-512 and 2-768
        if head_dim is None:
            head_dim = [5, 10, 20, 20]

    class_embed_type = None
    projection_class_embeddings_input_dim = None

    if "num_classes" in original_unet_config:
        if original_unet_config["num_classes"] == "sequential":
            class_embed_type = "projection"
            assert "adm_in_channels" in original_unet_config
            projection_class_embeddings_input_dim = original_unet_config["adm_in_channels"]
        else:
            raise NotImplementedError(
                f"Unknown conditional unet num_classes config: {original_unet_config['num_classes']}"
            )

    config = {
        "in_channels": original_unet_config["in_channels"],
        "down_block_types": tuple(down_block_types),
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": tuple(original_unet_config["num_res_blocks"]),
        "cross_attention_dim": original_unet_config["encoder_channels"],
        "attention_head_dim": head_dim,
        "use_linear_projection": use_linear_projection,
        "class_embed_type": class_embed_type,
        "projection_class_embeddings_input_dim": projection_class_embeddings_input_dim,
        "out_channels": original_unet_config["out_channels"],
        "up_block_types": tuple(up_block_types),
        "upcast_attention": False,  # TODO: guessing
        "cross_attention_norm": "group_norm",
        "mid_block_type": "UNetMidBlock2DSimpleCrossAttn",
        "act_fn": "gelu",
    }

    if original_unet_config["use_scale_shift_norm"]:
        config["resnet_time_scale_shift"] = "scale_shift"

    return config


def superres_convert_ldm_unet_checkpoint(unet_state_dict, config, path=None, extract_ema=False):
    """
    Takes a state dict and a config, and returns a converted checkpoint.
    """
    new_checkpoint = {}

    new_checkpoint["time_embedding.linear_1.weight"] = unet_state_dict["time_embed.0.weight"]
    new_checkpoint["time_embedding.linear_1.bias"] = unet_state_dict["time_embed.0.bias"]
    new_checkpoint["time_embedding.linear_2.weight"] = unet_state_dict["time_embed.2.weight"]
    new_checkpoint["time_embedding.linear_2.bias"] = unet_state_dict["time_embed.2.bias"]

    if config["class_embed_type"] is None:
        # No parameters to port
        ...
    elif config["class_embed_type"] == "timestep" or config["class_embed_type"] == "projection":
        new_checkpoint["class_embedding.linear_1.weight"] = unet_state_dict["aug_proj.0.weight"]
        new_checkpoint["class_embedding.linear_1.bias"] = unet_state_dict["aug_proj.0.bias"]
        new_checkpoint["class_embedding.linear_2.weight"] = unet_state_dict["aug_proj.2.weight"]
        new_checkpoint["class_embedding.linear_2.bias"] = unet_state_dict["aug_proj.2.bias"]
    else:
        raise NotImplementedError(f"Not implemented `class_embed_type`: {config['class_embed_type']}")

    if "encoder_proj.weight" in unet_state_dict:
        new_checkpoint["encoder_hid_proj.weight"] = unet_state_dict["encoder_proj.weight"]
        new_checkpoint["encoder_hid_proj.bias"] = unet_state_dict["encoder_proj.bias"]

    if "encoder_pooling.0.weight" in unet_state_dict:
        mapping = {
            "encoder_pooling.0": "add_embedding.norm1",
            "encoder_pooling.1": "add_embedding.pool",
            "encoder_pooling.2": "add_embedding.proj",
            "encoder_pooling.3": "add_embedding.norm2",
        }
        for key in unet_state_dict.keys():
            if key.startswith("encoder_pooling"):
                prefix = key[: len("encoder_pooling.0")]
                new_key = key.replace(prefix, mapping[prefix])
                new_checkpoint[new_key] = unet_state_dict[key]

    new_checkpoint["conv_in.weight"] = unet_state_dict["input_blocks.0.0.weight"]
    new_checkpoint["conv_in.bias"] = unet_state_dict["input_blocks.0.0.bias"]

    new_checkpoint["conv_norm_out.weight"] = unet_state_dict["out.0.weight"]
    new_checkpoint["conv_norm_out.bias"] = unet_state_dict["out.0.bias"]
    new_checkpoint["conv_out.weight"] = unet_state_dict["out.2.weight"]
    new_checkpoint["conv_out.bias"] = unet_state_dict["out.2.bias"]

    # Retrieves the keys for the input blocks only
    num_input_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "input_blocks" in layer})
    input_blocks = {
        layer_id: [key for key in unet_state_dict if f"input_blocks.{layer_id}." in key]
        for layer_id in range(num_input_blocks)
    }

    # Retrieves the keys for the middle blocks only
    num_middle_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "middle_block" in layer})
    middle_blocks = {
        layer_id: [key for key in unet_state_dict if f"middle_block.{layer_id}" in key]
        for layer_id in range(num_middle_blocks)
    }

    # Retrieves the keys for the output blocks only
    num_output_blocks = len({".".join(layer.split(".")[:2]) for layer in unet_state_dict if "output_blocks" in layer})
    output_blocks = {
        layer_id: [key for key in unet_state_dict if f"output_blocks.{layer_id}." in key]
        for layer_id in range(num_output_blocks)
    }
    if not isinstance(config["layers_per_block"], int):
        layers_per_block_list = [e + 1 for e in config["layers_per_block"]]
        layers_per_block_cumsum = list(np.cumsum(layers_per_block_list))
        downsampler_ids = layers_per_block_cumsum
    else:
        # TODO need better check than i in [4, 8, 12, 16]
        downsampler_ids = [4, 8, 12, 16]

    for i in range(1, num_input_blocks):
        if isinstance(config["layers_per_block"], int):
            layers_per_block = config["layers_per_block"]
            block_id = (i - 1) // (layers_per_block + 1)
            layer_in_block_id = (i - 1) % (layers_per_block + 1)
        else:
            block_id = next(k for k, n in enumerate(layers_per_block_cumsum) if (i - 1) < n)
            passed_blocks = layers_per_block_cumsum[block_id - 1] if block_id > 0 else 0
            layer_in_block_id = (i - 1) - passed_blocks

        resnets = [
            key for key in input_blocks[i] if f"input_blocks.{i}.0" in key and f"input_blocks.{i}.0.op" not in key
        ]
        attentions = [key for key in input_blocks[i] if f"input_blocks.{i}.1" in key]

        if f"input_blocks.{i}.0.op.weight" in unet_state_dict:
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.weight"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.weight"
            )
            new_checkpoint[f"down_blocks.{block_id}.downsamplers.0.conv.bias"] = unet_state_dict.pop(
                f"input_blocks.{i}.0.op.bias"
            )

        paths = renew_resnet_paths(resnets)

        block_type = config["down_block_types"][block_id]
        if (
            block_type == "ResnetDownsampleBlock2D" or block_type == "SimpleCrossAttnDownBlock2D"
        ) and i in downsampler_ids:
            meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.downsamplers.0"}
        else:
            meta_path = {"old": f"input_blocks.{i}.0", "new": f"down_blocks.{block_id}.resnets.{layer_in_block_id}"}

        assign_to_checkpoint(
            paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
        )

        if len(attentions):
            old_path = f"input_blocks.{i}.1"
            new_path = f"down_blocks.{block_id}.attentions.{layer_in_block_id}"

            assign_attention_to_checkpoint(
                new_checkpoint=new_checkpoint,
                unet_state_dict=unet_state_dict,
                old_path=old_path,
                new_path=new_path,
                config=config,
            )

            paths = renew_attention_paths(attentions)
            meta_path = {"old": old_path, "new": new_path}
            assign_to_checkpoint(
                paths,
                new_checkpoint,
                unet_state_dict,
                additional_replacements=[meta_path],
                config=config,
            )

    resnet_0 = middle_blocks[0]
    attentions = middle_blocks[1]
    resnet_1 = middle_blocks[2]

    resnet_0_paths = renew_resnet_paths(resnet_0)
    assign_to_checkpoint(resnet_0_paths, new_checkpoint, unet_state_dict, config=config)

    resnet_1_paths = renew_resnet_paths(resnet_1)
    assign_to_checkpoint(resnet_1_paths, new_checkpoint, unet_state_dict, config=config)

    old_path = "middle_block.1"
    new_path = "mid_block.attentions.0"

    assign_attention_to_checkpoint(
        new_checkpoint=new_checkpoint,
        unet_state_dict=unet_state_dict,
        old_path=old_path,
        new_path=new_path,
        config=config,
    )

    attentions_paths = renew_attention_paths(attentions)
    meta_path = {"old": "middle_block.1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(
        attentions_paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
    )
    if not isinstance(config["layers_per_block"], int):
        layers_per_block_list = list(reversed([e + 1 for e in config["layers_per_block"]]))
        layers_per_block_cumsum = list(np.cumsum(layers_per_block_list))

    for i in range(num_output_blocks):
        if isinstance(config["layers_per_block"], int):
            layers_per_block = config["layers_per_block"]
            block_id = i // (layers_per_block + 1)
            layer_in_block_id = i % (layers_per_block + 1)
        else:
            block_id = next(k for k, n in enumerate(layers_per_block_cumsum) if i < n)
            passed_blocks = layers_per_block_cumsum[block_id - 1] if block_id > 0 else 0
            layer_in_block_id = i - passed_blocks

        output_block_layers = [shave_segments(name, 2) for name in output_blocks[i]]
        output_block_list = {}

        for layer in output_block_layers:
            layer_id, layer_name = layer.split(".")[0], shave_segments(layer, 1)
            if layer_id in output_block_list:
                output_block_list[layer_id].append(layer_name)
            else:
                output_block_list[layer_id] = [layer_name]

        # len(output_block_list) == 1 -> resnet
        # len(output_block_list) == 2 -> resnet, attention or resnet, upscale resnet
        # len(output_block_list) == 3 -> resnet, attention, upscale resnet

        if len(output_block_list) > 1:
            resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.0" in key]

            has_attention = True
            if len(output_block_list) == 2 and any("in_layers" in k for k in output_block_list["1"]):
                has_attention = False

            maybe_attentions = [key for key in output_blocks[i] if f"output_blocks.{i}.1" in key]

            paths = renew_resnet_paths(resnets)

            meta_path = {"old": f"output_blocks.{i}.0", "new": f"up_blocks.{block_id}.resnets.{layer_in_block_id}"}

            assign_to_checkpoint(
                paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
            )

            output_block_list = {k: sorted(v) for k, v in output_block_list.items()}
            if ["conv.bias", "conv.weight"] in output_block_list.values():
                index = list(output_block_list.values()).index(["conv.bias", "conv.weight"])
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.weight"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.weight"
                ]
                new_checkpoint[f"up_blocks.{block_id}.upsamplers.0.conv.bias"] = unet_state_dict[
                    f"output_blocks.{i}.{index}.conv.bias"
                ]

                # this layer was no attention
                has_attention = False
                maybe_attentions = []

            if has_attention:
                old_path = f"output_blocks.{i}.1"
                new_path = f"up_blocks.{block_id}.attentions.{layer_in_block_id}"

                assign_attention_to_checkpoint(
                    new_checkpoint=new_checkpoint,
                    unet_state_dict=unet_state_dict,
                    old_path=old_path,
                    new_path=new_path,
                    config=config,
                )

                paths = renew_attention_paths(maybe_attentions)
                meta_path = {
                    "old": old_path,
                    "new": new_path,
                }
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )

            if len(output_block_list) == 3 or (not has_attention and len(maybe_attentions) > 0):
                layer_id = len(output_block_list) - 1
                resnets = [key for key in output_blocks[i] if f"output_blocks.{i}.{layer_id}" in key]
                paths = renew_resnet_paths(resnets)
                meta_path = {"old": f"output_blocks.{i}.{layer_id}", "new": f"up_blocks.{block_id}.upsamplers.0"}
                assign_to_checkpoint(
                    paths, new_checkpoint, unet_state_dict, additional_replacements=[meta_path], config=config
                )
        else:
            resnet_0_paths = renew_resnet_paths(output_block_layers, n_shave_prefix_segments=1)
            for path in resnet_0_paths:
                old_path = ".".join(["output_blocks", str(i), path["old"]])
                new_path = ".".join(["up_blocks", str(block_id), "resnets", str(layer_in_block_id), path["new"]])

                new_checkpoint[new_path] = unet_state_dict[old_path]

    return new_checkpoint


def verify_param_count(orig_path, unet_diffusers_config):
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


def superres_check_against_original(dump_path, unet_checkpoint_path):
    model_path = dump_path
    model = UNet2DConditionModel.from_pretrained(model_path)
    model.to("cuda")
    orig_path = unet_checkpoint_path

    if "-II-" in orig_path:
        from deepfloyd_if.modules import IFStageII

        if_II_model = IFStageII(device="cuda", dir_or_name=orig_path, model_kwargs={"precision": "fp32"}).model
    elif "-III-" in orig_path:
        from deepfloyd_if.modules import IFStageIII

        if_II_model = IFStageIII(device="cuda", dir_or_name=orig_path, model_kwargs={"precision": "fp32"}).model

    batch_size = 1
    channels = model.config.in_channels // 2
    height = model.config.sample_size
    width = model.config.sample_size
    height = 1024
    width = 1024

    torch.manual_seed(0)

    latents = torch.randn((batch_size, channels, height, width), device=model.device)
    image_small = torch.randn((batch_size, channels, height // 4, width // 4), device=model.device)

    interpolate_antialias = {}
    if "antialias" in inspect.signature(F.interpolate).parameters:
        interpolate_antialias["antialias"] = True
        image_upscaled = F.interpolate(
            image_small, size=[height, width], mode="bicubic", align_corners=False, **interpolate_antialias
        )

    latent_model_input = torch.cat([latents, image_upscaled], dim=1).to(model.dtype)
    t = torch.tensor([5], device=model.device).to(model.dtype)

    seq_len = 64
    encoder_hidden_states = torch.randn((batch_size, seq_len, model.config.encoder_hid_dim), device=model.device).to(
        model.dtype
    )

    fake_class_labels = torch.tensor([t], device=model.device).to(model.dtype)

    with torch.no_grad():
        out = if_II_model(latent_model_input, t, aug_steps=fake_class_labels, text_emb=encoder_hidden_states)

    if_II_model.to("cpu")
    del if_II_model
    import gc

    torch.cuda.empty_cache()
    gc.collect()
    print(50 * "=")

    with torch.no_grad():
        noise_pred = model(
            sample=latent_model_input,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=fake_class_labels,
            timestep=t,
        ).sample

    print("Out shape", noise_pred.shape)
    print("Diff", (out - noise_pred).abs().sum())


if __name__ == "__main__":
    main(parse_args())
