import argparse

import torch
from torch import nn

from diffusers import DiffusionPipeline
from diffusers.models.attention import AttentionBlock, assert_no_deprecated_attention_blocks
from diffusers.models.autoencoder_kl import AutoencoderKL
from diffusers.models.unet_2d import UNet2DModel
from diffusers.models.unet_2d_blocks import (
    AttnDownBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    AttnSkipUpBlock2D,
    AttnUpBlock2D,
    AttnUpDecoderBlock2D,
    UNetMidBlock2D,
)
from diffusers.models.vq_model import VQModel
from diffusers.pipelines.pipeline_utils import PipelineType
from diffusers.utils.torch_utils import randn_tensor


MODULES = [AutoencoderKL, VQModel, UNet2DModel]

UNET_BLOCKS = [
    UNetMidBlock2D,
    AttnDownBlock2D,
    AttnDownEncoderBlock2D,
    AttnSkipDownBlock2D,
    AttnUpBlock2D,
    AttnUpDecoderBlock2D,
    AttnSkipUpBlock2D,
]


unet_blocks_to_convert = []


def patch_unet_block(unet_block_class):
    orig_constructor = unet_block_class.__init__

    def new_constructor(self, *args, **kwargs):
        orig_constructor(self, *args, **kwargs)
        unet_blocks_to_convert.append(self)

    def convert_attention_blocks(self):
        new_attentions = []

        for attention_block in self.attentions:
            if isinstance(attention_block, AttentionBlock):
                new_attention_block = attention_block.as_cross_attention()
            else:
                new_attention_block = attention_block

            new_attentions.append(new_attention_block)

        self.attentions = nn.ModuleList(new_attentions)

    unet_block_class.__init__ = new_constructor
    unet_block_class.convert_attention_blocks = convert_attention_blocks


for unet_block_class in UNET_BLOCKS:
    patch_unet_block(unet_block_class)


def default_output(pipe: DiffusionPipeline):
    generator = torch.Generator("cpu").manual_seed(0)

    # TODO
    # height = 224
    # width = 224
    height = 512
    width = 512

    prompt = "a horse"
    class_labels = [0]
    image = randn_tensor((1, 3, height, width), generator=generator).clamp(-1, 1)
    mask_image = torch.bernoulli(torch.full((1, 1, height, width), 0.5), generator=generator)

    args = {"generator": generator, "return_dict": False, "output_type": "np"}

    if pipe.pipeline_type == PipelineType.UNCONDITIONAL_IMAGE_GENERATION:
        # No additional arguments
        ...
    elif pipe.pipeline_type == PipelineType.CLASS_CONDITIONAL_IMAGE_GENERATION:
        args["class_labels"] = class_labels
    elif pipe.pipeline_type == PipelineType.TEXT_TO_IMAGE:
        args["prompt"] = prompt
    elif pipe.pipeline_type == PipelineType.IMAGE_VARIATION:
        args["image"] = image
    elif pipe.pipeline_type == PipelineType.TEXT_GUIDED_IMAGE_VARIATION:
        args["prompt"] = prompt
        args["image"] = image
    elif pipe.pipeline_type == PipelineType.INPAINTING:
        args["image"] = image
        args["mask_image"] = mask_image
    elif pipe.pipeline_type == PipelineType.TEXT_GUIDED_INPAINTING:
        args["prompt"] = prompt
        args["image"] = image
        args["mask_image"] = mask_image
    elif pipe.pipeline_type == PipelineType.UNCONDITIONAL_AUDIO_GENERATION:
        # No additional arguments
        ...
    elif pipe.pipeline_type == PipelineType.AUDIO_VARIATION:
        # TODO not getting results equal

        # No additional arguments
        ...

        # HACK
        args.pop("output_type")
    else:
        assert False

    # We want to test that the outputs are the same regardless of if or not the safety checker
    # has a hit on the result. However, if a safety checker exists in the pipeline stored on the hub,
    # we still want it to be a property on the loaded pipeline for when we serialize the one with
    # converted weights.
    #
    # So, we temporarily remove the safety checker from the pipeline during inference and re-add it afterwards.

    if hasattr(pipe, "safety_checker"):
        safety_checker = pipe.safety_checker
        pipe.safety_checker = None
    else:
        safety_checker = None

    output = pipe(**args)

    if pipe.pipeline_type == PipelineType.AUDIO_VARIATION:
        # HACK
        output = output[1][1][0]
    else:
        output = output[0]

    if safety_checker is not None:
        pipe.safety_checker = safety_checker

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        default=None,
        type=str,
        required=True,
        help="Pipeline to convert the deprecated `AttentionBlock` to `CrossAttention`",
    )

    parser.add_argument(
        "--dump_path", default=None, type=str, required=True, help="Path to the save the converted pipeline."
    )

    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        required=False,
        help="Device to run the pipeline on for checking output is consistent.",
    )

    args = parser.parse_args()

    print(f"loading original pipeline {args.pipeline}")

    pipe = DiffusionPipeline.from_pretrained(args.pipeline)
    pipe.to(args.device)

    if pipe.pipeline_type is None:
        raise ValueError(
            f"{pipe.__class__.__name__} must have `pipeline_type` set. We need the pipeline type to know what"
            " parameters to pass so that we can check the conversion does not change the outputs."
        )

    print("Running original pipeline to check output against converted pipeline")
    orig_output = default_output(pipe)

    any_converted = False

    for attr_name in dir(pipe):
        attr = getattr(pipe, attr_name)

        for module in MODULES:
            if isinstance(attr, module):
                print(
                    f"converting `DiffusionPipeline.from_pretrained({args.pipeline}).{attr_name}.attention_block_type`"
                )
                attr.register_to_config(attention_block_type="CrossAttention")
                any_converted = True

    for unet_block in unet_blocks_to_convert:
        print(f"converting {unet_block.__class__}.attentions")
        unet_block.convert_attention_blocks()
        any_converted = True

    if not any_converted:
        print(f"`DiffusionPipeline.from_pretrained({args.pipeline})` did not have any deprecated attention blocks")
    else:
        print(f"Saving converted pipeline to {args.dump_path}")

        pipe.save_pretrained(args.dump_path)

        print(f"Converted pipeline saved to {args.dump_path}")

        print("Checking converted pipeline has no deprecated attention blocks")

        with assert_no_deprecated_attention_blocks():
            pipe = DiffusionPipeline.from_pretrained(args.dump_path)

        pipe.to(args.device)

        print("Running converted pipeline to check against original pipeline")
        converted_output = default_output(pipe)

        if (orig_output == converted_output).all():
            print("output of converted pipeline matches original pipeline")
        else:
            raise AssertionError("converted pipeline output differs from original pipeline")
