import argparse
import os
import shutil
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
import torch
from onnx import shape_inference
from packaging import version
from polygraphy.backend.onnx.loader import fold_constants
from torch.onnx import export

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetImg2ImgPipeline,
)
from diffusers.models.attention_processor import AttnProcessor
from diffusers.pipelines.controlnet.pipeline_controlnet_sd_xl import StableDiffusionXLControlNetPipeline


is_torch_less_than_1_11 = version.parse(version.parse(torch.__version__).base_version) < version.parse("1.11")
is_torch_2_0_1 = version.parse(version.parse(torch.__version__).base_version) == version.parse("2.0.1")


class Optimizer:
    def __init__(self, onnx_graph, verbose=False):
        self.graph = gs.import_onnx(onnx_graph)
        self.verbose = verbose

    def info(self, prefix):
        if self.verbose:
            print(
                f"{prefix} .. {len(self.graph.nodes)} nodes, {len(self.graph.tensors().keys())} tensors, {len(self.graph.inputs)} inputs, {len(self.graph.outputs)} outputs"
            )

    def cleanup(self, return_onnx=False):
        self.graph.cleanup().toposort()
        if return_onnx:
            return gs.export_onnx(self.graph)

    def select_outputs(self, keep, names=None):
        self.graph.outputs = [self.graph.outputs[o] for o in keep]
        if names:
            for i, name in enumerate(names):
                self.graph.outputs[i].name = name

    def fold_constants(self, return_onnx=False):
        onnx_graph = fold_constants(gs.export_onnx(self.graph), allow_onnxruntime_shape_inference=True)
        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph

    def infer_shapes(self, return_onnx=False):
        onnx_graph = gs.export_onnx(self.graph)
        if onnx_graph.ByteSize() > 2147483648:
            raise TypeError("ERROR: model size exceeds supported 2GB limit")
        else:
            onnx_graph = shape_inference.infer_shapes(onnx_graph)

        self.graph = gs.import_onnx(onnx_graph)
        if return_onnx:
            return onnx_graph


def optimize(onnx_graph, name, verbose):
    opt = Optimizer(onnx_graph, verbose=verbose)
    opt.info(name + ": original")
    opt.cleanup()
    opt.info(name + ": cleanup")
    opt.fold_constants()
    opt.info(name + ": fold constants")
    # opt.infer_shapes()
    # opt.info(name + ': shape inference')
    onnx_opt_graph = opt.cleanup(return_onnx=True)
    opt.info(name + ": finished")
    return onnx_opt_graph


class UNet2DConditionControlNetModel(torch.nn.Module):
    def __init__(
        self,
        unet,
        controlnets: ControlNetModel,
    ):
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_conds,
        controlnet_scales,
    ):
        for i, (controlnet_cond, conditioning_scale, controlnet) in enumerate(
            zip(controlnet_conds, controlnet_scales, self.controlnets)
        ):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
                return_dict=False,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            return_dict=False,
        )[0]
        return noise_pred


class UNet2DConditionXLControlNetModel(torch.nn.Module):
    def __init__(
        self,
        unet,
        controlnets: ControlNetModel,
    ):
        super().__init__()
        self.unet = unet
        self.controlnets = controlnets

    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_conds,
        controlnet_scales,
        text_embeds,
        time_ids,
    ):
        added_cond_kwargs = {"text_embeds": text_embeds, "time_ids": time_ids}
        for i, (controlnet_cond, conditioning_scale, controlnet) in enumerate(
            zip(controlnet_conds, controlnet_scales, self.controlnets)
        ):
            down_samples, mid_sample = controlnet(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_cond,
                conditioning_scale=conditioning_scale,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        noise_pred = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]
        return noise_pred


def onnx_export(
    model,
    model_args: tuple,
    output_path: Path,
    ordered_input_names,
    output_names,
    dynamic_axes,
    opset,
    use_external_data_format=False,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # PyTorch deprecated the `enable_onnx_checker` and `use_external_data_format` arguments in v1.11,
    # so we check the torch version for backwards compatibility
    with torch.inference_mode(), torch.autocast("cuda"):
        if is_torch_less_than_1_11:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                enable_onnx_checker=True,
                opset_version=opset,
            )
        else:
            export(
                model,
                model_args,
                f=output_path.as_posix(),
                input_names=ordered_input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                do_constant_folding=True,
                opset_version=opset,
            )


@torch.no_grad()
def convert_models(
    model_path: str, controlnet_path: list, output_path: str, opset: int, fp16: bool = False, sd_xl: bool = False
):
    """
    Function to convert models in stable diffusion controlnet pipeline into ONNX format

    Example:
    python convert_stable_diffusion_controlnet_to_onnx.py
    --model_path danbrown/RevAnimated-v1-2-2
    --controlnet_path lllyasviel/control_v11f1e_sd15_tile ioclab/brightness-controlnet
    --output_path path-to-models-stable_diffusion/RevAnimated-v1-2-2
    --fp16

    Example for SD XL:
    python convert_stable_diffusion_controlnet_to_onnx.py
    --model_path stabilityai/stable-diffusion-xl-base-1.0
    --controlnet_path SargeZT/sdxl-controlnet-seg
    --output_path path-to-models-stable_diffusion/stable-diffusion-xl-base-1.0
    --fp16
    --sd_xl

    Returns:
        create 4 onnx models in output path
        text_encoder/model.onnx
        unet/model.onnx + unet/weights.pb
        vae_encoder/model.onnx
        vae_decoder/model.onnx

        run test script in diffusers/examples/community
        python test_onnx_controlnet.py
        --sd_model danbrown/RevAnimated-v1-2-2
        --onnx_model_dir path-to-models-stable_diffusion/RevAnimated-v1-2-2
        --qr_img_path path-to-qr-code-image
    """
    dtype = torch.float16 if fp16 else torch.float32
    if fp16 and torch.cuda.is_available():
        device = "cuda"
    elif fp16 and not torch.cuda.is_available():
        raise ValueError("`float16` model export is only supported on GPUs with CUDA")
    else:
        device = "cpu"

    # init controlnet
    controlnets = []
    for path in controlnet_path:
        controlnet = ControlNetModel.from_pretrained(path, torch_dtype=dtype).to(device)
        if is_torch_2_0_1:
            controlnet.set_attn_processor(AttnProcessor())
        controlnets.append(controlnet)

    if sd_xl:
        if len(controlnets) == 1:
            controlnet = controlnets[0]
        else:
            raise ValueError("MultiControlNet is not yet supported.")
        pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            model_path, controlnet=controlnet, torch_dtype=dtype, variant="fp16", use_safetensors=True
        ).to(device)
    else:
        pipeline = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            model_path, controlnet=controlnets, torch_dtype=dtype
        ).to(device)

    output_path = Path(output_path)
    if is_torch_2_0_1:
        pipeline.unet.set_attn_processor(AttnProcessor())
        pipeline.vae.set_attn_processor(AttnProcessor())

    # # TEXT ENCODER
    num_tokens = pipeline.text_encoder.config.max_position_embeddings
    text_hidden_size = pipeline.text_encoder.config.hidden_size
    text_input = pipeline.tokenizer(
        "A sample prompt",
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    onnx_export(
        pipeline.text_encoder,
        # casting to torch.int32 until the CLIP fix is released: https://github.com/huggingface/transformers/pull/18515/files
        model_args=(text_input.input_ids.to(device=device, dtype=torch.int32)),
        output_path=output_path / "text_encoder" / "model.onnx",
        ordered_input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
        },
        opset=opset,
    )
    del pipeline.text_encoder

    # # UNET
    if sd_xl:
        controlnets = torch.nn.ModuleList(controlnets)
        unet_controlnet = UNet2DConditionXLControlNetModel(pipeline.unet, controlnets)
        unet_in_channels = pipeline.unet.config.in_channels
        unet_sample_size = pipeline.unet.config.sample_size
        text_hidden_size = 2048
        img_size = 8 * unet_sample_size
        unet_path = output_path / "unet" / "model.onnx"

        onnx_export(
            unet_controlnet,
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.tensor([1.0]).to(device=device, dtype=dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
                torch.randn(len(controlnets), 2, 3, img_size, img_size).to(device=device, dtype=dtype),
                torch.randn(len(controlnets), 1).to(device=device, dtype=dtype),
                torch.randn(2, 1280).to(device=device, dtype=dtype),
                torch.rand(2, 6).to(device=device, dtype=dtype),
            ),
            output_path=unet_path,
            ordered_input_names=[
                "sample",
                "timestep",
                "encoder_hidden_states",
                "controlnet_conds",
                "conditioning_scales",
                "text_embeds",
                "time_ids",
            ],
            output_names=["noise_pred"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "2B"},
                "controlnet_conds": {1: "2B", 3: "8H", 4: "8W"},
                "text_embeds": {0: "2B"},
                "time_ids": {0: "2B"},
            },
            opset=opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        # optimize onnx
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = optimize(onnx.load(unet_model_path), name="Unet", verbose=True)
        # clean up existing tensor files
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        # collate external tensor files into one
        onnx.save_model(
            unet_opt_graph,
            unet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        del pipeline.unet
    else:
        controlnets = torch.nn.ModuleList(controlnets)
        unet_controlnet = UNet2DConditionControlNetModel(pipeline.unet, controlnets)
        unet_in_channels = pipeline.unet.config.in_channels
        unet_sample_size = pipeline.unet.config.sample_size
        img_size = 8 * unet_sample_size
        unet_path = output_path / "unet" / "model.onnx"

        onnx_export(
            unet_controlnet,
            model_args=(
                torch.randn(2, unet_in_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
                torch.tensor([1.0]).to(device=device, dtype=dtype),
                torch.randn(2, num_tokens, text_hidden_size).to(device=device, dtype=dtype),
                torch.randn(len(controlnets), 2, 3, img_size, img_size).to(device=device, dtype=dtype),
                torch.randn(len(controlnets), 1).to(device=device, dtype=dtype),
            ),
            output_path=unet_path,
            ordered_input_names=[
                "sample",
                "timestep",
                "encoder_hidden_states",
                "controlnet_conds",
                "conditioning_scales",
            ],
            output_names=["noise_pred"],  # has to be different from "sample" for correct tracing
            dynamic_axes={
                "sample": {0: "2B", 2: "H", 3: "W"},
                "encoder_hidden_states": {0: "2B"},
                "controlnet_conds": {1: "2B", 3: "8H", 4: "8W"},
            },
            opset=opset,
            use_external_data_format=True,  # UNet is > 2GB, so the weights need to be split
        )
        unet_model_path = str(unet_path.absolute().as_posix())
        unet_dir = os.path.dirname(unet_model_path)
        # optimize onnx
        shape_inference.infer_shapes_path(unet_model_path, unet_model_path)
        unet_opt_graph = optimize(onnx.load(unet_model_path), name="Unet", verbose=True)
        # clean up existing tensor files
        shutil.rmtree(unet_dir)
        os.mkdir(unet_dir)
        # collate external tensor files into one
        onnx.save_model(
            unet_opt_graph,
            unet_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
            convert_attribute=False,
        )
        del pipeline.unet

    # VAE ENCODER
    vae_encoder = pipeline.vae
    vae_in_channels = vae_encoder.config.in_channels
    vae_sample_size = vae_encoder.config.sample_size
    # need to get the raw tensor output (sample) from the encoder
    vae_encoder.forward = lambda sample: vae_encoder.encode(sample).latent_dist.sample()
    onnx_export(
        vae_encoder,
        model_args=(torch.randn(1, vae_in_channels, vae_sample_size, vae_sample_size).to(device=device, dtype=dtype),),
        output_path=output_path / "vae_encoder" / "model.onnx",
        ordered_input_names=["sample"],
        output_names=["latent_sample"],
        dynamic_axes={
            "sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )

    # VAE DECODER
    vae_decoder = pipeline.vae
    vae_latent_channels = vae_decoder.config.latent_channels
    # forward only through the decoder part
    vae_decoder.forward = vae_encoder.decode
    onnx_export(
        vae_decoder,
        model_args=(
            torch.randn(1, vae_latent_channels, unet_sample_size, unet_sample_size).to(device=device, dtype=dtype),
        ),
        output_path=output_path / "vae_decoder" / "model.onnx",
        ordered_input_names=["latent_sample"],
        output_names=["sample"],
        dynamic_axes={
            "latent_sample": {0: "batch", 1: "channels", 2: "height", 3: "width"},
        },
        opset=opset,
    )
    del pipeline.vae

    del pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sd_xl", action="store_true", default=False, help="SD XL pipeline")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the `diffusers` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument(
        "--controlnet_path",
        nargs="+",
        required=True,
        help="Path to the `controlnet` checkpoint to convert (either a local directory or on the Hub).",
    )

    parser.add_argument("--output_path", type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--opset",
        default=14,
        type=int,
        help="The version of the ONNX operator set to use.",
    )
    parser.add_argument("--fp16", action="store_true", default=False, help="Export the models in `float16` mode")

    args = parser.parse_args()

    convert_models(args.model_path, args.controlnet_path, args.output_path, args.opset, args.fp16, args.sd_xl)
