import argparse
import os
from typing import Any

import safetensors.torch
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2LatentUpsamplePipeline,
    LTX2Pipeline,
    LTX2VideoDiffusionDecoderModel,
    LTX2VideoTransformer3DModel,
)
from diffusers.loaders.conversion.configs.ltx2 import (
    get_ltx2_audio_vae_config,
    get_ltx2_connectors_config,
    get_ltx2_diffusion_video_vae_config,
    get_ltx2_spatial_latent_upsampler_config,
    get_ltx2_temporal_latent_upsampler_config,
    get_ltx2_transformer_config,
    get_ltx2_video_vae_config,
    get_ltx2_vocoder_config,
)
from diffusers.pipelines.ltx2 import (
    LTX2DurationHead,
    LTX2LatentUpsamplerModel,
    LTX2TextConnectors,
    LTX2Vocoder,
    LTX2VocoderWithBWE,
)


# LTX-2.5's diffusion decoder replaces the conv decoder while keeping the same encoder, so only the
# `decoder.*` half of the VAE checkpoint is renamed with these rules.

# Where a checkpoint carries static AdaLN gates, each is folded into the Linear it gates (W <- g * W) and
# dropped, because the decoder's residuals are ungated. Maps a renamed parameter to its gate's suffix.


def split_transformer_and_connector_state_dict(state_dict: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    connector_prefixes = (
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "transformer_1d_blocks",
        "text_embedding_projection",
        "connectors.",
        "video_connector",
        "audio_connector",
        "text_proj_in",
    )

    transformer_state_dict, connector_state_dict = {}, {}
    for key, value in state_dict.items():
        if key.startswith(connector_prefixes):
            connector_state_dict[key] = value
        else:
            transformer_state_dict[key] = value

    return transformer_state_dict, connector_state_dict


def convert_ltx2_transformer(original_state_dict, version):
    config = get_ltx2_transformer_config(version)["diffusers_config"]
    state, _ = split_transformer_and_connector_state_dict(original_state_dict)
    return LTX2VideoTransformer3DModel.from_single_file(state, config=config)


def convert_ltx2_connectors(original_state_dict, version, gemma_text_config=None):
    config = get_ltx2_connectors_config(version, gemma_text_config=gemma_text_config)["diffusers_config"]
    _, state = split_transformer_and_connector_state_dict(original_state_dict)
    return LTX2TextConnectors.from_single_file(state, config=config)


def convert_ltx2_duration_head(original_state_dict):
    if not original_state_dict:
        return None
    state = original_state_dict
    config = {
        "video_cross_attention_dim": state["video_input_proj.weight"].shape[1],
        "audio_cross_attention_dim": state["audio_input_proj.weight"].shape[1],
        "pooler_hidden_dim": state["attention_pooler.cross_attn.in_proj_weight"].shape[1],
        "num_queries": state["attention_pooler.query_tokens"].shape[0],
        "mlp_hidden_dim": state["mlp_hidden.weight"].shape[0],
        "num_pooler_heads": 4,
    }
    return LTX2DurationHead.from_single_file(state, config=config)


def convert_ltx2_video_vae(original_state_dict, version, timestep_conditioning):
    config = get_ltx2_video_vae_config(version, timestep_conditioning)["diffusers_config"]

    return AutoencoderKLLTX2Video.from_single_file(original_state_dict, config=config)


def convert_ltx2_diffusion_video_vae(original_state_dict, version):
    config = get_ltx2_diffusion_video_vae_config(version)["diffusers_config"]

    return LTX2VideoDiffusionDecoderModel.from_single_file(original_state_dict, config=config)


def convert_ltx2_audio_vae(original_state_dict, version):
    config = get_ltx2_audio_vae_config(version)["diffusers_config"]

    return AutoencoderKLLTX2Audio.from_single_file(original_state_dict, config=config)


def convert_ltx2_vocoder(original_state_dict, version):
    config = get_ltx2_vocoder_config(version)["diffusers_config"]
    vocoder_cls = LTX2VocoderWithBWE if version in ("2.3", "2.5") else LTX2Vocoder
    return vocoder_cls.from_single_file(original_state_dict, config=config)


def convert_ltx2_latent_upsampler(original_state_dict, config, dtype):
    return LTX2LatentUpsamplerModel.from_single_file(original_state_dict, config=config, torch_dtype=dtype)


def load_original_checkpoint(args, filename: str | None) -> dict[str, Any]:
    if args.original_state_dict_repo_id is not None:
        ckpt_path = hf_hub_download(repo_id=args.original_state_dict_repo_id, filename=filename)
    elif args.checkpoint_path is not None:
        ckpt_path = args.checkpoint_path
    else:
        raise ValueError("Please provide either `original_state_dict_repo_id` or a local `checkpoint_path`")

    original_state_dict = safetensors.torch.load_file(ckpt_path)
    return original_state_dict


def load_hub_or_local_checkpoint(repo_id: str | None = None, filename: str | None = None) -> dict[str, Any]:
    if repo_id is None and filename is None:
        raise ValueError("Please supply at least one of `repo_id` or `filename`")

    if repo_id is not None:
        if filename is None:
            raise ValueError("If repo_id is specified, filename must also be specified.")
        ckpt_path = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        ckpt_path = filename

    _, ext = os.path.splitext(ckpt_path)
    if ext in [".safetensors", ".sft"]:
        state_dict = safetensors.torch.load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    return state_dict


def get_model_state_dict_from_combined_ckpt(combined_ckpt: dict[str, Any], prefix: str) -> dict[str, Any]:
    # Ensure that the key prefix ends with a dot (.)
    if not prefix.endswith("."):
        prefix = prefix + "."

    model_state_dict = {}
    for param_name, param in combined_ckpt.items():
        if param_name.startswith(prefix):
            model_state_dict[param_name.removeprefix(prefix)] = param

    if prefix == "model.diffusion_model.":
        # Some checkpoints store the text connector projection outside the diffusion model prefix.
        connector_prefixes = ["text_embedding_projection"]
        for param_name, param in combined_ckpt.items():
            for prefix in connector_prefixes:
                if param_name.startswith(prefix):
                    # Check to make sure we're not overwriting an existing key
                    if param_name not in model_state_dict:
                        model_state_dict[param_name] = combined_ckpt[param_name]

    return model_state_dict


def get_args():
    parser = argparse.ArgumentParser()

    def none_or_str(value: str):
        if isinstance(value, str) and value.lower() == "none":
            return None
        return value

    parser.add_argument(
        "--original_state_dict_repo_id",
        default="Lightricks/LTX-2",
        type=none_or_str,
        help="HF Hub repo id with LTX 2.0 checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        type=str,
        help="Local checkpoint path for LTX 2.0. Will be used if `original_state_dict_repo_id` is not specified.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="2.0",
        choices=["test", "2.0", "2.3", "2.5"],
        help="Version of the LTX 2.0 model",
    )

    parser.add_argument(
        "--combined_filename",
        default="ltx-2-19b-dev.safetensors",
        type=none_or_str,
        help="Filename for combined checkpoint with all LTX 2.0 models (VAE, DiT, etc.)",
    )
    parser.add_argument("--vae_prefix", default="vae.", type=str)
    parser.add_argument("--audio_vae_prefix", default="audio_vae.", type=str)
    parser.add_argument("--dit_prefix", default="model.diffusion_model.", type=str)
    parser.add_argument("--vocoder_prefix", default="vocoder.", type=str)
    parser.add_argument("--duration_head_prefix", default="duration_head.", type=str)

    parser.add_argument("--vae_filename", default=None, type=str, help="VAE filename; overrides combined ckpt if set")
    parser.add_argument(
        "--audio_vae_filename", default=None, type=str, help="Audio VAE filename; overrides combined ckpt if set"
    )
    parser.add_argument("--dit_filename", default=None, type=str, help="DiT filename; overrides combined ckpt if set")
    parser.add_argument(
        "--vocoder_filename", default=None, type=str, help="Vocoder filename; overrides combined ckpt if set"
    )
    parser.add_argument(
        "--text_encoder_model_id",
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        type=none_or_str,
        help=(
            "HF Hub id for the text encoder model. Default is Gemma 3, used by LTX 2.0/2.3. LTX-2.5 requires a "
            "Gemma 4 (`gemma4_unified`) checkpoint here instead -- passing the Gemma 3 default with `--version 2.5` "
            "raises an error."
        ),
    )
    parser.add_argument(
        "--tokenizer_id",
        default="google/gemma-3-12b-it-qat-q4_0-unquantized",
        type=none_or_str,
        help="HF Hub id for the text tokenizer. Should match --text_encoder_model_id's family (Gemma 3 vs Gemma 4).",
    )
    parser.add_argument(
        "--prompt_enhancer_model_id",
        default=None,
        type=none_or_str,
        help=(
            "HF Hub id for the prompt-enhancer model (used with --add_processor). For LTX-2.0/2.3, defaults to "
            "--text_encoder_model_id (the same Gemma 3 checkpoint serves both roles). LTX-2.5's fine-tuned text "
            "encoder is not trained for enhancement, so this must be set explicitly for --version 2.5 -- e.g. to "
            "google/gemma-4-E2B-it or google/gemma-4-E4B-it."
        ),
    )
    parser.add_argument(
        "--temporal_latent_upsampler_filename",
        default="ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        type=none_or_str,
        help="Temporal x2 latent upsampler filename (LTX-2.5, used by the DFR pipeline's temporal refine rounds)",
    )
    parser.add_argument(
        "--latent_upsampler_filename",
        default="ltx-2-spatial-upscaler-x2-1.0.safetensors",
        type=none_or_str,
        help="Latent upsampler filename",
    )

    parser.add_argument(
        "--timestep_conditioning", action="store_true", help="Whether to add timestep condition to the video VAE model"
    )
    parser.add_argument("--vae", action="store_true", help="Whether to convert the video VAE model")
    parser.add_argument(
        "--diffusion_vae",
        action="store_true",
        help=(
            "Whether to convert the LTX-2.5 diffusion decoder, saved to a `diffusion_decoder` subfolder â€” the "
            "component name `LTX2VideoDiffusionDecodePipeline` and the modular blocks resolve it by â€” so "
            "`from_pretrained` keeps returning the conv decoder in `vae` by default"
        ),
    )
    parser.add_argument("--audio_vae", action="store_true", help="Whether to convert the audio VAE model")
    parser.add_argument("--dit", action="store_true", help="Whether to convert the DiT model")
    parser.add_argument("--connectors", action="store_true", help="Whether to convert the connector model")
    parser.add_argument(
        "--duration_head",
        action="store_true",
        help="Whether to convert the duration head (present in LTX-2.5 and later checkpoints only)",
    )
    parser.add_argument("--vocoder", action="store_true", help="Whether to convert the vocoder model")
    parser.add_argument("--text_encoder", action="store_true", help="Whether to conver the text encoder")
    parser.add_argument("--latent_upsampler", action="store_true", help="Whether to convert the latent upsampler")
    parser.add_argument(
        "--temporal_latent_upsampler",
        action="store_true",
        help="Whether to convert the temporal x2 latent upsampler (LTX-2.5)",
    )
    parser.add_argument(
        "--full_pipeline",
        action="store_true",
        help="Whether to save the pipeline. This will attempt to convert all models (e.g. vae, dit, etc.)",
    )
    parser.add_argument(
        "--upsample_pipeline",
        action="store_true",
        help="Whether to save a latent upsampling pipeline",
    )
    parser.add_argument(
        "--add_processor",
        action="store_true",
        help="Whether to add a text-encoder processor to the pipeline for prompt enhancement.",
    )

    parser.add_argument("--vae_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--audio_vae_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--dit_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--vocoder_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--text_encoder_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    parser.add_argument("--output_path", type=str, required=True, help="Path where converted model should be saved")
    parser.add_argument(
        "--upsample_output_path",
        type=str,
        default=None,
        help="Path where converted upsampling pipeline should be saved",
    )

    return parser.parse_args()


DTYPE_MAPPING = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main(args):
    vae_dtype = DTYPE_MAPPING[args.vae_dtype]
    audio_vae_dtype = DTYPE_MAPPING[args.audio_vae_dtype]
    dit_dtype = DTYPE_MAPPING[args.dit_dtype]
    vocoder_dtype = DTYPE_MAPPING[args.vocoder_dtype]
    text_encoder_dtype = DTYPE_MAPPING[args.text_encoder_dtype]

    combined_ckpt = None
    load_combined_models = any(
        [
            args.vae,
            args.diffusion_vae,
            args.audio_vae,
            args.dit,
            args.vocoder,
            args.connectors,
            args.full_pipeline,
            args.upsample_pipeline,
        ]
    )
    if args.combined_filename is not None and load_combined_models:
        combined_ckpt = load_original_checkpoint(args, filename=args.combined_filename)

    # LTX-2.5 only works with a Gemma 4 (`gemma4_unified`) text encoder; --text_encoder_model_id defaults to
    # Gemma 3 (for 2.0/2.3), so silently proceeding would pair a 2.5 checkpoint with the wrong text encoder.
    gemma_text_config = None
    if args.version == "2.5" and (args.text_encoder or args.connectors or args.full_pipeline):
        gemma_config = AutoConfig.from_pretrained(args.text_encoder_model_id)
        if gemma_config.model_type != "gemma4_unified":
            raise ValueError(
                f"LTX-2.5 requires a Gemma 4 (`gemma4_unified`) text encoder, but --text_encoder_model_id="
                f"{args.text_encoder_model_id!r} has model_type={gemma_config.model_type!r}. Pass "
                "--text_encoder_model_id pointing at a Gemma 4 checkpoint (the default is Gemma 3, for 2.0/2.3)."
            )
        gemma_text_config = gemma_config.text_config

    # LTX-2.5's fine-tuned text encoder is never a valid prompt enhancer (unlike LTX-2.0/2.3, where the same Gemma 3
    # checkpoint serves both roles) -- require an explicit, separate --prompt_enhancer_model_id instead of silently
    # falling back to --text_encoder_model_id.
    if (
        args.version == "2.5"
        and args.add_processor
        and (args.text_encoder or args.full_pipeline)
        and args.prompt_enhancer_model_id is None
    ):
        raise ValueError(
            "LTX-2.5's text encoder is not trained for prompt enhancement, so --prompt_enhancer_model_id must be "
            "set explicitly when --add_processor is used with --version 2.5 -- e.g. to google/gemma-4-E2B-it or "
            "google/gemma-4-E4B-it."
        )

    if args.vae or args.full_pipeline or args.upsample_pipeline:
        if args.vae_filename is not None:
            original_vae_ckpt = load_hub_or_local_checkpoint(filename=args.vae_filename)
        elif combined_ckpt is not None:
            original_vae_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.vae_prefix)
        vae = convert_ltx2_video_vae(
            original_vae_ckpt, version=args.version, timestep_conditioning=args.timestep_conditioning
        )
        if not args.full_pipeline and not args.upsample_pipeline:
            vae.to(vae_dtype).save_pretrained(os.path.join(args.output_path, "vae"))

    if args.diffusion_vae:
        if args.vae_filename is not None:
            original_diffusion_vae_ckpt = load_hub_or_local_checkpoint(filename=args.vae_filename)
        elif combined_ckpt is not None:
            original_diffusion_vae_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.vae_prefix)
        diffusion_vae = convert_ltx2_diffusion_video_vae(original_diffusion_vae_ckpt, version=args.version)
        # "diffusion_decoder", not "vae_diffusion": pipeline-level `from_pretrained` resolves each component
        # from the subfolder named after it, so this folder name must match the `diffusion_decoder` component
        # of `LTX2VideoDiffusionDecodePipeline` (and the modular `ComponentSpec`) for those loads to work.
        diffusion_vae.to(vae_dtype).save_pretrained(os.path.join(args.output_path, "diffusion_decoder"))

    if args.audio_vae or args.full_pipeline:
        if args.audio_vae_filename is not None:
            original_audio_vae_ckpt = load_hub_or_local_checkpoint(filename=args.audio_vae_filename)
        elif combined_ckpt is not None:
            original_audio_vae_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.audio_vae_prefix)
        audio_vae = convert_ltx2_audio_vae(original_audio_vae_ckpt, version=args.version)
        if not args.full_pipeline:
            audio_vae.to(audio_vae_dtype).save_pretrained(os.path.join(args.output_path, "audio_vae"))

    if args.dit or args.full_pipeline:
        if args.dit_filename is not None:
            original_dit_ckpt = load_hub_or_local_checkpoint(filename=args.dit_filename)
        elif combined_ckpt is not None:
            original_dit_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.dit_prefix)
        transformer = convert_ltx2_transformer(original_dit_ckpt, version=args.version)
        if not args.full_pipeline:
            transformer.to(dit_dtype).save_pretrained(os.path.join(args.output_path, "transformer"))

    if args.connectors or args.full_pipeline:
        if args.dit_filename is not None:
            original_connectors_ckpt = load_hub_or_local_checkpoint(filename=args.dit_filename)
        elif combined_ckpt is not None:
            original_connectors_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.dit_prefix)
        connectors = convert_ltx2_connectors(
            original_connectors_ckpt, version=args.version, gemma_text_config=gemma_text_config
        )
        if not args.full_pipeline:
            connectors.to(dit_dtype).save_pretrained(os.path.join(args.output_path, "connectors"))

    duration_head = None
    if args.duration_head or args.full_pipeline:
        if combined_ckpt is not None:
            original_duration_head_ckpt = get_model_state_dict_from_combined_ckpt(
                combined_ckpt, args.duration_head_prefix
            )
            duration_head = convert_ltx2_duration_head(original_duration_head_ckpt)
        if duration_head is not None and not args.full_pipeline:
            duration_head.to(dit_dtype).save_pretrained(os.path.join(args.output_path, "duration_head"))

    if args.vocoder or args.full_pipeline:
        if args.vocoder_filename is not None:
            original_vocoder_ckpt = load_hub_or_local_checkpoint(filename=args.vocoder_filename)
        elif combined_ckpt is not None:
            original_vocoder_ckpt = get_model_state_dict_from_combined_ckpt(combined_ckpt, args.vocoder_prefix)
        vocoder = convert_ltx2_vocoder(original_vocoder_ckpt, version=args.version)
        if not args.full_pipeline:
            vocoder.to(vocoder_dtype).save_pretrained(os.path.join(args.output_path, "vocoder"))

    if args.text_encoder or args.full_pipeline:
        text_encoder = AutoModelForImageTextToText.from_pretrained(args.text_encoder_model_id)
        if not args.full_pipeline:
            text_encoder.to(text_encoder_dtype).save_pretrained(os.path.join(args.output_path, "text_encoder"))

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
        if not args.full_pipeline:
            tokenizer.save_pretrained(os.path.join(args.output_path, "tokenizer"))

        processor = None
        prompt_enhancer = None
        if args.add_processor:
            enhancer_model_id = args.prompt_enhancer_model_id or args.text_encoder_model_id
            processor = AutoProcessor.from_pretrained(enhancer_model_id)
            if not args.full_pipeline:
                processor.save_pretrained(os.path.join(args.output_path, "processor"))

            if args.prompt_enhancer_model_id is not None:
                # Separate, dedicated enhancer model (required for LTX-2.5); for LTX-2.0/2.3, the same
                # `text_encoder` checkpoint already serves as its own enhancer, so nothing extra is saved.
                prompt_enhancer = AutoModelForImageTextToText.from_pretrained(enhancer_model_id)
                if not args.full_pipeline:
                    prompt_enhancer.to(text_encoder_dtype).save_pretrained(
                        os.path.join(args.output_path, "prompt_enhancer")
                    )

    if args.latent_upsampler or args.upsample_pipeline:
        original_latent_upsampler_ckpt = load_hub_or_local_checkpoint(
            repo_id=args.original_state_dict_repo_id, filename=args.latent_upsampler_filename
        )
        latent_upsampler_config = get_ltx2_spatial_latent_upsampler_config(args.version)
        latent_upsampler = convert_ltx2_latent_upsampler(
            original_latent_upsampler_ckpt,
            latent_upsampler_config,
            dtype=vae_dtype,
        )
        if not args.full_pipeline and not args.upsample_pipeline:
            latent_upsampler.save_pretrained(os.path.join(args.output_path, "latent_upsampler"))

    if args.temporal_latent_upsampler:
        original_temporal_upsampler_ckpt = load_hub_or_local_checkpoint(
            repo_id=args.original_state_dict_repo_id, filename=args.temporal_latent_upsampler_filename
        )
        temporal_latent_upsampler = convert_ltx2_latent_upsampler(
            original_temporal_upsampler_ckpt,
            get_ltx2_temporal_latent_upsampler_config(args.version),
            dtype=vae_dtype,
        )
        temporal_latent_upsampler.save_pretrained(os.path.join(args.output_path, "temporal_latent_upsampler"))

    if args.full_pipeline:
        is_distilled_ckpt = "distilled" in args.combined_filename
        if is_distilled_ckpt:
            # Disable dynamic shifting and terminal shift so that distilled sigmas are used as-is
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=False,
                base_shift=0.95,
                max_shift=2.05,
                base_image_seq_len=1024,
                max_image_seq_len=4096,
                shift_terminal=None,
            )
        else:
            scheduler = FlowMatchEulerDiscreteScheduler(
                use_dynamic_shifting=True,
                base_shift=0.95,
                max_shift=2.05,
                base_image_seq_len=1024,
                max_image_seq_len=4096,
                shift_terminal=0.1,
            )

        pipe = LTX2Pipeline(
            scheduler=scheduler,
            vae=vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
            transformer=transformer,
            vocoder=vocoder,
            processor=processor,
            prompt_enhancer=prompt_enhancer,
        )

        pipe.save_pretrained(args.output_path, safe_serialization=True, max_shard_size="5GB")

    if args.upsample_pipeline:
        pipe = LTX2LatentUpsamplePipeline(vae=vae, latent_upsampler=latent_upsampler)

        # As two diffusers pipelines cannot be in the same directory, save the upsampling pipeline to its own directory
        if args.upsample_output_path:
            upsample_output_path = args.upsample_output_path
        else:
            upsample_output_path = args.output_path
        pipe.save_pretrained(upsample_output_path, safe_serialization=True, max_shard_size="5GB")


if __name__ == "__main__":
    args = get_args()
    main(args)
