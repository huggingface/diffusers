import inspect
import os
from argparse import ArgumentParser

import numpy as np
import torch
from muse import MaskGiTUViT, VQGANModel
from muse import PipelineMuse as OldPipelineMuse
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from diffusers import VQModel
from diffusers.loaders.conversion import get_conversion
from diffusers.models.attention_processor import AttnProcessor
from diffusers.models.unets.uvit_2d import UVit2DModel
from diffusers.pipelines.deprecated.amused.pipeline_amused import AmusedPipeline
from diffusers.schedulers import AmusedScheduler


torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True)

# Enable CUDNN deterministic mode
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False

device = "cuda"


def main():
    args = ArgumentParser()
    args.add_argument("--model_256", action="store_true")
    args.add_argument("--write_to", type=str, required=False, default=None)
    args.add_argument("--transformer_path", type=str, required=False, default=None)
    args = args.parse_args()

    transformer_path = args.transformer_path
    subfolder = "transformer"

    if transformer_path is None:
        if args.model_256:
            transformer_path = "openMUSE/muse-256"
        else:
            transformer_path = (
                "../research-run-512-checkpoints/research-run-512-with-downsample-checkpoint-554000/unwrapped_model/"
            )
            subfolder = None

    old_transformer = MaskGiTUViT.from_pretrained(transformer_path, subfolder=subfolder)

    old_transformer.to(device)

    old_vae = VQGANModel.from_pretrained("openMUSE/muse-512", subfolder="vae")
    old_vae.to(device)

    vqvae = make_vqvae(old_vae)

    tokenizer = CLIPTokenizer.from_pretrained("openMUSE/muse-512", subfolder="text_encoder")

    text_encoder = CLIPTextModelWithProjection.from_pretrained("openMUSE/muse-512", subfolder="text_encoder")
    text_encoder.to(device)

    transformer = make_transformer(old_transformer, args.model_256)

    scheduler = AmusedScheduler(mask_token_id=old_transformer.config.mask_token_id)

    new_pipe = AmusedPipeline(
        vqvae=vqvae, tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, scheduler=scheduler
    )

    old_pipe = OldPipelineMuse(
        vae=old_vae, transformer=old_transformer, text_encoder=text_encoder, tokenizer=tokenizer
    )
    old_pipe.to(device)

    if args.model_256:
        transformer_seq_len = 256
        orig_size = (256, 256)
    else:
        transformer_seq_len = 1024
        orig_size = (512, 512)

    old_out = old_pipe(
        "dog",
        generator=torch.Generator(device).manual_seed(0),
        transformer_seq_len=transformer_seq_len,
        orig_size=orig_size,
        timesteps=12,
    )[0]

    new_out = new_pipe("dog", generator=torch.Generator(device).manual_seed(0)).images[0]

    old_out = np.array(old_out)
    new_out = np.array(new_out)

    diff = np.abs(old_out.astype(np.float64) - new_out.astype(np.float64))

    # assert diff diff.sum() == 0
    print("skipping pipeline full equivalence check")

    print(f"max diff: {diff.max()}, diff.sum() / diff.size {diff.sum() / diff.size}")

    if args.model_256:
        assert diff.max() <= 3
        assert diff.sum() / diff.size < 0.7
    else:
        assert diff.max() <= 1
        assert diff.sum() / diff.size < 0.4

    if args.write_to is not None:
        new_pipe.save_pretrained(args.write_to)


def make_transformer(old_transformer, model_256):
    args = dict(old_transformer.config)
    force_down_up_sample = args["force_down_up_sample"]

    signature = inspect.signature(UVit2DModel.__init__)

    args_ = {
        "downsample": force_down_up_sample,
        "upsample": force_down_up_sample,
        "block_out_channels": args["block_out_channels"][0],
        "sample_size": 16 if model_256 else 32,
    }

    for s in list(signature.parameters.keys()):
        if s in ["self", "downsample", "upsample", "sample_size", "block_out_channels"]:
            continue

        args_[s] = args[s]

    new_transformer = UVit2DModel(**args_)
    new_transformer.to(device)

    new_transformer.set_attn_processor(AttnProcessor())

    state_dict = old_transformer.state_dict()

    state_dict = get_conversion("UVit2DModel", dict(new_transformer.config)).to_diffusers(state_dict)

    new_transformer.load_state_dict(state_dict)

    input_ids = torch.randint(0, 10, (1, 32, 32), device=old_transformer.device)
    encoder_hidden_states = torch.randn((1, 77, 768), device=old_transformer.device)
    cond_embeds = torch.randn((1, 768), device=old_transformer.device)
    micro_conds = torch.tensor([[512, 512, 0, 0, 6]], dtype=torch.float32, device=old_transformer.device)

    old_out = old_transformer(input_ids.reshape(1, -1), encoder_hidden_states, cond_embeds, micro_conds)
    old_out = old_out.reshape(1, 32, 32, 8192).permute(0, 3, 1, 2)

    new_out = new_transformer(input_ids, encoder_hidden_states, cond_embeds, micro_conds)

    # NOTE: these differences are solely due to using the geglu block that has a single linear layer of
    # double output dimension instead of two different linear layers
    max_diff = (old_out - new_out).abs().max()
    total_diff = (old_out - new_out).abs().sum()
    print(f"Transformer max_diff: {max_diff} total_diff:  {total_diff}")
    assert max_diff < 0.01
    assert total_diff < 1500

    return new_transformer


def make_vqvae(old_vae):
    new_vae = VQModel(
        act_fn="silu",
        block_out_channels=[128, 256, 256, 512, 768],
        down_block_types=[
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ],
        in_channels=3,
        latent_channels=64,
        layers_per_block=2,
        norm_num_groups=32,
        num_vq_embeddings=8192,
        out_channels=3,
        sample_size=32,
        up_block_types=[
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
            "UpDecoderBlock2D",
        ],
        mid_block_add_attention=False,
        lookup_from_codebook=True,
    )
    new_vae.to(device)

    new_state_dict = get_conversion("VQModel", dict(new_vae.config)).to_diffusers(old_vae.state_dict())

    new_vae.load_state_dict(new_state_dict)

    input = torch.randn((1, 3, 512, 512), device=device)
    input = input.clamp(-1, 1)

    old_encoder_output = old_vae.quant_conv(old_vae.encoder(input))
    new_encoder_output = new_vae.quant_conv(new_vae.encoder(input))
    assert (old_encoder_output == new_encoder_output).all()

    old_decoder_output = old_vae.decoder(old_vae.post_quant_conv(old_encoder_output))
    new_decoder_output = new_vae.decoder(new_vae.post_quant_conv(new_encoder_output))

    # assert (old_decoder_output == new_decoder_output).all()
    print("kipping vae decoder equivalence check")
    print(f"vae decoder diff {(old_decoder_output - new_decoder_output).float().abs().sum()}")

    old_output = old_vae(input)[0]
    new_output = new_vae(input)[0]

    # assert (old_output == new_output).all()
    print("skipping full vae equivalence check")
    print(f"vae full diff {(old_output - new_output).float().abs().sum()}")

    return new_vae

    # fmt: on


if __name__ == "__main__":
    main()
