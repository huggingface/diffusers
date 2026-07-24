import argparse

import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

from diffusers import (
    AutoencoderKLLTX2Audio,
    AutoencoderKLLTX2Video,
    FlowMatchEulerDiscreteScheduler,
    LTX2Pipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2 import LTX2TextConnectors
from diffusers.pipelines.ltx2.vocoder import LTX2Vocoder, LTX2VocoderWithBWE


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_text_encoder_ckpt_id)
    text_encoder = Gemma3ForConditionalGeneration.from_pretrained(args.base_text_encoder_ckpt_id)

    torch.manual_seed(0)
    transformer = LTX2VideoTransformer3DModel(
        in_channels=4,
        out_channels=4,
        patch_size=1,
        patch_size_t=1,
        num_attention_heads=2,
        attention_head_dim=8,
        # cross_attention_dim is the connector output dim; it must equal the video connector's
        # inner_dim (num_heads * head_dim). See the connector shape-dependency note below.
        cross_attention_dim=16,
        gated_attn=True,  # LTX-2.3
        cross_attn_mod=True,  # LTX-2.3
        audio_in_channels=4,
        audio_out_channels=4,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        # audio_cross_attention_dim must equal the audio connector's inner_dim (see note below).
        audio_cross_attention_dim=8,
        audio_gated_attn=True,  # LTX-2.3
        audio_cross_attn_mod=True,  # LTX-2.3
        num_layers=2,
        qk_norm="rms_norm_across_heads",
        caption_channels=text_encoder.config.text_config.hidden_size,
        rope_double_precision=True,
        rope_type="split",
        use_prompt_embeddings=False,  # LTX-2.3
        perturbed_attn=True,  # LTX-2.3
        ff_bias=False,  # LTX-2.4
        audio_ff_bias=False,  # LTX-2.4
        use_prompt_adaln_single=False,  # LTX-2.4
    )

    # Connector <-> transformer shape dependencies (must hold per modality, video and audio):
    #   1. The connector's LTX2ConnectorTransformer1d has no input projection, so its learnable
    #      registers and 1D transformer blocks all operate at `inner_dim = num_attention_heads *
    #      attention_head_dim`. The parent connector projects the text features to `*_hidden_dim`
    #      (via `video_text_proj_in` / `audio_text_proj_in`) before feeding them in, so
    #      `*_hidden_dim` MUST equal that connector `inner_dim`. `*_hidden_dim` defaults to the
    #      full-size 4096 / 2048, so it must be set explicitly when scaling the connector down.
    #   2. The connector output (its `inner_dim`) becomes the transformer's cross-attention K/V
    #      input, so the connector `inner_dim` MUST equal the transformer's `cross_attention_dim`
    #      (video) / `audio_cross_attention_dim` (audio).
    # Net invariant, per modality:
    #   connector num_heads * head_dim == connector *_hidden_dim == transformer *cross_attention_dim
    # Here we mirror the transformer's own head config so video -> 2 * 8 = 16 and audio -> 2 * 4 = 8.
    torch.manual_seed(0)
    connectors = LTX2TextConnectors(
        caption_channels=text_encoder.config.text_config.hidden_size,
        text_proj_in_factor=text_encoder.config.text_config.num_hidden_layers + 1,
        video_connector_num_attention_heads=2,  # 2 * 8 = 16 == transformer.cross_attention_dim
        video_connector_attention_head_dim=8,
        video_hidden_dim=16,  # must equal connector inner_dim (2 * 8)
        video_connector_num_layers=1,
        video_connector_num_learnable_registers=2,
        video_gated_attn=True,  # LTX-2.3
        audio_connector_num_attention_heads=2,  # 2 * 4 = 8 == transformer.audio_cross_attention_dim
        audio_connector_attention_head_dim=4,
        audio_hidden_dim=8,  # must equal connector inner_dim (2 * 4)
        audio_connector_num_layers=1,
        audio_connector_num_learnable_registers=2,
        audio_gated_attn=True,  # LTX-2.3
        connector_rope_base_seq_len=32,
        rope_theta=10000.0,
        rope_double_precision=True,
        causal_temporal_positioning=False,
        rope_type="split",
        per_modality_projections=True,  # LTX-2.3
        proj_bias=True,  # LTX-2.3
    )

    torch.manual_seed(0)
    vae = AutoencoderKLLTX2Video(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=(8,),
        decoder_block_out_channels=(8,),
        layers_per_block=(1,),
        decoder_layers_per_block=(1, 1),
        spatio_temporal_scaling=(True,),
        decoder_spatio_temporal_scaling=(True,),
        decoder_inject_noise=(False, False),
        downsample_type=("spatial",),
        upsample_residual=(False,),
        upsample_factor=(1,),
        timestep_conditioning=False,
        patch_size=1,
        patch_size_t=1,
        encoder_causal=True,
        decoder_causal=False,
    )
    vae.use_framewise_encoding = False
    vae.use_framewise_decoding = False

    torch.manual_seed(0)
    audio_vae = AutoencoderKLLTX2Audio(
        base_channels=4,
        output_channels=2,
        ch_mult=(1,),
        num_res_blocks=1,
        attn_resolutions=None,
        in_channels=2,
        resolution=32,
        latent_channels=2,
        norm_type="pixel",
        causality_axis="height",
        dropout=0.0,
        mid_block_add_attention=False,
        sample_rate=16000,
        mel_hop_length=160,
        is_causal=True,
        mel_bins=8,
    )

    torch.manual_seed(0)
    # TODO: use LTX2VocoderWithBWE since this is probably what LTX-2.4 uses??
    vocoder = LTX2Vocoder(
        in_channels=audio_vae.config.output_channels * audio_vae.config.mel_bins,
        hidden_channels=32,
        out_channels=2,
        upsample_kernel_sizes=[4, 4],
        upsample_factors=[2, 2],
        resnet_kernel_sizes=[3],
        resnet_dilations=[[1, 3, 5]],
        leaky_relu_negative_slope=0.1,
        output_sampling_rate=16000,
    )

    # NOTE: for now match LTX-2.3's scheduler params
    scheduler = FlowMatchEulerDiscreteScheduler(
        use_dynamic_shifting=True,
        base_shift=0.95,
        max_shift=2.05,
        shift_terminal=0.1,
        base_image_seq_len=1024,
        max_image_seq_len=4096,
    )

    components = {
        "transformer": transformer,
        "vae": vae,
        "audio_vae": audio_vae,
        "scheduler": scheduler,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "connectors": connectors,
        "vocoder": vocoder,
        "processor": None,
    }

    pipe = LTX2Pipeline(**components)
    pipe.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_text_encoder_ckpt_id", type=str, default="hf-internal-testing/tiny-gemma3")
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()

    main(args)
