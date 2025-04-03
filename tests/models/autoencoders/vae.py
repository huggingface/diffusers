def get_autoencoder_kl_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [2, 4]
    norm_num_groups = norm_num_groups or 2
    init_dict = {
        "block_out_channels": block_out_channels,
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "latent_channels": 4,
        "norm_num_groups": norm_num_groups,
    }
    return init_dict


def get_asym_autoencoder_kl_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [2, 4]
    norm_num_groups = norm_num_groups or 2
    init_dict = {
        "in_channels": 3,
        "out_channels": 3,
        "down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "down_block_out_channels": block_out_channels,
        "layers_per_down_block": 1,
        "up_block_types": ["UpDecoderBlock2D"] * len(block_out_channels),
        "up_block_out_channels": block_out_channels,
        "layers_per_up_block": 1,
        "act_fn": "silu",
        "latent_channels": 4,
        "norm_num_groups": norm_num_groups,
        "sample_size": 32,
        "scaling_factor": 0.18215,
    }
    return init_dict


def get_autoencoder_tiny_config(block_out_channels=None):
    block_out_channels = (len(block_out_channels) * [32]) if block_out_channels is not None else [32, 32]
    init_dict = {
        "in_channels": 3,
        "out_channels": 3,
        "encoder_block_out_channels": block_out_channels,
        "decoder_block_out_channels": block_out_channels,
        "num_encoder_blocks": [b // min(block_out_channels) for b in block_out_channels],
        "num_decoder_blocks": [b // min(block_out_channels) for b in reversed(block_out_channels)],
    }
    return init_dict


def get_consistency_vae_config(block_out_channels=None, norm_num_groups=None):
    block_out_channels = block_out_channels or [2, 4]
    norm_num_groups = norm_num_groups or 2
    return {
        "encoder_block_out_channels": block_out_channels,
        "encoder_in_channels": 3,
        "encoder_out_channels": 4,
        "encoder_down_block_types": ["DownEncoderBlock2D"] * len(block_out_channels),
        "decoder_add_attention": False,
        "decoder_block_out_channels": block_out_channels,
        "decoder_down_block_types": ["ResnetDownsampleBlock2D"] * len(block_out_channels),
        "decoder_downsample_padding": 1,
        "decoder_in_channels": 7,
        "decoder_layers_per_block": 1,
        "decoder_norm_eps": 1e-05,
        "decoder_norm_num_groups": norm_num_groups,
        "encoder_norm_num_groups": norm_num_groups,
        "decoder_num_train_timesteps": 1024,
        "decoder_out_channels": 6,
        "decoder_resnet_time_scale_shift": "scale_shift",
        "decoder_time_embedding_type": "learned",
        "decoder_up_block_types": ["ResnetUpsampleBlock2D"] * len(block_out_channels),
        "scaling_factor": 1,
        "latent_channels": 4,
    }


def get_autoencoder_oobleck_config(block_out_channels=None):
    init_dict = {
        "encoder_hidden_size": 12,
        "decoder_channels": 12,
        "decoder_input_channels": 6,
        "audio_channels": 2,
        "downsampling_ratios": [2, 4],
        "channel_multiples": [1, 2],
    }
    return init_dict
