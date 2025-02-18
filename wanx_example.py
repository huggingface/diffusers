import torch
from transformers import AutoTokenizer, T5EncoderModel
from diffusers import AutoencoderKLHunyuanVideo, WanxPipeline, WanxTransformer3DModel, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video

# model_id = "wanx/wanx"
# transformer = WanxTransformer3DModel.from_pretrained(
#     model_id, torch_dtype=torch.bfloat16
# )
# pipe = WanxVideoPipeline.from_pretrained(model_id, transformer=transformer, torch_dtype=torch.bfloat16)

device = "cuda"
seed = 0

# TODO: impl AutoencoderKLWanx
vae = AutoencoderKLHunyuanVideo(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=(
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
                "HunyuanVideoDownBlock3D",
            ),
            up_block_types=(
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
                "HunyuanVideoUpBlock3D",
            ),
            block_out_channels=(8, 8, 8, 8),
            layers_per_block=1,
            act_fn="silu",
            norm_num_groups=4,
            scaling_factor=0.476986,
            spatial_compression_ratio=8,
            temporal_compression_ratio=4,
            mid_block_add_attention=True,
        )

# TODO: impl FlowDPMSolverMultistepScheduler
scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)

text_encoder = T5EncoderModel.from_pretrained("hf-internal-testing/tiny-random-t5")
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5")


transformer = WanxTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads = 12,
            attention_head_dim = 128,
            in_channels = 16,
            out_channels = 16,
            text_dim = text_encoder.config.d_model,
            freq_dim = 256,
            ffn_dim = 8960,
            num_layers = 1,
            window_size = (-1, -1),
            cross_attn_norm = True,
            qk_norm = True,
            eps = 1e-6,
            # for i2v
            add_img_emb = False,
            added_kv_proj_dim = None,
        )

print(transformer)

components = {
    "transformer": transformer,
    "vae": vae,
    "scheduler": scheduler,
    "text_encoder": text_encoder,
    "tokenizer": tokenizer,
}

pipe = WanxPipeline(**components)

pipe.to(device)

generator = torch.Generator(device=device).manual_seed(seed)
inputs = {
    "prompt": "dance monkey",
    "negative_prompt": "negative", # TODO
    "generator": generator,
    "num_inference_steps": 2,
    "guidance_scale": 6.0,
    "height": 16,
    "width": 16,
    "num_frames": 8,
    "max_sequence_length": 16,
    "output_type": "pt",
}

video = pipe(**inputs).frames[0]

