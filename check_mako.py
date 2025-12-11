import torch
import torch.nn as nn
from diffusers import AutoModel, WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
import triton
from functools import partial
from argparse import ArgumentParser

CKPT_ID = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


@torch.no_grad()
def fuse_qkv_for_wan_transformer_3d_model(model: "WanTransformer3DModel") -> "WanTransformer3DModel":
    """
    In-place Q/K/V fusion for WanTransformer3DModel.

    For each WanTransformerBlock:
      * attn1: create (w_qkv_self, b_qkv_self) by concatenating Q/K/V.
      * attn2: create (w_kv_cross, b_kv_cross) by concatenating K/V.

    The fused tensors are registered as nn.Parameters on the corresponding
    WanAttention modules and populated via `load_state_dict`.
    """

    for block in getattr(model, "blocks", []):
        # ------------------------------------------------------------------
        # 1. Self-attention: fuse Q, K, V -> (w_qkv_self, b_qkv_self)
        # ------------------------------------------------------------------
        attn1 = getattr(block, "attn1", None)
        if attn1 is not None and not hasattr(attn1, "w_qkv_self"):
            # Grab existing projections
            w_q = attn1.to_q.weight.data
            w_k = attn1.to_k.weight.data
            w_v = attn1.to_v.weight.data
            b_q = attn1.to_q.bias.data
            b_k = attn1.to_k.bias.data
            b_v = attn1.to_v.bias.data

            # Fuse along the out_features dimension (dim=0)
            fused_w = torch.cat([w_q, w_k, w_v], dim=0)
            fused_b = torch.cat([b_q, b_k, b_v], dim=0)

            out_features, in_features = fused_w.shape
            device = fused_w.device
            dtype = fused_w.dtype

            # Register fused parameters with the requested names
            attn1.register_parameter(
                "w_qkv_self",
                nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype)),
            )
            attn1.register_parameter(
                "b_qkv_self",
                nn.Parameter(torch.empty((out_features,), device=device, dtype=dtype)),
            )

            # Load via state-dict mechanism (so it works nicely with checkpoints)
            attn1.load_state_dict(
                {"w_qkv_self": fused_w, "b_qkv_self": fused_b},
                strict=False,
            )

        # ------------------------------------------------------------------
        # 2. Cross-attention: fuse K, V -> (w_kv_cross, b_kv_cross)
        # ------------------------------------------------------------------
        attn2 = getattr(block, "attn2", None)
        if attn2 is not None and not hasattr(attn2, "w_kv_cross"):
            w_k = attn2.to_k.weight.data
            w_v = attn2.to_v.weight.data
            b_k = attn2.to_k.bias.data
            b_v = attn2.to_v.bias.data

            fused_w = torch.cat([w_k, w_v], dim=0)
            fused_b = torch.cat([b_k, b_v], dim=0)

            out_features, in_features = fused_w.shape
            device = fused_w.device
            dtype = fused_w.dtype

            attn2.register_parameter(
                "w_kv_cross",
                nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype)),
            )
            attn2.register_parameter(
                "b_kv_cross",
                nn.Parameter(torch.empty((out_features,), device=device, dtype=dtype)),
            )

            attn2.load_state_dict(
                {"w_kv_cross": fused_w, "b_kv_cross": fused_b},
                strict=False,
            )

    return model


def load_pipeline():
    vae = AutoModel.from_pretrained(CKPT_ID, subfolder="vae", torch_dtype=torch.float32)
    pipeline = WanPipeline.from_pretrained(
        CKPT_ID, vae=vae, torch_dtype=torch.bfloat16
    ).to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def get_prompts():
    prompt = """
    The camera rushes from far to near in a low-angle shot,
    revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in
    for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground.
    Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic
    shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
    """
    negative_prompt = """
    Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality,
    low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured,
    misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
    """
    return prompt, negative_prompt


# Fixing batch size of 2 and `max_sequence_length` of 256 because of the kernels.
def run_inference(pipeline, prompt, negative_prompt, num_inference_steps=50):
    output = pipeline(
        prompt=[prompt] * 2,
        negative_prompt=negative_prompt,
        num_frames=81,
        guidance_scale=5.0,
        num_inference_steps=num_inference_steps,
        max_sequence_length=256,
        generator=torch.manual_seed(0)
    ).frames[0]
    return output


def main(args):
    pipe = load_pipeline()
    if args.use_mako:
        from diffusers.models.transformers import wan_mako_attention_processor

        print("Using MaKO kernel.")
        pipe.transformer = fuse_qkv_for_wan_transformer_3d_model(pipe.transformer)
        pipe.transformer.set_attn_processor(wan_mako_attention_processor.WanMakoAttnProcessor())

    if args.use_compile:
        pipe.transformer.compile_repeated_blocks()

    prompt, negative_prompt = get_prompts()
    for _ in range(3):
        _ = run_inference(pipe, prompt, negative_prompt, 1)
    inference_func = partial(run_inference, pipe, prompt=prompt, negative_prompt=negative_prompt)

    latency = triton.testing.do_bench(inference_func, warmup=1, rep=1)
    print(f"{args=}, {latency=} seconds.")
    
    output = inference_func()
    export_to_video(output, f"output_mako@{args.use_mako}_compile@{args.use_compile}.mp4", fps=16)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--use_mako", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    args = parser.parse_args()
    main(args)