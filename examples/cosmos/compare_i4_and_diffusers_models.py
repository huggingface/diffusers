import argparse
import os
import sys
from pathlib import Path
import torch
from diffusers import Cosmos2_5_PredictBasePipeline


#sys.path.insert(0, "./cosmos-predict2.5/cosmos-predict2/")
#sys.path.insert(0, "./cosmos-predict2.5")
#sys.path.insert(0, ".")

from cosmos_predict2.config import SetupArguments
from cosmos_predict2.inference import Inference


class MockSafetyChecker:
    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, *args, **kwargs):
        return True

    def check_video_safety(self, video):
        return video


def load_models(i4_model: str, model_id: str, revision: str | None, dtype: torch.dtype):
    # i4
    setup_args = SetupArguments(
        output_dir=Path("test"),
        model=i4_model,
        disable_guardrails=True,
    )
    inference = Inference(setup_args)
    i4_model = inference.pipe.model

    # diffusers
    diffusers_pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        revision=revision,
        text_encoder_attn_implementation="flash_attention_2",
        safety_checker=MockSafetyChecker(),
    )
    return i4_model, diffusers_pipe

def compare_vae(pipe, i4_model, device, dtype):
    from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import retrieve_latents
    vae = pipe.vae.to(device=device, dtype=dtype)
    i4_vae = i4_model.tokenizer
    vae.eval()
    
    raw_state_shape = (1, 3, 9, 704, 1280)
    raw_state = torch.randint(0, 256, raw_state_shape, dtype=torch.uint8).to(device=device, dtype=dtype)
    raw_state = raw_state / 127.5 - 1.0 # [-1, 1]
    latents_mean = pipe.latents_mean.to(device)
    latents_std = pipe.latents_std.to(device)

    with torch.no_grad():
        diffusers_mu = retrieve_latents(vae.encode(raw_state), sample_mode="argmax")
        #diffusers_mu = vae.encode(raw_state).latent_dist.mean
        diffusers_out = (diffusers_mu - latents_mean) * latents_std
        i4_out = i4_vae.encode(raw_state)
    
    diff = (i4_out - diffusers_out).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    print("\n[VAE Encode] i4 vs diffusers")
    print(f"  output shape (i4): {i4_out.shape}")
    print(f"  output shape (diffusers): {diffusers_out.shape}")
    print(f"  output diff: max={max_diff:.6e}, mean={mean_diff:.6e}")
    

def make_dit_inputs(pipe, i4_model, device: str, dtype: torch.dtype, seed: int):
    i4_net = i4_model.net

    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_size = 1
    num_frames = 24
    height = 88
    width = 160
    in_channels = pipe.transformer.config.in_channels - 1

    latents = torch.randn(batch_size, in_channels, num_frames, height, width, device=device, dtype=dtype)
    cond_mask = torch.zeros(batch_size, 1, num_frames, height, width, device=device, dtype=dtype)
    padding_mask = torch.zeros(batch_size, 1, height, width, device=device, dtype=dtype)

    text_len = 512
    if pipe.transformer.config.use_crossattn_projection:
        text_dim = pipe.transformer.config.crossattn_proj_in_channels
    else:
        text_dim = pipe.transformer.config.encoder_hidden_states_channels
    text_embed = torch.randn(batch_size, text_len, text_dim, device=device, dtype=dtype)

    img_context = None
    if getattr(pipe.transformer.config, "img_context_dim_in", None):
        img_tokens = pipe.transformer.config.img_context_num_tokens or 256
        img_dim = pipe.transformer.config.img_context_dim_in
        img_context = torch.zeros(batch_size, img_tokens, img_dim, device=device, dtype=dtype)

    fps = torch.full((batch_size,), 24.0, device=device, dtype=dtype)
    timestep_scale = getattr(i4_net, "timestep_scale", 1.0)
    timesteps_raw = torch.randint(0, 1000, size=(batch_size, num_frames), device=device)
    
    inputs = {
        "latents": latents,
        "cond_mask": cond_mask,
        "padding_mask": padding_mask,
        "text_embed": text_embed,
        "img_context": img_context,
        "fps": fps,
        "i4_timesteps": timesteps_raw,
        "diff_timesteps": (timesteps_raw.float() * timestep_scale).view(batch_size, 1, num_frames, 1, 1),
    }

    print("Input shapes:")
    print(f"  latents: {inputs['latents'].shape}")
    print(f"  text_embed: {inputs['text_embed'].shape}")
    print(f"  cond_mask: {inputs['cond_mask'].shape}")
    print(f"  padding_mask: {inputs['padding_mask'].shape}")
    if inputs["img_context"] is not None:
        print(f"  img_context: {inputs['img_context'].shape}")
    else:
        print("  img_context: None")
    return inputs

def compare_dit(i4_model, pipe, inputs):
    device = inputs["latents"].device
    dtype = inputs["latents"].dtype
    i4_net = i4_model.net.to(device=device, dtype=dtype)
    pipe.transformer.to(device=device, dtype=dtype)
    i4_net.eval()
    pipe.transformer.eval()

    encoder_hidden_states = (
        (inputs["text_embed"], inputs["img_context"]) if inputs["img_context"] is not None else inputs["text_embed"]
    )
    fps = int(inputs["fps"][0].item())

    with torch.no_grad():
        i4_out = i4_net(
            x_B_C_T_H_W=inputs["latents"],
            timesteps_B_T=inputs["i4_timesteps"],
            crossattn_emb=inputs["text_embed"],
            condition_video_input_mask_B_C_T_H_W=inputs["cond_mask"],
            fps=inputs["fps"],
            padding_mask=inputs["padding_mask"],
            img_context_emb=inputs["img_context"],
        )

        diff_out = pipe.transformer(
            hidden_states=inputs["latents"],
            timestep=inputs["diff_timesteps"],
            encoder_hidden_states=encoder_hidden_states,
            fps=fps,
            condition_mask=inputs["cond_mask"],
            padding_mask=inputs["padding_mask"],
            return_dict=False,
        )[0]

    diff = (i4_out.float() - diff_out.float()).abs()

    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    print("\n[Forward pass] i4 vs diffusers")
    print(f"  output shape (i4): {i4_out.shape}")
    print(f"  output shape (diffusers): {diff_out.shape}")
    print(f"  output diff: max={max_diff:.6e}, mean={mean_diff:.6e}")
    return i4_out, diff_out, max_diff, mean_diff


def compare_text_encoder(i4_model, pipe, device, dtype):
    i4_model.text_encoder.model.to(device=device, dtype=dtype)
    pipe.text_encoder.to(device=device, dtype=dtype)
    i4_model.text_encoder.model.eval()
    pipe.text_encoder.eval()

    prompt = "a video of a dog playing in the park"
    i4_text_embeds = i4_model.text_encoder.compute_text_embeddings_online(
        data_batch={"ai_caption": [prompt], "images": None},
        input_caption_key="ai_caption",
    )
    diffuser_text_embeds = pipe._get_prompt_embeds(
        prompt=prompt,
        device=device,
    )

    diff = (i4_text_embeds.float() - diffuser_text_embeds.float()).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    print("\n[Text Embeds] i4 vs diffusers")
    print(f"  output diff: max={max_diff}, mean={mean_diff}")
    


def main():
    parser = argparse.ArgumentParser(description="i4 vs diffusers forward comparison for Cosmos Predict2.5 Base")
    parser.add_argument("--model-id", default="nvidia/Cosmos-Predict2.5-2B")
    parser.add_argument("--revision", default="diffusers/base/post-trained")
    parser.add_argument("--i4-model", default="2B/post-trained")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this comparison script.")

    dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    i4_model, diffusers_pipe = load_models(args.i4_model, args.model_id, args.revision, dtype=dtype)
    compare_vae(diffusers_pipe, i4_model, device="cuda", dtype=torch.float32)

    compare_text_encoder(i4_model, diffusers_pipe, device="cuda", dtype=dtype)

    inputs = make_dit_inputs(diffusers_pipe, i4_model, device="cuda", dtype=dtype, seed=args.seed)
    compare_dit(i4_model, diffusers_pipe, inputs)


if __name__ == "__main__":
    main()