"""DiT numerical parity test: original ACE-Step 1.5 turbo DiT vs diffusers AceStepDiTModel.

Loads the same converted weights into both models and compares forward-pass outputs
on identical random inputs, in fp32 eager/SDPA.

Goal: max-abs-diff ≤ 1e-5 on the final predicted latent.

Run:
    python scripts/dit_parity_test.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Path config via env vars so the script runs on jieyue too. Defaults target the
# local Mac layout. On jieyue set ACESTEP_ORIG / ACESTEP_DIFF.
ORIG_REPO = os.environ.get(
    "ACESTEP_ORIG", "/Users/gongjunmin/timedomain/diffusers_pr_workspace/ACE-Step-1.5"
)
DIFFUSERS_REPO = os.environ.get(
    "ACESTEP_DIFF", "/Users/gongjunmin/timedomain/diffusers_pr_workspace/diffusers"
)
CHECKPOINT_DIR = os.environ.get("ACESTEP_CKPT_DIR", os.path.join(ORIG_REPO, "checkpoints"))
DIT_CONFIG_NAME = os.environ.get("ACESTEP_VARIANT", "acestep-v15-turbo")
DIFFUSERS_CHECKPOINT_DIR = os.environ.get(
    "ACESTEP_DIFF_CKPT", f"/tmp/{DIT_CONFIG_NAME}-diffusers"
)

sys.path.insert(0, ORIG_REPO)
sys.path.insert(0, os.path.join(DIFFUSERS_REPO, "src"))

import torch  # noqa: E402

from safetensors.torch import load_file  # noqa: E402


def build_original_dit(device: torch.device) -> torch.nn.Module:
    """Instantiate the original DiT (from ACE-Step-1.5) and load the turbo weights."""
    from acestep.models.common.configuration_acestep_v15 import AceStepConfig
    from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepDiTModel as OrigDiT

    import json

    with open(os.path.join(CHECKPOINT_DIR, DIT_CONFIG_NAME, "config.json")) as f:
        config_dict = json.load(f)

    cfg = AceStepConfig(**{k: v for k, v in config_dict.items() if k not in ("architectures", "auto_map", "dtype", "transformers_version", "is_turbo", "model_type")})
    cfg._attn_implementation = "sdpa"

    model = OrigDiT(cfg)
    full_sd = load_file(os.path.join(CHECKPOINT_DIR, DIT_CONFIG_NAME, "model.safetensors"))
    dit_sd = {k[len("decoder.") :]: v.to(torch.float32) for k, v in full_sd.items() if k.startswith("decoder.")}
    missing, unexpected = model.load_state_dict(dit_sd, strict=False)
    if missing:
        # RoPE inv_freq buffer is created per-instance and not saved; that's fine.
        non_rope_missing = [k for k in missing if "rotary_emb.inv_freq" not in k]
        if non_rope_missing:
            raise RuntimeError(f"Original DiT missing keys: {non_rope_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"Original DiT unexpected keys: {unexpected[:10]}")

    return model.to(device).eval()


def build_diffusers_dit(device: torch.device) -> torch.nn.Module:
    """Instantiate the diffusers AceStepDiTModel and load converted weights."""
    from diffusers import AceStepDiTModel

    model = AceStepDiTModel.from_pretrained(
        os.path.join(DIFFUSERS_CHECKPOINT_DIR, "transformer"), torch_dtype=torch.float32
    )
    return model.to(device).eval()


def make_inputs(device, dtype, seed=12345, batch=1, seq_len=25 * 4, enc_seq_len=96):
    """Build a deterministic set of DiT inputs."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    # hidden_states: noise latents, (B, T, acoustic_dim=64)
    hidden_states = torch.randn(batch, seq_len, 64, generator=g, dtype=dtype, device="cpu")
    # context_latents: (B, T, acoustic_dim + 64). Original DiT concats to hidden_states on dim=-1
    # to make 192 in-channels. context_latents_dim = in_channels - acoustic_dim = 192 - 64 = 128.
    context_latents = torch.randn(batch, seq_len, 128, generator=g, dtype=dtype, device="cpu")
    # encoder_hidden_states: fused conditioning, (B, L_enc, hidden_size=2048)
    encoder_hidden_states = torch.randn(batch, enc_seq_len, 2048, generator=g, dtype=dtype, device="cpu")
    # timesteps: in [0, 1] typical for flow matching
    t = torch.rand(batch, generator=g, dtype=dtype, device="cpu") * 0.8 + 0.1  # avoid 0/1
    r = t.clone()  # timestep_r = t at inference

    return {
        "hidden_states": hidden_states.to(device),
        "timestep": t.to(device),
        "timestep_r": r.to(device),
        "encoder_hidden_states": encoder_hidden_states.to(device),
        "context_latents": context_latents.to(device),
    }


@torch.no_grad()
def run_original(model, inputs):
    """Call the original DiT. Returns the predicted velocity (same shape as hidden_states)."""
    # Original DiT forward expects: hidden_states, timestep, timestep_r, attention_mask,
    # encoder_hidden_states, encoder_attention_mask, context_latents.
    # Note: the forward actually overrides attention_mask/encoder_attention_mask to None
    # and rebuilds 4D masks internally — so what we pass is ignored.
    outputs = model(
        hidden_states=inputs["hidden_states"],
        timestep=inputs["timestep"],
        timestep_r=inputs["timestep_r"],
        attention_mask=None,
        encoder_hidden_states=inputs["encoder_hidden_states"],
        encoder_attention_mask=None,
        context_latents=inputs["context_latents"],
        use_cache=False,
    )
    # Returns (hidden_states, past_key_values[, attentions]) — take [0].
    return outputs[0]


@torch.no_grad()
def run_diffusers(model, inputs):
    out = model(
        hidden_states=inputs["hidden_states"],
        timestep=inputs["timestep"],
        timestep_r=inputs["timestep_r"],
        encoder_hidden_states=inputs["encoder_hidden_states"],
        context_latents=inputs["context_latents"],
        return_dict=False,
    )
    return out[0]


def diff_stats(a: torch.Tensor, b: torch.Tensor):
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rms": d.pow(2).mean().sqrt().item(),
        "a_std": a.std().item(),
        "b_std": b.std().item(),
        "shape": tuple(a.shape),
    }


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # MPS fp32 is fine for parity; keep on CPU if you want bit-exact eager SDPA.
    # Override via env:
    if os.environ.get("FORCE_CPU") == "1":
        device = torch.device("cpu")
    dtype = torch.float32

    print(f"[parity] device={device}, dtype={dtype}")

    print("[parity] loading original DiT (turbo, fp32) ...")
    orig = build_original_dit(device)
    print(f"[parity]   params: {sum(p.numel() for p in orig.parameters()) / 1e6:.2f}M")

    print("[parity] loading diffusers DiT (fp32) ...")
    diff = build_diffusers_dit(device)
    print(f"[parity]   params: {sum(p.numel() for p in diff.parameters()) / 1e6:.2f}M")

    # Run several configs to cover t==r, t!=r, various seq lengths, batch>1.
    cases = [
        dict(name="short_t=r", seed=12345, batch=1, seq_len=32, enc_seq_len=32, r_equals_t=True),
        dict(name="short_t!=r", seed=54321, batch=1, seq_len=32, enc_seq_len=32, r_equals_t=False),
        dict(name="medium_batch2", seed=777, batch=2, seq_len=64, enc_seq_len=80, r_equals_t=True),
        dict(name="odd_len_pad", seed=4242, batch=1, seq_len=33, enc_seq_len=17, r_equals_t=True),
    ]
    tol = 1e-5
    all_pass = True
    for case in cases:
        name = case.pop("name")
        r_equals_t = case.pop("r_equals_t")
        inputs = make_inputs(device, dtype, **case)
        if not r_equals_t:
            # Mean-flow style: r < t; time_embed_r sees (t - r) as a nonzero value.
            inputs["timestep_r"] = inputs["timestep_r"] * 0.3

        print(f"\n[parity:{name}] hidden={tuple(inputs['hidden_states'].shape)}, "
              f"enc={tuple(inputs['encoder_hidden_states'].shape)}, "
              f"t={inputs['timestep'].tolist()}, r={inputs['timestep_r'].tolist()}")
        y_orig = run_original(orig, inputs)
        y_diff = run_diffusers(diff, inputs)
        stats = diff_stats(y_orig.float().cpu(), y_diff.float().cpu())
        status = "PASS" if stats["max_abs"] <= tol else "FAIL"
        if stats["max_abs"] > tol:
            all_pass = False
        print(f"[parity:{name}] {status}  max_abs={stats['max_abs']:.3e}  "
              f"mean_abs={stats['mean_abs']:.3e}  rms={stats['rms']:.3e}  "
              f"out_std={stats['a_std']:.3f} (tol={tol:.0e})")

    print("\n[parity] ALL PASS" if all_pass else "\n[parity] SOME FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
