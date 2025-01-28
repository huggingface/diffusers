import torch
from collections import OrderedDict
from diffusers import CogView4Transformer2DModel

def load_state_dict_sat(file_path):
    """Load the SAT state dictionary from a given file path."""
    # Typically, the stored SAT ckpt is in the format: {'module': {...}}
    ckpt = torch.load(file_path, map_location="cuda")
    return ckpt["module"]


def extract_qkv_from_sat(state_dict, layer_idx):
    """
    Extract QKV weights and biases from a SAT state_dict.
    Expects keys like:
      model.diffusion_model.transformer.layers.{layer_idx}.attention.query_key_value
    """
    prefix = f"model.diffusion_model.transformer.layers.{layer_idx}.attention.query_key_value"
    w = state_dict[f"{prefix}.weight"].clone()
    b = state_dict[f"{prefix}.bias"].clone()
    return (w, b)


def load_state_dict_cogview(cogview_path):
    """
    Loads the CogView4 model from diffusers and returns its state_dict().
    NOTE: You should adjust 'torch_dtype' and 'device_map' as appropriate.
    """
    cogview_model = CogView4Transformer2DModel.from_pretrained(
        cogview_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return cogview_model.state_dict()


def extract_qkv_from_cogview(state_dict, layer_idx, num_heads, head_dim, hidden_dim):
    """
    Extract Q, K, V from CogView4 checkpoint and reshape them into the same shape as SAT’s QKV.
    For each layer i:
      Q prefix: transformer_blocks.{layer_idx}.attn1.to_q
      K prefix: transformer_blocks.{layer_idx}.attn1.to_k
      V prefix: transformer_blocks.{layer_idx}.attn1.to_v
    Final shape must match SAT's [3*hidden_dim, hidden_dim] for weight, and [3*hidden_dim] for bias.
    """
    q_prefix = f"transformer_blocks.{layer_idx}.attn1.to_q"
    k_prefix = f"transformer_blocks.{layer_idx}.attn1.to_k"
    v_prefix = f"transformer_blocks.{layer_idx}.attn1.to_v"

    # Extract
    q_weight = state_dict[f"{q_prefix}.weight"].clone()
    k_weight = state_dict[f"{k_prefix}.weight"].clone()
    v_weight = state_dict[f"{v_prefix}.weight"].clone()

    q_bias = state_dict[f"{q_prefix}.bias"].clone()
    k_bias = state_dict[f"{k_prefix}.bias"].clone()
    v_bias = state_dict[f"{v_prefix}.bias"].clone()

    # Reshape weights: [hidden_dim, hidden_dim] -> [num_heads, head_dim, hidden_dim]
    # Then concat along the first dimension (which will become 3*num_heads*head_dim)
    q_weight = q_weight.view(num_heads, head_dim, hidden_dim)
    k_weight = k_weight.view(num_heads, head_dim, hidden_dim)
    v_weight = v_weight.view(num_heads, head_dim, hidden_dim)

    qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)  # shape: (3*num_heads, head_dim, hidden_dim)
    qkv_weight = qkv_weight.view(3 * num_heads * head_dim, hidden_dim)  # flatten

    # Reshape biases: [hidden_dim] -> [num_heads, head_dim]
    q_bias = q_bias.view(num_heads, head_dim)
    k_bias = k_bias.view(num_heads, head_dim)
    v_bias = v_bias.view(num_heads, head_dim)

    qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)  # (3*num_heads, head_dim)
    qkv_bias = qkv_bias.view(3 * num_heads * head_dim)

    return (qkv_weight, qkv_bias)

def create_sat_state_dict_from_megatron(megatron_ckpt_dict, num_layers=48, num_heads=32, hidden_size=3072):
    """
    Convert a loaded Megatron checkpoint's 'model' dictionary into the same
    format used by SAT. This returns something like {'module': {...}} for
    easy comparison with SAT.

    The code below is adapted from your 'create_sat_state_dict' function,
    but we rename it here to keep it direct.
    """
    from tqdm import tqdm

    hidden_size_per_head = hidden_size // num_heads
    mega_weight = megatron_ckpt_dict["model"]
    sat_weight = {}

    # --- patch_embed ---
    sat_weight["model.diffusion_model.mixins.patch_embed.proj.weight"] = \
        mega_weight["encoder_expand_linear.weight"].reshape(hidden_size, 64).clone()
    sat_weight["model.diffusion_model.mixins.patch_embed.proj.bias"] = \
        mega_weight["encoder_expand_linear.bias"].clone()

    sat_weight["model.diffusion_model.mixins.patch_embed.text_proj.weight"] = \
        mega_weight["text_projector.weight"].clone()
    sat_weight["model.diffusion_model.mixins.patch_embed.text_proj.bias"] = \
        mega_weight["text_projector.bias"].clone()

    # --- time embedding ---
    sat_weight["model.diffusion_model.time_embed.0.weight"] = \
        mega_weight["time_embedding.time_embed.0.weight"].clone()
    sat_weight["model.diffusion_model.time_embed.0.bias"] = \
        mega_weight["time_embedding.time_embed.0.bias"].clone()
    sat_weight["model.diffusion_model.time_embed.2.weight"] = \
        mega_weight["time_embedding.time_embed.2.weight"].clone()
    sat_weight["model.diffusion_model.time_embed.2.bias"] = \
        mega_weight["time_embedding.time_embed.2.bias"].clone()

    # --- label embedding ---
    sat_weight["model.diffusion_model.label_emb.0.0.weight"] = \
        mega_weight["label_embedding.label_embed.0.weight"].clone()
    sat_weight["model.diffusion_model.label_emb.0.0.bias"] = \
        mega_weight["label_embedding.label_embed.0.bias"].clone()
    sat_weight["model.diffusion_model.label_emb.0.2.weight"] = \
        mega_weight["label_embedding.label_embed.2.weight"].clone()
    sat_weight["model.diffusion_model.label_emb.0.2.bias"] = \
        mega_weight["label_embedding.label_embed.2.bias"].clone()

    # --- layers ---
    for i in tqdm(range(num_layers), desc="Converting Megatron->SAT"):
        # attention output
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.attention.dense.weight"] = \
            mega_weight[f"decoder.layers.{i}.self_attention.linear_proj.weight"].clone()
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.attention.dense.bias"] = \
            mega_weight[f"decoder.layers.{i}.self_attention.linear_proj.bias"].clone()

        # QKV
        qkv_weight = mega_weight[f"decoder.layers.{i}.self_attention.linear_qkv.weight"].clone()
        qkv_bias = mega_weight[f"decoder.layers.{i}.self_attention.linear_qkv.bias"].clone()

        # Reshape QKV from Megatron format into SAT format
        # qkv_weight: [3*hidden_size, hidden_size] -> [num_heads, 3, hidden_size_per_head, hidden_size] -> ...
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.attention.query_key_value.weight"] = \
            qkv_weight.view(num_heads, 3, hidden_size_per_head, hidden_size) \
                .permute(1, 0, 2, 3) \
                .reshape(3 * hidden_size, hidden_size).clone()
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.attention.query_key_value.bias"] = \
            qkv_bias.view(num_heads, 3, hidden_size_per_head) \
                .permute(1, 0, 2) \
                .reshape(3 * hidden_size) \
                .clone()

        # MLP
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h.weight"] = \
            mega_weight[f"decoder.layers.{i}.mlp.linear_fc1.weight"].clone()
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.mlp.dense_h_to_4h.bias"] = \
            mega_weight[f"decoder.layers.{i}.mlp.linear_fc1.bias"].clone()

        sat_weight[f"model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h.weight"] = \
            mega_weight[f"decoder.layers.{i}.mlp.linear_fc2.weight"].clone()
        sat_weight[f"model.diffusion_model.transformer.layers.{i}.mlp.dense_4h_to_h.bias"] = \
            mega_weight[f"decoder.layers.{i}.mlp.linear_fc2.bias"].clone()

        # AdaLN
        adaln_weight = mega_weight[f"decoder.layers.{i}.adaln.weight"].clone()
        adaln_bias = mega_weight[f"decoder.layers.{i}.adaln.bias"].clone()

        sat_weight[f"model.diffusion_model.mixins.adaln.adaln_modules.{i}.1.weight"] = adaln_weight.clone()
        sat_weight[f"model.diffusion_model.mixins.adaln.adaln_modules.{i}.1.bias"] = adaln_bias.clone()

    # --- final layers ---
    sat_weight["model.diffusion_model.mixins.final_layer.adaln.1.weight"] = \
        mega_weight["adaln_final.weight"].clone()
    sat_weight["model.diffusion_model.mixins.final_layer.adaln.1.bias"] = \
        mega_weight["adaln_final.bias"].clone()
    sat_weight["model.diffusion_model.mixins.final_layer.linear.weight"] = \
        mega_weight["output_projector.weight"].clone()
    sat_weight["model.diffusion_model.mixins.final_layer.linear.bias"] = \
        mega_weight["output_projector.bias"].clone()

    return OrderedDict(sat_weight)


def load_state_dict_megatron_and_convert_to_sat(megatron_ckpt_path, num_layers, num_heads, hidden_size):
    """
    Load a Megatron checkpoint from <megatron_ckpt_path>, then convert it into
    an SAT-style OrderedDict for direct QKV comparison.

    Typically, <megatron_ckpt_path> = ".../iter_0287500/mp_rank_00/model_optim_rng.pt"
    """
    ckpt = torch.load(megatron_ckpt_path, map_location="cuda")
    # Convert to SAT
    sat_like_weight = create_sat_state_dict_from_megatron(
        ckpt, num_layers=num_layers, num_heads=num_heads, hidden_size=hidden_size
    )
    return sat_like_weight

def compute_l2_difference(tensor1, tensor2):
    """Compute L2 norm of the difference between two tensors."""
    return torch.norm(tensor1 - tensor2, p=2).item()


def compare_qkv(qkv1, qkv2, name1="Model1", name2="Model2", atol=1e-6):
    """
    Compare QKV from two different sources (each is a tuple of (weight, bias)).
    Returns (weight_match, bias_match, weight_l2, bias_l2).
    """
    w1, b1 = qkv1
    w2, b2 = qkv2

    weight_match = torch.allclose(w1, w2, atol=atol)
    bias_match = torch.allclose(b1, b2, atol=atol)
    weight_l2_diff = compute_l2_difference(w1, w2)
    bias_l2_diff = compute_l2_difference(b1, b2)

    if not (weight_match and bias_match):
        print(f"[QKV Mismatch] {name1} vs {name2}")
        print(f"  Weight L2: {weight_l2_diff:.6f}, Bias L2: {bias_l2_diff:.6f}")
    else:
        # If everything matches well:
        print(f"[QKV Match] {name1} vs {name2}  (Weight L2={weight_l2_diff:.6f}, Bias L2={bias_l2_diff:.6f})")

    return weight_match, bias_match, weight_l2_diff, bias_l2_diff

if __name__ == "__main__":
    num_layers = 28
    num_heads = 32
    hidden_dim = 4096
    head_dim = hidden_dim // num_heads

    sat_ckpt_path = "/share/home/zyx/Models/Megatron-VLM/examples/dit/ckpts/pt_sat/0287500/mp_rank_00_model_states.pt"
    sat_state_dict = load_state_dict_sat(sat_ckpt_path)

    cogview_path = "/share/zyx/CogView4-6B-0128/transformer"  # directory containing model index for diffusers
    cogview_state_dict = load_state_dict_cogview(cogview_path)

    megatron_ckpt_path = "/share/home/zyx/Models/Megatron-VLM/examples/dit/ckpts/pt_ema/iter_0287500/mp_rank_00/model_optim_rng.pt"
    mega_as_sat_state_dict = load_state_dict_megatron_and_convert_to_sat(
        megatron_ckpt_path,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_size=hidden_dim
    )

    print("\n==== Start QKV Comparison ====\n")
    for layer_idx in range(num_layers):
        print(f"--- Layer {layer_idx} ---")

        # Extract QKV from SAT
        sat_qkv = extract_qkv_from_sat(sat_state_dict, layer_idx)

        # Extract QKV from CogView
        cogview_qkv = extract_qkv_from_cogview(
            cogview_state_dict, layer_idx, num_heads, head_dim, hidden_dim
        )

        # Extract QKV from Megatron->SAT
        mega_qkv = extract_qkv_from_sat(mega_as_sat_state_dict, layer_idx)

        # Compare: SAT vs CogView
        compare_qkv(sat_qkv, cogview_qkv, name1="SAT", name2="CogView4")

        # Compare: SAT vs Megatron
        compare_qkv(sat_qkv, mega_qkv, name1="SAT", name2="Megatron")

        # Compare: CogView vs Megatron (optional)
        compare_qkv(cogview_qkv, mega_qkv, name1="CogView4", name2="Megatron")

        print()

    print("=== Done ===")
