import re

import torch


_QKV_ANCHORS_NUNCHAKU = ("to_qkv", "add_qkv_proj")
_ALLOWED_SUFFIXES_NUNCHAKU = {
    "bias",
    "lora_down",
    "lora_up",
    "qweight",
    "smooth_factor",
    "smooth_factor_orig",
    "wscales",
}

_QKV_NUNCHAKU_REGEX = re.compile(
    rf"^(?P<prefix>.*)\.(?:{'|'.join(map(re.escape, _QKV_ANCHORS_NUNCHAKU))})\.(?P<suffix>.+)$"
)


def _pick_split_dim(t: torch.Tensor, suffix: str) -> int:
    """
    Choose which dimension to split by 3. Heuristics:
      - 1D -> dim 0
      - 2D -> prefer dim=1 for 'qweight' (common layout [*, 3*out_features]),
              otherwise prefer dim=0 (common layout [3*out_features, *]).
      - If preferred dim isn't divisible by 3, try the other; else error.
    """
    shape = list(t.shape)
    if len(shape) == 0:
        raise ValueError("Cannot split a scalar into Q/K/V.")

    if len(shape) == 1:
        dim = 0
        if shape[dim] % 3 == 0:
            return dim
        raise ValueError(f"1D tensor of length {shape[0]} not divisible by 3.")

    # len(shape) >= 2
    preferred = 1 if suffix == "qweight" else 0
    other = 0 if preferred == 1 else 1

    if shape[preferred] % 3 == 0:
        return preferred
    if shape[other] % 3 == 0:
        return other

    # Fall back: any dim divisible by 3
    for d, s in enumerate(shape):
        if s % 3 == 0:
            return d

    raise ValueError(f"None of the dims {shape} are divisible by 3 for suffix '{suffix}'.")


def _split_qkv(t: torch.Tensor, dim: int):
    return torch.tensor_split(t, 3, dim=dim)


def _unpack_qkv_state_dict(
    state_dict: dict, anchors=_QKV_ANCHORS_NUNCHAKU, allowed_suffixes=_ALLOWED_SUFFIXES_NUNCHAKU
):
    """
    Convert fused QKV entries (e.g., '...to_qkv.bias', '...qkv_proj.wscales') into separate Q/K/V entries:
        '...to_q.bias', '...to_k.bias', '...to_v.bias' '...to_q.wscales', '...to_k.wscales', '...to_v.wscales'
    Returns a NEW dict; original is not modified.

    Only keys with suffix in `allowed_suffixes` are processed. Keys with non-divisible-by-3 tensors raise a ValueError.
    """
    anchors = tuple(anchors)
    allowed_suffixes = set(allowed_suffixes)

    new_sd: dict = {}
    for k, v in state_dict.items():
        m = _QKV_NUNCHAKU_REGEX.match(k)
        if m:
            suffix = m.group("suffix")
            if suffix not in allowed_suffixes:
                # keep as-is if it's not one of the targeted suffixes
                new_sd[k] = v
                continue

            prefix = m.group("prefix")  # everything before .to_qkv/.qkv_proj
            # Decide split axis
            split_dim = _pick_split_dim(v, suffix)
            q, k_, vv = _split_qkv(v, dim=split_dim)

            # Build new keys
            base_q = f"{prefix}.to_q.{suffix}"
            base_k = f"{prefix}.to_k.{suffix}"
            base_v = f"{prefix}.to_v.{suffix}"

            # Write into result dict
            new_sd[base_q] = q
            new_sd[base_k] = k_
            new_sd[base_v] = vv
        else:
            # not a fused qkv key
            new_sd[k] = v

    return new_sd
