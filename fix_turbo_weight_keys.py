"""
将 ERNIE-Image-Turbo/transformer 的权重键名修正为与 ERNIE-Image/transformer 一致。

差异均位于每层 self_attention 子模块，共 6 类 × 36 层 = 216 个键需要重命名：
  k_layernorm  -> norm_k
  q_layernorm  -> norm_q
  k_proj       -> to_k
  q_proj       -> to_q
  v_proj       -> to_v
  linear_proj  -> to_out.0
"""

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# ── 路径配置 ──────────────────────────────────────────────────────────────────
TURBO_DIR = Path("/root/paddlejob/gpfsspace/model_weights/turbo/ERNIE-Image-Turbo/transformer")
# 修正后的文件直接覆盖原目录（先备份），如需输出到新目录请修改此变量
OUTPUT_DIR = TURBO_DIR  # 或改为 Path("/your/output/path")
BACKUP_SUFFIX = ".bak"  # 原文件备份后缀，设为 None 则不备份

# ── 键名映射（只处理 self_attention 子键，前缀 layers.N. 由脚本动态拼接）───
KEY_REMAP = {
    "self_attention.k_layernorm.weight": "self_attention.norm_k.weight",
    "self_attention.q_layernorm.weight": "self_attention.norm_q.weight",
    "self_attention.k_proj.weight":      "self_attention.to_k.weight",
    "self_attention.q_proj.weight":      "self_attention.to_q.weight",
    "self_attention.v_proj.weight":      "self_attention.to_v.weight",
    "self_attention.linear_proj.weight": "self_attention.to_out.0.weight",
}

NUM_LAYERS = 36  # layers.0 ~ layers.35


def build_full_remap() -> dict[str, str]:
    """构建完整的旧键名 -> 新键名映射表（含层前缀）。"""
    remap = {}
    for layer_idx in range(NUM_LAYERS):
        prefix = f"layers.{layer_idx}."
        for old_suffix, new_suffix in KEY_REMAP.items():
            remap[prefix + old_suffix] = prefix + new_suffix
    return remap


def rename_keys_in_tensor_dict(
    tensors: dict[str, torch.Tensor],
    remap: dict[str, str],
) -> tuple[dict[str, torch.Tensor], int]:
    """重命名张量字典中的键，返回新字典和实际重命名的数量。"""
    renamed = 0
    new_tensors: dict[str, torch.Tensor] = {}
    for key, tensor in tensors.items():
        new_key = remap.get(key, key)
        if new_key != key:
            renamed += 1
        new_tensors[new_key] = tensor
    return new_tensors, renamed


def backup_file(path: Path) -> None:
    if BACKUP_SUFFIX is None:
        return
    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    shutil.copy2(path, backup)
    print(f"  [备份] {path.name} -> {backup.name}")


def process_safetensors_files(remap: dict[str, str]) -> None:
    index_path = TURBO_DIR / "diffusion_pytorch_model.safetensors.index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    # 找出所有需要处理的 shard 文件（去重）
    shard_files = sorted(set(index["weight_map"].values()))
    print(f"\n共发现 {len(shard_files)} 个 shard 文件，开始处理...\n")

    total_renamed = 0
    for shard_name in shard_files:
        shard_path = TURBO_DIR / shard_name
        print(f"[处理] {shard_name}")

        tensors = load_file(shard_path)
        new_tensors, renamed = rename_keys_in_tensor_dict(tensors, remap)
        total_renamed += renamed
        print(f"  本文件重命名: {renamed} 个键")

        if renamed > 0:
            # 保留原始 metadata（如果有）
            metadata = {}

            out_path = OUTPUT_DIR / shard_name
            if out_path == shard_path and BACKUP_SUFFIX:
                backup_file(shard_path)

            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            save_file(new_tensors, out_path, metadata=metadata)
            print(f"  [保存] {out_path}")
        else:
            if OUTPUT_DIR != TURBO_DIR:
                shutil.copy2(shard_path, OUTPUT_DIR / shard_name)
                print(f"  [复制（无变更）] {shard_name}")

    print(f"\n所有 shard 处理完毕，共重命名 {total_renamed} 个键。")

    # ── 更新 index.json 中的 weight_map ─────────────────────────────────────
    new_weight_map: dict[str, str] = {}
    for old_key, shard_name in index["weight_map"].items():
        new_key = remap.get(old_key, old_key)
        new_weight_map[new_key] = shard_name

    index["weight_map"] = new_weight_map

    out_index_path = OUTPUT_DIR / "diffusion_pytorch_model.safetensors.index.json"
    if out_index_path == index_path and BACKUP_SUFFIX:
        backup_file(index_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"[更新] index.json 已写入: {out_index_path}\n")


def verify_against_base() -> None:
    """（可选）验证修正后的 Turbo 键名与 Base 完全一致。"""
    BASE_DIR = Path("/root/paddlejob/gpfsspace/model_weights/base/ERNIE-Image/transformer")
    base_index_path = BASE_DIR / "diffusion_pytorch_model.safetensors.index.json"
    turbo_index_path = OUTPUT_DIR / "diffusion_pytorch_model.safetensors.index.json"

    if not base_index_path.exists() or not turbo_index_path.exists():
        print("[验证] 找不到 index.json，跳过验证。")
        return

    with open(base_index_path, "r") as f:
        base_keys = set(json.load(f)["weight_map"].keys())
    with open(turbo_index_path, "r") as f:
        turbo_keys = set(json.load(f)["weight_map"].keys())

    only_in_base  = base_keys - turbo_keys
    only_in_turbo = turbo_keys - base_keys

    if not only_in_base and only_in_turbo:
        print(f"[验证] 警告：Turbo 中多余的键 ({len(only_in_turbo)}):")
        for k in sorted(only_in_turbo):
            print(f"  + {k}")
    elif only_in_base:
        print(f"[验证] 警告：Base 中存在但 Turbo 中缺少的键 ({len(only_in_base)}):")
        for k in sorted(only_in_base):
            print(f"  - {k}")
    else:
        print("[验证] 通过！修正后 Turbo 的键名与 Base 完全一致。")


if __name__ == "__main__":
    remap = build_full_remap()
    print(f"键名映射表共 {len(remap)} 条（{NUM_LAYERS} 层 × {len(KEY_REMAP)} 类）")

    process_safetensors_files(remap)
    verify_against_base()
