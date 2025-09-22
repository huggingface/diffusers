import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

def compute_query_redundancy_cosine_lowmem(
    attn_2d: torch.Tensor,
    row_chunk: int = 1024,
    max_keys: int | None = None,
    topk_per_row: int | None = None,
    proj_dim: int | None = None,
    method: str = "cosine",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    返回 [Tq, Tq] 的冗余矩阵，使用 X_norm @ X_norm.T 的方式避免 [Tq,Tq,Tk] 的广播内存爆炸。
    - max_keys: 先截断列维 Tk（例如 2048）
    - topk_per_row: 每个 query 只保留注意力最大的前 k 个 key，其余置零
    - proj_dim: 用随机投影把列维从 Tk 降到 proj_dim（如 256/512）
    - row_chunk: 分块计算相似度，限制峰值内存
    """
    x = attn_2d.to(device=device, dtype=dtype, copy=False)

    # 1) 可选：列维降维/裁剪
    Tq, Tk = x.shape
    if max_keys is not None and Tk > max_keys:
        x = x[:, :max_keys]
        Tk = max_keys

    if topk_per_row is not None and topk_per_row < Tk:
        # 稀疏化列：每行保留 top-k
        vals, idxs = torch.topk(x, k=topk_per_row, dim=1)
        x_sparse = torch.zeros_like(x)
        x_sparse.scatter_(1, idxs, vals)
        x = x_sparse
        del x_sparse, vals, idxs

    if proj_dim is not None and proj_dim < Tk:
        # 随机高斯投影（Johnson–Lindenstrauss），把列维降到 proj_dim
        # 为保证可复现可设固定 seed
        with torch.no_grad():
            rand_proj = torch.randn(Tk, proj_dim, device=device, dtype=dtype) / (proj_dim ** 0.5)
            x = x @ rand_proj
        Tk = proj_dim

    # 2) 行归一化
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)

    # 3) 分块矩阵乘法：X_norm @ X_norm.T
    Tq = x.shape[0]
    out = torch.empty((Tq, Tq), dtype=dtype, device=device)
    for i in range(0, Tq, row_chunk):
        i_end = min(i + row_chunk, Tq)
        xi = x[i:i_end]                       # [chunk, Tk]
        # 直接乘全体行；如果内存仍吃紧，可双重分块再分列块
        out[i:i_end] = xi @ x.T               # [chunk, Tq]
        del xi
        torch.cuda.empty_cache() if device.startswith("cuda") else None

    return out

def save_redundancy_heatmap_lowmem(attn_2d: torch.Tensor, save_path: str, title: str = None):
    # 防止超大输入
    Tq, Tk = attn_2d.shape
    # 若 Tq 超大，行采样到最多 Nq（例如 1024）
    Nq = 1024
    if Tq > Nq:
        idx = torch.linspace(0, Tq - 1, steps=Nq).long()
        attn_2d = attn_2d[idx]

    # 用低内存余弦相似度
    R = compute_query_redundancy_cosine_lowmem(
        attn_2d,
        row_chunk=256,        # 可根据内存调小
        max_keys=2048,        # 限制列维
        topk_per_row=256,     # 每行只保留 top-256 的 key
        proj_dim=None,        # 或者用 proj_dim=256 做随机投影
        device="cpu",
        dtype=torch.float32,
        method="cosine",
    ).clamp_(0, 1).cpu()

    save_redundancy_heatmap(R, save_path, title=title, method="cosine")
    del R

def save_redundancy_heatmap(redundancy_matrix: torch.Tensor, save_path: str, 
                           title: str = None, method: str = "cosine"):
    """保存冗余度热力图"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 设置白色背景
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # 使用蓝色系颜色映射：从白色到深蓝
    colors = ['white', 'lightblue', 'skyblue', 'steelblue', 'blue', 'darkblue', 'navy']
    n_bins = 256
    blue_cmap = LinearSegmentedColormap.from_list('blue_gradient', colors, N=n_bins)
    
    # 绘制热力图
    im = ax.imshow(redundancy_matrix.detach().cpu().numpy(), 
                   cmap=blue_cmap, 
                   aspect='auto',
                   vmin=0,  # 最小值设为0（白色）
                   vmax=1)  # 最大值设为1（深蓝）
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label=f"Query Redundancy ({method})")
    cbar.ax.set_facecolor('white')
    
    # 设置坐标轴
    ax.set_xlabel("Query Token Index", fontsize=12)
    ax.set_ylabel("Query Token Index", fontsize=12)
    ax.invert_yaxis()  # 反转Y轴，使得第0行在顶部
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # 设置网格线
    # ax.grid(True, alpha=0.3, color='lightgray')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()

def analyze_query_redundancy(cap, out_dir: str, batch_index: int = 0, head_index: int = 0,
                           methods: list = ["cosine", "pearson"], 
                           aggregate_method: str = "single"):
    """
    分析并保存query token的冗余度
    
    Args:
        cap: 捕获的注意力数据
        out_dir: 输出目录
        batch_index: batch索引
        head_index: head索引
        methods: 冗余度计算方法列表
        aggregate_method: 注意力聚合方法
    """
    A = cap["attn"]  # [B*H, Tq, Tk]
    H = cap["num_heads"] or A.shape[0]
    b = batch_index or 0
    
    # 获取注意力矩阵
    if aggregate_method == "single":
        idx = b * H + head_index if (cap["num_heads"] is not None and cap["batch_size"] is not None) else head_index
        attn_2d = A[idx]  # [Tq, Tk]
        suffix = f"_head{head_index:02d}"
    else:
        if cap["num_heads"] is not None and cap["batch_size"] is not None:
            batch_start = b * H
            batch_end = (b + 1) * H
            heads_attn = A[batch_start:batch_end]  # [H, Tq, Tk]
        else:
            heads_attn = A  # [H, Tq, Tk]
        
        if aggregate_method == "average":
            attn_2d = heads_attn.mean(dim=0)  # [Tq, Tk]
        elif aggregate_method == "max":
            attn_2d = heads_attn.max(dim=0)[0]  # [Tq, Tk]
        elif aggregate_method == "sum":
            attn_2d = heads_attn.sum(dim=0)  # [Tq, Tk]
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")
        
        suffix = f"_{aggregate_method}"

    step = cap["step_index"]
    ts = cap["timestep"]
    tstr = f"t{int(ts)}" if ts is not None else (f"s{step}" if step is not None else "sNA")

    # 为每种方法计算冗余度
    for method in methods:
        print(f"Computing {method} redundancy for layer {cap['layer_id']}...")
        
        # 计算冗余度矩阵
        redundancy_matrix = compute_query_redundancy(attn_2d, method=method)
        
        # 保存冗余度热力图
        subdir = os.path.join(out_dir, f"{tstr}", f"layer{cap['layer_id']:03d}")
        fname = f"{cap['layer_name'].replace('.', '_')}{suffix}_redundancy_{method}.png"
        
        save_redundancy_heatmap(
            redundancy_matrix,
            os.path.join(subdir, fname),
            title=f"{cap['layer_name']} {tstr} Query Redundancy ({method})",
            method=method
        )
        
        print(f"Saved redundancy heatmap: {fname}")
        
        # 打印统计信息
        print(f"Redundancy statistics ({method}):")
        print(f"  Mean: {redundancy_matrix.mean().item():.4f}")
        print(f"  Std: {redundancy_matrix.std().item():.4f}")
        print(f"  Max: {redundancy_matrix.max().item():.4f}")
        print(f"  Min: {redundancy_matrix.min().item():.4f}")
        
        # 找出最冗余的query对
        # 排除对角线（自己与自己的相似度）
        mask = torch.eye(redundancy_matrix.shape[0], dtype=torch.bool)
        off_diagonal = redundancy_matrix.masked_select(~mask)
        
        if len(off_diagonal) > 0:
            max_redundancy = off_diagonal.max().item()
            print(f"  Max off-diagonal redundancy: {max_redundancy:.4f}")
            
            # 找出最冗余的query对
            max_indices = torch.nonzero(redundancy_matrix == max_redundancy, as_tuple=False)
            if len(max_indices) > 0:
                q1, q2 = max_indices[0]
                print(f"  Most redundant query pair: {q1.item()} <-> {q2.item()}")

