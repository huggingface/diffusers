
import sys
import os
sys.path.append('/home/lyc/diffusers/src')
# import torch
import logging

# 先配置 logging（在导入 diffusers 前）
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

output_path = '/home/lyc/diffusers_output/'
save_dir = '/home/lyc/diffusers_output/attn_maps'

# 文件 handler
file_handler = logging.FileHandler(output_path + 'cogvideox.log', mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
import torch
import os, re, math, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from diffusers.hooks import ModelHook, HookRegistry
from diffusers.maputils import compute_query_redundancy_cosine_lowmem
from diffusers.maputils import save_redundancy_heatmap_lowmem
from matplotlib.colors import LinearSegmentedColormap
import torch.nn.functional as F
from diffusers.pipelines import CogVideoXPipeline

def save_attention_heatmap(attn_2d: torch.Tensor, save_path: str, title: str = None, 
                          xlabel: str = "Query Tokens", ylabel: str = "Key Tokens"):
    """保存注意力热力图，白色背景，QK依赖越强越红"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 设置白色背景
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('white')
    
    # 使用红色系颜色映射：从白色/淡粉红到深红
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['white', 'mistyrose', 'lightpink', 'pink', 'hotpink', 'red', 'darkred']
    # n_bins = 256
    # red_cmap = LinearSegmentedColormap.from_list('red_gradient', colors, N=n_bins)

    # 自定义红色渐变：从纯白到深红
    colors = [
        (1.0, 1.0, 1.0),      # 纯白色 (RGB)
        (1.0, 0.9, 0.9),      # 极淡粉红
        (1.0, 0.8, 0.8),      # 淡粉红
        (1.0, 0.6, 0.6),      # 粉红
        (1.0, 0.4, 0.4),      # 中粉红
        (1.0, 0.2, 0.2),      # 红色
        (0.8, 0.1, 0.1),      # 深红
        (0.6, 0.0, 0.0)       # 极深红
    ]

    red_cmap = LinearSegmentedColormap.from_list('custom_red', colors, N=256)

    
    # 绘制热力图
    im = ax.imshow(attn_2d.detach().cpu().numpy(), 
                   cmap=red_cmap, 
                   aspect='auto',
                   vmin=0,  # 最小值设为0（白色）
                   vmax=0.0006)  # 最大值设为1（深红）
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, label="Attention Score")
    cbar.ax.set_facecolor('white')
    
    # 设置坐标轴
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.invert_yaxis()  # 反转Y轴，使得第0行在顶部
    
    if title:
        ax.set_title(title, fontsize=14, pad=20)
    
    # 设置网格线（可选，让图更清晰）
    # ax.grid(True, alpha=0.3, color='lightgray')
    
    # 保存图片
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved attention heatmap to {save_path}")

def save_cap_head(cap, out_dir: str, batch_index: int, head_index: int, 
                  aggregate_method: str = "single"):
    """保存注意力图，支持多种聚合方式"""
    A = cap["attn"]  # [B*H, Tq, Tk]
    H = cap["num_heads"] or A.shape[0]
    b = batch_index or 0
    
    if aggregate_method == "single":
        # 单个head
        idx = b * H + head_index if (cap["num_heads"] is not None and cap["batch_size"] is not None) else head_index
        attn_2d = A[idx]  # [Tq, Tk]
        suffix = f"_head{head_index:02d}"
    else:
        # 聚合所有head
        if cap["num_heads"] is not None and cap["batch_size"] is not None:
            # 有明确的batch和head信息
            batch_start = b * H
            batch_end = (b + 1) * H
            heads_attn = A[batch_start:batch_end]  # [H, Tq, Tk]
        else:
            # 退化情况：假设所有都是head
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

    # 目录/文件名：step/layer/head
    subdir = os.path.join(out_dir, f"{tstr}", f"layer{cap['layer_id']:03d}")
    fname = f"{cap['layer_name'].replace('.', '_')}{suffix}.png"
    
    save_attention_heatmap(
        attn_2d, 
        os.path.join(subdir, fname),
        title=f"{cap['layer_name']} {tstr} {aggregate_method}",
        xlabel="Query Tokens",
        ylabel="Key Tokens"
    )

def _to_scalar_timestep(ts):
    if ts is None:
        return None
    try:
        import numpy as np
    except Exception:
        np = None

    if torch.is_tensor(ts):
        if ts.numel() == 0:
            return None
        return ts.reshape(-1)[0].item()
    if isinstance(ts, (list, tuple)):
        return _to_scalar_timestep(ts[0]) if ts else None
    if np is not None and isinstance(ts, np.ndarray):
        return ts.reshape(-1)[0].item() if ts.size > 0 else None
    return int(ts)

class AttnCaptureHook(ModelHook):
    def __init__(
        self,
        shared_state: dict,
        target_layers: list[int] = None,
        target_heads: list[int] = None,
        target_steps: list[int] = None,  # 目标步数
        store_limit_per_layer: int = 1,
        eps: float = 1e-3,
        force_square: bool = True,
        max_sequence_length: int = 51200,
        process_immediately: bool = True,  # 是否立即处理
        attn_qk_map = False,          # 计算QK的attention score map
        redundancy_q_map = False,   # 计算query之间的冗余度
        output_dir: str = None,  # 输出目录
    ):
        super().__init__()
        self.state = shared_state
        self.target_layers = set(target_layers) if target_layers else None
        self.target_heads = set(target_heads) if target_heads else None
        self.target_steps = set(target_steps) if target_steps else None
        self.store_limit_per_layer = store_limit_per_layer
        self.eps = eps
        self.force_square = force_square
        self.max_sequence_length = max_sequence_length
        self.process_immediately = process_immediately
        self.attn_qk_map = attn_qk_map
        self.redundancy_q_map = redundancy_q_map
        self.output_dir = output_dir
        self.captured = []
        self._layer_store_count: dict[int, int] = {}

    def _layer_allowed(self, layer_id: int) -> bool:
        if self.target_layers is None:
            return True
        return layer_id in self.target_layers

    def _step_allowed(self, step_index: int | None) -> bool:
        if self.target_steps is None:
            return True
        return (step_index is not None) and (step_index in self.target_steps)

    def pre_forward(self, module, *args, **kwargs):
        ts = kwargs.get("timestep", None)
        ts_val = _to_scalar_timestep(ts)

        has_proj = all(hasattr(module, a) for a in ["to_q", "to_k", "to_v"])
        if not has_proj:
            self._can_capture = False
            return args, kwargs

        fqn = getattr(module, "_diffusers_fqn", module.__class__.__name__)
        layer_id = getattr(module, "_attn_layer_id", -1)
        
        if not self._layer_allowed(layer_id):
            self._can_capture = False
            return args, kwargs

        step_index = self.state.get("step_index", None)
        if not self._step_allowed(step_index):
            self._can_capture = False
            return args, kwargs

        if self._layer_store_count.get(layer_id, 0) >= self.store_limit_per_layer:
            self._can_capture = False
            return args, kwargs

        hidden_states = kwargs.get("hidden_states", None)
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)
        
        if hidden_states is not None:
            q_source = hidden_states
            k_source = hidden_states
            attention_type = "self"
        elif encoder_hidden_states is not None:
            q_source = encoder_hidden_states
            k_source = encoder_hidden_states
            attention_type = "self_encoder"
        else:
            self._can_capture = False
            return args, kwargs

        if q_source.shape[1] > self.max_sequence_length:
            print(f"Layer {layer_id}: sequence too long ({q_source.shape[1]}), skipping")
            self._can_capture = False
            return args, kwargs

        print(f"Capturing Layer {layer_id} at step {step_index}")

        q = module.to_q(q_source)
        k = module.to_k(k_source)

        if hasattr(module, "head_to_batch_dim"):
            q = module.head_to_batch_dim(q)
            k = module.head_to_batch_dim(k)
            num_heads = getattr(module, "heads", None)
            batch_size = None
        else:
            b, tq, _ = q.shape
            num_heads = getattr(module, "heads", None)
            if num_heads is None:
                self._can_capture = False
                return args, kwargs
            d = q.shape[-1] // num_heads
            q = q.view(b, tq, num_heads, d).permute(0, 2, 1, 3).reshape(b * num_heads, tq, d)
            b2, tk, _ = k.shape
            k = k.view(b2, tk, num_heads, d).permute(0, 2, 1, 3).reshape(b2 * num_heads, tk, d)
            batch_size = b

        self._can_capture = True
        self._ctx = {
            "fqn": fqn,
            "layer_id": layer_id,
            "num_heads": num_heads,
            "batch_size": batch_size,
            "q": q, "k": k,
            "step_index": step_index,
            "timestep": ts_val,
        }
        return args, kwargs

    def post_forward(self, module, output):
        if not getattr(self, "_can_capture", False):
            return output

        q, k = self._ctx["q"], self._ctx["k"]
        scale = getattr(module, "scale", None)
        if scale is None:
            scale = 1.0 / math.sqrt(q.shape[-1])
        
        if self.force_square and q.shape[1] != k.shape[1]:
            min_len = min(q.shape[1], k.shape[1])
            q = q[:, :min_len, :]
            k = k[:, :min_len, :]

        attn = torch.einsum("bqd,bkd->bqk", q, k).mul_(scale).softmax(dim=-1).detach().float().cpu()

        layer_id = self._ctx["layer_id"]
        step_index = self._ctx["step_index"]
        
        print(f"Processing Layer {layer_id} at step {step_index}")

        # 立即处理并保存
        if self.process_immediately:
            self._process_and_save_immediately(attn, layer_id, step_index)
        else:
            # 传统方式：存储到内存
            self._layer_store_count[layer_id] = self._layer_store_count.get(layer_id, 0) + 1
            self.captured.append({
                "layer_id": layer_id,
                "layer_name": self._ctx["fqn"],
                "step_index": step_index,
                "timestep": self._ctx["timestep"],
                "num_heads": self._ctx["num_heads"],
                "batch_size": self._ctx["batch_size"],
                "attn": attn,
            })

        return output

    def _process_and_save_immediately(self, attn: torch.Tensor, layer_id: int, step_index: int):
        """立即处理并保存注意力数据，然后丢弃"""
        try:
            H = self._ctx["num_heads"] or attn.shape[0]
            if self._ctx["num_heads"] is not None and self._ctx["batch_size"] is not None:
                # 有明确的batch和head信息
                batch_size = self._ctx["batch_size"]
                heads_attn = attn.view(batch_size, H, attn.shape[1], attn.shape[2])
                avg_attn = heads_attn.mean(dim=1)  # [B, Tq, Tk]
                attn_2d = avg_attn[0]  # 取第一个batch
            else:
                # 退化情况：假设所有都是head
                heads_attn = attn  # [H, Tq, Tk]
                attn_2d = heads_attn.mean(dim=0)  # [Tq, Tk]
            # 计算平均注意力（所有head的平均）
            if self.attn_qk_map:
                H = self._ctx["num_heads"] or attn.shape[0]
                # 保存注意力热力图
                tstr = f"s{step_index}"
                subdir = os.path.join(self.output_dir, f"{tstr}", f"layer{layer_id:03d}")
                fname = f"{self._ctx['fqn'].replace('.', '_')}_average.png"
                save_attention_heatmap(
                    attn_2d,
                    os.path.join(subdir, fname),
                    title=f"Layer {layer_id} Step {step_index} Average Attention",
                    xlabel="Query Tokens",
                    ylabel="Key Tokens"
                )

            # 计算并保存冗余度热力图
            if self.redundancy_q_map:
                # 计算并保存冗余度热力图（低内存）
                tstr = f"s{step_index}"
                subdir = os.path.join(self.output_dir, f"{tstr}", f"layer{layer_id:03d}")
                redundancy_fname = f"{self._ctx['fqn'].replace('.', '_')}_redundancy_cosine.png"
                save_redundancy_heatmap_lowmem(
                    attn_2d,
                    os.path.join(subdir, redundancy_fname),
                    title=f"Layer {layer_id} Step {step_index} Query Redundancy",
                )
                print(f"Saved attention maps for Layer {layer_id} at step {step_index}")
                
                # 打印统计信息
                print(f"  Attention shape: {attn_2d.shape}")
                print(f"  Mean attention: {attn_2d.mean().item():.4f}")
                # print(f"  Mean redundancy: {redundancy_matrix.mean().item():.4f}")
            
        except Exception as e:
            print(f"Error processing layer {layer_id}: {e}")
        finally:
            # 清理内存
            del attn
            if hasattr(self, '_ctx'):
                del self._ctx
            torch.cuda.empty_cache()  # 清理GPU缓存

class TransformerStepHook(ModelHook):
    def __init__(self, shared_state: dict):
        super().__init__()
        self.state = shared_state

    def pre_forward(self, module, *args, **kwargs):
        ts = kwargs.get("timestep", None)
        ts_val = _to_scalar_timestep(ts)
        if ts_val is not None:
            last = self.state.get("last_timestep", None)
            if last != ts_val:
                self.state["step_index"] = self.state.get("step_index", -1) + 1
                self.state["last_timestep"] = ts_val
            self.state["timestep"] = ts_val
        return args, kwargs
    
def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _save_heatmap(arr_2d: torch.Tensor, save_path: str, title: str = None,
                  cmap="magma", vmin=None, vmax=None, xlabel=None, ylabel=None,
                  invert_y=True, white_bg=True):
    _ensure_dir(save_path)
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white" if white_bg else None)
    if white_bg:
        ax.set_facecolor("white")
    im = ax.imshow(arr_2d.detach().cpu().numpy(), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title: ax.set_title(title)
    if invert_y: ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white" if white_bg else None)
    plt.close()

def _cosine_sim_lowmem(x: torch.Tensor, row_chunk: int = 1024, device="cpu", dtype=torch.float32):
    # x: [N, D] -> return [N, N] with X̂ X̂ᵀ, 分块避免内存峰值
    x = x.to(device=device, dtype=dtype, copy=False)
    x = x / (x.norm(dim=1, keepdim=True) + 1e-8)
    N, D = x.shape
    out = torch.empty((N, N), dtype=dtype, device=device)
    for i in range(0, N, row_chunk):
        i_end = min(i + row_chunk, N)
        out[i:i_end] = x[i:i_end] @ x.T
    return out

def _entropy_sparsity_from_sim(sim_row: torch.Tensor, temperature: float = 1.0):
    # sim_row: [N], -> p=softmax(sim/τ), 稀疏度 = 1 - H_norm
    p = F.softmax(sim_row / max(1e-6, temperature), dim=0)
    logp = torch.log(p + 1e-12)
    H = -(p * logp).sum()
    H_norm = H / math.log(p.shape[0] + 1e-12)
    return 1.0 - H_norm  # 高→更稀疏
    
class LatentFrameVizHook(ModelHook):
    """
    在 CogVideoX3DTransformer 入口（hidden_states）上做“帧级”可视化：
    - Full Attention（query->所有key 的相似度）: 取某个 query 的一行，重排为 H×W。
    - Query Sparsity map：每个 query 的稀疏度，重排为 H×W。
    计算结束立即落盘并释放内存。
    """
    def __init__(
        self,
        save_root: str,
        target_steps: list[int] = None,   # 只在这些 step_index 上可视化
        target_frames: list[int] = None,  # 只在这些帧索引上可视化
        query_indices: list[int] = None,  # 在 Full Attention 图中要展示的若干 query（在该帧的网格索引）
        max_hw_tokens: int = 4096,        # 控制 HxW 最大 tokens；过大则等距下采样
        row_chunk: int = 1024,            # 相似度分块
        cosine_device: str = "cpu",       # 相似度计算放 CPU，避免占用显存
        cosine_dtype = torch.float32,
        temperature: float = 1.0,         # 熵稀疏度的温度
        decode_latents = False,           # 是否解码 latent
    ):
        super().__init__()
        self.save_root = save_root
        self.target_steps = set(target_steps) if target_steps else None
        self.target_frames = set(target_frames) if target_frames else None
        self.query_indices = query_indices or [0]  # 默认画一个 query
        self.max_hw_tokens = max_hw_tokens
        self.row_chunk = row_chunk
        self.cosine_device = "cuda" if (cosine_device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.cosine_dtype = cosine_dtype
        self.temperature = temperature
        self.decode_latents = decode_latents

    def pre_forward(self, module, *args, **kwargs):
        self._do = False
        step_index = getattr(module._diffusers_hook.hooks["step_hook"], "state", {}).get("step_index", None) \
                     if hasattr(module, "_diffusers_hook") and "step_hook" in getattr(module._diffusers_hook, "hooks", {}) \
                     else None
        if (self.target_steps is not None) and (step_index not in self.target_steps):
            return args, kwargs

        x = kwargs.get("hidden_states", None)
        if x is None or x.dim() < 4:
            return args, kwargs

        # x 可能是 [B, C, T, H, W] 或 [B, T, C, H, W]；做个鲁棒判定
        # 认为 C 是 64~2048 间的典型通道数；T 通常 <= 33；H,W 16~128
        shape = x.shape
        if x.dim() == 5:
            b, d1, d2, d3, d4 = shape
            # 判断哪个是 C
            candidates = [d1, d2]
            if 64 <= d1 <= 4096 and d2 <= 64:
                layout = "B C T H W"
            elif 64 <= d2 <= 4096 and d1 <= 64:
                layout = "B T C H W"
            else:
                # 回退：若 d1 > d2 认为 d1 是 C
                layout = "B C T H W" if d1 >= d2 else "B T C H W"
        else:
            # 其他情况暂不处理
            return args, kwargs

        self._ctx = {"layout": layout, "step_index": step_index}
        self._x_ref = x.detach().to("cpu")  # 转 CPU，避免卡显存
        self._do = True
        return args, kwargs

    def post_forward(self, module, output):
        if not getattr(self, "_do", False):
            return output
        try:
            x = self._x_ref  # [B,*,*,H,W]
            layout = self._ctx["layout"]
            step = self._ctx["step_index"]
            B = x.shape[0]

            # decode 一下
            # 改进，动态使用不同 pipeline 的 vae 的 decode_latent

                
            # 选择目标帧集合
            if layout == "B C T H W":
                T = x.shape[2]; C = x.shape[1]; H = x.shape[3]; W = x.shape[4]
                frame_take = list(self.target_frames or range(T))
                for t in frame_take:
                    xt = x[0, :, t]  # [C, H, W]
                    self._process_one_frame(xt, t, H, W, step)
            else:  # "B T C H W"
                # 注意，这里的 vae 已经压缩过了。。。
                T = x.shape[1]; C = x.shape[2]; H = x.shape[3]; W = x.shape[4]
                frame_take = list(self.target_frames or range(T))
                for t in frame_take:
                    xt = x[0, t]  # [C, H, W]
                    self._process_one_frame(xt, t, H, W, step)

        finally:
            del self._x_ref
            torch.cuda.empty_cache()
        return output

    def _process_one_frame(self, xt: torch.Tensor, t: int, H: int, W: int, step: int):
        # xt: [C, H, W] on CPU; 下采样到 <= max_hw_tokens
        C = xt.shape[0]
        h, w = H, W
        N = h * w
        stride = 1
        while (h // stride) * (w // stride) > self.max_hw_tokens:
            stride *= 2
        if stride > 1:
            xt_ds = F.avg_pool2d(xt.unsqueeze(0), kernel_size=stride, stride=stride).squeeze(0)  # [C, h', w']
        else:
            xt_ds = xt
        h2, w2 = xt_ds.shape[1], xt_ds.shape[2]
        X = xt_ds.permute(1, 2, 0).reshape(-1, C)  # [N2, C]

        # 相似度矩阵（低内存）
        S = _cosine_sim_lowmem(X, row_chunk=self.row_chunk, device=self.cosine_device, dtype=self.cosine_dtype)  # [N2,N2]

        # Query Sparsity map（熵稀疏度）
        sparsity = torch.empty(S.shape[0], dtype=torch.float32, device=S.device)
        for i in range(S.shape[0]):
            sparsity[i] = _entropy_sparsity_from_sim(S[i], temperature=self.temperature)
        sparsity_map = sparsity.reshape(h2, w2).cpu()

        # Full attention for selected queries（每个 query 的一行）
        q_indices = [q for q in self.query_indices if 0 <= q < S.shape[0]]

        # 保存
        root = os.path.join(self.save_root, f"s{step}", f"frame{t:03d}")
        _save_heatmap(
            sparsity_map, os.path.join(root, "query_sparsity.png"),
            title=f"Query Sparsity (frame {t}, step {step})",
            cmap=LinearSegmentedColormap.from_list("green", ["white","lightgreen","green","darkgreen"], N=256),
            vmin=0.0, vmax=0.0010, xlabel="X", ylabel="Y"
        )
        for qi in q_indices:
            attn_q = S[qi].reshape(h2, w2).cpu()
            colors = ['white', 'lightblue', 'skyblue', 'steelblue', 'blue', 'darkblue', 'navy']
            n_bins = 256
            blue_cmap = LinearSegmentedColormap.from_list('blue_gradient', colors, N=n_bins)
            _save_heatmap(
                attn_q, os.path.join(root, f"full_attention_q{qi:05d}.png"),
                title=f"Full Attention of q={qi} (frame {t}, step {step})",
                 cmap=blue_cmap,
                vmin=-1.0, vmax=1.0, xlabel="Key X", ylabel="Key Y"
            )

def assign_layer_ids_and_register(model, attn_hook: AttnCaptureHook, layer_name_patterns=None):
    """为每个注意力模块分配连续的 layer_id，并注册捕获 hook"""
    patterns = [re.compile(p) for p in (layer_name_patterns or [])]
    def allow(fqn: str):
        return True if not patterns else any(p.search(fqn) for p in patterns)

    layer_id = 0
    for fqn, m in model.named_modules():
        try:
            m._diffusers_fqn = fqn
        except Exception:
            pass
        if hasattr(m, "to_q") and hasattr(m, "to_k") and allow(fqn):
            setattr(m, "_attn_layer_id", layer_id)
            HookRegistry.check_if_exists_or_initialize(m).register_hook(attn_hook, name=f"attn_capture_{layer_id}")
            layer_id += 1
    return layer_id