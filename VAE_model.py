import torch
import torch.nn as nn
import torch.nn.functional as F

# === Projected LFQ Quantizer ===
class ProjectedLFQ(nn.Module):
    def __init__(self, in_channels, quant_channels=16, entropy_loss_weight=0.1):
        super().__init__()
        self.project = nn.Conv3d(in_channels, quant_channels, 1)
        self.entropy_loss_weight = entropy_loss_weight

    def forward(self, x):
        x_proj = self.project(x)
        quantized_x = torch.where(x_proj > 0, torch.ones_like(x_proj), -torch.ones_like(x_proj))
        indices = (quantized_x > 0).long()
        probs = indices.float().mean(dim=(0,2,3,4))
        entropy = - (probs * torch.log(probs.clamp(min=1e-8)) +
                     (1 - probs) * torch.log((1 - probs).clamp(min=1e-8)))
        entropy_loss = -entropy.mean() * self.entropy_loss_weight
        return quantized_x, indices, entropy_loss

# === Efficient Cross Attention (memory-safe) ===
class EfficientCrossAttention3D(nn.Module):
    def __init__(self, channels, mode="channel"):
        super().__init__()
        self.mode = mode
        self.q_proj = nn.Conv3d(channels, channels, 1)
        self.k_proj = nn.Conv3d(channels, channels, 1)
        self.v_proj = nn.Conv3d(channels, channels, 1)
        self.out_proj = nn.Conv3d(channels, channels, 1)

    def forward(self, q, kv):
        B, C, T, H, W = q.shape
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(kv)
        v_proj = self.v_proj(kv)
        if self.mode == "channel":
            q_ = q_proj.permute(0,2,3,4,1).reshape(-1, C).unsqueeze(1)
            k_ = k_proj.permute(0,2,3,4,1).reshape(-1, C).unsqueeze(2)
            attn_scores = torch.softmax(torch.bmm(q_, k_), dim=-1)
            v_ = v_proj.permute(0,2,3,4,1).reshape(-1, C).unsqueeze(1)
            out = (attn_scores * v_).reshape(B, T, H, W, C).permute(0,4,1,2,3)
        else:
            q_ = q_proj.permute(0,1,3,4,2).reshape(-1, T)
            k_ = k_proj.permute(0,1,3,4,2).reshape(-1, T)
            v_ = v_proj.permute(0,1,3,4,2).reshape(-1, T)
            attn_scores = torch.softmax(q_ * k_, dim=-1)
            out = (attn_scores * v_).reshape(B, C, H, W, T).permute(0,1,4,2,3)
        return self.out_proj(out) + q

# === FFT and Wavelet Branches ===
class FFTBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temporal_convs = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
        )
        self.conv3d = nn.Conv3d(out_ch, out_ch, 3, padding=1)
    def forward(self, x):
        B, C, T, H, W = x.shape
        x2d = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        x2d = self.conv2d(x2d)
        x2d_fft = torch.fft.fft2(x2d.float()).real
        x2d_fft = x2d_fft.reshape(B, T, -1, H, W).permute(0,2,1,3,4)
        x1d = x2d_fft.permute(0,3,4,1,2).reshape(B*H*W, -1, T)
        x1d = self.temporal_convs(x1d)
        x1d = x1d.reshape(B, H, W, -1, T).permute(0, 3, 4, 1, 2)
        x3d = torch.fft.ifft2(x1d.float()).real
        return x1d, self.conv3d(x3d)

class WaveletBranch(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv2d = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv1d = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.conv3d = nn.Conv3d(out_ch, out_ch, 3, padding=1)
    def forward(self, x):
        B, C, T, H, W = x.shape
        x2d = x.permute(0,2,1,3,4).reshape(B*T, C, H, W)
        x2d = self.conv2d(x2d)
        x2d = x2d.reshape(B, T, -1, H, W).permute(0,2,1,3,4)
        x1d = x2d.permute(0,3,4,1,2).reshape(B*H*W, -1, T)
        x1d = self.conv1d(x1d)
        x1d = x1d.reshape(B, H, W, -1, T).permute(0, 3, 4, 1, 2)
        x3d = x1d
        return x1d, self.conv3d(x3d)

# === VAE Blocks (Encoder/Decoder) ===
import torch
import torch.nn as nn

class LearnableTanh(nn.Module):
    """
    y = a * tanh(b * x), with a,b > 0 (enforced via softplus)
    """
    def __init__(self, a_init=1.0, b_init=1.0):
        super().__init__()
        self._a = nn.Parameter(torch.tensor(float(a_init)))
        self._b = nn.Parameter(torch.tensor(float(b_init)))
        self.softplus = nn.Softplus()

    def forward(self, x):
        a = self.softplus(self._a)
        b = self.softplus(self._b)
        return a * torch.tanh(b * x)

class VideoVAEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fft_branch = FFTBranch(in_ch, out_ch)
        self.wavelet_branch = WaveletBranch(in_ch, out_ch)

        # Used to get q,k,v from conv features for channel attn (as you had)
        self.conv3d_branch = nn.Conv3d(in_ch, out_ch, 3, padding=1)

        self.channel_cross_attn  = EfficientCrossAttention3D(out_ch, mode="channel")
        self.temporal_cross_attn = EfficientCrossAttention3D(out_ch, mode="temporal")

        # Per-branch Conv3D before fusion (matches the two Conv3D boxes in the figure)
        self.conv3d_fft = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.conv3d_wav = nn.Conv3d(out_ch, out_ch, 3, padding=1)

        # Post-fusion: Norm → Learnable tanh → MaxPool → Conv3D
        self.fuse_norm = nn.BatchNorm3d(out_ch)
        self.ltan = LearnableTanh()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.out_conv = nn.Conv3d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        # Branches
        fft_1d, fft_out = self.fft_branch(x)      # shapes: (B,C_out,T,H,W)
        wav_1d, wav_out = self.wavelet_branch(x)

        # Channel then temporal cross-attention path
        conv3d_out = self.conv3d_branch(x)
        chan_attn  = self.channel_cross_attn(conv3d_out, fft_1d)
        temp_attn  = self.temporal_cross_attn(chan_attn, wav_1d)

        # Conv3D on each frequency branch before fusion (as in figure)
        fft_feat = self.conv3d_fft(fft_out)
        wav_feat = self.conv3d_wav(wav_out)

        # ⊕ elementwise fusion of three paths
        fused = fft_feat + wav_feat + temp_attn

        # Learnable tanh → MaxPool → Conv3D (final)
        fused = self.fuse_norm(fused)
        fused = self.ltan(fused)       # Learnable tanh
        fused = self.pool(fused)       # MaxPool3d
        out   = self.out_conv(fused)   # Conv3D

        return out


class VideoVAEBlockTranspose(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.fft_branch = FFTBranch(in_ch, out_ch)
        self.wavelet_branch = WaveletBranch(in_ch, out_ch)
        self.conv3d_branch = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.channel_cross_attn = EfficientCrossAttention3D(out_ch, mode="channel")
        self.temporal_cross_attn = EfficientCrossAttention3D(out_ch, mode="temporal")
        self.post_conv = nn.Conv3d(out_ch * 3, out_ch, 1)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()
        self.upsample = nn.ConvTranspose3d(out_ch, out_ch, 4, 2, 1, output_padding=0)
    def forward(self, x):
        fft_out = self.fft_branch(x)
        wav_out = self.wavelet_branch(x)
        conv3d_out = self.conv3d_branch(x)
        spectral_kv = fft_out + wav_out
        chan_attn = self.channel_cross_attn(conv3d_out, spectral_kv)
        temp_attn = self.temporal_cross_attn(chan_attn, spectral_kv)
        merged = torch.cat([fft_out, wav_out, temp_attn], dim=1)
        out = self.post_conv(merged)
        out = self.norm(out)
        out = self.act(out)
        out = self.upsample(out)
        return out

class RVQFusionBlock(nn.Module):
    def __init__(self, prev_fused_dim, quant_emb_dim, out_ch):
        super().__init__()
        self.tconv3d = nn.ConvTranspose3d(quant_emb_dim, quant_emb_dim, 3, padding=1)
        self.conv3d = nn.Conv3d(prev_fused_dim + quant_emb_dim, out_ch, 3, padding=1)
        self.act = nn.GELU()
    def forward(self, x_prev, x_qi):
        x_qi_up = F.interpolate(x_qi, size=x_prev.shape[-3:], mode='trilinear', align_corners=False)
        x_qi_up = self.tconv3d(x_qi_up)
        x_cat = torch.cat([x_prev, x_qi_up], dim=1)
        out = self.act(self.conv3d(x_cat))
        return out

class VideoVAE(nn.Module):
    def __init__(self, in_channels=3, base_ch=64, quant_emb_dim=16, num_blocks=4):
        super().__init__()
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8]  # [64, 128, 256, 512]
        self.enc_blocks = nn.ModuleList()
        self.lfqs = nn.ModuleList()
        for i, ch in enumerate(chs):
            self.enc_blocks.append(VideoVAEBlock(in_channels if i == 0 else chs[i-1], ch))
            self.lfqs.append(ProjectedLFQ(ch, quant_emb_dim))
        self.rvq_blocks = nn.ModuleList()
        self.rvq_blocks.append(RVQFusionBlock(chs[0], quant_emb_dim, chs[1]))
        self.rvq_blocks.append(RVQFusionBlock(chs[1], quant_emb_dim, chs[2]))
        self.rvq_blocks.append(RVQFusionBlock(chs[2], quant_emb_dim, chs[3]))
        self.rvq_blocks.append(RVQFusionBlock(chs[3], quant_emb_dim, chs[3]))
        rev_chs = list(reversed(chs))
        self.dec_blocks = nn.ModuleList()
        # Use the deepest 2 decoder blocks (from largest channel to in_channels)
        for i in range(1):
            ch = rev_chs[i]
            out_ch = rev_chs[i+1] if i+1 < len(rev_chs) else in_channels
            self.dec_blocks.append(VideoVAEBlockTranspose(ch, out_ch))
        
        self.out_conv = nn.Conv3d(out_ch, in_channels, 1, 1)

    def forward(self, x):
        # Strict: only allow shapes divisible by 16 (N=4 stages)
        B, C, T, H, W = x.shape
        for d in (T, H, W):
            if d % 16 != 0:
                raise ValueError(f"Input T/H/W must be divisible by 16, got {(T,H,W)}")
        hs, quantizeds, indices, entropies = [], [], [], []
        h = x
        out_for_transformer = []
        for enc_blk, lfq in zip(self.enc_blocks, self.lfqs):
            h = enc_blk(h)
            hs.append(h)
            q, idx, ent_loss = lfq(h)
            quantizeds.append(q)
            indices.append(idx)
            entropies.append(ent_loss)
        fused = self.rvq_blocks[0](hs[-1], quantizeds[0])
        out_for_transformer.append(fused)
        fused = self.rvq_blocks[1](fused, quantizeds[1])
        out_for_transformer.append(fused)
        fused = self.rvq_blocks[2](fused, quantizeds[2])
        out_for_transformer.append(fused)
        fused = self.rvq_blocks[3](fused, quantizeds[3])
        out_for_transformer.append(fused)
        dec_in = fused
        for dec_blk in self.dec_blocks:
            dec_in = dec_blk(dec_in)
        total_entropy_loss = sum(entropies)
        return self.out_conv(dec_in), indices, total_entropy_loss, out_for_transformer, hs[-1]

# ==== TEST ====
if __name__ == "__main__":
    # Input shape: must be (B, 3, T, H, W) with T, H, W divisible by 16 (e.g., 32, 64, 128, etc.)
    x = torch.randn(1, 3, 32, 64, 64)
    vae = VideoVAE()
    out, indices, entropy_loss = vae(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Entropy loss:", entropy_loss.item())
