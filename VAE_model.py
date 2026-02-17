# video_vae_modular_final.py

# ==============================================================================
# 1. IMPORTS & CONFIGURATION
# ==============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor, AutoModelForCausalLM
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class VideoVAEConfig:
    in_channels: int = 3
    base_ch: int = 64
    num_blocks: int = 4
    quant_emb_dim: int = 16
    alignment_dim: int = 256
    quant_align_loss_weight: float = 0.1
    likelihood_loss_weight: float = 0.2
    dino_loss_weight: float = 0.25

# ==============================================================================
# 2. PERCEPTUAL & TEXT MODULES
# ==============================================================================

class DINOv2Extractor(nn.Module):
    """
    A frozen DINOv2 model to extract perceptual features from video frames.
    """
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        model_name = "facebook/dinov2-base"
        print("Loading DINOv2 model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print("DINOv2 loaded and frozen successfully. 🦖")

    def forward(self, video_tensor: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = video_tensor.shape
        video_tensor = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.processor(images=video_tensor, return_tensors="pt", do_rescale=False).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Return the features of the [CLS] token
        return outputs.last_hidden_state[:, 0].view(b, t, -1)

class QwenVLTextEncoder(nn.Module):
    """A frozen Qwen-VL model to extract text embeddings."""
    def __init__(self, device="cuda"):
        super().__init__()
        model_id = "Qwen/Qwen2.5-VL-Instruct"
        self.device = device
        print("Loading Qwen-VL model and processor...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto", trust_remote_code=True).eval()
        for param in self.model.parameters(): param.requires_grad = False
        print("Qwen-VL loaded and frozen successfully. 🥶")

    def forward(self, text_prompts: list[str]):
        messages = [[{"role": "user", "content": [{"type": "text", "text": prompt}]}] for prompt in text_prompts]
        text_inputs = self.processor(conversations=messages, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**text_inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].to(self.device)

class TextVideoCrossAttention(nn.Module):
    """Performs cross-attention between video features (Q) and text features (K,V)."""
    def __init__(self, video_channels, text_embed_dim):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj = nn.Linear(video_channels, video_channels), nn.Linear(text_embed_dim, video_channels), nn.Linear(text_embed_dim, video_channels)
        self.out_proj = nn.Linear(video_channels, video_channels)

    def forward(self, video_feat, text_embedding):
        B, C, T, H, W = video_feat.shape
        video_seq = video_feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
        q, k, v = self.q_proj(video_seq), self.k_proj(text_embedding), self.v_proj(text_embedding)
        attn_output = F.scaled_dot_product_attention(q.unsqueeze(1), k, v).squeeze(1)
        return self.out_proj(attn_output).reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3)

# ==============================================================================
# 3. CORE ARCHITECTURAL BLOCKS
# ==============================================================================

class ProjectedLFQ(nn.Module):
    """Projects features and quantizes them, returning an entropy loss."""
    def __init__(self, in_channels, quant_channels, entropy_loss_weight=0.1):
        super().__init__()
        self.project = nn.Conv3d(in_channels, quant_channels, 1)
        self.entropy_loss_weight = entropy_loss_weight

    def forward(self, x):
        x_proj = self.project(x)
        quantized_x_hard = torch.where(x_proj > 0, 1.0, -1.0)
        quantized_x = x_proj + (quantized_x_hard - x_proj).detach()
        indices = (quantized_x > 0).long()
        probs = indices.float().mean(dim=(0, 2, 3, 4))
        entropy = - (probs * torch.log(probs.clamp(min=1e-8)) + (1 - probs) * torch.log((1 - probs).clamp(min=1e-8)))
        entropy_loss = -entropy.mean() * self.entropy_loss_weight
        return quantized_x, indices, entropy_loss

class VideoVAEEncoderBlock(nn.Module):
    """Standard VAE encoder block for downsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm(self.conv1(x)))
        h = self.act(self.norm(self.conv2(h)))
        return self.pool(h)

class PyramidalLFQBlock(nn.Module):
    """A block in the pyramidal upsampler: upsample -> fuse -> text-attend -> quantize."""
    def __init__(self, in_ch, skip_ch, out_ch, text_embed_dim, quant_emb_dim):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv3d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.text_cross_attn = TextVideoCrossAttention(out_ch, text_embed_dim)
        self.lfq = ProjectedLFQ(out_ch, quant_channels=quant_emb_dim)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x, skip, text_embedding):
        x_up = self.upsample(x)
        x_fused = self.act(self.norm(self.conv(torch.cat([x_up, skip], dim=1))))
        h_attn = x_fused + self.text_cross_attn(x_fused, text_embedding)
        q, indices, entropy_loss = self.lfq(h_attn)
        return h_attn, q, indices, entropy_loss

class VideoVAEDecoderBlock(nn.Module):
    """Standard VAE decoder block for upsampling."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.norm(self.upsample(x)))
        return self.act(self.norm(self.conv(h)))

# ==============================================================================
# 4. PRIMARY VideoVAE MODEL
# ==============================================================================

class VideoVAE(nn.Module):
    """
    A modular, text-conditioned Video VAE with a Pyramidal LFQ structure
    and multiple perception-based losses for high-quality synthesis.
    """
    def __init__(self, cfg: VideoVAEConfig, device="cuda"):
        super().__init__()
        self.cfg = cfg
        self.device = device

        # --- Sub-models (Text, Perception) ---
        self.text_encoder = QwenVLTextEncoder(device=device)
        text_embed_dim = self.text_encoder.model.config.hidden_size
        if self.training: # Only load DINOv2 if we are in training mode
            self.dino_extractor = DINOv2Extractor(device=device)

        # --- VAE Encoder ---
        self.enc_blocks = nn.ModuleList()
        chs = [cfg.base_ch * (2**i) for i in range(cfg.num_blocks)]
        current_ch = cfg.in_channels
        for ch in chs:
            self.enc_blocks.append(VideoVAEEncoderBlock(current_ch, ch))
            current_ch = ch

        # --- Pyramidal LFQ Upsampler ---
        rev_channels = list(reversed(chs))
        self.pyramid_blocks = nn.ModuleList()
        for i in range(2): # 2 stages for 4x total upscaling
            self.pyramid_blocks.append(
                PyramidalLFQBlock(rev_channels[i], rev_channels[i+1], rev_channels[i+1], text_embed_dim, cfg.quant_emb_dim)
            )

        # --- VAE Decoder ---
        self.dec_blocks = nn.ModuleList()
        decoder_channels = [chs[1], chs[0]]
        for i in range(len(decoder_channels)):
            in_ch = decoder_channels[i]
            out_ch = decoder_channels[i+1] if i + 1 < len(decoder_channels) else cfg.base_ch
            self.dec_blocks.append(VideoVAEDecoderBlock(in_ch, out_ch))
        self.out_conv = nn.Conv3d(cfg.base_ch, cfg.in_channels, 1)

        # --- Loss-specific Modules ---
        codebook_size = 2**cfg.quant_emb_dim
        self.quant_embedding = nn.Embedding(codebook_size, text_embed_dim)
        self.to_quant_logits = nn.Linear(text_embed_dim, codebook_size)
        quant_pooled_dim = chs[2] + chs[1]
        self.quant_proj = nn.Linear(quant_pooled_dim, cfg.alignment_dim)
        self.text_proj_for_quant = nn.Linear(text_embed_dim, cfg.alignment_dim)

    def forward(self, x: torch.Tensor, text_prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Core inference path. Encodes, quantizes via pyramid, and decodes.
        Returns all intermediate products needed for loss calculation.
        """
        text_embedding = self.text_encoder(text_prompts)
        
        encoder_features = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            encoder_features.append(h)

        rev_features = list(reversed(encoder_features))
        h = rev_features[0]
        pyramid_outputs = {'q': [], 'indices': [], 'entropies': []}
        for i, block in enumerate(self.pyramid_blocks):
            h, q, indices, entropy = block(h, rev_features[i + 1], text_embedding)
            pyramid_outputs['q'].append(q)
            pyramid_outputs['indices'].append(indices)
            pyramid_outputs['entropies'].append(entropy)
        
        dec_in = h
        for block in self.dec_blocks:
            dec_in = block(dec_in)
        reconstruction = torch.tanh(self.out_conv(dec_in))

        return {
            "reconstruction": reconstruction,
            "text_embedding": text_embedding,
            "pyramid_outputs": pyramid_outputs
        }

    def calculate_losses(self, original_video: torch.Tensor, forward_outputs: Dict) -> Dict:
        """
        Calculates all training-specific losses. This method should only be
        called during the training loop.
        """
        if not self.training:
            raise RuntimeError("calculate_losses() should only be called in training mode.")
            
        # Unpack forward pass results
        recon = forward_outputs["reconstruction"]
        text_emb = forward_outputs["text_embedding"]
        pyramid_out = forward_outputs["pyramid_outputs"]
        all_q, all_indices, all_entropies = pyramid_out['q'], pyramid_out['indices'], pyramid_out['entropies']

        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(recon, original_video)

        # 2. Entropy Loss
        entropy_loss = sum(all_entropies)
        
        # 3. P(Q|text) Likelihood Loss
        B = text_emb.size(0)
        seqs = [idx.view(B, self.cfg.quant_emb_dim, -1) for idx in all_indices]
        full_seq_bits = torch.cat(seqs, dim=2).permute(0, 2, 1)
        powers_of_2 = (2**torch.arange(self.cfg.quant_emb_dim, device=self.device)).float()
        quant_token_ids = (full_seq_bits * powers_of_2).sum(dim=2).long()
        quant_embeds = self.quant_embedding(quant_token_ids)
        combined_embeds = torch.cat([text_emb, quant_embeds], dim=1)
        with torch.no_grad():
            qwen_outputs = self.text_encoder.model(inputs_embeds=combined_embeds, output_hidden_states=True)
        last_hidden = qwen_outputs.hidden_states[-1][:, text_emb.shape[1] - 1:-1, :]
        pred_logits = self.to_quant_logits(last_hidden)
        likelihood_loss = F.cross_entropy(pred_logits.reshape(-1, pred_logits.size(-1)), quant_token_ids.reshape(-1))
        
        # 4. Quantized Vector-Text Alignment Loss
        q_pooled = [F.adaptive_avg_pool3d(q, 1).view(B, -1) for q in all_q]
        q_pooled_cat = torch.cat(q_pooled, dim=1)
        text_pooled = text_emb.mean(dim=1)
        q_aligned = self.quant_proj(q_pooled_cat)
        text_aligned = self.text_proj_for_quant(text_pooled)
        quant_align_loss = F.cosine_embedding_loss(q_aligned, text_aligned, torch.ones(B, device=self.device))
        
        # 5. DINOv2 Perceptual Loss (KL Divergence)
        orig_dino_feats = self.dino_extractor(original_video)
        recon_dino_feats = self.dino_extractor(recon)
        p = F.softmax(orig_dino_feats, dim=-1)
        q = F.log_softmax(recon_dino_feats, dim=-1)
        dino_loss = F.kl_div(q, p, reduction='batchmean')

        # --- Final Weighted Sum ---
        total_loss = (recon_loss + entropy_loss +
                      self.cfg.likelihood_loss_weight * likelihood_loss +
                      self.cfg.quant_align_loss_weight * quant_align_loss +
                      self.cfg.dino_loss_weight * dino_loss)

        return {
            "total_loss": total_loss, "reconstruction_loss": recon_loss,
            "entropy_loss": entropy_loss, "likelihood_loss": likelihood_loss,
            "quant_alignment_loss": quant_align_loss, "dino_perceptual_loss": dino_loss
        }

# ==============================================================================
# 5. EXAMPLE USAGE
# ==============================================================================
if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu": print("WARNING: Running on CPU. This will be extremely slow.")

    try:
        config = VideoVAEConfig(quant_emb_dim=16) # Set LFQ size to 16
        model = VideoVAE(config, device=device).to(device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("-" * 40)
        print(f"Trainable model parameters: {trainable_params:,}")
        print("(This should NOT include frozen DINOv2 or Qwen-VL models)")
        print("-" * 40)

        # --- SIMULATED TRAINING STEP ---
        print("\n--- 1. Simulating Training Step ---")
        model.train() # Set model to training mode
        batch_size = 2
        video_input = torch.randn(batch_size, 3, 16, 64, 64).to(device)
        prompts = ["A stunning sunrise over a calm ocean.", "A busy city street at night with neon lights."]
        
        # In a real training loop, this would be inside the loop
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad()
        
        forward_outputs = model(video_input, text_prompts=prompts)
        losses = model.calculate_losses(video_input, forward_outputs)
        
        # Backpropagation
        losses["total_loss"].backward()
        optimizer.step()
        
        print("Training step successful. Losses calculated:")
        for name, value in losses.items(): print(f"  - {name:<25}: {value.item():.4f}")

        # --- SIMULATED INFERENCE STEP ---
        print("\n--- 2. Simulating Inference Step ---")
        model.eval() # Set model to evaluation mode
        with torch.no_grad():
            # Notice we only call the forward pass and don't need the loss function
            inference_outputs = model(video_input, text_prompts=prompts)
            reconstructed_video = inference_outputs["reconstruction"]

        print("Inference step successful.")
        print("Input Video Shape:         ", video_input.shape)
        print("Reconstructed Video Shape: ", reconstructed_video.shape)

    except Exception as e:
        print(f"\n--- ❌ An Error Occurred ---")
        print(f"Error: {e}")
        if "out of memory" in str(e).lower():
            print("\n💡 Suggestion: CUDA Out-of-Memory. Try reducing `base_ch`, `num_blocks`, or input resolution.")
