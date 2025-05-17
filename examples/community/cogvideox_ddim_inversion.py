"""
This script performs DDIM inversion for video frames using a pre-trained model and generates
a video reconstruction based on a provided prompt. It utilizes the CogVideoX pipeline to
process video frames, apply the DDIM inverse scheduler, and produce an output video.

**Please notice that this script is based on the CogVideoX 5B model, and would not generate
a good result for 2B variants.**

Usage:
    python cogvideox_ddim_inversion.py
        --model-path /path/to/model
        --prompt "a prompt"
        --video-path /path/to/video.mp4
        --output-path /path/to/output

For more details about the cli arguments, please run `python cogvideox_ddim_inversion.py --help`.

Author:
    LittleNyima <littlenyima[at]163[dot]com>
"""

import argparse
import math
import os
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.models.attention_processor import Attention, CogVideoXAttnProcessor2_0
from diffusers.models.autoencoders import AutoencoderKLCogVideoX
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipeline, retrieve_timesteps
from diffusers.schedulers import CogVideoXDDIMScheduler, DDIMInverseScheduler
from diffusers.utils import export_to_video


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error.
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort: skip


class DDIMInversionArguments(TypedDict):
    model_path: str
    prompt: str
    video_path: str
    output_path: str
    guidance_scale: float
    num_inference_steps: int
    skip_frames_start: int
    skip_frames_end: int
    frame_sample_step: Optional[int]
    max_num_frames: int
    width: int
    height: int
    fps: int
    dtype: torch.dtype
    seed: int
    device: torch.device


def get_args() -> DDIMInversionArguments:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True, help="Path of the pretrained model")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the direct sample procedure")
    parser.add_argument("--video_path", type=str, required=True, help="Path of the video for inversion")
    parser.add_argument("--output_path", type=str, default="output", help="Path of the output videos")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--skip_frames_start", type=int, default=0, help="Number of skipped frames from the start")
    parser.add_argument("--skip_frames_end", type=int, default=0, help="Number of skipped frames from the end")
    parser.add_argument("--frame_sample_step", type=int, default=None, help="Temporal stride of the sampled frames")
    parser.add_argument("--max_num_frames", type=int, default=81, help="Max number of sampled frames")
    parser.add_argument("--width", type=int, default=720, help="Resized width of the video frames")
    parser.add_argument("--height", type=int, default=480, help="Resized height of the video frames")
    parser.add_argument("--fps", type=int, default=8, help="Frame rate of the output videos")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"], help="Dtype of the model")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the random number generator")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device for inference")

    args = parser.parse_args()
    args.dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    args.device = torch.device(args.device)

    return DDIMInversionArguments(**vars(args))


class CogVideoXAttnProcessor2_0ForDDIMInversion(CogVideoXAttnProcessor2_0):
    def __init__(self):
        super().__init__()

    def calculate_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn: Attention,
        batch_size: int,
        image_seq_length: int,
        text_seq_length: int,
        attention_mask: Optional[torch.Tensor],
        image_rotary_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Core attention computation with inversion-guided RoPE integration.

        Args:
            query (`torch.Tensor`): `[batch_size, seq_len, dim]` query tensor
            key (`torch.Tensor`): `[batch_size, seq_len, dim]` key tensor
            value (`torch.Tensor`): `[batch_size, seq_len, dim]` value tensor
            attn (`Attention`): Parent attention module with projection layers
            batch_size (`int`): Effective batch size (after chunk splitting)
            image_seq_length (`int`): Length of image feature sequence
            text_seq_length (`int`): Length of text feature sequence
            attention_mask (`Optional[torch.Tensor]`): Attention mask tensor
            image_rotary_emb (`Optional[torch.Tensor]`): Rotary embeddings for image positions

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                (1) hidden_states: [batch_size, image_seq_length, dim] processed image features
                (2) encoder_hidden_states: [batch_size, text_seq_length, dim] processed text features
        """
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                if key.size(2) == query.size(2):  # Attention for reference hidden states
                    key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
                else:  # RoPE should be applied to each group of image tokens
                    key[:, :, text_seq_length : text_seq_length + image_seq_length] = apply_rotary_emb(
                        key[:, :, text_seq_length : text_seq_length + image_seq_length], image_rotary_emb
                    )
                    key[:, :, text_seq_length * 2 + image_seq_length :] = apply_rotary_emb(
                        key[:, :, text_seq_length * 2 + image_seq_length :], image_rotary_emb
                    )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Process the dual-path attention for the inversion-guided denoising procedure.

        Args:
            attn (`Attention`): Parent attention module
            hidden_states (`torch.Tensor`): `[batch_size, image_seq_len, dim]` Image tokens
            encoder_hidden_states (`torch.Tensor`): `[batch_size, text_seq_len, dim]` Text tokens
            attention_mask (`Optional[torch.Tensor]`): Optional attention mask
            image_rotary_emb (`Optional[torch.Tensor]`): Rotary embeddings for image tokens

        Returns:
            `Tuple[torch.Tensor, torch.Tensor]`:
                (1) Final hidden states: `[batch_size, image_seq_length, dim]` Resulting image tokens
                (2) Final encoder states: `[batch_size, text_seq_length, dim]` Resulting text tokens
        """
        image_seq_length = hidden_states.size(1)
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query, query_reference = query.chunk(2)
        key, key_reference = key.chunk(2)
        value, value_reference = value.chunk(2)
        batch_size = batch_size // 2

        hidden_states, encoder_hidden_states = self.calculate_attention(
            query=query,
            key=torch.cat((key, key_reference), dim=1),
            value=torch.cat((value, value_reference), dim=1),
            attn=attn,
            batch_size=batch_size,
            image_seq_length=image_seq_length,
            text_seq_length=text_seq_length,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states_reference, encoder_hidden_states_reference = self.calculate_attention(
            query=query_reference,
            key=key_reference,
            value=value_reference,
            attn=attn,
            batch_size=batch_size,
            image_seq_length=image_seq_length,
            text_seq_length=text_seq_length,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        return (
            torch.cat((hidden_states, hidden_states_reference)),
            torch.cat((encoder_hidden_states, encoder_hidden_states_reference)),
        )


class OverrideAttnProcessors:
    r"""
    Context manager for temporarily overriding attention processors in CogVideo transformer blocks.

    Designed for DDIM inversion process, replaces original attention processors with
    `CogVideoXAttnProcessor2_0ForDDIMInversion` and restores them upon exit. Uses Python context manager
    pattern to safely manage processor replacement.

    Typical usage:
    ```python
    with OverrideAttnProcessors(transformer):
        # Perform DDIM inversion operations
    ```

    Args:
        transformer (`CogVideoXTransformer3DModel`):
            The transformer model containing attention blocks to be modified. Should have
            `transformer_blocks` attribute containing `CogVideoXBlock` instances.
    """

    def __init__(self, transformer: CogVideoXTransformer3DModel):
        self.transformer = transformer
        self.original_processors = {}

    def __enter__(self):
        for block in self.transformer.transformer_blocks:
            block = cast(CogVideoXBlock, block)
            self.original_processors[id(block)] = block.attn1.get_processor()
            block.attn1.set_processor(CogVideoXAttnProcessor2_0ForDDIMInversion())

    def __exit__(self, _0, _1, _2):
        for block in self.transformer.transformer_blocks:
            block = cast(CogVideoXBlock, block)
            block.attn1.set_processor(self.original_processors[id(block)])


def get_video_frames(
    video_path: str,
    width: int,
    height: int,
    skip_frames_start: int,
    skip_frames_end: int,
    max_num_frames: int,
    frame_sample_step: Optional[int],
) -> torch.FloatTensor:
    """
    Extract and preprocess video frames from a video file for VAE processing.

    Args:
        video_path (`str`): Path to input video file
        width (`int`): Target frame width for decoding
        height (`int`): Target frame height for decoding
        skip_frames_start (`int`): Number of frames to skip at video start
        skip_frames_end (`int`): Number of frames to skip at video end
        max_num_frames (`int`): Maximum allowed number of output frames
        frame_sample_step (`Optional[int]`):
            Frame sampling step size. If None, automatically calculated as:
            (total_frames - skipped_frames) // max_num_frames

    Returns:
        `torch.FloatTensor`: Preprocessed frames in `[F, C, H, W]` format where:
        - `F`: Number of frames (adjusted to 4k + 1 for VAE compatibility)
        - `C`: Channels (3 for RGB)
        - `H`: Frame height
        - `W`: Frame width
    """
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path, width=width, height=height)
        video_num_frames = len(video_reader)
        start_frame = min(skip_frames_start, video_num_frames)
        end_frame = max(0, video_num_frames - skip_frames_end)

        if end_frame <= start_frame:
            indices = [start_frame]
        elif end_frame - start_frame <= max_num_frames:
            indices = list(range(start_frame, end_frame))
        else:
            step = frame_sample_step or (end_frame - start_frame) // max_num_frames
            indices = list(range(start_frame, end_frame, step))

        frames = video_reader.get_batch(indices=indices)
        frames = frames[:max_num_frames].float()  # ensure that we don't go over the limit

        # Choose first (4k + 1) frames as this is how many is required by the VAE
        selected_num_frames = frames.size(0)
        remainder = (3 + selected_num_frames) % 4
        if remainder != 0:
            frames = frames[:-remainder]
        assert frames.size(0) % 4 == 1

        # Normalize the frames
        transform = T.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        frames = torch.stack(tuple(map(transform, frames)), dim=0)

        return frames.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]


class CogVideoXDDIMInversionOutput:
    inverse_latents: torch.FloatTensor
    recon_latents: torch.FloatTensor

    def __init__(self, inverse_latents: torch.FloatTensor, recon_latents: torch.FloatTensor):
        self.inverse_latents = inverse_latents
        self.recon_latents = recon_latents


class CogVideoXPipelineForDDIMInversion(CogVideoXPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDDIMScheduler,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.inverse_scheduler = DDIMInverseScheduler(**scheduler.config)

    def encode_video_frames(self, video_frames: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode video frames into latent space using Variational Autoencoder.

        Args:
            video_frames (`torch.FloatTensor`):
                Input frames tensor in `[F, C, H, W]` format from `get_video_frames()`

        Returns:
            `torch.FloatTensor`: Encoded latents in `[1, F, D, H_latent, W_latent]` format where:
            - `F`: Number of frames (same as input)
            - `D`: Latent channel dimension
            - `H_latent`: Latent space height (H // 2^vae.downscale_factor)
            - `W_latent`: Latent space width (W // 2^vae.downscale_factor)
        """
        vae: AutoencoderKLCogVideoX = self.vae
        video_frames = video_frames.to(device=vae.device, dtype=vae.dtype)
        video_frames = video_frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(x=video_frames).latent_dist.sample().transpose(1, 2)
        return latent_dist * vae.config.scaling_factor

    @torch.no_grad()
    def export_latents_to_video(self, latents: torch.FloatTensor, video_path: str, fps: int):
        r"""
        Decode latent vectors into video and export as video file.

        Args:
            latents (`torch.FloatTensor`): Encoded latents in `[B, F, D, H_latent, W_latent]` format from
                `encode_video_frames()`
            video_path (`str`): Output path for video file
            fps (`int`): Target frames per second for output video
        """
        video = self.decode_latents(latents)
        frames = self.video_processor.postprocess_video(video=video, output_type="pil")
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        export_to_video(video_frames=frames[0], output_video_path=video_path, fps=fps)

    # Modified from CogVideoXPipeline.__call__
    @torch.no_grad()
    def sample(
        self,
        latents: torch.FloatTensor,
        scheduler: Union[DDIMInverseScheduler, CogVideoXDDIMScheduler],
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        reference_latents: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        r"""
        Execute the core sampling loop for video generation/inversion using CogVideoX.

        Implements the full denoising trajectory recording for both DDIM inversion and
        generation processes. Supports dynamic classifier-free guidance and reference
        latent conditioning.

        Args:
            latents (`torch.FloatTensor`):
                Initial noise tensor of shape `[B, F, C, H, W]`.
            scheduler (`Union[DDIMInverseScheduler, CogVideoXDDIMScheduler]`):
                Scheduling strategy for diffusion process. Use:
                (1) `DDIMInverseScheduler` for inversion
                (2) `CogVideoXDDIMScheduler` for generation
            prompt (`Optional[Union[str, List[str]]]`):
                Text prompt(s) for conditional generation. Defaults to unconditional.
            negative_prompt (`Optional[Union[str, List[str]]]`):
                Negative prompt(s) for guidance. Requires `guidance_scale > 1`.
            num_inference_steps (`int`):
                Number of denoising steps. Affects quality/compute trade-off.
            guidance_scale (`float`):
                Classifier-free guidance weight. 1.0 = no guidance.
            use_dynamic_cfg (`bool`):
                Enable time-varying guidance scale (cosine schedule)
            eta (`float`):
                DDIM variance parameter (0 = deterministic process)
            generator (`Optional[Union[torch.Generator, List[torch.Generator]]]`):
                Random number generator(s) for reproducibility
            attention_kwargs (`Optional[Dict[str, Any]]`):
                Custom parameters for attention modules
            reference_latents (`torch.FloatTensor`):
                Reference latent trajectory for conditional sampling. Shape should match
                `[T, B, F, C, H, W]` where `T` is number of timesteps

        Returns:
            `torch.FloatTensor`:
                Full denoising trajectory tensor of shape `[T, B, F, C, H, W]`.
        """
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://huggingface.co/papers/2205.11487 . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        if reference_latents is not None:
            prompt_embeds = torch.cat([prompt_embeds] * 2, dim=0)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps, device)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latents = latents.to(device=device) * scheduler.init_noise_sigma

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        if isinstance(scheduler, DDIMInverseScheduler):  # Inverse scheduler does not accept extra kwargs
            extra_step_kwargs = {}

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(
                height=latents.size(3) * self.vae_scale_factor_spatial,
                width=latents.size(4) * self.vae_scale_factor_spatial,
                num_frames=latents.size(1),
                device=device,
            )
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)

        trajectory = torch.zeros_like(latents).unsqueeze(0).repeat(len(timesteps), 1, 1, 1, 1, 1)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                if reference_latents is not None:
                    reference = reference_latents[i]
                    reference = torch.cat([reference] * 2) if do_classifier_free_guidance else reference
                    latent_model_input = torch.cat([latent_model_input, reference], dim=0)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred.float()

                if reference_latents is not None:  # Recover the original batch size
                    noise_pred, _ = noise_pred.chunk(2)

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the noisy sample x_t-1 -> x_t
                latents = scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                latents = latents.to(prompt_embeds.dtype)
                trajectory[i] = latents

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    progress_bar.update()

        # Offload all models
        self.maybe_free_model_hooks()

        return trajectory

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        video_path: str,
        guidance_scale: float,
        num_inference_steps: int,
        skip_frames_start: int,
        skip_frames_end: int,
        frame_sample_step: Optional[int],
        max_num_frames: int,
        width: int,
        height: int,
        seed: int,
    ):
        """
        Performs DDIM inversion on a video to reconstruct it with a new prompt.

        Args:
            prompt (`str`): The text prompt to guide the reconstruction.
            video_path (`str`): Path to the input video file.
            guidance_scale (`float`): Scale for classifier-free guidance.
            num_inference_steps (`int`): Number of denoising steps.
            skip_frames_start (`int`): Number of frames to skip from the beginning of the video.
            skip_frames_end (`int`): Number of frames to skip from the end of the video.
            frame_sample_step (`Optional[int]`): Step size for sampling frames. If None, all frames are used.
            max_num_frames (`int`): Maximum number of frames to process.
            width (`int`): Width of the output video frames.
            height (`int`): Height of the output video frames.
            seed (`int`): Random seed for reproducibility.

        Returns:
            `CogVideoXDDIMInversionOutput`: Contains the inverse latents and reconstructed latents.
        """
        if not self.transformer.config.use_rotary_positional_embeddings:
            raise NotImplementedError("This script supports CogVideoX 5B model only.")
        video_frames = get_video_frames(
            video_path=video_path,
            width=width,
            height=height,
            skip_frames_start=skip_frames_start,
            skip_frames_end=skip_frames_end,
            max_num_frames=max_num_frames,
            frame_sample_step=frame_sample_step,
        ).to(device=self.device)
        video_latents = self.encode_video_frames(video_frames=video_frames)
        inverse_latents = self.sample(
            latents=video_latents,
            scheduler=self.inverse_scheduler,
            prompt="",
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        with OverrideAttnProcessors(transformer=self.transformer):
            recon_latents = self.sample(
                latents=torch.randn_like(video_latents),
                scheduler=self.scheduler,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                reference_latents=reversed(inverse_latents),
            )
        return CogVideoXDDIMInversionOutput(
            inverse_latents=inverse_latents,
            recon_latents=recon_latents,
        )


if __name__ == "__main__":
    arguments = get_args()
    pipeline = CogVideoXPipelineForDDIMInversion.from_pretrained(
        arguments.pop("model_path"),
        torch_dtype=arguments.pop("dtype"),
    ).to(device=arguments.pop("device"))

    output_path = arguments.pop("output_path")
    fps = arguments.pop("fps")
    inverse_video_path = os.path.join(output_path, f"{arguments.get('video_path')}_inversion.mp4")
    recon_video_path = os.path.join(output_path, f"{arguments.get('video_path')}_reconstruction.mp4")

    # Run DDIM inversion
    output = pipeline(**arguments)
    pipeline.export_latents_to_video(output.inverse_latents[-1], inverse_video_path, fps)
    pipeline.export_latents_to_video(output.recon_latents[-1], recon_video_path, fps)
