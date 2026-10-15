# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import html
import math
import re

import torch
import torch.nn.functional as F

from ...models.autoencoders.autoencoder_edenlive import EdenLiveLightTAEDecoder
from ...utils import is_ftfy_available
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from .video_processor import EdenLiveVideoProcessor


if is_ftfy_available():
    import ftfy


def initialize_pipeline(
    pipe,
    vae,
    text_encoder,
    tokenizer,
    transformer,
    scheduler,
    proxy_encoder,
    proxy_projection,
    image_encoder,
    image_processor,
    decoder,
    profile,
    proxy_resize_mode,
    proxy_unconditional,
    proxy_projection_autocast,
    proxy_image_size,
):
    if profile not in ("5b-bidir", "14b-bidir", "5b-causal", "14b-causal"):
        raise ValueError("`profile` must be one of 5b-bidir, 14b-bidir, 5b-causal, or 14b-causal.")
    pipe.model_size, mode = profile.split("-")
    pipe.is_causal = mode == "causal"
    if proxy_unconditional not in ("same", "zero"):
        raise ValueError("`proxy_unconditional` must be 'same' or 'zero'.")
    if decoder is not None:
        validate_decoder(decoder, vae)
    pipe.register_modules(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=scheduler,
        proxy_encoder=proxy_encoder,
        proxy_projection=proxy_projection,
        image_encoder=image_encoder,
        image_processor=image_processor,
        decoder=decoder,
    )
    pipe.register_to_config(
        profile=profile,
        proxy_resize_mode=proxy_resize_mode,
        proxy_unconditional=proxy_unconditional,
        proxy_projection_autocast=proxy_projection_autocast,
        proxy_image_size=proxy_image_size,
    )
    pipe.vae_scale_factor_temporal = vae.config.scale_factor_temporal if vae is not None else 4
    pipe.vae_scale_factor_spatial = vae.config.scale_factor_spatial if vae is not None else 8
    pipe.video_processor = VideoProcessor(vae_scale_factor=pipe.vae_scale_factor_spatial)
    pipe.proxy_video_processor = EdenLiveVideoProcessor(vae_scale_factor=1, do_normalize=False, do_convert_rgb=True)


def validate_decoder(decoder, vae):
    if not isinstance(decoder, torch.nn.Module) or not callable(getattr(decoder, "decode", None)):
        raise ValueError("`decoder` must be a module with a `decode(latents, return_dict=False)` method.")
    config = getattr(decoder, "config", None)
    latent_channels = getattr(config, "latent_channels", getattr(config, "z_dim", None))
    temporal = getattr(decoder, "scale_factor_temporal", getattr(config, "scale_factor_temporal", None))
    spatial = getattr(decoder, "scale_factor_spatial", getattr(config, "scale_factor_spatial", None))
    if vae is None or (latent_channels, temporal, spatial) != (
        vae.config.z_dim,
        vae.config.scale_factor_temporal,
        vae.config.scale_factor_spatial,
    ):
        raise ValueError("`decoder` latent channels and temporal/spatial scale factors must match the Wan VAE.")


def encode_prompt(
    pipe,
    prompt,
    negative_prompt=None,
    do_classifier_free_guidance=True,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    max_sequence_length=512,
    device=None,
    dtype=None,
):
    device = device or pipe._execution_device
    dtype = dtype or (pipe.transformer.dtype if pipe.transformer is not None else pipe.text_encoder.dtype)

    def encode(texts):
        if is_ftfy_available():
            texts = [ftfy.fix_text(text) for text in texts]
        texts = [re.sub(r"\s+", " ", html.unescape(html.unescape(text))).strip() for text in texts]
        inputs = pipe.tokenizer(
            texts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        mask = inputs.attention_mask.to(device)
        embeddings = pipe.text_encoder(inputs.input_ids.to(device), attention_mask=mask).last_hidden_state
        embeddings = embeddings.to(device=device, dtype=dtype)
        return embeddings.masked_fill(~mask.bool().unsqueeze(-1), 0)

    if prompt_embeds is None:
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if not prompt or not all(isinstance(text, str) for text in prompt):
            raise ValueError("Provide a string prompt, a list of strings, or `prompt_embeds`.")
        prompt_embeds = encode(prompt)
    else:
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
    if do_classifier_free_guidance:
        if negative_prompt_embeds is None:
            negative_prompt = "" if negative_prompt is None else negative_prompt
            negative_prompt = (
                [negative_prompt] * prompt_embeds.shape[0] if isinstance(negative_prompt, str) else negative_prompt
            )
            if len(negative_prompt) != prompt_embeds.shape[0]:
                raise ValueError("Negative and positive prompts must have the same batch size.")
            negative_prompt_embeds = encode(negative_prompt)
        else:
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
        if negative_prompt_embeds.shape != prompt_embeds.shape:
            raise ValueError("Positive and negative prompt embeddings must have matching shapes.")
    return prompt_embeds, negative_prompt_embeds


def check_inputs(
    pipe,
    prompt,
    prompt_embeds,
    negative_prompt,
    negative_prompt_embeds,
    image,
    video,
    height,
    width,
    output_type,
    callback_inputs,
):
    if image is None or video is None:
        raise ValueError("Both a first-frame `image` and a full-rate RGB proxy `video` are required.")
    if prompt is not None and prompt_embeds is not None:
        raise ValueError("Pass either `prompt` or `prompt_embeds`, not both.")
    if negative_prompt is not None and negative_prompt_embeds is not None:
        raise ValueError("Pass either `negative_prompt` or `negative_prompt_embeds`, not both.")
    patch = pipe.transformer.config.patch_size
    if (
        height <= 0
        or width <= 0
        or height % (pipe.vae_scale_factor_spatial * patch[1])
        or width % (pipe.vae_scale_factor_spatial * patch[2])
    ):
        raise ValueError(
            "Height and width must be positive and divisible by the VAE scale times the spatial patch size."
        )
    if output_type not in ("np", "pt", "pil", "latent"):
        raise ValueError("`output_type` must be 'np', 'pt', 'pil', or 'latent'.")
    if not set(callback_inputs).issubset(pipe._callback_tensor_inputs):
        raise ValueError(f"Callback tensor inputs must belong to {pipe._callback_tensor_inputs}.")


def prepare_latents(pipe, batch_size, height, width, num_frames, dtype, device, generator=None, latents=None):
    shape = (
        batch_size,
        pipe.vae.config.z_dim,
        (num_frames - 1) // pipe.vae_scale_factor_temporal + 1,
        height // pipe.vae_scale_factor_spatial,
        width // pipe.vae_scale_factor_spatial,
    )
    if latents is not None:
        if tuple(latents.shape) != shape:
            raise ValueError(f"Expected latents with shape {shape}, got {tuple(latents.shape)}.")
        return latents.to(device=device, dtype=dtype)
    return sample_noise(pipe, shape, generator, device, dtype)


def sample_noise(pipe, shape, generator, device, dtype):
    parallel_config = getattr(pipe.transformer, "_parallel_config", None)
    context_parallel_config = getattr(parallel_config, "context_parallel_config", None)
    if context_parallel_config is None:
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    group = context_parallel_config._flattened_mesh.get_group()
    source_rank = torch.distributed.get_global_rank(group, 0)
    if torch.distributed.get_rank() == source_rank:
        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        noise = torch.empty(shape, device=device, dtype=dtype)
    torch.distributed.broadcast(noise, src=source_rank, group=group)
    return noise


def prepare_conditions(pipe, image, proxy_video, height, width, num_frames, batch_size, device, dtype):
    pixels = pipe.proxy_video_processor.preprocess(image).float() * 255
    scale = max(height / pixels.shape[-2], width / pixels.shape[-1])
    resized_height = max(height, math.ceil(pixels.shape[-2] * scale))
    resized_width = max(width, math.ceil(pixels.shape[-1] * scale))
    pixels = F.interpolate(pixels, (resized_height, resized_width), mode="bilinear", align_corners=False)
    top, left = (resized_height - height) // 2, (resized_width - width) // 2
    pixels = pixels[:, :, top : top + height, left : left + width] / 127.5 - 1
    pixels = pixels.to(device=device)
    if pixels.shape[0] == 1:
        pixels = pixels.expand(batch_size, -1, -1, -1)
    if pixels.shape[0] != batch_size or proxy_video.shape[0] not in (1, batch_size):
        raise ValueError("Image and proxy-video batch sizes must be one or match the prompt batch.")
    clip_embeds = None
    if pipe.model_size == "14b":
        if pipe.image_encoder is None or pipe.image_processor is None:
            raise ValueError("The 14B profile requires a CLIP image encoder and processor.")
        rgb = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        clip_inputs = pipe.image_processor(images=list(rgb), return_tensors="pt").pixel_values
        clip_embeds = (
            pipe.image_encoder(
                clip_inputs.to(device=device, dtype=pipe.image_encoder.dtype), output_hidden_states=True
            )
            .hidden_states[-2]
            .to(dtype)
        )
        condition_video = torch.cat(
            [pixels.unsqueeze(2), pixels.new_zeros(batch_size, 3, num_frames - 1, height, width)], dim=2
        )
    else:
        condition_video = pixels.unsqueeze(2)
    condition = pipe.vae.encode(condition_video.to(pipe.vae.dtype)).latent_dist.mode()
    mean = condition.new_tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    inverse_std = 1 / condition.new_tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    condition = ((condition - mean) * inverse_std).to(dtype)
    first_frame = condition[:, :, :1]
    if pipe.model_size == "14b":
        mask = condition.new_zeros(batch_size, pipe.vae_scale_factor_temporal, *condition.shape[2:])
        mask[:, :, 0] = 1
        condition = torch.cat([mask, condition], dim=1)
    proxy_video = proxy_video.expand(batch_size, -1, -1, -1, -1)
    proxy_height, proxy_width = pipe.config.proxy_image_size
    proxy_features = []
    for frame in proxy_video.unbind(2):
        frame = frame.to(device=device, dtype=torch.float32)
        if pipe.model_size == "5b":
            # Match the original FP32 [-1, 1] -> [0, 1] round trip before DINO.
            frame = (frame * 2 - 1 + 1) * 0.5
        inputs = pipe.proxy_video_processor.preprocess_proxy_frames(
            frame, proxy_height, proxy_width, pipe.config.proxy_resize_mode
        )
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = pipe.proxy_encoder(pixel_values=inputs).last_hidden_state
        patch_size = pipe.proxy_encoder.config.patch_size
        grid_height, grid_width = proxy_height // patch_size, proxy_width // patch_size
        start = 1 + pipe.proxy_encoder.config.num_register_tokens
        hidden = hidden[:, start : start + grid_height * grid_width].reshape(batch_size, grid_height, grid_width, -1)
        with torch.autocast(
            device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and pipe.config.proxy_projection_autocast
        ):
            projected = pipe.proxy_projection(hidden if pipe.config.proxy_projection_autocast else hidden.float())
        proxy_features.append(projected.float().permute(0, 3, 1, 2))
    return condition, first_frame, clip_embeds, torch.stack(proxy_features, dim=2)


def prepare_model_input(pipe, latents, condition, first_frame, timestep, start=0, causal=False):
    model_input = latents
    pin_first = start == 0 and (causal or pipe.model_size == "5b")
    if pin_first:
        model_input = model_input.clone()
        model_input[:, :, :1] = first_frame
    if pipe.model_size == "14b":
        model_input = torch.cat([model_input, condition[:, :, start : start + latents.shape[2]]], dim=1)
    timestep = timestep.expand(latents.shape[0])
    if pin_first:
        patch = pipe.transformer.config.patch_size
        tokens_per_frame = (latents.shape[3] // patch[1]) * (latents.shape[4] // patch[2])
        timestep = timestep[:, None].expand(-1, latents.shape[2] * tokens_per_frame).clone()
        timestep[:, :tokens_per_frame] = 0
    return model_input, timestep


def decode_latents(pipe, latents, output_type):
    if output_type == "latent":
        return latents
    latents = latents.to(pipe.vae.dtype)
    mean = latents.new_tensor(pipe.vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latents.new_tensor(pipe.vae.config.latents_std).view(1, -1, 1, 1, 1)
    if isinstance(pipe.decoder, EdenLiveLightTAEDecoder):
        raw_latents = latents * std + mean
    else:
        raw_latents = latents / (1 / std) + mean
    decoder = pipe.decoder if pipe.decoder is not None else pipe.vae
    dtype = getattr(decoder, "dtype", None)
    if dtype is None:
        parameter = next(decoder.parameters(), None)
        dtype = parameter.dtype if parameter is not None else raw_latents.dtype
    video = decoder.decode(raw_latents.to(dtype=dtype), return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type=output_type)


def encode_timeline(pipe, timelines, negative_prompt, num_frames, latent_shape, max_sequence_length, guidance):
    timelines = [timelines] if isinstance(timelines, dict) else timelines
    batch_size = len(timelines)
    negatives = (
        [negative_prompt or ""] * batch_size if isinstance(negative_prompt, (str, type(None))) else negative_prompt
    )
    if len(negatives) != batch_size:
        raise ValueError("Negative prompt batch must match the timeline batch.")
    texts, negative_texts, lengths, counts = [], [], [], []
    tokens_per_frame = (latent_shape[-2] // pipe.transformer.config.patch_size[1]) * (
        latent_shape[-1] // pipe.transformer.config.patch_size[2]
    )
    for sample, negative in zip(timelines, negatives):
        fps = float(sample["fps"])
        if not math.isfinite(fps) or fps <= 0 or sample["num_frames"] != num_frames:
            raise ValueError("Each timeline needs positive finite fps and the requested decoded num_frames.")
        duration, previous = (num_frames - 1) / fps, 0.0
        segments = []
        for segment in sample["segments"]:
            start, end = float(segment["start_sec"]), float(segment["end_sec"])
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or end <= start
                or not math.isclose(start, previous, abs_tol=1e-6)
            ):
                raise ValueError("Timeline segments must be finite, ordered, contiguous, and start at zero.")
            previous = end
            if start < duration:
                segments.append({"text": segment["text"], "end_sec": min(end, duration)})
        if not segments or previous < duration:
            raise ValueError("Timeline segments must cover the requested video.")
        frame_counts = [0] * len(segments)
        for frame in range(latent_shape[2]):
            index = bisect.bisect_left(
                [segment["end_sec"] for segment in segments], frame * pipe.vae_scale_factor_temporal / fps - 1e-7
            )
            frame_counts[index] += 1
        if 0 in frame_counts:
            raise ValueError("A timeline segment is shorter than the latent time resolution.")
        lengths.append(tuple(count * tokens_per_frame for count in frame_counts))
        counts.append(len(segments))
        texts.extend(segment["text"] for segment in segments)
        negative_texts.extend([negative] * len(segments))
    positive, negative = encode_prompt(pipe, texts, negative_texts, guidance, max_sequence_length=max_sequence_length)

    def pack(embeddings):
        if embeddings is None:
            return None
        result = embeddings.new_zeros(batch_size, max(counts) * max_sequence_length, embeddings.shape[-1])
        offset = 0
        for index, count in enumerate(counts):
            result[index, : count * max_sequence_length] = embeddings[offset : offset + count].flatten(0, 1)
            offset += count
        return result

    return pack(positive), pack(negative), {"lengths": tuple(lengths), "text_length": max_sequence_length}
