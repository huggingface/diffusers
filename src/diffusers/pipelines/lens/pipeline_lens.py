# Copyright 2025 Microsoft and The HuggingFace Team. All rights reserved.
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

import inspect
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKLFlux2, LensTransformer2DModel
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import is_torch_xla_available, logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import LensPipelineOutput
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)


class LensGptOssEncoder(GptOssForCausalLM):
    """`GptOssForCausalLM` subclass that exposes selected hidden states.

    This text encoder extracts hidden states from specific intermediate layers of the
    GPT-OSS model, enabling multi-layer text features for the Lens transformer. It
    early-exits after the last selected layer to avoid unnecessary computation.

    Call `set_selected_layers` before the first forward pass to configure which
    layers to capture.
    """

    def set_selected_layers(self, layer_indices: Sequence[int]) -> None:
        layers = [int(i) for i in layer_indices]
        if not layers:
            raise ValueError("layer_indices must be non-empty")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layer_indices must be unique; got {layers}")
        if min(layers) < 0 or max(layers) >= len(self.model.layers):
            raise ValueError(
                f"layer_indices out of range; got {layers}, "
                f"model has {len(self.model.layers)} layers"
            )
        self._lens_selected_layers = layers
        self._lens_max_layer = max(layers)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """The [`LensGptOssEncoder`] forward method.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.

        Returns:
            `list[torch.Tensor]`: Hidden states at the configured selected layers.
        """
        if not hasattr(self, "_lens_selected_layers"):
            raise RuntimeError("Call set_selected_layers(...) before forward().")

        target_device = self.model.embed_tokens.weight.device
        if input_ids is not None and input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)

        model = self.model
        inputs_embeds = model.embed_tokens(input_ids)
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0).expand_as(input_ids)

        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        captured: list[torch.Tensor | None] = [None] * len(self._lens_selected_layers)
        index_lookup = {idx: pos for pos, idx in enumerate(self._lens_selected_layers)}

        for i, decoder_layer in enumerate(model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[model.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if i in index_lookup:
                captured[index_lookup[i]] = hidden_states
            if i == self._lens_max_layer:
                break

        for pos, layer_idx in enumerate(self._lens_selected_layers):
            if captured[pos] is None:
                raise RuntimeError(
                    f"Failed to capture hidden state for layer {layer_idx}"
                )
        return captured


import transformers as _transformers

if not hasattr(_transformers, "LensGptOssEncoder"):
    _transformers.LensGptOssEncoder = LensGptOssEncoder

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import LensPipeline

        >>> pipe = LensPipeline.from_pretrained("microsoft/Lens", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> prompt = "A cat holding a sign that says hello world"
        >>> image = pipe(prompt, num_inference_steps=50).images[0]
        >>> image.save("lens.png")
        ```
"""


RESOLUTION_BUCKETS = {
    1024: {
        "1:2": (1472, 736),
        "9:16": (1376, 768),
        "2:3": (1248, 832),
        "3:4": (1152, 864),
        "1:1": (1024, 1024),
        "4:3": (864, 1152),
        "3:2": (832, 1248),
        "16:9": (768, 1376),
        "2:1": (736, 1472),
    },
    1440: {
        "1:2": (2080, 1040),
        "9:16": (1936, 1088),
        "2:3": (1760, 1168),
        "3:4": (1616, 1216),
        "1:1": (1440, 1440),
        "4:3": (1216, 1616),
        "3:2": (1168, 1760),
        "16:9": (1088, 1936),
        "2:1": (1040, 2080),
    },
}


def resolve_resolution(base_resolution: int, aspect_ratio: str):
    if base_resolution not in RESOLUTION_BUCKETS:
        raise ValueError(
            f"Unsupported base_resolution={base_resolution}. "
            f"Supported: {tuple(RESOLUTION_BUCKETS.keys())}"
        )
    table = RESOLUTION_BUCKETS[base_resolution]
    if aspect_ratio not in table:
        raise ValueError(
            f"Unsupported aspect_ratio={aspect_ratio!r}. "
            f"Supported: {tuple(RESOLUTION_BUCKETS[1024].keys())}"
        )
    return table[aspect_ratio]


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


CHAT_SYSTEM = (
    "Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background."
)
CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97


class LensPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKLFlux2,
        text_encoder,
        tokenizer: PreTrainedTokenizerBase,
        transformer: LensTransformer2DModel,
    ):
        super().__init__()
        self.register_modules(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.vae_scale_factor = 16
        self.latent_channels = self.transformer.config.in_channels
        self.txt_offset = DEFAULT_TXT_OFFSET

        if hasattr(self.text_encoder, "set_selected_layers"):
            selected_layers = list(self.transformer.config.selected_layer_index)
            self.text_encoder.set_selected_layers(selected_layers)
        self.default_sample_size = 1024
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _build_chat_inputs(
        self, prompts: list[str], max_sequence_length: int, device: torch.device
    ):
        rendered = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": CHAT_SYSTEM, "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "thinking": CHAT_ASSISTANT_THINKING, "content": ""},
            ]
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=False
            )
            text = text.split("<|return|>")[0]
            rendered.append(text)

        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=max_sequence_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    @torch.no_grad()
    def _get_text_embeddings(
        self, prompts: list[str], max_sequence_length: int, device: torch.device
    ):
        input_ids, attn_mask = self._build_chat_inputs(prompts, max_sequence_length, device)
        layer_outputs = self.text_encoder(input_ids, attention_mask=attn_mask)

        offset = self.txt_offset
        if input_ids.shape[1] > offset:
            features = [feat[:, offset:, :].contiguous() for feat in layer_outputs]
            mask = attn_mask[:, offset:].bool()
        else:
            zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
            features = [layer_outputs[0].new_zeros(zero_shape) for _ in layer_outputs]
            mask = torch.zeros((input_ids.shape[0], 0), dtype=torch.bool, device=device)
        return features, mask

    def encode_prompt(
        self,
        prompt: Union[str, list[str]],
        negative_prompt: Union[str, list[str]] = "",
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[list[torch.Tensor]] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[list[torch.Tensor]] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        n = int(num_images_per_prompt)

        if isinstance(negative_prompt, str):
            negatives = [negative_prompt] * len(prompts)
        else:
            negatives = list(negative_prompt)
            if len(negatives) == 1:
                negatives = negatives * len(prompts)
            if len(negatives) != len(prompts):
                raise ValueError(
                    "negative_prompt must be a string or a list of the same length as prompt"
                )

        if prompt_embeds is None:
            prompt_embeds, prompt_mask = self._get_text_embeddings(prompts, max_sequence_length, device)
            prompt_embeds, prompt_mask = self._repeat_for_n(prompt_embeds, prompt_mask, n)
        elif prompt_mask is None:
            raise ValueError("`prompt_mask` must be provided when passing `prompt_embeds`.")

        if negative_prompt_embeds is None:
            if all(isinstance(neg, str) and not neg.strip() for neg in negatives):
                negative_prompt_embeds = [feat.new_zeros(feat.shape) for feat in prompt_embeds]
                negative_prompt_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)
            else:
                negative_prompt_embeds, negative_prompt_mask = self._get_text_embeddings(
                    negatives, max_sequence_length, device
                )
                negative_prompt_embeds, negative_prompt_mask = self._repeat_for_n(
                    negative_prompt_embeds, negative_prompt_mask, n
                )
        elif negative_prompt_mask is None:
            raise ValueError(
                "`negative_prompt_mask` must be provided when passing `negative_prompt_embeds`."
            )
        return prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask

    @staticmethod
    def _repeat_for_n(features: list[torch.Tensor], mask: torch.Tensor, n: int):
        if n == 1:
            return features, mask
        features = [f.repeat_interleave(n, dim=0) for f in features]
        mask = mask.repeat_interleave(n, dim=0)
        return features, mask

    @staticmethod
    def _align_text_features(
        pos_features: list[torch.Tensor],
        pos_mask: torch.Tensor,
        neg_features: list[torch.Tensor],
        neg_mask: torch.Tensor,
    ):
        seq_pos = pos_features[0].shape[1]
        seq_neg = neg_features[0].shape[1]
        target = max(seq_pos, seq_neg)

        def pad(features, cur):
            if cur == target:
                return features
            pad_len = target - cur
            return [
                torch.cat([feat, feat.new_zeros((feat.shape[0], pad_len, feat.shape[-1]))], dim=1)
                for feat in features
            ]

        def pad_mask(mask, cur):
            if cur == target:
                return mask
            return torch.cat(
                [mask, torch.zeros((mask.shape[0], target - cur), dtype=torch.bool, device=mask.device)], dim=1
            )

        pos_features = pad(pos_features, seq_pos)
        neg_features = pad(neg_features, seq_neg)
        pos_mask = pad_mask(pos_mask.bool(), seq_pos)
        neg_mask = pad_mask(neg_mask.bool(), seq_neg)
        return pos_features, pos_mask, neg_features, neg_mask

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        shape = (batch_size, latent_h * latent_w, num_channels_latents)
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
        return randn_tensor(shape, generator=generator, device=device, dtype=dtype)

    def check_inputs(
        self,
        prompt,
        height,
        width,
        prompt_embeds,
        callback_on_step_end_tensor_inputs,
    ):
        if height is None or width is None:
            raise ValueError("height and width must be provided (or use base_resolution + aspect_ratio).")
        if height % self.vae_scale_factor or width % self.vae_scale_factor:
            raise ValueError(
                f"height and width must be divisible by {self.vae_scale_factor}; got ({height}, {width})."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Either `prompt` or `prompt_embeds` must be provided.")
        if callback_on_step_end_tensor_inputs is not None:
            for k in callback_on_step_end_tensor_inputs:
                if k not in self._callback_tensor_inputs:
                    raise ValueError(
                        f"callback_on_step_end_tensor_inputs entry {k!r} is not in {self._callback_tensor_inputs}."
                    )

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(b, c * 4, h // 2, w // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        b, c, h, w = latents.shape
        latents = latents.reshape(b, c // 4, 2, 2, h, w)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(b, c // 4, h * 2, w * 2)

    @torch.no_grad()
    def _decode(self, latents: torch.Tensor, latent_h: int, latent_w: int):
        b = latents.shape[0]
        p1, p2 = 2, 2
        h, w = latent_h, latent_w
        c = latents.shape[-1] // (p1 * p2)
        latents = latents.reshape(b, h, w, c, p1, p2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(b, c, h * p1, w * p2)
        latents = latents.to(self.vae.dtype)

        bn = self.vae.bn
        mean = bn.running_mean.view(1, -1, 1, 1)
        var = bn.running_var.view(1, -1, 1, 1)
        std = torch.sqrt(var + self.vae.config.batch_norm_eps)
        shift = (-mean).to(device=latents.device, dtype=latents.dtype)
        scale = (1.0 / std).to(device=latents.device, dtype=latents.dtype)
        x = self._patchify_latents(latents)
        x = x / scale - shift
        x = self._unpatchify_latents(x)
        return self.vae.decode(x).sample

    @staticmethod
    def _to_pil(image: torch.Tensor):
        from PIL import Image

        image = image.clamp(-1.0, 1.0)
        image = (image + 1.0) * (255.0 / 2.0)
        image = image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
        return [Image.fromarray(im) for im in image]

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, list[str]] = None,
        negative_prompt: Union[str, list[str]] = "",
        height: Optional[int] = None,
        width: Optional[int] = None,
        base_resolution: Optional[int] = None,
        aspect_ratio: Optional[str] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[list[torch.Tensor]] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[list[torch.Tensor]] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        """
        Generate an image from a prompt.

        Examples:
        """
        if base_resolution is not None and aspect_ratio is not None:
            height, width = resolve_resolution(base_resolution, aspect_ratio)
        elif height is None or width is None:
            height = width = self.default_sample_size

        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        device = self._execution_device
        dtype = self.transformer.dtype

        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_mask=negative_prompt_mask,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = self._align_text_features(
            prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask
        )

        encoder_features = [
            torch.cat([pf, nf], dim=0).to(dtype=dtype)
            for pf, nf in zip(prompt_embeds, negative_prompt_embeds)
        ]
        encoder_mask = torch.cat([prompt_mask, negative_prompt_mask], dim=0)

        batch_size = prompt_embeds[0].shape[0]
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        seq_len = latent_h * latent_w
        latents = self.prepare_latents(
            batch_size, self.latent_channels, height, width,
            dtype=dtype, device=device, generator=generator, latents=latents,
        )

        mu = compute_empirical_mu(seq_len, num_inference_steps)
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(sigmas=sigmas, device=device, mu=mu)

        img_shapes = [(1, latent_h, latent_w)]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(self.scheduler.timesteps):
                timestep = t.expand(batch_size * 2).to(latents.dtype)
                hidden_states = latents.repeat(2, 1, 1)

                noise = self.transformer(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_features,
                    encoder_hidden_states_mask=encoder_mask,
                    timestep=timestep / 1000,
                    img_shapes=img_shapes,
                    return_dict=False,
                )[0]

                cond, uncond = noise.chunk(2)
                comb = uncond + guidance_scale * (cond - uncond)
                cond_norm = torch.norm(cond, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                scale = torch.where(
                    comb_norm > 0,
                    cond_norm / comb_norm.clamp_min(1e-12),
                    torch.ones_like(comb_norm),
                )
                noise_pred = comb * scale

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    cb_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    cb_out = callback_on_step_end(self, i, t, cb_kwargs)
                    latents = cb_out.pop("latents", latents)

                progress_bar.update()

        if output_type == "latent":
            images = latents
        else:
            decoded = self._decode(latents, latent_h, latent_w)
            if output_type == "pil":
                images = self._to_pil(decoded)
            elif output_type == "np":
                decoded = decoded.clamp(-1.0, 1.0)
                decoded = (decoded + 1.0) * 0.5
                images = decoded.permute(0, 2, 3, 1).to("cpu", torch.float32).numpy()
            else:
                raise ValueError(f"output_type must be one of 'pil', 'np', 'latent'; got {output_type!r}.")

        self.maybe_free_model_hooks()

        if not return_dict:
            return (images,)
        return LensPipelineOutput(images=images)
