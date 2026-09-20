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

import torch
from PIL import Image as PILImage
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...configuration_utils import FrozenDict
from ...guiders import ClassifierFreeGuidance
from ...models import AutoencoderKLQwenImage21
from ..modular_pipeline import ModularPipelineBlocks
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam


def get_qwenimage21_prompt_embeds(text_encoder, processor, prompt, image, device):
    sys_prompt = "Comprehend and analyze the provided prompt."
    prefix = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n"
    suffix = "{}<|im_end|>\n<|im_start|>assistant\n"
    prompt_template_t2i = prefix + suffix
    prompt_template_ti2i = prefix + "<image1><|vision_start|><|image_pad|><|vision_end|>" + suffix
    sys_message = [{"role": "system", "content": [{"type": "text", "text": sys_prompt}]}]
    drop_idx = len(processor.apply_chat_template(sys_message, tokenize=True, return_dict=False)[0])
    img_token_id = processor.tokenizer.encode("<|image_pad|>")[0]
    prompt = [prompt] if isinstance(prompt, str) else prompt
    # Qwen has no bos token, so an empty string leaves the encoder with nothing to read.
    prompt = [" " if not p else p for p in prompt]
    is_t2i = image is None

    if is_t2i:
        prompts = [prompt_template_t2i.format(t) for t in prompt]
    else:
        prompts = []
        condition_pil_list = []
        for t in prompt:
            n_imgs = len(image)
            replace = "<image1><|vision_start|><|image_pad|><|vision_end|>"
            for i in range(2, n_imgs + 1):
                replace += f" <image{i}><|vision_start|><|image_pad|><|vision_end|>"
            template = prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", replace)
            prompts.append(template.format(t))
        # Each prompt's template repeats the `<|image_pad|>` placeholders, so hand the processor one set of
        # images per prompt, in the order the placeholders appear.
        for _ in prompt:
            for img in image:
                if not isinstance(img, PILImage.Image):
                    img = PILImage.fromarray(img)
                if img.mode == "RGBA":
                    # The checkpoint was trained with the alpha composited over white for the vision encoder.
                    # Only this copy is flattened; the VAE still reads all four channels.
                    white = PILImage.new("RGB", img.size, (255, 255, 255))
                    white.paste(img, mask=img.getchannel("A"))
                    img = white
                condition_pil_list.append(img)

    # Left padding, as the checkpoint was trained with. `_extract_masked_hidden` drops the padding either way,
    # but the side decides the positions the encoder sees for a batch of prompts of different lengths.
    processor_kwargs = {
        "text": prompts,
        "padding": True,
        "padding_side": "left",
        "return_tensors": "pt",
    }
    if not is_t2i:
        processor_kwargs["images"] = condition_pil_list

    model_inputs = processor(**processor_kwargs).to(device)

    forward_kwargs = {
        "input_ids": model_inputs.input_ids,
        "attention_mask": model_inputs.attention_mask,
        "output_hidden_states": True,
    }
    if not is_t2i and hasattr(model_inputs, "pixel_values"):
        forward_kwargs.update(pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw)
    if hasattr(model_inputs, "mm_token_type_ids"):
        forward_kwargs["mm_token_type_ids"] = model_inputs.mm_token_type_ids

    # `hidden_states[-1]` has to be the last decoder layer's output, before the text encoder's final RMSNorm:
    # that is what the transformer was trained on. It is what transformers 4.x returns there, but from
    # transformers 5.0 the output capturing ties that entry to `last_hidden_state`, so it comes back normalized
    # instead — a third of the signal the transformer reads, which shows up first in rendered text. A forward hook
    # returning the module's input replaces its output, which neutralizes the norm for this call on either version.
    # TODO: replace this with `tie_last_hidden_states=False` in the text encoder's config, which
    # huggingface/transformers#48087 adds, once that ships in a stable transformers release (5.18).
    text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_encoder(**forward_kwargs)
    finally:
        handle.remove()
    hidden_states = outputs.hidden_states[-1]

    split_hidden_states = list(
        torch.split(
            hidden_states[model_inputs.attention_mask.bool()], model_inputs.attention_mask.sum(dim=1).tolist(), dim=0
        )
    )
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]

    image_pad_mask = [
        (sample_ids[sample_mask.bool()] == img_token_id)
        for sample_ids, sample_mask in zip(model_inputs.input_ids, model_inputs.attention_mask)
    ]
    image_pad_mask = [e[drop_idx:] for e in image_pad_mask]

    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    max_seq_len = max(e.size(0) for e in split_hidden_states)
    prompt_embeds = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
    )
    encoder_attention_mask = torch.stack(
        [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
    )
    image_pad_mask = torch.stack([torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in image_pad_mask])

    return prompt_embeds, encoder_attention_mask, image_pad_mask


class QwenImage21TextEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Encode prompts and optional condition images together with Qwen3-VL."

    @property
    def expected_components(self):
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("processor", Qwen3VLProcessor),
            ComponentSpec(
                "guider",
                ClassifierFreeGuidance,
                config=FrozenDict({"guidance_scale": 1.0}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def inputs(self):
        return [
            InputParam.template("prompt", required=True),
            InputParam.template("negative_prompt"),
            InputParam("condition_images", type_hint=list, description="Resized images to encode with each prompt."),
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam.template("prompt_embeds"),
            OutputParam.template("negative_prompt_embeds"),
            OutputParam.template("prompt_embeds_mask"),
            OutputParam.template("negative_prompt_embeds_mask"),
            OutputParam(
                "image_pad_mask", type_hint=torch.Tensor, description="Vision token positions in each positive prompt."
            ),
            OutputParam(
                "negative_image_pad_mask",
                type_hint=torch.Tensor,
                description="Vision token positions in each negative prompt.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        prompt = block_state.prompt
        if isinstance(prompt, str):
            prompt = [prompt]
        if not prompt or not all(isinstance(p, str) for p in prompt):
            raise ValueError("`prompt` must be a string or a nonempty list of strings.")
        block_state.prompt_embeds, block_state.prompt_embeds_mask, block_state.image_pad_mask = (
            get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                prompt,
                block_state.condition_images,
                components._execution_device,
            )
        )
        block_state.negative_prompt_embeds = None
        block_state.negative_prompt_embeds_mask = None
        block_state.negative_image_pad_mask = None
        components.guider.set_state(step=0, num_inference_steps=None, timestep=None)
        if components.guider.num_conditions > 1:
            negative_prompt = block_state.negative_prompt
            if negative_prompt is None:
                negative_prompt = ""
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * len(prompt)
            if len(negative_prompt) != len(prompt):
                raise ValueError("`negative_prompt` must have the same batch size as `prompt`.")
            (
                block_state.negative_prompt_embeds,
                block_state.negative_prompt_embeds_mask,
                block_state.negative_image_pad_mask,
            ) = get_qwenimage21_prompt_embeds(
                components.text_encoder,
                components.processor,
                negative_prompt,
                block_state.condition_images,
                components._execution_device,
            )
        self.set_block_state(state, block_state)
        return components, state


def encode_image(vae, image):
    latents = vae.encode(image.to(device=vae.device, dtype=vae.dtype)).latent_dist.mode()
    mean = latents.new_tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = latents.new_tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    return ((latents - mean) / std).flatten(2).transpose(1, 2)


class QwenImage21VaeEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Encode condition images into unpatched, normalized latent tokens."

    @property
    def expected_components(self):
        return [ComponentSpec("vae", AutoencoderKLQwenImage21)]

    @property
    def inputs(self):
        return [
            InputParam("vae_images", required=True, type_hint=list, description="Normalized RGBA condition tensors.")
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "condition_latents", type_hint=torch.Tensor, description="Concatenated condition image tokens."
            ),
            OutputParam(
                "condition_shapes",
                type_hint=list,
                description="Frame, height and width for each condition latent grid.",
            ),
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.condition_latents = torch.cat(
            [encode_image(components.vae, image) for image in block_state.vae_images], dim=1
        )
        block_state.condition_shapes = [
            (1, image.shape[-2] // 16, image.shape[-1] // 16) for image in block_state.vae_images
        ]
        self.set_block_state(state, block_state)
        return components, state


class QwenImage21InpaintVaeEncoderStep(ModularPipelineBlocks):
    model_name = "qwenimage21"

    @property
    def description(self):
        return "Encode the source image at the target resolution for inpainting preservation."

    @property
    def expected_components(self):
        return [ComponentSpec("vae", AutoencoderKLQwenImage21)]

    @property
    def inputs(self):
        return [
            InputParam(
                "source_image", required=True, type_hint=torch.Tensor, description="Normalized source RGBA image."
            )
        ]

    @property
    def intermediate_outputs(self):
        return [
            OutputParam(
                "source_latents",
                type_hint=torch.Tensor,
                description="Source tokens used to preserve the unmasked area.",
            )
        ]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        block_state.source_latents = encode_image(components.vae, block_state.source_image.unsqueeze(2))
        self.set_block_state(state, block_state)
        return components, state
