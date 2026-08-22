# Copyright 2026 The JoyAI-Video-Edit Team and The HuggingFace Team. All rights reserved.
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

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL.Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2Tokenizer

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from ...hooks import JoyVideoEditKVCacheConfig
from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLJoyVideoEdit, JoyVideoEditTransformer3DModel
from ...models.transformers.transformer_joyvideoedit import SELF_ATTN_MODE_REF_IMAGE_CACHE
from ...schedulers import FlowMatchEulerDiscreteScheduler
from ...utils import logging, replace_example_docstring
from ...utils.torch_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .pipeline_output import JoyVideoEditPipelineOutput


logger = logging.get_logger(__name__)


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import JoyVideoEditPipeline
        >>> from diffusers.utils import export_to_video, load_video
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model_id = "jdopensource/JoyAI-Video-Edit-Diffusers"
        >>> mimo_id = "XiaomiMiMo/MiMo-VL-7B-RL-2508"
        >>> processor = AutoProcessor.from_pretrained(mimo_id)
        >>> text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(mimo_id, dtype=torch.bfloat16)
        >>> pipe = JoyVideoEditPipeline.from_pretrained(
        ...     model_id,
        ...     text_encoder=text_encoder,
        ...     processor=processor,
        ...     dtype=torch.bfloat16,
        ... )
        >>> pipe.enable_model_cpu_offload()

        >>> video = load_video(
        ...     "https://raw.githubusercontent.com/jd-opensource/JoyAI-Video-Edit/main/assets/input.mp4"
        ... )
        >>> prompt = (
        ...     "Transform the scene into a British castle royal aristocratic style. Modify the characters' clothing "
        ...     "to aristocratic attire: dress the man in a tailored velvet suit with a ruffled cravat, and the women "
        ...     "in elegant silk gowns with lace details and embroidered bodices. Change their hairstyles to classic "
        ...     "aristocratic styles, such as elaborate updos with subtle jewels for the women and a neatly styled "
        ...     "classic cut for the man. Change the environmental decoration to a British castle interior: replace "
        ...     "the plain walls and abstract painting with stone walls and antique oil paintings in gilded frames, "
        ...     "and replace the white window curtains with heavy velvet drapes. The characters' ages and facial "
        ...     "features must remain completely unchanged. The dining table, white tablecloth, plates of food, wine "
        ...     "glasses, water glasses, and the characters' positions and actions must remain unchanged."
        ... )
        >>> output = pipe(
        ...     video=video,
        ...     prompt=prompt,
        ...     num_inference_steps=2,
        ...     generator=torch.Generator(device="cpu").manual_seed(0),
        ... )
        >>> export_to_video(output.frames[0], "joyvideoedit.mp4", fps=24)
        ```
"""


class JoyVideoEditPipeline(DiffusionPipeline):
    r"""
    Pipeline for chunk-wise causal video editing using the JoyAI-Video-Edit architecture.

    The source video is VAE-encoded into a latent sequence that conditions a dual-stream MM-DiT transformer. The
    transformer denoises the output latents one causal chunk at a time: each chunk attends to a sliding window of
    previously-denoised chunks (and an optional static reference image) through a per-layer KV cache, so later chunks
    stay temporally consistent with earlier ones without recomputing their key/value projections.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    MiMo-VL is an external runtime dependency. Load `XiaomiMiMo/MiMo-VL-7B-RL-2508` separately and pass its model and
    processor to this pipeline, or provide precomputed prompt embeddings.

    Args:
        transformer ([`JoyVideoEditTransformer3DModel`]):
            The streaming video-editing transformer that denoises the output latents.
        vae ([`AutoencoderKLJoyVideoEdit`]):
            Causal, chunk-streamable VAE to encode the source video and decode the edited latents.
        text_encoder ([`Qwen2_5_VLForConditionalGeneration`], *optional*):
            MiMo-VL model used to encode the prompt and first video frame. Load it from MiMo-VL's repository. Required
            unless `prompt_embeds` are provided.
        tokenizer ([`Qwen2Tokenizer`], *optional*):
            Tokenizer paired with `text_encoder`. Defaults to the processor's tokenizer when omitted.
        processor ([`Qwen2_5_VLProcessor`], *optional*):
            MiMo-VL processor loaded from `XiaomiMiMo/MiMo-VL-7B-RL-2508`. Required unless `prompt_embeds` are
            provided.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Flow-matching scheduler used to denoise each chunk.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]
    _optional_components = ["text_encoder", "tokenizer", "processor"]

    # Sentinel chunk id under which the static reference image's KV is prefilled (never a real chunk index).
    _KV_CACHE_ID_REF_IMAGE = -1

    def __init__(
        self,
        transformer: JoyVideoEditTransformer3DModel,
        vae: AutoencoderKLJoyVideoEdit,
        text_encoder: Optional[Qwen2_5_VLForConditionalGeneration],
        tokenizer: Optional[Qwen2Tokenizer],
        processor: Optional[Qwen2_5_VLProcessor],
        scheduler: FlowMatchEulerDiscreteScheduler,
    ):
        super().__init__()

        if tokenizer is None and processor is not None:
            tokenizer = processor.tokenizer

        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            scheduler=scheduler,
        )

        # The KV cache is required for chunk-wise denoising.
        if getattr(self, "transformer", None) is not None and not self.transformer.is_cache_enabled:
            self.transformer.enable_cache(JoyVideoEditKVCacheConfig())

        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 16
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 8
        transformer_patch_size = (
            self.transformer.config.patch_size if getattr(self, "transformer", None) else (1, 1, 1)
        )
        self.height_multiple = self.vae_scale_factor_spatial * transformer_patch_size[1]
        self.width_multiple = self.vae_scale_factor_spatial * transformer_patch_size[2]
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        # Encode the prompt and the video's first frame with the image-description template.
        self.prompt_template_encode = (
            "<|im_start|>system\n \\nDescribe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.prompt_template_encode_start_idx = None
        if self.tokenizer is not None:
            prefix_ids = self.tokenizer(self.prompt_template_encode.split("{}")[0]).input_ids
            user_id = self.tokenizer.convert_tokens_to_ids("user")
            self.prompt_template_encode_start_idx = prefix_ids.index(user_id)

        # The anchor frame is resized to a fixed ViT input area before being packed by the processor.
        self.vit_input_size = 512

    # ------------------------------------------------------------------
    # Prompt encoding (multimodal: text + video's first frame as an image anchor)
    # ------------------------------------------------------------------

    # Copied from diffusers.pipelines.qwenimage.pipeline_qwenimage.QwenImagePipeline._extract_masked_hidden
    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        image: PIL.Image.Image,
        device: torch.device,
        max_sequence_length: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        missing_components = [
            name for name in ("text_encoder", "tokenizer", "processor") if getattr(self, name, None) is None
        ]
        if missing_components:
            raise ValueError(
                f"Missing MiMo-VL components: {', '.join(missing_components)}. Load them from "
                "`XiaomiMiMo/MiMo-VL-7B-RL-2508` and pass them to `JoyVideoEditPipeline.from_pretrained`, or pass "
                "precomputed `prompt_embeds` and `prompt_embeds_mask`."
            )

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if self.prompt_template_encode_start_idx is None:
            prefix_ids = self.tokenizer(self.prompt_template_encode.split("{}")[0]).input_ids
            user_id = self.tokenizer.convert_tokens_to_ids("user")
            self.prompt_template_encode_start_idx = prefix_ids.index(user_id)
        drop_idx = self.prompt_template_encode_start_idx
        txt = [self.prompt_template_encode.format(e) for e in prompt]

        # Resize the anchor image to the fixed ViT input area while preserving its aspect ratio.
        target_area = self.vit_input_size * self.vit_input_size
        scale = (target_area / max(image.height * image.width, 1)) ** 0.5
        new_h = max(1, round(image.height * scale))
        new_w = max(1, round(image.width * scale))
        anchor = image.convert("RGB").resize((new_w, new_h), PIL.Image.BILINEAR)

        model_inputs = self.processor(text=txt, images=[anchor] * len(txt), padding=True, return_tensors="pt").to(
            device
        )
        # Forward all multimodal fields required for position encoding.
        outputs = self.text_encoder(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]

        # Remove padding and the template prefix, keep the most recent tokens, then left-align and pad the batch.
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs["attention_mask"])
        split_hidden_states = [e[drop_idx:][-max_sequence_length:] for e in split_hidden_states]
        attn_mask_list = [e.new_ones(e.size(0), dtype=torch.long) for e in split_hidden_states]
        max_seq_len = max(e.size(0) for e in split_hidden_states)
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        prompt_embeds_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )
        return prompt_embeds, prompt_embeds_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[PIL.Image.Image] = None,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Encode a text prompt together with the video's first frame into multimodal embeddings.

        Args:
            prompt (`str` or `List[str]`): Prompt(s) to encode.
            image (`PIL.Image.Image`, *optional*): Anchor image (the video's first frame) spliced into the prompt as
                an `<image>` token. Required unless `prompt_embeds` are provided.
            device (`torch.device`, *optional*): Target device.
            prompt_embeds (`torch.Tensor`, *optional*): Pre-computed embeddings that bypass encoding.
            prompt_embeds_mask (`torch.Tensor`, *optional*): Attention mask for pre-computed embeddings.
            max_sequence_length (`int`, *optional*, defaults to 1024): Maximum prompt length.

        Returns:
            Tuple of `(prompt_embeds, prompt_embeds_mask)`.
        """
        device = device or self._execution_device
        if prompt_embeds is None:
            if image is None:
                raise ValueError("`image` (the video's first frame) is required to encode a `prompt`.")
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(
                prompt, image, device, max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)
        prompt_embeds = prompt_embeds[:, :max_sequence_length].to(device=device)
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length].to(device=device)
        # The reference DiT uses unmasked attention for fully valid prompt sequences. Preserve that numerical path and
        # keep an explicit mask only when padding is present.
        if prompt_embeds_mask.all():
            prompt_embeds_mask = None
        return prompt_embeds, prompt_embeds_mask

    # ------------------------------------------------------------------
    # Latent (de)normalization
    # ------------------------------------------------------------------

    def normalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(device=latent.device, dtype=latent.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
        )
        return (latent - latents_mean) / latents_std

    def denormalize_latents(self, latent: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, -1, 1, 1, 1)
            .to(device=latent.device, dtype=latent.dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(device=latent.device, dtype=latent.dtype)
        )
        return latent * latents_std + latents_mean

    # ------------------------------------------------------------------
    # Chunk and temporal-id helpers
    # ------------------------------------------------------------------

    def _kv_cache_memory_id(self, kind: str, chunk_id: Optional[int] = None) -> int:
        if kind == "clean":
            if chunk_id is None:
                raise ValueError("`chunk_id` is required for clean cache ids.")
            return int(chunk_id)
        if kind == "ref_image":
            return self._KV_CACHE_ID_REF_IMAGE
        raise ValueError(f"Unsupported cache kind: {kind!r}")

    @staticmethod
    def _get_chunk_windows(
        total_latent_frames: int,
        chunk_size: int,
        window_size: int,
        global_sink_chunk: bool,
    ) -> List[Dict[str, Any]]:
        if window_size <= 0:
            raise ValueError(f"`window_size` must be positive, got {window_size}.")

        windows = []
        num_chunks = (total_latent_frames + chunk_size - 1) // chunk_size
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(total_latent_frames, chunk_start + chunk_size)
            if global_sink_chunk and chunk_idx > 0:
                tail_window_size = max(window_size - 1, 1)
                tail_chunk_start = max(1, chunk_idx - tail_window_size + 1)
                selected_chunk_ids = [0] + list(range(tail_chunk_start, chunk_idx + 1))
            else:
                window_chunk_start = max(0, chunk_idx - window_size + 1)
                selected_chunk_ids = list(range(window_chunk_start, chunk_idx + 1))

            windows.append(
                {
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "selected_chunk_ids": selected_chunk_ids,
                }
            )
        return windows

    @staticmethod
    def _chunk_frame_bounds(chunk_id: int, chunk_size: int, total_latent_frames: int) -> Tuple[int, int]:
        chunk_start = chunk_id * chunk_size
        chunk_end = min(total_latent_frames, chunk_start + chunk_size)
        return chunk_start, chunk_end

    @classmethod
    def _gather_window_temporal_ids(
        cls,
        selected_chunk_ids: List[int],
        chunk_size: int,
        total_latent_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        temporal_ids = []
        offset = 0
        for cid in selected_chunk_ids:
            frame_start, frame_end = cls._chunk_frame_bounds(cid, chunk_size, total_latent_frames)
            chunk_len = frame_end - frame_start
            temporal_ids.append(torch.arange(offset, offset + chunk_len, device=device, dtype=torch.long))
            offset += chunk_len
        return torch.cat(temporal_ids, dim=0)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def check_inputs(
        self,
        video,
        prompt,
        height,
        width,
        num_inference_steps,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        chunk_size=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if not isinstance(video, list) or len(video) == 0:
            raise ValueError("`video` must be a non-empty list of PIL images.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError("`callback_on_step_end_tensor_inputs` has invalid keys.")

        if height <= 0 or width <= 0:
            raise ValueError(f"`height` and `width` must be positive but are {height} and {width}.")
        if height % self.height_multiple != 0 or width % self.width_multiple != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by {self.height_multiple} and {self.width_multiple} but "
                f"are {height} and {width}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        if prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        if prompt is not None and not isinstance(prompt, (str, list)):
            raise ValueError("`prompt` has to be of type `str` or `list`.")
        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError("If `prompt_embeds` are provided, `prompt_embeds_mask` is required.")

        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"`chunk_size` must be positive when provided, got {chunk_size}.")
        if not isinstance(num_inference_steps, int) or num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be a positive integer, got {num_inference_steps}.")

    # ------------------------------------------------------------------
    # Pipeline properties
    # ------------------------------------------------------------------

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        video: List[PIL.Image.Image] = None,
        prompt: Union[str, List[str]] = None,
        ref_image: Optional[PipelineImageInput] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 2,
        chunk_size: Optional[int] = None,
        local_window_size: Optional[int] = None,
        global_sink_chunk: Optional[bool] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 1024,
    ):
        r"""
        The call method of the pipeline for chunk-wise causal video editing.

        Args:
            video (`List[PIL.Image.Image]`):
                The source video to edit, as a sequence of frames. If its length does not satisfy
                `temporal_compression_ratio * n + 1` for some integer `n`, trailing frames are truncated with a
                warning.
            prompt (`str` or `List[str]`):
                The prompt describing the desired edited video.
            ref_image (`PipelineImageInput`, *optional*):
                An optional static reference image whose KV is prefilled into the cache and attended to by every chunk
                (used to inject appearance/identity conditioning).
            height (`int`, *optional*):
                The height in pixels of the generated video. Defaults to the source video height and is adjusted down
                to the nearest valid spatial multiple when needed.
            width (`int`, *optional*):
                The width in pixels of the generated video. Defaults to the source video width and is adjusted down to
                the nearest valid spatial multiple when needed.
            num_inference_steps (`int`, *optional*, defaults to 2):
                The number of denoising steps applied per chunk.
            chunk_size (`int`, *optional*):
                Number of latent frames denoised per chunk. Defaults to `transformer.config.chunk_size`.
            local_window_size (`int`, *optional*):
                Number of recent chunks each chunk attends to. Defaults to `transformer.config.local_window_size`.
            global_sink_chunk (`bool`, *optional*):
                Whether every chunk additionally attends to chunk 0 (a global "sink"). Defaults to
                `transformer.config.global_sink_chunk`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A generator to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents for the first chunk, sampled from a Gaussian distribution when not
                provided.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-computed text embeddings. When provided, `prompt` can be omitted.
            prompt_embeds_mask (`torch.Tensor`, *optional*):
                Attention mask for `prompt_embeds`.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated video. Choose between `"np"`, `"pt"`, `"pil"`, or `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~pipelines.joyvideoedit.JoyVideoEditPipelineOutput`] instead of a plain tuple.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A callback invoked at the end of each denoising step.
            callback_on_step_end_tensor_inputs (`List[str]`, *optional*, defaults to `["latents"]`):
                Tensor keys passed to `callback_on_step_end`.
            max_sequence_length (`int`, *optional*, defaults to 1024):
                Maximum sequence length for prompt encoding.

        Examples:

        Returns:
            [`~pipelines.joyvideoedit.JoyVideoEditPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, a [`~pipelines.joyvideoedit.JoyVideoEditPipelineOutput`] is returned,
                otherwise a `tuple` where the first element is the generated frames.
        """
        if not isinstance(video, list) or len(video) == 0:
            raise ValueError("`video` must be a non-empty list of PIL images.")
        height = height if height is not None else video[0].height
        width = width if width is not None else video[0].width
        if height < self.height_multiple or width < self.width_multiple:
            raise ValueError(
                f"`height` and `width` must be at least {self.height_multiple} and {self.width_multiple} but are "
                f"{height} and {width}."
            )
        adjusted_height = height // self.height_multiple * self.height_multiple
        adjusted_width = width // self.width_multiple * self.width_multiple
        if height != adjusted_height or width != adjusted_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({self.height_multiple}, {self.width_multiple}). "
                f"Adjusting ({height}, {width}) to ({adjusted_height}, {adjusted_width})."
            )
            height, width = adjusted_height, adjusted_width

        self.check_inputs(
            video,
            prompt,
            height,
            width,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            chunk_size=chunk_size,
            num_inference_steps=num_inference_steps,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

        self._interrupt = False
        device = self._execution_device
        transformer_dtype = self.transformer.dtype

        chunk_size = chunk_size if chunk_size is not None else self.transformer.config.chunk_size
        local_window_size = (
            local_window_size if local_window_size is not None else self.transformer.config.local_window_size
        )
        global_sink_chunk = (
            global_sink_chunk if global_sink_chunk is not None else self.transformer.config.global_sink_chunk
        )
        if chunk_size is None or chunk_size <= 0:
            raise ValueError(f"`chunk_size` must resolve to a positive value, got {chunk_size}.")

        # 1. Encode prompt together with the video's first frame as an image anchor (no CFG, so no negative prompt).
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt=prompt,
            image=video[0] if video is not None else None,
            device=device,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            max_sequence_length=max_sequence_length,
        )
        prompt_embeds = prompt_embeds.to(transformer_dtype)

        # 2. Encode each conditioning latent frame from a causal pixel window and keep the window's final latent.
        video_tensor = self.video_processor.preprocess_video(video, height=height, width=width)
        video_tensor = video_tensor.to(device=device, dtype=self.vae.dtype)

        ffactor_temporal = self.vae_scale_factor_temporal
        total_pixel_frames = video_tensor.shape[2]
        valid_num_frames = (total_pixel_frames - 1) // ffactor_temporal * ffactor_temporal + 1
        if total_pixel_frames != valid_num_frames:
            logger.warning(
                f"Video contains {total_pixel_frames} frames, but its length must be of the form "
                f"`k * {ffactor_temporal} + 1`. Truncating to {valid_num_frames} frames."
            )
            video_tensor = video_tensor[:, :, :valid_num_frames]
            total_pixel_frames = valid_num_frames
        # The window size is fixed by the transformer's configured `chunk_size`, independent of any per-call
        # `chunk_size` override that only affects the denoising chunk layout.
        vae_chunk_size = self.transformer.config.chunk_size
        window_pixels = vae_chunk_size * ffactor_temporal
        window_frames = 1 + window_pixels
        stride = ffactor_temporal
        num_latents = (total_pixel_frames - 1) // stride + 1

        latent_frames = []
        for k in range(num_latents):
            if k == 0:
                window = video_tensor[:, :, :1]
            else:
                end_frame = k * stride
                start_frame = max(0, end_frame - window_pixels)
                window = video_tensor[:, :, start_frame : end_frame + 1]
                pad_needed = window_frames - window.shape[2]
                if pad_needed > 0:
                    pad = video_tensor[:, :, :1].expand(-1, -1, pad_needed, -1, -1)
                    window = torch.cat([pad, window], dim=2)
                    del pad
            window_latents = self.vae.encode(window).latent_dist.sample(generator=generator)
            latent_frames.append(window_latents[:, :, -1:])
        ref_video_latents = torch.cat(latent_frames, dim=2)
        ref_video_latents = self.normalize_latents(ref_video_latents).to(device=device, dtype=transformer_dtype)
        del video_tensor, latent_frames, window, window_latents

        # A single source video conditions every prompt in the batch, so broadcast its latents to the number of
        # prompts (`prompt=["edit A", "edit B"]` edits the same video two ways).
        num_prompts = prompt_embeds.shape[0]
        if ref_video_latents.shape[0] == 1 and num_prompts > 1:
            ref_video_latents = ref_video_latents.repeat(num_prompts, 1, 1, 1, 1)

        batch_size, latent_channels, total_latent_frames, latent_height, latent_width = ref_video_latents.shape

        # 3. Optionally prefill the KV cache with a static reference image.
        self.transformer._reset_stateful_cache()
        ref_image_kv_prefilled = False
        try:
            if ref_image is not None:
                ref_pixels = self.video_processor.preprocess(
                    ref_image,
                    height=latent_height * self.vae_scale_factor_spatial,
                    width=latent_width * self.vae_scale_factor_spatial,
                )
                ref_pixels = ref_pixels.unsqueeze(2).to(device=device, dtype=self.vae.dtype)  # (B, C, 1, H, W)
                reference_image_latents = self.vae.encode(ref_pixels).latent_dist.sample(generator=generator)
                reference_image_latents = self.normalize_latents(reference_image_latents)
                reference_image_latents = reference_image_latents[:, :, :1].to(device=device, dtype=transformer_dtype)
                if reference_image_latents.shape[0] == 1 and batch_size > 1:
                    reference_image_latents = reference_image_latents.repeat(batch_size, 1, 1, 1, 1)
                ref_frames = reference_image_latents.shape[2]
                with self.transformer.cache_context("inference"):
                    self.transformer(
                        hidden_states=reference_image_latents,
                        timestep=torch.zeros(
                            (reference_image_latents.shape[0],), device=device, dtype=transformer_dtype
                        ),
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        current_temporal_ids=torch.zeros(
                            (reference_image_latents.shape[0], ref_frames), device=device, dtype=torch.long
                        ),
                        kv_cache_mode="store",
                        kv_cache_chunk_id=self._kv_cache_memory_id("ref_image"),
                        kv_cache_selected_chunk_ids=[],
                        self_attn_input_mode=SELF_ATTN_MODE_REF_IMAGE_CACHE,
                        skip_text_stream=True,
                        return_dict=False,
                    )
                ref_image_kv_prefilled = True

            # 4. Set up chunk-wise causal denoising while cache cleanup is still guarded.
            windows = self._get_chunk_windows(total_latent_frames, chunk_size, local_window_size, global_sink_chunk)
            num_chunks = len(windows)
            self._num_timesteps = num_inference_steps

            # The first chunk may be seeded with user-provided `latents`; later chunks always start from fresh noise.
            initial_latents = latents
            raw_sigmas = torch.linspace(1, 0, num_inference_steps + 1)[:-1].numpy()
            chunk_outputs = []
        except Exception:
            self.transformer._reset_stateful_cache()
            raise

        try:
            for chunk_idx, window in enumerate(self.progress_bar(windows)):
                chunk_start = window["chunk_start"]
                chunk_end = window["chunk_end"]
                selected_chunk_ids = window["selected_chunk_ids"]
                history_chunk_ids = selected_chunk_ids[:-1]
                active_chunk_id = selected_chunk_ids[-1]
                current_chunk_len = chunk_end - chunk_start

                window_ids = self._gather_window_temporal_ids(
                    selected_chunk_ids, chunk_size, total_latent_frames, device
                )
                current_temporal_ids = window_ids[-current_chunk_len:]
                cached_temporal_ids = window_ids[:-current_chunk_len]
                if cached_temporal_ids.numel() == 0:
                    cached_temporal_ids = None

                ref_chunk_latent = ref_video_latents[:, :, chunk_start:chunk_end]

                noise_shape = (batch_size, latent_channels, current_chunk_len, latent_height, latent_width)
                # Keep Euler updates in float32 and cast only the transformer input.
                if chunk_idx == 0 and initial_latents is not None:
                    latents = initial_latents.to(device=device, dtype=torch.float32)
                else:
                    latents = randn_tensor(noise_shape, generator=generator, device=device, dtype=torch.float32)

                cache_memory_ids = [self._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids]
                if ref_image_kv_prefilled:
                    cache_memory_ids.append(self._kv_cache_memory_id("ref_image"))

                current_ids_batched = current_temporal_ids.unsqueeze(0).expand(batch_size, -1)
                cached_ids_batched = (
                    cached_temporal_ids.unsqueeze(0).expand(batch_size, -1)
                    if cached_temporal_ids is not None
                    else None
                )

                # Reset the scheduler for each independently denoised chunk.
                self.scheduler.set_timesteps(sigmas=raw_sigmas, device=device)
                timesteps = self.scheduler.timesteps
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue
                    t_expand = t.repeat(latents.shape[0])
                    with self.transformer.cache_context("inference"):
                        noise_pred = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=t_expand,
                            encoder_hidden_states=prompt_embeds,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            ref_video_latent=ref_chunk_latent,
                            current_temporal_ids=current_ids_batched,
                            cached_temporal_ids=cached_ids_batched,
                            kv_cache_mode="reuse",
                            kv_cache_chunk_id=active_chunk_id,
                            kv_cache_selected_chunk_ids=cache_memory_ids,
                            kv_cache_pre_rope=True,
                            return_dict=False,
                        )[0]
                    # Upcast the model output to keep the scheduler accumulator in float32.
                    latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

                    if callback_on_step_end is not None:
                        callback_kwargs = {}
                        for k in callback_on_step_end_tensor_inputs:
                            callback_kwargs[k] = locals()[k]
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # Evict every cached chunk that neither the current history nor the reference image needs, then store
                # this chunk's clean (denoised) KV so later chunks can attend to it.
                keep_before_store = {self._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
                if ref_image_kv_prefilled:
                    keep_before_store.add(self._kv_cache_memory_id("ref_image"))

                # Keep only the chunks the next window will attend to (plus the reference image).
                next_selected = windows[chunk_idx + 1]["selected_chunk_ids"] if chunk_idx + 1 < num_chunks else []
                keep_after_store = {self._kv_cache_memory_id("clean", cid) for cid in next_selected}
                if ref_image_kv_prefilled:
                    keep_after_store.add(self._kv_cache_memory_id("ref_image"))

                # `evict_kv_cache_chunks` and the store-mode forward both read/write cache state through the KV-cache
                # hook's `StateManager`, which requires an active context.
                with self.transformer.cache_context("inference"):
                    self.transformer.evict_kv_cache_chunks(keep_before_store)
                    self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=torch.zeros((latents.shape[0],), device=device, dtype=transformer_dtype),
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        current_temporal_ids=current_ids_batched,
                        kv_cache_mode="store",
                        kv_cache_chunk_id=self._kv_cache_memory_id("clean", active_chunk_id),
                        kv_cache_selected_chunk_ids=[],
                        kv_cache_pre_rope=True,
                        skip_text_stream=True,
                        return_dict=False,
                    )
                    self.transformer.evict_kv_cache_chunks(keep_after_store)

                chunk_outputs.append(latents)
        finally:
            # Never let KV-cache state leak into the next `__call__`.
            self.transformer._reset_stateful_cache()

        latents = torch.cat(chunk_outputs, dim=2)
        del chunk_outputs, ref_video_latents, ref_chunk_latent, prompt_embeds, prompt_embeds_mask, initial_latents

        if output_type == "latent":
            video = latents
        else:
            # Decode each chunk causally. Subsequent chunks prepend a latent encoded from the previous output frame.
            latents = self.denormalize_latents(latents.to(self.vae.dtype))
            total_decoded_frames = 1 + (latents.shape[2] - 1) * ffactor_temporal
            previous_frame = None
            video = None
            output_frame_start = 0
            for frame_start in range(0, latents.shape[2], chunk_size):
                chunk_latents = latents[:, :, frame_start : frame_start + chunk_size]
                if frame_start == 0:
                    decoded = self.vae.decode(chunk_latents, return_dict=False)[0]
                else:
                    pseudo_latent = self.vae.encode(previous_frame).latent_dist.sample(generator=generator)
                    decoded = self.vae.decode(torch.cat([pseudo_latent, chunk_latents], dim=2), return_dict=False)[0]
                    decoded = decoded[:, :, -chunk_latents.shape[2] * ffactor_temporal :]

                previous_frame = decoded[:, :, -1:].clone()
                num_decoded_frames = decoded.shape[2]
                decoded = self.video_processor.postprocess_video(decoded, output_type=output_type)
                if output_type == "pt":
                    decoded = decoded.cpu()

                if output_type == "pil":
                    if video is None:
                        video = [[] for _ in range(len(decoded))]
                    for batch_idx, frames in enumerate(decoded):
                        video[batch_idx].extend(frames)
                else:
                    if video is None:
                        output_shape = (decoded.shape[0], total_decoded_frames, *decoded.shape[2:])
                        if output_type == "np":
                            video = np.empty(output_shape, dtype=decoded.dtype)
                        else:
                            video = torch.empty(output_shape, dtype=decoded.dtype, device="cpu")
                    video[:, output_frame_start : output_frame_start + num_decoded_frames] = decoded
                output_frame_start += num_decoded_frames

            if output_type != "pil":
                video = video[:, :output_frame_start]

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)
        return JoyVideoEditPipelineOutput(frames=video)
