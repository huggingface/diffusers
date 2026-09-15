# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import numpy as np
import torch
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from ...models import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from ...utils import logging
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam
from .modular_pipeline import MiniMaxH3ModularPipeline
from .references import MiniMaxH3Reference


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_qwen3vl_prompt_embeds(
    text_encoder,
    processor,
    token_ids: list[int],
    vision_inputs: dict | None = None,
    text_encoder_layer: int = 50,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    r"""
    Encode one tokenized MiniMax-H3 presentation with the Qwen3-VL conditioner.

    Unlike its `get_<encoder>_prompt_embeds` siblings this takes the *tokenized* presentation rather than a prompt
    string: MiniMax-H3's tokenization is the presentation-building phase that precedes it, where every reference or
    keyframe prepends a label and a vision block. The `mm_token_type_ids` built here are Qwen-internal (`0` text, `1`
    image, `2` video — they drive Qwen3-VL's per-modality-run rotary layout and never leave this function) and are not
    MiniMax-H3's own per-row modality tags, which the presentation phase produces alongside `token_ids`.

    Args:
        text_encoder (`Qwen3VLForConditionalGeneration`): The conditioner.
        processor (`Qwen3VLProcessor`): Its processor, which derives the token type ids from the vision pad ids.
        token_ids (`list[int]`): The tokenized presentation.
        vision_inputs (`dict`, *optional*):
            The vision tensors of the presentation's blocks, by the conditioner's own parameter names — `pixel_values`
            / `image_grid_thw` for images, `pixel_values_videos` / `video_grid_thw` for videos.
        text_encoder_layer (`int`, *optional*, defaults to 50):
            Which hidden state conditions the transformer, i.e. `components.text_encoder_layer`.
        device (`torch.device`, *optional*): The device to run the conditioner on.
        dtype (`torch.dtype`, *optional*): The dtype of the returned embeddings.

    Returns:
        `torch.Tensor`: the `(1, num_text_tokens, 5120)` hidden state after decoder layer `text_encoder_layer`.
    """
    num_layers = text_encoder.config.text_config.num_hidden_layers
    if num_layers <= text_encoder_layer:
        raise ValueError(
            f"MiniMax-H3 conditions on `hidden_states[{text_encoder_layer}]` of its Qwen3-VL "
            f"conditioner, which needs more than {text_encoder_layer} decoder layers, but "
            f"`text_encoder` has {num_layers}. The last hidden state of a stack truncated to exactly "
            f"{text_encoder_layer} layers is post-norm and is not the conditioning MiniMax-H3 expects."
        )

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)

    vision_kwargs = {}
    for name, value in (vision_inputs or {}).items():
        vision_kwargs[name] = value.to(device, text_encoder.dtype) if name.startswith("pixel_") else value.to(device)

    # `text_encoder.model` is a submodule, and a CPU-offload hook — accelerate's or the one the
    # `ComponentsManager` attaches — wraps the *top-level* module's `forward` alone, so calling the submodule
    # directly would leave the conditioner on the CPU. Fire the hook by hand instead of routing through
    # `text_encoder(...)`: MiniMax-H3 reads `hidden_states[50]` and never uses the language-model head, whose
    # vocabulary-wide projection over every token is all the top-level forward would add.
    # TODO: firing another module's offload hook by hand is not something a block should do. It is here
    # because MiniMax-H3 reads `hidden_states[50]` off `text_encoder.model` while the hook wraps only the
    # top-level `forward`. Needs a real answer — an opt-in on the conditioner, or a hook that follows
    # submodule calls.
    hook = getattr(text_encoder, "_hf_hook", None)
    if hook is not None and hasattr(hook, "pre_forward"):
        hook.pre_forward(text_encoder)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        use_cache=False,
        output_hidden_states=True,
        **vision_kwargs,
    )
    return outputs.hidden_states[text_encoder_layer].to(device=device, dtype=dtype)


def encode_vae_condition(
    vae, pixels: torch.Tensor, pixel_mean: tuple, pixel_std: tuple, encode_seed: int = 42
) -> torch.Tensor:
    r"""
    Encode one visual condition — a keyframe or a `ref2va` reference — into normalized conditioning latents.

    The recipe is the released model's, and every part of it is needed to reproduce its conditioning: the pixels are
    ImageNet-normalized, the posterior is *sampled* under a fresh generator seeded independently of the request, and
    the sampled latent is rounded to float16 (~11 bits of every conditioning latent) before being normalized. A single
    frame is encoded by the (tiled) spatial encoder alone; a frame stack goes through the temporal chunking, which is
    what turns `17 * n + 5` frames into `5 * n + 2` latent frames.

    Args:
        vae (`AutoencoderKLMiniMaxH3`): The video VAE.
        pixels (`torch.Tensor` of shape `(1, 3, num_frames, height, width)`):
            The condition's `uint8` pixels, on the VAE's device. `num_frames` is 1 for a keyframe or an image
            reference.
        pixel_mean (`tuple[float, float, float]`), pixel_std (`tuple[float, float, float]`):
            The video VAE's pixel convention, i.e. `components.pixel_mean` / `components.pixel_std`.
        encode_seed (`int`, *optional*, defaults to 42):
            Seed the posterior is sampled under, i.e. `components.keyframe_encode_seed`.

    Returns:
        `torch.Tensor`: one `(1, latent_channels, num_latent_frames, latent_height, latent_width)` float32 CPU tensor.
    """
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    pixel_mean = torch.tensor(pixel_mean, device=pixels.device).view(1, -1, 1, 1, 1)
    pixel_std = torch.tensor(pixel_std, device=pixels.device).view(1, -1, 1, 1, 1)

    pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
    posterior = vae.encode(pixels, return_dict=False)[0]
    latents = posterior.sample(generator=torch.Generator().manual_seed(encode_seed))
    latents = latents.to(torch.float16).float().cpu()
    return (latents - latents_mean) / latents_std


class MiniMaxH3TextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes MiniMax-H3's presentation of a `t2va` request: the prompt verbatim, with no chat template and "
            "no special tokens. The checkpoint is guidance-distilled, so there is no negative prompt and no "
            "unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state MiniMax-H3 conditions on, of shape `(1, num_text_tokens, 5120)`, read after the "
                    "50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=(
                    "The per-row modality tag of every row of `prompt_embeds` — all text for this presentation."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )

        token_ids = components.tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"]

        # One conditioner call. The block emits the conditioner's dtype, as every other model does — it has no
        # denoiser of its own, since it is meant to run on its own.
        block_state.prompt_embeds = get_qwen3vl_prompt_embeds(
            components.text_encoder,
            components.processor,
            token_ids,
            {},
            text_encoder_layer=components.text_encoder_layer,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.full((len(token_ids),), components.text_tag, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3FL2VATextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes MiniMax-H3's presentation of a `fl2va` request: the prompt verbatim, preceded by a "
            '`"<Picture i>: "` label and a vision block per keyframe, with no chat template and no special tokens. '
            "The checkpoint is guidance-distilled, so there is no negative prompt and no unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam(
                name="keyframes",
                type_hint=list,
                description="The keyframes put onto the target canvas, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state MiniMax-H3 conditions on, of shape `(1, num_text_tokens, 5120)`, read after the "
                    "50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=(
                    "The per-row modality tag of every row of `prompt_embeds`; a vision block is tagged as video."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )
        tokenizer, processor = components.tokenizer, components.processor
        text_tag, video_tag = components.text_tag, components.video_tag

        # 1. Vision features of the keyframes, batched. The presentation is text-only without them.
        vision_inputs, image_grid_thw = {}, None
        if block_state.keyframes:
            vision = processor.image_processor(images=block_state.keyframes, return_tensors="pt")
            image_grid_thw = vision["image_grid_thw"]
            vision_inputs = {"pixel_values": vision["pixel_values"], "image_grid_thw": image_grid_thw}

        # 2. The presentation, tokenized: a `"<Picture i>: "` label and a vision block per keyframe, then the prompt
        # verbatim. The rows of a vision block are tagged as *video* rather than text, which is what the
        # transformer's AdaLN modulation keys off.
        token_ids, token_tags = [], []
        if block_state.keyframes:
            merge_size = processor.image_processor.merge_size**2
            for index in range(len(block_state.keyframes)):
                num_image_tokens = int(image_grid_thw[index].prod()) // merge_size
                label_ids = tokenizer(f"<Picture {index + 1}>: ", add_special_tokens=False)["input_ids"]
                vision_ids = (
                    [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                    + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * num_image_tokens
                    + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
                )
                token_ids += label_ids + vision_ids
                token_tags += [text_tag] * len(label_ids) + [video_tag] * len(vision_ids)
        prompt_ids = tokenizer(block_state.prompt, add_special_tokens=False)["input_ids"]
        token_ids += prompt_ids
        token_tags += [text_tag] * len(prompt_ids)

        # 3. One conditioner call. The block emits the conditioner's dtype, as every other model does — it has no
        # denoiser of its own, since it is meant to run on its own.
        block_state.prompt_embeds = get_qwen3vl_prompt_embeds(
            components.text_encoder,
            processor,
            token_ids,
            vision_inputs,
            text_encoder_layer=components.text_encoder_layer,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.tensor(token_tags, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3KeyframeVaeEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes the `fl2va` keyframes into conditioning latents. They become the anchors of the whole denoising "
            "loop, which only ever writes the generated rows, so they are never updated again — the prepare-latents "
            "step noises them to MiniMax-H3's conditioning level and packs them."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [ComponentSpec("vae", AutoencoderKLMiniMaxH3)]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="keyframes",
                type_hint=list,
                required=True,
                description="The keyframes put onto the target canvas, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The normalized video conditioning latents, one `(1, latent_channels, 1, latent_height, "
                    "latent_width)` tensor per keyframe, in packed order. One entry per condition is what the "
                    "prepare-latents step draws its noise against, so the list is the unit the request's generator "
                    "is consumed in."
                ),
            )
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        # A keyframe is a single frame, so the VAE runs its spatial encoder alone with none of the 17-frame
        # temporal chunking.
        block_state.condition_latents = [
            encode_vae_condition(
                components.vae,
                torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None, :, None],
                components.pixel_mean,
                components.pixel_std,
                components.keyframe_encode_seed,
            )
            for image in block_state.keyframes
        ]

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VATextEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    def __init__(self, video_sample_fps: float = 2.0):
        r"""
        Encode the presentation of a `ref2va` request.

        Args:
            video_sample_fps (`float`, defaults to 2.0):
                The rate the conditioner reads a reference video at: every `24 / video_sample_fps`-th of the normalized
                24 fps frames.
        """
        self.video_sample_fps = video_sample_fps
        super().__init__()

    @property
    def description(self) -> str:
        return (
            "Encodes MiniMax-H3's presentation of a `ref2va` request: a label per reference, numbered per modality "
            '(`"<Picture i>: "` plus a vision block, `"<Audio j>: "` alone, `"<Video k>: "` plus one timestamped '
            "vision block per merged frame pair), then the prompt verbatim. The checkpoint is guidance-distilled, so "
            "there is no negative prompt and no unconditional branch."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("text_encoder", Qwen3VLForConditionalGeneration),
            ComponentSpec("tokenizer", Qwen2TokenizerFast),
            ComponentSpec("processor", Qwen3VLProcessor),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam.template("prompt", description="The prompt to guide generation, a single string."),
            InputParam(
                name="normalized_references",
                type_hint=list[MiniMaxH3Reference],
                required=True,
                description="The references normalized by the setup step, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam.template(
                "prompt_embeds",
                description=(
                    "The hidden state MiniMax-H3 conditions on, of shape `(1, num_text_tokens, 5120)`, read after the "
                    "50th decoder layer of the Qwen3-VL conditioner."
                ),
            ),
            OutputParam(
                "text_token_tags",
                type_hint=torch.Tensor,
                description=(
                    "The per-row modality tag of every row of `prompt_embeds`; a vision block is tagged as video."
                ),
            ),
        ]

    @staticmethod
    def _sample_video_condition_frames(
        frames: np.ndarray, fps: float, sample_fps: float, temporal_patch: int
    ) -> tuple[list[np.ndarray], list[float]]:
        r"""
        Sample the frames the conditioner sees from a normalized reference video, and label their vision blocks.

        The conditioner reads a reference at `sample_fps`: every `fps / sample_fps`-th frame, deduplicated. Qwen3-VL
        then merges the sampled frames in groups of `temporal_patch` — repeating the last one when the count does not
        divide — and a merged group is labelled with the mean of its timestamps, which `"<{timestamp:.1f} seconds>"`
        renders with Python's round-half-to-even, so the first block of a 2 fps pair is `"<0.2 seconds>"` rather than
        `"<0.3 seconds>"`.

        Args:
            frames (`np.ndarray` of shape `(num_frames, height, width, 3)`): The normalized reference video.
            fps (`float`): The rate `frames` is at, i.e. `components.fps`.
            sample_fps (`float`): The rate the conditioner reads at, i.e. the block's `video_sample_fps`.
            temporal_patch (`int`): Qwen3-VL's temporal patch, read off the processor.

        Returns:
            `tuple[list[np.ndarray], list[float]]`: the sampled frames and one timestamp per vision block.
        """
        stride = fps / sample_fps
        indices, cursor = [], 0.0
        while round(cursor) < frames.shape[0]:
            if not indices or round(cursor) > indices[-1]:
                indices.append(round(cursor))
            cursor += stride
        if len(indices) < temporal_patch:
            minimum = round((temporal_patch - 1) * stride) + 1
            raise ValueError(
                f"A reference video is read at {sample_fps:g} fps and its sampled frames are merged in groups of "
                f"{temporal_patch}, so it must run at least {minimum} frames at {fps:g} fps "
                f"({minimum / fps:.2g} seconds), got {frames.shape[0]}."
            )

        timestamps = [index / sample_fps for index in range(len(indices))]
        timestamps += [timestamps[-1]] * (-len(timestamps) % temporal_patch)
        block_timestamps = [
            (timestamps[index] + timestamps[index + temporal_patch - 1]) / 2
            for index in range(0, len(timestamps), temporal_patch)
        ]
        return [frames[index] for index in indices], block_timestamps

    def _gather_vision_features(
        self, processor, references: list[MiniMaxH3Reference], fps: float
    ) -> tuple[dict, list[int], list[int], list[list[float]]]:
        r"""
        Run the references' pixels through the conditioner's processors, batched per modality.

        The vision tensors are batched per modality while the presentation is tokenized in request order; the two agree
        because the filtering here preserves relative order within each modality and Qwen3-VL fills the n-th pad *run*
        of a modality with the n-th entry of that modality's batch. Audio contributes nothing — a waveform never
        reaches the conditioner.

        Returns:
            `tuple`: the vision tensors by the conditioner's parameter names, the vision token count per image
            reference, the per-block token count per video reference, and the block timestamps per video reference.
        """
        merge_size = processor.image_processor.merge_size**2
        vision_inputs = {}

        image_token_counts = []
        images = [reference.image for reference in references if reference.kind == "image"]
        if images:
            image_features = processor.image_processor(images=images, return_tensors="pt")
            vision_inputs["pixel_values"] = image_features["pixel_values"]
            vision_inputs["image_grid_thw"] = image_features["image_grid_thw"]
            image_token_counts = [int(grid.prod()) // merge_size for grid in image_features["image_grid_thw"]]

        video_block_token_counts, video_block_timestamps = [], []
        videos = [reference for reference in references if reference.kind == "video"]
        if videos:
            temporal_patch = processor.video_processor.temporal_patch_size
            sampled = [
                self._sample_video_condition_frames(reference.frames, fps, self.video_sample_fps, temporal_patch)
                for reference in videos
            ]
            video_block_timestamps = [timestamps for _, timestamps in sampled]
            video_features = processor.video_processor(
                videos=[np.stack(frames) for frames, _ in sampled], do_sample_frames=False, return_tensors="pt"
            )
            vision_inputs["pixel_values_videos"] = video_features["pixel_values_videos"]
            vision_inputs["video_grid_thw"] = video_features["video_grid_thw"]
            video_block_token_counts = [
                int(grid[1]) * int(grid[2]) // merge_size for grid in video_features["video_grid_thw"]
            ]
            for timestamps, grid in zip(video_block_timestamps, video_features["video_grid_thw"]):
                if int(grid[0]) != len(timestamps):
                    raise ValueError(
                        f"The processor merged a reference video into {int(grid[0])} vision blocks, but MiniMax-H3 "
                        f"labels {len(timestamps)} of them."
                    )

        return vision_inputs, image_token_counts, video_block_token_counts, video_block_timestamps

    @staticmethod
    def _build_presentation(
        tokenizer,
        prompt: str,
        references: list[MiniMaxH3Reference],
        image_token_counts: list[int],
        video_block_token_counts: list[int],
        video_block_timestamps: list[list[float]],
        text_tag: int = 1,
        video_tag: int = 0,
    ) -> tuple[list[int], list[int]]:
        r"""
        Tokenize MiniMax-H3's presentation of a `ref2va` request.

        Every reference prepends a label, in packed order and numbered per modality: `"<Picture i>: "` plus a vision
        block for an image, `"<Audio j>: "` alone for audio — a waveform never reaches the conditioner — and `"<Video
        k>: "` plus one timestamped vision block per merged frame pair for a video. A video that carries sound is
        labelled `"<Audio j>: "` *before* `"<Video k>: "`, mirroring the order its rows are packed in. The prompt
        follows verbatim, with no chat template and no special tokens.

        Args:
            tokenizer (`Qwen2TokenizerFast`): Tokenizer of the conditioner.
            prompt (`str`): The prompt, appended verbatim.
            references (`list[MiniMaxH3Reference]`): The normalized references, in packed order.
            image_token_counts (`list[int]`): Number of vision tokens of every image reference's block.
            video_block_token_counts (`list[int]`): Number of vision tokens per block of every video reference.
            video_block_timestamps (`list[list[float]]`): The timestamp of every vision block, per video reference.
            text_tag (`int`, *optional*, defaults to 1): MiniMax-H3's modality tag for a text row.
            video_tag (`int`, *optional*, defaults to 0): MiniMax-H3's modality tag for a vision block's rows.

        Returns:
            `tuple[list[int], list[int]]`: the token ids and their modality tags.
        """

        def text(value: str) -> tuple[list[int], list[int]]:
            token_ids = tokenizer(value, add_special_tokens=False)["input_ids"]
            return token_ids, [text_tag] * len(token_ids)

        def vision(pad_token: str, num_tokens: int) -> tuple[list[int], list[int]]:
            token_ids = (
                [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
                + [tokenizer.convert_tokens_to_ids(pad_token)] * num_tokens
                + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
            )
            return token_ids, [video_tag] * len(token_ids)

        token_ids, token_tags = [], []

        def emit(segment: tuple[list[int], list[int]]) -> None:
            token_ids.extend(segment[0])
            token_tags.extend(segment[1])

        counts = {"image": 0, "video": 0, "audio": 0}
        for reference in references:
            if reference.has_audio:
                counts["audio"] += 1
                emit(text(f"<Audio {counts['audio']}>: "))
            if reference.kind == "image":
                counts["image"] += 1
                emit(text(f"<Picture {counts['image']}>: "))
                emit(vision("<|image_pad|>", image_token_counts[counts["image"] - 1]))
            elif reference.kind == "video":
                counts["video"] += 1
                emit(text(f"<Video {counts['video']}>: "))
                for timestamp in video_block_timestamps[counts["video"] - 1]:
                    # `"{:.1f}"` rounds half to even, so the mean of a 2 fps pair renders as "<0.2 seconds>".
                    emit(text(f"<{timestamp:.1f} seconds>"))
                    emit(vision("<|video_pad|>", video_block_token_counts[counts["video"] - 1]))
        emit(text(prompt))
        return token_ids, token_tags

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        if not isinstance(block_state.prompt, str):
            raise ValueError(
                "MiniMax-H3 packs one request into one sequence, so `prompt` must be a single string, got "
                f"{type(block_state.prompt)}."
            )
        references = block_state.normalized_references

        # 1. Vision features, batched per modality — audio contributes nothing to the conditioner.
        vision_inputs, image_token_counts, video_token_counts, video_timestamps = self._gather_vision_features(
            components.processor, references, components.fps
        )

        # 2. The presentation, tokenized in request order.
        token_ids, token_tags = self._build_presentation(
            components.tokenizer,
            block_state.prompt,
            references,
            image_token_counts,
            video_token_counts,
            video_timestamps,
            text_tag=components.text_tag,
            video_tag=components.video_tag,
        )

        # 3. One conditioner call. The block emits the conditioner's dtype, as every other model does — it has no
        # denoiser of its own, since it is meant to run on its own.
        block_state.prompt_embeds = get_qwen3vl_prompt_embeds(
            components.text_encoder,
            components.processor,
            token_ids,
            vision_inputs,
            text_encoder_layer=components.text_encoder_layer,
            device=components._execution_device,
            dtype=components.text_encoder.dtype,
        )
        block_state.text_token_tags = torch.tensor(token_tags, dtype=torch.long)

        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3Ref2VAReferenceEncoderStep(ModularPipelineBlocks):
    model_name = "minimax-h3"

    @property
    def description(self) -> str:
        return (
            "Encodes the `ref2va` references — image and video references through the video VAE, soundtracks through "
            "the audio VAE. They are the anchors of the whole denoising loop, which only ever writes the generated "
            "rows; the prepare-latents step noises the visual ones to MiniMax-H3's conditioning level and packs them, "
            "while soundtracks ride along clean at `t = 1.0`. The latent geometry of every reference is the shape of "
            "what this emits, which is what the packed layout is built from."
        )

    @property
    def expected_components(self) -> list[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLMiniMaxH3),
            ComponentSpec("audio_vae", AutoencoderKLMiniMaxH3Audio),
        ]

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam(
                name="normalized_references",
                type_hint=list[MiniMaxH3Reference],
                required=True,
                description="The references normalized by the setup step, in packed order.",
            ),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The encoded video conditioning latents of the image and video references, one `(1, "
                    "latent_channels, num_latent_frames, latent_height, latent_width)` tensor each in packed order, "
                    "or None when the references carry none."
                ),
            ),
            OutputParam(
                "audio_condition_latents",
                type_hint=list[torch.Tensor],
                description=(
                    "The clean audio conditioning rows of the reference soundtracks, one `(num_audio_latents * 2, "
                    "audio_latent_channels)` tensor per audio-bearing reference in packed order. One entry per "
                    "reference rather than one concatenated block, because the packed layout is built from the row "
                    "count of each."
                ),
            ),
        ]

    @torch.no_grad()
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        device = components._execution_device

        audio_latents_mean = torch.tensor(components.audio_vae.config.latents_mean).view(1, 1, -1)
        audio_latents_std = torch.tensor(components.audio_vae.config.latents_std).view(1, 1, -1)

        condition_latents, audio_condition_latents = [], []
        for reference in block_state.normalized_references:
            if reference.kind == "image":
                pixels = torch.from_numpy(np.array(reference.image)).to(device).permute(2, 0, 1)[None, :, None]
                condition_latents.append(
                    encode_vae_condition(
                        components.vae,
                        pixels,
                        components.pixel_mean,
                        components.pixel_std,
                        components.keyframe_encode_seed,
                    )
                )
            elif reference.kind == "video":
                # Snap *down* to `17 * n + 5` so the VAE encodes without padding; this only bites when the reference
                # is shorter than the target, whose own frame count already has that form.
                frames_per_chunk = components.vae_frames_per_chunk
                latents_per_chunk = components.vae_latents_per_chunk
                num_frames = reference.frames.shape[0]
                num_frames = (
                    max(1, (num_frames - latents_per_chunk) // frames_per_chunk) * frames_per_chunk + latents_per_chunk
                )
                pixels = torch.from_numpy(reference.frames[:num_frames].copy()).to(device).permute(3, 0, 1, 2)[None]
                condition_latents.append(
                    encode_vae_condition(
                        components.vae,
                        pixels,
                        components.pixel_mean,
                        components.pixel_std,
                        components.keyframe_encode_seed,
                    )
                )

            if reference.has_audio:
                posterior = components.audio_vae.encode(reference.audio.to(device)[:, None], return_dict=False)[0]
                # Channel-major rows: the two stereo channels are two batch items of the mono audio VAE. Soundtracks
                # take the posterior *mean*, and are never sampled.
                latents = posterior.mode().float().cpu().transpose(1, 2)
                normalized = (latents - audio_latents_mean) / audio_latents_std
                audio_condition_latents.append(normalized.reshape(-1, components.audio_latent_channels))

        block_state.condition_latents = condition_latents
        block_state.audio_condition_latents = audio_condition_latents

        self.set_block_state(state, block_state)
        return components, state
