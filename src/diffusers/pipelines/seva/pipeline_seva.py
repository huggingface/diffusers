# Copyright 2024 PixArt-Sigma Authors and The HuggingFace Team. All rights reserved.
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
import re
import warnings
from typing import List, Optional, Tuple, Union
import math
import copy
from tqdm import tqdm


import torch
from torch.nn import functional as F
from transformers import CLIPVisionModelWithProjection

from ...callbacks import MultiPipelineCallbacks, PipelineCallback
from .pipeline_output import SevaPipelineOutput
from ...image_processor import VaeImageProcessor
from ...models import AutoencoderKL, SevaUnet
from ...schedulers import DPMSolverMultistepScheduler
from ...utils import (
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
)
from ...utils.torch_utils import randn_tensor
from ..pipeline_utils import DiffusionPipeline


from .geometry import DEFAULT_FOV_RAD, get_preset_pose_fov, get_default_intrinsics
from .sampling import DiscreteDenoiser, DDPMDiscretization
from .infer_utils import (
    infer_prior_stats, 
    get_k_from_dict,
    get_value_dict, 
    chunk_input_and_test, 
    pad_indices, 
    create_samplers, 
    decode_output, 
    extend_dict,
    assemble,
    update_kv_for_dict
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_bs4_available():
    pass

if is_ftfy_available():
    pass


EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import SanaPipeline

        >>> pipe = SanaPipeline.from_pretrained(
        ...     "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.float32
        ... )
        >>> pipe.to("cuda")
        >>> pipe.text_encoder.to(torch.bfloat16)
        >>> pipe.transformer = pipe.transformer.to(torch.bfloat16)

        >>> image = pipe(prompt='a cyberpunk cat with a neon sign that says "Sana"')[0]
        >>> image[0].save("output.png")
        ```
"""

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class SevaPipeline(DiffusionPipeline):
    r"""Pipeline for text-to-image generation using [Seva]."""

    # fmt: off
    bad_punct_regex = re.compile(r"[" + "#®•©™&@·º½¾¿¡§~" + r"\)" + r"\(" + r"\]" + r"\[" + r"\}" + r"\{" + r"\|" + "\\" + r"\/" + r"\*" + r"]{1,}")
    # fmt: on

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        conditioner: CLIPVisionModelWithProjection,
        scheduler: DPMSolverMultistepScheduler,
        unet: SevaUnet,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            conditioner=conditioner,
            scheduler=scheduler,
            unet=unet,
        )

        self.vae_scale_factor = 64
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.discretization = DDPMDiscretization()
        self.denoiser = DiscreteDenoiser(
            discretization=self.discretization, num_idx=1000
        )

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        callback_on_step_end_tensor_inputs=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 32 but are {height} and {width}."
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds`."
            )

        if (
            negative_prompt_embeds is not None
            and negative_prompt_attention_mask is None
        ):
            raise ValueError(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds`."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_image(
        self,
        image,
        width,
        height,
        device,
        dtype,
    ):
        size_stride = 64
        height, width = (
            math.floor(height / size_stride + 0.5) * size_stride,
            math.floor(width / size_stride + 0.5) * size_stride,
        )
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(
                image, resize_mode="crop", height=height, width=width
            )
        image = image.to(device=device, dtype=dtype)
        return image

    def prepare_camera(
        self, image, preset_traj, num_frames, start_fov_rad=DEFAULT_FOV_RAD
    ):
        height, width = image.shape[2], image.shape[3]
        aspect_ratio = width / height
        input_cam_intrinsic = get_default_intrinsics(
            fov_rad=start_fov_rad,
            aspect_ratio=aspect_ratio,
        )
        input_c2ws = torch.eye(4)[None]

        start_c2w = input_c2ws[0]
        start_w2c = torch.linalg.inv(start_c2w)
        target_c2ws, target_fovs = get_preset_pose_fov(
            preset_traj=preset_traj,
            num_frames=num_frames,
            start_w2c=start_w2c,
            look_at=torch.tensor([0, 0, 10]),
            up_direction=-start_c2w[:3, 1],
            start_fov=start_fov_rad,
            spiral_radii=[1.0, 1.0, 0.5],
        )
        target_c2ws = torch.as_tensor(target_c2ws)
        target_fovs = torch.as_tensor(target_fovs)
        target_cam_intrinsic = get_default_intrinsics(
            fov_rad=target_fovs,
            aspect_ratio=aspect_ratio,
        )

        all_c2ws = torch.cat([input_c2ws, target_c2ws], 0)
        all_cam_intrinsic = (
            torch.cat([input_cam_intrinsic, target_cam_intrinsic], 0)
            * input_cam_intrinsic.new_tensor([width, height, 1])[:, None]
        )
        return all_c2ws, all_cam_intrinsic

    def get_camera_anchors(
        self, all_c2ws, all_cam_intrinsic, num_inputs, num_targets, version_dict
    ):
        num_anchors = infer_prior_stats(num_inputs, num_targets, version_dict)
        assert isinstance(num_anchors, int)

        anchor_indices = torch.linspace(
            num_inputs,
            num_inputs + num_targets - 1,
            num_anchors,
        ).tolist()

        anchor_c2ws = all_c2ws[[round(ind) for ind in anchor_indices]]
        anchor_Ks = all_cam_intrinsic[[round(ind) for ind in anchor_indices]]

        return anchor_c2ws, anchor_Ks, anchor_indices

    def get_conditioning(
        self,
        input_imgs,
        all_c2ws,
        all_cam_intrinsic,
        anchor_indices,
        num_inputs,
        num_targets,
    ):
        all_imgs_tensor = (
            F.pad(input_imgs, (0, 0, 0, 0, 0, 0, 0, num_targets), value=0.0)
            * 255.0
        ).to(torch.uint8)

        image_cond = {
            "img": all_imgs_tensor,
            "input_indices": list(range(num_inputs)),
            "prior_indices": anchor_indices,
        }
        camera_cond = {
            "c2w": all_c2ws,
            "K": all_cam_intrinsic,
            "input_indices": list(range(num_inputs + num_targets)),
        }
        return image_cond, camera_cond

    def do_sample(self, sampler, value_dict, version_dict, seq_len):
        imgs = value_dict["cond_frames"]
        input_masks = value_dict["cond_frames_mask"]
        pluckers = value_dict["plucker_coordinate"]

        num_samples = [1, seq_len]
        latents = F.pad(
            self.vae.encode(imgs[input_masks], version_dict["encoding_t"]),
            (0, 0, 0, 0, 0, 1),
            value=1.0,
        )

        c_crossattn = (
            self.conditioner(imgs[input_masks])
            .mean(0)
            .view(1, 1, -1)
            .repeat(seq_len, 1, 1)
        )
        uc_crossattn = torch.zeros_like(c_crossattn)
        c_replace = latents.new_zeros(seq_len, *latents.shape[1:])
        c_replace[input_masks] = latents
        uc_replace = torch.zeros_like(c_replace)
        c_concat = torch.cat(
            [
                input_masks.view(-1, 1, 1, 1).repeat(
                    1, 1, pluckers.shape[2], pluckers.shape[3]
                ),
                pluckers,
            ],
            1,
        )
        uc_concat = torch.cat(
            [pluckers.new_zeros(seq_len, 1, *pluckers.shape[-2:]), pluckers],
            1,
        )
        c_dense_vector = pluckers
        uc_dense_vector = c_dense_vector
        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": c_concat,
            "dense_vector": c_dense_vector,
        }
        uc = {
            "crossattn": uc_crossattn,
            "replace": uc_replace,
            "concat": uc_concat,
            "dense_vector": uc_dense_vector,
        }

        additional_model_inputs = {"num_frames": seq_len}
        additional_sampler_inputs = {
            "c2w": value_dict["c2w"],
            "K": value_dict["K"],
            "input_frame_mask": value_dict["cond_frames_mask"],
        }

        shape = (
            math.prod(num_samples),
            version_dict["C"],
            version_dict["H"] // version_dict["F"],
            version_dict["W"] // version_dict["F"],
        )
        randn = torch.randn(shape)

        samples_z = sampler(
            lambda input, sigma, c: self.denoiser(
                self.model,
                input,
                sigma,
                c,
                **additional_model_inputs,
            ),
            randn,
            scale=version_dict["options"]["cfg"],
            cond=c,
            uc=uc,
            **additional_sampler_inputs,
        )
        if samples_z is None:
            return
        samples = self.vae.decode(samples_z, version_dict["options"]["decoding_t"])
        return samples

    def first_pass(
        self,
        input_indices,
        input_imgs,
        input_c2ws,
        input_Ks,
        traj_prior_indices,
        traj_prior_imgs,
        traj_prior_c2ws,
        traj_prior_Ks,
        camera_cond,
        version_dict,
        task,
        samplers,
    ):
        T_first_pass = (
            version_dict["T"][0]
            if isinstance(version_dict["T"], (list, tuple))
            else version_dict["T"]
        )
        chunk_strategy_first_pass = version_dict["options"].get(
            "chunk_strategy_first_pass", "gt-nearest"
        )
        (
            _,
            input_inds_per_chunk,
            input_sels_per_chunk,
            prior_inds_per_chunk,
            prior_sels_per_chunk,
        ) = chunk_input_and_test(
            T_first_pass,
            input_c2ws,
            traj_prior_c2ws,
            input_indices,
            traj_prior_indices,
            options=version_dict["options"],
            task=task,
            chunk_strategy=chunk_strategy_first_pass,
            gt_input_inds=list(range(input_c2ws.shape[0])),
        )
        print(
            f"Two passes (first) - chunking with `{chunk_strategy_first_pass}` strategy: total "
            f"{len(input_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_prior_inds = []
        for i, (
            chunk_input_inds,
            chunk_input_sels,
            chunk_prior_inds,
            chunk_prior_sels,
        ) in tqdm(
            enumerate(
                zip(
                    input_inds_per_chunk,
                    input_sels_per_chunk,
                    prior_inds_per_chunk,
                    prior_sels_per_chunk,
                )
            ),
            total=len(input_inds_per_chunk),
            leave=False,
        ):
            (
                curr_input_sels,
                curr_prior_sels,
                curr_input_maps,
                curr_prior_maps,
            ) = pad_indices(
                chunk_input_sels,
                chunk_prior_sels,
                T=T_first_pass,
                padding_mode=version_dict["options"].get("t_padding_mode", "last"),
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_input_inds],
                    test=y[chunk_prior_inds],
                    input_maps=curr_input_maps,
                    test_maps=curr_prior_maps,
                )
                for x, y in zip(
                    [
                        torch.cat(
                            [
                                input_imgs,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat(
                            [
                                input_imgs_clip,
                                get_k_from_dict(all_samples, "samples-rgb").to(
                                    input_imgs.device
                                ),
                            ],
                            dim=0,
                        ),
                        torch.cat([input_c2ws, traj_prior_c2ws[all_prior_inds]], dim=0),
                        torch.cat([input_Ks, traj_prior_Ks[all_prior_inds]], dim=0),
                    ],  # procedually append generated prior views to the input views
                    [
                        traj_prior_imgs,
                        traj_prior_imgs_clip,
                        traj_prior_c2ws,
                        traj_prior_Ks,
                    ],
                )
            ]
            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_imgs_clip.to("cuda"),
                curr_input_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_first_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=version_dict["options"].get("camera_scale", 2.0),
            )
            samples = self.do_sample(
                sampler=samplers[1] if len(samplers) > 1 else samplers[0],
                value_dict=value_dict,
                version_dict=version_dict,
                seq_len=len(curr_imgs)
            )
            if samples is None:
                return
            samples = decode_output(
                samples, T_first_pass, chunk_prior_sels
            )  # decode into dict
            extend_dict(all_samples, samples)
            all_prior_inds.extend(chunk_prior_inds)
        

    def second_pass(
        self,
        input_indices,
        input_imgs,
        input_c2ws,
        input_Ks,
        traj_prior_indices,
        traj_prior_imgs,
        traj_prior_c2ws,
        traj_prior_Ks,
        test_indices,
        test_imgs,
        test_c2ws,
        test_Ks,
        camera_cond,
        version_dict,
        task,
        samplers,
    ):
        T_second_pass = (
            version_dict["T"][1]
            if isinstance(version_dict["T"], (list, tuple))
            else version_dict["T"]
        )
        assert (
            traj_prior_indices is not None
        ), "`prior_frame_indices` needs to be set if using 2-pass sampling."
        prior_argsort = torch.argsort(input_indices + traj_prior_indices).tolist()
        traj_prior_indices = torch.tensor(input_indices + traj_prior_indices)[prior_argsort].tolist()
        gt_input_inds = [prior_argsort.index(i) for i in range(input_c2ws.shape[0])]

        traj_prior_imgs = torch.cat(
            [input_imgs, get_k_from_dict(all_samples, "samples-rgb")], dim=0
        )[prior_argsort]
        traj_prior_imgs_clip = torch.cat(
            [
                input_imgs_clip,
                get_k_from_dict(all_samples, "samples-rgb"),
            ],
            dim=0,
        )[prior_argsort]
        traj_prior_c2ws = torch.cat([input_c2ws, traj_prior_c2ws], dim=0)[prior_argsort]
        traj_prior_Ks = torch.cat([input_Ks, traj_prior_Ks], dim=0)[prior_argsort]

        update_kv_for_dict(all_samples, "samples-rgb", traj_prior_imgs)
        update_kv_for_dict(all_samples, "samples-c2ws", traj_prior_c2ws)
        update_kv_for_dict(all_samples, "samples-intrinsics", traj_prior_Ks)

        chunk_strategy = version_dict["options"].get("chunk_strategy", "nearest")
        (
            _,
            prior_inds_per_chunk,
            prior_sels_per_chunk,
            test_inds_per_chunk,
            test_sels_per_chunk,
        ) = chunk_input_and_test(
            T_second_pass,
            traj_prior_c2ws,
            test_c2ws,
            traj_prior_indices,
            test_indices,
            options=version_dict["options"],
            task=task,
            chunk_strategy=chunk_strategy,
            gt_input_inds=gt_input_inds,
        )
        print(
            f"Two passes (second) - chunking with `{chunk_strategy}` strategy: total "
            f"{len(prior_inds_per_chunk)} forward(s) ..."
        )

        all_samples = {}
        all_test_inds = []
        for i, (
            chunk_prior_inds,
            chunk_prior_sels,
            chunk_test_inds,
            chunk_test_sels,
        ) in tqdm(
            enumerate(
                zip(
                    prior_inds_per_chunk,
                    prior_sels_per_chunk,
                    test_inds_per_chunk,
                    test_sels_per_chunk,
                )
            ),
            total=len(prior_inds_per_chunk),
            leave=False,
        ):
            (
                curr_prior_sels,
                curr_test_sels,
                curr_prior_maps,
                curr_test_maps,
            ) = pad_indices(
                chunk_prior_sels,
                chunk_test_sels,
                T=T_second_pass,
                padding_mode="last",
            )
            curr_imgs, curr_imgs_clip, curr_c2ws, curr_Ks = [
                assemble(
                    input=x[chunk_prior_inds],
                    test=y[chunk_test_inds],
                    input_maps=curr_prior_maps,
                    test_maps=curr_test_maps,
                )
                for x, y in zip(
                    [
                        traj_prior_imgs,
                        traj_prior_imgs_clip,
                        traj_prior_c2ws,
                        traj_prior_Ks,
                    ],
                    [test_imgs, test_imgs_clip, test_c2ws, test_Ks],
                )
            ]
            value_dict = get_value_dict(
                curr_imgs.to("cuda"),
                curr_imgs_clip.to("cuda"),
                curr_prior_sels,
                curr_c2ws,
                curr_Ks,
                list(range(T_second_pass)),
                all_c2ws=camera_cond["c2w"],
                camera_scale=version_dict["options"]["camera_scale"],
            )
            samples = self.do_sample(
                sampler=samplers[1] if len(samplers) > 1 else samplers[0],
                value_dict=value_dict,
                version_dict=version_dict,
                seq_len=T_second_pass
            )
            if samples is None:
                return
            samples = decode_output(
                samples, T_second_pass, chunk_test_sels
            )  # decode into dict
            extend_dict(all_samples, samples)
            all_test_inds.extend(chunk_test_inds)
        all_samples = {
            key: value[torch.argsort(all_test_inds)] for key, value in all_samples.items()
        }

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        input_imgs: torch.Tensor,
        height: int = 576,
        width: int = 576,
        max_sequence_length: int = 21,
        num_frames: int = 100,
        preset_traj: str = "spiral",
        num_steps: int = 50,
        conditioning_factor: float = 3.0,
        chunk_strategy: str = "interp",
        camera_scale: float = 2.0,
        task: str = "img2trajvid",
        # prompt: Union[str, List[str]] = None,
        # negative_prompt: str = "",
        # timesteps: List[int] = None,
        # sigmas: List[float] = None,
        # guidance_scale: float = 4.5,
        # num_images_per_prompt: Optional[int] = 1,
        # eta: float = 0.0,
        # generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        # latents: Optional[torch.Tensor] = None,
        # prompt_embeds: Optional[torch.Tensor] = None,
        # prompt_attention_mask: Optional[torch.Tensor] = None,
        # negative_prompt_embeds: Optional[torch.Tensor] = None,
        # negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        # output_type: Optional[str] = "pil",
        # return_dict: bool = True,
        # clean_caption: bool = False,
        # use_resolution_binning: bool = True,
        # attention_kwargs: Optional[Dict[str, Any]] = None,
        # callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        # callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # max_sequence_length: int = 300,
    ) -> SevaPipelineOutput:
        """
        Function invoked when calling the pipeline for generation.
        """

        batch_size = input_imgs.shape[0]
        assert batch_size == 1, "Batch size must be 1."

        device = self._execution_device
        input_imgs = self.prepare_image(
            input_imgs=input_imgs,
            width=width,
            height=height,
            device=device,
            dtype=self.vae.dtype,
        )

        # input + target
        all_c2ws, all_cam_intrinsic = self.prepare_camera(
            image=input_imgs,
            preset_traj=preset_traj,
            num_frames=num_frames,
        )

        VERSION_DICT = {
            "H": height,
            "W": width,
            "T": max_sequence_length,
            "C": self.vae.config.latent_channels,
            "f": self.image_processor.config.vae_scale_factor,
            "options": {},
        }
        options = {
            "num_steps": num_steps,
            "cfg": conditioning_factor,
            "chunk_strategy": chunk_strategy,
            "camera_scale": camera_scale,
            "video_save_fps": 30.0,
            "beta_linear_start": 5e-6,
            "log_snr_shift": 2.4,
            "guider_types": [1, 2],
            "num_steps": num_steps,
            "cfg_min": 1.2,
            "encoding_t": 1,
            "decoding_t": 1,
        }

        # anchor points based, not necessary equal to target
        anchor_c2ws, anchor_cam_intrinsic, anchor_indices = self.get_camera_anchors(
            all_c2ws=all_c2ws,
            all_cam_intrinsic=all_cam_intrinsic,
            num_inputs=input_imgs.shape[0],
            num_targets=all_c2ws.shape[0] - input_imgs.shape[0],
            version_dict=VERSION_DICT,
        )

        # image_cond --> all_ings(padded img for test), input_indices, anchor_indices
        # camera_cond --> all_c2ws, all_cam_intrinsics, all_indices
        image_cond, camera_cond = self.get_conditioning(
            input_imgs=input_imgs,
            all_c2ws=all_c2ws,
            all_cam_intrinsic=all_cam_intrinsic,
            anchor_indices=anchor_indices,
            num_inputs=input_imgs.shape[0],
            num_targets=all_c2ws.shape[0] - input_imgs.shape[0],
        )

        imgs = []
        for i, (img, K) in enumerate(zip(image_cond["img"], camera_cond["K"])):
            img = img.unsqueeze(0)
            img = self.image_processor.preprocess(
                img, resize_mode="crop", height=height, width=width
            )
            assert K is not None
            K[0] /= height
            K[1] /= width
            camera_cond["K"][i] = K
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)
        # imgs_clip = copy.deepcopy(imgs)

        for i, prior_k in enumerate(traj_prior_Ks):
            prior_k[0] /= height
            prior_k[1] /= width
            traj_prior_Ks[i] = prior_k

        # Get Data
        input_indices = image_cond["input_indices"]
        input_imgs = imgs[input_indices]
        input_c2ws = camera_cond["c2w"][input_indices]
        input_Ks = camera_cond["K"][input_indices]

        test_indices = [i for i in range(len(imgs)) if i not in input_indices]
        test_imgs = imgs[test_indices]
        test_c2ws = camera_cond["c2w"][test_indices]
        test_Ks = camera_cond["K"][test_indices]
        # test_imgs_clip = imgs_clip[test_indices]

        traj_prior_indices = anchor_indices
        traj_prior_imgs = imgs[[round(ind) for ind in anchor_indices]]
        traj_prior_c2ws = torch.as_tensor(anchor_c2ws)
        traj_prior_Ks = torch.as_tensor(anchor_cam_intrinsic)
        # traj_prior_imgs_clip = copy.deepcopy(test_imgs_clip)

        T_first_pass = (
            VERSION_DICT["T"][0]
            if isinstance(VERSION_DICT["T"], (list, tuple))
            else VERSION_DICT["T"]
        )
        T_second_pass = (
            VERSION_DICT["T"][1]
            if isinstance(VERSION_DICT["T"], (list, tuple))
            else VERSION_DICT["T"]
        )
        samplers = create_samplers(
            VERSION_DICT["options"]["guider_types"],
            self.discretization,
            [T_first_pass, T_second_pass],
            VERSION_DICT["options"]["num_steps"],
            VERSION_DICT["options"]["cfg_min"],
        )

        self.first_pass(
            input_indices,
            input_imgs,
            input_c2ws,
            input_Ks,
            traj_prior_indices,
            traj_prior_imgs,
            traj_prior_c2ws,
            traj_prior_Ks,
            camera_cond,
            VERSION_DICT,
            task,
            samplers
        )
        
        self.second_pass(
            self,
            input_indices,
            input_imgs,
            input_c2ws,
            input_Ks,
            traj_prior_indices,
            traj_prior_imgs,
            traj_prior_c2ws,
            traj_prior_Ks,
            test_indices,
            test_imgs,
            test_c2ws,
            test_Ks,
            camera_cond,
            VERSION_DICT,
            task,
            samplers,
        )
        

        # if not output_type == "latent":
        #     image = self.image_processor.postprocess(image, output_type=output_type)

        # # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image,)

        # return SanaPipelineOutput(images=image)
