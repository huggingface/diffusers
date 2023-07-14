# Copyright 2023 The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import List, Optional, Union

from PIL import Image
import numpy as np
import torch
from PIL.Image import Image
from transformers import CLIPTextModel, CLIPTokenizer

from ...models import PaellaVQModel
from ...schedulers import DDPMScheduler
from ...utils import BaseOutput, logging, randn_tensor
from ..pipeline_utils import DiffusionPipeline
from .modules import DiffNeXt, Prior, EfficientNetEncoder

# from .diffuzz import Diffuzz


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import WuerstchenPriorPipeline, WuerstchenGeneratorPipeline

        >>> prior_pipe = WuerstchenPriorPipeline.from_pretrained("kashif/wuerstchen-prior", torch_dtype=torch.float16).to("cuda")
        >>> gen_pipe = WuerstchenGeneratorPipeline.from_pretrain("kashif/wuerstchen-gen", torch_dtype=torch.float16).to("cuda")

        >>> prompt = "an image of a shiba inu, donning a spacesuit and helmet"
        >>> prior_output = pipe(prompt)
        >>> images = gen_pipe(prior_output.image_embeds, prior_output.text_embeds)
        ```
"""


default_inference_steps_c = {2 / 3: 20, 0.0: 10}
# default_inference_steps_c = {0.0: 60}
default_inference_steps_b = {0.0: 30}


@dataclass
class WuerstchenPriorPipelineOutput(BaseOutput):
    """
    Output class for WuerstchenPriorPipeline.

    Args:
        image_embeds (`torch.FloatTensor` or `np.ndarray`)
            Prior image embeddings for text prompt
        text_embeds (`torch.FloatTensor` or `np.ndarray`)
            Clip text embeddings for unconditional tokens
    """

    image_embeds: Union[torch.FloatTensor, np.ndarray]
    text_embeds: Union[torch.FloatTensor, np.ndarray]


@dataclass
class WuerstchenGeneratorPipelineOutput(BaseOutput):
    """
    Output class for WuerstchenPriorPipeline.

    Args:
        images (`torch.FloatTensor` or `np.ndarray`)
            Generated images for text prompt.
    """

    images: Union[torch.FloatTensor, np.ndarray]


class WuerstchenPriorPipeline(DiffusionPipeline):
    """
    Pipeline for generating image prior for Wuerstchen.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        prior ([`Prior`]):
            The canonical unCLIP prior to approximate the image embedding from the text embedding.
        text_encoder ([`CLIPTextModelWithProjection`]):
            Frozen text-encoder.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        prior: Prior,
        scheduler: DDPMScheduler,
    ) -> None:
        super().__init__()
        self.multiple = 128
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prior=prior,
            scheduler=scheduler,
        )
        # self.diffuzz = Diffuzz(device="cuda")
        self.register_to_config()

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
    ):
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        # get prompt text embeddings
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            text_input_ids = text_input_ids[:, : self.tokenizer.model_max_length]

        text_encoder_output = self.text_encoder(text_input_ids.to(device))

        text_encoder_hidden_states = text_encoder_output.last_hidden_state

        text_encoder_hidden_states = text_encoder_hidden_states.repeat_interleave(num_images_per_prompt, dim=0)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            negative_prompt_embeds_text_encoder_output = self.text_encoder(uncond_input.input_ids.to(device))

            uncond_text_encoder_hidden_states = negative_prompt_embeds_text_encoder_output.last_hidden_state

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method

            seq_len = uncond_text_encoder_hidden_states.shape[1]
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.repeat(1, num_images_per_prompt, 1)
            uncond_text_encoder_hidden_states = uncond_text_encoder_hidden_states.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            # done duplicates

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_encoder_hidden_states = torch.cat([text_encoder_hidden_states, uncond_text_encoder_hidden_states])

        return text_encoder_hidden_states

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.text_encoder, "_hf_hook"):
            return self.device
        for module in self.text_encoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    # @torch.no_grad()
    # def inference_loop(
    #     self, latents, steps, text_encoder_hidden_states, do_classifier_free_guidance, guidance_scale, generator
    # ):
    #     print(steps)
    #     print(steps[:-1])
    #     for i, t in enumerate(self.progress_bar(steps[:-1])):
    #         # print(torch.cat([latents] * 2).shape, latents.dtype)
    #         # print(t.expand(latents.size(0) * 2).shape, t.dtype)
    #         # print(text_encoder_hidden_states.shape, text_encoder_hidden_states.dtype)
    #         predicted_image_embedding = self.prior(
    #             torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
    #             r=t.expand(latents.size(0) * 2) if do_classifier_free_guidance else t,
    #             c=text_encoder_hidden_states,
    #         )

    #         # print(t.expand(latents.size(0) * 2))
    #         # print(i, predicted_image_embedding[0, 0, :4, :4])
    #         # print(text_encoder_hidden_states[0, 4, :4])

    #         if do_classifier_free_guidance:
    #             predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
    #             predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
    #                 predicted_image_embedding_text - predicted_image_embedding_uncond
    #             )
    #         # print(t)

    #         # latents = self.diffuzz.undiffuse(latents, t[None], steps[i + 1][None], predicted_image_embedding).to(
    #         #     dtype=t.dtype
    #         # )
    #         timestep = (t * 999).cpu().int()
    #         # print(timestep)
    #         latents = self.scheduler.step(
    #             predicted_image_embedding,
    #             timestep=timestep - 1,
    #             sample=latents,
    #             generator=generator,
    #         ).prev_sample

    #     return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        inference_steps: dict = None,
        guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if inference_steps is None:
            inference_steps = default_inference_steps_c

        if negative_prompt is None:
            negative_prompt = ""

        if isinstance(prompt, str):
            prompt = [prompt]
        elif not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        elif not isinstance(negative_prompt, list) and negative_prompt is not None:
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        text_encoder_hidden_states = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        dtype = text_encoder_hidden_states.dtype
        latent_height = 128 * (height // 128) // (1024 // 24)
        latent_width = 128 * (width // 128) // (1024 // 24)
        effnet_features_shape = (num_images_per_prompt, 16, latent_height, latent_width)

        total_num_inference_steps = sum(inference_steps.values())
        self.scheduler.set_timesteps(total_num_inference_steps, device=device)
        prior_timesteps_tensor = self.scheduler.timesteps

        # def seed_everything(seed: int):
        #     import random, os
        #     import numpy as np
        #     import torch

        #     random.seed(seed)
        #     os.environ["PYTHONHASHSEED"] = str(seed)
        #     np.random.seed(seed)
        #     torch.manual_seed(seed)
        #     torch.cuda.manual_seed(seed)
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = True

        # seed_everything(42)

        latents = self.prepare_latents(
            effnet_features_shape,
            dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        #  latents = torch.randn(effnet_features_shape, device=device)
        # print(latents[0, 0, :4, :4])
        # latents = latents.to(dtype=dtype)

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            ratio = (t / self.scheduler.config.num_train_timesteps).to(dtype)  # between 0 and 1
            predicted_image_embedding = self.prior(
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
                r=ratio.expand(latents.size(0) * 2) if do_classifier_free_guidance else ratio,
                c=text_encoder_hidden_states,
            )

            if do_classifier_free_guidance:
                predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
                predicted_image_embedding = torch.lerp(
                    predicted_image_embedding_uncond, predicted_image_embedding_text, guidance_scale
                )

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
            ).prev_sample

        # t_start = 1.0
        # for t_end, steps in inference_steps.items():
        #     steps = torch.linspace(t_start, t_end, steps + 1, dtype=dtype, device=device)
        #     latents = self.inference_loop(
        #         latents, steps, text_encoder_hidden_states, do_classifier_free_guidance, guidance_scale, generator
        #     )
        #     t_start = t_end

        # normalize the latents
        latents = latents * 42.0 - 1.0

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            latents = latents.cpu().numpy()
            text_encoder_hidden_states = text_encoder_hidden_states.cpu().numpy()

        if not return_dict:
            return (latents, text_encoder_hidden_states)

        return WuerstchenPriorPipelineOutput(latents, text_encoder_hidden_states)


class WuerstchenGeneratorPipeline(DiffusionPipeline):
    """
    Pipeline for generating images from the  Wuerstchen model.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        generator ([`DiffNeXt`]):
            The DiffNeXt unet generator.
        vqgan ([`PaellaVQModel`]):
            The VQGAN model.
        efficient_net ([`EfficientNetEncoder`]):
            The EfficientNet encoder.
        scheduler ([`DDPMScheduler`]):
            A scheduler to be used in combination with `prior` to generate image embedding.
    """

    def __init__(
        self,
        generator: DiffNeXt,
        scheduler: DDPMScheduler,
        vqgan: PaellaVQModel,
        efficient_net: EfficientNetEncoder,
    ) -> None:
        super().__init__()
        self.multiple = 128
        self.register_modules(
            generator=generator,
            scheduler=scheduler,
            vqgan=vqgan,
            efficient_net=efficient_net,
        )
        # self.diffuzz = Diffuzz(device="cuda")

        self.register_to_config()

    def prepare_latents(self, shape, dtype, device, generator, latents, scheduler):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * scheduler.init_noise_sigma
        return latents

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if self.device != torch.device("meta") or not hasattr(self.text_encoder, "_hf_hook"):
            return self.device
        for module in self.text_encoder.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def check_inputs(
        self, predicted_image_embeddings, text_encoder_hidden_states, do_classifier_free_guidance, device
    ):
        if not isinstance(text_encoder_hidden_states, torch.Tensor):
            raise TypeError(
                f"'text_encoder_hidden_states' must be of type 'torch.Tensor', but got {type(predicted_image_embeddings)}."
            )
        if isinstance(predicted_image_embeddings, np.ndarray):
            predicted_image_embeddings = torch.Tensor(predicted_image_embeddings, device=device).to(
                dtype=text_encoder_hidden_states.dtype
            )
        if not isinstance(predicted_image_embeddings, torch.Tensor):
            raise TypeError(
                f"'predicted_image_embeddings' must be of type 'torch.Tensor' or 'np.array', but got {type(predicted_image_embeddings)}."
            )

        if do_classifier_free_guidance:
            assert (
                predicted_image_embeddings.size(0) == text_encoder_hidden_states.size(0) // 2
            ), f"'text_encoder_hidden_states' must be double the size of 'predicted_image_embeddings' in the first dimension, but {predicted_image_embeddings.size(0)} != {text_encoder_hidden_states.size(0)}."
        else:
            if predicted_image_embeddings.size(0) * 2 == text_encoder_hidden_states.size(0):
                text_encoder_hidden_states = text_encoder_hidden_states.chunk(2)[0]
            assert predicted_image_embeddings.size(0) == text_encoder_hidden_states.size(
                0
            ), f"'text_encoder_hidden_states' must be the size of 'predicted_image_embeddings' in the first dimension, but {predicted_image_embeddings.size(0)} != {text_encoder_hidden_states.size(0)}."

        return predicted_image_embeddings, text_encoder_hidden_states

    # @torch.no_grad()
    # def inference_loop(
    #     self,
    #     latents,
    #     steps,
    #     predicted_effnet_latents,
    #     text_encoder_hidden_states,
    #     do_classifier_free_guidance,
    #     guidance_scale,
    #     generator,
    # ):
    #     for i, t in enumerate(self.progress_bar(steps[:-1])):
    #         # print(torch.cat([latents] * 2).shape, latents.dtype, latents.device)
    #         # print(t.expand(latents.size(0) * 2).shape, t.dtype, t.device)
    #         # print(text_encoder_hidden_states.shape, text_encoder_hidden_states.dtype, text_encoder_hidden_states.device)
    #         # print(predicted_effnet_latents.shape, predicted_effnet_latents.dtype, predicted_effnet_latents.device)
    #         predicted_image_embedding = self.generator(
    #             torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
    #             r=t.expand(latents.size(0) * 2) if do_classifier_free_guidance else t[None],
    #             effnet=torch.cat([predicted_effnet_latents, torch.zeros_like(predicted_effnet_latents)])
    #             if do_classifier_free_guidance
    #             else predicted_effnet_latents,
    #             clip=text_encoder_hidden_states,
    #         )

    #         if do_classifier_free_guidance:
    #             predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
    #             predicted_image_embedding = predicted_image_embedding_uncond + guidance_scale * (
    #                 predicted_image_embedding_text - predicted_image_embedding_uncond
    #             )
    #         # print(t)
    #         # latents = self.diffuzz.undiffuse(latents, t[None], steps[i + 1][None], predicted_image_embedding).to(
    #         #     dtype=t.dtype
    #         # )

    #         timestep = (t * 999).cpu().int()
    #         # print(timestep)
    #         latents = self.scheduler.step(
    #             predicted_image_embedding,
    #             timestep=timestep - 1,
    #             sample=latents,
    #             generator=generator,
    #         ).prev_sample

    #     return latents

    @torch.no_grad()
    def __call__(
        self,
        predicted_image_embeddings: torch.Tensor,
        text_encoder_hidden_states: torch.Tensor,
        inference_steps: dict = None,
        guidance_scale: float = 3.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",  # pt only
        return_dict: bool = True,
    ):
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0

        if inference_steps is None:
            inference_steps = default_inference_steps_b

        predicted_image_embeddings, text_encoder_hidden_states = self.check_inputs(
            predicted_image_embeddings, text_encoder_hidden_states, do_classifier_free_guidance, device
        )

        dtype = text_encoder_hidden_states.dtype
        latent_height = int(predicted_image_embeddings.size(2) * (256 / 24))
        latent_width = int(predicted_image_embeddings.size(3) * (256 / 24))
        effnet_features_shape = (predicted_image_embeddings.size(0), 4, latent_height, latent_width)

        total_num_inference_steps = sum(inference_steps.values())
        self.scheduler.set_timesteps(total_num_inference_steps, device=device)
        prior_timesteps_tensor = self.scheduler.timesteps

        latents = self.prepare_latents(
            effnet_features_shape,
            dtype,
            device,
            generator,
            latents,
            self.scheduler,
        )

        for i, t in enumerate(self.progress_bar(prior_timesteps_tensor)):
            ratio = (t / self.scheduler.config.num_train_timesteps).to(dtype)
            predicted_image_embedding = self.generator(
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents,
                r=ratio.expand(latents.size(0) * 2) if do_classifier_free_guidance else ratio[None],
                effnet=torch.cat([predicted_image_embeddings, torch.zeros_like(predicted_image_embeddings)])
                if do_classifier_free_guidance
                else predicted_image_embeddings,
                clip=text_encoder_hidden_states,
            )

            if do_classifier_free_guidance:
                predicted_image_embedding_text, predicted_image_embedding_uncond = predicted_image_embedding.chunk(2)
                predicted_image_embedding = torch.lerp(
                    predicted_image_embedding_uncond, predicted_image_embedding_text, guidance_scale
                )

            latents = self.scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=latents,
                generator=generator,
            ).prev_sample

        # # print(generator_timesteps_tensor)
        # t_start = 1.0
        # for t_end, steps in inference_steps.items():
        #     steps = torch.linspace(t_start, t_end, steps + 1, dtype=dtype, device=device)
        #     latents = self.inference_loop(
        #         latents,
        #         steps,
        #         predicted_image_embeddings,
        #         text_encoder_hidden_states,
        #         do_classifier_free_guidance,
        #         guidance_scale,
        #         generator,
        #     )
        #     t_start = t_end

        images = self.vqgan.decode(latents).sample

        if output_type not in ["pt", "np"]:
            raise ValueError(f"Only the output types `pt` and `np` are supported not output_type={output_type}")

        if output_type == "np":
            images = images.permute(0, 2, 3, 1).cpu().numpy()

        if not return_dict:
            return images

        return WuerstchenGeneratorPipelineOutput(images)
