from typing import Any, Callable, Dict, List, Optional, Union

import torch

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


pipe1_model_id = "CompVis/stable-diffusion-v1-1"
pipe2_model_id = "CompVis/stable-diffusion-v1-2"
pipe3_model_id = "CompVis/stable-diffusion-v1-3"
pipe4_model_id = "CompVis/stable-diffusion-v1-4"


class StableDiffusionComparisonPipeline(DiffusionPipeline):
    r"""
    Pipeline for parallel comparison of Stable Diffusion v1-v4
    This pipeline inherits from DiffusionPipeline and depends on the use of an Auth Token for
    downloading pre-trained checkpoints from Hugging Face Hub.
    Args:
        pipe1 ('StableDiffusionPipeline' or 'str', optional):
            A Stable Diffusion Pipeline prepared from the SD1.1 Checkpoints on Hugging Face Hub
        pipe2 ('StableDiffusionPipeline' or 'str', optional):
            A Stable Diffusion Pipeline prepared from the SD1.2 Checkpoints on Hugging Face Hub
        pipe3 ('StableDiffusionPipeline' or 'str', optional):
            A Stable Diffusion Pipeline prepared from the SD1.3 Checkpoints on Hugging Face Hub
        pipe4 ('StableDiffusionPipeline' or 'str', optional):
            A Stable Diffusion Pipeline prepared from the SD1.4 Checkpoints on Hugging Face Hub
    """

    def _init_(
        self,
        sd1_1: Union[StableDiffusionPipeline, str],
        sd1_2: Union[StableDiffusionPipeline, str],
        sd1_3: Union[StableDiffusionPipeline, str],
        sd1_4: Union[StableDiffusionPipeline, str],
    ):
        super()._init_()

        if not isinstance(sd1_1, StableDiffusionPipeline):
            self.pipe1 = StableDiffusionPipeline.from_pretrained(
                pipe1_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True
            )
        else:
            self.pipe1 = sd1_1
        if not isinstance(sd1_2, StableDiffusionPipeline):
            self.pipe2 = StableDiffusionPipeline.from_pretrained(
                pipe2_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True
            )
        else:
            self.pipe2 = sd1_2
        if not isinstance(sd1_3, StableDiffusionPipeline):
            self.pipe3 = StableDiffusionPipeline.from_pretrained(
                pipe3_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True
            )
        else:
            self.pipe3 = sd1_3
        if not isinstance(sd1_4, StableDiffusionPipeline):
            self.pipe4 = StableDiffusionPipeline.from_pretrained(
                pipe4_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token=True
            )
        else:
            self.pipe4 = sd1_4

        self.register_modules(pipeline1=self.pipe1, pipeline2=self.pipe2, pipeline3=self.pipe3, pipeline4=self.pipe4)

    @property
    def layers(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.config.keys() if not k.startswith("_")}

    @torch.no_grad()
    def text2img_sd1_1(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        return self.pipe1(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    @torch.no_grad()
    def text2img_sd1_2(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        return self.pipe2(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    @torch.no_grad()
    def text2img_sd1_3(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        return self.pipe3(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    @torch.no_grad()
    def text2img_sd1_4(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        return self.pipe4(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    @torch.no_grad()
    def _call_(
        self,
        prompt: Union[str, List[str]],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation. This function will generate 4 results as part
        of running all the 4 pipelines for SD1.1-1.4 together in a serial-processing, parallel-invocation fashion.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, optional, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, optional, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, optional, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, optional, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            eta (`float`, optional, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, optional):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, optional):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, optional, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, optional, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)

        # Checks if the height and width are divisible by 8 or not
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 8 but are {height} and {width}.")

        # Get first result from Stable Diffusion Checkpoint v1.1
        res1 = self.text2img_sd1_1(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

        # Get first result from Stable Diffusion Checkpoint v1.2
        res2 = self.text2img_sd1_2(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

        # Get first result from Stable Diffusion Checkpoint v1.3
        res3 = self.text2img_sd1_3(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

        # Get first result from Stable Diffusion Checkpoint v1.4
        res4 = self.text2img_sd1_4(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

        # Get all result images into a single list and pass it via StableDiffusionPipelineOutput for final result
        return StableDiffusionPipelineOutput([res1[0], res2[0], res3[0], res4[0]])
