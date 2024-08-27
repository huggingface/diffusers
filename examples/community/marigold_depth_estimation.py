# Copyright 2024 Bingxin Ke, ETH Zurich and The HuggingFace Team. All rights reserved.
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
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import logging
import math
from typing import Dict, Union

import matplotlib
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from scipy.optimize import minimize
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DiffusionPipeline,
    LCMScheduler,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput, check_min_version


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`None` or `PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Union[None, Image.Image]
    uncertainty: Union[None, np.ndarray]


def get_pil_resample_method(method_str: str) -> Resampling:
    resample_method_dic = {
        "bilinear": Resampling.BILINEAR,
        "bicubic": Resampling.BICUBIC,
        "nearest": Resampling.NEAREST,
    }
    resample_method = resample_method_dic.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None

    @torch.no_grad()
    def __call__(
        self,
        input_image: Image,
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        resample_method: str = "bilinear",
        batch_size: int = 0,
        seed: Union[int, None] = None,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `processing_res` > 0.
            resample_method: (`str`, *optional*, defaults to `bilinear`):
                Resampling method used to resize images and depth predictions. This can be one of `bilinear`, `bicubic` or `nearest`, defaults to: `bilinear`.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            seed (`int`, *optional*, defaults to `None`)
                Reproducibility seed.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`, pass `None` to skip colorized depth map generation):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1], None if `color_map` is `None`
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """

        device = self.device
        input_size = input_image.size

        if not match_input_res:
            assert processing_res is not None, "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert ensemble_size >= 1

        # Check if denoising step is reasonable
        self._check_inference_step(denoising_steps)

        resample_method: Resampling = get_pil_resample_method(resample_method)

        # ----------------- Image Preprocess -----------------
        # Resize image
        if processing_res > 0:
            input_image = self.resize_max_res(
                input_image,
                max_edge_resolution=processing_res,
                resample_method=resample_method,
            )
        # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
        input_image = input_image.convert("RGB")
        image = np.asarray(input_image)

        # Normalize rgb values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = self._find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(single_rgb_dataset, batch_size=_bs, shuffle=False)

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False)
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
                seed=seed,
            )
            depth_pred_ls.append(depth_pred_raw.detach())
        depth_preds = torch.concat(depth_pred_ls, dim=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = self.ensemble_depths(depth_preds, **(ensemble_kwargs or {}))
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size, resample=resample_method)
            depth_pred = np.asarray(pred_img)

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        if color_map is not None:
            depth_colored = self.colorize_depth_maps(
                depth_pred, 0, 1, cmap=color_map
            ).squeeze()  # [3, H, W], value in (0, 1)
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = self.chw2hwc(depth_colored)
            depth_colored_img = Image.fromarray(depth_colored_hwc)
        else:
            depth_colored_img = None

        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def _check_inference_step(self, n_step: int):
        """
        Check if denoising step is reasonable
        Args:
            n_step (`int`): denoising steps
        """
        assert n_step >= 1

        if isinstance(self.scheduler, DDIMScheduler):
            if n_step < 10:
                logging.warning(
                    f"Too few denoising steps: {n_step}. Recommended to use the LCM checkpoint for few-step inference."
                )
        elif isinstance(self.scheduler, LCMScheduler):
            if not 1 <= n_step <= 4:
                logging.warning(f"Non-optimal setting of denoising steps: {n_step}. Recommended setting is 1-4 steps.")
        else:
            raise RuntimeError(f"Unsupported scheduler type: {type(self.scheduler)}")

    def _encode_empty_text(self):
        """
        Encode text embedding for empty prompt.
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)

    @torch.no_grad()
    def single_infer(
        self,
        rgb_in: torch.Tensor,
        num_inference_steps: int,
        seed: Union[int, None],
        show_pbar: bool,
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        device = rgb_in.device

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        if seed is None:
            rand_num_generator = None
        else:
            rand_num_generator = torch.Generator(device=device)
            rand_num_generator.manual_seed(seed)
        depth_latent = torch.randn(
            rgb_latent.shape,
            device=device,
            dtype=self.dtype,
            generator=rand_num_generator,
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self._encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat((rgb_latent.shape[0], 1, 1))  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            unet_input = torch.cat([rgb_latent, depth_latent], dim=1)  # this order is important

            # predict the noise residual
            noise_pred = self.unet(unet_input, t, encoder_hidden_states=batch_empty_text_embed).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent, generator=rand_num_generator).prev_sample

        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # shift to [0, 1]
        depth = (depth + 1.0) / 2.0

        return depth

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent

    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    @staticmethod
    def resize_max_res(img: Image.Image, max_edge_resolution: int, resample_method=Resampling.BILINEAR) -> Image.Image:
        """
        Resize image to limit maximum edge length while keeping aspect ratio.

        Args:
            img (`Image.Image`):
                Image to be resized.
            max_edge_resolution (`int`):
                Maximum edge length (pixel).
            resample_method (`PIL.Image.Resampling`):
                Resampling method used to resize images.

        Returns:
            `Image.Image`: Resized image.
        """
        original_width, original_height = img.size
        downscale_factor = min(max_edge_resolution / original_width, max_edge_resolution / original_height)

        new_width = int(original_width * downscale_factor)
        new_height = int(original_height * downscale_factor)

        resized_img = img.resize((new_width, new_height), resample=resample_method)
        return resized_img

    @staticmethod
    def colorize_depth_maps(depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None):
        """
        Colorize depth maps.
        """
        assert len(depth_map.shape) >= 2, "Invalid dimension"

        if isinstance(depth_map, torch.Tensor):
            depth = depth_map.detach().clone().squeeze().numpy()
        elif isinstance(depth_map, np.ndarray):
            depth = depth_map.copy().squeeze()
        # reshape to [ (B,) H, W ]
        if depth.ndim < 3:
            depth = depth[np.newaxis, :, :]

        # colorize
        cm = matplotlib.colormaps[cmap]
        depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
        img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
        img_colored_np = np.rollaxis(img_colored_np, 3, 1)

        if valid_mask is not None:
            if isinstance(depth_map, torch.Tensor):
                valid_mask = valid_mask.detach().numpy()
            valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
            if valid_mask.ndim < 3:
                valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
            else:
                valid_mask = valid_mask[:, np.newaxis, :, :]
            valid_mask = np.repeat(valid_mask, 3, axis=1)
            img_colored_np[~valid_mask] = 0

        if isinstance(depth_map, torch.Tensor):
            img_colored = torch.from_numpy(img_colored_np).float()
        elif isinstance(depth_map, np.ndarray):
            img_colored = img_colored_np

        return img_colored

    @staticmethod
    def chw2hwc(chw):
        assert 3 == len(chw.shape)
        if isinstance(chw, torch.Tensor):
            hwc = torch.permute(chw, (1, 2, 0))
        elif isinstance(chw, np.ndarray):
            hwc = np.moveaxis(chw, 0, -1)
        return hwc

    @staticmethod
    def _find_batch_size(ensemble_size: int, input_res: int, dtype: torch.dtype) -> int:
        """
        Automatically search for suitable operating batch size.

        Args:
            ensemble_size (`int`):
                Number of predictions to be ensembled.
            input_res (`int`):
                Operating resolution of the input image.

        Returns:
            `int`: Operating batch size.
        """
        # Search table for suggested max. inference batch size
        bs_search_table = [
            # tested on A100-PCIE-80GB
            {"res": 768, "total_vram": 79, "bs": 35, "dtype": torch.float32},
            {"res": 1024, "total_vram": 79, "bs": 20, "dtype": torch.float32},
            # tested on A100-PCIE-40GB
            {"res": 768, "total_vram": 39, "bs": 15, "dtype": torch.float32},
            {"res": 1024, "total_vram": 39, "bs": 8, "dtype": torch.float32},
            {"res": 768, "total_vram": 39, "bs": 30, "dtype": torch.float16},
            {"res": 1024, "total_vram": 39, "bs": 15, "dtype": torch.float16},
            # tested on RTX3090, RTX4090
            {"res": 512, "total_vram": 23, "bs": 20, "dtype": torch.float32},
            {"res": 768, "total_vram": 23, "bs": 7, "dtype": torch.float32},
            {"res": 1024, "total_vram": 23, "bs": 3, "dtype": torch.float32},
            {"res": 512, "total_vram": 23, "bs": 40, "dtype": torch.float16},
            {"res": 768, "total_vram": 23, "bs": 18, "dtype": torch.float16},
            {"res": 1024, "total_vram": 23, "bs": 10, "dtype": torch.float16},
            # tested on GTX1080Ti
            {"res": 512, "total_vram": 10, "bs": 5, "dtype": torch.float32},
            {"res": 768, "total_vram": 10, "bs": 2, "dtype": torch.float32},
            {"res": 512, "total_vram": 10, "bs": 10, "dtype": torch.float16},
            {"res": 768, "total_vram": 10, "bs": 5, "dtype": torch.float16},
            {"res": 1024, "total_vram": 10, "bs": 3, "dtype": torch.float16},
        ]

        if not torch.cuda.is_available():
            return 1

        total_vram = torch.cuda.mem_get_info()[1] / 1024.0**3
        filtered_bs_search_table = [s for s in bs_search_table if s["dtype"] == dtype]
        for settings in sorted(
            filtered_bs_search_table,
            key=lambda k: (k["res"], -k["total_vram"]),
        ):
            if input_res <= settings["res"] and total_vram >= settings["total_vram"]:
                bs = settings["bs"]
                if bs > ensemble_size:
                    bs = ensemble_size
                elif bs > math.ceil(ensemble_size / 2) and bs < ensemble_size:
                    bs = math.ceil(ensemble_size / 2)
                return bs

        return 1

    @staticmethod
    def ensemble_depths(
        input_images: torch.Tensor,
        regularizer_strength: float = 0.02,
        max_iter: int = 2,
        tol: float = 1e-3,
        reduction: str = "median",
        max_res: int = None,
    ):
        """
        To ensemble multiple affine-invariant depth images (up to scale and shift),
            by aligning estimating the scale and shift
        """

        def inter_distances(tensors: torch.Tensor):
            """
            To calculate the distance between each two depth maps.
            """
            distances = []
            for i, j in torch.combinations(torch.arange(tensors.shape[0])):
                arr1 = tensors[i : i + 1]
                arr2 = tensors[j : j + 1]
                distances.append(arr1 - arr2)
            dist = torch.concatenate(distances, dim=0)
            return dist

        device = input_images.device
        dtype = input_images.dtype
        np_dtype = np.float32

        original_input = input_images.clone()
        n_img = input_images.shape[0]
        ori_shape = input_images.shape

        if max_res is not None:
            scale_factor = torch.min(max_res / torch.tensor(ori_shape[-2:]))
            if scale_factor < 1:
                downscaler = torch.nn.Upsample(scale_factor=scale_factor, mode="nearest")
                input_images = downscaler(torch.from_numpy(input_images)).numpy()

        # init guess
        _min = np.min(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
        _max = np.max(input_images.reshape((n_img, -1)).cpu().numpy(), axis=1)
        s_init = 1.0 / (_max - _min).reshape((-1, 1, 1))
        t_init = (-1 * s_init.flatten() * _min.flatten()).reshape((-1, 1, 1))
        x = np.concatenate([s_init, t_init]).reshape(-1).astype(np_dtype)

        input_images = input_images.to(device)

        # objective function
        def closure(x):
            l = len(x)
            s = x[: int(l / 2)]
            t = x[int(l / 2) :]
            s = torch.from_numpy(s).to(dtype=dtype).to(device)
            t = torch.from_numpy(t).to(dtype=dtype).to(device)

            transformed_arrays = input_images * s.view((-1, 1, 1)) + t.view((-1, 1, 1))
            dists = inter_distances(transformed_arrays)
            sqrt_dist = torch.sqrt(torch.mean(dists**2))

            if "mean" == reduction:
                pred = torch.mean(transformed_arrays, dim=0)
            elif "median" == reduction:
                pred = torch.median(transformed_arrays, dim=0).values
            else:
                raise ValueError

            near_err = torch.sqrt((0 - torch.min(pred)) ** 2)
            far_err = torch.sqrt((1 - torch.max(pred)) ** 2)

            err = sqrt_dist + (near_err + far_err) * regularizer_strength
            err = err.detach().cpu().numpy().astype(np_dtype)
            return err

        res = minimize(
            closure,
            x,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )
        x = res.x
        l = len(x)
        s = x[: int(l / 2)]
        t = x[int(l / 2) :]

        # Prediction
        s = torch.from_numpy(s).to(dtype=dtype).to(device)
        t = torch.from_numpy(t).to(dtype=dtype).to(device)
        transformed_arrays = original_input * s.view(-1, 1, 1) + t.view(-1, 1, 1)
        if "mean" == reduction:
            aligned_images = torch.mean(transformed_arrays, dim=0)
            std = torch.std(transformed_arrays, dim=0)
            uncertainty = std
        elif "median" == reduction:
            aligned_images = torch.median(transformed_arrays, dim=0).values
            # MAD (median absolute deviation) as uncertainty indicator
            abs_dev = torch.abs(transformed_arrays - aligned_images)
            mad = torch.median(abs_dev, dim=0).values
            uncertainty = mad
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")

        # Scale and shift to [0, 1]
        _min = torch.min(aligned_images)
        _max = torch.max(aligned_images)
        aligned_images = (aligned_images - _min) / (_max - _min)
        uncertainty /= _max - _min

        return aligned_images, uncertainty
