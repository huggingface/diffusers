# Copyright 2022 The HuggingFace Team. All rights reserved.
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
from typing import List, Optional, Tuple, Union

import torch

from ...utils import randn_tensor, DIFFUSERS_CACHE
from ..pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from pathlib import Path
from collections import OrderedDict

from .swinir import SwinIR

from .basicsr.utils import img2tensor, tensor2img
from .basicsr.archs.rrdbnet_arch import RRDBNet
from .basicsr.utils.realesrgan_utils import RealESRGANer
from .basicsr.utils.download_util import load_file_from_url
from .facelib.utils.face_restoration_helper import FaceRestoreHelper

_pretrain_model_url = {
    'dif_estimator':'https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth',
    'realesrgan':'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
    }

class DifFacePipeline(DiffusionPipeline):
    r"""
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        unet ([`UNet2DModel`]): U-Net architecture to denoise the encoded image.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    def init_diffuse_estimator(self):
        self.ir_model = SwinIR(
                img_size=64,
                patch_size=1,
                in_chans=3,
                embed_dim=180,
                depths=[6, 6, 6, 6, 6, 6, 6, 6],
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6],
                window_size=8,
                mlp_ratio=2,
                sf=8,
                img_range=1.0,
                upsampler="nearest+conv",
                resi_connection="1conv",
                unshuffle=True,
                unshuffle_scale=8,
                ).to(device=self.device)

        ckpt_path = load_file_from_url(
                _pretrain_model_url['dif_estimator'],
                model_dir=str(Path(DIFFUSERS_CACHE) / 'dif_estimator'),
                progress=True,
                )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt = OrderedDict({key[7:]:value for key, value in ckpt.items()})
        self.ir_model.load_state_dict(ckpt, strict=True)

    def init_face_tool(self):
        self.face_helper = FaceRestoreHelper(
                upscale_factor=2,
                face_size=512,
                crop_ratio=(1, 1),
                det_model='retinaface_resnet50',
                save_ext='png',
                use_parse=True,
                device=self.device,
                )
        bg_model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=2,
                )
        self.bg_model = RealESRGANer(
            scale=2,
            model_path=_pretrain_model_url['realesrgan'],
            model=bg_model,
            model_dir=str(Path(DIFFUSERS_CACHE) / 'realesrgan'),
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True,
            device=self.device,
            )  # need to set False in CPU mode

    @torch.no_grad()
    def __call__(
        self,
        y0: np.ndarray,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        bs: int = 8,
        started_steps: int = 100,
        num_inference_steps: int = 250,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        aligned: bool=False,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        Args:
            y0 ('np.ndarray'): Low quality image, uint8, BGR order
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            bs ('int'): batch size for diffusion model, only used for the unalinged case
            started_steps (`int`):
                Started denoising steps. Larger started steps usually lead to a higher quality and lower identity.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """
        def  _process_batch(cropped_faces_list):
            length = len(cropped_faces_list)
            cropped_face_t = np.stack(
                    img2tensor(cropped_faces_list, bgr2rgb=True, float32=True),
                    axis=0) / 255.
            cropped_face_t = torch.from_numpy(cropped_face_t).to(self.device)
            restored_faces = self.sample_alinged(
                    cropped_face_t,
                    started_steps=started_steps,
                    progress=False,
                    )      # [0, 1], b x c x h x w
            return restored_faces

        if not hasattr(self, 'ir_model'):
            self.init_diffuse_estimator()

        if not hasattr(self, 'face_helper') and not aligned:
            self.init_face_tool()

        assert isinstance(y0, np.ndarray), 'y0 must be numpy ndarray (image) with BGR order, uint8 format'

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        if aligned:
            y0 = torch.from_numpy(y0[:, :, ::-1].copy()).to(device=self.device, dtype=torch.float32)
            y0 = y0.permute(2,0,1).unsqueeze(0) / 255.    # b x c x h x w, [0,1]
            image = self.sample_alinged(y0, generator, started_steps, num_inference_steps) # [0, 1], torch.Tensor
            image = image.cpu().squeeze(0).permute(1,2,0).numpy()  # [0, 1], numpy array, RGB
        else:
            self.face_helper.clean_all()
            self.face_helper.read_image(y0)
            num_det_faces = self.face_helper.get_face_landmarks_5(
                    only_center_face=False,
                    resize=640,
                    eye_dist_threshold=5,
                    )
            self.face_helper.align_warp_face()

            num_cropped_face = len(self.face_helper.cropped_faces)
            if num_cropped_face > bs:
                restored_faces = []
                for idx_start in range(0, num_cropped_face, bs):
                    idx_end = idx_start + bs if idx_start + bs < num_cropped_face else num_cropped_face
                    current_cropped_faces = self.face_helper.cropped_faces[idx_start:idx_end]
                    current_restored_faces = _process_batch(current_cropped_faces)
                    current_restored_faces = tensor2img(
                            list(current_restored_faces.split(1, dim=0)),
                            rgb2bgr=True,
                            min_max=(0, 1),
                            out_type=np.uint8,
                            )
                    restored_faces.extend(current_restored_faces)
            else:
                restored_faces = _process_batch(self.face_helper.cropped_faces)
                restored_faces = tensor2img(
                        list(restored_faces.split(1, dim=0)),
                        rgb2bgr=True,
                        min_max=(0, 1),
                        out_type=np.uint8,
                        )
            for xx in restored_faces:
                self.face_helper.add_restored_face(xx)

            # paste_back
            bg_img = self.bg_model.enhance(y0, outscale=2)[0]
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=False,
                    )    # [0, 255], uint8, BGR order, h x w x c
            image = restored_img[:, :, ::-1].copy().astype(np.float32) / 255  # [0, 1], numpy array, RGB

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    @torch.no_grad()
    def sample_alinged(
        self,
        y0: torch.Tensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        started_steps: int = 100,
        progress: bool=True,
    ):
        r"""
        Args:
            y0 (torch.Tensor): Low quality image, b x c x h x w, [0, 1], RGB order
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            started_steps (`int`):
                Started denoising steps. Larger started steps usually lead to a higher quality and lower identity.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            progress: whether to display the progress bar
        Returns:
            torch.Tensor, b x c x h x w, [0, 1]
        """
        assert isinstance(y0, torch.Tensor), 'y0 must be torch tensor with 4 dimensions in range [-1, 1].'

        # Sample gaussian noise
        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            noise = randn_tensor(y0.shape, generator=generator)
            noise = noise.to(self.device)
        else:
            noise = randn_tensor(y0.shape, generator=generator, device=self.device)

        # enhanced by diffused estimator
        y0 = self.ir_model(y0)

        # [0, 1] to [-1, 1]
        y0 = (y0 - 0.5) / 0.5

        # forward diffusion
        image = self.scheduler.add_noise(y0, noise, torch.tensor([started_steps,], dtype=torch.int64))

        if progress:
            iter_steps = self.progress_bar(self.scheduler.timesteps[-started_steps:])
        else:
            iter_steps = self.scheduler.timesteps[-started_steps:].tolist()

        for t in iter_steps:
            # 1. predict noise model_output
            model_output = self.unet(image, t).sample

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image, generator=generator).prev_sample

        # [-1, 1] to [0, 1]
        image = image * 0.5 + 0.5

        return image
