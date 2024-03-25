# Copyright 2024 The HuggingFace Team. All rights reserved.
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


from math import pi
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from diffusers import DDPMScheduler, DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils.torch_utils import randn_tensor


class DPSPipeline(DiffusionPipeline):
    r"""
    Pipeline for Diffusion Posterior Sampling.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """

    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        measurement: torch.Tensor,
        operator: torch.nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        zeta: float = 0.3,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            measurement (`torch.Tensor`, *required*):
                A 'torch.Tensor', the corrupted image
            operator (`torch.nn.Module`, *required*):
                A 'torch.nn.Module', the operator generating the corrupted image
            loss_fn (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`, *required*):
                A 'Callable[[torch.Tensor, torch.Tensor], torch.Tensor]', the loss function used
                between the measurements, for most of the cases using RMSE is fine.
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            with torch.enable_grad():
                # 1. predict noise model_output
                image = image.requires_grad_()
                model_output = self.unet(image, t).sample

                # 2. compute previous image x'_{t-1} and original prediction x0_{t}
                scheduler_out = self.scheduler.step(model_output, t, image, generator=generator)
                image_pred, origi_pred = scheduler_out.prev_sample, scheduler_out.pred_original_sample

                # 3. compute y'_t = f(x0_{t})
                measurement_pred = operator(origi_pred)

                # 4. compute loss = d(y, y'_t-1)
                loss = loss_fn(measurement, measurement_pred)
                loss.backward()

                print("distance: {0:.4f}".format(loss.item()))

                with torch.no_grad():
                    image_pred = image_pred - zeta * image.grad
                    image = image_pred.detach()

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)


if __name__ == "__main__":
    import scipy
    from torch import nn
    from torchvision.utils import save_image

    # defining the operators f(.) of y = f(x)
    # super-resolution operator
    class SuperResolutionOperator(nn.Module):
        def __init__(self, in_shape, scale_factor):
            super().__init__()

            # Resizer local class, do not use outiside the SR operator class
            class Resizer(nn.Module):
                def __init__(self, in_shape, scale_factor=None, output_shape=None, kernel=None, antialiasing=True):
                    super(Resizer, self).__init__()

                    # First standardize values and fill missing arguments (if needed) by deriving scale from output shape or vice versa
                    scale_factor, output_shape = self.fix_scale_and_size(in_shape, output_shape, scale_factor)

                    # Choose interpolation method, each method has the matching kernel size
                    def cubic(x):
                        absx = np.abs(x)
                        absx2 = absx**2
                        absx3 = absx**3
                        return (1.5 * absx3 - 2.5 * absx2 + 1) * (absx <= 1) + (
                            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2
                        ) * ((1 < absx) & (absx <= 2))

                    def lanczos2(x):
                        return (
                            (np.sin(pi * x) * np.sin(pi * x / 2) + np.finfo(np.float32).eps)
                            / ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps)
                        ) * (abs(x) < 2)

                    def box(x):
                        return ((-0.5 <= x) & (x < 0.5)) * 1.0

                    def lanczos3(x):
                        return (
                            (np.sin(pi * x) * np.sin(pi * x / 3) + np.finfo(np.float32).eps)
                            / ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps)
                        ) * (abs(x) < 3)

                    def linear(x):
                        return (x + 1) * ((-1 <= x) & (x < 0)) + (1 - x) * ((0 <= x) & (x <= 1))

                    method, kernel_width = {
                        "cubic": (cubic, 4.0),
                        "lanczos2": (lanczos2, 4.0),
                        "lanczos3": (lanczos3, 6.0),
                        "box": (box, 1.0),
                        "linear": (linear, 2.0),
                        None: (cubic, 4.0),  # set default interpolation method as cubic
                    }.get(kernel)

                    # Antialiasing is only used when downscaling
                    antialiasing *= np.any(np.array(scale_factor) < 1)

                    # Sort indices of dimensions according to scale of each dimension. since we are going dim by dim this is efficient
                    sorted_dims = np.argsort(np.array(scale_factor))
                    self.sorted_dims = [int(dim) for dim in sorted_dims if scale_factor[dim] != 1]

                    # Iterate over dimensions to calculate local weights for resizing and resize each time in one direction
                    field_of_view_list = []
                    weights_list = []
                    for dim in self.sorted_dims:
                        # for each coordinate (along 1 dim), calculate which coordinates in the input image affect its result and the
                        # weights that multiply the values there to get its result.
                        weights, field_of_view = self.contributions(
                            in_shape[dim], output_shape[dim], scale_factor[dim], method, kernel_width, antialiasing
                        )

                        # convert to torch tensor
                        weights = torch.tensor(weights.T, dtype=torch.float32)

                        # We add singleton dimensions to the weight matrix so we can multiply it with the big tensor we get for
                        # tmp_im[field_of_view.T], (bsxfun style)
                        weights_list.append(
                            nn.Parameter(
                                torch.reshape(weights, list(weights.shape) + (len(scale_factor) - 1) * [1]),
                                requires_grad=False,
                            )
                        )
                        field_of_view_list.append(
                            nn.Parameter(
                                torch.tensor(field_of_view.T.astype(np.int32), dtype=torch.long), requires_grad=False
                            )
                        )

                    self.field_of_view = nn.ParameterList(field_of_view_list)
                    self.weights = nn.ParameterList(weights_list)

                def forward(self, in_tensor):
                    x = in_tensor

                    # Use the affecting position values and the set of weights to calculate the result of resizing along this 1 dim
                    for dim, fov, w in zip(self.sorted_dims, self.field_of_view, self.weights):
                        # To be able to act on each dim, we swap so that dim 0 is the wanted dim to resize
                        x = torch.transpose(x, dim, 0)

                        # This is a bit of a complicated multiplication: x[field_of_view.T] is a tensor of order image_dims+1.
                        # for each pixel in the output-image it matches the positions the influence it from the input image (along 1 dim
                        # only, this is why it only adds 1 dim to 5the shape). We then multiply, for each pixel, its set of positions with
                        # the matching set of weights. we do this by this big tensor element-wise multiplication (MATLAB bsxfun style:
                        # matching dims are multiplied element-wise while singletons mean that the matching dim is all multiplied by the
                        # same number
                        x = torch.sum(x[fov] * w, dim=0)

                        # Finally we swap back the axes to the original order
                        x = torch.transpose(x, dim, 0)

                    return x

                def fix_scale_and_size(self, input_shape, output_shape, scale_factor):
                    # First fixing the scale-factor (if given) to be standardized the function expects (a list of scale factors in the
                    # same size as the number of input dimensions)
                    if scale_factor is not None:
                        # By default, if scale-factor is a scalar we assume 2d resizing and duplicate it.
                        if np.isscalar(scale_factor) and len(input_shape) > 1:
                            scale_factor = [scale_factor, scale_factor]

                        # We extend the size of scale-factor list to the size of the input by assigning 1 to all the unspecified scales
                        scale_factor = list(scale_factor)
                        scale_factor = [1] * (len(input_shape) - len(scale_factor)) + scale_factor

                    # Fixing output-shape (if given): extending it to the size of the input-shape, by assigning the original input-size
                    # to all the unspecified dimensions
                    if output_shape is not None:
                        output_shape = list(input_shape[len(output_shape) :]) + list(np.uint(np.array(output_shape)))

                    # Dealing with the case of non-give scale-factor, calculating according to output-shape. note that this is
                    # sub-optimal, because there can be different scales to the same output-shape.
                    if scale_factor is None:
                        scale_factor = 1.0 * np.array(output_shape) / np.array(input_shape)

                    # Dealing with missing output-shape. calculating according to scale-factor
                    if output_shape is None:
                        output_shape = np.uint(np.ceil(np.array(input_shape) * np.array(scale_factor)))

                    return scale_factor, output_shape

                def contributions(self, in_length, out_length, scale, kernel, kernel_width, antialiasing):
                    # This function calculates a set of 'filters' and a set of field_of_view that will later on be applied
                    # such that each position from the field_of_view will be multiplied with a matching filter from the
                    # 'weights' based on the interpolation method and the distance of the sub-pixel location from the pixel centers
                    # around it. This is only done for one dimension of the image.

                    # When anti-aliasing is activated (default and only for downscaling) the receptive field is stretched to size of
                    # 1/sf. this means filtering is more 'low-pass filter'.
                    fixed_kernel = (lambda arg: scale * kernel(scale * arg)) if antialiasing else kernel
                    kernel_width *= 1.0 / scale if antialiasing else 1.0

                    # These are the coordinates of the output image
                    out_coordinates = np.arange(1, out_length + 1)

                    # since both scale-factor and output size can be provided simulatneously, perserving the center of the image requires shifting
                    # the output coordinates. the deviation is because out_length doesn't necesary equal in_length*scale.
                    # to keep the center we need to subtract half of this deivation so that we get equal margins for boths sides and center is preserved.
                    shifted_out_coordinates = out_coordinates - (out_length - in_length * scale) / 2

                    # These are the matching positions of the output-coordinates on the input image coordinates.
                    # Best explained by example: say we have 4 horizontal pixels for HR and we downscale by SF=2 and get 2 pixels:
                    # [1,2,3,4] -> [1,2]. Remember each pixel number is the middle of the pixel.
                    # The scaling is done between the distances and not pixel numbers (the right boundary of pixel 4 is transformed to
                    # the right boundary of pixel 2. pixel 1 in the small image matches the boundary between pixels 1 and 2 in the big
                    # one and not to pixel 2. This means the position is not just multiplication of the old pos by scale-factor).
                    # So if we measure distance from the left border, middle of pixel 1 is at distance d=0.5, border between 1 and 2 is
                    # at d=1, and so on (d = p - 0.5).  we calculate (d_new = d_old / sf) which means:
                    # (p_new-0.5 = (p_old-0.5) / sf)     ->          p_new = p_old/sf + 0.5 * (1-1/sf)
                    match_coordinates = shifted_out_coordinates / scale + 0.5 * (1 - 1 / scale)

                    # This is the left boundary to start multiplying the filter from, it depends on the size of the filter
                    left_boundary = np.floor(match_coordinates - kernel_width / 2)

                    # Kernel width needs to be enlarged because when covering has sub-pixel borders, it must 'see' the pixel centers
                    # of the pixels it only covered a part from. So we add one pixel at each side to consider (weights can zeroize them)
                    expanded_kernel_width = np.ceil(kernel_width) + 2

                    # Determine a set of field_of_view for each each output position, these are the pixels in the input image
                    # that the pixel in the output image 'sees'. We get a matrix whos horizontal dim is the output pixels (big) and the
                    # vertical dim is the pixels it 'sees' (kernel_size + 2)
                    field_of_view = np.squeeze(
                        np.int16(np.expand_dims(left_boundary, axis=1) + np.arange(expanded_kernel_width) - 1)
                    )

                    # Assign weight to each pixel in the field of view. A matrix whos horizontal dim is the output pixels and the
                    # vertical dim is a list of weights matching to the pixel in the field of view (that are specified in
                    # 'field_of_view')
                    weights = fixed_kernel(1.0 * np.expand_dims(match_coordinates, axis=1) - field_of_view - 1)

                    # Normalize weights to sum up to 1. be careful from dividing by 0
                    sum_weights = np.sum(weights, axis=1)
                    sum_weights[sum_weights == 0] = 1.0
                    weights = 1.0 * weights / np.expand_dims(sum_weights, axis=1)

                    # We use this mirror structure as a trick for reflection padding at the boundaries
                    mirror = np.uint(np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))))
                    field_of_view = mirror[np.mod(field_of_view, mirror.shape[0])]

                    # Get rid of  weights and pixel positions that are of zero weight
                    non_zero_out_pixels = np.nonzero(np.any(weights, axis=0))
                    weights = np.squeeze(weights[:, non_zero_out_pixels])
                    field_of_view = np.squeeze(field_of_view[:, non_zero_out_pixels])

                    # Final products are the relative positions and the matching weights, both are output_size X fixed_kernel_size
                    return weights, field_of_view

            self.down_sample = Resizer(in_shape, 1 / scale_factor)
            for param in self.parameters():
                param.requires_grad = False

        def forward(self, data, **kwargs):
            return self.down_sample(data)

    # Gaussian blurring operator
    class GaussialBlurOperator(nn.Module):
        def __init__(self, kernel_size, intensity):
            super().__init__()

            class Blurkernel(nn.Module):
                def __init__(self, blur_type="gaussian", kernel_size=31, std=3.0):
                    super().__init__()
                    self.blur_type = blur_type
                    self.kernel_size = kernel_size
                    self.std = std
                    self.seq = nn.Sequential(
                        nn.ReflectionPad2d(self.kernel_size // 2),
                        nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3),
                    )
                    self.weights_init()

                def forward(self, x):
                    return self.seq(x)

                def weights_init(self):
                    if self.blur_type == "gaussian":
                        n = np.zeros((self.kernel_size, self.kernel_size))
                        n[self.kernel_size // 2, self.kernel_size // 2] = 1
                        k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
                        k = torch.from_numpy(k)
                        self.k = k
                        for name, f in self.named_parameters():
                            f.data.copy_(k)

                def update_weights(self, k):
                    if not torch.is_tensor(k):
                        k = torch.from_numpy(k)
                    for name, f in self.named_parameters():
                        f.data.copy_(k)

                def get_kernel(self):
                    return self.k

            self.kernel_size = kernel_size
            self.conv = Blurkernel(blur_type="gaussian", kernel_size=kernel_size, std=intensity)
            self.kernel = self.conv.get_kernel()
            self.conv.update_weights(self.kernel.type(torch.float32))

            for param in self.parameters():
                param.requires_grad = False

        def forward(self, data, **kwargs):
            return self.conv(data)

        def transpose(self, data, **kwargs):
            return data

        def get_kernel(self):
            return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)

    # assuming the forward process y = f(x) is polluted by Gaussian noise, use l2 norm
    def RMSELoss(yhat, y):
        return torch.sqrt(torch.sum((yhat - y) ** 2))

    # set up source image
    src = Image.open("sample.png")
    # read image into [1,3,H,W]
    src = torch.from_numpy(np.array(src, dtype=np.float32)).permute(2, 0, 1)[None]
    # normalize image to [-1,1]
    src = (src / 127.5) - 1.0
    src = src.to("cuda")

    # set up operator and measurement
    # operator = SuperResolutionOperator(in_shape=src.shape, scale_factor=4).to("cuda")
    operator = GaussialBlurOperator(kernel_size=61, intensity=3.0).to("cuda")
    measurement = operator(src)

    # set up scheduler
    scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(1000)

    # set up model
    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to("cuda")

    save_image((src + 1.0) / 2.0, "dps_src.png")
    save_image((measurement + 1.0) / 2.0, "dps_mea.png")

    # finally, the pipeline
    dpspipe = DPSPipeline(model, scheduler)
    image = dpspipe(
        measurement=measurement,
        operator=operator,
        loss_fn=RMSELoss,
        zeta=1.0,
    ).images[0]

    image.save("dps_generated_image.png")
