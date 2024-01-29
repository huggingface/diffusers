import math
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...image_processor import PipelineImageInput


def load_images(
    images: PipelineImageInput, sizes=(512, 768), left=0, right=0, top=0, bottom=0, device=None, dtype=None
):
    def pre_process(im, sizes, left=0, right=0, top=0, bottom=0):
        if isinstance(im, str):
            image = np.array(Image.open(im).convert("RGB"))[:, :, :3]
        elif isinstance(im, Image.Image):
            image = np.array((im).convert("RGB"))[:, :, :3]
        else:
            image = im
        org_size = image.shape

        h, w, c = image.shape
        left = min(left, w - 1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top : h - bottom, left : w - right]

        ar = max(*image.shape[:2]) / min(*image.shape[:2])
        if ar > 1.25:
            h_max = image.shape[0] > image.shape[1]
            if h_max:
                resized = Image.fromarray(image).resize((sizes[0], sizes[1]))
            else:
                resized = Image.fromarray(image).resize((sizes[1], sizes[0]))
        else:
            resized = Image.fromarray(image).resize((sizes[0], sizes[0]))
        image = np.array(resized)
        if image.shape != org_size:
            print(
                f"Input image has been resized to {image.shape[1]}x{image.shape[0]}px (from {org_size[1]}x{org_size[0]}px)"
            )
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        return image, resized

    tmps = []
    resized_imgs = []
    if isinstance(images, list):
        for item in images:
            prep, resized = pre_process(item, sizes, left, right, top, bottom)
            if len(tmps) > 0 and prep.shape != tmps[0].shape:
                raise ValueError(
                    f"Mixed image resolution not supported in batch processing. Target resolution set to {tmps[0].shape[2]}x{tmps[0].shape[1]}px,"
                    f"but found image with resolution {prep.shape[2]}x{prep.shape[1]}px"
                )
            tmps.append(prep)
            resized_imgs.append(resized)
    else:
        prep, resized = pre_process(images, sizes, left, right, top, bottom)
        tmps.append(prep)
        resized_imgs.append(resized)
    image = torch.stack(tmps) / 127.5 - 1

    image = image.to(device=device, dtype=dtype)
    return image, resized_imgs


class LeditsAttentionStore:
    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [], "down_self": [], "mid_self": [], "up_self": []}

    def __call__(self, attn, is_cross: bool, place_in_unet: str, editing_prompts, PnP=False):
        # attn.shape = batch_size * head_size, seq_len query, seq_len_key
        if attn.shape[1] <= self.max_size:
            bs = 1 + int(PnP) + editing_prompts
            skip = 2 if PnP else 1  # skip PnP & unconditional
            attn = torch.stack(attn.split(self.batch_size)).permute(1, 0, 2, 3)
            source_batch_size = int(attn.shape[1] // bs)
            self.forward(attn[:, skip * source_batch_size :], is_cross, place_in_unet)

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"

        self.step_store[key].append(attn)

    def between_steps(self, store_step=True):
        if store_step:
            if self.average:
                if len(self.attention_store) == 0:
                    self.attention_store = self.step_store
                else:
                    for key in self.attention_store:
                        for i in range(len(self.attention_store[key])):
                            self.attention_store[key][i] += self.step_store[key][i]
            else:
                if len(self.attention_store) == 0:
                    self.attention_store = [self.step_store]
                else:
                    self.attention_store.append(self.step_store)

            self.cur_step += 1
        self.step_store = self.get_empty_store()

    def get_attention(self, step: int):
        if self.average:
            attention = {
                key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store
            }
        else:
            assert step is not None
            attention = self.attention_store[step]
        return attention

    def aggregate_attention(
        self, attention_maps, prompts, res: Union[int, Tuple[int]], from_where: List[str], is_cross: bool, select: int
    ):
        out = [[] for x in range(self.batch_size)]
        if isinstance(res, int):
            num_pixels = res**2
            resolution = (res, res)
        else:
            num_pixels = res[0] * res[1]
            resolution = res[:2]

        for location in from_where:
            for bs_item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
                for batch, item in enumerate(bs_item):
                    if item.shape[1] == num_pixels:
                        cross_maps = item.reshape(len(prompts), -1, *resolution, item.shape[-1])[select]
                        out[batch].append(cross_maps)

        out = torch.stack([torch.cat(x, dim=0) for x in out])
        # average over heads
        out = out.sum(1) / out.shape[1]
        return out

    def __init__(self, average: bool, batch_size=1, max_resolution=16, max_size: int = None):
        self.step_store = self.get_empty_store()
        self.attention_store = []
        self.cur_step = 0
        self.average = average
        self.batch_size = batch_size
        if max_size is None:
            self.max_size = max_resolution**2
        elif max_size is not None and max_resolution is None:
            self.max_size = max_size
        else:
            raise ValueError("Only allowed to set one of max_resolution or max_size")


# Modified from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionAttendAndExcitePipeline.GaussianSmoothing
class LeditsGaussianSmoothing:
    def __init__(self, device):
        kernel_size = [3, 3]
        sigma = [0.5, 0.5]

        # The gaussian kernel is the product of the gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-(((mgrid - mean) / (2 * std)) ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, *[1] * (kernel.dim() - 1))

        self.weight = kernel.to(device)

    def __call__(self, input):
        """
        Arguments:
        Apply gaussian filter to input.
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return F.conv2d(input, weight=self.weight.to(input.dtype))
