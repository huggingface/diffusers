import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from torchvision.transforms import Normalize as Normalize
from torchvision.utils import make_grid
import numpy as np
from IPython import display
from sklearn.cluster import KMeans
import torchvision.transforms.functional as TF

###
# Loss functions
###


## CLIP -----------------------------------------

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def make_clip_loss_fn(root, args):
    clip_size = root.clip_model.visual.input_resolution # for openslip: clip_model.visual.image_size

    def parse_prompt(prompt):
        if prompt.startswith('http://') or prompt.startswith('https://'):
            vals = prompt.rsplit(':', 2)
            vals = [vals[0] + ':' + vals[1], *vals[2:]]
        else:
            vals = prompt.rsplit(':', 1)
        vals = vals + ['', '1'][len(vals):]
        return vals[0], float(vals[1])

    def parse_clip_prompts(clip_prompt):
        target_embeds, weights = [], []
        for prompt in clip_prompt:
            txt, weight = parse_prompt(prompt)
            target_embeds.append(root.clip_model.encode_text(clip.tokenize(txt).to(root.device)).float())
            weights.append(weight)
        target_embeds = torch.cat(target_embeds)
        weights = torch.tensor(weights, device=root.device)
        if weights.sum().abs() < 1e-3:
            raise RuntimeError('Clip prompt weights must not sum to 0.')
        weights /= weights.sum().abs()
        return target_embeds, weights

    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                          std=[0.26862954, 0.26130258, 0.27577711])

    make_cutouts = MakeCutouts(clip_size, args.cutn, args.cut_pow)
    target_embeds, weights = parse_clip_prompts(args.clip_prompt)

    def clip_loss_fn(x, sigma, **kwargs):
        nonlocal target_embeds, weights, make_cutouts, normalize
        clip_in = normalize(make_cutouts(x.add(1).div(2)))
        image_embeds = root.clip_model.encode_image(clip_in).float()
        dists = spherical_dist_loss(image_embeds[:, None], target_embeds[None])
        dists = dists.view([args.cutn, 1, -1])
        losses = dists.mul(weights).sum(2).mean(0)
        return losses.sum()

    return clip_loss_fn

def make_aesthetics_loss_fn(root,args):
    clip_size = root.clip_model.visual.input_resolution # for openslip: clip_model.visual.image_size

    def aesthetics_cond_fn(x, sigma, **kwargs):
        clip_in = F.interpolate(x, (clip_size, clip_size))
        image_embeds = root.clip_model.encode_image(clip_in).float()
        losses = (10 - root.aesthetics_model(image_embeds)[0])
        return losses.sum()

    return aesthetics_cond_fn

## end CLIP -----------------------------------------

# blue loss from @johnowhitaker's tutorial on Grokking Stable Diffusion
def blue_loss_fn(x, sigma, **kwargs):
  # How far are the blue channel values to 0.9:
  error = torch.abs(x[:,-1, :, :] - 0.9).mean() 
  return error

# MSE loss from init
def make_mse_loss(target):
    def mse_loss(x, sigma, **kwargs):
        return (x - target).square().mean()
    return mse_loss

# MSE loss from init
def exposure_loss(target):
    def exposure_loss_fn(x, sigma, **kwargs):
        error = torch.abs(x-target).mean()
        return error
    return exposure_loss_fn

def mean_loss_fn(x, sigma, **kwargs):
  error = torch.abs(x).mean() 
  return error

def var_loss_fn(x, sigma, **kwargs):
  error = x.var()
  return error

def get_color_palette(root, n_colors, target, verbose=False):
    def display_color_palette(color_list):
        # Expand to 64x64 grid of single color pixels
        images = color_list.unsqueeze(2).repeat(1,1,64).unsqueeze(3).repeat(1,1,1,64)
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 8).cpu()
        display.display(TF.to_pil_image(grid))
        return

    # Create color palette
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(torch.flatten(target[0],1,2).T.cpu().numpy())
    color_list = torch.Tensor(kmeans.cluster_centers_).to(root.device)
    if verbose:
        display_color_palette(color_list)
    # Get ratio of each color class in the target image
    color_indexes, color_counts = np.unique(kmeans.labels_, return_counts=True)
    # color_list = color_list[color_indexes]
    return color_list, color_counts

def make_rgb_color_match_loss(root, target, n_colors, ignore_sat_weight=None, img_shape=None, device='cuda:0'):
    """
    target (tensor): Image sample (values from -1 to 1) to extract the color palette
    n_colors (int): Number of colors in the color palette
    ignore_sat_weight (None or number>0): Scale to ignore color saturation in color comparison
    img_shape (None or (int, int)): shape (width, height) of sample that the conditioning gradient is applied to, 
                                    if None then calculate target color distribution during gradient calculation 
                                    rather than once at the beginning
    """
    assert n_colors > 0, "Must use at least one color with color match loss"

    def adjust_saturation(sample, saturation_factor):
        # as in torchvision.transforms.functional.adjust_saturation, but for tensors with values from -1,1
        return blend(sample, TF.rgb_to_grayscale(sample), saturation_factor)

    def blend(img1, img2, ratio):
        return (ratio * img1 + (1.0 - ratio) * img2).clamp(-1, 1).to(img1.dtype)

    def color_distance_distributions(n_colors, img_shape, color_list, color_counts, n_images=1):
        # Get the target color distance distributions
        # Ensure color counts total the amout of pixels in the image
        n_pixels = img_shape[0]*img_shape[1]
        color_counts = (color_counts * n_pixels / sum(color_counts)).astype(int)

        # Make color distances for each color, sorted by distance
        color_distributions = torch.zeros((n_colors, n_images, n_pixels), device=device)
        for i_image in range(n_images):
            for ic,color0 in enumerate(color_list):
                i_dist = 0
                for jc,color1 in enumerate(color_list):
                    color_dist = torch.linalg.norm(color0 - color1)
                    color_distributions[ic, i_image, i_dist:i_dist+color_counts[jc]] = color_dist
                    i_dist += color_counts[jc]
        color_distributions, _ = torch.sort(color_distributions,dim=2)
        return color_distributions

    color_list, color_counts = get_color_palette(root, n_colors, target)
    color_distributions = None
    if img_shape is not None:
        color_distributions = color_distance_distributions(n_colors, img_shape, color_list, color_counts)

    def rgb_color_ratio_loss(x, sigma, **kwargs):
        nonlocal color_distributions
        all_color_norm_distances = torch.ones(len(color_list), x.shape[0], x.shape[2], x.shape[3]).to(device) * 6.0 # distance to color won't be more than max norm1 distance between -1 and 1 in 3 color dimensions

        for ic,color in enumerate(color_list):
            # Make a tensor of entirely one color
            color = color[None,:,None].repeat(1,1,x.shape[2]).unsqueeze(3).repeat(1,1,1,x.shape[3])
            # Get the color distances
            if ignore_sat_weight is None:
                # Simple color distance
                color_distances = torch.linalg.norm(x - color,  dim=1)
            else:
                # Color distance if the colors were saturated
                # This is to make color comparison ignore shadows and highlights, for example
                color_distances = torch.linalg.norm(adjust_saturation(x, ignore_sat_weight) - color,  dim=1)

            all_color_norm_distances[ic] = color_distances
        all_color_norm_distances = torch.flatten(all_color_norm_distances,start_dim=2)

        if color_distributions is None:
            color_distributions = color_distance_distributions(n_colors, 
                                                               (x.shape[2], x.shape[3]), 
                                                               color_list, 
                                                               color_counts, 
                                                               n_images=x.shape[0])

        # Sort the color distances so we can compare them as if they were a cumulative distribution function
        all_color_norm_distances, _ = torch.sort(all_color_norm_distances,dim=2)

        color_norm_distribution_diff = all_color_norm_distances - color_distributions

        return color_norm_distribution_diff.square().mean()

    return rgb_color_ratio_loss


###
# Thresholding functions for grad
###
def threshold_by(threshold, threshold_type, clamp_schedule):

  def dynamic_thresholding(vals, sigma):
      # Dynamic thresholding from Imagen paper (May 2022)
      s = np.percentile(np.abs(vals.cpu()), threshold, axis=tuple(range(1,vals.ndim)))
      s = np.max(np.append(s,1.0))
      vals = torch.clamp(vals, -1*s, s)
      vals = torch.FloatTensor.div(vals, s)
      return vals

  def static_thresholding(vals, sigma):
      vals = torch.clamp(vals, -1*threshold, threshold)
      return vals

  def mean_thresholding(vals, sigma): # Thresholding that appears in Jax and Disco
      magnitude = vals.square().mean(axis=(1,2,3),keepdims=True).sqrt()
      vals = vals * torch.where(magnitude > threshold, threshold / magnitude, 1.0)
      return vals

  def scheduling(vals, sigma):
      clamp_val = clamp_schedule[sigma.item()]
      magnitude = vals.square().mean().sqrt()
      vals = vals * magnitude.clamp(max=clamp_val) / magnitude
      #print(clamp_val)
      return vals

  if threshold_type == 'dynamic':
      return dynamic_thresholding
  elif threshold_type == 'static':
      return static_thresholding
  elif threshold_type == 'mean':
      return mean_thresholding
  elif threshold_type == 'schedule':
      return scheduling
  else:
      raise Exception(f"Thresholding type {threshold_type} not supported")
