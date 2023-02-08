from torch import nn
from k_diffusion import utils as k_utils
import torch
from k_diffusion.external import CompVisDenoiser
from torchvision.utils import make_grid
from IPython import display
from torchvision.transforms.functional import to_pil_image

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

class CFGDenoiserWithGrad(CompVisDenoiser):
    def __init__(self, model, 
                       loss_fns_scales, # List of [cond_function, scale] pairs
                       clamp_func=None,  # Gradient clamping function, clamp_func(grad, sigma)
                       gradient_wrt=None, # Calculate gradient with respect to ["x", "x0_pred", "both"]
                       gradient_add_to=None, # Add gradient to ["cond", "uncond", "both"]
                       cond_uncond_sync=True, # Calculates the cond and uncond simultaneously
                       decode_method=None, # Function used to decode the latent during gradient calculation
                       grad_inject_timing_fn=None, # Option to use grad in only a few of the steps
                       grad_consolidate_fn=None, # Function to add grad to image fn(img, grad, sigma)
                       verbose=False):
        super().__init__(model.inner_model)
        self.inner_model = model
        self.cond_uncond_sync = cond_uncond_sync 

        # Initialize gradient calculation variables
        self.clamp_func = clamp_func
        self.gradient_add_to = gradient_add_to
        if gradient_wrt is None:
            self.gradient_wrt = 'x'
        self.gradient_wrt = gradient_wrt
        if decode_method is None:
            decode_fn = lambda x: x
        elif decode_method == "autoencoder":
            decode_fn = model.inner_model.differentiable_decode_first_stage
        elif decode_method == "linear":
            decode_fn = model.inner_model.linear_decode
        self.decode_fn = decode_fn

        # Parse loss function-scale pairs
        cond_fns = []
        for loss_fn,scale in loss_fns_scales:
            if scale != 0:
                cond_fn = self.make_cond_fn(loss_fn, scale)
            else:
                cond_fn = None
            cond_fns += [cond_fn]
        self.cond_fns = cond_fns

        if grad_inject_timing_fn is None:
            self.grad_inject_timing_fn = lambda sigma: True
        else:
            self.grad_inject_timing_fn = grad_inject_timing_fn
        if grad_consolidate_fn is None:
            self.grad_consolidate_fn = lambda img, grad, sigma: img + grad * sigma
        else:
            self.grad_consolidate_fn = grad_consolidate_fn

        self.verbose = verbose
        self.verbose_print = print if self.verbose else lambda *args, **kwargs: None


    # General denoising model with gradient conditioning
    def cond_model_fn_(self, x, sigma, inner_model=None, **kwargs):

        # inner_model: optionally use a different inner_model function or a wrapper function around inner_model, see self.forward._cfg_model
        if inner_model is None:
            inner_model = self.inner_model

        total_cond_grad = torch.zeros_like(x)
        for cond_fn in self.cond_fns:
            if cond_fn is None: continue

            # Gradient with respect to x
            if self.gradient_wrt == 'x':
                with torch.enable_grad():
                    x = x.detach().requires_grad_()
                    denoised = inner_model(x, sigma, **kwargs)
                    cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()

            # Gradient wrt x0_pred, so save some compute: don't record grad until after denoised is calculated
            elif self.gradient_wrt == 'x0_pred':
                with torch.no_grad():
                    denoised = inner_model(x, sigma, **kwargs)
                with torch.enable_grad():
                    cond_grad = cond_fn(x, sigma, denoised=denoised.detach().requires_grad_(), **kwargs).detach()
            total_cond_grad += cond_grad

        total_cond_grad = torch.nan_to_num(total_cond_grad, nan=0.0, posinf=float('inf'), neginf=-float('inf'))

        # Clamp the gradient
        total_cond_grad = self.clamp_grad_verbose(total_cond_grad, sigma)

        # Add gradient to the image
        if self.gradient_wrt == 'x':
            x.copy_(self.grad_consolidate_fn(x.detach(), total_cond_grad, k_utils.append_dims(sigma, x.ndim)))
            cond_denoised = inner_model(x, sigma, **kwargs)
        elif self.gradient_wrt == 'x0_pred':
            x.copy_(self.grad_consolidate_fn(x.detach(), total_cond_grad, k_utils.append_dims(sigma, x.ndim)))
            cond_denoised = self.grad_consolidate_fn(denoised.detach(), total_cond_grad, k_utils.append_dims(sigma, x.ndim))

        return cond_denoised

    def forward(self, x, sigma, uncond, cond, cond_scale):

        def _cfg_model(x, sigma, cond, **kwargs):
            # Wrapper to add denoised cond and uncond as in a cfg model
            # input "cond" is both cond and uncond weights: torch.cat([uncond, cond])
            x_in = torch.cat([x] * 2)
            sigma_in = torch.cat([sigma] * 2)

            denoised = self.inner_model(x_in, sigma_in, cond=cond, **kwargs)
            uncond_x0, cond_x0 = denoised.chunk(2)
            x0_pred = uncond_x0 + (cond_x0 - uncond_x0) * cond_scale
            return x0_pred

        # Conditioning
        if self.check_conditioning_schedule(sigma):
            # Apply the conditioning gradient to the completed denoised (after both cond and uncond are combined into the diffused image)
            if self.cond_uncond_sync:
                # x0 = self.cfg_cond_model_fn_(x, sigma, uncond=uncond, cond=cond, cond_scale=cond_scale)
                cond_in = torch.cat([uncond, cond])
                x0 = self.cond_model_fn_(x, sigma, cond=cond_in, inner_model=_cfg_model)

            # Calculate cond and uncond separately
            else:
                if self.gradient_add_to == "uncond":
                    uncond = self.cond_model_fn_(x, sigma, cond=uncond)
                    cond = self.inner_model(x, sigma, cond=cond)
                    x0 = uncond + (cond - uncond) * cond_scale
                elif self.gradient_add_to == "cond":
                    uncond = self.inner_model(x, sigma, cond=uncond)
                    cond = self.cond_model_fn_(x, sigma, cond=cond)
                    x0 = uncond + (cond - uncond) * cond_scale
                elif self.gradient_add_to == "both":
                    uncond = self.cond_model_fn_(x, sigma, cond=uncond)
                    cond = self.cond_model_fn_(x, sigma, cond=cond)
                    x0 = uncond + (cond - uncond) * cond_scale
                else: 
                    raise Exception(f"Unrecognised option for gradient_add_to: {self.gradient_add_to}")

        # No conditioning
        else:
            # calculate cond and uncond simultaneously
            if self.cond_uncond_sync:
                cond_in = torch.cat([uncond, cond])
                x0 = _cfg_model(x, sigma, cond=cond_in)
            else:
                uncond = self.inner_model(x, sigma, cond=uncond)
                cond = self.inner_model(x, sigma, cond=cond)
                x0 = uncond + (cond - uncond) * cond_scale

        return x0

    def make_cond_fn(self, loss_fn, scale):
        # Turns a loss function into a cond function that is applied to the decoded RGB sample
        # loss_fn (function): func(x, sigma, denoised) -> number
        # scale (number): how much this loss is applied to the image

        # Cond function with respect to x
        def cond_fn(x, sigma, denoised, **kwargs):
            with torch.enable_grad():
                denoised_sample = self.decode_fn(denoised).requires_grad_()
                loss = loss_fn(denoised_sample, sigma, **kwargs) * scale
                grad = -torch.autograd.grad(loss, x)[0]
            self.verbose_print('Loss:', loss.item())
            return grad

        # Cond function with respect to x0_pred
        def cond_fn_pred(x, sigma, denoised, **kwargs):
            with torch.enable_grad():
                denoised_sample = self.decode_fn(denoised).requires_grad_()
                loss = loss_fn(denoised_sample, sigma, **kwargs) * scale
                grad = -torch.autograd.grad(loss, denoised)[0]
            self.verbose_print('Loss:', loss.item())
            return grad

        if self.gradient_wrt == 'x':
            return cond_fn
        elif self.gradient_wrt == 'x0_pred':
            return cond_fn_pred
        else:
            raise Exception(f"Variable gradient_wrt == {self.gradient_wrt} not recognised.")

    def clamp_grad_verbose(self, grad, sigma):
        if self.clamp_func is not None:
            if self.verbose:
                print("Grad before clamping:")
                self.display_samples(torch.abs(grad*2.0) - 1.0)
            grad = self.clamp_func(grad, sigma)
        if self.verbose:
            print("Conditioning gradient")
            self.display_samples(torch.abs(grad*2.0) - 1.0)
        return grad

    def check_conditioning_schedule(self, sigma):
        is_conditioning_step = False

        if (self.cond_fns is not None and 
            any(cond_fn is not None for cond_fn in self.cond_fns)):
            # Conditioning strength != 0
            # Check if this is a conditioning step
            if self.grad_inject_timing_fn(sigma):
                is_conditioning_step = True

                if self.verbose:
                    print(f"Conditioning step for sigma={sigma}")

        return is_conditioning_step

    def display_samples(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(images.numpy())
        grid = make_grid(images, 4).cpu()
        display.display(to_pil_image(grid))
        return
