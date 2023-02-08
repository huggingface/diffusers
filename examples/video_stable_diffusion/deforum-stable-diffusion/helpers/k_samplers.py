from typing import Any, Callable, Optional
from k_diffusion.external import CompVisDenoiser
from k_diffusion import sampling
import torch


def sampler_fn(
        c: torch.Tensor,
        uc: torch.Tensor,
        args,
        model_wrap: CompVisDenoiser,
        init_latent: Optional[torch.Tensor] = None,
        t_enc: Optional[torch.Tensor] = None,
        device=torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda"),
        cb: Callable[[Any], None] = None,
        verbose: Optional[bool] = False,
) -> torch.Tensor:
    shape = [args.C, args.H // args.f, args.W // args.f]
    sigmas: torch.Tensor = model_wrap.get_sigmas(args.steps)
    sigmas = sigmas[len(sigmas) - t_enc - 1 :]
    if args.use_init:
        if len(sigmas) > 0:
            x = (
                    init_latent
                    + torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
            )
        else:
            x = init_latent
    else:
        if len(sigmas) > 0:
            x = torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
        else:
            x = torch.zeros([args.n_samples, *shape], device=device)
    sampler_args = {
        "model": model_wrap,
        "x": x,
        "sigmas": sigmas,
        "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
        "disable": False,
        "callback": cb,
    }
    min = sigmas[0].item()
    max = min
    for i in sigmas:
        if i.item() < min and i.item() != 0.0:
            min = i.item()
    if args.sampler in ["dpm_fast"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigma_min": min,
            "sigma_max": max,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
            "n":args.steps,
        }
    elif args.sampler in ["dpm_adaptive"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigma_min": min,
            "sigma_max": max,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
        }
    sampler_map = {
        "klms": sampling.sample_lms,
        "dpm2": sampling.sample_dpm_2,
        "dpm2_ancestral": sampling.sample_dpm_2_ancestral,
        "heun": sampling.sample_heun,
        "euler": sampling.sample_euler,
        "euler_ancestral": sampling.sample_euler_ancestral,
        "dpm_fast": sampling.sample_dpm_fast,
        "dpm_adaptive": sampling.sample_dpm_adaptive,
        "dpmpp_2s_a": sampling.sample_dpmpp_2s_ancestral,
        "dpmpp_2m": sampling.sample_dpmpp_2m,
    }

    samples = sampler_map[args.sampler](**sampler_args)
    return samples


def make_inject_timing_fn(inject_timing, model, steps):
    """
    inject_timing (int or list of ints or list of floats between 0.0 and 1.0):
        int: compute every inject_timing steps
        list of floats: compute on these decimal fraction steps (eg, [0.5, 1.0] for 50 steps would be at steps 25 and 50)
        list of ints: compute on these steps
    model (CompVisDenoiser)
    steps (int): number of steps
    """
    all_sigmas = model.get_sigmas(steps)
    target_sigmas = torch.empty([0], device=all_sigmas.device)

    def timing_fn(sigma):
        is_conditioning_step = False
        if sigma in target_sigmas:
            is_conditioning_step = True
        return is_conditioning_step

    if inject_timing is None:
        timing_fn = lambda sigma: True
    elif isinstance(inject_timing,int) and inject_timing <= steps and inject_timing > 0:
        # Compute every nth step
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if (i+1) % inject_timing == 0]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)
    elif all(isinstance(t,float) for t in inject_timing) and all(t>=0.0 and t<=1.0 for t in inject_timing):
        # Compute on these steps (expressed as a decimal fraction between 0.0 and 1.0)
        target_indices = [int(frac_step*steps) if frac_step < 1.0 else steps-1 for frac_step in inject_timing]
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if i in target_indices]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)
    elif all(isinstance(t,int) for t in inject_timing) and all(t>0 and t<=steps for t in inject_timing):
        # Compute on these steps
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if i+1 in inject_timing]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)

    else:
        raise Exception(f"Not a valid input: inject_timing={inject_timing}\n" +
                        f"Must be an int, list of all ints (between step 1 and {steps}), or list of all floats between 0.0 and 1.0")
    return timing_fn