import torch
# from tqdm import trange, tqdm
# from . import utils


'''
helper functions:   append_zero(),
                    t_to_sigma(),
                    get_sigmas(),
                    append_dims(),
                    CFGDenoiserForward(),
                    get_scalings(),
                    DSsigma_to_t(),
                    DiscreteEpsDDPMDenoiserForward(),

need cleaning 
'''


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def t_to_sigma(t,sigmas):
    t = t.float()
    low_idx, high_idx, w = t.floor().long(), t.ceil().long(), t.frac()
    return (1 - w) * sigmas[low_idx] + w * sigmas[high_idx]


def get_sigmas(sigmas, n=None):
    if n is None:
        return append_zero(sigmas.flip(0))
    t_max = len(sigmas) - 1 # = 999
    t = torch.linspace(t_max, 0, n, device="cpu")
    # t = torch.linspace(t_max, 0, n, device=sigmas.device)
    return append_zero(t_to_sigma(t,sigmas))

#from k_samplers utils.py
def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]

def CFGDenoiserForward(Unet, x_in, sigma_in, cond_in, cond_scale,DSsigmas=None):
        # x_in = torch.cat([x] * 2)#A# concat the latent
        # sigma_in = torch.cat([sigma] * 2) #A# concat sigma
        # cond_in = torch.cat([uncond, cond])
        # uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        # uncond, cond = DiscreteEpsDDPMDenoiserForward(Unet,x_in, sigma_in,DSsigmas=DSsigmas, cond=cond_in).chunk(2)
        # return uncond + (cond - uncond) * cond_scale
        noise_pred = DiscreteEpsDDPMDenoiserForward(Unet,x_in, sigma_in,DSsigmas=DSsigmas, cond=cond_in)
        return noise_pred

def get_scalings(sigma):
        sigma_data = 1.
        c_out = -sigma
        c_in = 1 / (sigma ** 2 + sigma_data ** 2) ** 0.5
        return c_out, c_in

#DiscreteSchedule DS
def DSsigma_to_t(sigma, quantize=None,DSsigmas=None):
    # quantize = self.quantize if quantize is None else quantize
    quantize = False
    dists = torch.abs(sigma - DSsigmas[:, None])
    if quantize:
        return torch.argmin(dists, dim=0).view(sigma.shape)
    low_idx, high_idx = torch.sort(torch.topk(dists, dim=0, k=2, largest=False).indices, dim=0)[0]
    low, high = DSsigmas[low_idx], DSsigmas[high_idx]
    w = (low - sigma) / (low - high)
    w = w.clamp(0, 1)
    t = (1 - w) * low_idx + w * high_idx
    return t.view(sigma.shape)


def DiscreteEpsDDPMDenoiserForward(Unet,input,sigma,DSsigmas=None,**kwargs):
    c_out, c_in = [append_dims(x, input.ndim) for x in get_scalings(sigma)]
    #??? what is eps? 
    # eps = CVDget_eps(Unet,input * c_in, DSsigma_to_t(sigma), **kwargs)
    eps = Unet(input * c_in, DSsigma_to_t(sigma,DSsigmas=DSsigmas), encoder_hidden_states=kwargs['cond']).sample
    return input + eps * c_out


