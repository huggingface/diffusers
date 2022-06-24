#!/usr/bin/env python3
import numpy as np
import PIL
import torch
import ml_collections
#from configs.ve import ffhq_ncsnpp_continuous as configs
#  from configs.ve import cifar10_ncsnpp_continuous as configs


# ffhq_ncsnpp_continuous config
def get_config():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  training.batch_size = 8
  training.n_iters = 2400001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  training.snapshot_freq_for_preemption = 5000
  training.snapshot_sampling = True
  training.sde = 'vesde'
  training.continuous = True
  training.likelihood_weighting = False
  training.reduce_mean = True

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.method = 'pc'
  sampling.predictor = 'reverse_diffusion'
  sampling.corrector = 'langevin'
  sampling.probability_flow = False
  sampling.snr = 0.15
  sampling.n_steps_each = 1
  sampling.noise_removal = True

  # eval
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.batch_size = 1024
  evaluate.num_samples = 50000
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 96

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'FFHQ'
  data.image_size = 1024
  data.centered = False
  data.random_flip = True
  data.uniform_dequantization = False
  data.num_channels = 3
  # Plug in your own path to the tfrecords file.
  data.tfrecords_path = '/raid/song/ffhq-dataset/ffhq/ffhq-r10.tfrecords'

  # model
  config.model = model = ml_collections.ConfigDict()
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_max = 1348
  model.num_scales = 2000
  model.ema_rate = 0.9999
  model.sigma_min = 0.01
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 16
  model.ch_mult = (1, 2, 4, 8, 16, 32, 32, 32)
  model.num_res_blocks = 1
  model.attn_resolutions = (16,)
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.embedding_type = 'fourier'

  # optim
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config


torch.backends.cuda.matmul.allow_tf32 = False
torch.manual_seed(3)


class NewReverseDiffusionPredictor:
  def __init__(self, score_fn, probability_flow=False, sigma_min=0.0, sigma_max=0.0, N=0):
    super().__init__()
    self.sigma_min = sigma_min
    self.sigma_max = sigma_max
    self.N = N
    self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))

    self.probability_flow = probability_flow
    self.score_fn = score_fn

  def discretize(self, x, t):
    timestep = (t * (self.N - 1)).long()
    sigma = self.discrete_sigmas.to(t.device)[timestep]
    adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                 self.discrete_sigmas[timestep - 1].to(t.device))
    f = torch.zeros_like(x)
    G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)

    labels = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    result = self.score_fn(x, labels)

    rev_f = f - G[:, None, None, None] ** 2 * result * (0.5 if self.probability_flow else 1.)
    rev_G = torch.zeros_like(G) if self.probability_flow else G
    return rev_f, rev_G

  def update_fn(self, x, t):
    f, G = self.discretize(x, t)
    z = torch.randn_like(x)
    x_mean = x - f
    x = x_mean + G[:, None, None, None] * z
    return x, x_mean


class NewLangevinCorrector:
  def __init__(self, score_fn, snr, n_steps, sigma_min=0.0, sigma_max=0.0):
    super().__init__()
    self.score_fn = score_fn
    self.snr = snr
    self.n_steps = n_steps

    self.sigma_min = sigma_min
    self.sigma_max = sigma_max

  def update_fn(self, x, t):
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
#    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
#      timestep = (t * (sde.N - 1) / sde.T).long()
#      alpha = sde.alphas.to(t.device)[timestep]
#    else:
    alpha = torch.ones_like(t)

    for i in range(n_steps):
      labels = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
      grad = score_fn(x, labels)
      noise = torch.randn_like(x)
      grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None, None, None] * grad
      x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

    return x, x_mean



def save_image(x):
    image_processed = np.clip(x.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
    image_pil = PIL.Image.fromarray(image_processed[0])
    image_pil.save("../images/hey.png")


#  ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
#ckpt_filename = "exp/ve/ffhq_1024_ncsnpp_continuous/checkpoint_60.pth"
# Note usually we need to restore ema etc...
# ema restored checkpoint used from below



config = get_config()  

sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
N = config.model.num_scales

sampling_eps = 1e-5

batch_size = 1 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

from diffusers import NCSNpp
model = NCSNpp(config).to(config.device)
model = torch.nn.DataParallel(model)

loaded_state = torch.load("../score_sde_pytorch/ffhq_1024_ncsnpp_continuous_ema.pt")
del loaded_state["module.sigmas"]
model.load_state_dict(loaded_state, strict=False)

def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x

inverse_scaler = get_data_inverse_scaler(config)

img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
probability_flow = False
snr = 0.15 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}


device = config.device

new_corrector = NewLangevinCorrector(score_fn=model, snr=snr, n_steps=n_steps, sigma_min=sigma_min, sigma_max=sigma_max)
new_predictor = NewReverseDiffusionPredictor(score_fn=model, sigma_min=sigma_min, sigma_max=sigma_max, N=N)

with torch.no_grad():
    # Initial sample
    x = torch.randn(*shape) * sigma_max
    x = x.to(device)
    timesteps = torch.linspace(1, sampling_eps, N, device=device)

    for i in range(N):
        t = timesteps[i]
        vec_t = torch.ones(shape[0], device=t.device) * t
        x, x_mean = new_corrector.update_fn(x, vec_t)
        x, x_mean = new_predictor.update_fn(x, vec_t)

    x = inverse_scaler(x_mean)


save_image(x)

# for 5 cifar10
x_sum = 106071.9922
x_mean = 34.52864456176758

# for 1000 cifar10
x_sum = 461.9700
x_mean = 0.1504

# for 2 for 1024
x_sum = 3382810112.0
x_mean = 1075.366455078125

def check_x_sum_x_mean(x, x_sum, x_mean):
    assert (x.abs().sum() - x_sum).abs().cpu().item() < 1e-2, f"sum wrong {x.abs().sum()}"
    assert (x.abs().mean() - x_mean).abs().cpu().item() < 1e-4, f"mean wrong {x.abs().mean()}"


#check_x_sum_x_mean(x, x_sum, x_mean)
