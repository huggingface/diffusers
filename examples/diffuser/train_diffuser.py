import os
import warnings

import numpy as np
import torch

import d4rl  # noqa
import gym
import mediapy as media
import mujoco_py as mjc
import tqdm
from diffusers import DDPMScheduler, UNet1DModel


# Define some helper functions


DTYPE = torch.float


def normalize(x_in, data, key):
    means = data[key].mean(axis=0)
    stds = data[key].std(axis=0)
    return (x_in - means) / stds


def de_normalize(x_in, data, key):
    means = data[key].mean(axis=0)
    stds = data[key].std(axis=0)
    return x_in * stds + means


def to_torch(x_in, dtype=None, device="cuda"):
    dtype = dtype or DTYPE
    device = device
    if type(x_in) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x_in.items()}
    elif torch.is_tensor(x_in):
        return x_in.to(device).type(dtype)
    return torch.tensor(x_in, dtype=dtype, device=device)


def reset_x0(x_in, cond, act_dim):
    for key, val in cond.items():
        x_in[:, key, act_dim:] = val.clone()
    return x_in


def run_diffusion(x, scheduler, generator, network, unet, conditions, action_dim, config):
    y = None
    for i in tqdm.tqdm(scheduler.timesteps):
        # create batch of timesteps to pass into model
        timesteps = torch.full((config["n_samples"],), i, device=config["device"], dtype=torch.long)
        # 3. call the sample function
        for _ in range(config["n_guide_steps"]):
            with torch.enable_grad():
                x.requires_grad_()
                y = network(x, timesteps).sample
                grad = torch.autograd.grad([y.sum()], [x])[0]
            if config["scale_grad_by_std"]:
                posterior_variance = scheduler._get_variance(i)
                model_std = torch.exp(0.5 * posterior_variance)
                grad = model_std * grad
            grad[timesteps < config["t_grad_cutoff"]] = 0
            x = x.detach()
            x = x + config["scale"] * grad
            x = reset_x0(x, conditions, action_dim)
        # with torch.no_grad():
        prev_x = unet(x.permute(0, 2, 1), timesteps).sample.permute(0, 2, 1)
        x = scheduler.step(prev_x, i, x, predict_epsilon=False)["prev_sample"]

        # 3. [optional] add posterior noise to the sample
        if config["eta"] > 0:
            noise = torch.randn(x.shape).to(x.device)
            posterior_variance = scheduler._get_variance(i)  # * noise
            # no noise when t == 0
            # NOTE: original implementation missing sqrt on posterior_variance
            x = x + int(i > 0) * (0.5 * posterior_variance) * config["eta"] * noise  # MJ had as log var, exponentiated

        # 4. apply conditions to the trajectory
        x = reset_x0(x, conditions, action_dim)
        x = to_torch(x, device=config["device"])
    # y = network(x, timesteps).sample
    return x, y


def to_np(x_in):
    if torch.is_tensor(x_in):
        x_in = x_in.detach().cpu().numpy()
    return x_in


# from MJ's Diffuser code
# https://github.com/jannerm/diffuser/blob/76ae49ae85ba1c833bf78438faffdc63b8b4d55d/diffuser/utils/colab.py#L79
def mkdir(savepath):
    """
    returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False


def show_sample(renderer, observations, filename="sample.mp4", savebase="videos"):
    """
    observations : [ batch_size x horizon x observation_dim ]
    """

    mkdir(savebase)
    savepath = os.path.join(savebase, filename)

    images = []
    for rollout in observations:
        # [ horizon x height x width x channels ]
        img = renderer._renders(rollout, partial=True)
        images.append(img)

    # [ horizon x height x (batch_size * width) x channels ]
    images = np.concatenate(images, axis=2)
    media.write_video(savepath, images, fps=60)
    media.show_video(images, codec="h264", fps=60)
    return images


# Code adapted from Michael Janner
# source: https://github.com/jannerm/diffuser/blob/main/diffuser/utils/rendering.py


def env_map(env_name):
    """
    map D4RL dataset names to custom fully-observed
    variants for rendering
    """
    if "halfcheetah" in env_name:
        return "HalfCheetahFullObs-v2"
    elif "hopper" in env_name:
        return "HopperFullObs-v2"
    elif "walker2d" in env_name:
        return "Walker2dFullObs-v2"
    else:
        return env_name


def get_image_mask(img):
    background = (img == 255).all(axis=-1, keepdims=True)
    mask = ~background.repeat(3, axis=-1)
    return mask


def atmost_2d(x):
    while x.ndim > 2:
        x = x.squeeze(0)
    return x


def set_state(env, state):
    qpos_dim = env.sim.data.qpos.size
    qvel_dim = env.sim.data.qvel.size
    if not state.size == qpos_dim + qvel_dim:
        warnings.warn(
            f"[ utils/rendering ] Expected state of size {qpos_dim + qvel_dim}, but got state of size {state.size}"
        )
        state = state[: qpos_dim + qvel_dim]

    env.set_state(state[:qpos_dim], state[qpos_dim:])


class MuJoCoRenderer:
    """
    default mujoco renderer
    """

    def __init__(self, env):
        if type(env) is str:
            env = env_map(env)
            self.env = gym.make(env)
        else:
            self.env = env
        # - 1 because the envs in renderer are fully-observed
        # @TODO : clean up
        self.observation_dim = np.prod(self.env.observation_space.shape) - 1
        self.action_dim = np.prod(self.env.action_space.shape)
        try:
            self.viewer = mjc.MjRenderContextOffscreen(self.env.sim)
        except:
            print("[ utils/rendering ] Warning: could not initialize offscreen renderer")
            self.viewer = None

    def pad_observation(self, observation):
        state = np.concatenate(
            [
                np.zeros(1),
                observation,
            ]
        )
        return state

    def pad_observations(self, observations):
        qpos_dim = self.env.sim.data.qpos.size
        # xpos is hidden
        xvel_dim = qpos_dim - 1
        xvel = observations[:, xvel_dim]
        xpos = np.cumsum(xvel) * self.env.dt
        states = np.concatenate(
            [
                xpos[:, None],
                observations,
            ],
            axis=-1,
        )
        return states

    def render(self, observation, dim=256, partial=False, qvel=True, render_kwargs=None, conditions=None):
        if type(dim) == int:
            dim = (dim, dim)

        if self.viewer is None:
            return np.zeros((*dim, 3), np.uint8)

        if render_kwargs is None:
            xpos = observation[0] if not partial else 0
            render_kwargs = {"trackbodyid": 2, "distance": 3, "lookat": [xpos, -0.5, 1], "elevation": -20}

        for key, val in render_kwargs.items():
            if key == "lookat":
                self.viewer.cam.lookat[:] = val[:]
            else:
                setattr(self.viewer.cam, key, val)

        if partial:
            state = self.pad_observation(observation)
        else:
            state = observation

        qpos_dim = self.env.sim.data.qpos.size
        if not qvel or state.shape[-1] == qpos_dim:
            qvel_dim = self.env.sim.data.qvel.size
            state = np.concatenate([state, np.zeros(qvel_dim)])

        set_state(self.env, state)

        self.viewer.render(*dim)
        data = self.viewer.read_pixels(*dim, depth=False)
        data = data[::-1, :, :]
        return data

    def _renders(self, observations, **kwargs):
        images = []
        for observation in observations:
            img = self.render(observation, **kwargs)
            images.append(img)
        return np.stack(images, axis=0)

    def renders(self, samples, partial=False, **kwargs):
        if partial:
            samples = self.pad_observations(samples)
            partial = False

        sample_images = self._renders(samples, partial=partial, **kwargs)

        composite = np.ones_like(sample_images[0]) * 255

        for img in sample_images:
            mask = get_image_mask(img)
            composite[mask] = img[mask]

        return composite

    def __call__(self, *args, **kwargs):
        return self.renders(*args, **kwargs)


env_name = "hopper-medium-expert-v2"
env = gym.make(env_name)
data = env.get_dataset()  # dataset is only used for normalization in this colab

# Cuda settings for colab
# torch.cuda.get_device_name(0)
DEVICE = "cpu"
DTYPE = torch.float

# diffusion model settings
n_samples = 4  # number of trajectories planned via diffusion
horizon = 128  # length of sampled trajectories
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
num_inference_steps = 100  # number of difusion steps

obs = env.reset()
obs_raw = obs

# normalize observations for forward passes
obs = normalize(obs, data, "observations")


# Two generators for different parts of the diffusion loop to work in colab
generator = torch.Generator(device="cuda")
generator_cpu = torch.Generator(device="cpu")
network = UNet1DModel.from_pretrained("fusing/ddpm-unet-rl-hopper-hor128").to(device=DEVICE)

scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")
optimizer = torch.optim.AdamW(
    network.parameters(),
    lr=0.001,
    betas=(0.95, 0.99),
    weight_decay=1e-6,
    eps=1e-8,
)

# TODO: Flesh this out using accelerate library (a la other examples)
