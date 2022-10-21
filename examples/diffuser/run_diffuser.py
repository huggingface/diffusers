import numpy as np
import torch

import d4rl  # noqa
import gym
import tqdm
import train_diffuser
from diffusers import DDPMScheduler, UNet1DModel


env_name = "hopper-medium-expert-v2"
env = gym.make(env_name)
data = env.get_dataset()  # dataset is only used for normalization in this colab

DEVICE = "cpu"
DTYPE = torch.float

# diffusion model settings
n_samples = 4  # number of trajectories planned via diffusion
horizon = 128  # length of sampled trajectories
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
num_inference_steps = 100  # number of difusion steps


# Two generators for different parts of the diffusion loop to work in colab
generator_cpu = torch.Generator(device="cpu")

scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2")

# 3 different pretrained models are available for this task.
# The horizion represents the length of trajectories used in training.
network = UNet1DModel.from_pretrained("fusing/ddpm-unet-rl-hopper-hor128").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor256").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor512").to(device=DEVICE)


# network specific constants for inference
clip_denoised = network.clip_denoised
predict_epsilon = network.predict_epsilon

# [ observation_dim ] --> [ n_samples x observation_dim ]
obs = env.reset()
total_reward = 0
done = False
T = 300
rollout = [obs.copy()]

try:
    for t in tqdm.tqdm(range(T)):
        obs_raw = obs

        # normalize observations for forward passes
        obs = train_diffuser.normalize(obs, data, "observations")
        obs = obs[None].repeat(n_samples, axis=0)
        conditions = {0: train_diffuser.to_torch(obs, device=DEVICE)}

        # constants for inference
        batch_size = len(conditions[0])
        shape = (batch_size, horizon, state_dim + action_dim)

        # sample random initial noise vector
        x1 = torch.randn(shape, device=DEVICE, generator=generator_cpu)

        # this model is conditioned from an initial state, so you will see this function
        #  multiple times to change the initial state of generated data to the state
        #  generated via env.reset() above or env.step() below
        x = train_diffuser.reset_x0(x1, conditions, action_dim)

        # convert a np observation to torch for model forward pass
        x = train_diffuser.to_torch(x)

        eta = 1.0  # noise factor for sampling reconstructed state

        # run the diffusion process
        # for i in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
        for i in tqdm.tqdm(scheduler.timesteps):
            # create batch of timesteps to pass into model
            timesteps = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)

            # 1. generate prediction from model
            with torch.no_grad():
                residual = network(x, timesteps).sample

            # 2. use the model prediction to reconstruct an observation (de-noise)
            obs_reconstruct = scheduler.step(residual, i, x, predict_epsilon=predict_epsilon)["prev_sample"]

            # 3. [optional] add posterior noise to the sample
            if eta > 0:
                noise = torch.randn(obs_reconstruct.shape, generator=generator_cpu).to(obs_reconstruct.device)
                posterior_variance = scheduler._get_variance(i)  # * noise
                # no noise when t == 0
                # NOTE: original implementation missing sqrt on posterior_variance
                obs_reconstruct = (
                    obs_reconstruct + int(i > 0) * (0.5 * posterior_variance) * eta * noise
                )  # MJ had as log var, exponentiated

            # 4. apply conditions to the trajectory
            obs_reconstruct_postcond = train_diffuser.reset_x0(obs_reconstruct, conditions, action_dim)
            x = train_diffuser.to_torch(obs_reconstruct_postcond)
        plans = train_diffuser.helpers.to_np(x[:, :, :action_dim])
        # select random plan
        idx = np.random.randint(plans.shape[0])
        # select action at correct time
        action = plans[idx, 0, :]
        actions = train_diffuser.de_normalize(action, data, "actions")
        # execute action in environment
        next_observation, reward, terminal, _ = env.step(action)

        # update return
        total_reward += reward
        print(f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}")

        # save observations for rendering
        rollout.append(next_observation.copy())
        obs = next_observation
except KeyboardInterrupt:
    pass

print(f"Total reward: {total_reward}")
render = train_diffuser.MuJoCoRenderer(env)
train_diffuser.show_sample(render, np.expand_dims(np.stack(rollout), axis=0))
