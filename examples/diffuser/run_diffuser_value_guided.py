import d4rl

import torch
import tqdm
import numpy as np
import gym 
from diffusers import DDPMScheduler, TemporalUNet, ValueFunction, ValueFunctionScheduler
from helpers import MuJoCoRenderer, show_sample


# model = torch.load("../diffuser/test.torch")
# hf_value_function = ValueFunction(training_horizon=32, dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14, cond_dim=11)
# hf_value_function.load_state_dict(model.state_dict())
# hf_value_function.to_hub("bglick13/hf_value_function")

env_name = "hopper-medium-expert-v2"
env = gym.make(env_name)
data = env.get_dataset() # dataset is only used for normalization in this colab

# Cuda settings for colab
# torch.cuda.get_device_name(0)
DEVICE = 'cpu'
DTYPE = torch.float

# diffusion model settings
n_samples = 4   # number of trajectories planned via diffusion
horizon = 32   # length of sampled trajectories
state_dim = env.observation_space.shape[0] 
action_dim = env.action_space.shape[0]
num_inference_steps = 100 # number of difusion steps

def normalize(x_in, data, key):
  upper = np.max(data[key], axis=0)
  lower = np.min(data[key], axis=0)
  x_out = 2*(x_in - lower)/(upper-lower) - 1
  return x_out

def de_normalize(x_in, data, key):
	upper = np.max(data[key], axis=0)
	lower = np.min(data[key], axis=0)
	x_out = lower + (upper - lower)*(1 + x_in) /2
	return x_out
	
def to_torch(x_in, dtype=None, device=None):
	dtype = dtype or DTYPE
	device = device or DEVICE
	if type(x_in) is dict:
		return {k: to_torch(v, dtype, device) for k, v in x_in.items()}
	elif torch.is_tensor(x_in):
		return x_in.to(device).type(dtype)
	return torch.tensor(x_in, dtype=dtype, device=device)



# Two generators for different parts of the diffusion loop to work in colab
# generator = torch.Generator(device='cuda')
generator_cpu = torch.Generator(device='cpu')

scheduler = ValueFunctionScheduler(num_train_timesteps=20,beta_schedule="squaredcos_cap_v2", clip_sample=False)

# 3 different pretrained models are available for this task. 
# The horizion represents the length of trajectories used in training.
# network = ValueFunction(training_horizon=horizon, dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14, cond_dim=11)

network = ValueFunction.from_pretrained("bglick13/hopper-medium-expert-v2-value-function-hor32").to(device=DEVICE)
unet = TemporalUNet.from_pretrained("bglick13/hopper-medium-expert-v2-unet-hor32").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor256").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor512").to(device=DEVICE)
def reset_x0(x_in, cond, act_dim):
	for key, val in cond.items():
		x_in[:, key, act_dim:] = val.clone()
	return x_in

# network specific constants for inference
clip_denoised = False
predict_epsilon = False
n_guide_steps = 2
scale_grad_by_std = True
scale = 0.001
## add a batch dimension and repeat for multiple samples
## [ observation_dim ] --> [ n_samples x observation_dim ]
obs = env.reset()
total_reward = 0
done = False
T = 300
rollout = [obs.copy()]
try:
    for t in tqdm.tqdm(range(T)):
        obs_raw = obs
        # 1. Call the policy
        # normalize observations for forward passes
        obs = normalize(obs, data, 'observations')

        obs = obs[None].repeat(n_samples, axis=0)
        conditions = {
            0: to_torch(obs, device=DEVICE)
        }

        # 2. Call the diffusion model
        # constants for inference
        batch_size = len(conditions[0])
        shape = (batch_size, horizon, state_dim+action_dim)

        # sample random initial noise vector
        x1 = torch.randn(shape, device=DEVICE, generator=generator_cpu)

        # this model is conditioned from an initial state, so you will see this function
        #  multiple times to change the initial state of generated data to the state 
        #  generated via env.reset() above or env.step() below
        x = reset_x0(x1, conditions, action_dim)

        # convert a np observation to torch for model forward pass
        x = to_torch(x)

        eta = 1.0 # noise factor for sampling reconstructed state

        # run the diffusion process
        # for i in tqdm.tqdm(reversed(range(num_inference_steps)), total=num_inference_steps):
        for i in tqdm.tqdm(scheduler.timesteps):

            # create batch of timesteps to pass into model
            timesteps = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
            
            # 3. call the sample function
            for _ in range(n_guide_steps):
                with torch.enable_grad():
                    x.requires_grad_()
                    y = network(x, timesteps).sample
                    grad = torch.autograd.grad([y.sum()], [x])[0]
                if scale_grad_by_std:
                    posterior_variance = scheduler._get_variance(i)
                    grad = posterior_variance * 0.5 * grad
                grad[i < 4] = 0
                x = x.detach()
                x = x + scale * grad
                x = reset_x0(x, conditions, action_dim)
            prev_x = unet(x, timesteps).sample
            # TODO: This should really be a TemporalUnet that predicts previos state given x
            x = scheduler.step(prev_x, i, x)["prev_sample"]
            x = reset_x0(x, conditions, action_dim)
            if clip_denoised:
                x.clamp_(-1., 1.)
            # 2. use the model prediction to reconstruct an observation (de-noise)
            

            # # 3. [optional] add posterior noise to the sample
            # if eta > 0:
            #     noise = torch.randn(obs_reconstruct.shape, generator=generator_cpu).to(obs_reconstruct.device)
            #     posterior_variance = scheduler._get_variance(i) # * noise
            #     # no noise when t == 0
            #     # NOTE: original implementation missing sqrt on posterior_variance
            #     obs_reconstruct = obs_reconstruct + int(i>0) * (0.5 * posterior_variance) * eta* noise  # MJ had as log var, exponentiated

            # 4. apply conditions to the trajectory
            # obs_reconstruct_postcond = reset_x0(obs_reconstruct, conditions, action_dim)
            x = to_torch(x)
        sorted_idx = y.argsort(0, descending=True).squeeze()
        sorted_values = x[sorted_idx]
        actions = sorted_values[:, :, :action_dim]
        actions = actions[0, 0].detach().cpu().numpy()
        actions = de_normalize(actions, data, key='actions')
        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(actions)

        ## update return
        total_reward += reward
        print(f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}")
        # save observations for rendering
        rollout.append(next_observation.copy())

        obs = next_observation
except KeyboardInterrupt:
    pass

print(f"Total reward: {total_reward}")
render = MuJoCoRenderer(env)
show_sample(render, np.expand_dims(np.stack(rollout),axis=0))