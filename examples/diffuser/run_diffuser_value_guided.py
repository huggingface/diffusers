import d4rl

import torch
import tqdm
import numpy as np
import gym 
from diffusers import DDPMScheduler, UNet1DModel, ValueFunction, ValueFunctionScheduler
from helpers import MuJoCoRenderer, show_sample
import helpers
import wandb
wandb.init(project="diffusers-value-guided-rl")

config = dict(
    n_samples=4,
    horizon=32,
    num_inference_steps=200,
    n_guide_steps=0,
    scale_grad_by_std=True,
    scale=0.001,
    eta=0.0,
    t_grad_cutoff=4
)

# model = torch.load("../diffuser/test.torch")
# hf_value_function = ValueFunction(training_horizon=32, dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14, cond_dim=11)
# hf_value_function.load_state_dict(model.state_dict())
# hf_value_function.to_hub("bglick13/hf_value_function")

env_name = "hopper-medium-expert-v2"
env = gym.make(env_name)
data = env.get_dataset() # dataset is only used for normalization in this colab
render = MuJoCoRenderer(env)

# Cuda settings for colab
# torch.cuda.get_device_name(0)
DEVICE = 'cpu'
DTYPE = torch.float

# diffusion model settings
state_dim = env.observation_space.shape[0] 
action_dim = env.action_space.shape[0]




# Two generators for different parts of the diffusion loop to work in colab
# generator = torch.Generator(device='cuda')
generator_cpu = torch.Generator(device='cpu')

scheduler = ValueFunctionScheduler(num_train_timesteps=config['num_inference_steps'],beta_schedule="squaredcos_cap_v2", clip_sample=False)

# 3 different pretrained models are available for this task. 
# The horizion represents the length of trajectories used in training.
# network = ValueFunction(training_horizon=horizon, dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14, cond_dim=11)

network = ValueFunction.from_pretrained("bglick13/hopper-medium-expert-v2-value-function-hor32").to(device=DEVICE)
unet = UNet1DModel.from_pretrained("bglick13/hopper-medium-expert-v2-unet-hor32").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor256").to(device=DEVICE)
# network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor512").to(device=DEVICE)

## add a batch dimension and repeat for multiple samples
## [ observation_dim ] --> [ n_samples x observation_dim ]
obs = env.reset()
total_reward = 0
done = False
T = 400
rollout = [obs.copy()]
trajectories = []
y_maxes = []
try:
    for t in tqdm.tqdm(range(T)):
        obs_raw = obs
        # 1. Call the policy
        # normalize observations for forward passes
        obs = helpers.normalize(obs, data, 'observations')

        obs = obs[None].repeat(config['n_samples'], axis=0)
        conditions = {
            0: helpers.to_torch(obs, device=DEVICE)
        }

        # 2. Call the diffusion model
        # constants for inference
        batch_size = len(conditions[0])
        shape = (batch_size, config['horizon'], state_dim+action_dim)

        # sample random initial noise vector
        x1 = torch.randn(shape, device=DEVICE, generator=generator_cpu)

        # this model is conditioned from an initial state, so you will see this function
        #  multiple times to change the initial state of generated data to the state 
        #  generated via env.reset() above or env.step() below
        x = helpers.reset_x0(x1, conditions, action_dim)

        # convert a np observation to torch for model forward pass
        x = helpers.to_torch(x)
        x, y = helpers.run_diffusion(x, scheduler, generator_cpu, network, unet, conditions, action_dim, config)
        sorted_idx = y.argsort(0, descending=True).squeeze()
        y_maxes.append(y[sorted_idx[0]].detach().cpu().numpy())
        sorted_values = x[sorted_idx]
        actions = sorted_values[:, :, :action_dim]
        if t % 10 == 0:
            trajectory = sorted_values[:, :, action_dim:][0].unsqueeze(0).detach().numpy()
            trajectory = helpers.de_normalize(trajectory, data, 'observations')
            trajectories.append(trajectory)

        actions = actions.detach().cpu().numpy()
        denorm_actions = helpers.de_normalize(actions, data, key='actions')
        # denorm_actions = denorm_actions[np.random.randint(config['n_samples']), 0]
        denorm_actions = denorm_actions[0, 0]


        ## execute action in environment
        next_observation, reward, terminal, _ = env.step(denorm_actions)

        ## update return
        total_reward += reward
        wandb.log({"total_reward": total_reward, "reward": reward, "y_max": y_maxes[-1], "diff_from_expert_reward": reward - data['rewards'][t]})
        print(f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}")
        # save observations for rendering
        rollout.append(next_observation.copy())

        obs = next_observation
except KeyboardInterrupt:
    pass

print(f"Total reward: {total_reward}")

images = show_sample(render, np.expand_dims(np.stack(rollout),axis=0))
wandb.log({"rollout": wandb.Video('videos/sample.mp4', fps=60, format='mp4')})