import numpy as np
import torch

import d4rl  # noqa
import gym
import helpers
import tqdm
from diffusers import DDPMScheduler, UNet1DModel, ValueFunction
from helpers import MuJoCoRenderer, show_sample


config = dict(
    n_samples=64,
    horizon=32,
    num_inference_steps=20,
    n_guide_steps=2,
    scale_grad_by_std=True,
    scale=0.1,
    eta=0.0,
    t_grad_cutoff=2,
    device="cpu",
)


def _run():
    env_name = "hopper-medium-v2"
    env = gym.make(env_name)
    data = env.get_dataset()  # dataset is only used for normalization in this colab
    render = MuJoCoRenderer(env)

    # Cuda settings for colab
    # torch.cuda.get_device_name(0)
    DEVICE = config["device"]

    # diffusion model settings
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Two generators for different parts of the diffusion loop to work in colab
    # generator = torch.Generator(device='cuda')
    generator = torch.Generator(device=DEVICE)

    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_inference_steps"],
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,
        variance_type="fixed_small_log",
    )

    # 3 different pretrained models are available for this task.
    # The horizion represents the length of trajectories used in training.
    # network = ValueFunction(training_horizon=horizon, dim=32, dim_mults=(1, 2, 4, 8), transition_dim=14, cond_dim=11)

    network = ValueFunction.from_pretrained("bglick13/hopper-medium-v2-value-function-hor32").to(device=DEVICE).eval()
    unet = UNet1DModel.from_pretrained("bglick13/hopper-medium-v2-unet-hor32").to(device=DEVICE).eval()
    # unet = UNet1DModel.from_pretrained("fusing/ddpm-unet-rl-hopper-hor128").to(device=DEVICE)
    # network = TemporalUNet.from_pretrained("fusing/ddpm-unet-rl-hopper-hor512").to(device=DEVICE)

    # add a batch dimension and repeat for multiple samples
    # [ observation_dim ] --> [ n_samples x observation_dim ]
    env.seed(0)
    obs = env.reset()
    total_reward = 0
    total_score = 0
    T = 1000
    rollout = [obs.copy()]
    trajectories = []
    y_maxes = [0]
    try:
        for t in tqdm.tqdm(range(T)):
            # 1. Call the policy
            # normalize observations for forward passes
            obs = helpers.normalize(obs, data, "observations")

            obs = obs[None].repeat(config["n_samples"], axis=0)
            conditions = {0: helpers.to_torch(obs, device=DEVICE)}

            # 2. Call the diffusion model
            # constants for inference
            batch_size = len(conditions[0])
            shape = (batch_size, config["horizon"], state_dim + action_dim)

            # sample random initial noise vector
            x1 = torch.randn(shape, device=DEVICE)

            # this model is conditioned from an initial state, so you will see this function
            #  multiple times to change the initial state of generated data to the state
            #  generated via env.reset() above or env.step() below
            x = helpers.reset_x0(x1, conditions, action_dim)

            # convert a np observation to torch for model forward pass
            x = helpers.to_torch(x, device=DEVICE)
            x, y = helpers.run_diffusion(x, scheduler, generator, network, unet, conditions, action_dim, config)
            if y is not None:
                sorted_idx = y.argsort(0, descending=True).squeeze()
                y_maxes.append(y[sorted_idx[0]].detach().cpu().numpy())
                sorted_values = x[sorted_idx]
            else:
                sorted_values = x
            actions = sorted_values[:, :, :action_dim]
            if t % 10 == 0:
                trajectory = sorted_values[:, :, action_dim:][0].unsqueeze(0).detach().cpu().numpy()
                trajectory = helpers.de_normalize(trajectory, data, "observations")
                trajectories.append(trajectory)

            actions = actions.detach().cpu().numpy()
            denorm_actions = helpers.de_normalize(actions, data, key="actions")
            # denorm_actions = denorm_actions[np.random.randint(config['n_samples']), 0]
            denorm_actions = denorm_actions[0, 0]

            # execute action in environment
            next_observation, reward, terminal, _ = env.step(denorm_actions)
            score = env.get_normalized_score(total_reward)
            # update return
            total_reward += reward
            total_score += score
            print(
                f"Step: {t}, Reward: {reward}, Total Reward: {total_reward}, Score: {score}, Total Score:"
                f" {total_score}"
            )
            # save observations for rendering
            rollout.append(next_observation.copy())

            obs = next_observation
    except KeyboardInterrupt:
        pass

    print(f"Total reward: {total_reward}")

    show_sample(render, np.expand_dims(np.stack(rollout), axis=0))


def run():
    _run()


if __name__ == "__main__":
    run()
