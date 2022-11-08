# Overview

These examples show how to run (Diffuser)[https://arxiv.org/abs/2205.09991] in Diffusers. 
There are four scripts, 
1. `run_diffuser_value_guided.py` to sample high reward trajectories for the RL agent and render them,
2. `run_diffuser_locomotion.py` to sample actions and run them in the environment,
3. and `run_diffuser_gen_trajectories.py` to just sample actions from the pre-trained diffusion model,
4. and finally `train_diffuser.py` to train one of these diffusion models.

You will need some RL specific requirements to run the examples:

```
pip install -f https://download.pytorch.org/whl/torch_stable.html \
                free-mujoco-py \
                einops \
                gym \
                protobuf==3.20.1 \
                git+https://github.com/rail-berkeley/d4rl.git \
                mediapy \
                Pillow==9.0.0
```
