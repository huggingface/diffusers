
## Diffusion-based Policy Learning for RL

`diffusion_policy` implements [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/), a diffusion model that predicts robot action sequences in reinforcement learning tasks.

This example implements a robot control model for pushing a T-shaped block into a target area. The model takes in current state observations as input, and outputs a trajectory of subsequent steps to follow.

To execute the script, run `diffusion_policy.py`

## Diffuser Locomotion

These examples show how to run [Diffuser](https://huggingface.co/papers/2205.09991) in Diffusers.
There are two ways to use the script, `run_diffuser_locomotion.py`.

The key option is a change of the variable `n_guide_steps`.
When `n_guide_steps=0`, the trajectories are sampled from the diffusion model, but not fine-tuned to maximize reward in the environment.
By default, `n_guide_steps=2` to match the original implementation.


You will need some RL specific requirements to run the examples:

```sh
pip install -f https://download.pytorch.org/whl/torch_stable.html \
                free-mujoco-py \
                einops \
                gym==0.24.1 \
                protobuf==3.20.1 \
                git+https://github.com/rail-berkeley/d4rl.git \
                mediapy \
                Pillow==9.0.0
```
