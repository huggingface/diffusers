# Overview

These examples show how to run (Diffuser)[https://arxiv.org/pdf/2205.09991.pdf] in Diffusers. There are two scripts, `run_diffuser_value_guided.py` and `run_diffuser.py`.

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
