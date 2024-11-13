
## OpenAI Compatible `/v1/images/generations` Server

This is a concurrent, multithreaded solution for running a server that can generate images using the `diffusers` library. This examples uses the Stable Diffusion 3 pipeline, but you can use any pipeline that you would like by swapping out the model and pipeline to be the ones that you want to use.

### Installing Dependencies

The pipeline can have its dependencies installed with:
```
pip install -f requirements.txt
```

### Upgrading Dependencies

If you need to upgrade some dependencies, you can do that with either [pip-tools](https://github.com/jazzband/pip-tools) or [uv](https://github.com/astral-sh/uv). With `uv`, this looks like:
```
uv pip compile requirements.in -o requirements.txt
```
