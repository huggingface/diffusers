
# Create a server

Diffusers' pipelines can be used as an inference engine for a server. It supports concurrent and multithreaded requests to generate images that may be requested by multiple users at the same time.

This guide will show you how to use the [`StableDiffusion3Pipeline`] in a server, but feel free to use any pipeline you want.


Start by navigating to the `examples/server` folder and installing all the dependencies.

``py
pip install .
pip install -f requirements.txt



Launch the server with the following command.

```py
python server.py


If you need to upgrade some dependencies, you can use either [pip-tools](https://github.com/jazzband/pip-tools) or [uv](https://github.com/astral-sh/uv). For example, upgrade the dependencies with `uv` using the following command.

```

